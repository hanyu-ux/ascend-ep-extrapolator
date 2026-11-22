"""Bounded, read-only evidence from an existing Host-stack diagnostic report."""

from collections import Counter, defaultdict
from bisect import bisect_left, bisect_right
import copy
import csv
import hashlib
import json
from pathlib import Path
import re
from statistics import fmean


SCHEMA = "u_host_stack_evidence_v1"
LIMIT = 65536


def _is_wait(name):
    """Evidence selection only; do not change the core's timing categories."""
    return not re.search(r"no[_ ]?wait", name, re.I) and bool(re.search(
        r"synchroni[sz]e|\bwait\b|(?:stream|event).*wait|wait.*(?:stream|event)", name, re.I))


def _focus_kind(name):
    match = re.search(r"(?:^|[ :])(run_fullgraph|prepare_inputs|_graph_replay|propose|mixed_quant_sparse_flash_mla_metadata_fp8)$", name)
    if match:
        return match[1]
    if "MixedQuantSparseFlashMlaMetadataGetWorkspaceSize" in name:
        return "metadata_workspace"
    return None


def _round_boundaries(stages, windows, round_ids):
    result = []
    for round_id in sorted(round_ids):
        regions, missing = [], []
        def add(name, bounds):
            if bounds is None or bounds[1] <= bounds[0]:
                missing.append(name)
            else:
                regions.append({"name": name, "start_ns": bounds[0], "end_ns": bounds[1],
                                "duration_ns": bounds[1] - bounds[0]})
        head, tail = windows.get((round_id, "main_to_draft")), windows.get((round_id, "draft_to_nextmain"))
        match = re.fullmatch(r"hc-(\d+)", round_id)
        prior = windows.get(("hc-" + str(int(match[1]) - 1), "draft_to_nextmain")) if match else None
        add("main_hc", (prior[1], head[0]) if prior and head else None)
        add("main_to_draft", head)
        metadata = [(int(r["start_ns"]), int(r["end_ns"])) for r in stages
                    if r["round_id"] == round_id and r["stage_key"] == "draft_metadata"]
        add("draft_metadata", metadata[0] if len(metadata) == 1 else None)
        add("draft_hc", (head[1], tail[0]) if head and tail else None)
        add("draft_to_nextmain", tail)
        result.append({"round_id": round_id, "regions": regions, "missing_regions": missing,
                       "scope": "Same-round saved CSV stage boundaries; HC envelopes are not full graph ranges"})
    return result


def _path(core, value, base):
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return core._source_path(path)
    for candidate in (base / path, base.parent / path):
        if core._io_path(candidate).is_file():
            return core._source_path(candidate)
    return core._source_path(base / path)


def _stamp(path):
    s = path.stat()
    return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns


def _hash(core, path):
    digest = hashlib.sha256()
    with core._io_path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rows(core, path):
    with core._io_path(path).open(encoding="utf-8-sig", newline="") as stream:
        yield from csv.DictReader(stream)


def _source_sample(row):
    keep = ("row_index", "name", "status", "reason", "device_start_ns", "device_end_ns", "cpu_event_index",
            "cpu_name", "cpu_category", "cpu_start_ns", "cpu_end_ns", "cpu_leaf_name", "node_event_index", "node_start_ns",
            "node_end_ns", "enqueue_event_index", "enqueue_start_ns", "enqueue_end_ns", "dequeue_event_index",
            "dequeue_start_ns", "dequeue_end_ns", "queue_wait_ns", "submit_to_device_ns")
    result = {k: row.get(k) for k in keep}
    try:
        ancestors = json.loads(row.get("cpu_ancestors_json") or "[]")
        result["cpu_ancestors"] = ancestors[-32:]
        result["cpu_ancestors_omitted"] = max(0, len(ancestors) - 32)
        result["cpu_ancestors_scope"] = "Nearest 32 saved ancestors; saved causal link is not revalidated"
    except (ValueError, TypeError):
        result["cpu_ancestors_error"] = "Invalid saved ancestor JSON"
    return result


def _stage_evidence(core, entry, base):
    details = entry.get("stage_details") or {}
    stages_path = _path(core, details.get("stages_csv"), base)
    device_path = _path(core, details.get("device_tasks_csv"), base)
    if stages_path is None:
        return {"status": "NO_SAVED_STAGES", "gaps": []}
    access = core._io_path(stages_path)
    stamp = _stamp(access)
    stages = [r for r in _rows(core, stages_path) if r.get("level") == "0" and r.get("status") == "MATCHED"]
    if any(r.get("rank") != entry.get("rank") for r in stages):
        raise ValueError("Saved stage CSV rank differs from selected case")
    if _stamp(access) != stamp:
        raise ValueError("Saved stages changed during evidence export")
    windows = {}
    for row in stages:
        key = row["round_id"], row["window_key"]
        start, end = int(row["start_ns"]), int(row["end_ns"])
        previous = windows.get(key, (start, end))
        windows[key] = min(start, previous[0]), max(end, previous[1])
    draft, main, paired = [], [], []
    for (round_id, kind), bounds in windows.items():
        if kind != "main_to_draft":
            continue
        tail = windows.get((round_id, "draft_to_nextmain"))
        if tail and tail[0] > bounds[1]:
            draft.append((tail[0] - bounds[1]) / 1e6)
        match = re.fullmatch(r"hc-(\d+)", round_id)
        if match:
            prior = windows.get(("hc-" + str(int(match[1]) - 1), "draft_to_nextmain"))
            if prior and bounds[0] > prior[1]:
                main.append((bounds[0] - prior[1]) / 1e6)
                if tail and tail[0] > bounds[1]:
                    paired.append({"main_hc": bounds[0] - prior[1], "draft_hc": tail[0] - bounds[1],
                                   "u": bounds[1] - bounds[0] + tail[1] - tail[0], "cycle": tail[1] - prior[1]})
    result = {"status": "MATCHED", "scope": "HC envelopes reconstructed from saved matched stage boundaries; not full graph",
              "main_hc_ms_mean": fmean(main) if main else None, "main_hc_n": len(main),
              "draft_hc_ms_mean": fmean(draft) if draft else None, "draft_hc_n": len(draft), "gaps": []}
    result["paired_rounds"] = {"n": len(paired),
        **{key + "_ms_mean": fmean(r[key] for r in paired) / 1e6 if paired else None
           for key in ("main_hc", "draft_hc", "u", "cycle")},
        "closure_max_abs_ns": max((abs(r["cycle"] - r["main_hc"] - r["draft_hc"] - r["u"]) for r in paired), default=None),
        "scope": "Only same rounds with head, tail and consecutive prior tail; CSV HC envelopes, not full graph"}
    selected = [r for r in stages if r["stage_key"] in ("draft_metadata", "main_inputs")]
    result["stage7_rounds"] = sum(r["stage_key"] == "draft_metadata" for r in selected)
    if device_path is None:
        result["gap_status"] = "NO_SAVED_DEVICE_TASKS"
        return result
    device_access = core._io_path(device_path)
    stamp = _stamp(device_access)
    tasks = defaultdict(list)
    for row in _rows(core, device_path):
        if row.get("rank") != entry.get("rank"):
            raise ValueError("Saved device CSV rank differs from selected case")
        tasks[row["round_id"]].append(row)
    if _stamp(device_access) != stamp:
        raise ValueError("Saved device task export changed during evidence export")
    gaps = []
    for row in selected:
        left, right = int(row["start_ns"]), int(row["end_ns"])
        group = [r for r in tasks[row["round_id"]] if int(r["start_ns"]) < right and int(r["end_ns"]) > left]
        union = core.a_union([(max(left, int(r["start_ns"])), min(right, int(r["end_ns"]))) for r in group])
        for a, b in core.residual_complement(left, right, union):
            following = [r for r in group if int(r["start_ns"]) == b]
            previous = [r for r in group if int(r["end_ns"]) == a]
            slim = lambda r: {k: r.get(k) for k in ("row_index", "name", "start_ns", "end_ns", "stream_id", "task_id", "model_id")}
            gaps.append({"round_id": row["round_id"], "stage_key": row["stage_key"], "start_ns": a, "end_ns": b,
                         "duration_ns": b - a, "following_devices": [slim(r) for r in following[:4]],
                         "previous_devices": [slim(r) for r in previous[:4]],
                         "scope": "during_device_gap; CSV complement only, not causal Host attribution"})
    result["gaps"] = sorted([r for r in gaps if r["stage_key"] == "draft_metadata"], key=lambda r: -r["duration_ns"])[:2]
    result["main_inputs_gaps"] = sorted([r for r in gaps if r["stage_key"] == "main_inputs"], key=lambda r: -r["duration_ns"])[:1]
    result["gap_count"] = len(gaps)
    round_ids = {r["round_id"] for r in result["gaps"] + result["main_inputs_gaps"]}
    round_ids.update(w["round_id"] for w in entry.get("windows", []) if w.get("round_id"))
    result["round_boundaries"] = _round_boundaries(stages, windows, round_ids)
    return result


def _sources(core, entry, base, gaps):
    path = _path(core, (entry.get("stage_details") or {}).get("source_links_csv"), base)
    if path is None:
        return {"status": "NO_SAVED_SOURCE_LINKS", "groups": {}}
    access = core._io_path(path)
    stamp = _stamp(access)
    digest = _hash(core, path)
    counts, reasons, samples, by_row = defaultdict(Counter), defaultdict(Counter), defaultdict(list), {}
    wanted = {str(t["row_index"]) for gap in gaps for t in gap["following_devices"]}
    for row in _rows(core, path):
        name = row.get("name", "").replace("_", "").lower()
        group = "expand" if "expandidxmapping" in name else "metadata" if "metadata" in name else None
        if group:
            counts[group][row.get("status", "UNKNOWN")] += 1
            reasons[group][row.get("reason") or "none"] += 1
            samples[group].append(row)
            samples[group].sort(key=lambda r: (r.get("status") in ("LINKED", "CPU_ONLY"), int(r["row_index"])))
            del samples[group][3:]
        if row["row_index"] in wanted:
            if row["row_index"] in by_row and by_row[row["row_index"]] != row:
                raise ValueError("Conflicting saved source rows")
            by_row[row["row_index"]] = row
    if _stamp(access) != stamp:
        raise ValueError("Saved source-link export changed during evidence export")
    for gap in gaps:
        gap["following_source_links"] = [_source_sample(by_row[str(t["row_index"])]) for t in gap["following_devices"]
                                         if str(t["row_index"]) in by_row]
    return {"status": "READ", "sha256": digest, "hash_scope": "Current stable CSV; no expected source-link hash was stored in full report",
            "groups": {group: {"status_counts": dict(counts[group]), "reason_counts": dict(reasons[group]),
                                "samples": [_source_sample(r) for r in samples[group]]} for group in counts}}


def _select_windows(entry):
    result = []
    groups = defaultdict(list)
    for row in entry.get("windows", []):
        groups[row["kind"]].append(row)
    for kind, rows in groups.items():
        row = copy.deepcopy(max(rows, key=lambda r: r.get("window_ns", -1)))
        for key in ("inclusive_functions", "exclusive_functions"):
            values = sorted(row.get(key, []), key=lambda r: -r["ns"])
            waits = [r for r in values if _is_wait(r["name"])]
            chosen = values[:20]
            chosen.extend(r for r in waits[:20] if r not in chosen)
            row[key] = chosen
            row[key + "_export_omitted"] = len(values) - len(chosen)
        row["scope"] = "Longest saved window example; function times are clipped to this window, not whole-call times"
        result.append(row)
    return result


def _event(core, number, raw, offset):
    if raw.get("ph") != "X":
        return None
    try:
        start = core.a_ns(raw["ts"]) + offset
        duration = core.a_ns(raw["dur"])
    except (KeyError, ValueError, ArithmeticError):
        return None
    if duration <= 0:
        return None
    args = core.residual_args(raw)
    # The streaming reader retains JSON fractional times as Decimal. Export
    # their exact microsecond text; float conversion can lose epoch-scale ns.
    return {"event_index": number, "name": str(raw.get("name", "")), "pid": core.residual_id(raw.get("pid")),
            "tid": core.residual_id(raw.get("tid")), "cat": str(raw.get("cat", "")), "ts": str(raw["ts"]), "dur": str(raw["dur"]),
            "start_ns": start, "end_ns": start + duration, "duration_ns": duration,
            "thread_id": core.residual_id(args.get("threadid")),
            "python_id": core.residual_id(args.get("pythonid")),
            "python_parent_id": core.residual_id(args.get("pythonparentid")),
            "device_hint": core.residual_device_hint(raw, args)}


def _trace_evidence(core, entry, windows, stage7, sources, base):
    trace = entry.get("trace") or {}
    if trace.get("status") != "ALIGNED_X_EVENTS":
        return {"status": "CLOCK_NOT_ALIGNED", "reason": "No trace-to-device-gap attribution without saved alignment"}
    path = _path(core, trace.get("path"), base)
    expected = entry.get("trace_sha256") or trace.get("sha256")
    if entry.get("trace_sha256") and trace.get("sha256") and entry["trace_sha256"] != trace["sha256"]:
        raise ValueError("Saved report contains conflicting trace hashes")
    if path is None or not expected:
        return {"status": "MISSING_TRACE_PATH_OR_EXPECTED_HASH"}
    offset = int(trace.get("clock_offset_ns") or 0)
    targets, lanes, primary_specs = set(), set(), []
    for w in windows:
        anchor = w.get("anchor") or {}
        if anchor.get("event_index") is not None:
            targets.add(int(anchor["event_index"]))
        if anchor.get("pid") is not None and anchor.get("tid") is not None:
            lanes.add((str(anchor["pid"]), str(anchor["tid"])))
        wait_rows = []
        for key in ("exclusive_functions", "inclusive_functions"):
            for row in w.get(key, []):
                if _is_wait(row["name"]):
                    targets.update(int(i) for i in row.get("event_indexes", []))
                    if row.get("event_indexes"):
                        wait_rows.append(row)
        if wait_rows:
            longest = max(wait_rows, key=lambda r: (r["ns"], r["name"]))
            primary_specs.append({"window_id": w["id"], "window_kind": w["kind"], "round_id": w.get("round_id"),
                                  "function_clipped_ns": longest["ns"],
                                  "candidate_event_indexes": sorted(set(int(i) for i in longest["event_indexes"]))})
    gaps = stage7.get("gaps", []) + stage7.get("main_inputs_gaps", [])
    for gap in gaps:
        for row in gap.get("following_source_links", []):
            if row.get("cpu_event_index"):
                targets.add(int(row["cpu_event_index"]))
    if not targets and not gaps:
        return {"status": "NO_TARGET_EVENT_INDEXES", "expected_sha256": expected}
    access = core._io_path(path)
    stamp = _stamp(access)
    processes, events, scan = {}, {}, {}
    gap_candidates, gap_waits, focus_candidates = defaultdict(list), defaultdict(list), defaultdict(list)
    focus_ranges = []
    stage7_rounds = {g["round_id"] for g in stage7.get("gaps", [])}
    for boundary in stage7.get("round_boundaries", []):
        if boundary["round_id"] not in stage7_rounds:
            continue
        regions = {r["name"]: r for r in boundary["regions"]}
        head = regions.get("main_to_draft")
        if head:
            focus_ranges.append({"round_id": boundary["round_id"],
                                 "start_ns": regions.get("main_hc", head)["start_ns"], "end_ns": head["end_ns"]})
    label = str(entry.get("case", "unknown"))
    print(f"  {label}: scanning selected trace for wait events (pass 1/2)", flush=True)
    for number, raw in enumerate(core.residual_iter_trace_events(path, stats=scan), 1):
        if number % 100000 == 0:
            print(f"  {label}: trace pass 1/2, {number:,} events", flush=True)
        if raw.get("ph") == "M" and raw.get("name") == "process_name":
            processes[core.residual_id(raw.get("pid"))] = str(core.residual_args(raw).get("name", ""))
        e = _event(core, number, raw, offset)
        if e is None:
            continue
        if number in targets:
            events[number] = e
        kind = _focus_kind(e["name"])
        if kind and any(e["tid"] == tid for pid, tid in lanes):
            for region in focus_ranges:
                overlap = min(e["end_ns"], region["end_ns"]) - max(e["start_ns"], region["start_ns"])
                if overlap > 0:
                    key = region["round_id"], kind
                    focus_candidates[key].append(e)
                    focus_candidates[key].sort(key=lambda r: (-min(r["end_ns"], region["end_ns"]) + max(r["start_ns"], region["start_ns"]), r["event_index"]))
                    del focus_candidates[key][8:]
        for n, gap in enumerate(gaps):
            if max(e["start_ns"], gap["start_ns"]) < min(e["end_ns"], gap["end_ns"]):
                if (e["pid"], e["tid"]) in lanes or any(e["tid"] == lane[1] for lane in lanes):
                    gap_candidates[n].append(e)
                    gap_candidates[n].sort(key=lambda r: -min(r["end_ns"], gap["end_ns"]) + max(r["start_ns"], gap["start_ns"]))
                    del gap_candidates[n][100:]
                    if _is_wait(e["name"]) or _focus_kind(e["name"]):
                        gap_waits[n].append(e)
                        gap_waits[n].sort(key=lambda r: -r["duration_ns"])
                        del gap_waits[n][12:]
    if _stamp(access) != stamp or scan.get("sha256") != expected:
        raise ValueError("Trace changed or SHA256 differs from existing full report")
    focus_calls = defaultdict(list)
    for n, gap in enumerate(gaps):
        retained = {e["event_index"]: e for e in gap_candidates[n] + gap_waits[n]}
        candidates = [e for e in retained.values() if not e["device_hint"]
                      and not core.residual_device_process(processes.get(e["pid"], ""))
                      and ((e["pid"], e["tid"]) in lanes or re.search(r"cann|runtime", processes.get(e["pid"], ""), re.I)
                           and any(e["tid"] == tid and e["thread_id"] in (None, tid) for pid, tid in lanes))]
        candidates.sort(key=lambda e: (not bool(_focus_kind(e["name"])), not _is_wait(e["name"]), -e["duration_ns"], e["event_index"]))
        chosen = candidates[:20]
        waits = [e for e in candidates if _is_wait(e["name"]) or _focus_kind(e["name"])]
        chosen.extend(e for e in waits[:12] if e not in chosen)
        if gap["stage_key"] == "main_inputs":
            actual_waits = [e for e in candidates if _is_wait(e["name"])]
            if actual_waits:
                longest = max(actual_waits, key=lambda e: (min(e["end_ns"], gap["end_ns"]) - max(e["start_ns"], gap["start_ns"]),
                                                          e["duration_ns"], -e["event_index"]))
                primary_specs.append({"window_id": "device-gap:" + gap["round_id"], "window_kind": "main_inputs_device_gap",
                                      "round_id": gap["round_id"],
                                      "function_clipped_ns": min(longest["end_ns"], gap["end_ns"]) - max(longest["start_ns"], gap["start_ns"]),
                                      "candidate_event_indexes": [longest["event_index"]]})
                if longest not in chosen:
                    chosen.append(longest)
        gap["host_event_indexes"] = [e["event_index"] for e in chosen]
        gap["host_scope"] = "Same Host thread as saved sample anchors, during device gap; temporal co-occurrence only"
        gap["host_candidates_retained"] = len(candidates)
        for e in chosen:
            events[e["event_index"]] = e
            if _is_wait(e["name"]) or _focus_kind(e["name"]):
                targets.add(e["event_index"])
            if _focus_kind(e["name"]) in ("run_fullgraph", "prepare_inputs"):
                focus_calls[e["event_index"]].append({"round_id": gap["round_id"], "kind": _focus_kind(e["name"])})
    for (round_id, kind), values in sorted(focus_candidates.items()):
        eligible = [e for e in values if not e["device_hint"] and not core.residual_device_process(processes.get(e["pid"], ""))
                    and ((e["pid"], e["tid"]) in lanes or re.search(r"cann|runtime", processes.get(e["pid"], ""), re.I)
                         and any(e["tid"] == tid and e["thread_id"] in (None, tid) for pid, tid in lanes))]
        if eligible:
            e = eligible[0]
            events[e["event_index"]] = e
            targets.add(e["event_index"])
            focus_calls[e["event_index"]].append({"round_id": round_id, "kind": kind})
    target_events = {i: events[i] for i in targets if i in events}
    primary = defaultdict(list)
    for spec in primary_specs:
        available = [target_events[i] for i in spec["candidate_event_indexes"] if i in target_events]
        # A per-function window total is not an individual call duration. Pick
        # one recorded instance by its actual duration, with a stable ID tie.
        chosen = max(available, key=lambda e: (e["duration_ns"], -e["event_index"])) if available else None
        identity = chosen["event_index"] if chosen else spec["candidate_event_indexes"][0]
        primary[identity].append(spec)
    ordered_targets = sorted(target_events.values(), key=lambda e: e["start_ns"])
    target_starts = [e["start_ns"] for e in ordered_targets]
    parents = defaultdict(list)
    scan2 = {}
    print(f"  {label}: scanning containing parent frames (pass 2/2)", flush=True)
    for number, raw in enumerate(core.residual_iter_trace_events(path, stats=scan2), 1):
        if number % 100000 == 0:
            print(f"  {label}: trace pass 2/2, {number:,} events", flush=True)
        e = _event(core, number, raw, offset)
        if e is None or e["device_hint"] or core.residual_device_process(processes.get(e["pid"], "")):
            continue
        for target in ordered_targets[bisect_left(target_starts, e["start_ns"]):bisect_right(target_starts, e["end_ns"])]:
            identity = target["event_index"]
            same = (e["pid"], e["tid"]) == (target["pid"], target["tid"])
            cross = ((target["pid"], target["tid"]) not in lanes and (e["pid"], e["tid"]) in lanes
                     and e["tid"] == target["tid"] and target["thread_id"] in (None, target["tid"])
                     and re.search(r"cann|runtime", processes.get(target["pid"], ""), re.I))
            if number != identity and (same or cross) and e["start_ns"] <= target["start_ns"] and e["end_ns"] >= target["end_ns"]:
                parents[identity].append(e)
                parents[identity].sort(key=lambda r: (r["duration_ns"], r["event_index"]))
                del parents[identity][64:]
    if _stamp(access) != stamp or scan2.get("sha256") != expected:
        raise ValueError("Trace changed or SHA256 differs during parent-chain scan")
    chains = []
    for identity, target in target_events.items():
        current, chain, used = target, [], {identity}
        pool = parents[identity]
        for _ in range(12):
            explicit = [e for e in pool if e["event_index"] not in used and current["python_parent_id"] is not None
                        and current["cat"] == "python_function" and e["cat"] == "python_function"
                        and (e["pid"], e["tid"], e["python_id"]) == (current["pid"], current["tid"], current["python_parent_id"])
                        and e["start_ns"] <= current["start_ns"] and e["end_ns"] >= current["end_ns"]]
            if len(explicit) > 1:
                chain.append({"relation": "AMBIGUOUS_PYTHON_PARENT_ID", "matches": len(explicit)})
                break
            candidates = [e for e in pool if e["event_index"] not in used and e["start_ns"] <= current["start_ns"]
                          and e["end_ns"] >= current["end_ns"]]
            parent = explicit[0] if explicit else next(iter(candidates), None)
            if parent is None:
                break
            relation = "python_parent_id" if explicit else "inferred_time_containment_not_direct_caller"
            chain.append({"event_index": parent["event_index"], "relation": relation})
            events[parent["event_index"]] = parent
            used.add(parent["event_index"])
            current = parent
        chains.append({"target_event_index": identity, "target_is_wait": _is_wait(target["name"]),
                       "target_duration_ns": target["duration_ns"], "primary_windows": primary.get(identity, []),
                       "focus_calls": focus_calls.get(identity, []),
                       "parents": chain, "parent_depth_limit": 12,
                       "candidate_limit": 64})
    for e in events.values():
        if not (_is_wait(e["name"]) or _focus_kind(e["name"])):
            continue
        overlaps = []
        for boundary in stage7.get("round_boundaries", []):
            for region in boundary["regions"]:
                amount = min(e["end_ns"], region["end_ns"]) - max(e["start_ns"], region["start_ns"])
                if amount > 0:
                    overlaps.append({"round_id": boundary["round_id"], "region": region["name"], "overlap_ns": amount})
        e["round_overlaps"] = overlaps
    return {"status": "VERIFIED", "sha256": expected, "clock_alignment_status": trace.get("status"),
            "timestamp_scope": "Saved trace timestamps plus saved constant offset; graph causality not inferred",
            "raw_time_encoding": "ts/dur are exact decimal microsecond strings; start_ns/end_ns/duration_ns are integer nanoseconds",
            "missing_target_event_indexes": sorted(targets - set(target_events)),
            "primary_waits_expected": sorted(primary), "primary_waits_retained": sorted(set(primary) & set(target_events)),
            "primary_waits_missing": sorted(set(primary) - set(target_events)),
            "overlap_scope": "Per-event intersections with same-round CSV regions; samples only, temporal overlap is not causal proof",
            "events": [events[k] for k in sorted(events)], "wait_parent_chains": chains,
            "passes": 2}


def _chain_priority(chain):
    return (not bool(chain.get("primary_windows")), not bool(chain.get("focus_calls")),
            not chain.get("target_is_wait", False), -chain.get("target_duration_ns", 0), chain["target_event_index"])


def _prune_trace(case, omissions):
    trace = case.get("trace_evidence") or {}
    chains = trace.get("wait_parent_chains", [])
    required = {c["target_event_index"] for c in chains}
    required.update(p["event_index"] for c in chains for p in c["parents"] if "event_index" in p)
    events = trace.get("events", [])
    trace["events"] = [e for e in events if e["event_index"] in required]
    omissions["trace_events"] += len(events) - len(trace["events"])
    for gap in (case.get("stage7") or {}).get("gaps", []) + (case.get("stage7") or {}).get("main_inputs_gaps", []):
        gap["host_event_indexes"] = [i for i in gap.get("host_event_indexes", []) if i in required]


def _validate_primary(case):
    trace = case.get("trace_evidence") or {}
    expected = set(case.get("primary_waits_expected", trace.get("primary_waits_expected", [])))
    events = {e["event_index"] for e in trace.get("events", [])}
    chains = {c["target_event_index"] for c in trace.get("wait_parent_chains", [])}
    retained = expected & events & chains
    for destination in (case, trace):
        destination["primary_waits_expected"] = sorted(expected)
        destination["primary_waits_retained"] = sorted(retained)
        destination["primary_waits_missing"] = sorted(expected - retained)
    if expected - retained:
        trace["status"] = "PRIMARY_WAIT_EVIDENCE_MISSING"
        case["status"] = "ERROR"
        case["error"] = "Primary wait target or parent-chain evidence is missing; inspect primary_waits_missing"


def bounded_evidence(payload, limit=LIMIT):
    result = copy.deepcopy(payload)
    result.pop("evidence_checkpoint", None)
    result.pop("checkpoint_sources", None)
    result["export_max_bytes"] = limit
    result["export_omissions"] = Counter()
    for case in result["cases"]:
        case["primary_waits_expected"] = list((case.get("trace_evidence") or {}).get("primary_waits_expected", []))
    def encoded():
        return (json.dumps(result, ensure_ascii=True, allow_nan=False, separators=(",", ":")) + "\n").encode("ascii")
    for cap in (20, 10, 5, 2, 0):
        if len(encoded()) <= limit:
            break
        for case in reversed(result["cases"]):
            for summary in case.get("window_summary") or []:
                values = summary.get("top_exclusive_functions", [])
                waits = [r for r in values if _is_wait(r.get("name", ""))]
                keep = values[:cap]
                keep.extend(r for r in waits[:max(1, min(cap, 8))] if r not in keep)
                result["export_omissions"]["summary_function_rows"] += len(values) - len(keep)
                summary["top_exclusive_functions"] = keep
            for w in case.get("windows", []):
                for key in ("inclusive_functions", "exclusive_functions"):
                    rows = w.get(key, [])
                    waits = [r for r in rows if _is_wait(r.get("name", ""))]
                    keep = rows[:cap]
                    keep.extend(r for r in waits[:max(1, min(cap, 8))] if r not in keep)
                    result["export_omissions"]["function_rows"] += len(rows) - len(keep)
                    w[key] = keep
            for group in (case.get("source_links") or {}).get("groups", {}).values():
                samples = group.get("samples", [])
                if cap < 5:
                    result["export_omissions"]["source_samples"] += max(0, len(samples) - 1)
                    group["samples"] = samples[:1]
    for cap in (12, 6, 3):
        if len(encoded()) <= limit:
            break
        for case in reversed(result["cases"]):
            samples = [s for g in (case.get("source_links") or {}).get("groups", {}).values() for s in g.get("samples", [])]
            for gap in (case.get("stage7") or {}).get("gaps", []) + (case.get("stage7") or {}).get("main_inputs_gaps", []):
                samples.extend(gap.get("following_source_links", []))
            for sample in samples:
                values = sample.get("cpu_ancestors", [])
                omitted = max(0, len(values) - cap)
                sample["cpu_ancestors"] = values[-cap:]
                sample["cpu_ancestors_omitted"] = sample.get("cpu_ancestors_omitted", 0) + omitted
                result["export_omissions"]["source_ancestors"] += omitted
    if len(encoded()) > limit:
        for case in reversed(result["cases"]):
            evidence = case.get("trace_evidence") or {}
            for chain in evidence.get("wait_parent_chains", []):
                if len(encoded()) <= limit:
                    break
                depth = 6 if chain.get("primary_windows") else 4
                result["export_omissions"]["parent_links"] += max(0, len(chain["parents"]) - depth)
                chain["parents"] = chain["parents"][:depth]
            _prune_trace(case, result["export_omissions"])
    # Preserve a few representative waits even for unusually large Python
    # names or many repeated call samples. Means and per-case errors remain.
    for cap in (8, 4, 2, 1, 0):
        if len(encoded()) <= limit:
            break
        for case in reversed(result["cases"]):
            trace = case.get("trace_evidence") or {}
            chains = trace.get("wait_parent_chains", [])
            # Focus calls are useful context, but can be numerous. Only the
            # requested primary waits are mandatory under the byte budget.
            protected = [c for c in chains if c.get("primary_windows")]
            optional = [c for c in chains if c not in protected]
            selected = sorted(protected, key=_chain_priority) + sorted(optional, key=_chain_priority)[:cap]
            trace["wait_parent_chains"] = selected
            result["export_omissions"]["target_parent_chains"] += len(chains) - len(selected)
            _prune_trace(case, result["export_omissions"])
    if len(encoded()) > limit and any(c.get("primary_waits_expected") for c in result["cases"]):
        # Discard reproducible summaries before losing the purpose of this
        # export. The complete collected result is saved in the checkpoint.
        for case in result["cases"]:
            for key in ("windows", "window_summary", "phase_summary", "all_stage_wall_means", "source_links"):
                if key in case:
                    case.pop(key)
                    result["export_omissions"][key] += 1
            for gap in (case.get("stage7") or {}).get("gaps", []) + (case.get("stage7") or {}).get("main_inputs_gaps", []):
                if gap.pop("following_source_links", None) is not None:
                    result["export_omissions"]["gap_source_links"] += 1
            _prune_trace(case, result["export_omissions"])
    if len(encoded()) > limit:
        # Keep actual primary waits and concrete Host calls ahead of source
        # samples and wrapper ancestors. They are the purpose of this export.
        for case in result["cases"]:
            case["source_links"] = {"status": "OMITTED_FOR_BYTE_LIMIT"}
            result["export_omissions"]["source_links"] += 1
            for chain in (case.get("trace_evidence") or {}).get("wait_parent_chains", []):
                depth = 4 if chain.get("primary_windows") else 0
                result["export_omissions"]["parent_links"] += max(0, len(chain["parents"]) - depth)
                chain["parents"] = chain["parents"][:depth]
            _prune_trace(case, result["export_omissions"])
    for case in result["cases"]:
        _validate_primary(case)
    data = encoded()
    if len(data) > limit:
        # Pathological display strings must not lose all twelve-case progress.
        # Retain scalar means and identities; explicitly mark detailed omission.
        for case in result["cases"]:
            for key in ("windows", "source_links", "trace_evidence"):
                if key in case:
                    case[key] = {"status": "OMITTED_FOR_BYTE_LIMIT"}
                    result["export_omissions"][key] += 1
            for gap in (case.get("stage7") or {}).get("gaps", []) + (case.get("stage7") or {}).get("main_inputs_gaps", []):
                gap.pop("following_source_links", None)
                result["export_omissions"]["gap_host_event_references"] += len(gap.get("host_event_indexes", []))
                gap["host_event_indexes"] = []
                gap["host_evidence_status"] = "OMITTED_FOR_BYTE_LIMIT"
            _validate_primary(case)
        data = encoded()
    if len(data) > limit:
        raise ValueError("Required scalar summaries exceed evidence byte limit")
    return data


def export_existing(core, full_path, out_dir, names=None, rank=None, max_bytes=LIMIT):
    if type(max_bytes) is not int or max_bytes < LIMIT:
        raise ValueError("max_bytes must be an integer of at least 65536")
    full_path = core._source_path(full_path)
    access = core._io_path(full_path)
    stamp = _stamp(access)
    raw = access.read_bytes()
    saved = json.loads(raw.decode("utf-8-sig"))
    checkpoint_input = saved.get("schema") == SCHEMA and saved.get("evidence_checkpoint") is True
    if saved.get("schema") != "u_host_stack_v1" and not checkpoint_input:
        raise ValueError("--export-existing expects u_host_stack_full.json or a full evidence checkpoint")
    wanted = names or ([c["case"] for c in saved["cases"]] if checkpoint_input else ["bs8-s7", "bs16-s7"])
    if len(wanted) != len(set(wanted)):
        raise ValueError("--case selections must be unique")
    chosen = []
    for name in wanted:
        found = [c for c in saved["cases"] if c.get("case") == name]
        if len(found) != 1:
            raise ValueError("Expected exactly one saved case: " + name)
        chosen.append(found[0])
    out = core._source_path(out_dir) / "u_host_stack_evidence.json"
    checkpoint = out.parent / "u_host_stack_evidence_full.json"
    protected = {full_path}
    for value in saved.get("checkpoint_sources", []):
        protected.add(_path(core, value, full_path.parent))
    for entry in saved["cases"]:
        for value in [entry.get("kernel_path"), (entry.get("trace") or {}).get("path"),
                      *((entry.get("stage_details") or {}).values()),
                      *(m.get("path") for m in entry.get("profiler_metadata", []))]:
            if value:
                protected.add(_path(core, value, full_path.parent))
    if out in protected or not checkpoint_input and checkpoint in protected:
        raise ValueError("Evidence output overlaps an existing source input")
    if checkpoint_input:
        print("Repacking saved evidence checkpoint; no trace or CSV scans", flush=True)
        result = copy.deepcopy(saved)
        result["cases"] = copy.deepcopy(chosen)
        result["repacked_from_checkpoint_sha256"] = hashlib.sha256(raw).hexdigest()
        result["repack_scope"] = "Saved collection and trace verification retained; source trace and CSV files are not rescanned"
        for case in result["cases"]:
            if rank is not None and case.get("rank") != rank:
                raise ValueError("Saved case rank differs from --rank")
        data = bounded_evidence(result, limit=max_bytes)
        if _stamp(access) != stamp:
            raise ValueError("Source checkpoint changed during repacking")
        core._io_path(out.parent).mkdir(parents=True, exist_ok=True)
        core._io_path(out).write_bytes(data)
        return json.loads(data), out
    result = {"schema": SCHEMA, "profile_version": saved.get("profile_version"), "diagnostic_only": True,
              "exporter_profile_version": core.PROFILE_VERSION,
              "full_report_sha256": hashlib.sha256(raw).hexdigest(), "rank": saved.get("rank"), "cases": [],
              "event_index_scope": "Window/function/source indexes refer to the original trace; trace_evidence chains and gap host_event_indexes refer to retained trace_evidence.events",
              "scope": "Existing-report examples and original trace parent evidence; no HC remeasurement, no extrapolation. "
                       "Function samples are not means; inferred containment is not a direct caller or critical-path proof."}
    for entry in chosen:
        print(f"Exporting existing evidence: {entry.get('case')} / {entry.get('rank')}", flush=True)
        case = {k: copy.deepcopy(entry.get(k)) for k in ("case", "rank", "bs", "S", "status", "error", "warnings",
                "declared_with_stack", "stack_metadata_verified", "profiler_metadata", "phase_summary", "window_summary",
                "all_stage_wall_means", "cycle_residual_ms_mean")}
        case["source_status"] = case.pop("status")
        case["source_error"] = case.pop("error")
        case["status"], case["error"] = "EXPORTED", None
        case["evidence_warning"] = "Saved Host source links and overlap values are producer results; " \
                                   "broad Python ancestors are not revalidated here as causal links"
        # Means already computed from all candidates stay intact; only examples
        # and supporting function rows are bounded in this export.
        for row in case.get("window_summary") or []:
            values = row.get("top_exclusive_functions", [])
            row["top_exclusive_functions"] = values[:20]
            row["evidence_function_rows_omitted"] = max(0, len(values) - 20)
        try:
            if rank is not None and entry.get("rank") != rank:
                raise ValueError("Saved case rank differs from --rank")
            case["windows"] = _select_windows(entry)
            case["stage7"] = _stage_evidence(core, entry, full_path.parent)
            case["source_links"] = _sources(core, entry, full_path.parent,
                                            case["stage7"]["gaps"] + case["stage7"].get("main_inputs_gaps", []))
            case["trace_evidence"] = _trace_evidence(core, entry, case["windows"], case["stage7"], case["source_links"], full_path.parent)
        except (OSError, ValueError, KeyError, TypeError, ArithmeticError, csv.Error) as error:
            case["status"], case["error"] = "ERROR", type(error).__name__ + ": " + str(error)
        result["cases"].append(case)
    if _stamp(access) != stamp:
        raise ValueError("Source full report changed during evidence export")
    core._io_path(out.parent).mkdir(parents=True, exist_ok=True)
    # Save the complete collection first. A budget error or later export issue
    # can then be retried without reading million-event traces again.
    result["evidence_checkpoint"] = True
    result["checkpoint_sources"] = sorted(str(p) for p in protected if p is not None)
    core.atomic_json(checkpoint, result)
    print("Full evidence checkpoint: " + str(checkpoint), flush=True)
    data = bounded_evidence(result, limit=max_bytes)
    core._io_path(out).write_bytes(data)
    return json.loads(data), out
