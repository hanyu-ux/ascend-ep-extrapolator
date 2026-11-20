#!/usr/bin/env python3
"""Inspect recorded Host functions before input expansion and metadata enqueue."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

from u_gap_submission_probe import read_rows, resolve_file, row_ids


def cpu_anchor(source, fragment):
    normalized = fragment.replace("_", "").lower()
    ancestors = json.loads(source.get("cpu_ancestors_json") or "[]")
    matches = [event for event in ancestors if normalized in event["name"].replace("_", "").lower()]
    if matches:
        event = matches[0]
        return {"anchor_event_index": int(event["id"]), "anchor_name": event["name"],
                "anchor_start_ns": int(event["start"]), "anchor_end_ns": int(event["end"])}
    if normalized in source.get("cpu_name", "").replace("_", "").lower() and source.get("cpu_event_index"):
        return {"anchor_event_index": int(source["cpu_event_index"]), "anchor_name": source["cpu_name"],
                "anchor_start_ns": int(source["cpu_start_ns"]), "anchor_end_ns": int(source["cpu_end_ns"])}
    return None


def read_evidence(core, diagnostic, report_dir, rank):
    devices, sources, members, gaps = {}, {}, defaultdict(set), {}
    for task in read_rows(core, resolve_file(core, diagnostic.get("device_tasks_csv"), report_dir)):
        if task["rank"] != rank:
            raise ValueError("Device export rank differs from selected rank")
        key = task["round_id"], int(task["row_index"])
        old = devices.get(key)
        if old and any(old.get(field) != task.get(field) for field in
                       ("start_ns", "end_ns", "name", "stream_id", "model_id", "task_id")):
            raise ValueError("Conflicting device task identity")
        devices[key] = task
    for source in read_rows(core, resolve_file(core, diagnostic.get("source_links_csv"), report_dir)):
        identity = int(source["row_index"])
        sources[identity] = None if identity in sources and sources[identity] != source else source
    aliases = {}
    for key, task in devices.items():
        for identity in row_ids(task.get("row_indices")) or [key[1]]:
            alias = key[0], identity
            if alias in aliases and aliases[alias] != key:
                raise ValueError("Conflicting device row aliases")
            aliases[alias] = key
    focus = diagnostic.get("focus_diagnostics") or {}
    for segment in read_rows(core, resolve_file(core, focus.get("segments_csv"), report_dir)):
        if segment["rank"] != rank:
            raise ValueError("Focus export rank differs from selected rank")
        phase, round_id = segment["stage_key"], segment["round_id"]
        for identity in row_ids(segment.get("row_indices")):
            if (round_id, identity) in aliases:
                members[phase].add(aliases[round_id, identity])
        if segment["kind"] == "no_device_record":
            gaps[phase, round_id, int(segment["end_ns"])] = segment
    return devices, sources, members, gaps


def select_windows(core, diagnostic, report_dir, rank, samples):
    """Select longest examples, or all attributable windows when samples=None.

    Counts retain the historical ``sampled`` spelling even when all candidates
    are selected for an explicitly separate aggregate diagnostic.
    """
    if samples is not None and (type(samples) is not int or samples < 1):
        raise ValueError("samples must be a positive integer or None")
    devices, sources, members, gaps = read_evidence(core, diagnostic, report_dir, rank)
    candidates, unavailable, counts = defaultdict(dict), defaultdict(list), Counter()
    by_round = defaultdict(list)
    for (round_id, _), task in devices.items():
        by_round[round_id].append(task)

    def add(kind, left, right, task, anchor, gap, start_anchor=None):
        if right <= left:
            counts[kind + ":no_positive_interval"] += 1
            return
        window = {"kind": kind, "start_ns": left, "end_ns": right, **anchor,
                  "round_id": task["round_id"], "device_row": int(task["row_index"]),
                  "device_name": task["name"], "device_gap_ns":
                  int(gap["end_ns"]) - int(gap["start_ns"]) if gap else 0,
                  "device_start_ns": int(task["start_ns"])}
        if start_anchor:
            window["start_anchor"] = start_anchor
        identity = left, right, anchor["anchor_event_index"]
        candidates[kind][identity] = window

    for phase, fragment in (("main_inputs", "_expand_idx_mapping_kernel"),
                            ("main_metadata", "mixed_quant_sparse_flash_mla_metadata")):
        for key in sorted(members[phase]):
            task = devices[key]
            if fragment.replace("_", "") not in task["name"].replace("_", "").lower():
                continue
            counts[fragment + ":tasks"] += 1
            source = sources.get(int(task["row_index"]))
            gap = gaps.get((phase, key[0], int(task["start_ns"])))
            valid = (source is not None and source.get("status") in ("LINKED", "CPU_ONLY")
                     and int(source["device_start_ns"]) == int(task["start_ns"])
                     and int(source["device_end_ns"]) == int(task["end_ns"]))
            anchor = cpu_anchor(source, fragment) if valid else None
            if anchor is None:
                reason = source.get("status") if source else "MISSING_OR_CONFLICTING"
                counts[fragment + ":unavailable:" + str(reason)] += 1
                unavailable[phase].append({"round_id": key[0], "row_index": key[1],
                    "source_status": reason, "reason": source.get("reason", "")[:200] if source else "",
                    "gap_ns": int(gap["duration_ns"]) if gap else 0})
                continue
            if phase == "main_metadata":
                if (source.get("enqueue_start_ns") and anchor["anchor_start_ns"]
                        <= int(source["enqueue_start_ns"]) <= anchor["anchor_end_ns"]):
                    add("metadata_before_enqueue", anchor["anchor_start_ns"], int(source["enqueue_start_ns"]),
                        task, anchor, gap)
                else:
                    counts["metadata_before_enqueue:enqueue_outside_cpu_scope_or_missing"] += 1
                continue
            if gap:
                add("prepare_gap_before_expand", int(gap["start_ns"]), anchor["anchor_start_ns"], task, anchor, gap)
            predecessors = [previous for previous in by_round[key[0]] if previous["name"] == "_apply_write_kernel"
                            and int(previous["end_ns"]) <= int(task["start_ns"])]
            if predecessors:
                previous = max(predecessors, key=lambda event: int(event["end_ns"]))
                previous_source = sources.get(int(previous["row_index"]))
                previous_anchor = (cpu_anchor(previous_source, "_apply_write_kernel")
                                   if previous_source and previous_source.get("status") in ("LINKED", "CPU_ONLY")
                                   and int(previous_source["device_start_ns"]) == int(previous["start_ns"])
                                   and int(previous_source["device_end_ns"]) == int(previous["end_ns"]) else None)
                if previous_anchor:
                    add("prepare_between_host_calls", previous_anchor["anchor_end_ns"], anchor["anchor_start_ns"],
                        task, anchor, gap, previous_anchor)
                else:
                    counts["prepare_between_host_calls:no_previous_cpu_source"] += 1

    selected = []
    for kind in ("prepare_gap_before_expand", "prepare_between_host_calls", "metadata_before_enqueue"):
        group = sorted(candidates[kind].values(), key=lambda row: (-(row["end_ns"] - row["start_ns"]), row["round_id"]))
        chosen = group if samples is None else group[:samples]
        counts[kind + ":candidates"] = len(group)
        counts[kind + ":sampled"] = len(chosen)
        for window in chosen:
            window["id"] = kind + ":" + str(len(selected) + 1)
            selected.append(window)
    return selected, dict(counts), {phase: sorted(rows, key=lambda row: -row["gap_ns"])[:2]
                                   for phase, rows in unavailable.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report_dir", type=Path)
    parser.add_argument("--case", dest="cases", action="append")
    parser.add_argument("--rank", default="dp4")
    parser.add_argument("--samples", type=int, choices=range(1, 5), default=2)
    parser.add_argument("--out", type=Path, default=Path("u_host_functions_dp4.json"))
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import profile_ep128 as core
    from ep128_host_function_trace import analyze_trace

    report_path = args.report_dir / "model_architecture_report.json"
    report = json.loads(core._io_path(report_path).read_text(encoding="utf-8-sig"))
    names = args.cases or ["bs8-s7", "bs16-s15"]
    if len(set(names)) != len(names):
        raise ValueError("Case names must be unique")
    result = {"schema": "u_host_function_v1", "profile_version": report.get("profile_version"), "rank": args.rank,
              "scope": "Bounded longest-window samples; exclusive recorded-scope wall time, with unrecorded and ambiguous time retained",
              "cases": [], "omitted_window_details": 0}
    protected = {core._source_path(report_path)}
    failed = False
    for name in names:
        entry = {"case": name}
        try:
            cases = [case for case in report["cases"] if case["name"] == name]
            if len(cases) != 1:
                raise ValueError("Expected exactly one requested case")
            measurements = [item for item in cases[0].get("measurements", []) if item.get("rank") == args.rank]
            if len(measurements) != 1:
                raise ValueError("Expected exactly one selected-rank measurement")
            diagnostic = measurements[0].get("step_residuals") or {}
            trace = diagnostic.get("trace") or {}
            if trace.get("status") != "ALIGNED_X_EVENTS":
                raise ValueError("Existing report must have an aligned trace")
            windows, counts, unavailable = select_windows(core, diagnostic, args.report_dir, args.rank, args.samples)
            entry.update(selection_counts=counts, largest_unresolved_sources=unavailable)
            if not windows:
                entry["status"] = "NO_ATTRIBUTABLE_WINDOWS"
                failed = True
                result["cases"].append(entry)
                continue
            trace_path = resolve_file(core, trace.get("path"), args.report_dir)
            protected.add(core._source_path(trace_path))
            print(name + ": scanning existing trace for " + str(len(windows)) + " Host windows", flush=True)
            scanned = analyze_trace(core, trace_path, trace, windows)
            identities = {window["id"]: window for window in windows}
            for window in scanned["windows"]:
                identity = identities[window["id"]]
                window.update(round_id=identity["round_id"], device_row=identity["device_row"],
                              device_gap_ns=identity["device_gap_ns"],
                              window_start_relative_to_device_ns=identity["start_ns"] - identity["device_start_ns"])
            entry.update(scanned)
            failed |= any(window["status"] != "MATCHED" for window in scanned["windows"])
        except (OSError, ValueError, KeyError, TypeError, ArithmeticError) as error:
            entry["error"] = type(error).__name__ + ": " + str(error)
            failed = True
        result["cases"].append(entry)

    def encoded():
        return (json.dumps(result, ensure_ascii=True, allow_nan=False, separators=(",", ":")) + "\n").encode("ascii")

    payload = encoded()
    for case in result["cases"]:
        for window in case.get("windows", []):
            if len(payload) <= 65536:
                break
            for field in ("inclusive_functions", "exclusive_functions", "longest_unrecorded"):
                window.pop(field, None)
            window["detail_omitted_for_byte_limit"] = True
            result["omitted_window_details"] += 1
            payload = encoded()
    if len(payload) > 65536:
        raise ValueError("Host diagnostic output exceeds 64 KiB")
    if core._source_path(args.out) in protected:
        raise ValueError("Output path overlaps a source input")
    core._io_path(args.out).write_bytes(payload)
    print(str(args.out.resolve()) + " (" + str(len(payload)) + " bytes)")
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
