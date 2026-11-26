#!/usr/bin/env python3
"""Export complete fixed-rank Host/HC swimlane coordinates from existing captures.

Reads schema_version=2 cases JSON, kernel CSV and trace JSON only. No model,
service, benchmark, accelerator package or new capture is started. All complete
source HC rounds and all recognized Host calls survive lossless ZIP/text export.
"""

import argparse
import base64
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
import csv
import gzip
import hashlib
import json
import multiprocessing
from pathlib import Path
import sys
import tempfile
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import profile_ep128 as core
from ep128_swimlane_trace import collect_host_events, pair_rounds
from ep128_swimlane_links import export_anchor_links


SCHEMA = "u_swimlane_v2"
TEXT_CHUNK = 60000
ROLE_KEYS = ("outer_propose", "runner_inputs", "state_attn")
STAMP_KEYS = {"start_ns", "end_ns", "origin_ns", "timestamp_ns", "clock_offset_ns",
              "csv_start_ns", "trace_start_ns"}


def stamp(path):
    info = core._io_path(path).stat()
    return info.st_size, info.st_mtime_ns


def wire(value, key=""):
    """Absolute epoch coordinates stay decimal strings for JS/PPT consumers."""
    if isinstance(value, dict):
        return {k: wire(v, k) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [wire(v) for v in value]
    if type(value) is int and (key in STAMP_KEYS or key.endswith("_start_ns") or key.endswith("_end_ns")):
        return str(value)
    return value


def task_anchor(task):
    return {key: getattr(task, key, None) for key in
            ("row_index", "name", "op_type", "start_ns", "end_ns", "device_id", "stream_id", "task_id", "model_id",
             "context_id", "batch_id", "subtask_id", "task_type")}


def round_coordinates(measured, capture, case):
    """Keep full parent-stage boundaries, including unresolved original windows."""
    windows = core.residual_windows(measured)
    index = core.ATimeIndex(sorted(capture["tasks"], key=lambda t: (t.start_ns, t.end_ns)))
    by_round = {}
    anchor_rows = set()
    for window in windows:
        left, right = int(window["start_ns"]), int(window["end_ns"])
        window_tasks = index.overlapping(left, right)
        stages = [r for r in core.residual_stage_plan(window, window_tasks, case)
                  if r["level"] == 0]
        if not stages or int(stages[0]["start_ns"]) != left or int(stages[-1]["end_ns"]) != right:
            raise ValueError("U stage endpoints do not close to the source window")
        if sum(int(r["end_ns"]) - int(r["start_ns"]) for r in stages) != right - left or any(
                a["end_ns"] != b["start_ns"] for a, b in zip(stages, stages[1:])):
            raise ValueError("U stages have overlapping or missing time")
        for row in stages:
            row["stage_number"] = row["stage_order"] + (8 if window["window_key"] == "draft_to_nextmain" else 0)
            for side in ("start_anchor", "end_anchor"):
                if row[side].get("row_index") is not None:
                    anchor_rows.add(row[side]["row_index"])
        saved_window = {**window, "stages": stages}
        if any(row["status"] == "UNRESOLVED" for row in stages):
            saved_window["unresolved_device_tasks"] = [
                {**task_anchor(task), **{key: getattr(task, key, None) for key in
                                       ("input_shapes", "output_shapes", "input_dtypes", "output_dtypes")}}
                for task in window_tasks]
        by_round.setdefault(window["round_id"], []).append(saved_window)
    rounds = []
    for source in measured["rounds"]:
        start, end = int(source["start_ns"]), int(source["end_ns"])
        regions = {}
        for role in ("main", "draft"):
            layers = sorted((r for r in source["layers"] if r["role"] == role), key=lambda r: int(r["start_ns"]))
            a, b = int(layers[0]["start_ns"]), int(layers[-1]["end_ns"])
            regions[role + "_hc"] = {"start_ns": a, "end_ns": b, "duration_ns": b-a}
            for task in capture["tasks"]:
                if (task.start_ns == a and (core.A_PRE.search(task.name) or core.A_PRE.search(task.op_type))
                        or task.end_ns == b and (core.A_POST.search(task.name) or core.A_POST.search(task.op_type))):
                    anchor_rows.add(task.row_index)
        own = by_round[source["round_id"]]
        u_ns = sum(int(w["end_ns"]) - int(w["start_ns"]) for w in own)
        if sum(r["duration_ns"] for r in regions.values()) + u_ns != end-start:
            raise ValueError("Main HC, draft HC and U do not close to the source cycle")
        all_stages = [r for w in own for r in w["stages"]]
        rounds.append({"round_id": source["round_id"], "start_ns": start, "end_ns": end,
                       "origin_ns": start, "duration_ns": end-start, "u_ns": u_ns,
                       "regions": regions, "u_windows": own,
                       "stage_status": "MATCHED" if len(all_stages) == 15 and all(
                           r["status"] == "MATCHED" for r in all_stages) else "UNRESOLVED"})
    return rounds, [task_anchor(t) for t in capture["tasks"] if t.row_index in anchor_rows]


def coordinate(event, origin):
    start, end = int(event["start_ns"]), int(event["end_ns"])
    return {"event_index": event["event_index"], "start_ns": start, "end_ns": end,
            "relative_start_ns": start-origin, "relative_end_ns": end-origin, "duration_ns": end-start}


def coordination_breakdown(prefix, child):
    """Disjoint wall ranges; verified nested waits never become additive costs."""
    left, right = int(prefix["start_ns"]), int(prefix["end_ns"])
    partition = child.get("prefix_partition", {})
    sync = partition.get("dp_sync", {}).get("events", [])
    waits = partition.get("wait_candidates", [])
    refs = child.get("by_role", {}).get("preparation_dp_aggregate", [])
    verified = lambda ref: ref.get("parent_chain_verified") is True
    contained = lambda ref: left <= int(ref["start_ns"]) < int(ref["end_ns"]) <= right
    span = lambda ref: (int(ref["start_ns"]), int(ref["end_ns"]))
    result = {"status": "PARTIAL", "duration_ns": right-left,
              "dp_coordination_wall_ns": None, "outside_coordination_wall_ns": None,
              "verified_dp_wait_union_ns": None, "wait_status": "NOT_RECORDED_OR_NOT_VERIFIED",
              "scope": "Wall intervals, not CPU busy or pure communication latency. DP sync includes its local setup and graph selection. "
                       "Nested all_reduce and wait scopes are not added to their parent."}
    if any(verified(r) and not contained(r) for r in sync):
        result["reason"] = "Verified DP scope crosses preparation aggregate boundary"
        return result
    d = core.a_union([span(r) for r in sync if verified(r) and contained(r)])
    def subtract(intervals, mask):
        return core.a_union([piece for a, b in intervals for piece in core.residual_complement(
            a, b, core.a_union([(max(a, c), min(b, e)) for c, e in mask if max(a,c)<min(b,e)]))])
    def amount(intervals):
        return sum(b-a for a,b in intervals)
    local_names = {"update_pp_decode_requests", "finish_requests", "free_states", "add_requests", "update_requests",
                   "apply_staged_writes", "gather_batch_req_state"}
    def name(ref):
        return str(ref.get("name", "")).rsplit(": ", 1)[-1]
    graph = core.a_union([span(r) for r in refs if verified(r) and contained(r) and name(r)=="dispatch"])
    graph = subtract(graph, d)
    local = core.a_union([span(r) for r in refs if verified(r) and contained(r) and name(r) in local_names])
    local = subtract(local, core.a_union(d+graph))
    rest = subtract([(left,right)], core.a_union(d+graph+local))
    categories = (("local_request_batch", local), ("local_graph_selection", graph),
                  ("dp_coordination", d), ("unattributed_wall", rest))
    parts = sorted(({"key": key, "start_ns": a, "end_ns": b, "duration_ns": b-a}
                    for key, intervals in categories for a,b in intervals), key=lambda p:p["start_ns"])
    if sum(p["duration_ns"] for p in parts) != right-left or any(
            a["end_ns"] != b["start_ns"] for a,b in zip(parts,parts[1:])):
        raise ValueError("Preparation/DP wall partition does not close")
    result.update(parts=parts, known_local_request_batch_ns=amount(local), local_graph_outside_dp_ns=amount(graph),
                  unattributed_wall_ns=amount(rest), unknown_sync_events=sum(not verified(r) for r in sync))
    if d:
        result.update(dp_coordination_wall_ns=amount(d), outside_coordination_wall_ns=right-left-amount(d))
        w = core.a_union([span(r) for r in waits if verified(r) and contained(r)])
        if w and subtract(w,d):
            result["reason"] = "Verified wait is outside verified DP synchronization scopes"
        elif w:
            result.update(verified_dp_wait_union_ns=amount(w), wait_status="PYTHON_PARENT_CHAIN_VERIFIED",
                          dp_other_wall_ns=amount(d)-amount(w))
        if partition.get("status") in ("MATCHED", "VERIFIED_HOST_BOUNDARIES") and not result.get("unknown_sync_events") and not result.get("reason"):
            result["status"] = "MATCHED"
    return result


def attach_host(rounds, pairs, events):
    """Store original call envelopes and separate temporal intersections."""
    lookup = {e["event_index"]: e for e in events}
    matches = {r["round_id"]: r for r in pairs}
    for row in rounds:
        pair = matches.get(row["round_id"], {"status": "PARTIAL", "functions": {}, "missing": ["round pairing"]})
        row["host_pairing"] = pair
        functions = pair.get("functions", {})
        row["host_lanes"] = {key: coordinate(lookup[functions[key]], row["origin_ns"])
                             for key in ROLE_KEYS if functions.get(key) in lookup}
        next_id, input_id = functions.get("next_execute"), functions.get("runner_inputs")
        if next_id in lookup and input_id in lookup:
            a, b = int(lookup[next_id]["start_ns"]), int(lookup[input_id]["start_ns"])
            if b < a:
                raise ValueError("Host preparation/DP aggregate endpoints are reversed")
            row["host_lanes"]["preparation_dp_aggregate"] = {
                "start_ns": a, "end_ns": b, "relative_start_ns": a-row["origin_ns"],
                "relative_end_ns": b-row["origin_ns"], "duration_ns": b-a,
                "start_event_index": next_id, "end_event_index": input_id,
                "scope": "Aggregate from next core execute entry to runner inputs entry; not DP wait duration"}
        child = pair.get("child_layer", {})
        row["host_children"] = {role: [{**ref, **coordinate(ref, row["origin_ns"])} for ref in refs]
                                for role, refs in child.get("by_role", {}).items()}
        if "preparation_dp_aggregate" in row["host_lanes"]:
            row["coordination_breakdown"] = coordination_breakdown(row["host_lanes"]["preparation_dp_aggregate"], child)
        regions = {**row["regions"], **{w["window_key"]: w for w in row["u_windows"]}}
        regions.update({"u" + str(stage["stage_number"]): stage
                        for w in row["u_windows"] for stage in w["stages"]})
        for lane in row["host_lanes"].values():
            lane["overlap_ns"] = {name: max(0, min(int(lane["end_ns"]), int(bounds["end_ns"]))
                                              - max(int(lane["start_ns"]), int(bounds["start_ns"])))
                                  for name, bounds in regions.items()}
    return rounds


def detail_rounds(rounds, limit):
    """Suggest complete business families without discarding any saved round."""
    ready = [r for r in rounds if r["host_pairing"]["status"] == "MATCHED"
             and len(r["host_lanes"]) == 4 and r["stage_status"] == "MATCHED"]
    def detailed(row):
        child = row["host_pairing"].get("child_layer", {})
        roles = child.get("by_role", {})
        return (row.get("coordination_breakdown", {}).get("status") == "MATCHED"
                and all(roles.get(role) for role in ROLE_KEYS))
    candidates = [r for r in ready if detailed(r)] or ready
    ordered = sorted(candidates, key=lambda r: (r["duration_ns"], r["round_id"]))
    if not ordered:
        return [], "PARTIAL"
    middle = len(ordered)//2
    sequence = sorted(range(len(ordered)), key=lambda i: (abs(i-middle), i))[:limit]
    return [ordered[i] for i in sequence], "READY" if any(detailed(r) for r in candidates) else "PARTIAL"


def analyze_case(item, config, base, rank, progress=print, detail_limit=1):
    result = {"case": item.get("name", "invalid"), "rank": rank, "status": "ERROR", "error": None,
              "rounds": [], "host_events": [], "device_anchors": [], "complete_rounds": 0,
              "incomplete_rounds": 0, "ready_requires_all_rounds": False, "detail_status": "UNAVAILABLE",
              "ready_definition": "At least one retained round has all four verified Host scopes and 15 matched U stages; "
                                  "READY does not mean every retained round is complete"}
    try:
        case = core.normalize_case(config.get("defaults", {}), item, base,
                                   capture_purpose="host_stack_diagnostic")
        result.update(bs=case["bs"], S=case["draft_tokens"], declared_with_stack=case["with_stack"])
        model = core.referenced_object(config["sample_model"], base, "sample_model")
        core.validate_models(model, None)
        result["source_model"] = {key: model.get(key) for key in
                                  ("hidden_size", "vocab_size", "num_hidden_layers", "compress_ratios")}
        result["diagnostic_config"] = {key: case.get(key) for key in ("draft_layers", "draft_passes")}
        kernel = core._source_path(core._fixed_overlap_input(case, rank, base))
        trace = core._source_path(core._case_trace_path(case, rank, base) or kernel.parent / "trace_view.json")
        before = {p: stamp(p) for p in (kernel, trace)}
        progress("Reading existing kernel CSV and resolving every complete HC cycle")
        capture = core.read_capture(kernel, rank)
        diagnostic_case = {**case, "trim_rounds": 0}
        measured = core.measure_capture(capture, model, diagnostic_case, base)
        result.update(kernel_path=str(kernel), kernel_sha256=capture["sha256"], trace_path=str(trace),
                      declared_trim_rounds=case.get("trim_rounds", 2), exported_trim_rounds=0)
        result["rounds"], result["device_anchors"] = round_coordinates(measured, capture, diagnostic_case)
        result["incomplete_rounds"] = len(result["rounds"])
        progress("Pass 1/4: verifying trace/device clock anchors")
        clock, offset = core.residual_trace_scan(trace, capture["tasks"], progress)
        result["trace"] = clock
        if clock.get("status") != "ALIGNED_X_EVENTS" or offset is None:
            raise ValueError("Trace alignment failed: " + str(clock.get("status")))
        progress("Pass 2/4: retaining Host children, coordination, wait and native API calls")
        host = collect_host_events(core, trace, offset, clock["sha256"], progress)
        result["host_events"] = host["events"]
        result["host_scan"] = {key: value for key, value in host.items() if key != "events"}
        pairs = pair_rounds(core, measured["rounds"], host["events"])
        result["rounds"] = attach_host(result["rounds"], pairs, host["events"])
        ready = [r for r in result["rounds"] if r["host_pairing"]["status"] == "MATCHED"
                 and len(r["host_lanes"]) == 4 and r["stage_status"] == "MATCHED"]
        result["complete_rounds"] = len(ready)
        result["incomplete_rounds"] = len(result["rounds"]) - len(ready)
        result["complete_host_rounds"] = sum(len(r["host_lanes"]) == 4 for r in result["rounds"])
        if ready:
            selected, result["detail_status"] = detail_rounds(result["rounds"], detail_limit)
            result["suggested_round_id"] = selected[0]["round_id"]
            result["source_link_round_ids"] = [r["round_id"] for r in selected]
            result["source_link_scope"] = "Detailed source links for these complete representative rounds; all rounds and target Host calls remain retained"
            progress("Pass 3/4: resolving device-anchor CPU/API/queue/flow sources")
            try:
                with tempfile.TemporaryDirectory(prefix="u-swimlane-links-") as directory:
                    result["anchor_links"] = export_anchor_links(core, capture, selected, host["events"],
                        trace, clock, offset, Path(directory), progress)
            except Exception as error:
                result["anchor_links"] = {"status": "ERROR", "error": type(error).__name__+": "+str(error),
                                          "scope": "Valid Host/HC coordinates are preserved; no device arrows are confirmed"}
                result["detail_status"] = "PARTIAL"
            result["status"] = "READY"
        else:
            result["status"] = "PARTIAL"
            result["error"] = "No round has both four verified Host call scopes and all 15 matched U stages"
        result["round_selection"] = "All complete CSV HC rounds retained; suggested round is median-cycle among complete candidates"
        if any(stamp(path) != info for path, info in before.items()):
            raise ValueError("An input changed during offline export")
    except Exception as error:
        result["status"], result["error"] = "ERROR", type(error).__name__ + ": " + str(error)
    return result


def worker(index, item, config, base, rank, detail_limit=1):
    with redirect_stdout(core._CaseProgressStream(sys.stdout, item["name"] + " / " + rank)):
        return index, analyze_case(item, config, base, rank, lambda message: print(message, flush=True), detail_limit)


def worker_failure(index, item, rank, error, stage):
    return index, {"case": item["name"], "rank": rank, "status": "ERROR",
                   "error": "CASE_WORKER_ERROR (" + stage + "): " + type(error).__name__ + ": " + str(error),
                   "rounds": [], "host_events": [], "complete_rounds": 0, "incomplete_rounds": 0}


def checked_outcome(outcome, index, item, rank):
    if (not isinstance(outcome, tuple) or len(outcome) != 2 or outcome[0] != index
            or not isinstance(outcome[1], dict) or outcome[1].get("case") != item["name"]
            or outcome[1].get("rank") != rank):
        raise ValueError("Case worker identity mismatch")
    return outcome


def run_cases(items, config, base, rank, workers, detail_limit=1):
    if workers == 1:
        for index, item in enumerate(items):
            try:
                outcome = checked_outcome(worker(index, item, config, base, rank, detail_limit), index, item, rank)
            except Exception as error:
                outcome = worker_failure(index, item, rank, error, "serial")
            yield outcome
        return
    try:
        executor = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"))
    except Exception as error:
        for index, item in enumerate(items):
            yield worker_failure(index, item, rank, error, "pool")
        return
    pending, reported = {}, set()
    try:
        for index, item in enumerate(items):
            try:
                future = executor.submit(worker, index, item, config, base, rank, detail_limit)
            except Exception as error:
                reported.add(index)
                yield worker_failure(index, item, rank, error, "submit")
            else:
                pending[future] = index, item
        try:
            for future in as_completed(pending):
                index, item = pending[future]
                try:
                    outcome = checked_outcome(future.result(), index, item, rank)
                except Exception as error:
                    outcome = worker_failure(index, item, rank, error, "collect")
                reported.add(index)
                yield outcome
        except Exception as error:
            for index, item in pending.values():
                if index not in reported:
                    reported.add(index)
                    yield worker_failure(index, item, rank, error, "collect")
    finally:
        try:
            executor.shutdown(wait=True)
        except Exception as error:
            # Already collected per-case evidence remains useful and must still be packaged.
            print("Worker pool shutdown warning: " + type(error).__name__ + ": " + str(error), flush=True)


def write_packages(payload, out, rank):
    """No byte budget: compress and split every byte, never remove coordinates."""
    stem = "u_swimlane_" + rank
    normalized = wire(payload)
    raw = (json.dumps(normalized, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n").encode("utf-8")
    destination = out / (stem + ".json")
    # The checkpoint is durable before any packaging step.
    with core.atomic_writer(core._io_path(destination)) as stream:
        stream.write(raw.decode("utf-8"))
    # ZIP and text parts contain the exact compact JSON payload, with its SHA.
    packed = gzip.compress(raw, compresslevel=9, mtime=0)
    data = base64.b64encode(packed).decode("ascii")
    count = max(1, (len(data)+TEXT_CHUNK-1)//TEXT_CHUNK)
    raw_sha, gzip_sha = hashlib.sha256(raw).hexdigest(), hashlib.sha256(packed).hexdigest()
    zip_path = out / (stem + ".zip")
    with zipfile.ZipFile(core._io_path(zip_path), "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        info = zipfile.ZipInfo(stem + ".json", date_time=(2020, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        archive.writestr(info, raw)
    names = []
    parts_dir = out / (stem + "_parts_" + raw_sha[:12])
    core._io_path(parts_dir).mkdir(parents=True, exist_ok=True)
    for index in range(count):
        envelope = {"schema": "u_swimlane_gzip_base64_v1", "part": index+1, "parts": count,
                    "raw_bytes": len(raw), "raw_sha256": raw_sha, "gzip_sha256": gzip_sha,
                    "encoding": "gzip+base64", "data": data[index*TEXT_CHUNK:(index+1)*TEXT_CHUNK]}
        blob = (json.dumps(envelope, ensure_ascii=True, separators=(",", ":"))+"\n").encode("ascii")
        if len(blob) > 65536:
            raise ValueError("Lossless text part exceeds 64 KiB")
        path = parts_dir / ("part" + str(index+1).zfill(3) + ".txt")
        core._io_path(path).write_bytes(blob)
        names.append(path.relative_to(out).as_posix())
    manifest = {"schema": "u_swimlane_package_v1", "raw_sha256": raw_sha, "raw_bytes": len(raw),
                "zip": zip_path.name, "parts": names, "gzip_sha256": gzip_sha,
                "truncated": False, "send": "Send the ZIP, or every text part listed here; do not send only the first part"}
    core.atomic_json(core._io_path(out / (stem + "_package.json")), manifest)
    return zip_path, names


def write_status(payload, out, rank):
    fields = ("case", "rank", "bs", "S", "status", "detail_status", "rounds", "complete_host_rounds", "complete_rounds", "incomplete_rounds",
              "suggested_round_id", "trace_status", "anchor_link_status", "error")
    with core._io_path(out / ("u_swimlane_" + rank + "_status.csv")).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        for entry in payload["cases"]:
            row = {key: entry.get(key) for key in fields}
            row.update(rounds=len(entry.get("rounds", [])), trace_status=(entry.get("trace") or {}).get("status"),
                       anchor_link_status=(entry.get("anchor_links") or {}).get("status"))
            writer.writerow(row)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", type=Path, help="Existing schema_version=2 cases JSON, not trace_view.json")
    parser.add_argument("--rank", default="dp4")
    parser.add_argument("--workers", type=core._cli_positive_workers, default=None)
    parser.add_argument("--case", dest="names", action="append", help="Optional case selection; default exports ALL cases")
    parser.add_argument("--expected-cases", type=core._cli_positive_workers)
    parser.add_argument("--detail-rounds", type=core._cli_positive_workers, default=1,
                        help="Complete representative rounds per case for device-source links; all Host calls and HC/U rounds are still retained")
    parser.add_argument("--out", type=Path, default=Path("u_swimlane_all_dp4"))
    args = parser.parse_args(argv)
    if not args.rank or any(c not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.-" for c in args.rank):
        parser.error("--rank must be a simple rank identifier")
    manifest = core._source_path(args.cases)
    try:
        config = json.loads(core._io_path(manifest).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError, UnicodeError) as error:
        parser.error("Cannot read cases JSON: " + str(error))
    if not isinstance(config, dict) or config.get("schema_version") != core.SCHEMA_VERSION or not isinstance(config.get("cases"), list):
        parser.error("Provide the existing schema_version=2 cases JSON")
    names = [item.get("name") for item in config["cases"] if isinstance(item, dict)]
    if len(names) != len(config["cases"]) or not names or any(not isinstance(n, str) or not n for n in names) or len(set(names)) != len(names):
        parser.error("Cases must have unique nonempty names")
    if args.names and (len(set(args.names)) != len(args.names) or set(args.names)-set(names)):
        parser.error("--case names must be unique and present in cases JSON")
    selected = [item for item in config["cases"] if not args.names or item["name"] in args.names]
    if args.expected_cases is not None and len(selected) != args.expected_cases:
        parser.error("Expected " + str(args.expected_cases) + " cases, found " + str(len(selected)))
    out = core._source_path(args.out)
    protected = {manifest}
    for key in ("sample_model", "target_model", "communication_reference"):
        if isinstance(config.get(key), str):
            p = Path(config[key]); protected.add(core._source_path(p if p.is_absolute() else manifest.parent/p))
    for item in config["cases"]:
        raw = {**config.get("defaults", {}), **item}
        explicit = raw.get("inputs")
        sources = [source for source in explicit if isinstance(source, dict)] if isinstance(explicit, list) else []
        for source in sources:
            for key in ("path", "trace_path"):
                if isinstance(source.get(key), str) and source[key]:
                    p = Path(source[key]).expanduser()
                    protected.add(core._source_path(p if p.is_absolute() else manifest.parent/p))
        checks = [({**raw, "inputs": [source]}, source.get("rank")) for source in sources]
        if explicit is None:
            checks.append((raw, args.rank))
        for source_case, source_rank in checks:
            if not isinstance(source_rank, str) or not source_rank:
                continue
            try:
                kernel = core._source_path(core._fixed_overlap_input(source_case, source_rank, manifest.parent))
                protected.add(kernel)
                protected.add(core._source_path(core._case_trace_path(source_case, source_rank, manifest.parent)
                                                or kernel.parent/"trace_view.json"))
            except (ValueError, KeyError, OSError, TypeError):
                pass  # Raw explicit paths stay protected even when a case cannot be normalized.
    stem = "u_swimlane_" + args.rank
    for p in protected:
        try:
            relative = p.relative_to(out)
        except ValueError:
            continue
        if (p.parent == out and p.name.casefold().startswith(stem.casefold())
                or relative.parts and relative.parts[0].casefold().startswith(stem.casefold()+"_parts_")):
            parser.error("Output directory would overwrite a source input")
    core._io_path(out).mkdir(parents=True, exist_ok=True)
    workers = core._cli_case_workers(args.workers, len(selected), "swimlane case (fixed " + args.rank + ")")
    payload = {"schema": SCHEMA, "profile_version": core.PROFILE_VERSION, "rank": args.rank,
               "manifest_sha256": hashlib.sha256(core._io_path(manifest).read_bytes()).hexdigest(),
               "planned_cases": len(selected), "completed_cases": 0,
               "execution": {"parallelism": "case", "workers": workers}, "cases": [],
               "detail_rounds_per_case": args.detail_rounds,
               "trace_passes_per_aligned_case": 4,
               "diagram_contract": {"host_color_by": "function_kind", "host_labels": "function name and wall time",
                                    "show_hc_overlap_labels": False, "children_depth_to_display": 1,
                                    "confirmed_device_edge": "solid", "inferred_or_source_only_edge": "dashed",
                                    "cpu_wait": "Do not invent a directly submitted device kernel"},
               "retention": {"all_complete_hc_rounds": True, "all_recognized_host_calls": True,
                             "host_calls_clipped": False, "byte_budget_truncation": False},
               "scope": "Existing CSV/trace offline diagnostic only. HC envelopes are not full graphs or continuous busy time. "
                        "Host calls are full wall ranges. Round association by Host order/graph entry is labeled inference, "
                        "not a complete device flow proof. Missing/ambiguous/unaligned data are never filled from other rounds."}
    complete = {}
    for index, entry in run_cases(selected, config, manifest.parent, args.rank, workers, args.detail_rounds):
        complete[index] = entry
        payload["cases"] = [complete[k] for k in sorted(complete)]
        payload["completed_cases"] = len(complete)
        core.atomic_json(core._io_path(out/("u_swimlane_"+args.rank+".json")), wire(payload))
        print(f"[{len(complete)}/{len(selected)}] {entry['case']} {entry['status']}; complete rounds={entry.get('complete_rounds', 0)}", flush=True)
    write_status(payload, out, args.rank)
    archive, parts = write_packages(payload, out, args.rank)
    ready = sum(e["status"] == "READY" for e in payload["cases"])
    detailed = sum(e.get("detail_status") == "READY" for e in payload["cases"])
    print(f"READY={ready}/{len(selected)}; child-detail READY={detailed}/{len(selected)}; full coordinates were not truncated", flush=True)
    print("Send ZIP: " + str(archive), flush=True)
    print("Or send ALL " + str(len(parts)) + " text parts listed in u_swimlane_"+args.rank+"_package.json", flush=True)
    return int(ready != len(selected))


if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.exit(main())
