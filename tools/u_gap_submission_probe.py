#!/usr/bin/env python3
"""Join existing U13/14 device gaps to their recorded CANN submission times."""

import argparse
from collections import Counter, defaultdict
import csv
from decimal import Decimal
import json
from pathlib import Path
import sys


PHASES = ("main_inputs", "main_metadata")
PARTS = ("before_node_ns", "inside_node_ns", "after_node_ns", "unknown_ns")
TOTALS = (*PARTS, "duration_ns", "node_returned_before_gap_ns", "queue_wait_in_gap_ns")


def row_ids(value):
    return [int(item) for item in (json.loads(value) if isinstance(value, str) and value else value or [])]


def classify_gap(gap, links, devices):
    """Partition a CSV gap around its uniquely linked following CANN Node."""
    left, right = int(gap["start_ns"]), int(gap["end_ns"])
    if right <= left:
        raise ValueError("A device gap must have positive duration")
    result = dict.fromkeys(PARTS, 0)
    result.update(duration_ns=right - left, status="UNKNOWN", operator="unknown",
                  node_returned_before_gap_ns=0, queue_wait_in_gap_ns=0)
    follows = row_ids(gap.get("following_row_indices"))
    boundary = not follows
    if boundary:
        follows = sorted({int(task["row_index"]) for (round_id, _), task in devices.items()
                          if round_id == gap["round_id"] and int(task["start_ns"]) == right})
    tasks = {}
    for identity in follows:
        task = devices.get((gap["round_id"], identity))
        if task is None or int(task["start_ns"]) != right:
            tasks[identity] = None
        else:
            tasks[int(task["row_index"])] = task
    example = {"round_id": gap["round_id"], "gap_ns": right - left,
               "stage_boundary": boundary, "following_rows": follows[:6],
               "previous_rows": row_ids(gap.get("previous_row_indices"))[:6]}
    previous = {devices[(gap["round_id"], identity)].get("name", "unknown")
                for identity in row_ids(gap.get("previous_row_indices"))
                if (gap["round_id"], identity) in devices}
    example["previous_names"] = sorted(name[:140] for name in previous)[:3]
    result["example"] = example
    if len(tasks) != 1 or next(iter(tasks.values()), None) is None:
        result.update(unknown_ns=right - left, status="MISSING_OR_MULTIPLE_FOLLOWING_TASKS")
        return result
    identity, task = next(iter(tasks.items()))
    result["operator"] = task.get("name", "unknown")
    example.update(following_name=task.get("name", "")[:140], stream_id=task.get("stream_id"),
                   model_id=task.get("model_id"), task_id=task.get("task_id"),
                   wait_ms_diagnostic=task.get("wait_ms_diagnostic"))
    source = links.get(identity)
    if (source is None or source.get("status") not in ("LINKED", "API_ONLY")
            or not source.get("node_start_ns") or not source.get("node_end_ns")
            or int(source["device_start_ns"]) != right
            or int(source["device_end_ns"]) != int(task["end_ns"])):
        result.update(unknown_ns=right - left, status="NO_UNIQUE_NODE_SOURCE")
        example["source_status"] = source.get("status") if source else "MISSING_OR_CONFLICTING"
        return result
    start, end = int(source["node_start_ns"]), int(source["node_end_ns"])
    if end < start or start > right:
        result.update(unknown_ns=right - left, status="INVALID_NODE_ORDER")
        return result
    result.update(before_node_ns=max(0, min(right, start) - left),
                  inside_node_ns=max(0, min(right, end) - max(left, start)),
                  after_node_ns=max(0, right - max(left, end)), status="NODE_LINKED",
                  node_returned_before_gap_ns=right - left if end <= left else 0)
    example.update(source_status=source["status"], cpu_name=source.get("cpu_name", "")[:140],
                   cpu_leaf_name=source.get("cpu_leaf_name", "")[:140])
    for key in ("node_start_ns", "node_end_ns", "cpu_start_ns", "cpu_end_ns", "enqueue_start_ns",
                "enqueue_end_ns", "dequeue_start_ns", "dequeue_end_ns"):
        example[key.replace("_ns", "_relative_ns")] = int(source[key]) - left if source.get(key) else None
    if source.get("enqueue_end_ns") and source.get("dequeue_start_ns"):
        queue_left, queue_right = int(source["enqueue_end_ns"]), int(source["dequeue_start_ns"])
        if queue_right >= queue_left:
            result["queue_wait_in_gap_ns"] = max(0, min(right, queue_right) - max(left, queue_left))
    if sum(result[key] for key in PARTS) != right - left:
        raise ValueError("Submission partitions do not close to the CSV gap")
    example.update({key: result[key] for key in PARTS})
    return result


def read_rows(core, path):
    with core._io_path(path).open(encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle)


def resolve_file(core, value, report_dir):
    if not value:
        raise ValueError("This report has no saved focus or source-link export")
    path = Path(value)
    candidates = [path] if path.is_absolute() else [report_dir.parent / path, report_dir / path]
    for candidate in candidates:
        if core._io_path(candidate).is_file():
            return candidate
    raise FileNotFoundError("Saved diagnostic CSV is unavailable: " + path.name)


def analyze_capture(core, diagnostic, rank, report_dir):
    focus = diagnostic.get("focus_diagnostics") or {}
    segment_path = resolve_file(core, focus.get("segments_csv"), report_dir)
    device_path = resolve_file(core, diagnostic.get("device_tasks_csv"), report_dir)
    devices = {}
    for row in read_rows(core, device_path):
        if row["rank"] != rank:
            raise ValueError("Device CSV rank differs from selected rank")
        for identity in row_ids(row.get("row_indices")) or [int(row["row_index"])]:
            key = row["round_id"], identity
            old = devices.get(key)
            if old is not None and any(old.get(field) != row.get(field) for field in
                                       ("start_ns", "end_ns", "name", "stream_id", "model_id", "task_id")):
                raise ValueError("Conflicting device task rows")
            devices[key] = row
    links = {}
    aligned = (diagnostic.get("trace") or {}).get("status") == "ALIGNED_X_EVENTS"
    if aligned and diagnostic.get("source_links_csv"):
        for row in read_rows(core, resolve_file(core, diagnostic["source_links_csv"], report_dir)):
            identity = int(row["row_index"])
            if identity in links and links[identity] != row:
                links[identity] = None
            else:
                links[identity] = row
    groups = {}
    for stage in focus.get("stages", []):
        key = stage.get("stage_key")
        if key not in PHASES:
            continue
        steps = {}
        for step in stage.get("per_step", []):
            if step.get("status") != "MATCHED":
                continue
            if step["round_id"] in steps:
                raise ValueError("Duplicate retained phase round")
            if step.get("no_recorded_ms") is None:
                raise ValueError("Report lacks per-round CSV gap totals")
            steps[step["round_id"]] = round(Decimal(str(step["no_recorded_ms"])) * 1_000_000)
        if key in groups:
            raise ValueError("Duplicate focus phase")
        groups[key] = {"rounds": len(steps), "expected_gap_ns": steps, "observed_gap_ns": Counter(),
                       "totals": Counter(dict.fromkeys(TOTALS, 0)), "counts": Counter(),
                       "operators": defaultdict(Counter), "examples": []}
    seen = set()
    for gap in read_rows(core, segment_path):
        if gap["rank"] != rank:
            raise ValueError("Focus CSV rank differs from selected rank")
        if gap.get("kind") != "no_device_record" or gap.get("stage_key") not in groups:
            continue
        identity = gap["stage_key"], gap["round_id"], gap["start_ns"], gap["end_ns"]
        if identity in seen:
            raise ValueError("Duplicate gap in focus export")
        seen.add(identity)
        group = groups[gap["stage_key"]]
        if gap["round_id"] not in group["expected_gap_ns"]:
            raise ValueError("Focus gap has no matched retained phase round")
        row = classify_gap(gap, links, devices)
        group["observed_gap_ns"][gap["round_id"]] += row["duration_ns"]
        group["counts"][row["status"]] += 1
        for key in TOTALS:
            group["totals"][key] += row[key]
        group["operators"][row["operator"]].update({key: row[key] for key in (*PARTS, "duration_ns")})
        group["examples"].append(row["example"])
        group["examples"] = sorted(group["examples"], key=lambda item: -item["gap_ns"])[:2]
    output = []
    for key in PHASES:
        if key not in groups or not groups[key]["rounds"]:
            output.append({"stage_key": key, "error": "No matched retained phase rounds"})
            continue
        group = groups[key]
        if any(group["observed_gap_ns"][identity] != expected
               for identity, expected in group["expected_gap_ns"].items()):
            raise ValueError("Focus gap totals differ from retained report rounds")
        rounds = group["rounds"]
        total = group["totals"]
        row = {"stage_key": key, "rounds": rounds, "trace_aligned": aligned,
               "gap_count": sum(group["counts"].values()), "gap_status_counts": dict(group["counts"]),
               "unsampled_gaps": sum(group["counts"].values()) - len(group["examples"]),
               "means_ms": {name.replace("_ns", ""): value / rounds / 1e6 for name, value in total.items()},
               "top_following_operators": [], "largest_gaps": group["examples"]}
        for name, values in sorted(group["operators"].items(), key=lambda pair: -pair[1]["duration_ns"])[:3]:
            row["top_following_operators"].append({"name": name[:140], "means_ms": {
                field.replace("_ns", ""): value / rounds / 1e6 for field, value in values.items()}})
        output.append(row)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report_dir", type=Path)
    parser.add_argument("--rank", default="dp4")
    parser.add_argument("--out", type=Path, default=Path("u_gap_dp4.json"))
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import profile_ep128 as core

    report = json.loads(core._io_path(args.report_dir / "model_architecture_report.json").read_text(encoding="utf-8-sig"))
    result = {"schema": "u_gap_submission_v1", "profile_version": report.get("profile_version"), "rank": args.rank,
              "scope": "CSV gaps split by the following task's recorded CANN Node; after-return time has no assigned wait cause; queue overlap is nonadditive; DMA placement is absent from these CSV gaps",
              "cases": [], "omitted_examples": 0}
    failed = False
    for case in report["cases"]:
        captures = [item for item in case.get("measurements", []) if item.get("rank") == args.rank]
        config = case.get("config") or {}
        row = {"case": case["name"], "bs": config.get("bs"), "S": config.get("draft_tokens")}
        try:
            if len(captures) != 1:
                raise ValueError("Expected exactly one selected-rank measurement")
            row["stages"] = analyze_capture(core, captures[0].get("step_residuals") or {}, args.rank, args.report_dir)
            failed |= any("error" in stage for stage in row["stages"])
        except (OSError, ValueError, KeyError, TypeError) as error:
            row["error"] = type(error).__name__ + ": " + str(error)
            failed = True
        result["cases"].append(row)
    def payload():
        return (json.dumps(result, ensure_ascii=True, allow_nan=False, separators=(",", ":")) + "\n").encode("ascii")
    data = payload()
    for case in reversed(result["cases"]):
        for stage in reversed(case.get("stages", [])):
            while len(data) > 65536 and stage.get("largest_gaps"):
                stage["largest_gaps"].pop()
                stage["unsampled_gaps"] += 1
                result["omitted_examples"] += 1
                data = payload()
    if len(data) > 65536:
        raise ValueError("Aggregate diagnostic output exceeds 64 KiB")
    core._io_path(args.out).write_bytes(data)
    print(str(args.out.resolve()) + " (" + str(len(data)) + " bytes)")
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
