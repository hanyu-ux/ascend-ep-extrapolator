#!/usr/bin/env python3
"""Read the existing dp4 S15 profiler inputs and export only diagnostic evidence."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys


def diagnose_graph(core, capture, measured, case):
    from ep128_next_main_overlap import _draft_graph

    task_index = core.ATimeIndex(sorted(capture["tasks"], key=lambda task: (task.start_ns, task.end_ns)))
    parents = defaultdict(dict)
    for window in core.residual_windows(measured):
        start, end = (core.residual_ns(window[k]) for k in ("start_ns", "end_ns"))
        stages = core.residual_stage_plan(window, task_index.overlapping(start, end), case)
        parents[window["round_id"]].update(
            (stage["stage_key"], stage) for stage in stages if stage["level"] == 0
        )

    failed_counts, reason_counts, examples = Counter(), Counter(), {}
    matched_rounds = 0
    for cycle in measured["rounds"]:
        round_id = cycle["round_id"]
        draft = [layer for layer in cycle["layers"] if layer["role"] == "draft"]
        hc = int(draft[0]["start_ns"]), int(draft[-1]["end_ns"])
        stages = parents[round_id]
        graph, reason = _draft_graph(cycle, stages, task_index, hc)
        checks, evidence = _graph_checks(cycle, stages, task_index, hc)
        failed = tuple(key for key, ok in checks.items() if not ok)
        if bool(failed) != (graph is None):
            raise RuntimeError("Graph probe disagrees with the installed draft graph detector")
        matched_rounds += graph is not None
        reason_counts[reason or "MATCHED"] += 1
        failed_counts.update(failed)
        if failed not in examples:
            examples[failed] = {"round_id": round_id, "failed": list(failed), "rounds": 0, **evidence}
        examples[failed]["rounds"] += 1

    ordered = sorted(examples.values(), key=lambda row: (-row["rounds"], row["round_id"]))
    return {
        "rounds": len(measured["rounds"]),
        "matched_rounds": matched_rounds,
        "reasons": dict(reason_counts),
        "failed_check_counts": dict(failed_counts),
        "failure_patterns_omitted": max(0, len(ordered) - 6),
        "examples": ordered[:6],
    }


def _graph_checks(cycle, parents, task_index, hc):
    embedding, markov, next_inputs = [parents.get(key) for key in ("draft_embedding", "markov", "main_inputs")]
    checks = {
        key + "_anchor_matched": stage is not None and stage["status"] == "MATCHED"
        for key, stage in zip(("embedding", "markov", "next_inputs"), (embedding, markov, next_inputs))
    }
    if not all(checks.values()):
        return checks, {}

    main_end = max(int(layer["end_ns"]) for layer in cycle["layers"] if layer["role"] == "main")
    next_start = int(next_inputs["start_ns"])
    tasks = task_index.overlapping(main_end, next_start)
    embedding_row = embedding["start_anchor"].get("row_index")
    markov_row = markov["start_anchor"].get("row_index")
    matches = [task for task in tasks if task.row_index == embedding_row and task.start_ns == int(embedding["start_ns"])]
    checks["embedding_has_unique_native_model_id"] = (
        len(matches) == 1 and matches[0].model_id not in (None, "", "N/A", "4294967295", "-1")
    )

    def task_sample(task):
        return {
            "row": task.row_index, "model": task.model_id, "stream": task.stream_id,
            "task": task.task_id, "op": task.name[:120],
            "start_ns": task.start_ns - main_end, "end_ns": task.end_ns - main_end,
        }

    if not checks["embedding_has_unique_native_model_id"]:
        return checks, {"embedding_row": embedding_row, "embedding_matches": [task_sample(task) for task in matches[:3]]}

    embed = matches[0]
    graph_tasks = [task for task in tasks if task.model_id == embed.model_id]
    task_zero = [task for task in tasks if task.task_id == "0"]
    starts = [task for task in task_zero if task.model_id == embed.model_id and task.stream_id == embed.stream_id]
    first = min(graph_tasks, key=lambda task: task.start_ns)
    last = max(graph_tasks, key=lambda task: task.end_ns)
    graph_hc_start = [task for task in graph_tasks if task.start_ns == hc[0]]
    graph_hc_end = [task for task in graph_tasks if task.end_ns == hc[1]]
    checks.update(
        one_task_zero_on_embedding_stream=len(starts) == 1,
        task_zero_starts_model_envelope=starts[0].start_ns == first.start_ns if len(starts) == 1 else True,
        envelope_starts_after_main_end=main_end <= first.start_ns,
        envelope_starts_at_or_before_embedding=first.start_ns <= embed.start_ns,
        embedding_starts_before_hc=embed.start_ns < hc[0],
        hc_interval_positive=hc[0] < hc[1],
        hc_ends_before_envelope_end=hc[1] < last.end_ns,
        envelope_ends_at_or_before_next_inputs=last.end_ns <= next_start,
        hc_start_is_same_model_task_start=bool(graph_hc_start),
        hc_end_is_same_model_task_end=bool(graph_hc_end),
        markov_row_has_embedding_model_id=any(task.row_index == markov_row for task in graph_tasks),
    )
    alternate_zeros = [task for task in task_zero if task.model_id == embed.model_id and task.stream_id != embed.stream_id]
    return checks, {
        "time_origin": "main_hc_end",
        "hc_start_ns": hc[0] - main_end,
        "hc_end_ns": hc[1] - main_end,
        "next_inputs_start_ns": next_start - main_end,
        "embedding": task_sample(embed),
        "model_first": task_sample(first),
        "model_last": task_sample(last),
        "same_model_tasks": len(graph_tasks),
        "task_zero_on_embedding_stream_count": len(starts),
        "task_zero_on_embedding_stream": [task_sample(task) for task in starts[:2]],
        "task_zero_same_model_other_stream_count": len(alternate_zeros),
        "task_zero_same_model_other_stream": [task_sample(task) for task in alternate_zeros[:2]],
        "hc_start_tasks": [task_sample(task) for task in (graph_hc_start or [t for t in tasks if t.start_ns == hc[0]])[:2]],
        "hc_end_tasks": [task_sample(task) for task in (graph_hc_end or [t for t in tasks if t.end_ns == hc[1]])[:2]],
        "markov_row": markov_row,
        "markov_tasks": [task_sample(task) for task in tasks if task.row_index == markov_row][:2],
    }


def diagnose_clock(core, capture, trace_path):
    """Record provenance for the exact anchor-selection rules in core."""
    lookup = core.residual_anchor_candidates(capture["tasks"])
    candidate_starts = {task.start_ns for task in lookup.values()}
    matches, processes, trace_stats = {}, {}, {}
    counts = Counter(rejected_id_conflicts=0)
    rejected = []
    id_fields = ("stream_id", "task_id", "model_id")
    for event in core.residual_iter_trace_events(trace_path, stats=trace_stats):
        counts["events"] += 1
        phase = event.get("ph")
        if phase == "M" and event.get("name") == "process_name" and event.get("pid") is not None:
            processes[core.residual_id(event["pid"])] = str(core.residual_args(event).get("name", ""))
        if phase != "X":
            continue
        kind = core.residual_kind(event.get("name", ""))
        if not kind:
            continue
        try:
            duration = core.a_ns(event["dur"])
            start = core.a_ns(event["ts"])
        except (ValueError, KeyError, ArithmeticError):
            counts["invalid_timed_events"] += 1
            continue
        args = core.residual_args(event)
        task, matched_key, ids = core.residual_find_clock_anchor(lookup, kind, duration, args)
        if task is None:
            continue
        if matched_key == "id_conflict":
            counts["rejected_id_conflicts"] += 1
            if len(rejected) < 8:
                rejected.append({
                    "csv_row_index": task.row_index, "trace_event_index": counts["events"] - 1,
                    "kind": kind, "duration_ns": duration,
                    "csv_ids": dict(zip(id_fields, core.residual_clock_task_ids(task))),
                    "trace_ids": dict(zip(id_fields, ids)),
                    "csv_minus_trace_ns": str(task.start_ns - start),
                })
            continue
        hits = matches.setdefault(task.start_ns, [])
        if len(hits) < 9:
            hits.append({
                "csv_task": task,
                "trace_start": start, "trace_pid": core.residual_id(event.get("pid")),
                "trace_tid": core.residual_id(event.get("tid")), "trace_event_index": counts["events"] - 1,
                "trace_name": str(event.get("name", ""))[:120], "trace_ids": ids,
                "device_hint": core.residual_device_hint(event, args), "key": matched_key,
                "kind": kind, "duration_ns": duration,
            })
    accepted = []
    for source, hits in matches.items():
        selected = [hit for hit in hits if hit["device_hint"] or
                    core.residual_device_process(processes.get(hit["trace_pid"], ""))]
        if len(selected) != 1 or len(hits) >= 9:
            counts["rejected_hit_cap" if len(hits) >= 9 else "rejected_device_hit_count"] += 1
            continue
        hit = selected[0]
        hit.update(csv_start=source, offset=source - hit["trace_start"], candidate_trace_hits=len(hits))
        accepted.append(hit)
    result = {
        "anchor_count": len(accepted),
        "candidate_csv_starts": len(candidate_starts),
        "candidates_with_trace_hits": len(matches),
        "counts": dict(counts),
        "required_anchor_count": core.RESIDUAL_MIN_ANCHORS,
        "clock_tolerance_ns": core.RESIDUAL_CLOCK_TOLERANCE_NS,
        "trace_sha256": trace_stats.get("sha256"),
        "accepted_by_key": dict(Counter(hit["key"] for hit in accepted)),
        "rejected_conflict_examples": rejected,
        "status": "CLOCK_UNCONFIRMED",
        "clock_offset_ns": None,
        "max_clock_residual_ns": None,
        "worst_anchors": [],
    }
    if len(accepted) < core.RESIDUAL_MIN_ANCHORS:
        return result
    offsets = sorted(hit["offset"] for hit in accepted)
    offset = (offsets[len(offsets) // 2] + offsets[(len(offsets) - 1) // 2]) // 2
    tolerance = core.RESIDUAL_CLOCK_TOLERANCE_NS
    for hit in accepted:
        hit["residual"] = hit["offset"] - offset
    worst = sorted(accepted, key=lambda hit: -abs(hit["residual"]))[:8]
    maximum = abs(worst[0]["residual"])
    csv_origin = min(hit["csv_start"] for hit in accepted)
    for hit in worst:
        task = hit["csv_task"]
        csv_ids = core.residual_clock_task_ids(task)
        result["worst_anchors"].append({
            "csv_row_index": task.row_index,
            "csv_name": str(task.name)[:120],
            "trace_event_index": hit["trace_event_index"],
            "trace_name": hit["trace_name"],
            "trace_pid": hit["trace_pid"],
            "trace_tid": hit["trace_tid"],
            "matched_key": "kind+duration+ids" if hit["key"] == "full" else "kind+duration",
            "kind": hit["kind"],
            "duration_ns": hit["duration_ns"],
            "csv_ids": dict(zip(id_fields, csv_ids)),
            "trace_ids": dict(zip(id_fields, hit["trace_ids"])),
            "id_disagreements": [field for field, csv_id, trace_id
                                 in zip(id_fields, csv_ids, hit["trace_ids"]) if csv_id != trace_id],
            "csv_full_key_available": lookup.get(("full", (hit["kind"], hit["duration_ns"]) + csv_ids)) is task,
            "candidate_trace_hits": hit["candidate_trace_hits"],
            "device_verified_by": "event_hint" if hit["device_hint"] else "trace_process_name",
            "csv_start_ns_relative": hit["csv_start"] - csv_origin,
            "trace_shifted_start_ns_relative": hit["trace_start"] + offset - csv_origin,
            "csv_minus_trace_ns": str(hit["offset"]),
            "signed_residual_ns": hit["residual"],
            "absolute_residual_ns": abs(hit["residual"]),
        })
    stats_by_key = {}
    for key in ("full", "base"):
        residuals = sorted(abs(hit["residual"]) for hit in accepted if hit["key"] == key)
        if residuals:
            stats_by_key[key] = {
                "n": len(residuals),
                "above_tolerance": sum(value > tolerance for value in residuals),
                "p95_absolute_residual_ns": residuals[(95 * len(residuals) + 99) // 100 - 1],
                "max_absolute_residual_ns": residuals[-1],
            }
    result.update(
        status="CLOCK_MISMATCH" if maximum > tolerance else "ALIGNED_X_EVENTS",
        clock_offset_ns=str(offset), max_clock_residual_ns=maximum, residuals_by_key=stats_by_key,
    )
    return result


def main():
    parser = argparse.ArgumentParser(description="Inspect the existing dp4 S15 graph and clock evidence.")
    parser.add_argument("--cases", type=Path, default=Path("cases.local.json"))
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path, default=Path("u_overlap_dp4_s15_probe.json"))
    args = parser.parse_args()

    sys.path.insert(0, str(args.repo.resolve()))
    import profile_ep128 as core

    if Path(core.__file__).resolve() != (args.repo / "profile_ep128.py").resolve():
        raise ValueError("Use --repo to point to the extrapolator checkout used for the fixed-rank report.")

    manifest = core._source_path(args.cases)
    config = json.loads(core._io_path(manifest).read_text(encoding="utf-8-sig"))
    if config.get("schema_version") != core.SCHEMA_VERSION:
        raise ValueError("Use the same schema_version=2 cases.local.json as the dp4 export.")

    rank = "dp4"
    result = {"schema": "u_overlap_s15_probe_v1", "profile_version": core.PROFILE_VERSION, "rank": rank}
    failed = False
    for bs, section, probe in ((16, "graph", run_graph), (8, "clock", run_clock)):
        case_name = f"bs{bs}-s15"
        print(f"Reading existing {case_name} / {rank} inputs", flush=True)
        try:
            result[section] = probe(core, config, manifest.parent, case_name, rank, bs)
        except (ValueError, KeyError, OSError, ArithmeticError, TypeError, RuntimeError, AssertionError) as error:
            print(f"{case_name}: {type(error).__name__}: {error}", file=sys.stderr)
            result[section] = {"case": case_name, "error_type": type(error).__name__}
            failed = True

    payload = (json.dumps(result, ensure_ascii=True, allow_nan=False, separators=(",", ":")) + "\n").encode("ascii")
    if len(payload) > 65536:
        raise ValueError(f"Diagnostic output exceeds 64 KiB ({len(payload)} bytes).")
    core._io_path(args.out).write_bytes(payload)
    print(f"{args.out.resolve()} ({len(payload)} bytes)")
    return int(failed)


def load_case(core, config, base, name, rank, bs):
    selected = [item for item in config["cases"] if item.get("name") == name]
    if len(selected) != 1:
        raise ValueError(f"Expected one {name} entry in the existing cases manifest, found {len(selected)}.")
    case = core.normalize_case(config.get("defaults", {}), selected[0], base)
    if (case["bs"], case["draft_tokens"]) != (bs, 15):
        raise ValueError(f"{name} must have BS={bs} and S=15.")
    capture = core.read_capture(core._fixed_overlap_input(case, rank, base), rank)
    return case, capture


def run_graph(core, config, base, name, rank, bs):
    case, capture = load_case(core, config, base, name, rank, bs)
    model = core.referenced_object(config["sample_model"], base, "sample_model")
    core.validate_models(model, None)
    measured = core.measure_capture(capture, model, case, base)
    return {"case": name, "kernel_sha256": capture["sha256"], **diagnose_graph(core, capture, measured, case)}


def run_clock(core, config, base, name, rank, bs):
    case, capture = load_case(core, config, base, name, rank, bs)
    trace_path = core._case_trace_path(case, rank, base)
    if trace_path is None:
        trace_path = Path(capture["path"]).parent / "trace_view.json"
    trace_path = core._source_path(trace_path)
    return {"case": name, "kernel_sha256": capture["sha256"], **diagnose_clock(core, capture, trace_path)}


if __name__ == "__main__":
    sys.exit(main())
