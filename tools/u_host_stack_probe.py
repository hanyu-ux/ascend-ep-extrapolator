#!/usr/bin/env python3
"""Diagnose saved stack-enabled captures without producing timing extrapolations.

Run in the repository: python tools/u_host_stack_probe.py cases.json --rank dp4
Every case uses its own CSV, trace, clock anchors and profiler metadata. Original
captures are read only. Full local evidence and a bounded share file are saved.
"""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
import copy
import csv
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import pickle
import re
import sqlite3
from statistics import fmean
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import profile_ep128 as core
from ep128_host_function_trace import analyze_trace
from ep128_next_main_overlap import summarize_next_main_overlap
from u_host_function_probe import select_windows


SCHEMA = "u_host_stack_v1"
SHARE_LIMIT = 65536


def capture_metadata(path):
    """Observe real profiler settings, retaining missing and conflicting states."""
    rows = []
    for info in sorted(core._io_path(path.parent.parent).glob("profiler_info*.json")):
        data = info.read_bytes()
        common = json.loads(data.decode("utf-8-sig")).get("config", {}).get("common_config", {})
        rows.append({"path": str(core._source_path(info)), "sha256": hashlib.sha256(data).hexdigest(),
                     **{key: common.get(key) for key in ("with_stack", "with_modules", "record_shapes")}})
    if any(r[key] is not None and type(r[key]) is not bool
           for r in rows for key in ("with_stack", "with_modules")):
        raise ValueError("Profiler stack settings must be booleans")
    if any(len({r[key] for r in rows if r[key] is not None}) > 1
           for key in ("with_stack", "with_modules")):
        raise ValueError("Conflicting profiler_info stack settings in selected capture")
    if rows and all(r["with_stack"] is False and r["with_modules"] in (None, False) for r in rows):
        raise ValueError("Selected profiler_info declares stack/modules disabled")
    verified = bool(rows) and any(r["with_stack"] is True or r["with_modules"] is True for r in rows)
    return rows, verified


def summarize_windows(windows, requests, samples=2):
    """Aggregate all attributable windows, never the longest examples' mean.

    Different kinds are different intervals and must not be added. Within one
    kind, overlapping requests would double-count functions, so such aggregates
    remain unavailable. Counts always expose the actually contributing windows.
    """
    identities = {w["id"]: w for w in requests}
    groups = defaultdict(list)
    for window in windows:
        request = identities[window["id"]]
        groups[request["kind"]].append((window, request))
    summaries, examples = [], []
    for kind in ("prepare_gap_before_expand", "prepare_between_host_calls", "metadata_before_enqueue"):
        group = groups[kind]
        good = [(w, r) for w, r in group if w.get("status") == "MATCHED"]
        intervals = sorted((r["start_ns"], r["end_ns"]) for w, r in good)
        overlap = any(a[1] > b[0] for a, b in zip(intervals, intervals[1:]))
        row = {"kind": kind, "windows": len(group), "matched_windows": len(good),
               "matched_rounds": len({r["round_id"] for w, r in good}),
               "windows_with_python_functions": sum(bool(w.get("python_function_events")) for w, r in good),
               "status_counts": dict(Counter(w.get("status", "UNKNOWN") for w, r in group)),
               "status": "OVERLAPPING_WINDOWS" if overlap else "MATCHED" if good else "NO_MATCHED_WINDOWS",
               "scope": "Means across all matched candidate windows, not all capture rounds; function times are clipped "
                        "to each window, not whole-call durations; kinds are nonadditive",
               "top_exclusive_functions": []}
        for field in ("window_ns", "attributed_ns", "ambiguous_ns", "unrecorded_ns"):
            row[field.replace("_ns", "_mean_ns")] = fmean(w[field] for w, r in good) if good and not overlap else None
        row["python_function_covered_mean_ns"] = (fmean(w.get("python_function_covered_ns", 0) for w, r in good)
                                                   if good and not overlap else None)
        if good and not overlap:
            functions = defaultdict(lambda: {"total_ns": 0, "windows_present": 0})
            for w, r in good:
                if w["window_ns"] != sum(w[k] for k in ("attributed_ns", "ambiguous_ns", "unrecorded_ns")):
                    raise ValueError("Host window partition does not close")
                if w.get("exclusive_functions_omitted"):
                    raise ValueError("Cannot aggregate truncated function lists")
                for function in w.get("exclusive_functions", []):
                    item = functions[function["name"], function["lane"]]
                    item["total_ns"] += function["ns"]
                    item["windows_present"] += 1
            row["top_exclusive_functions"] = [
                {"name": name, "lane": lane, **amount, "ns_mean_per_window": amount["total_ns"] / len(good)}
                for (name, lane), amount in sorted(functions.items(), key=lambda item: -item[1]["total_ns"])]
        summaries.append(row)
        for w, r in sorted(group, key=lambda pair: -(pair[1]["end_ns"] - pair[1]["start_ns"]))[:samples]:
            examples.append({**w, "kind": kind, "round_id": r["round_id"], "device_row": r["device_row"],
                             "device_gap_ns": r["device_gap_ns"],
                             "window_start_relative_to_device_ns": r["start_ns"] - r["device_start_ns"]})
    return summaries, examples


def bounded_share(payload, limit=SHARE_LIMIT):
    """Keep every case's status/means; omit example detail before core results."""
    share = {**{k: v for k, v in payload.items() if k != "cases"},
             "cases": [{k: copy.deepcopy(v) for k, v in case.items() if k not in ("all_windows", "stage_details")}
                       for case in payload["cases"]]}
    share["share_limit_bytes"] = limit
    share["omitted_examples"] = 0
    share["omitted_function_rows"] = 0
    share["omitted_unresolved_source_examples"] = 0
    phase_fields = ("stage_number", "stage_key", "rounds", "matched_rounds", "draft_graph_matched_rounds",
                    "host_complete_rounds", "host_partial_rounds", "trace_status", "case_error")
    metrics = ("stage_wall_ms", "device_busy_csv_ms", "dma_additional_ms", "device_uncovered_after_dma_ms",
               "draft_hc_ms", "host_linked_ms", "host_in_draft_hc_ms", "host_in_draft_graph_ms")
    for case in share["cases"]:
        case.pop("all_windows", None)
        case.pop("stage_details", None)
        case.pop("kernel_path", None)
        unresolved = case.pop("largest_unresolved_sources", {})
        share["omitted_unresolved_source_examples"] += sum(len(v) for v in unresolved.values())
        if "trace" in case:
            case["trace"] = {k: v for k, v in case["trace"].items()
                             if k in ("status", "reason", "sha256", "clock_offset_ns", "anchor_count",
                                      "max_clock_residual_ns", "rejected_id_conflicts")}
        for metadata in case.get("profiler_metadata", []):
            metadata.pop("path", None)
        stages = case.pop("all_stage_wall_means", [])
        case["stage_wall_means"] = {r["stage_key"]: {"ms": r["duration_ms_mean"], "n": r["rounds"],
                                                    "status": r["status"]} for r in stages}
        case["phase_summary"] = [{k: v for k, v in row.items() if k in phase_fields
                                  or any(k == m + suffix for m in metrics for suffix in ("_mean", "_n"))}
                                 for row in case.get("phase_summary", [])]
        for row in case.get("window_summary", []):
            row.pop("scope", None)
            functions = row.get("top_exclusive_functions", [])
            row["function_rows_total"] = len(functions)
            share["omitted_function_rows"] += max(0, len(functions) - 8)
            row["top_exclusive_functions"] = functions[:8]
        for window in case.get("windows", []):
            window.pop("inclusive_functions", None)
            window.pop("longest_unrecorded", None)
            window.pop("anchor", None)
            functions = window.get("exclusive_functions", [])
            window["exclusive_function_rows_total"] = len(functions)
            share["omitted_function_rows"] += max(0, len(functions) - 5)
            window["exclusive_functions"] = functions[:5]

    def encoded():
        return (json.dumps(share, ensure_ascii=True, allow_nan=False, separators=(",", ":")) + "\n").encode("ascii")

    # All arithmetic/aggregation used full names. Only exported display strings
    # are shortened, with an explicit marker rather than merging identities.
    for case in share["cases"]:
        for row in case.get("window_summary", []):
            for function in row.get("top_exclusive_functions", []):
                if len(function["name"]) > 240:
                    function["name"] = function["name"][:240]
                    function["name_truncated"] = True
        for window in case.get("windows", []):
            for function in window.get("exclusive_functions", []):
                if len(function["name"]) > 240:
                    function["name"] = function["name"][:240]
                    function["name_truncated"] = True
    for case in reversed(share["cases"]):
        while len(encoded()) > limit and case.get("windows"):
            case["windows"].pop()
            share["omitted_examples"] += 1
    for maximum in (5, 3, 1, 0):
        if len(encoded()) <= limit:
            break
        for case in reversed(share["cases"]):
            for row in case.get("window_summary", []):
                functions = row["top_exclusive_functions"]
                while len(functions) > maximum and len(encoded()) > limit:
                    functions.pop()
                    share["omitted_function_rows"] += 1
    result = encoded()
    if len(result) > limit:
        raise ValueError("Required case summaries exceed share limit; select fewer cases with --case")
    return result


def analyze_case(item, config, base, rank, out, samples):
    name = item.get("name", "unnamed") if isinstance(item, dict) else "invalid"
    entry = {"case": name, "rank": rank, "status": "ERROR", "error": None, "warnings": [],
             "phase_summary": [], "window_summary": [], "windows": []}
    try:
        case = core.normalize_case(config.get("defaults", {}), item, base, capture_purpose="host_stack_diagnostic")
        entry.update(case=case["name"], bs=case["bs"], S=case["draft_tokens"], declared_with_stack=case["with_stack"],
                     configured_effective_tokens_per_round=case["effective_tokens_per_round"])
        path = core._fixed_overlap_input(case, rank, base)
        metadata, verified = capture_metadata(path)
        entry.update(profiler_metadata=metadata, stack_metadata_verified=verified)
        if not verified:
            entry["warnings"].append("PROFILER_METADATA_MISSING_OR_INCOMPLETE")
        if metadata and any(m.get("with_modules") is True for m in metadata) and not any(m.get("with_stack") is True for m in metadata):
            entry["warnings"].append("PYTHON_TRACING_VIA_WITH_MODULES")
        observed_stack = {m["with_stack"] for m in metadata if m.get("with_stack") is not None}
        if observed_stack and observed_stack != {case["with_stack"]}:
            entry["warnings"].append("MANIFEST_STACK_FLAG_DIFFERS_FROM_CAPTURE; diagnostic mode preserves both")
        model = core.referenced_object(config["sample_model"], base, "sample_model")
        core.validate_models(model, None)
        capture = core.read_capture(path, rank)
        entry.update(kernel_path=capture["path"], kernel_sha256=capture["sha256"])
        measured = core.measure_capture(capture, model, case, base)
        slug = re.sub(r"[^\w.-]+", "_", case["name"])[:48] or "case"
        detail_dir = out / "stack_details" / (slug + "-" + hashlib.sha256(case["name"].encode()).hexdigest()[:8])
        diagnostic = core.residual_analyze_capture(capture, measured, detail_dir, case=case,
            trace_path=core._case_trace_path(case, rank, base), include_next_main_overlap=True,
            progress=lambda message: print("    " + message, flush=True))
        entry.update(trace=diagnostic["trace"], cycle_residual_ms_mean=diagnostic["residual_ms_mean"],
                     stage_details={k: diagnostic.get(k) for k in ("device_tasks_csv", "stages_csv", "source_links_csv")},
                     all_stage_wall_means=[{k: r[k] for k in ("window_key", "stage_key", "rounds", "status", "duration_ms_mean")}
                                           for r in diagnostic["stage_summary"]])
        identity = {"case": case["name"], "rank": rank, "bs": case["bs"], "S": case["draft_tokens"]}
        entry["phase_summary"] = summarize_next_main_overlap(diagnostic["next_main_overlap"], identity)
        if diagnostic["trace"].get("status") != "ALIGNED_X_EVENTS":
            raise ValueError("Selected trace is not aligned: " + str(diagnostic["trace"].get("status")))
        requests, counts, unavailable = select_windows(core, diagnostic, out, rank, None)
        entry.update(selection_counts=counts, largest_unresolved_sources=unavailable)
        trace_path = Path(diagnostic["trace"]["path"])
        scanned = analyze_trace(core, trace_path, diagnostic["trace"], requests, function_limit=None)
        entry.update({k: scanned[k] for k in ("trace_sha256", "sha256_matches_report", "counts", "scope")})
        entry["all_windows"] = scanned["windows"]
        entry["window_summary"], entry["windows"] = summarize_windows(scanned["windows"], requests, samples)
        if not scanned["counts"]["python_function_events"]:
            entry["status"] = "NO_PYTHON_FUNCTION_EVENTS"
            entry["error"] = "No timed python_function events in this trace"
        elif not requests:
            entry["status"] = "NO_ATTRIBUTABLE_WINDOWS"
            entry["error"] = "No uniquely attributable preparation/metadata windows"
        else:
            ready = verified and all(w["status"] == "MATCHED" for w in scanned["windows"])
            ready = ready and all(r["status"] != "OVERLAPPING_WINDOWS" for r in entry["window_summary"])
            missing_python = sum(w.get("status") == "MATCHED" and not w.get("python_function_events")
                                 for w in scanned["windows"])
            if missing_python:
                entry["warnings"].append("TARGET_WINDOWS_WITHOUT_PYTHON_FUNCTION_EVENTS: " + str(missing_python))
                ready = False
            if not any(r["kind"].startswith("prepare_") for r in requests):
                entry["warnings"].append("NO_ATTRIBUTABLE_PREPARE_WINDOW")
                ready = False
            if not any(r["kind"] == "metadata_before_enqueue" for r in requests):
                entry["warnings"].append("NO_ATTRIBUTABLE_METADATA_WINDOW")
                ready = False
            entry["status"] = "MATCHED" if ready else "PARTIAL"
        return entry
    except (ValueError, KeyError, OSError, ArithmeticError, TypeError, csv.Error, sqlite3.Error, re.error) as error:
        entry["error"] = type(error).__name__ + ": " + str(error)
        return entry


def _worker_failure(item, rank, error, stage):
    return {"case": item["name"], "rank": rank, "status": "ERROR",
            "error": "CASE_WORKER_ERROR (" + stage + "): " + type(error).__name__ + ": " + str(error),
            "warnings": [], "phase_summary": [], "window_summary": [], "windows": []}


def _case_worker(index, item, config, base, rank, out, samples, spool):
    """Keep large results on local disk; only result identity crosses IPC."""
    with redirect_stdout(core._CaseProgressStream(sys.stdout, item["name"] + " / " + rank)):
        print("Starting Host-stack analysis; worker=" + str(os.getpid()), flush=True)
        try:
            entry = analyze_case(item, config, base, rank, out, samples)
        except Exception as error:
            entry = _worker_failure(item, rank, error, "analysis")
        destination = Path(spool) / ("case-" + str(index) + ".pickle")
        with destination.open("wb") as stream:
            pickle.dump({"index": index, "entry": entry}, stream, protocol=pickle.HIGHEST_PROTOCOL)
    return index, str(destination)


def run_cases(selected, config, base, rank, out, samples, workers):
    """Yield completed cases; caller alone updates ordered aggregate files.

    Spawn is explicit on every platform. Per-case detail directories already
    include a case-name hash. Temporary IPC files are created by this run, not
    chosen by the capture manifest, and are removed after collection.
    """
    if workers <= 1 or len(selected) <= 1:
        for index, item in enumerate(selected):
            print(f"[{index + 1}/{len(selected)}] {item['name']} / {rank}", flush=True)
            try:
                entry = analyze_case(item, config, base, rank, out, samples)
            except Exception as error:
                entry = _worker_failure(item, rank, error, "analysis")
            yield index, entry
        return
    with tempfile.TemporaryDirectory(prefix="u-stack-workers-") as spool:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as executor:
            pending = {}
            for index, item in enumerate(selected):
                try:
                    future = executor.submit(_case_worker, index, item, config, base, rank, out, samples, spool)
                except Exception as error:
                    yield index, _worker_failure(item, rank, error, "submit")
                else:
                    pending[future] = index, item
            for future in as_completed(pending):
                index, item = pending[future]
                expected = Path(spool) / ("case-" + str(index) + ".pickle")
                try:
                    outcome = future.result()
                    if outcome != (index, str(expected)):
                        raise ValueError("Worker result file identity differs from submitted case")
                    with expected.open("rb") as stream:
                        result = pickle.load(stream)
                    entry = result["entry"]
                    if result["index"] != index or entry["case"] != item["name"] or entry["rank"] != rank:
                        raise ValueError("Worker result case/rank identity mismatch")
                except Exception as error:
                    entry = _worker_failure(item, rank, error, "collect")
                finally:
                    expected.unlink(missing_ok=True)
                yield index, entry


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", type=Path, nargs="?")
    parser.add_argument("--export-existing", type=Path,
                        help="Read an existing u_host_stack_full.json and export selected wait/parent evidence only")
    parser.add_argument("--max-bytes", type=int, default=65536,
                        help="Small evidence export byte limit (at least 65536); use 262144 for more context")
    parser.add_argument("--rank", default="dp4")
    parser.add_argument("--case", dest="names", action="append")
    parser.add_argument("--out", type=Path, default=Path("u_host_stack_dp4"))
    parser.add_argument("--samples", type=int, choices=range(1, 5), default=2)
    parser.add_argument("--workers", type=core._cli_positive_workers, default=None,
                        help="Case analysis processes; automatic 1-2 by default, --workers 1 for serial")
    args = parser.parse_args(argv)
    if args.max_bytes < 65536:
        parser.error("--max-bytes must be at least 65536")
    if args.export_existing is None and args.max_bytes != 65536:
        parser.error("--max-bytes is only supported with --export-existing")
    if args.export_existing is not None:
        if args.cases is not None:
            parser.error("Use either cases JSON or --export-existing, not both")
        from ep128_stack_evidence import export_existing
        result, path = export_existing(core, args.export_existing, args.out, args.names, rank=args.rank,
                                       max_bytes=args.max_bytes)
        print("Evidence: " + str(path) + " (" + str(core._io_path(path).stat().st_size) + " bytes)")
        return int(any(c["status"] == "ERROR" for c in result["cases"]))
    if args.cases is None:
        parser.error("Provide cases JSON, or --export-existing FULL_JSON")
    manifest = core._source_path(args.cases)
    config = json.loads(core._io_path(manifest).read_text(encoding="utf-8-sig"))
    if config.get("schema_version") != core.SCHEMA_VERSION or not isinstance(config.get("cases"), list):
        parser.error("Use the existing schema_version=2 cases JSON")
    names = [item.get("name") for item in config["cases"] if isinstance(item, dict)]
    if len(names) != len(config["cases"]) or len(set(names)) != len(names) or not names:
        parser.error("Cases must have unique names")
    if args.names and (len(set(args.names)) != len(args.names) or set(args.names) - set(names)):
        parser.error("--case names must be unique and present in the manifest")
    selected = [item for item in config["cases"] if not args.names or item["name"] in args.names]
    # Prioritize the root-cause comparison so useful evidence is saved first.
    priority = {name: i for i, name in enumerate(("bs8-s7", "bs16-s7", "bs8-s15", "bs16-s15"))}
    selected.sort(key=lambda item: priority.get(item["name"], 99))
    workers = core._cli_case_workers(args.workers, len(selected), "case (fixed " + args.rank + ")")
    out = core._source_path(args.out)
    full_path, share_path = out / "u_host_stack_full.json", out / "u_host_stack_share.json"
    protected = {manifest}
    for key in ("sample_model", "target_model", "communication_reference"):
        if isinstance(config.get(key), str):
            candidate = Path(config[key])
            protected.add(core._source_path(candidate if candidate.is_absolute() else manifest.parent / candidate))
    for item in selected:
        raw = {**config.get("defaults", {}), **item}
        for source in raw.get("inputs") or []:
            if isinstance(source, dict) and source.get("rank") == args.rank:
                for key in ("path", "trace_path"):
                    if isinstance(source.get(key), str):
                        candidate = Path(source[key])
                        protected.add(core._source_path(candidate if candidate.is_absolute() else manifest.parent / candidate))
        try:
            path = core._fixed_overlap_input(raw, args.rank, manifest.parent)
            protected.add(core._source_path(path))
            protected.add(core._source_path(core._case_trace_path(raw, args.rank, manifest.parent) or path.parent / "trace_view.json"))
        except (ValueError, KeyError, OSError, TypeError):
            # Preserve this case's error in the checkpoint while analyzing the
            # remaining cases. This preflight only protects existing inputs.
            pass
    if any(core._source_path(path) in protected for path in (full_path, share_path)):
        parser.error("Output path overlaps a source input")
    core._io_path(out).mkdir(parents=True, exist_ok=True)
    payload = {"schema": SCHEMA, "profile_version": core.PROFILE_VERSION, "diagnostic_only": True,
               "rank": args.rank, "planned_cases": len(selected), "completed_cases": 0, "cases": [],
               "execution": {"parallelism": "case", "requested_workers": args.workers, "workers": workers},
               "scope": "Stack-enabled diagnostic capture; no throughput/extrapolation; timings include profiler overhead. "
                        "Function means use all attributable windows of each kind, not longest samples; "
                        "self wall time includes uninstrumented work and descheduling. "
                        "Temporal overlap does not establish critical-path causality."}
    completed = {}
    for index, entry in run_cases(selected, config, manifest.parent, args.rank, out, args.samples, workers):
        completed[index] = entry
        payload["cases"] = [completed[position] for position in sorted(completed)]
        payload["completed_cases"] = len(payload["cases"])
        core.atomic_json(full_path, payload)
        core._io_path(share_path).write_bytes(bounded_share(payload))
        print(f"Completed [{len(completed)}/{len(selected)}] {entry['case']} / {args.rank}: "
              + entry["status"] + (": " + entry["error"] if entry["error"] else ""), flush=True)
        print("  Checkpoint: " + str(share_path), flush=True)
    print("Share: " + str(share_path) + " (" + str(core._io_path(share_path).stat().st_size) + " bytes)")
    return int(any(c["error"] or c["status"] != "MATCHED" for c in payload["cases"]))


if __name__ == "__main__":
    sys.exit(main())
