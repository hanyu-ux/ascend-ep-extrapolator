"""Host-stack diagnostics stay separate from timing and use fresh capture evidence."""

from copy import deepcopy
from decimal import Decimal
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import ep128_host_function_trace as attribution
import profile_ep128 as core
import test_ep128_next_main_overlap as overlap_fixtures
import test_ep128_v2_measure as fixtures


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "u_host_stack_probe.py"


def load_tool():
    spec = importlib.util.spec_from_file_location("stack_probe_for_tests", TOOL)
    module = importlib.util.module_from_spec(spec)
    with patch.object(sys, "path", [str(ROOT / "tools"), *sys.path]):
        spec.loader.exec_module(module)
    return module


class HostStackContractTests(unittest.TestCase):
    def test_explicit_diagnostic_preserves_stack_declaration_and_default_rejects_it(self):
        config = json.loads((ROOT / "profile_ep128_cases.example.json").read_text(encoding="utf-8"))
        item = dict(config["cases"][0], with_stack=True)
        original = deepcopy(item)
        with self.assertRaisesRegex(ValueError, "with_stack=false"):
            core.normalize_case(config["defaults"], item, ROOT)
        diagnostic = core.normalize_case(config["defaults"], item, ROOT,
                                         capture_purpose="host_stack_diagnostic")
        self.assertIs(diagnostic["with_stack"], True)
        self.assertEqual(item, original)
        self.assertEqual(diagnostic["bs"], item["bs"])
        with self.assertRaisesRegex(ValueError, "capture_purpose"):
            core.normalize_case(config["defaults"], item, ROOT, capture_purpose="silently_skip_checks")

    def test_full_function_analysis_keeps_all_names_and_exclusive_time(self):
        with tempfile.TemporaryDirectory(prefix="ep128 stack full ") as temporary:
            path = Path(temporary) / "trace.json"
            epoch = fixtures.RESIDUAL_EPOCH

            def event(name, start, end):
                return {"ph": "X", "name": name, "pid": 10, "tid": 7,
                        "cat": "python_function", "ts": str(Decimal(epoch + start) / 1000),
                        "dur": str(Decimal(end - start) / 1000)}

            names = ["same_very_long_prefix_" * 12 + str(index) for index in range(8)]
            events = [{"ph": "M", "name": "process_name", "pid": 10,
                       "args": {"name": "Python"}}, event("prepare_inputs", 0, 100)]
            events.extend(event(name, index * 10, index * 10 + 5) for index, name in enumerate(names))
            payload = json.dumps(events).encode("utf-8")
            path.write_bytes(payload)
            request = {"id": "prepare-1", "kind": "prepare_between_host_calls",
                       "start_ns": str(epoch), "end_ns": str(epoch + 100),
                       "anchor_event_index": 2, "anchor_name": "prepare_inputs",
                       "anchor_start_ns": str(epoch), "anchor_end_ns": str(epoch + 100)}
            trace = {"status": "ALIGNED_X_EVENTS", "clock_offset_ns": "0",
                     "sha256": hashlib.sha256(payload).hexdigest()}
            default = attribution.analyze_trace(core, path, trace, [request])["windows"][0]
            full = attribution.analyze_trace(core, path, trace, [request], function_limit=None)["windows"][0]
            self.assertEqual(len(default["exclusive_functions"]), 5)
            self.assertEqual(len(full["exclusive_functions"]), 9)
            self.assertEqual(full["exclusive_functions_omitted"], 0)
            self.assertEqual({row["name"] for row in full["exclusive_functions"]}, set(names) | {"prepare_inputs"})
            self.assertEqual(sum(row["ns"] for row in full["exclusive_functions"]), 100)
            self.assertEqual(full["attributed_ns"] + full["ambiguous_ns"] + full["unrecorded_ns"], 100)
            self.assertEqual(full["python_function_events"], 9)
            self.assertEqual(full["python_function_covered_ns"], 100)


class StackWindowSummaryTests(unittest.TestCase):
    def setUp(self):
        self.tool = load_tool()

    def rows(self):
        windows, requests = [], []
        prefix = "shared_prefix_" * 20
        for index, duration in enumerate((10, 20, 30)):
            identity = "window-" + str(index)
            requests.append({"id": identity, "kind": "prepare_between_host_calls",
                             "round_id": "hc-" + str(index), "start_ns": 100 * index,
                             "end_ns": 100 * index + duration, "device_start_ns": 100 * index + duration,
                             "device_row": index, "device_gap_ns": duration})
            name = prefix + ("A" if index == 0 else "B")
            windows.append({"id": identity, "status": "MATCHED", "window_ns": duration,
                            "attributed_ns": duration, "ambiguous_ns": 0, "unrecorded_ns": 0,
                            "exclusive_functions_omitted": 0,
                            "exclusive_functions": [{"name": name, "lane": "cpu_thread", "ns": duration}]})
        return windows, requests

    def test_means_use_all_windows_and_absent_functions_still_use_same_denominator(self):
        windows, requests = self.rows()
        original = deepcopy((windows, requests))
        summaries, examples = self.tool.summarize_windows(windows, requests, samples=1)
        row = next(row for row in summaries if row["kind"] == "prepare_between_host_calls")
        self.assertEqual((row["windows"], row["matched_windows"], row["matched_rounds"]), (3, 3, 3))
        self.assertEqual(row["window_mean_ns"], 20)
        functions = {row["name"]: row for row in row["top_exclusive_functions"]}
        self.assertEqual(len(functions), 2)
        first, second = ["shared_prefix_" * 20 + suffix for suffix in ("A", "B")]
        self.assertAlmostEqual(functions[first]["ns_mean_per_window"], 10 / 3)
        self.assertAlmostEqual(functions[second]["ns_mean_per_window"], 50 / 3)
        self.assertEqual(functions[first]["windows_present"], 1)
        self.assertEqual(functions[second]["windows_present"], 2)
        self.assertEqual(len(examples), 1)
        self.assertEqual(examples[0]["window_ns"], 30)
        self.assertEqual((windows, requests), original)

    def test_overlapping_same_kind_windows_cannot_be_added_or_averaged(self):
        windows, requests = self.rows()
        requests[1].update(start_ns=5, end_ns=25, round_id=requests[0]["round_id"])
        summaries, examples = self.tool.summarize_windows(windows, requests, samples=1)
        row = next(row for row in summaries if row["kind"] == "prepare_between_host_calls")
        self.assertEqual(row["status"], "OVERLAPPING_WINDOWS")
        self.assertIsNone(row["window_mean_ns"])
        self.assertFalse(row["top_exclusive_functions"])
        self.assertTrue(examples)

    def test_truncated_or_nonclosing_windows_cannot_produce_complete_aggregates(self):
        for invalid in ("truncated", "nonclosing"):
            with self.subTest(invalid=invalid):
                windows, requests = self.rows()
                if invalid == "truncated":
                    windows[0]["exclusive_functions_omitted"] = 1
                else:
                    windows[0]["attributed_ns"] -= 1
                with self.assertRaisesRegex(ValueError, "truncated|does not close"):
                    self.tool.summarize_windows(windows, requests)

    def test_twelve_case_small_export_preserves_status_and_means_without_source_mutation(self):
        payload = {"schema": "u_host_stack_v1", "diagnostic_only": True, "rank": "dp4", "cases": []}
        for case_index in range(12):
            name = "case-" + str(case_index)
            windows, requests = self.rows()
            summaries, examples = self.tool.summarize_windows(windows, requests, samples=2)
            for row in summaries:
                row["top_exclusive_functions"] = [
                    {"name": "x" * 1000 + str(index), "lane": "cpu_thread", "total_ns": 100 - index,
                     "windows_present": 3, "ns_mean_per_window": (100 - index) / 3}
                    for index in range(40)
                ]
            payload["cases"].append({"case": name, "rank": "dp4", "status": "PARTIAL" if case_index else "ERROR",
                                     "error": "missing evidence" if not case_index else None,
                                     "counts": {"python_function_events": 100},
                                     "phase_summary": [{"stage_number": 13, "stage_key": "main_inputs",
                                                        "stage_wall_ms_mean": 3.4, "rounds": 30}],
                                     "window_summary": summaries, "windows": examples, "all_windows": windows})
        original = deepcopy(payload)
        small = self.tool.bounded_share(payload)
        self.assertLessEqual(len(small), 65536)
        decoded = json.loads(small)
        self.assertEqual(len(decoded["cases"]), 12)
        self.assertEqual([row["case"] for row in decoded["cases"]], [row["case"] for row in original["cases"]])
        self.assertEqual(decoded["cases"][0]["status"], "ERROR")
        self.assertEqual(decoded["cases"][0]["error"], "missing evidence")
        for row in decoded["cases"]:
            self.assertEqual(row["phase_summary"][0]["stage_wall_ms_mean"], 3.4)
            prepare = next(item for item in row["window_summary"] if item["kind"] == "prepare_between_host_calls")
            self.assertEqual(prepare["window_mean_ns"], 20)
            self.assertNotIn("all_windows", row)
        self.assertGreater(decoded["omitted_function_rows"], 0)
        self.assertEqual(payload, original)


class HostStackProbeCLITests(unittest.TestCase):
    native_capture = overlap_fixtures.FixedRankCLITests.native_capture

    def stack_capture(self, parent, bs=8, *, python_events=True, observed_stack=True):
        directory = parent / "ASCEND_PROFILER_OUTPUT"
        directory.mkdir(parents=True)
        path = Path(self.native_capture(directory, bs, 7))
        capture = core.read_capture(path, "dp4")
        trace_path = directory / "trace_view.json"
        events = json.loads(trace_path.read_text(encoding="utf-8"))
        tasks = capture["tasks"]
        apply = [task for task in tasks if task.name == "_apply_write_kernel"]
        expands = [task for task in tasks if task.name == "_expand_idx_mapping_kernel"]
        offset, epoch = 423, fixtures.RESIDUAL_EPOCH
        for index, (previous, following) in enumerate(zip(apply, expands)):
            start = following.start_ns - epoch
            for number, name, left, right, task in (
                (500 + index * 2, "_apply_write_kernel", start - 11_000, start - 9_000, previous),
                (501 + index * 2, "_expand_idx_mapping_kernel", start - 1_000, start - 500, following),
            ):
                before = len(events)
                overlap_fixtures.trace_source(events, name, left, right, number, [task], offset)
                # CPU submissions run on the same physical Host thread. Queue
                # correlation identities remain distinct for every launch.
                events[before]["tid"] = 7
                events[before + 1]["tid"] = 7
            if python_events:
                events.append(overlap_fixtures.event(
                    "model_runner.py(789): NPUModelRunner.prepare_inputs", start - 9_000,
                    start - 500, 10, 7, offset, kind="python_function"))
                events.append(overlap_fixtures.event(
                    "numpy.cumsum", start - 8_000, start - 6_000,
                    10, 7, offset, kind="python_function"))
        trace_path.write_text(json.dumps(events), encoding="utf-8")
        (parent / "profiler_info_4.json").write_text(json.dumps({"config": {"common_config": {
            "with_stack": observed_stack, "with_modules": False,
            "activities": ["ProfilerActivity.CPU", "ProfilerActivity.NPU"],
        }}}), encoding="utf-8")
        return path, trace_path

    def config(self, base, captures):
        config = json.loads((ROOT / "profile_ep128_cases.example.json").read_text(encoding="utf-8"))
        config["sample_model"] = fixtures.MODEL
        config["cases"] = [
            {"name": name, "bs": bs, "draft_tokens": 7,
             "synthetic_acceptance_length": 4.4,
             "inputs": [{"rank": "dp4", "path": str(path)}]}
            for name, bs, path in captures
        ]
        # Existing Windows manifests usually retain the old timing declaration;
        # the diagnostic must observe and expose the real capture metadata.
        config["defaults"]["with_stack"] = False
        path = base / "cases.json"
        path.write_text(json.dumps(config), encoding="utf-8-sig")
        return path

    def run_probe(self, manifest, out, *arguments):
        return subprocess.run(
            [sys.executable, "-B", "-S", str(TOOL), str(manifest), "--rank", "dp4",
             "--out", str(out), "--samples", "1", *arguments],
            cwd=manifest.parent, capture_output=True, text=True, encoding="utf-8", timeout=60,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )

    def test_fresh_stack_capture_all_windows_and_partial_case_failures_are_retained(self):
        with tempfile.TemporaryDirectory(prefix="ep128 堆栈 CLI ") as temporary:
            base = Path(temporary)
            good_csv, good_trace = self.stack_capture(base / "bs8-s7" / "dp4", 8)
            no_python_csv, _ = self.stack_capture(base / "bs16-s7" / "dp4", 16, python_events=False)
            manifest = self.config(base, [("bs8-s7", 8, good_csv), ("bs16-s7", 16, no_python_csv)])
            source = {p: p.read_bytes() for p in base.rglob("*") if p.is_file()}
            out = base / "independent diagnostics"
            run = self.run_probe(manifest, out)
            self.assertNotEqual(run.returncode, 0, run.stdout + run.stderr)
            share = out / "u_host_stack_share.json"
            self.assertLessEqual(share.stat().st_size, 65536)
            report = json.loads((out / "u_host_stack_full.json").read_text(encoding="utf-8"))
            self.assertEqual(report["schema"], "u_host_stack_v1")
            self.assertIs(report["diagnostic_only"], True)
            cases = {row["case"]: row for row in report["cases"]}
            good = cases["bs8-s7"]
            self.assertEqual(good["rank"], "dp4")
            self.assertIsNone(good.get("error"), good)
            self.assertIs(good["declared_with_stack"], False)
            self.assertIs(good["stack_metadata_verified"], True)
            self.assertTrue(any("MANIFEST_STACK_FLAG_DIFFERS" in warning for warning in good["warnings"]))
            self.assertGreater(good["counts"]["python_function_events"], 0)
            self.assertEqual(good["trace_sha256"], hashlib.sha256(good_trace.read_bytes()).hexdigest())
            summaries = {row["kind"]: row for row in good["window_summary"]}
            prepare = summaries["prepare_between_host_calls"]
            self.assertGreater(prepare["matched_windows"], 1)
            self.assertEqual(prepare["window_mean_ns"], 8_000)
            self.assertEqual(prepare["unrecorded_mean_ns"], 0)
            functions = {row["name"]: row for row in prepare["top_exclusive_functions"]}
            self.assertEqual(functions["numpy.cumsum"]["ns_mean_per_window"], 2_000)
            self.assertEqual(cases["bs16-s7"]["status"], "NO_PYTHON_FUNCTION_EVENTS")
            self.assertTrue(good["phase_summary"])
            self.assertFalse((out / "model_architecture_report.json").exists())
            self.assertEqual({path: path.read_bytes() for path in source}, source)

    def test_missing_profiler_metadata_is_explicit_partial_and_real_stack_still_blocks_timing(self):
        with tempfile.TemporaryDirectory(prefix="ep128 metadata partial ") as temporary:
            base = Path(temporary)
            csv_path, _ = self.stack_capture(base / "capture" / "dp4")
            manifest = self.config(base, [("bs8-s7", 8, csv_path)])
            cfg = json.loads(manifest.read_text(encoding="utf-8-sig"))
            timing_case = core.normalize_case(cfg["defaults"], cfg["cases"][0], base)
            # A stale false manifest declaration must not bypass the normal
            # driver's check against the genuine stack-enabled capture metadata.
            with self.assertRaisesRegex(ValueError, "enabled stack/modules"):
                core._analyze_case_rank({"case": timing_case, "model": fixtures.MODEL, "base": base,
                                        "files": [csv_path], "names": ["dp4"], "residual_dir": None,
                                        "schedule": None, "excluded": set(), "required": {}}, 0)
            (csv_path.parent.parent / "profiler_info_4.json").unlink()
            out = base / "partial"
            run = self.run_probe(manifest, out)
            self.assertNotEqual(run.returncode, 0, run.stdout + run.stderr)
            case = json.loads((out / "u_host_stack_full.json").read_text(encoding="utf-8"))["cases"][0]
            self.assertEqual(case["status"], "PARTIAL")
            self.assertIsNone(case.get("error"), case)
            self.assertIs(case["stack_metadata_verified"], False)
            self.assertTrue(any("METADATA_MISSING" in warning for warning in case["warnings"]))
            self.assertGreater(case["counts"]["python_function_events"], 0)

    def test_conflicting_real_profiler_metadata_stops_only_that_case(self):
        with tempfile.TemporaryDirectory(prefix="ep128 metadata conflict ") as temporary:
            base = Path(temporary)
            csv_path, _ = self.stack_capture(base / "capture" / "dp4")
            (csv_path.parent.parent / "profiler_info_other.json").write_text(json.dumps({
                "config": {"common_config": {"with_stack": False, "with_modules": False}}
            }), encoding="utf-8")
            manifest = self.config(base, [("bs8-s7", 8, csv_path)])
            out = base / "conflict"
            run = self.run_probe(manifest, out)
            self.assertNotEqual(run.returncode, 0)
            case = json.loads((out / "u_host_stack_full.json").read_text(encoding="utf-8"))["cases"][0]
            self.assertEqual(case["status"], "ERROR")
            self.assertIn("Conflicting", case["error"])

    def test_python_events_on_an_unrelated_thread_do_not_verify_target_windows(self):
        with tempfile.TemporaryDirectory(prefix="ep128 wrong Python thread ") as temporary:
            base = Path(temporary)
            csv_path, trace_path = self.stack_capture(base / "capture" / "dp4")
            events = json.loads(trace_path.read_text(encoding="utf-8"))
            for event in events:
                if event.get("cat") == "python_function":
                    event.update(pid=11, tid=99)
            events.insert(0, {"ph": "M", "name": "process_name", "pid": 11,
                              "args": {"name": "Python"}})
            trace_path.write_text(json.dumps(events), encoding="utf-8")
            manifest = self.config(base, [("bs8-s7", 8, csv_path)])
            out = base / "wrong-thread"
            run = self.run_probe(manifest, out)
            self.assertNotEqual(run.returncode, 0, run.stdout + run.stderr)
            case = json.loads((out / "u_host_stack_full.json").read_text(encoding="utf-8"))["cases"][0]
            self.assertEqual(case["status"], "PARTIAL")
            self.assertGreater(case["counts"]["python_function_events"], 0)
            self.assertTrue(case["all_windows"])
            self.assertTrue(all(row["python_function_events"] == 0 for row in case["all_windows"]))
            self.assertTrue(all(row["python_function_covered_ns"] == 0 for row in case["all_windows"]))
            self.assertTrue(any("PYTHON" in warning for warning in case["warnings"]))

    def test_output_directory_cannot_overwrite_referenced_model_json(self):
        with tempfile.TemporaryDirectory(prefix="ep128 protected model ") as temporary:
            base = Path(temporary)
            csv_path, _ = self.stack_capture(base / "capture" / "dp4")
            manifest = self.config(base, [("bs8-s7", 8, csv_path)])
            out = base / "model-files"
            out.mkdir()
            model_path = out / "u_host_stack_full.json"
            model_bytes = json.dumps(fixtures.MODEL).encode("utf-8")
            model_path.write_bytes(model_bytes)
            cfg = json.loads(manifest.read_text(encoding="utf-8-sig"))
            cfg["sample_model"] = str(model_path)
            manifest.write_text(json.dumps(cfg), encoding="utf-8")
            run = self.run_probe(manifest, out)
            self.assertNotEqual(run.returncode, 0, run.stdout + run.stderr)
            self.assertEqual(model_path.read_bytes(), model_bytes)

    def test_twelve_complete_capture_entries_fit_share_and_keep_priority_functions(self):
        """Budget full native diagnostics, including trace and stage metadata."""
        with tempfile.TemporaryDirectory(prefix="ep128 realistic share ") as temporary:
            base = Path(temporary)
            csv_path, _ = self.stack_capture(base / "capture" / "dp4")
            manifest = self.config(base, [("bs8-s7", 8, csv_path)])
            out = base / "diagnostics"
            run = self.run_probe(manifest, out)
            full_path = out / "u_host_stack_full.json"
            self.assertTrue(full_path.is_file(), run.stdout + run.stderr)
            payload = json.loads(full_path.read_text(encoding="utf-8"))
            original = payload["cases"][0]
            self.assertIsNone(original["error"], original)
            self.assertEqual(len(original["all_stage_wall_means"]), 15)
            names = ["bs8-s7", "bs16-s7", "bs8-s15", "bs16-s15"] + [
                "bs" + str(bs) + "-s" + str(steps)
                for steps in (7, 9, 15) for bs in (2, 4, 8, 16)
                if (bs, steps) not in ((8, 7), (16, 7), (8, 15), (16, 15))]
            payload["cases"] = [dict(deepcopy(original), case=name) for name in names]
            payload["planned_cases"] = payload["completed_cases"] = 12
            before = deepcopy(payload)
            share = load_tool().bounded_share(payload)
            self.assertLessEqual(len(share), 65536)
            decoded = json.loads(share)
            self.assertEqual([row["case"] for row in decoded["cases"]], names)
            for row in decoded["cases"]:
                self.assertEqual(len(row["stage_wall_means"]), 15)
                self.assertEqual(len(row["phase_summary"]), 2)
                self.assertEqual(row["trace"]["status"], "ALIGNED_X_EVENTS")
            for row in decoded["cases"][:2]:
                prepare = next(item for item in row["window_summary"]
                               if item["kind"] == "prepare_between_host_calls")
                functions = {item["name"] for item in prepare["top_exclusive_functions"]}
                self.assertIn("numpy.cumsum", functions)
                self.assertEqual(prepare["window_mean_ns"], 8000)
            self.assertEqual(payload, before)

    def test_real_metadata_false_and_missing_rank_are_errors_without_fallback(self):
        with tempfile.TemporaryDirectory(prefix="ep128 stack metadata ") as temporary:
            base = Path(temporary)
            csv_path, _ = self.stack_capture(base / "capture" / "dp4", observed_stack=False)
            manifest = self.config(base, [("metadata-false", 8, csv_path)])
            out = base / "out"
            run = self.run_probe(manifest, out)
            self.assertNotEqual(run.returncode, 0, run.stdout + run.stderr)
            case = json.loads((out / "u_host_stack_full.json").read_text(encoding="utf-8"))["cases"][0]
            self.assertTrue(case.get("error"))
            cfg = json.loads(manifest.read_text(encoding="utf-8-sig"))
            cfg["cases"][0]["inputs"][0]["rank"] = "dp5"
            manifest.write_text(json.dumps(cfg), encoding="utf-8")
            run = self.run_probe(manifest, base / "missing-rank")
            self.assertNotEqual(run.returncode, 0)
            case = json.loads((base / "missing-rank" / "u_host_stack_full.json").read_text(encoding="utf-8"))["cases"][0]
            self.assertTrue(case.get("error"))


if __name__ == "__main__":
    unittest.main()
