"""Existing-report evidence extraction does not repeat HC analysis."""

import csv
from copy import deepcopy
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import ep128_stack_evidence as evidence
import profile_ep128 as core
import test_ep128_host_stack_probe as stack_fixtures


ROOT = Path(__file__).resolve().parents[1]
EPOCH = 1790000000000000000


def event(name, start, end, pid=10, tid=7, cat="python_function", **args):
    return {"ph": "X", "name": name, "pid": pid, "tid": tid, "cat": cat,
            "ts": str(Decimal(EPOCH + start) / 1000), "dur": str(Decimal(end - start) / 1000), "args": args}


def rows(path, values):
    fields = list(dict.fromkeys(k for row in values for k in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(values)


class ExistingStackEvidenceTests(unittest.TestCase):
    def fixture(self, base):
        events = [
            {"ph": "M", "name": "process_name", "pid": 10, "args": {"name": "Python"}},
            {"ph": "M", "name": "process_name", "pid": 20, "args": {"name": "CANN"}},
            event("prepare_inputs", 0, 200, **{"Python id": 1, "Python parent id": 0}),
            event("wait_wrapper", 10, 80, **{"Python id": 2, "Python parent id": 1}),
            event("Event::synchronize", 20, 70, cat="cpu_op"),
            event("AscendCL@aclrtSynchronizeEvent", 25, 65, pid=20, cat="cann_api", thread_id=7),
            event("wrong_worker", 0, 200, pid=11, tid=99),
            event("_expand_idx_mapping_kernel", 100, 110, cat="cpu_op"),
        ]
        trace = base / "trace.json"
        trace.write_text(json.dumps(events), encoding="utf-8")
        stage = base / "stages.csv"
        stage_rows = []
        for round_id, offset in (("hc-1", 0), ("hc-2", 400)):
            for key, kind, start, end in (("main_head", "main_to_draft", 0, 10),
                                        ("draft_metadata", "main_to_draft", 10, 100),
                                        ("draft_embedding", "main_to_draft", 100, 110),
                                        ("main_inputs", "draft_to_nextmain", 200, 260),
                                        ("main_metadata", "draft_to_nextmain", 260, 280)):
                stage_rows.append({"rank": "dp4", "round_id": round_id, "level": 0, "status": "MATCHED",
                                   "stage_key": key, "window_key": kind,
                                   "start_ns": EPOCH + offset + start, "end_ns": EPOCH + offset + end})
        rows(stage, stage_rows)
        device = base / "device.csv"
        tasks = []
        for round_id, offset in (("hc-1", 0), ("hc-2", 400)):
            for index, name, start, end in ((1, "MlaMetadata", 20, 30), (2, "Fill", 80, 90),
                                          (3, "_expand_idx_mapping_kernel", 250, 255)):
                tasks.append({"rank": "dp4", "round_id": round_id, "row_index": index,
                              "name": name, "start_ns": EPOCH + offset + start, "end_ns": EPOCH + offset + end,
                              "stream_id": 4, "task_id": index, "model_id": 49})
        rows(device, tasks)
        source = base / "source.csv"
        rows(source, [
            {"row_index": 1, "name": "MlaMetadata", "status": "AMBIGUOUS", "reason": "two explicit candidates",
             "cpu_ancestors_json": json.dumps([{"id": 3, "name": "prepare_inputs", "start": EPOCH, "end": EPOCH + 200}])},
            {"row_index": 2, "name": "Fill", "status": "LINKED", "reason": "", "cpu_event_index": 5,
             "cpu_name": "Event::synchronize", "cpu_start_ns": EPOCH + 20, "cpu_end_ns": EPOCH + 70},
            {"row_index": 3, "name": "_expand_idx_mapping_kernel", "status": "LINKED", "reason": "",
             "cpu_event_index": 8, "cpu_name": "_expand_idx_mapping_kernel",
             "cpu_ancestors_json": "[]"},
        ])
        functions = [{"name": "Event::synchronize", "lane": "cpu_thread", "ns": 50, "sync": True, "event_indexes": [5]},
                     {"name": "AscendCL@aclrtSynchronizeEvent", "lane": "cann_thread", "ns": 40, "sync": True, "event_indexes": [6]},
                     {"name": "prepare_inputs", "lane": "cpu_thread", "ns": 90, "sync": False, "event_indexes": [3]}]
        window = {"id": "prepare-1", "kind": "prepare_gap_before_expand", "round_id": "hc-1", "status": "MATCHED",
                  "window_ns": 90, "attributed_ns": 90, "ambiguous_ns": 0, "unrecorded_ns": 0,
                  "anchor": {"event_index": 8, "name": "_expand_idx_mapping_kernel", "pid": "10", "tid": "7"},
                  "exclusive_functions": functions, "inclusive_functions": functions}
        entry = {"case": "bs8-s7", "rank": "dp4", "bs": 8, "S": 7, "status": "PARTIAL", "error": None,
                 "warnings": [], "declared_with_stack": False, "stack_metadata_verified": True,
                 "profiler_metadata": [{"with_stack": False, "with_modules": True}],
                 "trace": {"status": "ALIGNED_X_EVENTS", "path": str(trace), "clock_offset_ns": "0",
                           "sha256": hashlib.sha256(trace.read_bytes()).hexdigest()},
                 "stage_details": {"stages_csv": str(stage), "device_tasks_csv": str(device), "source_links_csv": str(source)},
                 "phase_summary": [], "window_summary": [], "all_stage_wall_means": [], "windows": [window]}
        second = json.loads(json.dumps(entry))
        second.update(case="bs16-s7", bs=16)
        full = base / "u_host_stack_full.json"
        full.write_text(json.dumps({"schema": "u_host_stack_v1", "rank": "dp4", "cases": [entry, second]}), encoding="utf-8")
        return full, trace

    def test_full_examples_wait_parents_stage7_and_hc_are_read_without_remeasurement(self):
        with tempfile.TemporaryDirectory(prefix="ep128 existing stack ") as temporary:
            base = Path(temporary)
            full, trace = self.fixture(base)
            original = {p: p.read_bytes() for p in base.iterdir() if p.is_file()}
            with patch.object(core, "read_capture", side_effect=AssertionError("must not remeasure")), \
                    patch.object(core, "measure_capture", side_effect=AssertionError("must not remeasure")):
                result, destination = evidence.export_existing(core, full, base / "export", rank="dp4")
            self.assertLessEqual(destination.stat().st_size, 65536)
            case = result["cases"][0]
            self.assertEqual(case["status"], "EXPORTED", case)
            self.assertIs(case["profiler_metadata"][0]["with_stack"], False)
            self.assertIs(case["profiler_metadata"][0]["with_modules"], True)
            self.assertEqual(case["source_links"]["groups"]["metadata"]["status_counts"], {"AMBIGUOUS": 1})
            self.assertEqual(case["stage7"]["draft_hc_n"], 2)
            self.assertEqual(case["stage7"]["main_hc_n"], 1)
            self.assertEqual(case["stage7"]["main_hc_ms_mean"], 120 / 1e6)
            self.assertEqual(case["stage7"]["gaps"][0]["duration_ns"], 50)
            self.assertTrue(case["stage7"]["main_inputs_gaps"])
            traced = case["trace_evidence"]
            self.assertEqual(traced["status"], "VERIFIED")
            events = {e["event_index"]: e for e in traced["events"]}
            self.assertEqual(events[4]["python_parent_id"], "1")
            chain = next(c for c in traced["wait_parent_chains"] if c["target_event_index"] == 5)
            self.assertEqual(chain["parents"][0]["event_index"], 4)
            self.assertEqual(chain["parents"][1], {"event_index": 3, "relation": "python_parent_id"})
            self.assertNotIn(7, [p.get("event_index") for p in chain["parents"]])
            self.assertEqual({p: p.read_bytes() for p in original}, original)

    def test_cli_exports_native_decimal_trace_times_without_epoch_precision_loss(self):
        with tempfile.TemporaryDirectory(prefix="ep128 numeric trace ") as temporary:
            base = Path(temporary)
            full, trace = self.fixture(base)
            raw_events = json.loads(trace.read_text(encoding="utf-8"))
            # Real profiler traces use JSON numbers, not the timestamp strings
            # used by the older fixture. Keep the original decimal lexemes:
            # binary floats cannot represent these epoch + nanosecond values.
            raw_events[2]["ts"] = str(EPOCH // 1000) + ".001"
            raw_events[2]["dur"] = "0.200"
            raw_events[3]["dur"] = "0.123"
            numeric_json = re.sub(r'"(ts|dur)": "(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"',
                                  r'"\1": \2', json.dumps(raw_events))
            trace.write_text(numeric_json, encoding="utf-8")
            saved = json.loads(full.read_text(encoding="utf-8"))
            for case in saved["cases"]:
                case["trace"]["sha256"] = hashlib.sha256(trace.read_bytes()).hexdigest()
            full.write_text(json.dumps(saved), encoding="utf-8")
            parsed = list(core.residual_iter_trace_events(trace))
            self.assertIs(type(parsed[2]["ts"]), Decimal)
            self.assertIs(type(parsed[3]["dur"]), Decimal)
            self.assertEqual(parsed[2]["ts"], Decimal("1790000000000000.001"))
            original = {p: p.read_bytes() for p in base.iterdir() if p.is_file()}
            out = base / "export"
            run = subprocess.run([sys.executable, "-B", "-S", str(ROOT / "tools/u_host_stack_probe.py"),
                                  "--export-existing", str(full), "--rank", "dp4", "--case", "bs8-s7",
                                  "--case", "bs16-s7", "--out", str(out)],
                                 capture_output=True, text=True, timeout=30)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            destination = out / "u_host_stack_evidence.json"
            self.assertLessEqual(destination.stat().st_size, 65536)
            result = json.loads(destination.read_text(encoding="utf-8"))
            self.assertEqual(result["schema"], "u_host_stack_evidence_v1")
            self.assertEqual([c["case"] for c in result["cases"]], ["bs8-s7", "bs16-s7"])
            for case in result["cases"]:
                self.assertEqual(case["status"], "EXPORTED", case)
                traced = case["trace_evidence"]
                self.assertEqual(traced["status"], "VERIFIED")
                events = {e["event_index"]: e for e in traced["events"]}
                self.assertEqual(events[3]["ts"], "1790000000000000.001")
                self.assertEqual(events[3]["dur"], "0.200")
                self.assertEqual(events[3]["start_ns"], EPOCH + 1)
                self.assertEqual(events[3]["end_ns"], EPOCH + 201)
                self.assertIs(type(events[3]["start_ns"]), int)
                self.assertEqual(events[4]["dur"], "0.123")
                self.assertEqual(events[4]["duration_ns"], 123)
                self.assertEqual(events[4]["end_ns"], EPOCH + 133)
                self.assertEqual(traced["sha256"], hashlib.sha256(trace.read_bytes()).hexdigest())
            self.assertEqual({p: p.read_bytes() for p in original}, original)

    def test_changed_trace_is_reported_and_does_not_supply_parent_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            full, trace = self.fixture(base)
            trace.write_bytes(trace.read_bytes() + b"\n")
            result, out = evidence.export_existing(core, full, base / "export")
            self.assertTrue(out.is_file())
            self.assertTrue(all(c["status"] == "ERROR" and "SHA256" in c["error"] for c in result["cases"]))
            self.assertTrue(all("trace_evidence" not in c for c in result["cases"]))

    def test_generic_wait_gap_and_same_round_preparation_overlaps_are_retained(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            full, trace = self.fixture(base)
            raw = json.loads(trace.read_text(encoding="utf-8"))
            raw.extend([
                event("dispatch_and_sync", 180, 275, **{"Python id": 9, "Python parent id": 0}),
                event("<built-in method wait of pybind11 object>", 20, 85, **{"Python id": 10, "Python parent id": 1}),
                event("<built-in method wait of pybind11 object>", 210, 255, **{"Python id": 11, "Python parent id": 9}),
                event("_npu_getCurrentRawStreamNoWait", 205, 259, **{"Python id": 12, "Python parent id": 9}),
                event("draft.py(10): run_fullgraph", 390, 530, **{"Python id": 13, "Python parent id": 0}),
                event("metadata.py(20): mixed_quant_sparse_flash_mla_metadata_fp8", 350, 420,
                      **{"Python id": 14, "Python parent id": 0}),
                event("AscendCL@aclnnMixedQuantSparseFlashMlaMetadataGetWorkspaceSize", 355, 395,
                      pid=20, cat="cann_api", thread_id=7),
            ])
            trace.write_text(json.dumps(raw), encoding="utf-8")
            saved = json.loads(full.read_text(encoding="utf-8"))
            for case in saved["cases"]:
                case["trace"]["sha256"] = hashlib.sha256(trace.read_bytes()).hexdigest()
            longest = {"name": "<built-in method wait of pybind11 object>", "ns": 65,
                       "sync": False, "event_indexes": [10]}
            for key in ("exclusive_functions", "inclusive_functions"):
                saved["cases"][0]["windows"][0][key].insert(0, deepcopy(longest))
                saved["cases"][1]["windows"][0][key] = []
            saved["cases"][1]["windows"][0]["kind"] = "metadata_before_enqueue"
            full.write_text(json.dumps(saved), encoding="utf-8")
            result, _ = evidence.export_existing(core, full, base / "export")
            for case in result["cases"]:
                self.assertEqual(case["status"], "EXPORTED", case)
                traced = case["trace_evidence"]
                events = {e["event_index"]: e for e in traced["events"]}
                chains = {c["target_event_index"]: c for c in traced["wait_parent_chains"]}
                self.assertIn(11, traced["primary_waits_retained"])
                self.assertEqual(traced["primary_waits_missing"], [])
                self.assertEqual(chains[11]["parents"][0], {"event_index": 9, "relation": "python_parent_id"})
                self.assertNotIn(12, chains)
                self.assertTrue(chains[11]["target_is_wait"])
                workspace_overlap = events[15]["round_overlaps"]
                self.assertIn({"round_id": "hc-2", "region": "main_hc", "overlap_ns": 40}, workspace_overlap)
                call_overlaps = events[13]["round_overlaps"]
                self.assertIn({"round_id": "hc-2", "region": "main_hc", "overlap_ns": 10}, call_overlaps)
                self.assertIn({"round_id": "hc-2", "region": "draft_metadata", "overlap_ns": 90}, call_overlaps)
                paired = case["stage7"]["paired_rounds"]
                self.assertEqual(paired["n"], 1)
                self.assertEqual(paired["closure_max_abs_ns"], 0)
                self.assertEqual(paired["cycle_ms_mean"], 400 / 1e6)
            first = result["cases"][0]["trace_evidence"]
            self.assertIn(10, first["primary_waits_retained"])
            self.assertFalse(evidence._is_wait("_npu_getCurrentRawStreamNoWait"))

    def test_budget_protects_primary_generic_wait_and_reports_primary_omission(self):
        cases = []
        for name in ("first", "second"):
            primary = {"target_event_index": 1, "target_is_wait": True, "target_duration_ns": 9000000,
                       "primary_windows": [{"window_kind": "prepare_gap_before_expand", "round_id": "hc-2"}],
                       "parents": [{"event_index": i, "relation": "python_parent_id"} for i in range(2, 10)]}
            events = [{"event_index": i, "name": "dispatch_and_sync" if i == 2 else "generic wait or parent " + str(i),
                       "duration_ns": 9000000} for i in range(1, 10)]
            chains = [primary]
            for j in range(30):
                numbers = [100 + 20 * j + i for i in range(13)]
                chains.append({"target_event_index": numbers[0], "target_is_wait": True, "target_duration_ns": 1000,
                               "parents": [{"event_index": i, "relation": "python_parent_id"} for i in numbers[1:]]})
                events.extend({"event_index": i, "name": "Event::synchronize " + "long_wrapper_path/" * 30,
                               "duration_ns": 1000} for i in numbers)
            cases.append({"case": name, "status": "EXPORTED", "windows": [], "window_summary": [],
                          "trace_evidence": {"status": "VERIFIED", "primary_waits_expected": [1],
                                             "events": events, "wait_parent_chains": chains}})
        payload = {"cases": cases}
        before = deepcopy(payload)
        data = evidence.bounded_evidence(payload)
        self.assertLessEqual(len(data), 65536)
        for case in json.loads(data)["cases"]:
            self.assertEqual(case["primary_waits_retained"], [1])
            self.assertEqual(case["primary_waits_missing"], [])
            traced = case["trace_evidence"]
            primary = next(c for c in traced["wait_parent_chains"] if c["target_event_index"] == 1)
            self.assertGreaterEqual(len(primary["parents"]), 4)
            self.assertEqual(primary["parents"][0], {"event_index": 2, "relation": "python_parent_id"})
            available = {e["event_index"] for e in traced["events"]}
            self.assertTrue(all(p["event_index"] in available for c in traced["wait_parent_chains"] for p in c["parents"]))
        self.assertEqual(payload, before)
        pathological = {"cases": [deepcopy(cases[0])]}
        pathological["cases"][0]["windows"] = [{"huge_display": "x" * 10000}]
        output = json.loads(evidence.bounded_evidence(pathological, limit=1800))["cases"][0]
        self.assertEqual(output["status"], "ERROR")
        self.assertEqual(output["primary_waits_missing"], [1])
        self.assertEqual(output["trace_evidence"]["status"], "PRIMARY_WAIT_EVIDENCE_MISSING")

    def test_cli_reports_missing_primary_wait_as_nonzero_and_saves_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            full, _ = self.fixture(base)
            saved = json.loads(full.read_text(encoding="utf-8"))
            missing = {"name": "<built-in method wait of pybind11 object>", "ns": 80,
                       "sync": False, "event_indexes": [999999]}
            for key in ("exclusive_functions", "inclusive_functions"):
                saved["cases"][0]["windows"][0][key].insert(0, deepcopy(missing))
            full.write_text(json.dumps(saved), encoding="utf-8")
            out = base / "export"
            run = subprocess.run([sys.executable, "-B", "-S", str(ROOT / "tools/u_host_stack_probe.py"),
                                  "--export-existing", str(full), "--out", str(out)],
                                 capture_output=True, text=True, timeout=30)
            self.assertEqual(run.returncode, 1, run.stdout + run.stderr)
            result = json.loads((out / "u_host_stack_evidence.json").read_text(encoding="utf-8"))
            first, second = result["cases"]
            self.assertEqual(first["status"], "ERROR")
            self.assertEqual(first["primary_waits_missing"], [999999])
            self.assertEqual(first["trace_evidence"]["status"], "PRIMARY_WAIT_EVIDENCE_MISSING")
            self.assertEqual(second["status"], "EXPORTED")

    def test_source_output_collision_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            full, _ = self.fixture(base)
            colliding = base / "u_host_stack_evidence.json"
            colliding.write_bytes(full.read_bytes())
            original = colliding.read_bytes()
            with self.assertRaisesRegex(ValueError, "overlaps"):
                evidence.export_existing(core, colliding, base)
            self.assertEqual(colliding.read_bytes(), original)

    def test_unselected_case_input_is_also_protected(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            full, _ = self.fixture(base)
            saved = json.loads(full.read_text())
            colliding = base / "u_host_stack_evidence.json"
            colliding.write_text("source input", encoding="utf-8")
            saved["cases"][1]["kernel_path"] = str(colliding)
            full.write_text(json.dumps(saved), encoding="utf-8")
            original = colliding.read_bytes()
            with self.assertRaisesRegex(ValueError, "overlaps"):
                evidence.export_existing(core, full, base, names=["bs8-s7"])
            self.assertEqual(colliding.read_bytes(), original)

    def test_pathological_budget_fallback_clears_local_gap_references(self):
        payload = {"cases": [{"windows": [{"long_display_value": "x" * 10000}],
                              "window_summary": [],
                              "trace_evidence": {"events": [], "wait_parent_chains": []},
                              "stage7": {"gaps": [{"host_event_indexes": [1]}],
                                         "main_inputs_gaps": [{"host_event_indexes": [2]}]}}]}
        output = json.loads(evidence.bounded_evidence(payload, limit=1500))
        case = output["cases"][0]
        self.assertEqual(case["trace_evidence"]["status"], "OMITTED_FOR_BYTE_LIMIT")
        for gap in case["stage7"]["gaps"] + case["stage7"]["main_inputs_gaps"]:
            self.assertEqual(gap["host_event_indexes"], [])
            self.assertEqual(gap["host_evidence_status"], "OMITTED_FOR_BYTE_LIMIT")

    def test_cli_reuses_same_entrypoint_and_preserves_saved_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            full, _ = self.fixture(base)
            saved = json.loads(full.read_text())
            saved["cases"][1]["status"] = "ERROR"
            saved["cases"][1]["error"] = "saved clock mismatch"
            saved["cases"][1]["trace"]["status"] = "CLOCK_MISMATCH"
            full.write_text(json.dumps(saved), encoding="utf-8")
            out = base / "export"
            run = subprocess.run([sys.executable, "-B", "-S", str(ROOT / "tools/u_host_stack_probe.py"),
                                  "--export-existing", str(full), "--out", str(out)],
                                 capture_output=True, text=True, timeout=30)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            result = json.loads((out / "u_host_stack_evidence.json").read_text())
            second = result["cases"][1]
            self.assertEqual(second["source_error"], "saved clock mismatch")
            self.assertEqual(second["trace_evidence"]["status"], "CLOCK_NOT_ALIGNED")

    def test_large_two_case_parent_export_is_bounded_and_keeps_representative_waits(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            full, _ = self.fixture(base)
            result, _ = evidence.export_existing(core, full, base / "export")
            for case in result["cases"]:
                traced = case["trace_evidence"]
                traced["events"], traced["wait_parent_chains"] = [], []
                for identity in range(30):
                    numbers = [identity * 20 + j for j in range(13)]
                    traced["wait_parent_chains"].append({"target_event_index": numbers[0], "target_is_wait": True,
                        "parents": [{"event_index": j, "relation": "python_parent_id"} for j in numbers[1:]]})
                    traced["events"].extend({"event_index": j, "name": "long_python_path/" * 35 + str(j),
                                             "duration_ns": 100, "pid": "10", "tid": "7"} for j in numbers)
                funcs = [{"name": "Event::synchronize/" + "long_python_path/" * 35 + str(i), "ns": 100 - i,
                          "sync": True, "event_indexes": [i * 20]} for i in range(20)]
                case["windows"][0]["inclusive_functions"] = deepcopy(funcs)
                case["windows"][0]["exclusive_functions"] = deepcopy(funcs)
                case["window_summary"] = [{"kind": "prepare_gap_before_expand", "window_mean_ns": 90,
                                           "matched_windows": 10, "top_exclusive_functions": deepcopy(funcs)}]
            before = deepcopy(result)
            data = evidence.bounded_evidence(result)
            self.assertLessEqual(len(data), 65536)
            output = json.loads(data)
            self.assertTrue(output["export_omissions"])
            for case in output["cases"]:
                traced = case["trace_evidence"]
                self.assertTrue(traced["wait_parent_chains"])
                self.assertTrue(any(c["target_is_wait"] for c in traced["wait_parent_chains"]))
                available = {e["event_index"] for e in traced["events"]}
                for chain in traced["wait_parent_chains"]:
                    self.assertIn(chain["target_event_index"], available)
                    self.assertTrue(all(p["event_index"] in available for p in chain["parents"]))
                for gap in case["stage7"]["gaps"] + case["stage7"]["main_inputs_gaps"]:
                    self.assertTrue(all(i in available for i in gap.get("host_event_indexes", [])))
                self.assertEqual(case["window_summary"][0]["window_mean_ns"], 90)
            self.assertEqual(result, before)

    def test_with_modules_true_is_not_falsely_reported_as_stack_flag_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            fixtures = stack_fixtures.HostStackProbeCLITests()
            csv_path, _ = fixtures.stack_capture(base / "capture" / "dp4", observed_stack=False)
            info = csv_path.parent.parent / "profiler_info_4.json"
            metadata = json.loads(info.read_text())
            metadata["config"]["common_config"]["with_modules"] = True
            info.write_text(json.dumps(metadata), encoding="utf-8")
            manifest = fixtures.config(base, [("bs8-s7", 8, csv_path)])
            run = fixtures.run_probe(manifest, base / "out")
            case = json.loads((base / "out/u_host_stack_full.json").read_text())["cases"][0]
            self.assertIsNone(case["error"], run.stdout + run.stderr)
            self.assertIn("PYTHON_TRACING_VIA_WITH_MODULES", case["warnings"])
            self.assertFalse(any("MANIFEST_STACK_FLAG_DIFFERS" in w for w in case["warnings"]))


if __name__ == "__main__":
    unittest.main()
