"""The CLI keeps task interiors and source failures independently of U/Host readiness."""

from contextlib import ExitStack
from copy import deepcopy
from decimal import Decimal
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import profile_ep128 as core
from test_ep128_swimlane_probe import ROOT, TOOL, input_fixture, probe


class FullDeviceExportTests(unittest.TestCase):
    def analyze(self, directory, *, source_error=None, clock_failure=False, copies=1, detail_limit=0, trace_only=False):
        case, capture, measured, kernel, trace, clock = input_fixture(directory)
        if trace_only:
            records = json.loads(trace.read_bytes())
            records.append({"ph": "X", "name": "LateMemcpyAsync", "cat": "dma", "pid": 900, "tid": 4,
                "ts": str(Decimal(measured["rounds"][0]["end_ns"])/1000), "dur": "0.007", "args": {}})
            trace.write_text(json.dumps(records), encoding="utf-8")
            clock["sha256"] = hashlib.sha256(trace.read_bytes()).hexdigest()
            clock["process_names"]["900"] = "Ascend Hardware"
        rounds, anchors = probe.round_coordinates(measured, capture, case)
        for index in range(1, copies):
            row = deepcopy(rounds[0])
            row["round_id"] = "hc-" + str(index+1)
            rounds.append(row)
        if clock_failure:
            clock["status"] = "CLOCK_MISMATCH"
        with ExitStack() as stack:
            for name, value in (("normalize_case", case), ("_fixed_overlap_input", kernel),
                                ("_case_trace_path", trace), ("read_capture", capture),
                                ("measure_capture", measured),
                                ("residual_trace_scan", (clock, None if clock_failure else 0))):
                stack.enter_context(patch.object(core, name, return_value=value))
            stack.enter_context(patch.object(core, "validate_models"))
            stack.enter_context(patch.object(probe, "round_coordinates", return_value=(rounds, anchors)))
            real_export = probe.export_device_task_links
            exporter = stack.enter_context(patch.object(probe, "export_device_task_links",
                side_effect=source_error if source_error else real_export))
            result = probe.analyze_case(case, {"sample_model": {}}, directory, "dp4", lambda _: None, detail_limit)
        return result, capture, exporter

    def test_default_keeps_every_round_and_full_task_catalog(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, capture, exporter = self.analyze(Path(tmp), copies=3)
        self.assertEqual(result["status"], "READY", result.get("error"))
        self.assertEqual([r["round_id"] for r in exporter.call_args.args[2]], ["hc-1", "hc-2", "hc-3"])
        device = result["device_task_links"]
        self.assertEqual(len(device["device_tasks"]), len(capture["tasks"]))
        self.assertEqual(len(device["links"]), len(capture["tasks"]))
        self.assertEqual(result["device_source_round_ids"], ["hc-1", "hc-2", "hc-3"])
        self.assertEqual(result["anchor_links"]["resolution_reused_from"], "device_task_links")
        self.assertNotIn("anchor_links", device)
        self.assertNotIn("trace_device_events", result["host_scan"])

    def test_positive_detail_limit_only_limits_source_round_membership(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, capture, exporter = self.analyze(Path(tmp), copies=3, detail_limit=1)
        self.assertEqual(len(exporter.call_args.args[2]), 1)
        self.assertEqual(len(result["rounds"]), 3)
        self.assertEqual(len(result["device_task_links"]["device_tasks"]), len(capture["tasks"]))

    def test_source_exception_keeps_valid_host_and_device_catalog(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, capture, _ = self.analyze(Path(tmp), source_error=OSError("source fixture failure"))
        self.assertEqual(result["status"], "READY")
        self.assertTrue(result["rounds"][0]["host_lanes"])
        self.assertEqual(result["device_task_status"], "ERROR")
        self.assertEqual(len(result["device_task_links"]["device_tasks"]), len(capture["tasks"]))
        self.assertIn("source fixture failure", result["device_task_links"]["error"])
        self.assertEqual(result["device_task_links"]["links"], [])

    def test_source_exception_also_keeps_already_read_trace_only_DMA(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, capture, _ = self.analyze(Path(tmp), trace_only=True, source_error=OSError("source fixture failure"))
        device = result["device_task_links"]
        self.assertEqual(result["device_task_status"], "ERROR")
        self.assertEqual(len(device["device_tasks"]), len(capture["tasks"])+1)
        dma, = [t for t in device["device_tasks"] if t["source_kind"] == "TRACE_ONLY"]
        self.assertEqual(dma["name"], "LateMemcpyAsync")
        self.assertEqual(int(dma["end_ns"])-int(dma["start_ns"]), 7)
        self.assertEqual(device["trace_only_task_status"], "RETAINED")
        self.assertFalse(device["task_edges"] or device["source_events"])

    def test_clock_failure_still_exports_every_csv_task_without_host_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, capture, exporter = self.analyze(Path(tmp), clock_failure=True)
        self.assertEqual(result["status"], "ERROR")
        self.assertEqual(result["device_task_status"], "CSV_ONLY_CLOCK_UNVERIFIED")
        self.assertEqual(len(result["device_task_links"]["device_tasks"]), len(capture["tasks"]))
        self.assertEqual(result["device_task_links"]["source_events"], [])
        self.assertEqual(result["host_events"], [])
        self.assertTrue(all(b["relation"] == "DEVICE_CYCLE_OVERLAP"
                            for b in result["device_task_links"]["task_round_bindings"]))
        exporter.assert_not_called()

    def test_tables_preserve_trace_only_identity_and_unclipped_nanoseconds(self):
        origin = 1790000000000000001
        task = {"task_key": "trace:9", "source_kind": "TRACE_ONLY", "csv_row_index": None,
                "trace_event_index": 9, "name": "MemcpyAsync", "start_ns": origin-11,
                "end_ns": origin+19, "stream_id": "2", "task_id": "987"}
        link = {"task_key": task["task_key"], "source_kind": "TRACE_ONLY", "status": "LINKED",
                "cpu_event_index": 2, "cpu_name": "metadata builder enqueue", "cpu_start_ns": origin-40,
                "cpu_end_ns": origin-20, "flow_ids": ["5"],
                "host_associations": [{"host_event_index": 1, "runtime_submission_confirmed": True}]}
        binding = {"task_key": task["task_key"], "round_id": "hc-3", "relation": "EXPLICIT_SOURCE_OWNER",
                   "role": "state_attn", "host_event_index": 1}
        payload = {"cases": [{"case": "bs16-s15", "rank": "dp4", "device_task_links": {
            "device_tasks": [task], "links": [link], "task_round_bindings": [binding]}}]}
        with tempfile.TemporaryDirectory() as tmp:
            paths = probe.write_device_tables(payload, Path(tmp), "dp4")
            def rows(path):
                with path.open(encoding="utf-8-sig", newline="") as stream:
                    return list(csv.DictReader(stream))
            catalog, sources, rounds = [rows(p) for p in paths]
        self.assertEqual(catalog[0]["csv_row_index"], "")
        self.assertEqual(catalog[0]["trace_event_index"], "9")
        self.assertEqual(catalog[0]["start_ns"], str(origin-11))
        self.assertEqual(catalog[0]["duration_ns"], "30")
        self.assertEqual(sources[0]["cpu_start_ns"], str(origin-40))
        self.assertEqual(json.loads(sources[0]["host_associations_json"]), link["host_associations"])
        self.assertEqual(json.loads(rounds[0]["binding_json"]), binding)
        self.assertEqual(rounds[0]["host_role"], "state_attn")

    def test_twelve_spawn_errors_finish_without_packaging_and_keep_readable_indexes(self):
        with tempfile.TemporaryDirectory(prefix="all tasks cases ") as tmp:
            directory = Path(tmp)
            config = json.loads((ROOT/"profile_ep128_cases.example.json").read_text(encoding="utf-8"))
            template = config["cases"][0]
            config["cases"] = [{**deepcopy(template), "name": f"bs{bs}-s{s}", "bs": bs, "draft_tokens": s,
                                "inputs": [{"rank": "dp4", "path": "missing.csv"}]}
                               for bs in (2, 4, 8, 16) for s in (7, 9, 15)]
            source = directory/"cases.json"
            source.write_text(json.dumps(config), encoding="utf-8-sig")
            original = source.read_bytes()
            out = directory/"out"
            result = subprocess.run([sys.executable, "-B", "-S", str(TOOL), str(source),
                "--rank", "dp4", "--workers", "2", "--expected-cases", "12", "--no-package", "--out", str(out)],
                cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
            self.assertEqual(result.returncode, 1, result.stdout+result.stderr)
            report = json.loads((out/"u_swimlane_dp4.json").read_bytes())
            self.assertEqual(report["completed_cases"], 12)
            self.assertEqual(report["detail_rounds_per_case"], 0)
            self.assertTrue(report["retention"]["all_round_source_relations"])
            self.assertTrue(report["retention"]["all_csv_device_tasks"])
            self.assertEqual(source.read_bytes(), original)
            for suffix in ("_status.csv", "_device_tasks.csv", "_task_sources.csv", "_task_rounds.csv"):
                self.assertTrue((out/("u_swimlane_dp4"+suffix)).is_file())
            self.assertFalse(list(out.glob("*.zip")))
            self.assertFalse(list(out.glob("*_parts_*")))
            self.assertIn("packaging skipped", result.stdout)

    def test_negative_detail_count_rejected_before_input_reads(self):
        result = subprocess.run([sys.executable, "-B", "-S", str(TOOL), "missing.json", "--detail-rounds", "-1"],
            cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
        self.assertEqual(result.returncode, 2)
        self.assertIn("0 (all rounds)", result.stderr)


if __name__ == "__main__":
    unittest.main()
