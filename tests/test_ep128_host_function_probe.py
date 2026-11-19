"""Exercise report-to-trace attribution through the Windows-compatible CLI."""

import csv
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "u_host_function_probe.py"
EPOCH = 1790000000000000007


def event(name, start, end, pid=10, category="cpu_op"):
    return {"ph": "X", "name": name, "pid": pid, "tid": 7, "cat": category,
            "ts": str(Decimal(EPOCH + start) / 1000),
            "dur": str(Decimal(end - start) / 1000)}


class HostFunctionProbeCLITests(unittest.TestCase):
    def test_existing_exports_select_exact_host_windows_and_reject_changed_trace(self):
        with tempfile.TemporaryDirectory(prefix="ep128 host CLI ") as temporary:
            base = Path(temporary)
            report_dir = base / "existing report"
            report_dir.mkdir()
            metadata = "cann_ops_transformer::mixed_quant_sparse_flash_mla_metadata"
            events = [
                {"ph": "M", "name": "process_name", "pid": 10, "args": {"name": "Python"}},
                {"ph": "M", "name": "process_name", "pid": 20, "args": {"name": "CANN"}},
                event("_apply_write_kernel", 0, 10),
                event("_expand_idx_mapping_kernel", 100, 110),
                event(metadata, 120, 180),
                event("aclnnMetadataGetWorkspaceSize", 130, 150, 20, "cann_api"),
                event("Enqueue@metadata", 170, 175),
                event("Event::synchronize", 20, 90),
            ]
            trace_path = report_dir / "trace view.json"
            trace_bytes = json.dumps({"traceEvents": events}).encode("utf-8")
            trace_path.write_bytes(trace_bytes)

            def device(row, name, start, end, round_id="hc-7"):
                return {"rank": "dp4", "round_id": round_id, "row_index": row,
                        "row_indices": json.dumps([row]), "name": name,
                        "start_ns": str(EPOCH + start), "end_ns": str(EPOCH + end),
                        "stream_id": "4", "model_id": "52", "task_id": str(row + 1000)}

            devices = [device(1, "_apply_write_kernel", 10, 20),
                       device(2, "_expand_idx_mapping_kernel", 100, 120),
                       device(3, "aclnnMixedQuantSparseFlashMlaMetadata", 200, 220),
                       device(4, "_expand_idx_mapping_kernel", 1000, 1100, "hc-8")]
            sources = []
            for task, anchor_index in zip(devices[:3], (3, 4, 5)):
                anchor = events[anchor_index - 1]
                start = int(Decimal(anchor["ts"]) * 1000)
                end = start + int(Decimal(anchor["dur"]) * 1000)
                sources.append({"row_index": task["row_index"], "status": "LINKED",
                                "device_start_ns": task["start_ns"], "device_end_ns": task["end_ns"],
                                "cpu_event_index": anchor_index, "cpu_name": anchor["name"],
                                "cpu_start_ns": str(start), "cpu_end_ns": str(end),
                                "cpu_ancestors_json": "[]",
                                "enqueue_start_ns": str(EPOCH + 170) if task["row_index"] == 3 else "",
                                "reason": ""})
            sources.append({**sources[1], "row_index": 4, "status": "AMBIGUOUS",
                            "device_start_ns": str(EPOCH + 1000), "device_end_ns": str(EPOCH + 1100)})

            def segment(phase, start, end, rows, *, kind="single_operator", previous=(), following=(), round_id="hc-7"):
                return {"rank": "dp4", "round_id": round_id, "stage_key": phase, "kind": kind,
                        "start_ns": str(EPOCH + start), "end_ns": str(EPOCH + end), "duration_ns": end - start,
                        "row_indices": json.dumps(rows), "previous_row_indices": json.dumps(previous),
                        "following_row_indices": json.dumps(following)}

            segments = [
                segment("main_inputs", 20, 100, [], kind="no_device_record", previous=[1], following=[2]),
                segment("main_inputs", 100, 120, [2]),
                segment("main_metadata", 200, 220, [3]),
                segment("main_inputs", 0, 1000, [], kind="no_device_record", following=[4], round_id="hc-8"),
                segment("main_inputs", 1000, 1100, [4], round_id="hc-8"),
            ]
            for name, rows in (("device.csv", devices), ("source.csv", sources), ("focus.csv", segments)):
                with (report_dir / name).open("w", encoding="utf-8-sig", newline="") as handle:
                    writer = csv.DictWriter(handle, tuple(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
            diagnostic = {"device_tasks_csv": "device.csv", "source_links_csv": "source.csv",
                          "focus_diagnostics": {"segments_csv": "focus.csv"},
                          "trace": {"status": "ALIGNED_X_EVENTS", "clock_offset_ns": "0",
                                    "path": trace_path.name, "sha256": hashlib.sha256(trace_bytes).hexdigest()}}
            report = {"profile_version": "synthetic-cli", "cases": [{"name": "demo", "measurements": [
                {"rank": "dp4", "step_residuals": diagnostic}]}]}
            (report_dir / "model_architecture_report.json").write_text(json.dumps(report), encoding="utf-8-sig")
            original = {path.name: path.read_bytes() for path in report_dir.iterdir()}
            output = base / "small functions.json"
            command = [sys.executable, "-B", "-S", str(TOOL), str(report_dir), "--case", "demo",
                       "--rank", "dp4", "--samples", "1", "--out", str(output)]

            def run():
                return subprocess.run(command, cwd=base, capture_output=True, text=True,
                                      encoding="utf-8", timeout=30,
                                      env={**os.environ, "PYTHONIOENCODING": "utf-8"})

            result = run()
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            payload = output.read_bytes()
            self.assertLessEqual(len(payload), 65536)
            case = json.loads(payload)["cases"][0]
            windows = {window["kind"]: window for window in case["windows"]}
            self.assertEqual(set(windows), {"prepare_gap_before_expand", "prepare_between_host_calls",
                                            "metadata_before_enqueue"})
            self.assertTrue(all(window["status"] == "MATCHED" for window in windows.values()))
            self.assertTrue(all(window["device_row"] != 4 for window in windows.values()))
            self.assertEqual(windows["prepare_gap_before_expand"]["window_ns"], 80)
            self.assertEqual(windows["prepare_between_host_calls"]["window_ns"], 90)
            self.assertEqual(windows["metadata_before_enqueue"]["device_gap_ns"], 0)
            functions = {row["name"]: row["ns"] for row in windows["metadata_before_enqueue"]["exclusive_functions"]}
            self.assertEqual(functions, {metadata: 30, "aclnnMetadataGetWorkspaceSize": 20})
            unresolved = case["largest_unresolved_sources"]["main_inputs"][0]
            self.assertEqual((unresolved["row_index"], unresolved["source_status"]), (4, "AMBIGUOUS"))
            self.assertEqual({path.name: path.read_bytes() for path in report_dir.iterdir()}, original)

            trace_path.write_bytes(trace_bytes + b"\n")
            changed = run()
            self.assertEqual(changed.returncode, 1)
            self.assertIn("SHA256", json.loads(output.read_bytes())["cases"][0]["error"])
            for name, content in original.items():
                if name != trace_path.name:
                    self.assertEqual((report_dir / name).read_bytes(), content)


if __name__ == "__main__":
    unittest.main()
