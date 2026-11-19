"""Locate existing CSV gaps relative to their recorded following CANN calls."""

import csv
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "u_gap_submission_probe.py"
EPOCH = 1790000000000000007


def load_probe():
    spec = importlib.util.spec_from_file_location("u_gap_submission_probe", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class GapSubmissionTests(unittest.TestCase):
    def setUp(self):
        self.probe = load_probe()

    def gap(self, **changes):
        return {"rank": "dp4", "round_id": "hc-7", "stage_key": "main_metadata",
                "kind": "no_device_record", "start_ns": str(EPOCH + 10),
                "end_ns": str(EPOCH + 100), "duration_ns": "90", "duration_ms": "0.00009",
                "previous_row_indices": "[1]", "following_row_indices": "[2]", **changes}

    def device(self, row=2, **changes):
        return {"rank": "dp4", "round_id": "hc-7", "row_index": str(row),
                "row_indices": json.dumps([row]), "name": "metadata_kernel", "stream_id": "4",
                "task_id": str(row + 1000), "model_id": "52", "start_ns": str(EPOCH + 100),
                "end_ns": str(EPOCH + 110), "wait_ms_diagnostic": "0.00003", **changes}

    def source(self, **changes):
        return {"row_index": "2", "name": "metadata_kernel", "status": "API_ONLY",
                "device_start_ns": str(EPOCH + 100), "device_end_ns": str(EPOCH + 110),
                "node_event_index": "17", "node_start_ns": str(EPOCH + 30),
                "node_end_ns": str(EPOCH + 70), **changes}

    def classify(self, gap=None, links=None, devices=None):
        return self.probe.classify_gap(
            self.gap() if gap is None else gap,
            {2: self.source()} if links is None else links,
            {("hc-7", 2): self.device()} if devices is None else devices,
        )

    def partition(self, result):
        names = ("before_node_ns", "inside_node_ns", "after_node_ns", "unknown_ns")
        values = [result[name] for name in names]
        self.assertEqual(sum(values), result["duration_ns"])
        return values

    def test_api_only_preserves_exact_nanoseconds_and_splits_at_node_boundaries(self):
        result = self.classify()
        self.assertEqual(self.partition(result), [20, 40, 30, 0])

    def test_node_returned_before_gap_accounts_for_the_whole_gap(self):
        source = self.source(node_start_ns=str(EPOCH - 20), node_end_ns=str(EPOCH - 10))
        result = self.classify(links={2: source})
        self.assertEqual(self.partition(result), [0, 0, 90, 0])

    def test_missing_and_conflicting_sources_keep_the_gap_unknown(self):
        for sources in ({}, {2: None}, {2: self.source(status="AMBIGUOUS")},
                        {2: self.source(device_start_ns=str(EPOCH + 101))},
                        {2: self.source(device_end_ns=str(EPOCH + 111))}):
            with self.subTest(sources=sources):
                result = self.classify(links=sources)
                self.assertEqual(self.partition(result), [0, 0, 0, 90])

    def test_concurrent_following_tasks_are_not_assigned_to_one_node(self):
        devices = {("hc-7", 2): self.device(), ("hc-7", 3): self.device(3)}
        result = self.classify(gap=self.gap(following_row_indices="[2,3]"), devices=devices)
        self.assertEqual(self.partition(result), [0, 0, 0, 90])

    def test_tail_gap_uses_the_task_starting_at_the_next_stage_boundary(self):
        result = self.classify(gap=self.gap(following_row_indices="[]"))
        self.assertEqual(self.partition(result), [20, 40, 30, 0])
        self.assertTrue(result["example"]["stage_boundary"])

    def test_cli_reads_existing_bom_exports_from_a_separate_working_directory(self):
        with tempfile.TemporaryDirectory(prefix="ep128 gap submission ") as temporary:
            base = Path(temporary)
            report_dir = base / "existing report"
            report_dir.mkdir()
            rows = (("segments.csv", [self.gap()]), ("links.csv", [self.source()]),
                    ("device.csv", [self.device()]))
            for name, content in rows:
                with (report_dir / name).open("w", encoding="utf-8-sig", newline="") as stream:
                    writer = csv.DictWriter(stream, tuple(content[0]))
                    writer.writeheader()
                    writer.writerows(content)
            diagnostic = {
                "residual_ms_mean": 0.001,
                "stage_summary": [{"stage_key": "main_metadata", "window_key": "draft_to_nextmain",
                                   "rounds": 1, "status": "MATCHED", "duration_ms_mean": 0.0001}],
                "trace": {"status": "ALIGNED_X_EVENTS"},
                "source_links_csv": "links.csv", "device_tasks_csv": "device.csv",
                "focus_diagnostics": {
                    "segments_csv": "segments.csv", "stages": [
                        {"stage_key": key, "per_step": [{
                            "round_id": "hc-7", "status": "MATCHED",
                            "no_recorded_ms": 0.00009 if key == "main_metadata" else 0,
                        }]}
                        for key in ("main_inputs", "main_metadata")
                    ],
                },
            }
            report = {"cases": [{"name": "bs16-s15", "config": {"bs": 16, "draft_tokens": 15},
                                 "measurements": [{"rank": "dp4", "step_residuals": diagnostic}]}]}
            report_path = report_dir / "model_architecture_report.json"
            report_path.write_text(json.dumps(report), encoding="utf-8-sig")
            original = {path.name: path.read_bytes() for path in report_dir.iterdir()}
            output = base / "small result.json"
            result = subprocess.run(
                [sys.executable, "-B", "-S", str(TOOL), str(report_dir), "--rank", "dp4", "--out", str(output)],
                cwd=base, capture_output=True, text=True, encoding="utf-8", timeout=30,
                env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            payload = output.read_bytes()
            self.assertLessEqual(len(payload), 65536)
            report = json.loads(payload)
            self.assertEqual(report["rank"], "dp4")
            self.assertEqual(report["cases"][0]["cycle_residual_ms_mean"], 0.001)
            self.assertEqual(report["cases"][0]["all_stage_wall_means"], diagnostic["stage_summary"])
            stages = {row["stage_key"]: row for row in report["cases"][0]["stages"]}
            metadata = stages["main_metadata"]
            self.assertEqual(metadata["gap_count"], 1)
            self.assertEqual(metadata["means_ms"]["before_node"], 20 / 1e6)
            self.assertEqual(metadata["means_ms"]["inside_node"], 40 / 1e6)
            self.assertEqual(metadata["means_ms"]["after_node"], 30 / 1e6)
            self.assertEqual(stages["main_inputs"]["means_ms"]["duration"], 0)
            self.assertEqual({path.name: path.read_bytes() for path in report_dir.iterdir()}, original)

            with (report_dir / "links.csv").open("a", encoding="utf-8", newline="") as stream:
                csv.DictWriter(stream, tuple(self.source())).writerow(self.source(node_end_ns=str(EPOCH + 80)))
            duplicate = subprocess.run(
                [sys.executable, "-B", "-S", str(TOOL), str(report_dir), "--rank", "dp4", "--out", str(output)],
                cwd=base, capture_output=True, text=True, encoding="utf-8", timeout=30,
                env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            )
            self.assertEqual(duplicate.returncode, 0, duplicate.stdout + duplicate.stderr)
            report = json.loads(output.read_bytes())
            metadata = next(row for row in report["cases"][0]["stages"] if row["stage_key"] == "main_metadata")
            self.assertEqual(metadata["means_ms"]["unknown"], 90 / 1e6)


if __name__ == "__main__":
    unittest.main()
