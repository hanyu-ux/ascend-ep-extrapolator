"""Source-rank U diagnostics: exact wall coverage and honest source mapping."""

import copy
import csv
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import profile_ep128 as M
try:
    from . import test_ep128_v2_measure as legacy
except ImportError:  # unittest discover -s tests imports this as a top-level module.
    import test_ep128_v2_measure as legacy


EPOCH = 1790000000000000007


def task(name, start, end, row, stream="1", **extra):
    return SimpleNamespace(name=name, op_type=name, start_ns=EPOCH + start, end_ns=EPOCH + end,
        row_index=row, row_indices=[row], stream_id=stream, input_shapes="4,8", output_shapes="4,8",
        task_type="AI_VECTOR_CORE", wait_ns=0, **extra)


def parent(key="main_inputs", start=0, end=100, round_id="hc-7"):
    return {"stage_key": key, "label": key, "level": 0, "status": "MATCHED", "round_id": round_id,
            "start_ns": str(EPOCH + start), "end_ns": str(EPOCH + end),
            "duration_ms": (end - start) / 1e6}


class FocusCoverageTests(unittest.TestCase):
    def test_multistream_boundary_clipping_duplicates_and_gaps_close_in_integer_ns(self):
        tasks = [task("_prepare_pos_seq_lens_kernel", -10, 30, 1),
                 task("_prepare_pos_seq_lens_kernel", 5, 15, 2),
                 task("unknown_kernel", 20, 50, 3, "2"), task("other_kernel", 50, 80, 4)]
        tasks[0].row_indices = [1, 17]
        before = copy.deepcopy(tasks)
        operators = {}
        measured = M.residual_focus_partition(parent(), list(reversed(tasks)), operators)
        rows = measured["segments"]
        self.assertEqual(sum(row["duration_ns"] for row in rows), 100)
        self.assertEqual(measured["closure_error_ns"], 0)
        self.assertEqual(int(rows[0]["start_ns"]), EPOCH)
        self.assertEqual(int(rows[-1]["end_ns"]), EPOCH + 100)
        self.assertTrue(all(a["end_ns"] == b["start_ns"] for a, b in zip(rows, rows[1:])))
        self.assertEqual(sum(row["duration_ns"] for row in rows if row["kind"] == "no_device_record"), 20)
        repeated = next(row for row in measured["components"] if row["operator_multiplicity"]
                        and 2 in row["operator_multiplicity"].values())
        self.assertEqual(repeated["kind"], "concurrent_operators")
        self.assertIn("×2", repeated["label"])
        self.assertTrue(any(17 in row["row_indices"] for row in rows))
        # Adjacent tasks on an identical timestamp are never falsely concurrent.
        self.assertFalse(any(3 in row["row_indices"] and 4 in row["row_indices"] for row in rows))
        self.assertEqual(tasks, before)
        self.assertEqual(measured, M.residual_focus_partition(parent(), tasks, {}))

    def test_missing_optional_or_ambiguous_anchors_do_not_erase_fine_coverage(self):
        tasks = [task("SparseAttnSharedkvMetadata", 10, 60, 1),
                 task("SparseAttnSharedkvMetadata", 20, 70, 2, "2")]
        result = M.residual_focus_partition(parent("main_metadata"), tasks)
        self.assertEqual(result["closure_error_ns"], 0)
        self.assertEqual(sum(row["duration_ns"] for row in result["segments"] if row["kind"] == "concurrent_operators"), 40)
        self.assertTrue(any(row["kind"] == "no_device_record" for row in result["segments"]))
        empty = M.residual_focus_partition(parent(), [])
        self.assertEqual(len(empty["components"]), 1)
        self.assertEqual(empty["components"][0]["mapping_status"], "UNMAPPED")
        self.assertEqual(empty["components"][0]["source_function_mapping"], [])

    def test_gaps_keep_neighbor_locations_and_do_not_claim_python_function_duration(self):
        tasks = [task("_prepare_pos_seq_lens_kernel", 10, 20, 1),
                 task("_combine_sampled_and_draft_tokens_kernel", 40, 50, 2)]
        report = M.residual_focus_partition(parent(), tasks)
        gaps = [row for row in report["components"] if row["kind"] == "no_device_record"]
        self.assertEqual(len(gaps), 3)
        middle = next(row for row in gaps if row["previous_operator_ids"] and row["following_operator_ids"])
        self.assertIn("准备位置与序列长度", middle["label"])
        self.assertIn("拼接采样与草稿token", middle["label"])
        self.assertEqual(middle["source_function_mapping"], [])
        self.assertEqual(middle["mapping_status"], "UNMAPPED")
        segment = next(row for row in report["segments"] if row["component_key"] == middle["component_key"])
        self.assertEqual((segment["previous_row_indices"], segment["following_row_indices"]), ([1], [2]))

    def test_static_entries_are_pinned_and_generic_ops_do_not_claim_builder_ownership(self):
        label, status, entries = M.residual_focus_source_mapping(task("_compute_slot_mappings_kernel", 0, 5, 1))
        self.assertEqual(status, "STATIC_KERNEL_NAME_MATCH")
        self.assertEqual(entries[0]["function"], "AscendBlockTables.compute_slot_mappings")
        self.assertIn("66b5a5e", entries[0]["version_basis"])
        self.assertIn("unverified", entries[0]["measurement_scope"])
        self.assertTrue(entries[0]["path"] and entries[0]["line"] and entries[0]["evidence"])
        _, status, entries = M.residual_focus_source_mapping(task("aclnnCast_Cast", 1, 4, 2))
        self.assertEqual((status, entries), ("UNMAPPED", []))
        for name in ("SparseAttnSharedkvMetadata", "MixedQuantSparseFlashMlaMetadata"):
            _, status, entries = M.residual_focus_source_mapping(task(name, 1, 4, 2))
            self.assertEqual(status, "STATIC_KERNEL_FAMILY_MATCH")
            self.assertIn("different ratios", entries[0]["evidence"])
            self.assertTrue(entries[0]["call_chain"])


class FocusStepTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="ep128-focus-")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)

    def diagnostics(self, rows, tasks, rounds=("hc-7", "hc-8")):
        windows = [{"round_id": round_id, "window_key": "draft_to_nextmain",
                    "start_ns": str(EPOCH + i * 1000), "end_ns": str(EPOCH + i * 1000 + 100)}
                   for i, round_id in enumerate(rounds)]
        for row in rows:
            start, end = int(row["start_ns"]), int(row["end_ns"])
            row.update(M.diagnostic_metrics(tasks, start, end))
        return M.residual_focus_diagnostics(rows, windows, M.ATimeIndex(sorted(tasks, key=lambda row: row.start_ns)),
                                            "dp9", self.directory, "capture")

    def test_per_step_parent_and_missing_component_zero_mean_close_without_cross_rank(self):
        rows = [parent(start=0, end=50), parent("main_metadata", 50, 100),
                parent(start=1000, end=1050, round_id="hc-8"),
                parent("main_metadata", 1050, 1100, round_id="hc-8")]
        tasks = [task("_prepare_pos_seq_lens_kernel", 0, 20, 1),
                 task("SparseAttnSharedkvMetadata", 50, 70, 2),
                 task("SparseAttnSharedkvMetadata", 1050, 1090, 3)]
        report = self.diagnostics(rows, tasks)
        self.assertEqual(report["rank"], "dp9")
        self.assertEqual(report["status"], "MATCHED")
        for stage in report["stages"]:
            self.assertEqual(stage["rounds"], 2)
            self.assertEqual(stage["duration_ms_mean"], .00005)
            self.assertAlmostEqual(sum(row["duration_ms_mean"] for row in stage["summary"]), stage["duration_ms_mean"])
            self.assertAlmostEqual(sum(row["share_of_parent"] for row in stage["summary"]), 1.)
            self.assertEqual([row["round_id"] for row in stage["per_step"]], ["hc-7", "hc-8"])
            self.assertEqual([row["step_index"] for row in stage["per_step"]], [1, 2])
            self.assertTrue(all(row["closure_error_ns"] == 0 for row in stage["per_step"]))
        single = next(row for row in report["stages"][0]["summary"] if row["kind"] == "single_operator")
        self.assertEqual(single["observed_steps"], 1)
        self.assertEqual(single["duration_ms_mean"], .00001)
        self.assertEqual(single["duration_ms_p50"], .00001)
        self.assertEqual(single["duration_ms_max"], .00002)
        metadata = report["stages"][1]["per_step"]
        self.assertEqual([row["recorded_busy_ms"] for row in metadata], [.00002, .00004])
        with Path(report["segments_csv"]).open(encoding="utf-8-sig", newline="") as stream:
            segments = list(csv.DictReader(stream))
        self.assertTrue(all(row["rank"] == "dp9" for row in segments))
        self.assertIn([1], [json.loads(row["row_indices"]) for row in segments])
        self.assertTrue(Path(report["steps_csv"]).is_file())

    def test_missing_whole_parent_is_unknown_not_zero_and_preserves_original_step_gap(self):
        report = self.diagnostics([parent(start=0, end=50), parent("main_metadata", 50, 100)], [])
        self.assertEqual(report["status"], "PARTIAL")
        for stage in report["stages"]:
            self.assertEqual(stage["observed_steps"], 1)
            self.assertIsNone(stage["duration_ms_mean"])
            self.assertEqual(stage["per_step"][1]["status"], "UNAVAILABLE")
            self.assertIn("missing " + stage["stage_key"] + " parent interval", stage["per_step"][1]["reason"])
            self.assertIsNone(stage["per_step"][0]["reason"])
            self.assertIsNone(stage["per_step"][1]["duration_ms"])
            self.assertIsNone(stage["summary"][0]["duration_ms_mean"])
        for mode in ("multiple", "nonmatched"):
            rows = [parent(start=0, end=50), parent("main_metadata", 50, 100)]
            if mode == "multiple":
                rows.append(copy.deepcopy(rows[0]))
            else:
                rows[0].update(status="UNRESOLVED", reason="missing preparation anchor")
            stage = self.diagnostics(rows, [])["stages"][0]
            with self.subTest(mode=mode):
                sample = stage["per_step"][0]
                self.assertEqual(sample["status"], "UNAVAILABLE")
                self.assertIsNone(sample["duration_ms"])
                if mode == "multiple":
                    self.assertIn("ambiguous main_inputs parent intervals: observed 2", sample["reason"])
                else:
                    self.assertIn("parent status=UNRESOLVED", sample["reason"])
                    self.assertIn("missing preparation anchor", sample["reason"])

    def test_capture_integration_adds_focus_without_changing_15_parent_or_level1_results(self):
        probe = legacy.ResidualStageTests()
        probe.directory = self.directory
        capture, measured = probe.matched_capture()
        expected = []
        index = M.ATimeIndex(capture["tasks"])
        for window in M.residual_windows(measured):
            expected.extend(M.residual_stage_plan(window, index.overlapping(int(window["start_ns"]), int(window["end_ns"])),
                                                  {"draft_tokens": 7, "draft_layers": 3}))
        before = copy.deepcopy(measured)
        report = M.residual_analyze_capture(capture, measured, self.directory / "out",
                                            case={"draft_tokens": 7, "draft_layers": 3})
        self.assertEqual(measured, before)
        self.assertEqual(len(report["stage_summary"]), 15)
        expected_parents = {row["stage_key"]: row["duration_ms"] for row in expected if row["level"] == 0}
        self.assertEqual({row["stage_key"]: row["duration_ms_mean"] for row in report["stage_summary"]}, expected_parents)
        expected_children = {row["stage_key"]: row["duration_ms"] for row in expected if row["level"] == 1}
        self.assertEqual({row["stage_key"]: row["duration_ms_mean"] for row in report["substage_summary"]}, expected_children)
        self.assertEqual(report["focus_diagnostics"]["status"], "MATCHED")
        for stage in report["focus_diagnostics"]["stages"]:
            self.assertAlmostEqual(stage["duration_ms_mean"], expected_parents[stage["stage_key"]])
            self.assertAlmostEqual(sum(row["duration_ms_mean"] for row in stage["summary"]), stage["duration_ms_mean"])


if __name__ == "__main__":
    unittest.main()
