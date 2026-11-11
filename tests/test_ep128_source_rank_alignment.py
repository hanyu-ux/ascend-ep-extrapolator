"""Offline old-report rank alignment is pure and never fabricates PMU evidence."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import profile_ep128 as core
import test_ep128_layer_mean_excel as excel_tests
from test_ep128_layer_mean import attach
from test_ep128_target_stages import native_stage_report

HERE = Path(__file__).resolve().parents[1]
renderer = excel_tests.renderer


def legacy_report():
    """Valid native scalar estimates with old method-specific selected ranks."""
    report, native = native_stage_report({"dp0": [1., 3.], "dp1": [2.2, 2.2]})
    report["generated_at"] = "2026-11-10T00:00:00+00:00"
    report["source_kind"] = "offline_kernel_csv"
    report["contract"] = {"projection": "legacy per-method maximum target rank mean"}
    report["communication_reference"] = {
        "rows": core.read_reference(HERE / "examples/communication-reference.synthetic.csv")}
    single = copy.deepcopy(report)
    case = single["cases"][0]
    case["captures"] = [item for item in case["captures"] if item["rank"] == "dp0"]
    case["measurements"] = [item for item in case["measurements"] if item["rank"] == "dp0"]
    attach(single, {"dp0": native["dp0"]})
    attach(report, native)
    case = report["cases"][0]
    # This fixture represents a saved report from before method3 existed.
    case["projection"]["layer_mean_methods"] = case["projection"]["layer_mean_methods"][:2]
    original = single["cases"][0]["projection"]["layer_mean_methods"][0]
    full = case["projection"]["layer_mean_methods"][0]
    old_first = dict(original, per_rank=full["per_rank"], detail_rows=full["detail_rows"],
                     compute_evidence=full["compute_evidence"], coverage=full["coverage"])
    case["projection"]["layer_mean_methods"][0] = old_first
    case["projection"].pop("rank_selection", None)
    for method in case["projection"]["layer_mean_methods"]:
        method.pop("rank_selection", None)
        method.pop("source_rank", None)
        method["aggregation"] = "max_of_per_rank_mean_of_per_round_layer_type_mean_estimates"
        method["focus_selected_rank"] = "dp0"
    for measured in case["measurements"]:
        residual = 18. * sum(item["duration_ms"] / 1245. for item in measured["rounds"]) / len(measured["rounds"])
        parents, summaries = [], []
        for window, count in (("main_to_draft", 8), ("draft_to_nextmain", 7)):
            summaries.append({"window_key": window, "rounds": 2, "duration_ms_mean": residual * count / 15})
            for number in range(1, count + 1):
                parents.append({"window_key": window, "stage_key": window + "_" + str(number),
                    "stage_order": number, "label": "SOURCE_" + measured["rank"] + "_U_" + str(number),
                    "level": 0, "parent_stage_key": None, "duration_ms_mean": residual / 15,
                    "rounds": 2, "matched_windows": 2, "status": "MATCHED"})
        measured["step_residuals"] = {"rank": measured["rank"], "status": "DEVICE_ONLY",
            "trace": {"status": "NOT_FOUND"}, "residual_ms_mean": residual,
            "summary": summaries, "stage_summary": parents, "substage_summary": []}
    source_rows = [{"rank": item["rank"], "cycle_ms_mean": sum(row["duration_ms"] for row in item["rounds"]) / 2,
                    "rounds": 2} for item in case["measurements"]]
    source = next(item for item in source_rows if item["rank"] == "dp1")
    config = case["config"]
    rate = 1000 * config["bs"] * config["effective_tokens_per_round"] / source["cycle_ms_mean"]
    case["source"] = {"status": "SOURCE_CONDITIONAL_RATE", "per_rank": source_rows, "class_stats": [],
        "aggregate": {"selected_rank": "dp1", "cycle_ms": source["cycle_ms_mean"],
                      "per_device_tokens_s": rate, "rank_coverage_complete": False}}
    return report


class SourceRankAlignmentTests(unittest.TestCase):
    def test_offline_alignment_uses_own_stored_rank_rows_and_does_not_mutate_or_read_csv(self):
        report = legacy_report()
        before = copy.deepcopy(report)
        with mock.patch.object(core, "read_capture", side_effect=AssertionError("no native CSV reread")):
            aligned = core.align_report_source_rank(report)
        self.assertEqual(report, before)
        self.assertIn("retained source full-cycle mean before extrapolation", aligned["contract"]["projection"])
        self.assertEqual(aligned["contract"]["source_rank_alignment_previous_projection"],
                         before["contract"]["projection"])
        case = aligned["cases"][0]
        self.assertEqual(case["projection"]["rank_selection"]["selected_rank"], "dp1")
        for original, method in zip(before["cases"][0]["projection"]["layer_mean_methods"],
                                    case["projection"]["layer_mean_methods"]):
            expected = next(item for item in original["per_rank"] if item["rank"] == "dp1")
            self.assertEqual(method["selected_rank"], "dp1")
            self.assertEqual(method["source_rank"], "dp1")
            self.assertNotIn("focus_selected_rank", method)
            for field in ("cycle_ms", "per_device_tokens_s", "global_tokens_s", "tpot_ms",
                          "class_means", "role_details", "source_partition", "breakdown"):
                self.assertEqual(method[field], expected[field])
            self.assertEqual(method["per_rank"], original["per_rank"])
        self.assertNotEqual(case["projection"]["layer_mean_methods"][0]["cycle_ms"],
                            before["cases"][0]["projection"]["layer_mean_methods"][0]["cycle_ms"])
        self.assertEqual(core.align_report_source_rank(aligned), aligned)

    def test_other_method_operator_evidence_is_not_borrowed_and_same_method_rank_is_retained(self):
        report = legacy_report()
        original = report["cases"][0]["projection"]["layer_mean_methods"]
        self.assertEqual([method["selected_rank"] for method in original], ["dp0", "dp1"])
        aligned = core.align_report_source_rank(report)["cases"][0]["projection"]["layer_mean_methods"]
        for group in aligned[0]["target_stage_breakdowns"]:
            self.assertTrue(all("operator_breakdown" not in stage for stage in group["stage_rows"]))
        # Method 2 had rich same-rank data. Keep its PMU/operators; their model
        # deltas must never leak into method 1's source-rank row.
        self.assertEqual(aligned[1]["target_stage_breakdowns"], original[1]["target_stage_breakdowns"])
        self.assertTrue(any("operator_breakdown" in stage for group in aligned[1]["target_stage_breakdowns"]
                            for stage in group["stage_rows"]))

    def test_unavailable_method_stays_unavailable_and_drops_old_selected_diagnostics(self):
        report = legacy_report()
        method = report["cases"][0]["projection"]["layer_mean_methods"][1]
        method.update(status="UNAVAILABLE", reasons=["SYNTHETIC_COMMUNICATION_MISSING"], source_cycle_ms=12345.)
        aligned = core.align_report_source_rank(report)["cases"][0]["projection"]["layer_mean_methods"][1]
        self.assertEqual(aligned["status"], "UNAVAILABLE")
        self.assertEqual(aligned["reasons"], ["SYNTHETIC_COMMUNICATION_MISSING"])
        self.assertIsNone(aligned["selected_rank"])
        self.assertEqual(aligned["source_rank"], "dp1")
        self.assertEqual(aligned["source_cycle_ms"], aligned["rank_selection"]["selected_source_cycle_ms"])
        self.assertEqual(aligned["target_stage_breakdowns"], [])
        self.assertIsNone(aligned["cycle_ms"])
        self.assertNotIn("focus_selected_rank", aligned)

    def test_renderer_rejects_conflicting_shared_rank_declarations(self):
        for corrupted in ("method_selected", "source_aggregate", "method_policy"):
            report = core.align_report_source_rank(legacy_report())
            case = report["cases"][0]
            method = case["projection"]["layer_mean_methods"][0]
            if corrupted == "method_selected":
                method["selected_rank"] = "dp0"
            elif corrupted == "source_aggregate":
                case["source"]["aggregate"]["selected_rank"] = "dp0"
            else:
                method["rank_selection"] = dict(method["rank_selection"], selected_rank="dp0")
            with self.subTest(corrupted=corrupted), self.assertRaisesRegex(ValueError, "共[同享]"):
                renderer._validate_report(report)

    def test_missing_rank_or_incomplete_row_raises_without_falling_back_or_changing_input(self):
        for missing in ("rank", "headline"):
            report = legacy_report()
            method = report["cases"][0]["projection"]["layer_mean_methods"][0]
            if missing == "rank":
                method["per_rank"] = [item for item in method["per_rank"] if item["rank"] != "dp1"]
            else:
                next(item for item in method["per_rank"] if item["rank"] == "dp1").pop("tpot_ms")
            before = copy.deepcopy(report)
            with self.subTest(missing=missing), self.assertRaisesRegex(ValueError, "cannot align stored source rank"):
                core.align_report_source_rank(report)
            self.assertEqual(report, before)

    def test_mismatched_parent_stages_do_not_reuse_old_operator_breakdowns(self):
        report = legacy_report()
        method = report["cases"][0]["projection"]["layer_mean_methods"][1]
        selected = next(item for item in method["per_rank"] if item["rank"] == "dp1")
        selected["target_stage_breakdowns"][0]["stage_rows"][0]["delta_ms"] += .01
        aligned = core.align_report_source_rank(report)["cases"][0]["projection"]["layer_mean_methods"][1]
        self.assertNotIn("operator_breakdown", aligned["target_stage_breakdowns"][0]["stage_rows"][0])

    def test_renderer_flag_writes_new_sidecar_uses_common_u_and_preserves_legacy_default(self):
        report = legacy_report()
        before = copy.deepcopy(report)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source = directory / "original.json"
            source.write_text(json.dumps(report))
            source_bytes = source.read_bytes()
            aligned_out, stored_out = directory / "aligned.xlsx", directory / "stored.xlsx"
            with mock.patch.object(core, "read_capture", side_effect=AssertionError("no CSV reread")):
                renderer.build_workbook(aligned_out, report=source, report_data=report,
                                        expected_method_set="layer_mean_methods", align_source_rank=True)
                renderer.build_workbook(stored_out, report=source, report_data=report,
                                        expected_method_set="layer_mean_methods")
            self.assertEqual(report, before)
            self.assertEqual(source.read_bytes(), source_bytes)
            self.assertFalse(stored_out.with_suffix(".source_rank.json").exists())
            derived = json.loads(aligned_out.with_suffix(".source_rank.json").read_text())
            methods = derived["cases"][0]["projection"]["layer_mean_methods"]
            self.assertEqual([method["selected_rank"] for method in methods], ["dp1", "dp1"])
            self.assertEqual(methods[0]["target_stage_breakdowns"],
                             next(item for item in before["cases"][0]["projection"]["layer_mean_methods"][0]["per_rank"]
                                  if item["rank"] == "dp1")["target_stage_breakdowns"])
            book, _ = excel_tests.read_book(aligned_out)
            text = " ".join(map(str, book["方法1估算"][1].values()))
            self.assertIn("SOURCE_dp1_U_", text)
            self.assertNotIn("SOURCE_dp0_U_", text)
            self.assertIn("缺少operator_breakdown", text)
            preview = aligned_out.with_suffix(".html").read_text()
            self.assertIn("共同", preview)
            legacy, _ = excel_tests.read_book(stored_out)
            self.assertIn(before["cases"][0]["projection"]["layer_mean_methods"][0]["cycle_ms"],
                          legacy["方法1估算"][1].values())
            self.assertIn("旧", " ".join(map(str, legacy["复核记录"][1].values())))

    def test_renderer_cli_alignment_is_offline_and_produces_sidecar(self):
        report = legacy_report()
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source = directory / "legacy.json"
            source.write_text(json.dumps(report))
            before = source.read_bytes()
            output = directory / "aligned.xlsx"
            result = subprocess.run([sys.executable, "-B", "-S", str(HERE / "build_ep128_leadership_excel.py"),
                                     "--report", str(source), "--out", str(output), "--align-source-rank"],
                                    text=True, capture_output=True, timeout=45)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn(".source_rank.json", result.stdout)
            self.assertTrue(output.is_file())
            self.assertTrue(output.with_suffix(".html").is_file())
            self.assertTrue(output.with_suffix(".source_rank.json").is_file())
            self.assertEqual(source.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
