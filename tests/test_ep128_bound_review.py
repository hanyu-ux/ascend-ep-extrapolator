"""Independent adversarial checks for offline operator-bound evidence."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import profile_ep128 as core
import build_ep128_leadership_excel as renderer
from tests.test_ep128_roofline import fixture, explicit_profile, refreshed_kernel


class BoundIndependentReviewTests(unittest.TestCase):
    def test_primary_table_reports_real_competitor_pipeline_counts(self):
        report = core.refresh_report_operator_bounds(fixture())
        rows, _ = renderer._operator_pmu_data([refreshed_kernel(report)])
        self.assertIn("AIC流水=5/5", rows[0][-2])
        self.assertNotIn("AIC流水=0/0", rows[0][-2])

    def test_declared_source_rank_is_checked_even_when_no_operator_is_configured(self):
        profile = explicit_profile()
        profile["cases"]["case-a"].update(source_rank="dp999", operators={})
        with self.assertRaisesRegex(ValueError, "source_rank"):
            core.refresh_report_operator_bounds(fixture(), profile)

    def test_whitespace_does_not_satisfy_hardware_or_workload_provenance(self):
        for location in ("hardware", "operator"):
            with self.subTest(location=location):
                profile = explicit_profile()
                case = profile["cases"]["case-a"]
                destination = case["hardware"] if location == "hardware" else case["operators"]["op_exact"]
                destination["provenance"] = " \t\r\n "
                try:
                    updated = core.refresh_report_operator_bounds(fixture(), profile)
                except ValueError:
                    continue
                model = refreshed_kernel(updated)["pmu"]["roofline"]
                self.assertEqual(model["status"], "unavailable")
                self.assertTrue(model["missing"])

    def test_finite_but_overflowing_native_duration_never_serializes_infinity(self):
        report = fixture()
        for method in report["cases"][0]["projection"]["layer_mean_methods"]:
            kernel = next(core._bound_method_kernels(method))
            kernel["pmu"]["coverage"]["total_duration_ms"] = 1e308
        updated = core.refresh_report_operator_bounds(report, explicit_profile())
        self.assertIn(refreshed_kernel(updated)["pmu"]["roofline"]["status"], ("unavailable", "inconsistent"))
        json.dumps(updated, allow_nan=False)

    def test_unrepresentably_large_json_integer_is_a_validation_error(self):
        for destination in ("bandwidth", "flops", "peak"):
            with self.subTest(destination=destination):
                profile = explicit_profile()
                case = profile["cases"]["case-a"]
                if destination == "bandwidth":
                    case["hardware"]["memory_bandwidth_gbps"] = 10 ** 400
                elif destination == "peak":
                    case["hardware"]["peak_tflops_by_dtype"]["FP16"] = 10 ** 400
                else:
                    case["operators"]["op_exact"]["flops_per_call"] = 10 ** 400
                with self.assertRaises(ValueError):
                    core.validate_bound_profile(profile)

    def test_all_offline_output_artifacts_protect_the_original_report_and_profile(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for collision in ("workbook", "html", "template", "bound", "profile"):
                with self.subTest(collision=collision):
                    output = directory / (collision + ".xlsx")
                    report_path = directory / (collision + ".input.json")
                    profile_path = directory / (collision + ".profile.json")
                    kwargs = {"refresh_bound": True, "bound_profile": profile_path}
                    if collision == "workbook":
                        report_path = output
                    elif collision == "html":
                        report_path = output.with_suffix(".html")
                    elif collision == "template":
                        kwargs["bound_template"] = report_path
                    elif collision == "bound":
                        report_path = output.with_suffix(".bound.json")
                    else:
                        profile_path = output.with_suffix(".bound.json")
                        kwargs["bound_profile"] = profile_path
                    report_path.write_text("ORIGINAL_REPORT", encoding="utf-8")
                    profile_path.write_text("ORIGINAL_PROFILE", encoding="utf-8")
                    before = (report_path.read_bytes(), profile_path.read_bytes())
                    with self.assertRaisesRegex(ValueError, "不能覆盖"):
                        renderer.build_workbook(output, report=report_path, report_data={}, **kwargs)
                    self.assertEqual((report_path.read_bytes(), profile_path.read_bytes()), before)

    def test_refresh_preserves_non_pmu_values_and_separate_core_memory_scopes(self):
        report = fixture()
        for method in report["cases"][0]["projection"]["layer_mean_methods"]:
            kernel = next(core._bound_method_kernels(method))
            kernel["pmu"]["metrics"].update({
                "aic_read_main_memory_datas_kb": {"value": 100., "unit": "KB"},
                "aiv_read_main_memory_datas_kb": {"value": 200., "unit": "KB"}})
        before = copy.deepcopy(report)
        updated = core.refresh_report_operator_bounds(report)
        self.assertEqual(report, before)
        kernel = refreshed_kernel(updated)
        self.assertEqual(kernel["pmu"]["metrics"]["aic_read_main_memory_datas_kb"]["value"], 100.)
        self.assertEqual(kernel["pmu"]["metrics"]["aiv_read_main_memory_datas_kb"]["value"], 200.)
        self.assertEqual(kernel["pmu"]["roofline"]["status"], "unavailable")
        self.assertNotIn("memory_bytes_per_call", kernel["pmu"]["roofline"])
        self.assertEqual(updated["cases"][0]["projection"]["layer_mean_methods"][0]["cycle_ms"],
                         before["cases"][0]["projection"]["layer_mean_methods"][0]["cycle_ms"])


if __name__ == "__main__":
    unittest.main()
