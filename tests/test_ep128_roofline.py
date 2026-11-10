"""Explicit analytic bounds must not turn pipeline activity into hardware MFU."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import profile_ep128 as core
import build_ep128_leadership_excel as renderer
from tests.test_ep128_operator_pmu import summary, task


def fixture():
    raw = {"aic_mac_ratio": .975, "aic_scalar_ratio": .1, "aic_mte1_ratio": .2,
           "aic_mte2_ratio": .3, "aic_fixpipe_ratio": .15, "aicore_time_us": 98.}
    pmu = summary([task(1, 100, raw), task(2, 100, raw)])
    kernel = {"key": "op_exact", "name": "GroupedMatmul", "op_type": "GroupedMatmul",
              "task_type": "AI_CORE", "input_shapes": "57344,7168;14,224,512,16,32",
              "pmu": pmu, "duration_mean_ms": .001, "kernel_sum_ms": .002}
    methods = [{"id": name, "selected_rank": "dp1", "cycle_ms": 123., "tpot_ms": 24.6,
                "per_device_tokens_s": 42., "target_stage_breakdowns": [{"attention": attention,
                 "stage_rows": [{"operator_breakdown": {"kernels": [copy.deepcopy(kernel)],
                                                           "rows": [{"source_ms": 1.5}]}}]}]}
               for name, attention in (("method1", "CSA"), ("method2", "HCA"))]
    return {"cases": [{"name": "case-a", "projection": {"layer_mean_methods": methods}}]}


def explicit_profile():
    return {"schema_version": 1, "cases": {"case-a": {"source_rank": "dp1", "hardware": {
        "name": "SYNTHETIC ONLY", "architecture": "synthetic", "scope": "logical_npu",
        "peak_tflops_by_dtype": {"FP16": 100.}, "memory_bandwidth_gbps": 1000., "memory_level": "HBM",
        "provenance": "synthetic example; no production hardware constants"}, "operators": {"op_exact": {
            "compute_dtype": "FP16", "flops_per_call": 1e9, "memory_bytes_per_call": 1e6,
            "memory_bytes_basis": "measured", "memory_level": "HBM", "workload_basis": "constant_per_call",
            "provenance": "synthetic fixed-work evidence"}}}}}


def refreshed_kernel(report):
    return report["cases"][0]["projection"]["layer_mean_methods"][0]["target_stage_breakdowns"][0]["stage_rows"][0]["operator_breakdown"]["kernels"][0]


class RooflineTests(unittest.TestCase):
    def test_missing_hardware_is_unknown_even_for_high_mac_and_packed_gmm(self):
        report = core.refresh_report_operator_bounds(fixture())
        pmu = refreshed_kernel(report)["pmu"]
        self.assertEqual(pmu["roofline"]["status"], "unavailable")
        self.assertNotIn("flops_per_call", pmu["roofline"])
        self.assertFalse(pmu["bound"]["bound_verified"])
        self.assertFalse(pmu["roofline"]["bound_verified"])
        self.assertTrue(any("有效" in item for item in pmu["roofline"]["missing"]))

    def test_quantitative_units_use_unique_whole_calls_not_clipped_hotspot_mean(self):
        report = core.refresh_report_operator_bounds(fixture(), explicit_profile())
        model = refreshed_kernel(report)["pmu"]["roofline"]
        self.assertEqual(model["status"], "modeled")
        self.assertEqual(model["duration_mean_us"], 100.)  # clipped duration_mean_ms is deliberately 1us
        self.assertEqual(model["compute_lower_bound_us"], 10.)
        self.assertEqual(model["memory_lower_bound_us"], 1.)
        self.assertEqual(model["achieved_tflops"], 10.)
        self.assertEqual(model["achieved_bandwidth_gbps"], 10.)
        self.assertEqual(model["arithmetic_intensity"], 1000.)
        self.assertEqual(model["ridge_point_flops_per_byte"], 100.)
        self.assertAlmostEqual(model["compute_peak_fraction"], .1)
        self.assertAlmostEqual(model["model_efficiency"], .1)
        self.assertEqual(model["direction"], "compute")
        self.assertFalse(model["bound_verified"])

    def test_memory_and_balanced_directions_are_separate_model_conclusions(self):
        profile = explicit_profile()
        op = profile["cases"]["case-a"]["operators"]["op_exact"]
        for traffic, direction in ((2e7, "memory"), (1e7, "balanced")):
            op["memory_bytes_per_call"] = traffic
            result = core.refresh_report_operator_bounds(fixture(), profile)
            self.assertEqual(refreshed_kernel(result)["pmu"]["roofline"]["direction"], direction)

    def test_modeled_bytes_never_report_measured_bandwidth_saturation(self):
        profile = explicit_profile()
        profile["cases"]["case-a"]["operators"]["op_exact"]["memory_bytes_basis"] = "model"
        model = refreshed_kernel(core.refresh_report_operator_bounds(fixture(), profile))["pmu"]["roofline"]
        self.assertEqual(model["confidence"], "低")
        self.assertTrue(any("不是实测" in item for item in model["caveats"]))

    def test_memory_traffic_and_peak_must_explicitly_name_the_same_level(self):
        profile = explicit_profile()
        op = profile["cases"]["case-a"]["operators"]["op_exact"]
        for level, status in ((None, "unavailable"), ("L1", "inconsistent"), ("hbm", "modeled")):
            op["memory_level"] = level
            result = core.refresh_report_operator_bounds(fixture(), profile)
            self.assertEqual(refreshed_kernel(result)["pmu"]["roofline"]["status"], status)

    def test_optional_aic_mte3_does_not_inflate_required_pipeline_count(self):
        report = fixture()
        for method in report["cases"][0]["projection"]["layer_mean_methods"]:
            for kernel in core._bound_method_kernels(method):
                kernel["pmu"]["metrics"]["aic_mte3_ratio"] = {
                    "value": .1, "min": .1, "max": .1, "duration_coverage": 1., "unit": "ratio"}
        kernel = refreshed_kernel(core.refresh_report_operator_bounds(report))
        rows, _ = renderer._operator_pmu_data([kernel])
        self.assertIn("AIC流水=5/5", rows[0][-2])
        self.assertNotIn("6/5", rows[0][-2])

    def test_dtype_specific_peak_missing_does_not_apply_int8_or_fp8_multiplier(self):
        profile = explicit_profile()
        profile["cases"]["case-a"]["operators"]["op_exact"]["compute_dtype"] = "FP8"
        model = refreshed_kernel(core.refresh_report_operator_bounds(fixture(), profile))["pmu"]["roofline"]
        self.assertEqual(model["status"], "unavailable")
        self.assertNotIn("peak_tflops", model)

    def test_missing_fixed_active_work_does_not_treat_shape_capacity_as_flops(self):
        profile = explicit_profile()
        profile["cases"]["case-a"]["operators"]["op_exact"].pop("workload_basis")
        model = refreshed_kernel(core.refresh_report_operator_bounds(fixture(), profile))["pmu"]["roofline"]
        self.assertEqual(model["status"], "unavailable")

    def test_impossible_efficiency_is_flagged_not_clamped_or_accepted(self):
        profile = explicit_profile()
        profile["cases"]["case-a"]["operators"]["op_exact"]["flops_per_call"] = 2e10
        model = refreshed_kernel(core.refresh_report_operator_bounds(fixture(), profile))["pmu"]["roofline"]
        self.assertEqual(model["status"], "inconsistent")
        self.assertEqual(model["model_efficiency"], 2.)
        self.assertEqual(model["confidence"], "未判断")

    def test_invalid_hardware_and_unit_scope_values_fail_early(self):
        for field, invalid in (("scope", "physical_card"), ("memory_bandwidth_gbps", 0),
                               ("memory_bandwidth_gbps", float("nan")), ("memory_bandwidth_gbps", True)):
            with self.subTest(field=field, invalid=invalid):
                profile = explicit_profile()
                profile["cases"]["case-a"]["hardware"][field] = invalid
                with self.assertRaises(ValueError):
                    core.validate_bound_profile(profile)
        profile = explicit_profile()
        profile["cases"]["case-a"]["hardware"]["peak_tflops_by_dtype"]["fp16"] = 200.
        with self.assertRaisesRegex(ValueError, "duplicate"):
            core.validate_bound_profile(profile)

    def test_case_rank_and_exact_fingerprint_are_checked(self):
        for change in ("case", "rank", "operator"):
            with self.subTest(change=change):
                profile = explicit_profile()
                if change == "case":
                    profile["cases"]["not-this-case"] = profile["cases"].pop("case-a")
                elif change == "rank":
                    profile["cases"]["case-a"]["source_rank"] = "dp0"
                else:
                    ops = profile["cases"]["case-a"]["operators"]
                    ops["op_other"] = ops.pop("op_exact")
                with self.assertRaises(ValueError):
                    core.refresh_report_operator_bounds(fixture(), profile)

    def test_refresh_changes_only_pmu_does_not_mutate_or_read_csv_and_is_idempotent(self):
        report, profile = fixture(), explicit_profile()
        before, original_profile = copy.deepcopy(report), copy.deepcopy(profile)
        with mock.patch.object(core, "read_capture", side_effect=AssertionError("CSV reread forbidden")):
            result = core.refresh_report_operator_bounds(report, profile)
        self.assertEqual(report, before)
        self.assertEqual(profile, original_profile)
        def strip(value):
            if isinstance(value, dict):
                return {key: strip(item) for key, item in value.items() if key not in ("pmu", "operator_bound_analysis")}
            return [strip(item) for item in value] if isinstance(value, list) else value
        self.assertEqual(strip(result), strip(before))
        self.assertEqual(core.refresh_report_operator_bounds(result), result)

    def test_old_json_without_individual_samples_cannot_recover_short_or_joint_checks(self):
        report = fixture()
        for method in report["cases"][0]["projection"]["layer_mean_methods"]:
            refreshed_kernel({"cases": [{"projection": {"layer_mean_methods": [method]}}]})["pmu"].pop("sample_diagnostics", None)
        bound = refreshed_kernel(core.refresh_report_operator_bounds(report))["pmu"]["bound"]
        self.assertEqual(bound["confidence"], "低")
        self.assertTrue(any("逐调用" in item or "联合" in item for item in bound.get("caveats", []) + bound["evidence"]))

    def test_communication_and_aicpu_never_receive_ai_core_roofline(self):
        pmu = refreshed_kernel(fixture())["pmu"]
        for member in ({"name": "AllToAllV"}, {"record_kind": "synchronization_control"}, {"task_type": "AI_CPU"}):
            self.assertEqual(core._lm_operator_roofline(member, pmu)["status"], "not_applicable")

    def test_template_leaves_all_hardware_and_work_values_unknown(self):
        report = fixture()
        profile = core.operator_bound_profile_template(report)
        self.assertEqual(profile, core.validate_bound_profile(profile))
        hw = profile["cases"]["case-a"]["hardware"]
        self.assertIsNone(hw["name"])
        self.assertEqual(hw["peak_tflops_by_dtype"], {})
        self.assertIsNone(profile["cases"]["case-a"]["operators"]["op_exact"]["flops_per_call"])
        model = refreshed_kernel(core.refresh_report_operator_bounds(report, profile))["pmu"]["roofline"]
        self.assertEqual(model["status"], "unavailable")

    def test_templates_reject_legacy_different_method_ranks(self):
        report = fixture()
        report["cases"][0]["projection"]["layer_mean_methods"][1]["selected_rank"] = "dp0"
        with self.assertRaisesRegex(ValueError, "shared source rank"):
            core.operator_bound_profile_template(report)

    def test_main_tables_include_mte1_fixpipe_and_model_not_just_advanced_details(self):
        report = core.refresh_report_operator_bounds(fixture(), explicit_profile())
        kernel = refreshed_kernel(report)
        rows, _ = renderer._operator_pmu_data([kernel])
        self.assertEqual(len(rows[0]), len(renderer.PMU_HEADERS))
        self.assertIn("aic_mte1_ratio", dict(renderer.PMU_METRICS))
        self.assertIn("aic_fixpipe_ratio", dict(renderer.PMU_METRICS))
        model_rows, details = renderer._operator_roofline_data([kernel])
        self.assertEqual(model_rows[0][1], "100")
        self.assertIn("compute_peak_fraction=10.00%", details[0][1])

    def test_overflowing_or_underflowing_model_is_not_serialized_as_nonfinite(self):
        profile = explicit_profile()
        profile["cases"]["case-a"]["hardware"]["peak_tflops_by_dtype"]["FP16"] = 1e-300
        profile["cases"]["case-a"]["operators"]["op_exact"]["flops_per_call"] = 1e308
        result = core.refresh_report_operator_bounds(fixture(), profile)
        self.assertEqual(refreshed_kernel(result)["pmu"]["roofline"]["status"], "inconsistent")
        json.dumps(result, allow_nan=False)

    def test_finite_counter_weighting_avoids_intermediate_overflow_and_unknown_sum(self):
        pmu = summary([task(1, 100, {"aic_cube_fops": 1e308}),
                       task(2, 100, {"aic_cube_fops": 1e308})])
        field = pmu["metrics"]["aic_cube_fops"]
        self.assertEqual(field["value"], 1e308)
        self.assertEqual(field["sample_mean"], 1e308)
        self.assertIsNone(field["sample_sum"])
        self.assertEqual(pmu["coverage"]["duration_coverage"], 1.)
        json.dumps(pmu, allow_nan=False)

    def test_amount_details_distinguish_arithmetic_call_mean_from_time_weighted_value(self):
        kernel = refreshed_kernel(fixture())
        kernel["pmu"] = summary([task(1, 100, {"gm_to_l1_datas_kb": 10.}),
                                 task(2, 300, {"gm_to_l1_datas_kb": 30.})])
        field = kernel["pmu"]["metrics"]["gm_to_l1_datas_kb"]
        self.assertEqual(field["value"], 25.)
        self.assertEqual(field["sample_mean"], 20.)
        _, details = renderer._operator_pmu_data([kernel])
        self.assertIn("算术均值=20.0", details[0][1])
        self.assertIn("导出行求和=40.0", details[0][1])
        self.assertIn("不自动等于全设备总量", details[0][1])


if __name__ == "__main__":
    unittest.main()
