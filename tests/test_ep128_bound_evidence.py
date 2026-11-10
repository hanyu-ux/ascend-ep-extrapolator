"""Pipeline screening separates activity evidence from a verified NPU bound."""

import json
import unittest
from unittest.mock import patch

import profile_ep128 as core
from tests.test_ep128_operator_pmu import summary, task


def aic(**updates):
    return {"aic_mac_ratio": .98, "aic_scalar_ratio": .1, "aic_mte1_ratio": .2,
            "aic_mte2_ratio": .3, "aic_fixpipe_ratio": .1, **updates}


def aiv(**updates):
    return {"aiv_vec_ratio": .98, "aiv_scalar_ratio": .1,
            "aiv_mte2_ratio": .3, "aiv_mte3_ratio": .1, **updates}


class BoundEvidenceTests(unittest.TestCase):
    def test_complete_aic_competitors_do_not_require_mte3_or_claim_verified_bound(self):
        result = summary([task(1, 100, aic(aicore_time_us=99))])
        bound = result["bound"]
        self.assertEqual(bound["code"], "aic_compute")
        self.assertEqual(bound["confidence"], "中")
        self.assertEqual(bound["evidence_level"], "pipeline_screening")
        self.assertFalse(bound["bound_verified"])
        self.assertIn("未验证", bound["label"])
        self.assertEqual(bound["missing_metrics"], [])
        self.assertEqual(set(bound["required_metrics"]), set(aic()))
        self.assertNotIn("aic_mte3_ratio", bound["required_metrics"])
        self.assertTrue(bound["pipeline_coverage"]["aic"]["complete"])
        self.assertEqual(bound["pipeline_coverage"]["aic"]["joint_duration_coverage"], 1.)
        self.assertTrue(result["sample_diagnostics"]["per_call_checks_observed"])

    def test_mte1_fixpipe_and_gm_are_distinct_transfer_candidates(self):
        cases = (("aic_mte1_ratio", "片内供数"), ("aic_fixpipe_ratio", "结果搬运"),
                 ("aic_mte2_ratio", "GM读入搬运"), ("aic_mte3_ratio", "通路依架构"))
        for metric, phrase in cases:
            with self.subTest(metric=metric):
                result = summary([task(1, 100, aic(aic_mac_ratio=.3, **{metric: .96}))])
                self.assertEqual(result["bound"]["code"], "aic_transfer")
                self.assertIn(phrase, result["bound"]["label"])
                self.assertFalse(result["bound"]["bound_verified"])
                self.assertTrue(any("拓扑" in note for note in result["bound"]["caveats"]))
                if metric == "aic_mte2_ratio":
                    self.assertIn("非HBM带宽结论", result["bound"]["label"])
                if metric == "aic_fixpipe_ratio":
                    self.assertNotIn("片内搬运", result["bound"]["label"])

    def test_missing_mte1_or_fixpipe_limits_high_mac_to_busy_screening(self):
        for missing in ("aic_mte1_ratio", "aic_fixpipe_ratio", "aic_scalar_ratio"):
            with self.subTest(missing=missing):
                metrics = aic()
                del metrics[missing]
                bound = summary([task(1, 100, metrics)])["bound"]
                self.assertEqual(bound["code"], "aic_busy_compute")
                self.assertEqual(bound["confidence"], "低")
                self.assertEqual(bound["missing_metrics"], [missing])
                self.assertIsNone(bound["pipeline_coverage"]["aic"]["duration_coverage"][missing])
                self.assertFalse(bound["pipeline_coverage"]["aic"]["complete"])

    def test_aiv_requires_its_own_four_pipelines(self):
        bound = summary([task(1, 100, aiv(), core_type="AI_VECTOR_CORE")])["bound"]
        self.assertEqual(bound["code"], "aiv_compute")
        self.assertEqual(bound["confidence"], "中")
        self.assertEqual(set(bound["required_metrics"]), set(aiv()))
        self.assertNotIn("aiv_mte1_ratio", bound["required_metrics"])
        self.assertNotIn("aiv_fixpipe_ratio", bound["required_metrics"])
        self.assertEqual(set(bound["pipeline_coverage"]), {"aiv"})

    def test_scalar_without_competitors_is_busy_not_proven_dominant(self):
        bound = summary([task(1, 100, {"aic_scalar_ratio": .95})])["bound"]
        self.assertEqual(bound["code"], "aic_busy_scalar")
        self.assertEqual(bound["confidence"], "低")
        self.assertFalse(bound["bound_verified"])
        self.assertEqual(set(bound["missing_metrics"]), set(aic()) - {"aic_scalar_ratio"})
        complete = summary([task(1, 100, aic(aic_mac_ratio=.2, aic_scalar_ratio=.9))])["bound"]
        self.assertEqual(complete["code"], "aic_scalar")
        self.assertEqual(complete["confidence"], "中")
        self.assertIn("非CPU", complete["label"])

    def test_each_metric_eighty_percent_does_not_prove_joint_coverage(self):
        calls = []
        for index, missing in enumerate(aic(), 1):
            metrics = aic()
            del metrics[missing]
            calls.append(task(index, 100, metrics))
        result = summary(calls)
        self.assertEqual(result["coverage"]["duration_coverage"], 1.)
        self.assertTrue(all(result["metrics"][field]["duration_coverage"] == .8 for field in aic()))
        bound = result["bound"]
        self.assertEqual(bound["missing_metrics"], [])
        self.assertEqual(bound["pipeline_coverage"]["aic"]["joint_duration_coverage"], 0.)
        self.assertFalse(bound["pipeline_coverage"]["aic"]["complete"])
        self.assertEqual(bound["confidence"], "低")
        self.assertTrue(any("联合Duration覆盖不足80%" in note for note in bound["evidence"]))

    def test_field_coverage_limit_is_not_reported_as_absent_or_zero(self):
        first, second = aic(), aic()
        del second["aic_fixpipe_ratio"]
        result = summary([task(1, 100, first), task(2, 900, second)])
        pipeline = result["bound"]["pipeline_coverage"]["aic"]
        self.assertEqual(pipeline["missing_metrics"], [])
        self.assertIn("aic_fixpipe_ratio", pipeline["low_coverage_metrics"])
        self.assertEqual(pipeline["duration_coverage"]["aic_fixpipe_ratio"], .1)
        self.assertEqual(result["metrics"]["aic_fixpipe_ratio"]["value"], .1)
        self.assertEqual(result["bound"]["confidence"], "低")

    def test_mixed_core_keeps_separate_denominators_and_required_sets(self):
        result = summary([task(1, 100, {**aic(), **aiv(aiv_vec_ratio=.2, aiv_mte2_ratio=.96)}, core_type="MIX_AIC")])
        bound = result["bound"]
        self.assertEqual(bound["code"], "mixed_core")
        self.assertEqual(set(bound["pipeline_coverage"]), {"aic", "aiv"})
        self.assertEqual(set(bound["pipeline_coverage"]["aic"]["required_metrics"]), set(aic()))
        self.assertEqual(set(bound["pipeline_coverage"]["aiv"]["required_metrics"]), set(aiv()))
        self.assertTrue(any("分母不同" in note for note in bound["evidence"]))
        self.assertFalse(bound["bound_verified"])

    def test_offline_aggregate_without_call_diagnostics_remains_low_confidence(self):
        result = summary([task(1, 100, aic(aicore_time_us=60))])
        # Aggregate time=60us cannot establish the distribution of 100us calls.
        member = {"name": "GroupedMatmul", "task_type": "AI_CORE"}
        for samples in ([], [{"core_type": "AI_CORE", "metrics": aic(aicore_time_us=60)}]):
            with self.subTest(samples=samples):
                bound = core._lm_operator_pmu_bound(member, result["metrics"], result["coverage"], samples)
                self.assertEqual(bound["code"], "aic_compute")
                self.assertEqual(bound["confidence"], "低")
                self.assertIsNone(bound["pipeline_coverage"]["aic"]["joint_duration_coverage"])
                self.assertTrue(any("真实单次调用" in note for note in bound["evidence"]))

    def test_compact_native_diagnostics_roundtrip_preserves_short_and_core_gap_facts(self):
        result = summary([task(1, 100, aic(aicore_time_us=50)), task(2, 100, aic(aicore_time_us=70))])
        diagnostic = json.loads(json.dumps(result["sample_diagnostics"]))
        self.assertEqual(diagnostic["core_time_gap_duration_share"], .5)
        self.assertEqual(diagnostic["core_time_comparable_duration_share"], 1.)
        pseudo = [{"core_type": "AI_CORE", "metrics": aic()}]
        bound = core._lm_operator_pmu_bound({"name": "GroupedMatmul", "task_type": "AI_CORE"},
            result["metrics"], result["coverage"], pseudo, sample_diagnostics=diagnostic)
        self.assertEqual(bound["code"], "task_core_time_gap")
        self.assertEqual(bound["evidence_level"], "task_core_time_discrepancy")
        self.assertEqual(bound["confidence"], "低")
        self.assertFalse(bound["bound_verified"])

    def test_mix_core_time_gap_needs_both_native_core_times(self):
        metrics = {**aic(), **aiv(aiv_vec_ratio=.2, aiv_mte2_ratio=.96), "aicore_time_us": 20}
        one = summary([task(1, 100, metrics, core_type="MIX_AIC")])
        self.assertFalse(one["sample_diagnostics"]["core_time_gap_checks_observed"])
        self.assertIsNone(one["sample_diagnostics"]["core_time_gap_duration_share"])
        self.assertEqual(one["bound"]["code"], "mixed_core")
        both = summary([task(1, 100, dict(metrics, aiv_time_us=10), core_type="MIX_AIC")])
        self.assertEqual(both["bound"]["code"], "task_core_time_gap")
        self.assertEqual(both["sample_diagnostics"]["core_time_gap_duration_share"], 1.)

    def test_short_task_and_variation_checks_survive_pipeline_completeness(self):
        short = summary([task(1, 5, aic())])
        self.assertEqual(short["sample_diagnostics"]["short_duration_share"], 1.)
        self.assertEqual(short["bound"]["confidence"], "低")
        mixed = summary([task(1, 5, aic()), task(2, 100, aic())])
        self.assertLess(mixed["sample_diagnostics"]["short_duration_share"], .5)
        self.assertEqual(mixed["bound"]["confidence"], "中")
        varied = summary([task(1, 100, aic()), task(2, 100, aic(aic_mac_ratio=.5))])
        self.assertEqual(varied["bound"]["confidence"], "低")
        self.assertTrue(any("35个百分点" in note for note in varied["bound"]["evidence"]))

    def test_communication_control_and_aicpu_are_outside_compute_transfer_bound(self):
        for name, kind, code in (("MoeDistributeDispatchV3", "AI_CORE", "communication_mixed"),
                                 ("EventWait", "AI_CORE", "synchronization_control"),
                                 ("HostService", "AI_CPU", "aicpu")):
            with self.subTest(name=name):
                bound = summary([task(1, 100, aic(), core_type=kind, name=name)])["bound"]
                self.assertEqual(bound["code"], code)
                self.assertEqual(bound["evidence_level"], "not_applicable")
                self.assertFalse(bound["bound_verified"])

    def test_flow_work_fields_offer_unweighted_sum_mean_and_scope_metadata(self):
        field = "explicit_work_test_flops"
        info = {"unit": "flops", "scope": "explicit_test_work", "description": "Synthetic work only"}
        with patch.dict(core._A_PMU_FIELDS, {field: "flops"}), patch.dict(core._A_PMU_FIELD_INFO, {field: info}):
            result = summary([task(1, 100, {field: 100}), task(2, 300, {field: 900})])
        metric = result["metrics"][field]
        self.assertEqual(metric["value"], 700.)
        self.assertEqual(metric["sample_sum"], 1000.)
        self.assertEqual(metric["sample_mean"], 500.)
        self.assertEqual(metric["sample_aggregation"], "arithmetic_over_known_whole_calls")
        self.assertEqual(metric["field_info"], info)
        self.assertIsNot(metric["field_info"], info)


if __name__ == "__main__":
    unittest.main()
