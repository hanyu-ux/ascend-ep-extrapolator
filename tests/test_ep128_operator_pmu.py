"""Native PMU interpretation must never change extrapolated timing accounts."""
import copy
import csv
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import profile_ep128 as core
from tests.test_ep128_layer_mean import attach
from tests.test_ep128_target_stages import native_stage_report


def task(index, duration_us, metrics=None, *, core_type="AI_CORE", name="GroupedMatmul", start_ns=0):
    return SimpleNamespace(row_index=index, row_indices=[index], start_ns=start_ns,
        end_ns=start_ns + round(duration_us * 1000), name=name, op_type=name,
        label=name + " " + name, stream_id="2", task_id=str(index), task_type=core_type,
        input_shapes="32,7168;14,224,512,16,32", output_shapes="32,8192",
        input_dtypes="DT_FLOAT8_E4M3FN;DT_FLOAT4_E2M1", output_dtypes="DT_BF16",
        pmu={"metrics": metrics or {}, "invalid_fields": [], "core_type": core_type})


def summary(tasks, clips=None):
    index = core.ATimeIndex(sorted(tasks, key=lambda item: item.start_ns))
    member = core._lm_operator_member(tasks[0], index)
    observations = []
    for i, item in enumerate(tasks):
        left, right = clips[i] if clips else (item.start_ns, item.end_ns)
        observations.append(core._lm_operator_pmu_observation(item, left, right, {"rank": "dp1"}))
    return core._lm_operator_pmu_summary(member, observations)


class OperatorPmuTests(unittest.TestCase):
    def read_rows(self, rows):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "kernel_details.csv"
        fields = list(dict.fromkeys(key for row in rows for key in row))
        with path.open("w", newline="", encoding="utf-8-sig") as output:
            writer = csv.DictWriter(output, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        return core.read_capture(path, "dp0")

    @staticmethod
    def row(**extra):
        return {"Name": "GroupedMatmul", "Type": "GroupedMatmul", "Start Time(us)": "1793934945917078.394",
                "Duration(us)": "100", "Task ID": "22", "Stream ID": "2", "Accelerator Core": "AI_CORE", **extra}

    def test_native_headers_percent_units_and_observed_zero(self):
        capture = self.read_rows([self.row(**{"aic_mac_ratio": ".975", "aic_mte2_ratio (%)": "77.1",
            "aic_mte3_ratio": "0", "aicore_time (μs)": "99.3", "aiv_time(ns)": "0",
            "aic_total_cycles": "150000", "Block Num": "32", "Mix Block Num": "0",
            "cube_utilization(%)": "99.2", "aic_scalar_time(µs)": "1.5", "aic_icache_miss_rate": "N/A"})])
        observed = capture["tasks"][0].pmu["metrics"]
        self.assertEqual(observed["aic_mac_ratio"], .975)
        self.assertAlmostEqual(observed["aic_mte2_ratio"], .771)
        self.assertEqual(observed["aic_mte3_ratio"], 0)
        self.assertEqual(observed["aiv_time_us"], 0)
        self.assertAlmostEqual(observed["cube_utilization"], .992)
        self.assertEqual(observed["aic_scalar_time_us"], 1.5)
        self.assertEqual(observed["block_num"], 32)
        self.assertNotIn("aic_icache_miss_rate", observed)
        self.assertEqual(capture["tasks"][0].start_ns, 1793934945917078394)

    def test_invalid_optional_values_and_alias_conflicts_do_not_damage_timing(self):
        capture = self.read_rows([self.row(**{"aic_mac_ratio": "nan", "aic_mte2_ratio": "1.2",
            "aic_mte3_ratio": "-1", "aicore_time(us)": "inf", "Block Num": "1.5",
            "aiv_vec_ratio": "60%", "aiv_vec_ratio (%)": "50"})])
        item = capture["tasks"][0]
        self.assertEqual(item.end_ns - item.start_ns, 100000)
        self.assertEqual(item.pmu["metrics"], {})
        self.assertEqual(set(item.pmu["invalid_fields"]), {"aic_mac_ratio", "aic_mte2_ratio", "aic_mte3_ratio",
            "aicore_time_us", "block_num", "aiv_vec_ratio"})

    def test_percent_suffix_is_rejected_for_time_count_and_cycles_but_kept_for_ratios(self):
        capture = self.read_rows([self.row(**{"aicore_time(us)": "100%", "Block Num": "100%",
            "aic_total_cycles": "100%", "aic_mac_ratio": "98%"})])
        item = capture["tasks"][0]
        self.assertEqual(item.end_ns - item.start_ns, 100000)
        self.assertEqual(item.pmu["metrics"], {"aic_mac_ratio": .98})
        self.assertEqual(set(item.pmu["invalid_fields"]), {"aicore_time_us", "block_num", "aic_total_cycles"})

    def test_duplicate_native_rows_stay_one_call_even_if_optional_pmu_differs(self):
        first = self.row(**{"aic_mac_ratio": ".9", "aic_mte2_ratio": ".2"})
        second = self.row(**{"aic_mac_ratio": ".8", "aic_mte2_ratio": ".2"})
        capture = self.read_rows([first, second, second])
        self.assertEqual(capture["duplicate_rows"], 2)
        self.assertEqual(len(capture["tasks"]), 1)
        item = capture["tasks"][0]
        self.assertEqual(item.row_indices, [1, 2, 3])
        self.assertNotIn("aic_mac_ratio", item.pmu["metrics"])
        self.assertEqual(item.pmu["metrics"]["aic_mte2_ratio"], .2)
        self.assertEqual(item.pmu["invalid_fields"], ["aic_mac_ratio"])

    def test_real_call_duration_weighting_and_per_metric_coverage(self):
        result = summary([task(1, 100, {"aic_mac_ratio": .2}), task(2, 300, {"aic_mac_ratio": .8}),
                          task(3, 100)])
        metric = result["metrics"]["aic_mac_ratio"]
        self.assertAlmostEqual(metric["value"], .65)
        self.assertEqual((metric["known_calls"], metric["total_calls"]), (2, 3))
        self.assertAlmostEqual(metric["known_duration_ms"], .4)
        self.assertAlmostEqual(metric["duration_coverage"], .8)
        self.assertEqual(result["coverage"]["total_calls"], 3)
        self.assertNotIn("aiv_vec_ratio", result["metrics"])

    def test_native_call_deduplicated_and_clipped_stage_not_claimed_local_pmu(self):
        item = task(1, 100, {"aic_mac_ratio": .95, "aic_mte2_ratio": .2, "aic_mte3_ratio": .1})
        result = summary([item, item], [(0, 10000), (10000, 100000)])
        self.assertEqual(result["coverage"]["total_calls"], 1)
        self.assertAlmostEqual(result["coverage"]["total_duration_ms"], .1)
        self.assertEqual(result["coverage"]["clipped_calls"], 1)
        self.assertIn("整次调用", result["scope_note"])
        self.assertTrue(any("局部测量" in message for message in result["bound"]["evidence"]))

    def test_stage_hotspot_keeps_clipped_time_but_pmu_weights_whole_calls(self):
        tasks = [task(1, 100, {"aic_mac_ratio": .2}),
                 task(2, 300, {"aic_mac_ratio": .8}, start_ns=200000)]
        stage = core._lm_operator_stage({"start_ns": "20000", "end_ns": "450000"}, core.ATimeIndex(tasks),
                                       {"rank": "dp1", "round_id": "hc-1", "layer": 2})
        self.assertEqual(stage["status"], "MATCHED")
        means = core._lm_operator_source_means([{"class_stage_values": {("main", "CSA"): [
            {"layer": 2, "_operator_stages": {"experts": stage}}]}}])
        hotspot = means[("main", "CSA"), "experts"]["kernels"][0]
        self.assertAlmostEqual(hotspot["kernel_sum_ms"], .330)
        self.assertAlmostEqual(hotspot["duration_mean_ms"], .165)
        self.assertAlmostEqual(hotspot["pmu"]["metrics"]["aic_mac_ratio"]["value"], .65)
        self.assertAlmostEqual(hotspot["pmu"]["coverage"]["total_duration_ms"], .4)
        self.assertEqual(hotspot["pmu"]["coverage"]["clipped_calls"], 2)

    def test_vector_zeros_on_cube_do_not_make_vector_inefficient(self):
        result = summary([task(1, 100, {"aic_mac_ratio": 0, "aicore_time_us": 0,
            "aiv_vec_ratio": .97, "aiv_mte2_ratio": .3, "aiv_mte3_ratio": .2,
            "aiv_time_us": 99}, core_type="AI_VECTOR_CORE")])
        self.assertEqual(result["bound"]["code"], "aiv_compute")
        self.assertEqual(result["bound"]["confidence"], "中")
        self.assertNotIn("AIC：", " ".join(result["bound"]["evidence"]))
        self.assertEqual(result["metrics"]["aic_mac_ratio"]["value"], 0)

    def test_mixed_core_evaluates_each_core_in_its_own_denominator(self):
        result = summary([task(1, 100, {"aic_mac_ratio": .98, "aic_mte2_ratio": .3,
            "aic_mte3_ratio": .1, "aiv_vec_ratio": .2, "aiv_mte2_ratio": .96,
            "aiv_mte3_ratio": .1}, core_type="MIX_AIC")])
        self.assertEqual(result["bound"]["code"], "mixed_core")
        self.assertIn("矩阵计算", result["bound"]["label"])
        self.assertIn("读入搬运", result["bound"]["label"])
        self.assertTrue(any("分母不同" in value for value in result["bound"]["evidence"]))

    def test_communication_and_control_are_not_guessed_memory_or_cpu_bound(self):
        result = summary([task(1, 2000, {"aic_mac_ratio": 0, "aic_mte2_ratio": .99},
                               name="MoeDistributeDispatchV3")])
        self.assertEqual(result["bound"]["code"], "communication_mixed")
        self.assertEqual(result["bound"]["confidence"], "低")
        result = summary([task(1, 2000, name="EventWait")])
        self.assertEqual(result["bound"]["code"], "synchronization_control")

    def test_high_mac_is_only_compute_direction_when_other_pipes_observed(self):
        full = {"aic_mac_ratio": .98, "aic_mte2_ratio": .77, "aic_mte3_ratio": .01,
                "aicore_time_us": 99}
        result = summary([task(1, 100, full)])
        self.assertEqual(result["bound"]["code"], "aic_compute")
        self.assertEqual(result["bound"]["confidence"], "中")
        incomplete = summary([task(1, 100, {"aic_mac_ratio": .98})])
        self.assertEqual(incomplete["bound"]["code"], "aic_busy_compute")
        self.assertEqual(incomplete["bound"]["confidence"], "低")

    def test_short_tasks_and_task_core_time_gap_lower_confidence(self):
        metrics = {"aiv_vec_ratio": .99, "aiv_mte2_ratio": .2, "aiv_mte3_ratio": .1}
        short = summary([task(1, 5, metrics, core_type="AI_VECTOR_CORE")])
        self.assertEqual(short["bound"]["confidence"], "低")
        self.assertTrue(any("短任务" in value for value in short["bound"]["evidence"]))
        gap = summary([task(1, 100, dict(metrics, aiv_time_us=20), core_type="AI_VECTOR_CORE")])
        self.assertEqual(gap["bound"]["code"], "task_core_time_gap")
        self.assertEqual(gap["bound"]["confidence"], "低")

    def test_core_time_gap_uses_active_core_and_requires_both_mix_times(self):
        metrics = {"aic_mac_ratio": .98, "aic_mte2_ratio": .2, "aic_mte3_ratio": .1,
                   "aiv_vec_ratio": .3, "aiv_mte2_ratio": .96, "aiv_mte3_ratio": .1}
        # A small irrelevant AIV time must not replace an unknown AIC time.
        aic = summary([task(1, 100, dict(metrics, aiv_time_us=5), core_type="AI_CORE")])
        self.assertEqual(aic["bound"]["code"], "aic_compute")
        # MIX could spend the whole interval in the core whose time is absent.
        for known_time in ("aicore_time_us", "aiv_time_us"):
            mixed = summary([task(1, 100, dict(metrics, **{known_time: 5}), core_type="MIX_AIC")])
            self.assertEqual(mixed["bound"]["code"], "mixed_core")
        both = summary([task(1, 100, dict(metrics, aicore_time_us=20, aiv_time_us=5), core_type="MIX_AIC")])
        self.assertEqual(both["bound"]["code"], "task_core_time_gap")

    def test_missing_pmu_old_simple_namespace_is_unknown_not_zero(self):
        item = task(1, 100)
        del item.pmu
        result = summary([item])
        self.assertEqual(result["metrics"], {})
        self.assertEqual(result["coverage"]["known_calls"], 0)
        self.assertEqual(result["coverage"]["total_calls"], 1)
        self.assertEqual(result["bound"]["code"], "unknown")

    def test_pipeline_evidence_low_duration_coverage_reduces_confidence(self):
        metrics = {"aic_mac_ratio": .97, "aic_mte2_ratio": .2, "aic_mte3_ratio": .1}
        result = summary([task(1, 100, metrics), task(2, 900)])
        self.assertAlmostEqual(result["coverage"]["duration_coverage"], .1)
        self.assertEqual(result["bound"]["confidence"], "低")

    def test_optional_pmu_does_not_change_any_existing_layer_or_cycle_numbers(self):
        report, native = native_stage_report({"dp0": [1., 2.]})
        for item in native["dp0"]["tasks"]:
            item.task_type = "AI_CORE"
            item.pmu = {"metrics": {"aic_mac_ratio": .96, "aic_mte2_ratio": .3, "aic_mte3_ratio": .1},
                        "invalid_fields": [], "core_type": "AI_CORE"}
        # Setting task_type would deliberately change the operator fingerprint;
        # keep it identical in both fixtures to isolate optional PMU metadata.
        unchanged = copy.deepcopy(native)
        for item in unchanged["dp0"]["tasks"]:
            del item.pmu
        expected = attach(copy.deepcopy(report), unchanged)
        actual = attach(copy.deepcopy(report), native)
        def strip_pmu(value):
            if isinstance(value, dict):
                return {key: strip_pmu(item) for key, item in value.items() if key != "pmu"}
            if isinstance(value, list):
                return [strip_pmu(item) for item in value]
            return value
        self.assertEqual(strip_pmu(actual), strip_pmu(expected))
        for group in actual[0]["target_stage_breakdowns"]:
            for stage in group["stage_rows"]:
                for kernel in stage["operator_breakdown"]["kernels"]:
                    self.assertAlmostEqual(kernel["pmu"]["metrics"]["aic_mac_ratio"]["value"], .96)
                    self.assertEqual(kernel["pmu"]["aggregation"], "native_whole_call_duration_weighted")


if __name__ == "__main__":
    unittest.main()
