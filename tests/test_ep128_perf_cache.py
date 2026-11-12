"""Analysis caches preserve native fingerprints, PMU validation and weighting."""
import copy
import csv
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import profile_ep128 as core


def task(index=1, **updates):
    values = dict(row_index=index, row_indices=[index], start_ns=1000,
                  end_ns=101000, name="GroupedMatmul", op_type="GroupedMatmul",
                  stream_id="2", task_id=str(index), task_type="AI_CORE",
                  input_shapes="32,7168;14,224,512,16,32", output_shapes="32,8192",
                  input_dtypes="DT_FLOAT8_E4M3FN;DT_FLOAT4_E2M1", output_dtypes="DT_BF16",
                  pmu={"metrics": {"aic_mac_ratio": .95}, "invalid_fields": [], "core_type": "AI_CORE"})
    values.update(updates)
    return SimpleNamespace(**values)


class PerformanceCacheTests(unittest.TestCase):
    def read_native(self, rows):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        path = Path(temporary.name) / "kernel_details.csv"
        fields = list(dict.fromkeys(key for row in rows for key in row))
        with path.open("w", newline="", encoding="utf-8-sig") as output:
            writer = csv.DictWriter(output, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        return core.read_capture(path, "dp0")["tasks"][0]

    @staticmethod
    def row(**pmu):
        return {"Name": "GroupedMatmul", "Type": "GroupedMatmul", "Start Time(us)": "1",
                "Duration(us)": "100", "Task ID": "7", "Stream ID": "2", **pmu}

    def test_same_task_fingerprint_is_reused_across_complete_hc_window_scans(self):
        item = task()
        index = core.ATimeIndex([item])
        expected = core._lm_operator_member(item, index)
        self.assertIs(core._lm_operator_member(item, index), expected)
        scope = {"rank": "dp0", "round_id": "r1", "layer": 1}
        first = core._lm_operator_layer({"layer": 1, "start_ns": "1000", "end_ns": "11000"}, index, scope)
        second = core._lm_operator_layer({"layer": 2, "start_ns": "11000", "end_ns": "101000"},
                                         index, {**scope, "layer": 2})
        for result, duration in ((first, 10000), (second, 90000)):
            self.assertEqual(result["status"], "MATCHED", result["reason"])
            kernel, = result["kernels"].values()
            self.assertIs(kernel["member"], expected)
            self.assertEqual(kernel["member"]["key"], expected["key"])
            self.assertEqual(kernel["durations_ns"], [duration])
            self.assertEqual(kernel["example"]["task_start_ns"], str(item.start_ns))
            self.assertEqual(kernel["example"]["task_end_ns"], str(item.end_ns))
            self.assertNotIn("rows", result)
        self.assertEqual(len(index._operator_members), 1)
        self.assertEqual(expected["input_shapes"], item.input_shapes)

    def test_fingerprint_cache_is_index_scoped_and_preserves_full_identity(self):
        first, same = task(), task(2)
        changed = task(3, output_shapes="32,4096")
        index = core.ATimeIndex([first, same, changed])
        member = core._lm_operator_member(first, index)
        self.assertIs(core._lm_operator_member(same, index), member)
        self.assertNotEqual(core._lm_operator_member(changed, index)["key"], member["key"])
        fresh = copy.copy(first)
        fresh.name = "AnotherKernel"
        self.assertNotEqual(core._lm_operator_member(fresh, core.ATimeIndex([fresh]))["key"], member["key"])

    def test_validated_native_pmu_is_shared_without_changing_clip_or_call_identity(self):
        item = self.read_native([self.row(aic_mac_ratio=".95", aic_mte2_ratio="25%")])
        first = core._lm_operator_pmu_observation(item, 1000, 11000, {"rank": "dp0"})
        second = core._lm_operator_pmu_observation(item, 11000, 101000, {"rank": "dp0"})
        self.assertIs(first["metrics"], item.pmu["metrics"])
        self.assertIs(second["metrics"], first["metrics"])
        self.assertEqual(first["identity"], second["identity"])
        self.assertTrue(first["clipped"])
        member = core._lm_operator_member(item, core.ATimeIndex([item]))
        result = core._lm_operator_pmu_summary(member, [first, second])
        self.assertEqual(result["coverage"]["total_calls"], 1)
        self.assertEqual(result["metrics"]["aic_mte2_ratio"]["value"], .25)
        self.assertEqual(result["coverage"]["total_duration_ms"], .1)

    def test_duplicate_conflicts_still_override_native_validated_metrics(self):
        item = self.read_native([self.row(aic_mac_ratio=".95", aic_mte2_ratio=".25"),
                                 self.row(aic_mac_ratio=".90", aic_mte2_ratio=".25")])
        result = core._lm_operator_pmu_observation(item, 1000, 101000, {"rank": "dp0"})
        self.assertEqual(result["metrics"], {"aic_mte2_ratio": .25})
        self.assertEqual(result["invalid_fields"], ["aic_mac_ratio"])
        self.assertEqual(item.row_indices, [1, 2])

    def test_manual_task_pmu_remains_validated_and_independent_of_original_dict(self):
        item = task(pmu={"metrics": {"aic_mac_ratio": True, "aiv_vec_ratio": math.nan,
                         "aic_mte2_ratio": 1.1, "aic_mte3_ratio": -.1, "aic_scalar_ratio": .2,
                         "unrecognized": 1}, "invalid_fields": ["aic_scalar_ratio"]})
        first = core._lm_operator_pmu_observation(item, 1000, 101000, {"rank": "dp0"})
        self.assertEqual(first["metrics"], {})
        item.pmu["metrics"]["aic_mac_ratio"] = .8
        second = core._lm_operator_pmu_observation(item, 1000, 101000, {"rank": "dp0"})
        self.assertEqual(second["metrics"], {"aic_mac_ratio": .8})
        self.assertEqual(first["metrics"], {})

    def test_numeric_cache_keeps_units_invalids_and_row_conflicts_separate(self):
        columns = [("ratio", "aic_mac_ratio", 1.), ("percent", "aic_mte2_ratio", .01),
                   ("time", "aicore_time_us", .001), ("count", "block_num", 1.)]
        row = {"ratio": "50%", "percent": "50", "time": "50", "count": "50%"}
        first = core._a_read_pmu(row, columns)
        self.assertEqual(first["metrics"], {"aic_mac_ratio": .5, "aic_mte2_ratio": .5,
                                           "aicore_time_us": .05})
        self.assertEqual(first["invalid_fields"], ["block_num"])
        first["metrics"].clear()
        second = core._a_read_pmu(row, columns)
        self.assertEqual(second["metrics"]["aic_mac_ratio"], .5)
        conflict = core._a_read_pmu({"a": ".5", "b": "60%"},
                                   [("a", "aic_mac_ratio", 1.), ("b", "aic_mac_ratio", 1.)])
        self.assertEqual(conflict["metrics"], {})
        self.assertEqual(conflict["invalid_fields"], ["aic_mac_ratio"])

    def test_numeric_cache_has_a_fixed_memory_bound(self):
        for value in range(10000):
            core._a_pmu_value(str(value), "cycles", 1.)
        self.assertLessEqual(core._a_pmu_value.cache_info().currsize, 8192)


if __name__ == "__main__":
    unittest.main()
