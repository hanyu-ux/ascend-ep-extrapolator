"""Documented PMU units remain distinct across memory levels and core scopes."""
import csv
from pathlib import Path
import tempfile
import unittest

import profile_ep128 as core


class PmuMemoryFieldTests(unittest.TestCase):
    @staticmethod
    def parse(row):
        columns = [(header, *core._a_pmu_header(header)) for header in row
                   if core._a_pmu_header(header) is not None]
        return core._a_read_pmu(row, columns)

    def test_memory_kb_headers_keep_export_units_and_do_not_expand_core_prefixes(self):
        for name in ("read_main_memory", "write_main_memory", "GM_to_L1", "L0C_to_L1",
                     "L0C_to_GM", "GM_to_UB", "UB_to_GM"):
            header = name + "_datas(KB)"
            expected = name.lower() + "_datas_kb"
            with self.subTest(header=header):
                self.assertEqual(core._a_pmu_header(header), (expected, 1.))
                self.assertEqual(core._a_pmu_header(expected), (expected, 1.))
                result = self.parse({header: "1.25"})
                self.assertEqual(result["metrics"], {expected: 1.25})
                self.assertEqual(core._A_PMU_FIELDS[expected], "KB")
            for prefix in ("aic_", "aiv_"):
                self.assertIsNone(core._a_pmu_header(prefix + header))

    def test_unknown_memory_units_and_undocumented_throughput_are_not_guessed(self):
        for header in ("read_main_memory_datas", "read_main_memory_datas(B)",
                       "read_main_memory_datas(bytes)", "read_main_memory_datas(MB)",
                       "read_main_memory_datas(KiB)", "aic_main_mem_read_bw(MB/s)",
                       "aic_main_mem_read_bw(GB*s)", "aic_main_mem_read_bw(GBs)",
                       "read_main_memory_datas(KB)%",
                       "aic_main_mem_read_bw", "aic_cube_fops(TFLOP/s)",
                       "aic_mac_flops", "hbm_bandwidth(GB/s)", "HBM read bytes",
                       "aiv_cube_fops", "aic_vector_fops", "aiv_vector_fops",
                       "aiv_l1_read_bw(GB/s)", "aiv_l0a_read_bw(GB/s)",
                       "aic_l2_read_bw", "aic_write_cache_hit", "aiv_read_hit_rate(%)"):
            with self.subTest(header=header):
                self.assertIsNone(core._a_pmu_header(header))

    def test_bandwidth_is_gbps_for_one_documented_channel_and_not_summed(self):
        row = {"aic_main_mem_read_bw(GB/s)": "100", "aic_main_mem_write_bw(GB/s)": "20",
               "aic_l1_read_bw(GB/s)": "300", "aic_l0c_read_bw_cube(GB/s)": "400",
               "aiv_ub_read_bw_vector(GB/s)": "500", "aiv_ub_write_bw_scalar(GB/s)": "2.5",
               "aic_gm_to_ub_bw(GB/s)": "200", "aiv_ub_to_gm_bw(GB/s)": "10"}
        result = self.parse(row)
        self.assertEqual(result["metrics"], {"aic_main_mem_read_bw_gbps": 100.,
            "aic_main_mem_write_bw_gbps": 20., "aic_l1_read_bw_gbps": 300.,
            "aic_l0c_read_bw_cube_gbps": 400., "aiv_ub_read_bw_vector_gbps": 500.,
            "aiv_ub_write_bw_scalar_gbps": 2.5, "aic_gm_to_ub_bw_gbps": 200.,
            "aiv_ub_to_gm_bw_gbps": 10.})
        self.assertEqual(core._A_PMU_FIELD_INFO["aic_l1_read_bw_gbps"]["scope"], "internal_l1")
        self.assertEqual(core._A_PMU_FIELD_INFO["aic_main_mem_read_bw_gbps"]["scope"], "reported_main_memory")
        self.assertEqual(core._A_PMU_FIELD_INFO["aic_gm_to_ub_bw_gbps"]["products"], "Atlas inference (310P)")

    def test_fops_are_reported_operations_not_mac_times_two_or_a_rate(self):
        row = {"aic_cube_fops": "1024", "aic_vec_fops": "64", "aiv_vec_fops": "32"}
        parsed = self.parse(row)
        self.assertEqual(parsed["metrics"], {key: float(value) for key, value in row.items()})
        for field in row:
            self.assertEqual(core._A_PMU_FIELDS[field], "operations")
            self.assertIn("not effective algorithm FLOPs", core._A_PMU_FIELD_INFO[field]["description"])

    def test_cache_and_channel_ratios_keep_percentage_scaling_and_estimate_flag(self):
        row = {"aic_l2_cache_hit_rate(%)": "97.5",
               "GM_to_L1_bw_usage_rate(%)": "20", "L1_to_GM_bw_usage_rate(%)(estimate)": "40",
               "L1_to_GM_datas(KB)(estimate)": "1.5"}
        parsed = self.parse(row)
        self.assertEqual(parsed["metrics"], {"aic_l2_cache_hit_rate": .975,
            "gm_to_l1_bw_usage_rate": .2, "l1_to_gm_bw_usage_rate_estimate": .4,
            "l1_to_gm_datas_kb_estimate": 1.5})
        self.assertTrue(core._A_PMU_FIELD_INFO["l1_to_gm_datas_kb_estimate"]["estimated"])
        self.assertFalse(core._A_PMU_FIELD_INFO["gm_to_l1_datas_kb"]["estimated"])
        self.assertIsNone(core._a_pmu_header("L1_to_GM_datas(KB)"))

    def test_nonfinite_negative_invalid_suffix_and_conflicting_aliases_stay_invalid(self):
        for value in ("NaN", "Infinity", "-1", "20%", "bad"):
            with self.subTest(value=value):
                parsed = self.parse({"read_main_memory_datas(KB)": value})
                self.assertEqual(parsed["metrics"], {})
                self.assertEqual(parsed["invalid_fields"], ["read_main_memory_datas_kb"])
        for missing in ("", "N/A", "--"):
            self.assertEqual(self.parse({"read_main_memory_datas(KB)": missing}), {})
        conflict = self.parse({"read_main_memory_datas(KB)": "1", "read_main_memory_datas_kb": "2"})
        self.assertEqual(conflict["metrics"], {})
        self.assertEqual(conflict["invalid_fields"], ["read_main_memory_datas_kb"])
        count = self.parse({"block_num": "1.5"})
        self.assertEqual(count["invalid_fields"], ["block_num"])

    def test_native_duplicate_merge_preserves_new_units_and_conflict_validation(self):
        row = {"Name": "GroupedMatmul", "Type": "GroupedMatmul", "Start Time(us)": "1",
               "Duration(us)": "100", "Task ID": "7", "Stream ID": "2",
               "read_main_memory_datas(KB)": "1024", "aic_l1_read_bw(GB/s)": "200",
               "aic_mac_ratio": ".9"}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "kernel_details.csv"
            with path.open("w", encoding="utf-8", newline="") as output:
                writer = csv.DictWriter(output, fieldnames=list(row))
                writer.writeheader()
                writer.writerows([row, dict(row, **{"aic_l1_read_bw(GB/s)": "201"})])
            tasks = core.read_capture(path, "dp0")["tasks"]
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0].row_indices, [1, 2])
        self.assertEqual(tasks[0].pmu["metrics"], {"read_main_memory_datas_kb": 1024., "aic_mac_ratio": .9})
        self.assertEqual(tasks[0].pmu["invalid_fields"], ["aic_l1_read_bw_gbps"])

    def test_all_supported_fields_describe_units_export_scope_and_official_provenance(self):
        self.assertEqual(set(core._A_PMU_FIELDS), set(core._A_PMU_FIELD_INFO))
        for field, unit in core._A_PMU_FIELDS.items():
            with self.subTest(field=field):
                info = core._A_PMU_FIELD_INFO[field]
                self.assertEqual(info["unit"], unit)
                self.assertTrue(info["scope"])
                self.assertTrue(info["description"])
                self.assertTrue(any(info["source"].startswith(source) for source in (core._A_PMU_DOC, core._A_PMU_MSPROF_DOC)))
                self.assertIn("not inferred as a whole-device total", info["aggregation"])
        legacy = {field for field, _ in core._A_PMU_LEGACY_FIELDS}
        added = set(core._A_PMU_FIELDS) - legacy
        self.assertEqual(len(added), 44)
        for field in added:
            info = core._A_PMU_FIELD_INFO[field]
            self.assertTrue(info["products"])
            self.assertEqual(core._a_pmu_header(info["source_header"])[0], field)


if __name__ == "__main__":
    unittest.main()
