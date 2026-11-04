"""Input discovery/rank coverage using temporary paths, never device measurements."""
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import profile_ep128 as p


class InputCoverageTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)

    def capture(self, host, rank):
        directory = (self.base / host /
                     f"dp{rank}_pp0_tp0_dcp0_ep0_rank0_236_20261012041225380_ascend_pt" /
                     "ASCEND_PROFILER_OUTPUT")
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "kernel_details.csv"
        path.write_text("INTENTIONALLY NOT A KERNEL CSV\n", encoding="utf-8")
        return path.resolve()

    def host(self, name, ranks):
        return [self.capture(name, rank) for rank in ranks]

    def test_user_style_directory_uses_dp_before_rank_zero(self):
        for rank in range(16):
            path = self.capture("node", rank)
            with self.subTest(rank=rank):
                self.assertEqual(p.rank_name(path, 99), f"dp{rank}")

    def test_single_host_reports_the_eight_missing_global_ranks(self):
        selected = self.host("node0", range(8))
        self.host("node1", range(8, 16))
        files, names = p.case_inputs({"path": "node0"}, self.base)
        self.assertEqual(set(files), set(selected))
        self.assertEqual(names, [f"dp{i}" for i in range(8)])
        details = p.rank_coverage_details(names, {"expected_ranks": 16})
        self.assertEqual(details["expected"], [f"dp{i}" for i in range(16)])
        self.assertEqual(details["observed"], names)
        self.assertEqual(details["missing"], [f"dp{i}" for i in range(8, 16)])
        self.assertEqual(details["unexpected"], [])
        self.assertFalse(details["complete"])
        self.assertFalse(p.rank_coverage(names, {"expected_ranks": 16}))

    def test_paths_merge_both_hosts_without_an_eight_file_limit(self):
        selected = self.host("node0", range(8)) + self.host("node1", range(8, 16))
        files, names = p.case_inputs({"paths": ["node0", "node1"]}, self.base)
        self.assertEqual(len(files), 16)
        self.assertEqual(set(files), set(selected))
        self.assertEqual(set(names), {f"dp{i}" for i in range(16)})
        self.assertTrue(p.rank_coverage(names, {"expected_ranks": 16}))
        details = p.rank_coverage_details(names, {"expected_ranks": 16})
        self.assertEqual(details["missing"], [])
        self.assertEqual(details["unexpected"], [])
        self.assertTrue(details["complete"])

    def test_explicit_inputs_take_priority_without_mixing_other_paths(self):
        self.host("node0", range(8))
        selected = self.capture("node1", 15)
        files, names = p.case_inputs({
            "inputs": [{"path": str(selected), "rank": "dp15"}],
            "paths": ["node0"], "path": "does-not-exist"}, self.base)
        self.assertEqual(files, [selected])
        self.assertEqual(names, ["dp15"])

    def test_paths_take_priority_over_path(self):
        selected = self.host("node0", range(8))
        self.host("node1", range(8, 16))
        files, names = p.case_inputs({"paths": ["node0"], "path": "node1"}, self.base)
        self.assertEqual(set(files), set(selected))
        self.assertEqual(set(names), {f"dp{i}" for i in range(8)})

    def test_empty_higher_priority_inputs_do_not_silently_fall_back(self):
        self.capture("node0", 0)
        with self.assertRaisesRegex(ValueError, "inputs"):
            p.case_inputs({"inputs": [], "paths": ["node0"], "path": "node0"}, self.base)
        with self.assertRaisesRegex(ValueError, "path"):
            p.case_inputs({"paths": [], "path": "node0"}, self.base)

    def test_windows_backslashes_identify_dp8_and_dp15(self):
        for rank in (8, 15):
            path = "\\".join((r"D:\技术开发部\10T采样-Dspark入图\node1",
                              f"dp{rank}_pp0_tp0_dcp0_ep0_rank0_236_20261012041225380_ascend_pt",
                              "ASCEND_PROFILER_OUTPUT", "kernel_details.csv"))
            with self.subTest(rank=rank):
                self.assertEqual(p.rank_name(path, 99), f"dp{rank}")

    def test_details_preserve_expected_and_observed_order(self):
        details = p.rank_coverage_details(
            ["dp3", "dp3", "dp1", "foreign", "dp0", "foreign"],
            {"expected_ranks": 4})
        self.assertEqual(details, {
            "expected": ["dp0", "dp1", "dp2", "dp3"],
            "observed": ["dp3", "dp1", "foreign", "dp0"],
            "missing": ["dp2"], "unexpected": ["foreign"], "complete": False})
        custom = p.rank_coverage_details(["node-b", "outside"], {
            "expected_ranks": 3, "expected_rank_names": ["node-c", "node-b", "node-a"]})
        self.assertEqual(custom["expected"], ["node-c", "node-b", "node-a"])
        self.assertEqual(custom["missing"], ["node-c", "node-a"])
        self.assertEqual(custom["unexpected"], ["outside"])

    def test_partial_list_prints_coverage_and_does_not_read_csv(self):
        self.host("node0", range(8))
        config = {"defaults": {"expected_ranks": 16},
                  "cases": [{"name": "synthetic-paths-only", "path": "node0"}]}
        output = io.StringIO()
        with redirect_stdout(output), \
                mock.patch.object(p, "read_capture", side_effect=AssertionError("Must not read CSV")), \
                mock.patch.object(Path, "open", side_effect=AssertionError("Must not open CSV")):
            code = p.list_inputs(config, self.base)
        text = output.getvalue()
        self.assertEqual(code, 0, text)
        self.assertIn("配置期望rank：16；实际发现CSV：8；已识别全局rank：8", text)
        self.assertIn("缺少全局rank：" + ", ".join(f"dp{i}" for i in range(8, 16)), text)
        self.assertIn("Expected global-rank coverage: False", text)
        self.assertEqual(text.count("kernel_details.csv\n"), 8)

    def test_two_hosts_with_duplicate_local_names_are_not_renumbered(self):
        self.host("node0", range(8))
        self.host("node1", range(8))
        with self.assertRaisesRegex(ValueError, "Duplicate global rank/capture"):
            p.case_inputs({"paths": ["node0", "node1"]}, self.base)


if __name__ == "__main__":
    unittest.main()
