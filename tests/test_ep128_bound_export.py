"""Offline bound views preserve all nine sheets, source evidence and input JSON."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
from zipfile import ZipFile

import profile_ep128 as core
import build_ep128_leadership_excel as renderer
from test_ep128_layer_mean_excel import operator_drilldown_report
from tests.test_ep128_operator_pmu import summary, task


def report_fixture():
    report = operator_drilldown_report()
    for case in report["cases"]:
        for method in case["projection"]["layer_mean_methods"]:
            for kernel in core._bound_method_kernels(method):
                kernel["pmu"] = summary([task(1, 100, {"aic_mac_ratio": .975, "aic_scalar_ratio": .1,
                    "aic_mte1_ratio": .2, "aic_mte2_ratio": .3, "aic_fixpipe_ratio": .15})])
    return report


class BoundExportTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def test_refresh_xlsx_html_and_derived_json_preserve_original_numerics(self):
        report = report_fixture()
        source, out = self.root / "core.json", self.root / "bound.xlsx"
        source.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
        original = source.read_bytes()
        before = copy.deepcopy(report)
        with mock.patch.object(core, "read_capture", side_effect=AssertionError("no CSV reads")):
            renderer.build_workbook(out, report=source, report_data=report, refresh_bound=True)
        self.assertEqual(source.read_bytes(), original)
        self.assertEqual(report, before)
        derived = json.loads(out.with_suffix(".bound.json").read_text(encoding="utf-8"))
        for old_case, new_case in zip(before["cases"], derived["cases"]):
            for old, new in zip(old_case["projection"]["layer_mean_methods"], new_case["projection"]["layer_mean_methods"]):
                for key in ("cycle_ms", "tpot_ms", "per_device_tokens_s", "selected_rank", "breakdown", "class_means"):
                    self.assertEqual(old[key], new[key])
        self.assertFalse(derived["operator_bound_analysis"]["source_csv_read"])
        with ZipFile(out) as archive:
            self.assertEqual(archive.read("xl/workbook.xml").count(b"<sheet "), 10)
            for index in (5, 6):
                text = archive.read(f"xl/worksheets/sheet{index}.xml").decode()
                self.assertIn("MTE1", text)
                self.assertIn("Fixpipe", text)
                self.assertIn("Roofline", text)
        html = out.with_suffix(".html").read_text(encoding="utf-8")
        for value in ("MTE1", "Fixpipe", "Roofline定量核对", "CSA", "HCA", "SWA"):
            self.assertIn(value, html)

    def test_cli_refresh_does_not_require_original_csv_or_cases(self):
        source, out = self.root / "core.json", self.root / "cli.xlsx"
        source.write_text(json.dumps(report_fixture()), encoding="utf-8")
        result = subprocess.run([sys.executable, "-B", "-S", renderer.__file__, "--report", str(source),
            "--refresh-bound", "--out", str(out)], text=True, capture_output=True, timeout=40)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue(out.with_suffix(".bound.json").is_file())
        self.assertIn("bound更新核心JSON", result.stdout)

    def test_output_input_profile_and_template_collisions_fail_without_modifying_inputs(self):
        source = self.root / "core.json"
        report = report_fixture()
        source.write_text(json.dumps(report), encoding="utf-8")
        before = source.read_bytes()
        for kwargs in ({"bound_template": source}, {"bound_template": self.root / "output.xlsx"},
                       {"bound_profile": self.root / "output.bound.json"}):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, "覆盖|独立路径"):
                renderer.build_workbook(self.root / "output.xlsx", report=source, report_data=report,
                                        refresh_bound=True, **kwargs)
        self.assertEqual(source.read_bytes(), before)

    def test_bound_flags_reject_configuration_preview_without_fabricating_evidence(self):
        for kwargs in ({"refresh_bound": True}, {"bound_template": self.root / "profile.json"},
                       {"bound_profile": self.root / "profile.json"}):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, "预览"):
                renderer.build_workbook(self.root / "preview.xlsx", cases=self.root / "missing.json", preview=True, **kwargs)


if __name__ == "__main__":
    unittest.main()
