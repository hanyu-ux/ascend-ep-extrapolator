#!/usr/bin/env python3
"""Render configuration scopes without claiming these fixtures are measurements."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from test_ep128_v2_excel import fixture, read_book, renderer


class ReportScopeTests(unittest.TestCase):
    def export(self, report, root):
        origin, out = Path(root) / "report.json", Path(root) / "scope.xlsx"
        origin.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
        renderer.build_workbook(out, report=origin)
        return read_book(out)[0], out.with_suffix(".html").read_text(encoding="utf-8")

    def test_explicit_ep2_and_ep128_scopes_survive_xlsx_and_html_export(self):
        for layers, ep in ((8, 2), (61, 128)):
            with self.subTest(layers=layers, ep=ep), tempfile.TemporaryDirectory() as root:
                report = fixture()
                case = report["cases"][0]
                case["target_model"]["main"]["num_hidden_layers"] = layers
                case["config"].update(draft_layers=3, base_ep=2 if ep == 2 else 16, target_ep=ep)
                book, page = self.export(report, root)
                expected = f"目标{layers}主层＋3草稿层、EP{ep}"
                self.assertIn(expected, book["估算总览"][1]["A3"])
                self.assertIn(expected, page)
                # Scope text is display metadata: the core's numeric result is untouched.
                self.assertIn(519.4321, book["估算总览"][1].values())
                if ep == 2:
                    self.assertNotIn("EP128", str(book))
                    self.assertNotIn("61主层", page)

    def test_mixed_and_missing_targets_are_listed_per_case_without_default_target(self):
        report = fixture()
        first = report["cases"][0]
        first["name"] = "SYNTHETIC_EP2"
        first["target_model"]["main"]["num_hidden_layers"] = 8
        first["config"].update(draft_layers=3, base_ep=2, target_ep=2)
        missing = copy.deepcopy(first)
        missing["name"] = "SYNTHETIC_MISSING_TARGET"
        missing["target_model"] = None
        # A leftover EP setting or structural diagnostics must not invent a model.
        missing["config"]["target_ep"] = 128
        missing["projection"]["structure"]["target_counts"] = {"main_HCA_normal": 61}
        for case in (first, missing):
            for method in case["projection"]["layer_mean_methods"]:
                method.update(status="PENDING", reasons=["Unverified synthetic evidence"])
        report["cases"].append(missing)
        with tempfile.TemporaryDirectory() as root:
            book, page = self.export(report, root)
            preview_path = Path(root) / "preview.json"
            preview_path.write_text(json.dumps(dict(schema_version=2,
                sample_model=first["source_model"], cases=[dict(case["config"],
                    name=case["name"], target_model=case["target_model"]) for case in report["cases"]])),
                encoding="utf-8")
            preview = renderer._preview_report(preview_path)
            self.assertIn("目标8主层＋3草稿层、EP2", renderer._case_scope(preview["cases"][0]))
            self.assertIn("目标模型未配置", renderer._case_scope(preview["cases"][1]))
        overview = book["估算总览"][1]
        self.assertIn("按采样组合分别列示", overview["A3"])
        self.assertIn("目标8主层＋3草稿层、EP2", str(overview))
        self.assertIn("目标模型未配置", str(overview))
        self.assertIn("目标模型未配置", page)
        self.assertNotIn("61主层", page)
        self.assertNotIn(519.4321, overview.values())
        self.assertNotIn(40.125, overview.values())
        self.assertNotIn("EP128", renderer._case_scope(missing))


if __name__ == "__main__":
    unittest.main()
