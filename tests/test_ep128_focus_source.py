"""FOCUS source tables live once on baseline; method charts keep exact references."""
import copy
from pathlib import Path
import re
import tempfile
import unittest

from test_ep128_focus_excel import CHART_NS, focus_fixture
from test_ep128_v2_excel import read_book, renderer
import test_ep128_layer_mean_excel as layer_excel


class FocusSourceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="synthetic-focus-source-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def export(self, report):
        before = copy.deepcopy(report)
        out = self.root / "focus-source.xlsx"
        renderer.build_workbook(out, report=self.root / "synthetic.json", report_data=report)
        self.assertEqual(report, before)
        return out, read_book(out)[0], out.with_suffix(".html").read_text(encoding="utf-8")

    def focus_line(self, out, book, sheet):
        charts = layer_excel.LayerMeanRendererTests.chart_parts_by_sheet(self, out, book)[sheet]
        return next(chart for chart in charts if chart.find(".//c:lineChart", CHART_NS) is not None
                    and "两项U逐轮耗时变化" in "".join(chart.itertext()))

    def test_common_rank_has_one_source_table_and_both_methods_reference_it(self):
        report = focus_fixture()
        case = report["cases"][0]
        methods = case["projection"]["layer_mean_methods"]
        methods[1]["selected_rank"] = methods[0]["selected_rank"]
        out, book, page = self.export(report)
        source = book["源采样基线"][1]
        self.assertEqual([str(value).split("\n", 1)[0] for value in source.values()].count("保留序号"), 1)
        formulas = []
        for name in ("方法1估算", "方法2估算"):
            values = list(book[name][1].values())
            self.assertNotIn("保留序号", values)
            self.assertIn("AscendBlockTables.compute_slot_mappings", "\n".join(map(str, values)))
            line = self.focus_line(out, book, name)
            references = [node.text for node in line.findall(".//c:f", CHART_NS)]
            self.assertTrue(references)
            self.assertTrue(all(reference.startswith("'源采样基线'!") for reference in references))
            formulas.append(references)
            links = book[name][0].findall(".//s:hyperlink", {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"})
            self.assertTrue(any("源采样基线" in link.get("location", "") for link in links))
        self.assertEqual(formulas[0], formulas[1])
        self.assertEqual(page.count("<th>保留序号</th>"), 1)
        anchor = renderer._focus_source_anchor(case, "dp7")
        self.assertIn('id="' + anchor + '"', page)
        self.assertEqual(page.count('href="#' + anchor + '"'), 2)
        for method in methods:
            method_html = renderer._method_focus_html(case, method)
            self.assertNotIn("<th>保留序号</th>", method_html)
            self.assertIn("两项U逐轮耗时变化", method_html)

    def test_legacy_different_ranks_keep_separate_true_source_values(self):
        out, book, page = self.export(focus_fixture())
        source = book["源采样基线"][1]
        self.assertEqual([str(value).split("\n", 1)[0] for value in source.values()].count("保留序号"), 2)
        references = []
        for name, factor in (("方法1估算", 1.), ("方法2估算", 10.)):
            line = self.focus_line(out, book, name)
            series = line.findall(".//c:lineChart/c:ser", CHART_NS)
            values = [float(point.findtext("c:v", namespaces=CHART_NS)) for point in
                      series[0].findall("c:val/c:numRef/c:numCache/c:pt", CHART_NS)]
            self.assertEqual(values, [factor * value for value in (.5, 1., 1.5)])
            references.append(series[0].findtext("c:val/c:numRef/c:f", namespaces=CHART_NS))
        self.assertNotEqual(*references)
        self.assertEqual(page.count("<th>保留序号</th>"), 2)

    def test_missing_step_is_blank_in_source_and_gap_stays_at_original_hc_id(self):
        report = focus_fixture()
        stage = report["cases"][0]["measurements"][0]["step_residuals"]["focus_diagnostics"]["stages"][0]
        stage["per_step"][1].update(status="UNAVAILABLE", duration_ms=None,
            recorded_busy_ms=None, no_recorded_ms=None, reason="SYNTHETIC_SOURCE_ONLY_MISSING")
        out, book, page = self.export(report)
        line = self.focus_line(out, book, "方法1估算")
        series = line.findall(".//c:lineChart/c:ser", CHART_NS)[0]
        cache = series.find("c:val/c:numRef/c:numCache", CHART_NS)
        self.assertEqual(cache.find("c:ptCount", CHART_NS).get("val"), "3")
        self.assertEqual([point.get("idx") for point in cache.findall("c:pt", CHART_NS)], ["0", "2"])
        reference = series.findtext("c:val/c:numRef/c:f", namespaces=CHART_NS)
        first = int(re.search(r"\$(\d+):", reference).group(1))
        self.assertIsNone(book["源采样基线"][1].get("D" + str(first + 1)))
        self.assertEqual(book["源采样基线"][1]["B" + str(first + 1)], "hc-13")
        self.assertIn("SYNTHETIC_SOURCE_ONLY_MISSING", "\n".join(map(str, book["源采样基线"][1].values())))
        self.assertNotIn("SYNTHETIC_SOURCE_ONLY_MISSING", "\n".join(map(str, book["方法1估算"][1].values())))
        self.assertIn("SYNTHETIC_SOURCE_ONLY_MISSING", page)

    def test_long_status_continuations_keep_chart_references_exact(self):
        report = focus_fixture()
        stage = report["cases"][0]["measurements"][0]["step_residuals"]["focus_diagnostics"]["stages"][0]
        stage["per_step"][1]["reason"] = "SYNTHETIC_LONG_REASON\n" * 90
        out, book, _ = self.export(report)
        line = self.focus_line(out, book, "方法1估算")
        series = line.findall(".//c:lineChart/c:ser", CHART_NS)[0]
        reference = series.findtext("c:val/c:numRef/c:f", namespaces=CHART_NS)
        self.assertRegex(reference, r"^'源采样基线'!\$G\$\d+:\$G\$\d+$")
        first = int(re.search(r"\$(\d+):", reference).group(1))
        self.assertEqual([book["源采样基线"][1]["G" + str(first + index)] for index in range(3)], [.5, 1., 1.5])
        self.assertEqual([book["源采样基线"][1]["A" + str(first + index)] for index in range(3)], ["hc-10", "hc-13", "hc-18"])
        text = "\n".join(map(str, book["源采样基线"][1].values()))
        self.assertEqual(text.count("SYNTHETIC_LONG_REASON"), 90)
        self.assertIn("长文本已保留", text)

    def test_source_anchor_is_stable_and_distinguishes_case_and_rank(self):
        case = focus_fixture()["cases"][0]
        anchor = renderer._focus_source_anchor(case, "dp7")
        self.assertEqual(anchor, renderer._focus_source_anchor(copy.deepcopy(case), "dp7"))
        self.assertNotEqual(anchor, renderer._focus_source_anchor(case, "dp3"))
        self.assertNotEqual(anchor, renderer._focus_source_anchor(dict(case, name="another-case"), "dp7"))
        self.assertRegex(anchor, r"^focus-source-[a-f0-9]{16}$")


if __name__ == "__main__":
    unittest.main()
