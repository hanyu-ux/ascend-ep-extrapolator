"""Workbook navigation links point to real, correctly scoped report evidence."""
from collections import Counter
from contextlib import ExitStack
import copy
from html.parser import HTMLParser
from pathlib import Path
import re
import tempfile
import unittest
from unittest import mock
import xml.etree.ElementTree as ET
from zipfile import ZipFile

import profile_ep128 as core
from test_ep128_focus_excel import focus_fixture
from test_ep128_v2_excel import read_book, renderer


NS = {"s": renderer.xlsx_writer.MAIN}
NAV_SHEETS = ("估算总览", "源采样基线", "方法1估算", "方法2估算")


def two_case_report(*, duplicate_names=False):
    report = focus_fixture()
    first = report["cases"][0]
    first["name"] = "SYNTHETIC_甲'case & <图> 🚀"
    second = copy.deepcopy(first)
    second["name"] = first["name"] if duplicate_names else "SYNTHETIC_乙case 🧪"
    second["config"]["bs"] += 1
    # Make same-rank source data distinguishable across the two cases.
    for capture in second["measurements"]:
        for stage in capture["step_residuals"]["focus_diagnostics"]["stages"]:
            for step in stage["per_step"]:
                for key in ("duration_ms", "recorded_busy_ms", "no_recorded_ms", "kernel_sum_ms"):
                    step[key] *= 2
                step["component_durations_ms"] = {key: value * 2
                    for key, value in step["component_durations_ms"].items()}
            for key in ("duration_ms_mean", "duration_ms_p50", "duration_ms_p90", "duration_ms_max"):
                stage[key] *= 2
            for item in stage["summary"]:
                for key in ("duration_ms_mean", "duration_ms_p50", "duration_ms_p90", "duration_ms_max"):
                    item[key] *= 2
    report["cases"].append(second)
    return report


def location_parts(location):
    match = re.fullmatch(r"'((?:[^']|'')*)'!([A-Z]+[1-9][0-9]*)", location)
    if match is None:
        raise AssertionError("Unexpected internal location: " + repr(location))
    return match[1].replace("''", "'"), match[2]


class HTMLTargets(HTMLParser):
    def __init__(self, document):
        super().__init__(convert_charrefs=True)
        self.ids, self.fragments, self.targets = [], [], []
        self.feed(document)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if attrs.get("id"):
            self.ids.append(attrs["id"])
        if tag == "a" and attrs.get("href", "").startswith("#"):
            self.fragments.append(attrs["href"][1:])
        if tag in ("svg", "table") or tag == "div" and "u-bars" in attrs.get("class", "").split():
            self.targets.append((tag, attrs.get("id")))


class ReportNavigationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="synthetic-report-navigation-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def build(self, report, *, navigation=True, stem="navigation"):
        before = copy.deepcopy(report)
        saved = []
        original_save = renderer.Workbook.save

        def save(book, path):
            saved.append(book)
            return original_save(book, path)

        out = self.root / (stem + ".xlsx")
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(renderer.Workbook, "save", new=save))
            stack.enter_context(mock.patch.object(core, "read_capture",
                side_effect=AssertionError("Presentation must not reread profiling inputs")))
            if not navigation:
                stack.enter_context(mock.patch.object(renderer, "_navigation_start", return_value=5))
                stack.enter_context(mock.patch.object(renderer, "_navigation_case", return_value=None))
                stack.enter_context(mock.patch.object(renderer, "_navigation_finish", return_value=None))
            renderer.build_workbook(out, report=self.root / "synthetic-source.json", report_data=report)
        self.assertEqual(report, before)
        self.assertEqual(len(saved), 1)
        return out, read_book(out)[0], {sheet.name: sheet for sheet in saved[0].sheets}

    def test_nine_sheets_and_all_serialized_links_have_written_internal_targets(self):
        out, book, _ = self.build(two_case_report())
        self.assertEqual(tuple(book), renderer.SHEETS)
        for name, (root, cells) in book.items():
            links = root.findall("s:hyperlinks/s:hyperlink", NS)
            if name in NAV_SHEETS:
                self.assertGreater(len(links), 10, name)
                self.assertIn("A4", [link.get("ref") for link in links])
                pane = root.find("s:sheetViews/s:sheetView/s:pane", NS)
                self.assertGreaterEqual(int(pane.get("ySplit")), 4)
            for link in links:
                self.assertIn(link.get("ref"), cells, (name, link.attrib))
                target, reference = location_parts(link.get("location"))
                self.assertIn(target, book)
                self.assertIn(reference, book[target][1], (name, link.attrib))
                self.assertEqual(link.get("display"), str(cells[link.get("ref")]))
                self.assertIsNone(link.get("{" + renderer.xlsx_writer.REL + "}id"))
        with ZipFile(out) as archive:
            for part in archive.namelist():
                if part.endswith(".rels"):
                    root = ET.fromstring(archive.read(part))
                    self.assertFalse(any(item.get("TargetMode") == "External" for item in root))

    def test_every_chart_and_recorded_table_has_a_directory_link_to_its_real_anchor(self):
        _, _, sheets = self.build(two_case_report())
        for name in NAV_SHEETS:
            sheet = sheets[name]
            self.assertTrue(sheet.charts, name)
            links = [(position, target, sheet.cells[position][1])
                     for position, target in sheet.internal_links.items()]
            for chart in sheet.charts:
                row, col = chart["anchor"][:2]
                entries = [(position, label) for position, target, label in links
                           if target == (name, row, col) and str(label).startswith("图 · ")]
                self.assertEqual(len(entries), 1, (name, chart["title"]))
                self.assertEqual(entries[0][1], "图 · " + chart["title"])
                self.assertIn((row, col), sheet.cells)
            for entry in sheet.navigation_entries:
                self.assertTrue(any(target == (name, entry["row"], entry.get("col", 1))
                    and str(label).startswith("表 · ") for _, target, label in links), (name, entry))

    def test_case_entries_and_chart_directory_groups_do_not_cross_case_boundaries(self):
        report = two_case_report()
        _, _, sheets = self.build(report)
        for name in NAV_SHEETS:
            sheet = sheets[name]
            case_starts = [(row, value) for row, value in sheet.navigation_cases]
            self.assertEqual([value for _, value in case_starts], [case["name"] for case in report["cases"]])
            self.assertLess(case_starts[0][0], case_starts[1][0])
            for index, (case_row, case_name) in enumerate(case_starts, 1):
                index_links = [(position, target) for position, target in sheet.internal_links.items()
                               if str(sheet.cells[position][1]).startswith(str(index) + ". " + case_name)]
                self.assertEqual(len(index_links), 1)
                self.assertEqual(index_links[0][1], (name, case_row, 1))
            for chart in sheet.charts:
                row, col = chart["anchor"][:2]
                expected = next((case_name for start, case_name in reversed(case_starts) if start <= row),
                                "跨case汇总与趋势")
                source = next(position for position, target in sheet.internal_links.items()
                              if target == (name, row, col) and str(sheet.cells[position][1]).startswith("图 · "))
                headings = [(position[0], cell[1]) for position, cell in sheet.cells.items()
                            if position[1] == 1 and cell[2] == renderer.xlsx_writer.STYLE_IDS["header"]
                            and cell[1] in ["跨case汇总与趋势", *[case["name"] for case in report["cases"]]]
                            and position[0] < source[0]]
                self.assertTrue(headings)
                self.assertEqual(max(headings)[1], expected, (name, chart["title"]))

    def test_shared_source_links_and_line_ranges_resolve_to_each_case_own_values(self):
        report = two_case_report()
        _, book, sheets = self.build(report)
        source_name = "源采样基线"
        for name, factor in (("方法1估算", 1.), ("方法2估算", 10.)):
            sheet = sheets[name]
            lines = [chart for chart in sheet.charts if chart["title"].startswith("两项U逐轮耗时变化")]
            self.assertEqual(len(lines), 2)
            for index, chart in enumerate(lines, 1):
                self.assertEqual(chart["series"][0]["values"], [index * factor * value for value in (.5, 1., 1.5)])
                for series in chart["series"]:
                    match = re.fullmatch(r"'源采样基线'!\$([A-Z]+)\$(\d+):\$([A-Z]+)\$(\d+)", series["values_ref"])
                    self.assertIsNotNone(match)
                    first_col, first, last_col, last = match.groups()
                    self.assertEqual(first_col, last_col)
                    actual = [book[source_name][1].get(first_col + str(row)) for row in range(int(first), int(last) + 1)]
                    self.assertEqual(actual, series["values"])
            source_links = [(position, target) for position, target in sheet.internal_links.items()
                            if "逐轮明细" in str(sheet.cells[position][1])]
            self.assertEqual(len(source_links), 2)
            self.assertTrue(all(target[0] == source_name and target[1] > 1 for _, target in source_links))
            self.assertNotIn("保留序号", book[name][1].values())

    def test_navigation_does_not_change_any_numeric_values_or_chart_values(self):
        report = two_case_report()
        _, _, actual = self.build(report, stem="with-navigation")
        _, _, before = self.build(report, navigation=False, stem="without-navigation")
        for name in renderer.SHEETS:
            numeric = lambda sheet: Counter((cell[0], cell[1], cell[3]) for cell in sheet.cells.values()
                if cell[0] in ("n", "b"))
            self.assertEqual(numeric(actual[name]), numeric(before[name]), name)
            self.assertEqual(len(actual[name].charts), len(before[name].charts))
            for current, previous in zip(actual[name].charts, before[name].charts):
                self.assertEqual(current["title"], previous["title"])
                self.assertEqual(current["categories"], previous["categories"])
                self.assertEqual([(series["name"], series["values"]) for series in current["series"]],
                                 [(series["name"], series["values"]) for series in previous["series"]])

    def test_very_long_unicode_case_labels_continue_without_losing_link_targets(self):
        writer = renderer.xlsx_writer
        book = writer.Workbook()
        sheet = book.add_sheet("方法'目录 & 详图")
        for name in (renderer.SHEETS[0], renderer.SHEETS[1]):
            book.add_sheet(name).write(1, 1, "真实目标")
        sheet.write(1, 1, "独立目录检验")
        long_name = "SYNTHETIC_'长名 & <🧪>" * 2400
        report = {"cases": [{"name": long_name, "config": {"bs": 8, "draft_tokens": 3}}]}
        row = renderer._navigation_start(sheet, report)
        renderer._navigation_case(sheet, row, report["cases"][0])
        sheet.write(row, 1, "case正文")
        renderer._table(sheet, row + 1, ["数据"], [[1.25]])
        sheet.add_line_chart("真实折线", ["hc-1"], [{"name": "耗时", "values": [1.25]}], (row + 4, 1, row + 15, 12))
        renderer._navigation_finish(sheet)
        path = book.save(self.root / "long-unicode.xlsx")
        cells = read_book(path)[0][sheet.name][1]
        pieces = [cell[1] for position, cell in sorted(sheet.cells.items())
                  if position[1] == 1 and position[0] < row and position in sheet.internal_links
                  and cell[1] != "跨case趋势与汇总 · 点击进入"
                  and sheet.internal_links[position] == (sheet.name, row, 1)]
        expected = "1. " + long_name + " · BS=8 · S=3"
        self.assertEqual("".join(pieces), expected)
        self.assertGreater(len(pieces), 1)
        self.assertTrue(all(len(str(value).encode("utf-16-le")) // 2 <= 32767 for value in cells.values()))
        for position in sheet.internal_links:
            self.assertIn(writer.cell_reference(*position), cells)

    def test_duplicate_case_names_retain_distinct_case_and_source_destinations(self):
        report = two_case_report(duplicate_names=True)
        out, _, sheets = self.build(report)
        for name in NAV_SHEETS:
            sheet = sheets[name]
            starts = [row for row, _ in sheet.navigation_cases]
            targets = []
            directory_targets = []
            for index in (1, 2):
                position = next(position for position, cell in sheet.cells.items()
                                if str(cell[1]).startswith(str(index) + ". " + report["cases"][0]["name"]))
                targets.append(sheet.internal_links[position])
                directory_targets.append(sheet.internal_links[position[0], 9])
            self.assertEqual(targets, [(name, start, 1) for start in starts])
            self.assertNotEqual(directory_targets[0], directory_targets[1])
        for name, factor in (("方法1估算", 1.), ("方法2估算", 10.)):
            lines = [chart for chart in sheets[name].charts if chart["title"].startswith("两项U逐轮耗时变化")]
            self.assertEqual([chart["series"][0]["values"] for chart in lines],
                             [[index * factor * value for value in (.5, 1., 1.5)] for index in (1, 2)])
            self.assertNotEqual(lines[0]["series"][0]["values_ref"], lines[1]["series"][0]["values_ref"])
        page = HTMLTargets(out.with_suffix(".html").read_text(encoding="utf-8"))
        self.assertEqual(len(page.ids), len(set(page.ids)))
        self.assertTrue(set(page.fragments).issubset(set(page.ids)))
        focus = [identity for identity in page.ids if identity.startswith("focus-source-")]
        self.assertEqual(len(focus), 4)
        for index, case in enumerate(report["cases"]):
            for rank in ("dp7", "dp3"):
                self.assertIn(renderer._focus_source_anchor(case, rank, index), focus)

    def test_html_directory_has_unique_live_fragments_for_every_chart_and_table(self):
        out, _, _ = self.build(two_case_report(), stem="html-navigation")
        document = out.with_suffix(".html").read_text(encoding="utf-8")
        page = HTMLTargets(document)
        self.assertEqual(len(page.ids), len(set(page.ids)))
        self.assertTrue(set(page.fragments).issubset(set(page.ids)))
        self.assertIn("report-chart-directory", page.ids)
        self.assertTrue(page.targets)
        for kind, identity in page.targets:
            self.assertIsNotNone(identity, kind)
            self.assertIn(identity, page.fragments, (kind, identity))
        self.assertIn('type="search"', document)
        self.assertIn("返回图表目录", document)

    def test_html_index_preserves_original_tables_svg_points_and_existing_anchors(self):
        fragment = '''<html><body><main><h1>报告</h1><h2>甲' &amp; &lt;图&gt;</h2>
<section id="original-source"><h3>源表 🚀</h3><table><tr><th>耗时</th></tr><tr><td>7.125</td></tr></table></section>
<svg viewBox="0 0 100 20"><title>真实曲线</title><polyline points="0,4.25 12,0 19,7.125"/><text>7.125</text></svg>
<div class="u-bars" role="img" aria-label="分段耗时"><div class="u-bar"><span>1.25 ms</span></div><div class="u-bar">2.5 ms</div></div>
<p><a href="#original-source">源数据</a></p></main></body></html>'''
        document = renderer._html_navigation(fragment)
        parser = HTMLTargets(document)
        self.assertTrue(set(parser.fragments).issubset(set(parser.ids)))
        self.assertEqual(len(parser.ids), len(set(parser.ids)))
        self.assertIn('<polyline points="0,4.25 12,0 19,7.125"/>', document)
        self.assertIn('<td>7.125</td>', document)
        self.assertIn('href="#original-source"', document)
        self.assertIn('id="original-source"', document)
        self.assertIn('<span>1.25 ms</span>', document)

    def test_html_generated_ids_do_not_collide_with_preexisting_ids_on_any_tag(self):
        fragment = '<html><body><main><h1>报告</h1><h2>目录</h2><p id="report-nav-target-1">保留</p><table id="report-nav-target-3"><tr><th>值</th></tr></table><svg><title>图</title></svg></main></body></html>'
        document = renderer._html_navigation(fragment)
        parsed = HTMLTargets(document)
        self.assertEqual(len(parsed.ids), len(set(parsed.ids)))
        self.assertTrue(set(parsed.fragments).issubset(set(parsed.ids)))
        self.assertIn('<p id="report-nav-target-1">保留</p>', document)

    def test_html_parser_keeps_nested_u_bars_open_until_matching_outer_div(self):
        parser = renderer._HTMLNavigationParser("")
        parser.feed('<div class="u-bars" role="img" aria-label="完整分段图"><div class="u-bar">第一项</div>')
        self.assertEqual([block["tag"] for block in parser.active], ["div"])
        parser.feed('<div class="u-bar">第二项</div></div>')
        parser.close()
        self.assertEqual(parser.active, [])
        self.assertEqual(parser.blocks[0]["label"], "完整分段图")


if __name__ == "__main__":
    unittest.main()
