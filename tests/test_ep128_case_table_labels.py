"""Per-table BS/S context is visible without changing synthetic report evidence."""
from collections import Counter
from contextlib import ExitStack
import copy
from html.parser import HTMLParser
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import profile_ep128 as core
from test_ep128_report_navigation import two_case_report
from test_ep128_v2_excel import renderer


LABELS = ("BS8 · S9", "BS16 · S15")
CASE_SHEETS = ("估算总览", "源采样基线", "层与组件实测", "时间账本",
               "方法1估算", "方法2估算", "方法3估算", "公式与假设", "复核记录")


def label_report():
    report = two_case_report(duplicate_names=True)
    for case, bs, draft in zip(report["cases"], (8, 16), (9, 15)):
        # A deliberately misleading, shared name must never supply the values.
        case["name"] = "SYNTHETIC_SHARED_BS99_S99_<case>&"
        case["config"].update(bs=bs, draft_tokens=draft)
        case["config"].pop("num_speculative_tokens", None)
    report["failures"] = [dict(case="SYNTHETIC_GLOBAL_FAILURE", error="SYNTHETIC_ONLY")]
    return report


class TableCaptions(HTMLParser):
    """Read captions and original table cells without counting navigation text."""
    def __init__(self, document):
        super().__init__(convert_charrefs=True)
        self.tables, self.stack = [], []
        self.in_caption, self.in_cell = False, False
        self.feed(document)
        self.close()

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            table = dict(captions=[], headers=[], values=[])
            self.tables.append(table)
            self.stack.append(table)
        elif self.stack and tag == "caption":
            self.in_caption = True
            self.stack[-1]["captions"].append(dict(text="", attrs=dict(attrs)))
        elif self.stack and tag in ("th", "td"):
            self.in_cell = tag
            self.stack[-1]["headers" if tag == "th" else "values"].append("")

    def handle_data(self, data):
        if not self.stack:
            return
        if self.in_caption:
            self.stack[-1]["captions"][-1]["text"] += data
        elif self.in_cell:
            key = "headers" if self.in_cell == "th" else "values"
            self.stack[-1][key][-1] += data

    def handle_endtag(self, tag):
        if tag == "caption":
            self.in_caption = False
        elif tag in ("th", "td"):
            self.in_cell = False
        elif tag == "table" and self.stack:
            self.stack.pop()


class CaseTableLabelTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="synthetic-case-table-labels-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def build(self, report, *, suppress_labels=False, stem="labels"):
        before = copy.deepcopy(report)
        saved, tables = [], []
        native_save, native_table = renderer.Workbook.save, renderer._table

        def save(book, path):
            saved.append(book)
            return native_save(book, path)

        def table(sheet, row, headers, records, **kwargs):
            label = getattr(sheet, "table_case_label", "")
            output = native_table(sheet, row, headers, records, **kwargs)
            first = kwargs["spans"][0][0] if kwargs.get("spans") else 1
            tables.append(dict(sheet=sheet, row=row, label=label,
                               headers=list(headers), column=first))
            return output

        out = self.root / (stem + ".xlsx")
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(renderer.Workbook, "save", new=save))
            stack.enter_context(mock.patch.object(renderer, "_table", new=table))
            stack.enter_context(mock.patch.object(core, "read_capture",
                side_effect=AssertionError("Labels must not reread profiling data")))
            if suppress_labels:
                stack.enter_context(mock.patch.object(renderer, "_case_table_label", return_value=""))
            renderer.build_workbook(out, report=self.root / "synthetic-core.json", report_data=report)
        self.assertEqual(report, before)
        self.assertEqual(len(saved), 1)
        return saved[0], tables, out.with_suffix(".html").read_text(encoding="utf-8")

    def test_label_reads_declared_values_and_supported_s_alias_without_guessing_name(self):
        self.assertEqual(renderer._case_table_label(
            dict(name="BS99_S99", config=dict(bs=8, draft_tokens=9))), LABELS[0])
        self.assertEqual(renderer._case_table_label(
            dict(name="BS99_S99", config=dict(bs=16, num_speculative_tokens=15))), LABELS[1])
        self.assertEqual(renderer._case_table_label(
            dict(name="BS8_S9", config={})), "BS未声明 · S未声明")
        self.assertEqual(renderer._case_table_label(dict(name="BS8_S9")), "BS未声明 · S未声明")
        self.assertEqual(renderer._case_table_label(
            dict(config=dict(bs=True, draft_tokens=True))), "BS未声明 · S未声明")
        self.assertEqual(renderer._case_table_label(
            dict(config=dict(bs=0, draft_tokens=-1))), "BS未声明 · S未声明")
        self.assertEqual(renderer._case_table_label(
            dict(config=dict(bs=8, draft_tokens=9, num_speculative_tokens=15))), "BS8 · S未声明")
        self.assertEqual(renderer._case_table_label(
            dict(config=dict(bs=8, draft_tokens=0))), "BS8 · S0")

    def test_context_can_be_set_and_cleared_without_changing_case(self):
        book = renderer.Workbook()
        sheet = book.add_sheet("SYNTHETIC")
        case = dict(name="not_a_parameter_source", config=dict(bs=8, draft_tokens=9))
        before = copy.deepcopy(case)
        renderer._set_table_case(sheet, case)
        self.assertEqual(sheet.table_case_label, LABELS[0])
        renderer._set_table_case(sheet, None)
        self.assertFalse(sheet.table_case_label)
        self.assertEqual(case, before)

    def test_table_keeps_positions_return_value_data_and_caller_header_list(self):
        book = renderer.Workbook()
        source, labeled = book.add_sheet("source"), book.add_sheet("labeled")
        headers = ["类别", "耗时(ms)"]
        records = [["SYNTHETIC_A", 1.25], ["SYNTHETIC_B", 0.]]
        written_source, written_labeled = [], []
        spans = ((1, 4), (5, 12))
        renderer._set_table_case(labeled, dict(config=dict(bs=8, draft_tokens=9)))
        source_end = renderer._table(source, 7, headers, records, spans=spans, record_rows=written_source)
        labeled_end = renderer._table(labeled, 7, headers, records, spans=spans, record_rows=written_labeled)
        self.assertEqual(headers, ["类别", "耗时(ms)"])
        self.assertEqual(source_end, labeled_end)
        self.assertEqual(written_source, written_labeled)
        self.assertEqual(labeled.cells[7, 1][1], "类别\n" + LABELS[0])
        self.assertEqual(labeled.cells[7, 5], source.cells[7, 5])
        self.assertEqual(set(labeled.cells), set(source.cells))
        for position in source.cells:
            if position != (7, 1):
                self.assertEqual(labeled.cells[position], source.cells[position], position)

    def test_workbook_labels_all_case_tables_and_does_not_tag_global_tables(self):
        report = label_report()
        book, tables, _ = self.build(report)
        by_sheet = {sheet.name: sheet for sheet in book.sheets}
        seen = {}
        for item in tables:
            sheet, row, label = item["sheet"], item["row"], item["label"]
            actual = sheet.cells[row, item["column"]][1]
            expected = str(item["headers"][0]) + ("\n" + label if label else "")
            self.assertEqual(actual, expected, (sheet.name, row))
            if label:
                self.assertIn(label, LABELS)
                seen.setdefault(sheet.name, set()).add(label)
            elif "navigation_cases" in vars(sheet) and sheet.navigation_cases:
                self.assertLess(row, sheet.navigation_cases[0][0], (sheet.name, row))
        for name in CASE_SHEETS:
            self.assertEqual(seen.get(name), set(LABELS), name)
        for name in ("估算总览", "源采样基线", "方法1估算", "方法2估算", "方法3估算"):
            sheet = by_sheet[name]
            starts = [row for row, _ in sheet.navigation_cases]
            self.assertEqual(len(starts), 2)
            for item in (item for item in tables if item["sheet"] is sheet):
                expected = next((LABELS[index] for index in reversed(range(2))
                                 if starts[index] <= item["row"]), "")
                self.assertEqual(item["label"], expected, (name, item["row"]))
        # The final failures table is report-wide, after the last case section.
        review = by_sheet[renderer.SHEETS[8]]
        failure_rows = [position[0] for position, cell in review.cells.items()
                        if cell[1] == "SYNTHETIC_GLOBAL_FAILURE"]
        self.assertEqual(len(failure_rows), 1)
        final = next(item for item in tables if item["sheet"] is review
                     and item["row"] == failure_rows[0] - 1)
        self.assertFalse(final["label"])
        self.assertTrue(any(not item["label"] for item in tables), "Global summaries must remain report-wide")

    def test_labels_preserve_every_numeric_cell_coordinate_and_chart_reference(self):
        report = label_report()
        actual, _, _ = self.build(report, stem="with-labels")
        before, _, _ = self.build(report, suppress_labels=True, stem="without-labels")
        for current, original in zip(actual.sheets, before.sheets):
            self.assertEqual(current.name, original.name)
            numeric = lambda sheet: {position: cell for position, cell in sheet.cells.items()
                                    if cell[0] in ("n", "b")}
            self.assertEqual(numeric(current), numeric(original), current.name)
            self.assertEqual(len(current.charts), len(original.charts), current.name)
            for chart, baseline in zip(current.charts, original.charts):
                for key in ("anchor", "categories", "categories_ref"):
                    self.assertEqual(chart.get(key), baseline.get(key), (current.name, key))
                self.assertEqual(chart["series"], baseline["series"], current.name)
            self.assertEqual(current.result_cells if hasattr(current, "result_cells") else None,
                             original.result_cells if hasattr(original, "result_cells") else None)

    def test_html_helper_adds_escaped_caption_to_each_table_without_changing_cells(self):
        fragment = ('<section><table><thead><tr><th>原列</th></tr></thead>'
                    '<tbody><tr><td>1.25 &amp; 原数据</td></tr></tbody></table>'
                    '<details><table><tr><th>第二列</th><td>0</td></tr></table></details></section>')
        label = 'BS8 · S9 <&" source'
        with mock.patch.object(renderer, "_case_table_label", return_value=label):
            result = renderer._html_case_tables(fragment, dict(config=dict(bs=8, draft_tokens=9)))
        before, after = TableCaptions(fragment), TableCaptions(result)
        self.assertEqual(len(after.tables), 2)
        self.assertNotIn(label, result)
        self.assertIn('&lt;&amp;&quot;', result)
        for original, table in zip(before.tables, after.tables):
            self.assertEqual(table["headers"], original["headers"])
            self.assertEqual(table["values"], original["values"])
            self.assertEqual(len(table["captions"]), 1)
            self.assertEqual(table["captions"][0]["text"], label)
            self.assertIn("case-table-caption", table["captions"][0]["attrs"].get("class", "").split())

    def test_html_main_case_and_focus_source_tables_keep_distinct_same_name_case_labels(self):
        _, _, document = self.build(label_report())
        parsed = TableCaptions(document)
        counts = Counter()
        for table in parsed.tables:
            self.assertLessEqual(len(table["captions"]), 1)
            if table["captions"]:
                label = table["captions"][0]["text"]
                self.assertIn(label, LABELS)
                counts[label] += 1
        self.assertGreater(counts[LABELS[0]], 20)
        self.assertEqual(counts[LABELS[0]], counts[LABELS[1]])
        self.assertTrue(any(not table["captions"] for table in parsed.tables), "Global tables need no single-case caption")
        focus = [table for table in parsed.tables if table["headers"]
                 and table["headers"][0] == "保留序号"]
        self.assertEqual(len(focus), 4)
        self.assertEqual(Counter(table["captions"][0]["text"] for table in focus),
                         Counter({LABELS[0]: 2, LABELS[1]: 2}))


if __name__ == "__main__":
    unittest.main()
