"""Editable combination charts and cell-backed percentage/time point labels."""
import importlib.util
import posixpath
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import ZipFile

from ep128_xlsx_writer import ART, CHART, CONTENT, PACKAGE_REL, Workbook

NS = {"a": ART, "c": CHART, "p": PACKAGE_REL, "ct": CONTENT}


class ComboChartTests(unittest.TestCase):
    def workbook(self, secondary_axis=False):
        book = Workbook()
        chart_sheet = book.add_sheet("额外图表统计")
        data = book.add_sheet("图表 数据")
        categories = ["2", "4", "8", "16"]
        fractions = [.25, None, 0, .125]
        absolute = [1.5, None, 0, 2.25]
        labels = ["25.0% · 1.500 ms", None, "0.0% · 0.000 ms", "12.5% · 2.250 ms"]
        for row, (category, fraction, duration, label) in enumerate(
                zip(categories, fractions, absolute, labels), 2):
            for column, value in enumerate((category, fraction, duration, label), 1):
                data.write(row, column, value)
        data.write(1, 2, "轮次剩余占比")
        data.write(1, 3, "轮次剩余时间")
        columns = [{"name": "轮次剩余时间" if secondary_axis else "轮次剩余占比",
                    "values": absolute if secondary_axis else fractions,
                    "values_ref": "'图表 数据'!$C$2:$C$5" if secondary_axis else "'图表 数据'!$B$2:$B$5",
                    "name_ref": "'图表 数据'!$C$1" if secondary_axis else "'图表 数据'!$B$1",
                    "color": "167D8D", "data_labels": secondary_axis}]
        lines = [{"name": "占比与时间", "values": fractions, "values_ref": "'图表 数据'!$B$2:$B$5",
                  "color": "D59440", "data_labels": True if secondary_axis else labels}]
        if not secondary_axis:
            lines[0]["data_labels_ref"] = "='图表 数据'!$D$2:$D$5"
        chart_sheet.add_combo_chart("方法1 · S=9", categories, columns, lines, (3, 1, 24, 12),
                                    categories_ref="='图表 数据'!$A$2:$A$5",
                                    secondary_axis=secondary_axis,
                                    column_axis_title="剩余时间（ms）" if secondary_axis else "占完整周期（%）",
                                    column_value_format='0.000" ms"' if secondary_axis else "0.0%")
        return book

    def chart_xml(self, book):
        with tempfile.TemporaryDirectory() as directory, \
                ZipFile(book.save(Path(directory) / "combo.xlsx")) as archive:
            return ET.fromstring(archive.read("xl/charts/chart1.xml"))

    def test_shared_axes_equal_data_put_points_on_column_tops(self):
        chart = self.chart_xml(self.workbook())
        bar, line = chart.find(".//c:barChart", NS), chart.find(".//c:lineChart", NS)
        self.assertEqual(bar.find("c:barDir", NS).get("val"), "col")
        self.assertEqual(bar.find("c:grouping", NS).get("val"), "clustered")
        self.assertEqual(line.find("c:grouping", NS).get("val"), "standard")
        self.assertEqual([axis.get("val") for axis in bar.findall("c:axId", NS)], ["100", "200"])
        self.assertEqual([axis.get("val") for axis in line.findall("c:axId", NS)], ["100", "200"])
        self.assertEqual(len(chart.findall(".//c:catAx", NS)), 1)
        self.assertEqual(len(chart.findall(".//c:valAx", NS)), 1)
        bar_series, line_series = bar.find("c:ser", NS), line.find("c:ser", NS)
        self.assertEqual(bar_series.find("c:val/c:numRef/c:f", NS).text,
                         line_series.find("c:val/c:numRef/c:f", NS).text)
        for item in (bar_series, line_series):
            self.assertEqual(item.find("c:cat/c:strRef/c:f", NS).text, "'图表 数据'!$A$2:$A$5")
            self.assertEqual(item.find("c:val/c:numRef/c:numCache/c:ptCount", NS).get("val"), "4")
            points = {int(point.get("idx")): float(point.find("c:v", NS).text)
                      for point in item.findall("c:val/c:numRef/c:numCache/c:pt", NS)}
            self.assertEqual(points, {0: .25, 2: 0, 3: .125})
        self.assertIsNone(bar_series.find("c:dLbls", NS))
        self.assertEqual(line_series.find("c:marker/c:symbol", NS).get("val"), "circle")
        self.assertEqual(chart.find(".//c:valAx/c:numFmt", NS).get("formatCode"), "0.0%")
        self.assertEqual(chart.find(".//c:dispBlanksAs", NS).get("val"), "gap")
        self.assertEqual([item.find("c:idx", NS).get("val") for item in (bar_series, line_series)], ["0", "1"])

    def test_editable_custom_labels_keep_both_units_and_sparse_indices(self):
        chart = self.chart_xml(self.workbook())
        labels = chart.findall(".//c:lineChart/c:ser/c:dLbls/c:dLbl", NS)
        self.assertEqual([label.find("c:idx", NS).get("val") for label in labels], ["0", "2", "3"])
        self.assertEqual([label.find("c:tx/c:strRef/c:f", NS).text for label in labels],
                         ["'图表 数据'!$D$2", "'图表 数据'!$D$4", "'图表 数据'!$D$5"])
        self.assertEqual([label.find("c:tx/c:strRef/c:strCache/c:pt/c:v", NS).text for label in labels],
                         ["25.0% · 1.500 ms", "0.0% · 0.000 ms", "12.5% · 2.250 ms"])
        self.assertEqual(chart.find(".//c:lineChart/c:ser/c:dLbls/c:showVal", NS).get("val"), "0")

    def test_secondary_axes_have_distinct_ids_and_formatted_value_labels(self):
        chart = self.chart_xml(self.workbook(secondary_axis=True))
        self.assertEqual([axis.get("val") for axis in chart.findall(".//c:barChart/c:axId", NS)], ["100", "200"])
        self.assertEqual([axis.get("val") for axis in chart.findall(".//c:lineChart/c:axId", NS)], ["100", "300"])
        axes = chart.findall(".//c:valAx", NS)
        self.assertEqual([axis.find("c:axId", NS).get("val") for axis in axes], ["200", "300"])
        self.assertEqual([axis.find("c:axPos", NS).get("val") for axis in axes], ["l", "r"])
        self.assertEqual([axis.find("c:crossAx", NS).get("val") for axis in axes], ["100", "100"])
        self.assertEqual(axes[1].find("c:crosses", NS).get("val"), "max")
        for path, expected_format in ((".//c:barChart", '0.000" ms"'), (".//c:lineChart", "0.0%")):
            labels = chart.find(path + "/c:ser/c:dLbls", NS)
            self.assertEqual(labels.find("c:numFmt", NS).get("formatCode"), expected_format)
            self.assertEqual(labels.find("c:showVal", NS).get("val"), "1")

    def test_percentage_bar_labels_link_horizontal_cell_ranges_and_stay_outside_bars(self):
        book = Workbook()
        sheet = book.add_sheet("分解")
        sheet.add_bar_chart("分解", ["one", "two"],
                            [{"name": "stage", "values": [.25, .5],
                              "data_labels": ["25.0% · 1.000 ms", "50.0% · 2.000 ms"],
                              "data_labels_ref": "'图表 数据'!D2:E2"}],
                            (1, 1, 12, 8), stacked=False,
                            x_axis_title="占完整周期（%）", value_format="0.0%")
        chart = self.chart_xml(book)
        self.assertEqual(chart.find(".//c:barDir", NS).get("val"), "bar")
        self.assertEqual(chart.find(".//c:valAx/c:numFmt", NS).get("formatCode"), "0.0%")
        self.assertEqual(chart.find(".//c:val/c:numLit/c:formatCode", NS).text, "0.0%")
        self.assertEqual(chart.find(".//c:dLbls/c:dLblPos", NS).get("val"), "outEnd")
        self.assertEqual([label.find("c:tx/c:strRef/c:f", NS).text
                          for label in chart.findall(".//c:dLbl", NS)],
                         ["'图表 数据'!$D$2", "'图表 数据'!$E$2"])
        sheet.charts[0]["stacked"] = True
        self.assertEqual(self.chart_xml(book).find(".//c:dLbls/c:dLblPos", NS).get("val"), "ctr")

    def test_literal_labels_escape_text_and_legacy_chart_defaults_stay_plain(self):
        book = Workbook()
        sheet = book.add_sheet("test")
        sheet.add_bar_chart("legacy", ["x"], [{"name": "stage", "values": [1]}], (1, 1, 12, 8))
        plain = self.chart_xml(book)
        self.assertIsNone(plain.find(".//c:dLbls", NS))
        self.assertEqual(plain.find(".//c:valAx/c:numFmt", NS).get("formatCode"), "0.000")
        book = Workbook()
        book.add_sheet("literal").add_combo_chart("literal", ["x"],
            [{"name": "column", "values": [.25], "data_labels": False}],
            [{"name": "line", "values": [.25], "data_labels": ["25.0% < 1.500 ms & test"]}],
            (1, 1, 12, 8), secondary_axis=False, column_value_format="0.0%")
        self.assertEqual(self.chart_xml(book).find(".//c:dLbl/c:tx/c:rich/a:p/a:r/a:t", NS).text,
                         "25.0% < 1.500 ms & test")

    def test_package_relationships_and_chart_content_types_resolve(self):
        with tempfile.TemporaryDirectory() as directory, \
                ZipFile(self.workbook().save(Path(directory) / "package.xlsx")) as archive:
            self.assertIsNone(archive.testzip())
            names = set(archive.namelist())
            for name in names:
                if name.endswith((".xml", ".rels")):
                    root = ET.fromstring(archive.read(name))
                    if name.endswith(".rels"):
                        source = posixpath.dirname(name).replace("/_rels", "")
                        if source == "_rels":
                            source = ""
                        for relationship in root:
                            target = posixpath.normpath(posixpath.join(source, relationship.get("Target")))
                            self.assertIn(target, names)
            content = ET.fromstring(archive.read("[Content_Types].xml"))
            overrides = {item.get("PartName") for item in content.findall("ct:Override", NS)}
            self.assertIn("/xl/charts/chart1.xml", overrides)
            self.assertIn("/xl/drawings/drawing1.xml", overrides)

    def test_invalid_combo_or_label_input_is_rejected_atomically(self):
        base = {"title": "test", "categories": ["one", "two"],
                "columns": [{"name": "column", "values": [1, 2]}],
                "lines": [{"name": "line", "values": [.1, .2]}], "anchor": (1, 1, 12, 8)}
        invalid = [{"columns": []}, {"lines": []}, {"secondary_axis": 1},
                   {"title": ""}, {"categories": []}, {"anchor": (10, 1, 1, 8)}]
        for value in (True, "unknown", float("nan"), float("inf")):
            invalid.append({"lines": [{"name": "line", "values": [value, .2]}]})
        for updates in ({"data_labels": ["one"]}, {"data_labels": "one"},
                        {"data_labels_ref": "'Data'!A1:A2"},
                        {"data_labels": ["one", "two"], "data_labels_ref": "'Data'!A1:B2"},
                        {"data_labels": ["one", "two"], "data_labels_ref": "'Data'!A1:A3"},
                        {"data_labels": ["one", "two"], "data_labels_ref": "'Data'!A2:A1"},
                        {"data_labels": ["one", "two"], "data_labels_ref": "'Data'!A1\n:A2"}):
            invalid.append({"lines": [{"name": "line", "values": [.1, .2], **updates}]})
        for changes in invalid:
            with self.subTest(changes=changes):
                sheet = Workbook().add_sheet("invalid")
                with self.assertRaises(ValueError):
                    sheet.add_combo_chart(**{**base, **changes})
                self.assertEqual(sheet.charts, [])

    @unittest.skipUnless(importlib.util.find_spec("openpyxl"), "optional reader validation")
    def test_openpyxl_reads_combined_charts_and_sparse_label_indices(self):
        import openpyxl
        with tempfile.TemporaryDirectory() as directory:
            path = self.workbook().save(Path(directory) / "reader.xlsx")
            book = openpyxl.load_workbook(path)
            combo = book["额外图表统计"]._charts[0]
            self.assertEqual(len(combo._charts), 2)
            column, line = combo._charts
            self.assertEqual(column.type, "col")
            self.assertEqual(column.y_axis.axId, line.y_axis.axId)
            # openpyxl's label model omits c:tx, whose cell refs and caches are
            # checked directly above. It still reads the actual combo package.
            self.assertEqual([label.idx for label in line.ser[0].dLbls.dLbl], [0, 2, 3])
            self.assertFalse(line.ser[0].dLbls.showVal)
            book.close()


if __name__ == "__main__":
    unittest.main()
