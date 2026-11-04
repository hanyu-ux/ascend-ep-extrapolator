#!/usr/bin/env python3
"""Native chart package checks; no Office or third-party writer is required."""
import importlib.util
from pathlib import Path
import posixpath
import tempfile
import unittest
import xml.etree.ElementTree as ET
from zipfile import ZipFile

from ep128_xlsx_writer import Workbook, CHART, DRAWING, MAIN, PACKAGE_REL, CONTENT, ART


NS = {"c": CHART, "xdr": DRAWING, "s": MAIN, "a": ART,
      "p": PACKAGE_REL, "ct": CONTENT}


class NativeBarChartTests(unittest.TestCase):
    def workbook(self):
        book = Workbook()
        sheet = book.add_sheet("估算图", widths=[20] * 8)
        data = book.add_sheet("图表 数据")
        categories = ["方法一", "方法二"]
        for row, label in enumerate(categories, 2):
            data.write(row, 1, label)
        data.write(1, 2, "主模型")
        data.write(2, 2, 10.25)
        data.write(3, 2, 9.5)
        data.write(1, 3, "其余轮次开销")
        data.write(2, 3, 2.75)
        data.write(3, 3, 2.5)
        sheet.add_bar_chart("设备完整轮次构成", categories,
                            [{"name": "主模型", "name_ref": "'图表 数据'!$B$1",
                              "values": [10.25, 9.5], "values_ref": "'图表 数据'!$B$2:$B$3",
                              "color": "167D8D"},
                             {"name": "其余轮次开销", "values": [2.75, 2.5],
                              "values_ref": "'图表 数据'!$C$2:$C$3", "color": "D59440"}],
                            (3, 1, 20, 9), categories_ref="'图表 数据'!$A$2:$A$3")
        return book

    def test_range_references_caches_and_style(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.workbook().save(Path(directory) / "chart.xlsx")
            with ZipFile(path) as archive:
                root = ET.fromstring(archive.read("xl/charts/chart1.xml"))
                bar = root.find(".//c:barChart", NS)
                self.assertEqual(bar.find("c:barDir", NS).get("val"), "bar")
                self.assertEqual(bar.find("c:grouping", NS).get("val"), "stacked")
                series = bar.findall("c:ser", NS)
                self.assertEqual(len(series), 2)
                self.assertEqual(series[0].find("c:val/c:numRef/c:f", NS).text,
                                 "'图表 数据'!$B$2:$B$3")
                self.assertEqual([float(v.text) for v in series[0].findall(
                    "c:val/c:numRef/c:numCache/c:pt/c:v", NS)], [10.25, 9.5])
                self.assertEqual([v.text for v in series[0].findall(
                    "c:cat/c:strRef/c:strCache/c:pt/c:v", NS)], ["方法一", "方法二"])
                self.assertEqual(series[0].find("c:tx/c:strRef/c:f", NS).text,
                                 "'图表 数据'!$B$1")
                art = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main"}
                self.assertEqual(series[0].find("c:spPr", NS).find(
                    "a:solidFill/a:srgbClr", art).get("val"), "167D8D")
                drawing = ET.fromstring(archive.read("xl/drawings/drawing1.xml"))
                self.assertEqual(drawing.find("xdr:twoCellAnchor/xdr:from/xdr:row", NS).text, "2")
                self.assertEqual(drawing.find("xdr:twoCellAnchor/xdr:to/xdr:col", NS).text, "8")
                sheet = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
                self.assertIsNotNone(sheet.find("s:drawing", NS))
                self.assertEqual(sheet.find("s:dimension", NS).get("ref"), "A1:I20")

    def test_all_relationships_and_content_parts_resolve(self):
        book = self.workbook()
        book.sheets[1].add_bar_chart("对比", ["一", "二"],
                                    [{"name": "草稿", "values": [1, 2]}], (5, 1, 18, 6),
                                    stacked=False)
        with tempfile.TemporaryDirectory() as directory:
            with ZipFile(book.save(Path(directory) / "many.xlsx")) as archive:
                self.assertIsNone(archive.testzip())
                names = set(archive.namelist())
                self.assertIn("xl/charts/chart2.xml", names)
                for name in names:
                    if name.endswith(".xml") or name.endswith(".rels"):
                        root = ET.fromstring(archive.read(name))
                        if name.endswith(".rels"):
                            source_dir = posixpath.dirname(name).replace("/_rels", "")
                            if source_dir == "_rels":
                                source_dir = ""
                            for relationship in root:
                                target = posixpath.normpath(posixpath.join(source_dir, relationship.get("Target")))
                                self.assertIn(target, names, (name, target))
                content = ET.fromstring(archive.read("[Content_Types].xml"))
                overrides = {item.get("PartName") for item in content.findall("ct:Override", NS)}
                for name in ("xl/charts/chart1.xml", "xl/charts/chart2.xml",
                             "xl/drawings/drawing1.xml", "xl/drawings/drawing2.xml"):
                    self.assertIn("/" + name, overrides)

    def test_clustered_literal_data_and_negative_values_preserved(self):
        book = Workbook()
        sheet = book.add_sheet("test")
        sheet.add_bar_chart("delta", ["a", "b"],
                            [{"name": "change", "values": [-1.25, 2]}],
                            (1, 1, 15, 8), stacked=False)
        with tempfile.TemporaryDirectory() as directory:
            with ZipFile(book.save(Path(directory) / "negative.xlsx")) as archive:
                chart = ET.fromstring(archive.read("xl/charts/chart1.xml"))
                self.assertEqual(chart.find(".//c:grouping", NS).get("val"), "clustered")
                values = chart.findall(".//c:val/c:numLit/c:pt/c:v", NS)
                self.assertEqual([float(value.text) for value in values], [-1.25, 2])

    def test_missing_and_nonfinite_values_are_rejected_not_zeroed(self):
        for unknown in (None, "待导入", float("nan"), float("inf"), True):
            with self.subTest(unknown=unknown):
                book = Workbook()
                sheet = book.add_sheet("test")
                with self.assertRaisesRegex(ValueError, "finite numbers"):
                    sheet.add_bar_chart("test", ["one"],
                                        [{"name": "a", "values": [unknown]}], (1, 1, 10, 8))
                self.assertEqual(sheet.charts, [])

    def test_chart_validation(self):
        valid = {"title": "test", "categories": ["one"],
                 "series": [{"name": "a", "values": [1]}], "anchor": (1, 1, 10, 8)}
        for changes in ({"title": ""}, {"categories": []}, {"series": []},
                        {"anchor": (10, 1, 1, 8)}, {"anchor": (0, 1, 10, 8)},
                        {"series": [{"name": "a", "values": [1, 2]}]},
                        {"series": [{"name": "a", "values": [1], "color": "red"}]},
                        {"categories_ref": ""}):
            with self.subTest(changes=changes):
                sheet = Workbook().add_sheet("test")
                with self.assertRaises(ValueError):
                    sheet.add_bar_chart(**{**valid, **changes})

    def test_plain_workbook_keeps_no_drawing_parts(self):
        book = Workbook()
        book.add_sheet("plain").write(1, 1, 5)
        with tempfile.TemporaryDirectory() as directory:
            with ZipFile(book.save(Path(directory) / "plain.xlsx")) as archive:
                self.assertFalse(any("drawing" in name or "chart" in name for name in archive.namelist()))
                self.assertIsNone(ET.fromstring(archive.read("xl/worksheets/sheet1.xml")).find("s:drawing", NS))

    @unittest.skipUnless(importlib.util.find_spec("openpyxl"), "openpyxl is only an optional reader for this test")
    def test_openpyxl_parses_editable_chart_and_anchors(self):
        import openpyxl
        with tempfile.TemporaryDirectory() as directory:
            path = self.workbook().save(Path(directory) / "reader.xlsx")
            book = openpyxl.load_workbook(path)
            chart = book["估算图"]._charts[0]
            self.assertEqual(chart.type, "bar")
            self.assertEqual(chart.grouping, "stacked")
            self.assertEqual(len(chart.ser), 2)
            self.assertEqual(chart.ser[0].val.numRef.f, "'图表 数据'!$B$2:$B$3")
            self.assertEqual(chart.ser[0].val.numRef.numCache.pt[0].v, 10.25)
            self.assertEqual(chart.ser[0].graphicalProperties.solidFill.srgbClr, "167D8D")
            self.assertEqual(chart.anchor._from.row, 2)
            self.assertEqual(chart.anchor.to.col, 8)
            book.close()


class NativeLineChartTests(unittest.TestCase):
    def workbook(self):
        book = Workbook()
        sheet = book.add_sheet("BS趋势", widths=[20] * 8)
        data = book.add_sheet("趋势 数据")
        categories = ["4", "8", "16", "32"]
        first, second = [10, None, 0, 15.25], [None, 9.5, 20, None]
        data.write(1, 2, "方法一")
        data.write(1, 3, "方法二")
        for row, (label, one, two) in enumerate(zip(categories, first, second), 2):
            data.write(row, 1, int(label))
            data.write(row, 2, one)
            data.write(row, 3, two)
        sheet.add_line_chart("单卡吞吐随BS变化", categories,
            [{"name": "方法一", "name_ref": "='趋势 数据'!$B$1", "values": first,
              "values_ref": "='趋势 数据'!$B$2:$B$5", "color": "167d8d"},
             {"name": "方法二", "values": second, "values_ref": "'趋势 数据'!$C$2:$C$5"}],
            (6, 1, 24, 9), categories_ref="='趋势 数据'!$A$2:$A$5",
            x_axis_title="BS", y_axis_title="token/s/card")
        return book

    def test_line_references_sparse_caches_markers_and_axis_orientation(self):
        with tempfile.TemporaryDirectory() as directory:
            with ZipFile(self.workbook().save(Path(directory) / "line.xlsx")) as archive:
                root = ET.fromstring(archive.read("xl/charts/chart1.xml"))
                line = root.find(".//c:lineChart", NS)
                self.assertIsNotNone(line)
                self.assertIsNone(root.find(".//c:barChart", NS))
                self.assertEqual(line.find("c:grouping", NS).get("val"), "standard")
                self.assertEqual(root.find(".//c:dispBlanksAs", NS).get("val"), "gap")
                series = line.findall("c:ser", NS)
                self.assertEqual(len(series), 2)
                for index, item in enumerate(series):
                    self.assertEqual(item.find("c:idx", NS).get("val"), str(index))
                    self.assertEqual(item.find("c:order", NS).get("val"), str(index))
                    self.assertEqual(item.find("c:cat/c:strRef/c:f", NS).text,
                                     "'趋势 数据'!$A$2:$A$5")
                    self.assertEqual([point.text for point in item.findall(
                        "c:cat/c:strRef/c:strCache/c:pt/c:v", NS)], ["4", "8", "16", "32"])
                    self.assertEqual(item.find("c:marker/c:symbol", NS).get("val"), "circle")
                    self.assertEqual(item.find("c:marker/c:size", NS).get("val"), "5")
                    self.assertEqual(item.find("c:smooth", NS).get("val"), "0")
                self.assertEqual(series[0].find("c:tx/c:strRef/c:f", NS).text, "'趋势 数据'!$B$1")
                self.assertEqual(series[0].find("c:val/c:numRef/c:f", NS).text,
                                 "'趋势 数据'!$B$2:$B$5")
                for item, expected in zip(series, ({0: 10.0, 2: 0.0, 3: 15.25}, {1: 9.5, 2: 20.0})):
                    cache = item.find("c:val/c:numRef/c:numCache", NS)
                    self.assertEqual(cache.find("c:ptCount", NS).get("val"), "4")
                    actual = {int(point.get("idx")): float(point.find("c:v", NS).text)
                              for point in cache.findall("c:pt", NS)}
                    self.assertEqual(actual, expected)
                self.assertEqual(series[0].find("c:spPr/a:ln/a:solidFill/a:srgbClr", NS).get("val"), "167D8D")
                self.assertEqual(series[0].find("c:marker/c:spPr/a:solidFill/a:srgbClr", NS).get("val"), "167D8D")
                cat_axis, val_axis = root.find(".//c:catAx", NS), root.find(".//c:valAx", NS)
                self.assertEqual(cat_axis.find("c:axPos", NS).get("val"), "b")
                self.assertEqual(val_axis.find("c:axPos", NS).get("val"), "l")
                self.assertEqual(cat_axis.find("c:scaling/c:orientation", NS).get("val"), "minMax")
                self.assertEqual(cat_axis.find("c:crossAx", NS).get("val"), val_axis.find("c:axId", NS).get("val"))
                self.assertEqual(val_axis.find("c:crossAx", NS).get("val"), cat_axis.find("c:axId", NS).get("val"))
                self.assertEqual([item.get("val") for item in line.findall("c:axId", NS)], ["100", "200"])
                self.assertEqual(cat_axis.find("c:title/c:tx/c:rich/a:p/a:r/a:t", NS).text, "BS")
                self.assertEqual(val_axis.find("c:title/c:tx/c:rich/a:p/a:r/a:t", NS).text, "token/s/card")
                sheet = ET.fromstring(archive.read("xl/worksheets/sheet2.xml"))
                self.assertEqual(sheet.find(".//s:c[@r='A2']/s:v", NS).text, "4")
                self.assertIsNone(sheet.find(".//s:c[@r='B3']/s:v", NS))
                self.assertEqual(sheet.find(".//s:c[@r='B4']/s:v", NS).text, "0")

    def test_literal_values_preserve_zero_negative_and_fully_unknown_series(self):
        book = Workbook()
        book.add_sheet("literal").add_line_chart("delta", ["one", "two", "three", "four", "five"],
            [{"name": "delta", "values": [None, -1.25, None, 0, None]},
             {"name": "unknown", "values": [None] * 5}], (1, 1, 15, 8))
        with tempfile.TemporaryDirectory() as directory:
            with ZipFile(book.save(Path(directory) / "literal.xlsx")) as archive:
                chart = ET.fromstring(archive.read("xl/charts/chart1.xml"))
                caches = chart.findall(".//c:ser/c:val/c:numLit", NS)
                self.assertEqual(len(caches), 2)
                for cache in caches:
                    self.assertEqual(cache.find("c:ptCount", NS).get("val"), "5")
                self.assertEqual([(point.get("idx"), float(point.find("c:v", NS).text))
                                  for point in caches[0].findall("c:pt", NS)], [("1", -1.25), ("3", 0.0)])
                self.assertEqual(caches[1].findall("c:pt", NS), [])
                self.assertEqual(chart.find(".//c:dispBlanksAs", NS).get("val"), "gap")

    def test_line_validation_is_atomic_and_bar_unknown_contract_is_unchanged(self):
        valid = {"title": "trend", "categories": ["one"],
                 "series": [{"name": "method", "values": [1]}], "anchor": (1, 1, 10, 8)}
        invalid = [{"title": ""}, {"categories": []}, {"categories": "one"}, {"series": []},
                   {"anchor": (10, 1, 1, 8)}, {"anchor": (0, 1, 10, 8)},
                   {"series": [{"name": "method", "values": [1, None]}]},
                   {"series": [{"name": "method", "values": [1], "color": "red"}]},
                   {"categories_ref": ""}, {"series": [{"name": "method", "values": [1], "values_ref": "a\nb"}]}]
        for value in ("待估算", float("nan"), float("inf"), True):
            invalid.append({"series": [{"name": "method", "values": [value]}]})
        for changes in invalid:
            with self.subTest(changes=changes):
                sheet = Workbook().add_sheet("validation")
                with self.assertRaises(ValueError):
                    sheet.add_line_chart(**{**valid, **changes})
                self.assertEqual(sheet.charts, [])
        sheet = Workbook().add_sheet("bar")
        with self.assertRaisesRegex(ValueError, "finite numbers"):
            sheet.add_bar_chart("bar", ["one"], [{"name": "method", "values": [None]}], (1, 1, 10, 8))
        self.assertEqual(sheet.charts, [])

    def test_mixed_bar_line_chart_package_and_relationships_resolve(self):
        book = NativeBarChartTests().workbook()
        book.sheets[0].add_line_chart("BS比较", ["4", "8"],
            [{"name": "method", "values": [2, None]}], (23, 1, 40, 9))
        with tempfile.TemporaryDirectory() as directory:
            with ZipFile(book.save(Path(directory) / "mixed.xlsx")) as archive:
                self.assertIsNone(archive.testzip())
                self.assertIsNotNone(ET.fromstring(archive.read("xl/charts/chart1.xml")).find(".//c:barChart", NS))
                self.assertIsNotNone(ET.fromstring(archive.read("xl/charts/chart2.xml")).find(".//c:lineChart", NS))
                names = set(archive.namelist())
                for name in names:
                    if name.endswith(".rels"):
                        source_dir = posixpath.dirname(name).replace("/_rels", "")
                        if source_dir == "_rels":
                            source_dir = ""
                        for relationship in ET.fromstring(archive.read(name)):
                            target = posixpath.normpath(posixpath.join(source_dir, relationship.get("Target")))
                            self.assertIn(target, names)
                drawing = ET.fromstring(archive.read("xl/drawings/drawing1.xml"))
                self.assertEqual(len(drawing.findall("xdr:twoCellAnchor", NS)), 2)
                sheet = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
                self.assertEqual(sheet.find("s:dimension", NS).get("ref"), "A1:I40")

    @unittest.skipUnless(importlib.util.find_spec("openpyxl"), "openpyxl is only an optional reader for this test")
    def test_openpyxl_reads_editable_sparse_line_series_and_markers(self):
        import openpyxl
        with tempfile.TemporaryDirectory() as directory:
            path = self.workbook().save(Path(directory) / "reader.xlsx")
            book = openpyxl.load_workbook(path)
            chart = book["BS趋势"]._charts[0]
            self.assertEqual(type(chart).__name__, "LineChart")
            self.assertEqual(chart.grouping, "standard")
            self.assertEqual(len(chart.ser), 2)
            self.assertEqual(chart.ser[0].val.numRef.f, "'趋势 数据'!$B$2:$B$5")
            self.assertEqual([(point.idx, point.v) for point in chart.ser[0].val.numRef.numCache.pt],
                             [(0, 10.0), (2, 0.0), (3, 15.25)])
            self.assertEqual(chart.ser[0].marker.symbol, "circle")
            self.assertEqual(chart.ser[0].marker.size, 5)
            self.assertEqual(chart.x_axis.axPos, "b")
            self.assertEqual(chart.y_axis.axPos, "l")
            self.assertEqual(chart.anchor._from.row, 5)
            self.assertEqual(chart.anchor.to.col, 8)
            self.assertEqual(book["趋势 数据"]["A2"].value, 4)
            book.close()


if __name__ == "__main__":
    unittest.main()
