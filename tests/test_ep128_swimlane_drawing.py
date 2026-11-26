"""Native vector panels preserve precise swimlane geometry beside Excel charts."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET
from zipfile import ZipFile

import ep128_xlsx_writer as writer


NS = {"a": writer.ART, "xdr": writer.DRAWING, "r": writer.REL,
      "m": writer.MAIN, "p": writer.PACKAGE_REL, "ct": writer.CONTENT}


def elements():
    return [
        {"type": "rect", "x": 10.25, "y": 20, "width": 130.125, "height": 22,
         "fill": "#168B8D", "stroke": None, "opacity": .8, "description": "主模型 HC 范围"},
        {"type": "line", "x1": 145.75, "y1": 15, "x2": 145.75, "y2": 110,
         "stroke": "D2675B", "stroke_width": 1.25, "dash": True, "arrow": "end"},
        {"type": "text", "x": 5, "y": 120, "width": 285, "height": 60,
         "text": '=HYPERLINK("https://example.invalid/") <&>\n块表 / slot',
         "font_size": 10.5, "bold": True, "color": "16324F", "align": "center"},
    ]


class NativeSwimlaneDrawingTests(unittest.TestCase):
    def contents(self, book):
        with tempfile.TemporaryDirectory() as directory:
            path = book.save(Path(directory) / "native.xlsx")
            with ZipFile(path) as archive:
                return {name: archive.read(name) for name in archive.namelist()}

    def drawing(self, files, number=1):
        return ET.fromstring(files["xl/drawings/drawing" + str(number) + ".xml"])

    def test_vector_only_sheet_has_self_contained_native_shapes_and_plain_escaped_text(self):
        book = writer.Workbook()
        sheet = book.add_sheet("额外图表统计", widths=[12] * 24)
        supplied = elements()
        original = deepcopy(supplied)
        sheet.add_vector_drawing(supplied, (300, 200), (4, 1, 14, 25), description="CPU / NPU 时间线 <&>")
        self.assertEqual(supplied, original)
        supplied[0]["x"] = 250
        files = self.contents(book)
        self.assertFalse(any(name.startswith("xl/charts/") for name in files))
        self.assertFalse(any(name.startswith("xl/media/") for name in files))
        self.assertNotIn("xl/drawings/_rels/drawing1.xml.rels", files)
        drawing = self.drawing(files)
        group = drawing.find("xdr:twoCellAnchor/xdr:grpSp", NS)
        self.assertIsNotNone(group)
        self.assertEqual(len(group.findall("xdr:sp", NS)), 3)
        description = group.find("xdr:nvGrpSpPr/xdr:cNvPr", NS).get("descr")
        self.assertEqual(description, "CPU / NPU 时间线 <&>")
        text = [node.text or "" for node in group.findall(".//a:t", NS)]
        self.assertEqual(text, ['=HYPERLINK("https://example.invalid/") <&>', "块表 / slot"])
        self.assertIn(b"&lt;&amp;&gt;", files["xl/drawings/drawing1.xml"])
        self.assertEqual(group.find("xdr:sp/xdr:spPr/a:xfrm/a:off", NS).get("x"), str(round(10.25 * 9525)))
        sheet_xml = ET.fromstring(files["xl/worksheets/sheet1.xml"])
        self.assertEqual(sheet_xml.find("m:drawing", NS).get("{" + writer.REL + "}id"), "rIdDrawing")
        self.assertEqual(sheet_xml.findall(".//m:f", NS), [])
        self.assertEqual(sheet_xml.find("m:dimension", NS).get("ref"), "A1:Y14")
        for path, body in files.items():
            if path.endswith(".rels"):
                self.assertNotIn(b'TargetMode="External"', body)

    def test_fractional_coordinates_dash_alpha_fonts_and_endpoint_arrow_survive(self):
        book = writer.Workbook()
        sheet = book.add_sheet("泳道")
        sheet.add_vector_drawing(elements(), (300, 200), (1, 1, 11, 9))
        drawing = self.drawing(self.contents(book))
        shapes = drawing.findall(".//xdr:grpSp/xdr:sp", NS)
        rect, line, text = shapes
        self.assertEqual(rect.find("xdr:spPr/a:xfrm/a:ext", NS).get("cx"), str(round(130.125 * 9525)))
        self.assertEqual(rect.find("xdr:spPr/a:solidFill/a:srgbClr/a:alpha", NS).get("val"), "80000")
        self.assertIsNotNone(rect.find("xdr:spPr/a:ln/a:noFill", NS))
        self.assertEqual(line.find("xdr:spPr/a:xfrm/a:off", NS).get("x"), str(round(145.75 * 9525)))
        self.assertEqual(line.find("xdr:spPr/a:ln", NS).get("w"), str(round(1.25 * 9525)))
        self.assertEqual(line.find("xdr:spPr/a:ln/a:prstDash", NS).get("val"), "dash")
        self.assertEqual(line.find("xdr:spPr/a:ln/a:headEnd", NS).get("type"), "none")
        self.assertEqual(line.find("xdr:spPr/a:ln/a:tailEnd", NS).get("type"), "triangle")
        self.assertEqual(text.find("xdr:txBody/a:p/a:pPr", NS).get("algn"), "ctr")
        style = text.find("xdr:txBody/a:p/a:r/a:rPr", NS)
        self.assertEqual(style.get("sz"), "1050")
        self.assertEqual(style.get("b"), "1")
        self.assertEqual(style.find("a:ea", NS).get("typeface"), "微软雅黑")

    def test_two_charts_and_twelve_panels_share_one_drawing_with_unique_shape_ids(self):
        book = writer.Workbook()
        sheet = book.add_sheet("额外图表统计", widths=[12] * 24)
        sheet.add_bar_chart("原横条", ["A"], [{"name": "耗时", "values": [1]}], (1, 1, 10, 10))
        sheet.add_line_chart("原折线", ["A"], [{"name": "占比", "values": [.5]}], (11, 1, 20, 10))
        for panel in range(12):
            labels = [{"type": "text", "x": 5, "y": index * 15, "width": 280, "height": 14,
                       "text": f"case {panel} 函数 {index}", "font_size": 9} for index in range(20)]
            sheet.add_vector_drawing(labels, (300, 300), (21 + panel * 16, 1, 36 + panel * 16, 25))
        files = self.contents(book)
        drawing = self.drawing(files)
        self.assertEqual(len(drawing.findall("xdr:twoCellAnchor/xdr:graphicFrame", NS)), 2)
        self.assertEqual(len(drawing.findall("xdr:twoCellAnchor/xdr:grpSp", NS)), 12)
        ids = [node.get("id") for node in drawing.findall(".//xdr:cNvPr", NS)]
        self.assertEqual(len(ids), 2 + 12 * 21)
        self.assertEqual(len(ids), len(set(ids)))
        references = [node.get("{" + writer.REL + "}id") for node in drawing.findall(".//{*}chart")]
        self.assertEqual(references, ["rId1", "rId2"])
        rels = ET.fromstring(files["xl/drawings/_rels/drawing1.xml.rels"])
        self.assertEqual([(node.get("Id"), node.get("Target")) for node in rels],
                         [("rId1", "../charts/chart1.xml"), ("rId2", "../charts/chart2.xml")])
        self.assertEqual(len(drawing.findall(".//a:t", NS)), 240)
        content = ET.fromstring(files["[Content_Types].xml"])
        self.assertEqual(len([node for node in content if node.get("PartName") == "/xl/drawings/drawing1.xml"]), 1)

    def test_cell_frame_uses_actual_column_widths_and_row_heights(self):
        book = writer.Workbook()
        sheet = book.add_sheet("泳道", widths=[10, 12, 8, 14])
        sheet.row_height(1, 30).row_height(3, 15).row_height(4, 25)
        sheet.add_vector_drawing([{"type": "rect", "x": 0, "y": 0, "width": 1000, "height": 300,
                                   "fill": "FFFFFF"}], (1000, 300), (3, 2, 7, 5))
        drawing = self.drawing(self.contents(book))
        transform = drawing.find(".//xdr:grpSp/xdr:grpSpPr/a:xfrm", NS)
        self.assertEqual(transform.find("a:off", NS).attrib, {"x": "714375", "y": "635000"})
        self.assertEqual(transform.find("a:ext", NS).attrib, {"cx": "2409825", "cy": "1016000"})
        self.assertEqual(transform.find("a:chExt", NS).attrib, {"cx": "9525000", "cy": "2857500"})

    def test_nonzero_view_box_and_reversed_line_preserve_direction(self):
        book = writer.Workbook()
        sheet = book.add_sheet("泳道")
        sheet.add_vector_drawing([{"type": "line", "x1": 50, "y1": 50, "x2": -10, "y2": -5,
                                   "dash": "dot", "arrow": "both"}], (-20, -10, 100, 80), (1, 1, 9, 8))
        drawing = self.drawing(self.contents(book))
        group_transform = drawing.find(".//xdr:grpSp/xdr:grpSpPr/a:xfrm", NS)
        self.assertEqual(group_transform.find("a:chOff", NS).attrib, {"x": "-190500", "y": "-95250"})
        transform = drawing.find(".//xdr:sp/xdr:spPr/a:xfrm", NS)
        self.assertEqual(transform.get("flipH"), "1")
        self.assertEqual(transform.get("flipV"), "1")
        self.assertEqual(transform.find("a:off", NS).attrib, {"x": "-95250", "y": "-47625"})
        self.assertEqual([node.get("type") for node in drawing.findall(".//a:headEnd", NS)], ["triangle"])
        self.assertEqual([node.get("type") for node in drawing.findall(".//a:tailEnd", NS)], ["triangle"])

    def test_invalid_geometry_styles_text_and_out_of_bounds_are_rejected_without_mutation(self):
        sheet = writer.Workbook().add_sheet("泳道")
        good = {"type": "text", "x": 0, "y": 0, "width": 100, "height": 20, "text": "函数"}
        failures = [dict(good, x=float("nan")), dict(good, y=True), dict(good, width=0),
                    dict(good, height=-1), dict(good, x=-1), dict(good, width=101),
                    dict(good, color="red"), dict(good, stroke="ABC<DE"), dict(good, opacity=1.1),
                    dict(good, dash="dashed"), dict(good, align="justify"), dict(good, valign="middle"),
                    dict(good, font_size=float("inf")), dict(good, text="bad\x00text"),
                    {"type": "image", "x": 0, "y": 0},
                    {"type": "line", "x1": 1, "y1": 1, "x2": 1, "y2": 1},
                    {"type": "line", "x1": 0, "y1": 0, "x2": 20, "y2": 20, "arrow": "triangle"}]
        for value in failures:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    sheet.add_vector_drawing([value], (100, 100), (1, 1, 10, 10))
                self.assertEqual(sheet.vector_drawings, [])
        for box in ((0, 100), (100,), (0, 0, 100, -5)):
            with self.subTest(box=box), self.assertRaises(ValueError):
                sheet.add_vector_drawing([good], box, (1, 1, 10, 10))
        for anchor in ((1, 1, 1, 10), (1, 1, 10, 1), (0, 1, 10, 10), (1, 1, 10.5, 10)):
            with self.subTest(anchor=anchor), self.assertRaises(ValueError):
                sheet.add_vector_drawing([good], (100, 100), anchor)


if __name__ == "__main__":
    unittest.main()
