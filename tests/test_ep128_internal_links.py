"""Safe workbook-local hyperlinks survive later table writes and merged titles."""
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET
from zipfile import ZipFile

import ep128_xlsx_writer as writer


NS = {"s": writer.MAIN}


class InternalLinkTests(unittest.TestCase):
    def archive(self, book):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        path = book.save(Path(temporary.name) / "中文 workbook.xlsx")
        archive = ZipFile(path)
        self.addCleanup(archive.close)
        return archive

    def sheet(self, archive, index=1):
        return ET.fromstring(archive.read("xl/worksheets/sheet" + str(index) + ".xml"))

    def test_unicode_and_apostrophes_are_escaped_in_pure_internal_locations(self):
        book = writer.Workbook()
        index = book.add_sheet("图表目录")
        target = book.add_sheet("方法'详图 & <x>")
        target.write(12, 3, "目标位置")
        self.assertIs(index.add_internal_link(2, 1, "吞吐 & <图> →", target.name, "$c$12"), index)
        archive = self.archive(book)
        sheet = self.sheet(archive)
        link = sheet.find("s:hyperlinks/s:hyperlink", NS)
        self.assertEqual(link.attrib, {"ref": "A2", "location": "'方法''详图 & <x>'!C12",
                                      "display": "吞吐 & <图> →"})
        self.assertEqual(sheet.find("s:sheetData/s:row/s:c/s:is/s:t", NS).text, "吞吐 & <图> →")
        self.assertFalse(any(name.endswith("sheet1.xml.rels") for name in archive.namelist()))
        self.assertIsNone(sheet.find(".//s:f", NS))

    def test_forward_registration_and_later_writes_keep_link_and_current_display(self):
        book = writer.Workbook()
        index = book.add_sheet("目录")
        index.add_internal_link(1, 1, "临时目录名", "方法1估算", "A9")
        index.write(1, 1, "最终目录名", "header")
        target = book.add_sheet("方法1估算")
        target.merge(9, 1, 9, 12, "图表标题")
        link = self.sheet(self.archive(book)).find("s:hyperlinks/s:hyperlink", NS)
        self.assertEqual(link.get("display"), "最终目录名")
        self.assertEqual(link.get("location"), "'方法1估算'!A9")

    def test_merged_source_anchor_is_linkable_and_merge_rewrite_keeps_link(self):
        book = writer.Workbook()
        sheet = book.add_sheet("目录")
        sheet.write(9, 1, "目标")
        sheet.add_internal_link(1, 1, "旧名", sheet.name, "A9")
        sheet.merge(1, 1, 1, 5, "合并目录", "body")
        root = self.sheet(self.archive(book))
        self.assertEqual(root.find("s:mergeCells/s:mergeCell", NS).get("ref"), "A1:E1")
        link = root.find("s:hyperlinks/s:hyperlink", NS)
        self.assertEqual(link.get("ref"), "A1")
        self.assertEqual(link.get("display"), "合并目录")
        with self.assertRaisesRegex(ValueError, "non-anchor"):
            sheet.add_internal_link(1, 2, "藏在合并内", sheet.name, "A9")
        self.assertNotIn((1, 2), sheet.internal_links)

    def test_unwritten_sheet_cell_and_merged_nonanchor_fail_before_replacing_file(self):
        for missing in ("sheet", "cell", "merged_nonanchor", "source"):
            with self.subTest(missing=missing), tempfile.TemporaryDirectory() as temporary:
                book = writer.Workbook()
                sheet = book.add_sheet("目录")
                sheet.add_internal_link(1, 1, "跳转", "目标", "B2")
                if missing != "sheet":
                    target = book.add_sheet("目标")
                    if missing == "merged_nonanchor":
                        target.merge(2, 1, 2, 3, "仅首格可跳")
                    elif missing == "source":
                        target.write(2, 2, "目标")
                        sheet.cells.pop((1, 1))
                path = Path(temporary) / "existing.xlsx"
                path.write_bytes(b"KEEP EXISTING WORKBOOK")
                with self.assertRaisesRegex(ValueError, "Internal hyperlink"):
                    book.save(path)
                self.assertEqual(path.read_bytes(), b"KEEP EXISTING WORKBOOK")
                self.assertEqual(list(Path(temporary).iterdir()), [path])

    def test_target_is_exactly_one_valid_cell_and_never_a_url_formula_or_range(self):
        book = writer.Workbook()
        sheet = book.add_sheet("目录")
        for address in (None, True, 1, "", "A0", "A01", "A-1", "1A", "A1:B2", "Sheet!A1",
                        "https://example.com/A1", "file:///A1", "=A1", "#A1", "A1;B2", "A1\nB2",
                        "XFE1", "A1048577", "AAAA1", "[book.xlsx]A1"):
            with self.subTest(address=address), self.assertRaises(ValueError):
                sheet.add_internal_link(1, 1, "跳转", "目标", address)
        self.assertEqual(sheet.cells, {})
        self.assertEqual(sheet.internal_links, {})

    def test_source_coordinates_and_labels_follow_existing_strict_cell_rules(self):
        book = writer.Workbook()
        sheet = book.add_sheet("目录")
        for row, col in ((True, 1), (1., 1), (0, 1), (1, 0), (writer.MAX_ROW + 1, 1),
                         (1, writer.MAX_COL + 1), (1, False)):
            with self.subTest(row=row, col=col), self.assertRaises(ValueError):
                sheet.add_internal_link(row, col, "跳转", "目标", "A1")
        for label in (None, True, 2, "", " \t ", "bad\x00label"):
            with self.subTest(label=label), self.assertRaises(ValueError):
                sheet.add_internal_link(1, 1, label, "目标", "A1")
        self.assertEqual(sheet.cells, {})

    def test_target_sheet_cannot_contain_external_reference_or_invalid_xml(self):
        book = writer.Workbook()
        sheet = book.add_sheet("目录")
        for name in (None, 1, "", "x" * 32, "https://example.com", "file://target", "[other.xlsx]Data",
                     "Data!A1/A2", "A:B", "'quoted'", "bad\x00sheet"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                sheet.add_internal_link(1, 1, "跳转", name, "A1")
        self.assertEqual(sheet.internal_links, {})

    def test_label_looking_like_formula_is_stored_as_text_not_an_external_action(self):
        book = writer.Workbook()
        sheet = book.add_sheet("目录")
        sheet.write(2, 1, "目标")
        label = '=HYPERLINK("https://example.com","not executed")'
        sheet.add_internal_link(1, 1, label, sheet.name, "A2")
        archive = self.archive(book)
        root = self.sheet(archive)
        self.assertIsNone(root.find(".//s:f", NS))
        self.assertEqual(root.find("s:hyperlinks/s:hyperlink", NS).get("display"), label)
        self.assertFalse(any(b'TargetMode="External"' in archive.read(name)
                             for name in archive.namelist() if name.endswith(".rels")))

    def test_repeated_registration_updates_one_link_and_formula_rewrite_keeps_it(self):
        book = writer.Workbook()
        sheet = book.add_sheet("目录")
        sheet.write(2, 1, "目标2").write(3, 1, "目标3")
        sheet.add_internal_link(1, 1, "第二行", sheet.name, "A2")
        sheet.add_internal_link(1, 1, "第三行", sheet.name, "A3")
        sheet.formula(1, 1, "1+1", 2)
        root = self.sheet(self.archive(book))
        links = root.findall("s:hyperlinks/s:hyperlink", NS)
        self.assertEqual(len(links), 1)
        self.assertEqual(links[0].get("location"), "'目录'!A3")
        self.assertEqual(links[0].get("display"), "2")
        self.assertEqual(root.find("s:sheetData/s:row/s:c/s:f", NS).text, "1+1")

    def test_maximum_coordinates_and_explicit_blank_target_are_valid(self):
        book = writer.Workbook()
        index = book.add_sheet("Index")
        target = book.add_sheet("Target")
        target.write(writer.MAX_ROW, writer.MAX_COL, None)
        index.add_internal_link(1, 1, "Last cell", "target", "$xFd$1048576")
        root = self.sheet(self.archive(book))
        self.assertEqual(root.find("s:hyperlinks/s:hyperlink", NS).get("location"), "'target'!XFD1048576")

    def test_links_and_chart_drawing_coexist_without_hyperlink_relationships(self):
        book = writer.Workbook()
        sheet = book.add_sheet("目录")
        sheet.merge(1, 1, 1, 2, "目录")
        sheet.write(3, 1, "数据")
        sheet.add_internal_link(1, 1, "数据和图", sheet.name, "A3", style="title")
        sheet.add_line_chart("图表", ["step1"], [{"name": "time", "values": [1]}], (5, 1, 15, 6))
        archive = self.archive(book)
        root = self.sheet(archive)
        tags = [element.tag.rsplit("}", 1)[-1] for element in root]
        self.assertLess(tags.index("mergeCells"), tags.index("hyperlinks"))
        self.assertLess(tags.index("hyperlinks"), tags.index("printOptions"))
        self.assertLess(tags.index("hyperlinks"), tags.index("drawing"))
        rels = ET.fromstring(archive.read("xl/worksheets/_rels/sheet1.xml.rels"))
        self.assertEqual(len(rels), 1)
        self.assertTrue(rels[0].get("Type").endswith("/drawing"))
        self.assertIsNone(root.find("s:hyperlinks/s:hyperlink", NS).get("{" + writer.REL + "}id"))

    def test_link_font_is_visible_and_existing_style_indices_remain_stable(self):
        book = writer.Workbook()
        sheet = book.add_sheet("目录")
        sheet.write(2, 1, "目标")
        sheet.add_internal_link(1, 1, "跳转", sheet.name, "A2")
        archive = self.archive(book)
        styles = ET.fromstring(archive.read("xl/styles.xml"))
        self.assertEqual(writer.STYLE_IDS["title"], 1)
        self.assertEqual(writer.STYLE_IDS["hero_integer"], 23)
        self.assertEqual(writer.STYLE_IDS["link"], 24)
        link = styles.find("s:cellXfs", NS)[writer.STYLE_IDS["link"]]
        font = styles.find("s:fonts", NS)[int(link.get("fontId"))]
        self.assertIsNotNone(font.find("s:u", NS))
        self.assertEqual(font.find("s:color", NS).get("rgb"), "FF0563C1")


if __name__ == "__main__":
    unittest.main()
