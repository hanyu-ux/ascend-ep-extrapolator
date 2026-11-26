"""Indexed merge validation retains legacy semantics without whole-sheet scans."""

from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile

import ep128_xlsx_writer as writer


class LegacyWorksheet(writer.Worksheet):
    """The previous exact checks, retained here only as an independent oracle."""
    def _put(self, row, col, value, style, expression=None):
        self._check_position(row, col)
        if style not in writer.STYLE_IDS:
            raise ValueError("Unknown worksheet style: " + str(style))
        for r1, c1, r2, c2 in self.merges:
            if r1 <= row <= r2 and c1 <= col <= c2 and (row, col) != (r1, c1):
                raise ValueError("Cannot write into a non-anchor merged cell")
        value_type, serialized = writer._value_type(value)
        self.cells[row, col] = (value_type, serialized, writer.STYLE_IDS[style], expression)
        return self

    def merge(self, r1, c1, r2, c2, value, style="title"):
        self._check_position(r1, c1)
        self._check_position(r2, c2)
        if r1 > r2 or c1 > c2:
            raise ValueError("Merged range is reversed")
        if any(not (r2 < a or r1 > b or c2 < c or c1 > d)
               for a, c, b, d in self.merges):
            raise ValueError("Merged ranges must not overlap")
        if any(r1 <= row <= r2 and c1 <= col <= c2 and (row, col) != (r1, c1)
               for row, col in self.cells):
            raise ValueError("Merged range would hide existing cells")
        self.write(r1, c1, value, style)
        if (r1, c1) != (r2, c2):
            self.merges.append((r1, c1, r2, c2))
        return self


def sheet(kind=writer.Worksheet):
    return kind("时间账本", [12] * 12, True, 8, None)


def outcome(target, operation):
    name, args = operation
    try:
        getattr(target, name)(*args)
    except Exception as error:
        return type(error), str(error)
    return None


class MergeIndexTests(unittest.TestCase):
    def test_six_thousand_rows_never_scan_previous_merges_or_cell_keys(self):
        class MembershipOnlyCells(dict):
            visits = 0
            def __iter__(self):
                raise AssertionError("ordinary row merges must not scan all written cells")
            def __contains__(self, key):
                self.visits += 1
                return super().__contains__(key)
        target = sheet()
        target.cells = MembershipOnlyCells()
        target._ensure_merge_index()  # The initially empty cache may be built once.
        def no_full_list_scan(_values):
            raise AssertionError("ordinary row merges must not scan the full merge list")
        with patch.object(writer._MergeList, "__iter__", no_full_list_scan):
            for row in range(1, 6001):
                target.merge(row, 1, row, 6, "阶段")
                target.merge(row, 7, row, 8, 1.25, "number")
                for col in range(9, 13):
                    target.write(row, col, col)
        self.assertEqual(len(target.merges), 12000)
        self.assertEqual(len(target.cells), 36000)
        self.assertEqual(target.cells.visits, 36000)

    def test_random_order_and_boundary_operations_match_previous_checks(self):
        rng = random.Random(271828)
        operations = [
            ("write", (5, 5, "existing")),
            ("merge", (4, 4, 6, 6, "cannot hide")),
            ("merge", (1, 1, 1, 3, "normal")),
            ("write", (1, 1, "anchor overwrite")),
            ("write", (1, 2, "covered")),
            ("merge", (1, 1, 1, 1, "still overlaps")),
            ("merge", (writer.MAX_ROW, writer.MAX_COL-1, writer.MAX_ROW, writer.MAX_COL, "boundary")),
            ("write", (writer.MAX_ROW, writer.MAX_COL, "covered boundary")),
            ("merge", (0, 1, 1, 2, "out of range")),
            ("merge", (3, 3, 2, 3, "reversed")),
        ]
        for number in range(600):
            r1, c1 = rng.randrange(1, 100), rng.randrange(1, 14)
            if number % 3:
                operations.append(("merge", (r1, c1, r1+rng.randrange(0, 5), c1+rng.randrange(0, 5), str(number))))
            else:
                operations.append(("write", (r1, c1, str(number))))
        indexed, legacy = sheet(), sheet(LegacyWorksheet)
        for operation in operations:
            with self.subTest(operation=operation):
                self.assertEqual(outcome(indexed, operation), outcome(legacy, operation))
                self.assertEqual(indexed.cells, legacy.cells)
                self.assertEqual(indexed.merges, legacy.merges)

    def test_adjacent_same_row_spans_and_anchor_overwrite_remain_legal(self):
        target = sheet()
        target.merge(12, 5, 12, 8, "middle")
        target.merge(12, 1, 12, 4, "left")
        target.merge(12, 9, 12, 12, "right")
        target.write(12, 5, "new anchor")
        self.assertEqual(target.merges, [(12, 5, 12, 8), (12, 1, 12, 4), (12, 9, 12, 12)])
        with self.assertRaisesRegex(ValueError, "must not overlap"):
            target.merge(11, 4, 13, 5, "crosses")
        with self.assertRaisesRegex(ValueError, "non-anchor"):
            target.write(12, 6, "inside")

    def test_small_rectangles_detect_manually_written_and_blank_cells(self):
        target = sheet()
        target.cells[50, 3] = ("blank", None, 0, None)
        with self.assertRaisesRegex(ValueError, "hide existing cells"):
            target.merge(49, 1, 51, 5, "hides blank")
        self.assertEqual(target.merges, [])
        target.merge(50, 3, 50, 4, "existing anchor is allowed")
        self.assertEqual(target.cells[50, 3][1], "existing anchor is allowed")

    def test_tall_ranges_use_bounded_storage_and_all_conflicts_are_checked(self):
        target = sheet()
        target.merge(1, 2, writer.MAX_ROW, 2, "tall")
        self.assertEqual(target._merge_rows, {})
        self.assertEqual(len(target._tall_merges), 1)
        for row in (1, 65, 500000, writer.MAX_ROW):
            if row != 1:
                with self.subTest(row=row), self.assertRaisesRegex(ValueError, "non-anchor"):
                    target.write(row, 2, "covered")
            with self.subTest(row=row), self.assertRaisesRegex(ValueError, "must not overlap"):
                target.merge(row, 1, row, 3, "cross tall")
        target.merge(100, 5, 1000, 6, "second tall")
        self.assertEqual(len(target._tall_merges), 2)
        target.write(1001, 2+5, "outside")

    def test_large_rectangle_fallback_detects_existing_far_cell(self):
        target = sheet()
        target.write(1000, 1000, "far")
        with self.assertRaisesRegex(ValueError, "hide existing cells"):
            target.merge(1, 1, 2000, 2000, "large")
        self.assertEqual(target.merges, [])
        target.merge(1, 1, 100, 100, "large empty")
        with self.assertRaisesRegex(ValueError, "must not overlap"):
            target.merge(60, 60, 150, 150, "tall cross")

    def test_external_merge_mutations_invalidate_same_length_and_tall_indexes(self):
        target = sheet()
        target.merges.append((10, 1, 10, 3))
        with self.assertRaisesRegex(ValueError, "non-anchor"):
            target.write(10, 2, "covered")
        target.merges[0] = (20, 1, 20, 3)
        target.write(10, 2, "old range removed")
        with self.assertRaisesRegex(ValueError, "non-anchor"):
            target.write(20, 2, "replacement")
        target.merges[:] = [(30, 1, 300, 3)]
        with self.assertRaisesRegex(ValueError, "non-anchor"):
            target.write(200, 2, "tall replacement")
        target.merges.clear()
        target.write(200, 2, "cleared")
        target.merges = [(40, 1, 40, 4)]
        target.merges += [(50, 1, 50, 4)]
        target.merges.reverse()
        target.merges.sort()
        with self.assertRaisesRegex(ValueError, "non-anchor"):
            target.write(50, 2, "augmented")
        target.merges.pop()
        target.write(50, 2, "popped")
        del target.merges[:]
        target.write(40, 2, "deleted")

    def test_short_and_tall_thresholds_and_out_of_order_ranges_match_legacy(self):
        indexed, legacy = sheet(), sheet(LegacyWorksheet)
        for operation in [
            ("merge", (300, 20, 364, 23, "65 rows")),
            ("merge", (100, 2, 163, 5, "64 rows")),
            ("merge", (160, 6, 165, 7, "adjacent cols")),
            ("merge", (120, 3, 121, 4, "overlap short")),
            ("merge", (200, 1, 500, 2, "tall empty")),
            ("merge", (400, 20, 403, 24, "after tall")),
            ("write", (363, 22, "inside tall")),
            ("write", (163, 4, "inside short")),
            ("write", (164, 4, "outside short")),
        ]:
            with self.subTest(operation=operation):
                self.assertEqual(outcome(indexed, operation), outcome(legacy, operation))
        self.assertEqual(indexed.cells, legacy.cells)
        self.assertEqual(indexed.merges, legacy.merges)

    def test_written_worksheet_xml_is_unchanged_for_identical_operations(self):
        operations = [("merge", (1, 1, 2, 12, "标题", "title")),
                      ("merge", (3, 1, 3, 12, "说明", "note")),
                      ("merge", (5, 1, 5, 4, "阶段")),
                      ("merge", (5, 5, 5, 8, 1.25, "number")),
                      ("write", (5, 9, "MATCHED")),
                      ("merge", (8, 2, 80, 3, "长范围"))]
        books = []
        for kind in (writer.Worksheet, LegacyWorksheet):
            book = writer.Workbook()
            target = sheet(kind)
            book.sheets.append(target)
            for operation in operations:
                self.assertIsNone(outcome(target, operation))
            books.append(book)
        with tempfile.TemporaryDirectory() as directory:
            paths = [book.save(Path(directory) / (str(i) + ".xlsx")) for i, book in enumerate(books)]
            with ZipFile(paths[0]) as indexed, ZipFile(paths[1]) as legacy:
                self.assertEqual(indexed.read("xl/worksheets/sheet1.xml"), legacy.read("xl/worksheets/sheet1.xml"))


if __name__ == "__main__":
    unittest.main()
