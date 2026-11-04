"""Large source previews and Excel text limits; synthetic data only."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from test_ep128_v2_excel import fixture, read_book, renderer, NS


class RendererScaleTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="synthetic-ep128-scale-")
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)

    def build(self, data, **kwargs):
        origin = self.base / "model_architecture_report.json"
        out = self.base / "leadership.xlsx"
        result = renderer.build_workbook(out, report=origin, report_data=data, **kwargs)
        self.assertEqual(result, out)
        return out, origin, read_book(out)[0]

    def test_large_reference_table_is_bounded_without_changing_core_results(self):
        data = fixture()
        row = {"bs": 4, "ep32_send": 0.123, "scope": "SYNTHETIC_ONLY"}
        data["communication_reference"] = {"rows": [row] * 40000}
        before = copy.deepcopy(data['cases'])
        out, origin, book = self.build(data)
        self.assertEqual(tuple(book), renderer.SHEETS)
        for name, (root, _) in book.items():
            rows = [int(row.attrib["r"]) for row in root.findall(".//s:sheetData/s:row", NS)]
            self.assertLess(max(rows, default=0), 10000, name)
        text = "\n".join(str(value) for value in book["通信参考原表"][1].values())
        self.assertIn("显示前 1000 / 40000 条", text)
        self.assertIn("汇总值来自全量核心统计", text)
        self.assertIn("不作代表性抽样", text)
        self.assertIn(str(origin), text)
        self.assertIn("communication_reference.rows[]", text)
        self.assertEqual(data['cases'], before)
        self.assertEqual(len(data['communication_reference']['rows']), 40000)
        self.assertIn(519.4321, book['估算总览'][1].values())
        self.assertTrue(out.with_suffix(".html").is_file())

    def test_custom_detail_limit_applies_to_source_partitions_and_reference_rows(self):
        data = fixture()
        data['communication_reference'] = {'rows': [{'row': number} for number in range(12)]}
        for method in data['cases'][0]['projection']['layer_mean_methods']:
            method['per_rank'].extend(dict(method['per_rank'][0], rank='extra' + str(number)) for number in range(10))
        before = copy.deepcopy(data)
        _, _, book = self.build(data, detail_rows=3)
        for name in ('时间账本', '通信参考原表'):
            text = "\n".join(str(value) for value in book[name][1].values())
            self.assertIn("显示前 3 / 12 条", text)
        self.assertEqual(data, before)
        self.assertIn(519.4321, book['估算总览'][1].values())

    def test_astral_text_is_split_by_utf16_units_and_preserves_all_evidence(self):
        data = fixture()
        evidence = "BEGIN_SCALE_PROOF:" + "\U0001f600" * 20000 + ":END_SCALE_PROOF"
        self.assertLess(len(evidence), 32767)
        self.assertGreater(len(evidence.encode("utf-16-le")) // 2, 32767)
        data['cases'][0]['projection']['layer_mean_methods'][0]['assumptions'] = [evidence]
        _, _, book = self.build(data)
        parts = [value for value in book["公式与假设"][1].values() if isinstance(value, str)
                 and ("\U0001f600" in value or "BEGIN_SCALE_PROOF:" in value or ":END_SCALE_PROOF" in value)]
        self.assertGreater(len(parts), 1)
        self.assertEqual("".join(parts), evidence)
        for _, cells in book.values():
            for value in cells.values():
                if isinstance(value, str):
                    self.assertLessEqual(len(value.encode("utf-16-le")) // 2, 32767)
        self.assertEqual(data['cases'][0]['projection']['layer_mean_methods'][0]['assumptions'], [evidence])

    def test_detail_limit_rejects_out_of_range_or_noninteger_values(self):
        for limit in (0, -1, 10001, True, 1.5):
            with self.subTest(limit=limit), self.assertRaises((ValueError, TypeError)):
                self.build(fixture(), detail_rows=limit)

    def test_cli_reuses_saved_core_json_without_changing_its_bytes(self):
        data = fixture()
        data['communication_reference'] = {'rows': [{'row': number} for number in range(12)]}
        source = self.base / 'saved core report.json'
        source.write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')
        original_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        out = self.base / 'cli output' / 'leadership.xlsx'
        result = subprocess.run([sys.executable, '-S', '-u', str(Path(renderer.__file__).resolve()),
             '--report', str(source), '--out', str(out), '--detail-rows', '3'],
            cwd=self.base, capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(out.is_file())
        self.assertTrue(out.with_suffix('.html').is_file())
        book, _ = read_book(out)
        self.assertEqual(tuple(book), renderer.SHEETS)
        text = "\n".join(str(value) for value in book['通信参考原表'][1].values())
        self.assertIn('显示前 3 / 12 条', text)
        self.assertIn(str(source), text)
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), original_hash)


if __name__ == '__main__':
    unittest.main()
