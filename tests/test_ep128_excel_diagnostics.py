"""Renderer/writer failure localization using synthetic CPU fixtures only."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import ep128_xlsx_writer as writer
from test_ep128_v2_excel import fixture, read_book, renderer


BARE_OLD_MESSAGE = "Excel row/column is outside the one-based worksheet range"
INPUT_DIAGNOSTIC_PREFIXES = ("输入报告：", "输入清单：", "报告生成时间：", "输入组合：", "BS分组：", "失败组合：")


class ExcelDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="synthetic-excel-diagnostics-")
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)

    def test_runtime_identity_uses_the_actual_loaded_modules_and_code(self):
        identity = renderer.runtime_diagnostics()
        self.assertEqual(identity["renderer_path"], str(Path(renderer.__file__).resolve()))
        self.assertEqual(identity["renderer_version"], renderer.RENDERER_VERSION)
        self.assertEqual(identity["writer_path"], str(Path(writer.__file__).resolve()))
        self.assertEqual(identity["writer_version"], writer.WRITER_VERSION)
        self.assertEqual(identity["position_code_path"], writer._position.__code__.co_filename)
        self.assertEqual(identity["position_code_line"], writer._position.__code__.co_firstlineno)
        self.assertEqual(identity["python_executable"], sys.executable)
        self.assertEqual(identity["python_version"], sys.version.split()[0])

    def test_even_an_old_bare_error_retains_stage_paths_and_original_cause(self):
        original = ValueError(BARE_OLD_MESSAGE)
        stages = []
        with mock.patch.object(renderer, "_read_json", return_value=fixture()), \
                mock.patch.object(renderer, "_ledger_sheet", side_effect=original):
            with self.assertRaises(ValueError) as caught:
                renderer.build_workbook(self.base / "result.xlsx",
                                        report=self.base / "saved report.json",
                                        progress=stages.append)
        error = caught.exception
        self.assertIs(error.__cause__, original)
        self.assertIn("阶段=时间账本", str(error))
        self.assertIn(BARE_OLD_MESSAGE, str(error))
        self.assertIn(str(Path(renderer.__file__).resolve()), str(error))
        self.assertIn(str(Path(writer.__file__).resolve()), str(error))
        self.assertEqual(stages[-1], "时间账本")
        self.assertNotIn("写入XLSX", stages)
        self.assertFalse((self.base / "result.xlsx").exists())

    def test_loaded_report_skips_json_read_preserves_payload_and_renders_all_stages(self):
        data = fixture()
        before = copy.deepcopy(data)
        source = self.base / "saved report.json"
        # A source label is still available in the workbook, but this content
        # must not be parsed when its already loaded report is supplied.
        source.write_text("SYNTHETIC deliberately unparsed source label", encoding="utf-8")
        out = self.base / "loaded.xlsx"
        stages = []
        with mock.patch.object(renderer, "_read_json", side_effect=AssertionError("must reuse report")) as reader, \
                mock.patch.object(renderer, "_validate_report", wraps=renderer._validate_report) as validator:
            self.assertEqual(renderer.build_workbook(out, report=source, report_data=data,
                                                    progress=stages.append), out)
        reader.assert_not_called()
        validator.assert_called_once_with(data)
        self.assertEqual(data, before)
        self.assertEqual(source.read_text(encoding="utf-8"), "SYNTHETIC deliberately unparsed source label")
        book, _ = read_book(out)
        self.assertEqual(tuple(book), renderer.SHEETS)
        self.assertIn(519.4321, book["估算总览"][1].values())
        self.assertIn(str(source), book["复核记录"][1].values())
        self.assertTrue(out.with_suffix(".html").is_file())
        self.assertEqual([message for message in stages if not message.startswith(INPUT_DIAGNOSTIC_PREFIXES)],
                         ["复用已加载核心报告", "校验核心报告", *renderer.SHEETS,
                          "核对Excel关键结果", "写入XLSX", "写入HTML预览"])
        origin = next(message for message in stages if message.startswith("输入报告："))
        self.assertIn("复用内存报告（来源路径仅为标签）", origin)
        self.assertIn(str(source.resolve()), origin)
        self.assertTrue(any(message.startswith("输入清单：case=1；") for message in stages))
        self.assertIn("报告生成时间：" + data["generated_at"], stages)
        self.assertTrue(any(message.startswith("输入组合：") and data["cases"][0]["name"] in message
                            for message in stages))
        self.assertTrue(any(message.startswith("BS分组：") for message in stages))
        self.assertNotIn("读取核心JSON", stages)

    def test_in_memory_report_and_deduplicated_saved_json_render_the_same_values(self):
        data = fixture()
        for method in data['cases'][0]['projection']['layer_mean_methods']:
            method['comm_model'] = copy.deepcopy(method['communication'])
        archived = copy.deepcopy(data)
        for method in archived['cases'][0]['projection']['layer_mean_methods']:
            method.pop('communication')
        source = self.base / 'saved-slim-report.json'
        source.write_text(json.dumps(archived, ensure_ascii=False), encoding='utf-8')
        memory_out, saved_out = self.base / 'memory.xlsx', self.base / 'saved.xlsx'
        renderer.build_workbook(memory_out, report=source, report_data=data)
        renderer.build_workbook(saved_out, report=source)
        memory_book, saved_book = read_book(memory_out)[0], read_book(saved_out)[0]
        self.assertEqual({name: cells for name, (_, cells) in memory_book.items()},
                         {name: cells for name, (_, cells) in saved_book.items()})
        self.assertIn(.014, saved_book['公式与假设'][1].values())

    def test_loaded_report_requires_source_label_and_cannot_be_a_preview(self):
        source = self.base / "source.json"
        cases = self.base / "cases.json"
        invalid = [dict(report_data=fixture()),
                   dict(cases=cases, preview=True, report_data=fixture()),
                   dict(report=source, cases=cases, report_data=fixture()),
                   dict(report=source, preview=True, report_data=fixture()),
                   dict(report=source, report_data=[]),
                   dict(report=source, report_data="not a report")]
        for arguments in invalid:
            with self.subTest(arguments=tuple(arguments)), \
                    mock.patch.object(renderer, "_read_json", side_effect=AssertionError("invalid arguments")), \
                    self.assertRaises(ValueError):
                renderer.build_workbook(self.base / "invalid.xlsx", **arguments)
        self.assertFalse((self.base / "invalid.xlsx").exists())

    def test_loaded_report_keeps_normal_schema_and_numeric_validation(self):
        for mutate in (lambda data: data.update(schema_version=1),
                       lambda data: data["cases"][0]["config"].update(synthetic_nonfinite=float("nan"))):
            data = fixture()
            mutate(data)
            stages = []
            with mock.patch.object(renderer, "_read_json", side_effect=AssertionError("must reuse report")), \
                    self.assertRaisesRegex(ValueError, "阶段=校验核心报告"):
                renderer.build_workbook(self.base / "invalid.xlsx", report=self.base / "source.json",
                                        report_data=data, progress=stages.append)
            self.assertEqual([message for message in stages if not message.startswith(INPUT_DIAGNOSTIC_PREFIXES)],
                             ["复用已加载核心报告", "校验核心报告"])
        self.assertFalse((self.base / "invalid.xlsx").exists())

    def test_obsolete_selector_identifies_source_and_case_before_building(self):
        data = fixture()
        data['cases'][0]['projection']['display_method_set'] = 'methods'
        source = self.base / 'obsolete-source.json'
        stages = []
        with mock.patch.object(renderer, '_overview', side_effect=AssertionError('must reject selector first')), \
                self.assertRaisesRegex(ValueError, '阶段=校验核心报告') as caught:
            renderer.build_workbook(self.base / 'wrong.xlsx', report=source, report_data=data,
                                    expected_method_set='layer_mean_methods', progress=stages.append)
        message = str(caught.exception)
        self.assertIn(data['cases'][0]['name'], message)
        self.assertIn("display_method_set='methods'", message)
        self.assertEqual(stages[-1], '校验核心报告')
        self.assertFalse((self.base / 'wrong.xlsx').exists())

    def test_selector_checks_every_case_and_rejects_obsolete_internal_expectations(self):
        data = fixture()
        second = copy.deepcopy(data['cases'][0])
        second['name'] = 'SECOND_CASE_WITH_DIFFERENT_SELECTOR'
        second['projection']['display_method_set'] = 'assumption_methods'
        data['cases'].append(second)
        with self.assertRaisesRegex(ValueError, second['name']):
            renderer.build_workbook(self.base / 'mixed.xlsx', report=self.base / 'source.json',
                                    report_data=data, expected_method_set='layer_mean_methods')
        for expected in ('methods', 'source_point_methods', 'assumption_methods', 'unknown'):
            with self.subTest(expected=expected), self.assertRaisesRegex(ValueError, 'expected_method_set must be layer_mean_methods'):
                renderer.build_workbook(self.base / 'bad-choice.xlsx', report=self.base / 'source.json',
                                        report_data=fixture(), expected_method_set=expected)

    def test_cli_rejects_removed_method_selector_flag(self):
        source = self.base / 'strict-source.json'
        source.write_text(json.dumps(fixture(), ensure_ascii=False), encoding='utf-8')
        out = self.base / 'wrong.xlsx'
        result = subprocess.run(
            [sys.executable, '-S', '-u', str(Path(renderer.__file__).resolve()),
             '--report', str(source), '--expected-method-set', 'source_point_methods', '--out', str(out)],
            cwd=self.base, capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn('unrecognized arguments: --expected-method-set', result.stderr)
        self.assertFalse(out.exists())

    def test_writer_position_failure_reports_sheet_coordinate_and_type(self):
        for row, column, typename in ((0, 1, "int"), (1, 0, "int"),
                                      (writer.MAX_ROW + 1, 1, "int"),
                                      (1, writer.MAX_COL + 1, "int"),
                                      (1.5, 1, "float"), (True, 1, "bool")):
            with self.subTest(row=row, column=column):
                sheet = writer.Workbook().add_sheet("SYNTHETIC_DIAGNOSTIC_SHEET")
                with self.assertRaises(ValueError) as caught:
                    sheet.write(row, column, "no hardware data")
                message = str(caught.exception)
                self.assertIn("SYNTHETIC_DIAGNOSTIC_SHEET", message)
                self.assertIn("row=" + repr(row), message)
                self.assertIn("column=" + repr(column), message)
                self.assertIn("(" + typename + ")", message)
                self.assertIsInstance(caught.exception.__cause__, ValueError)

    def test_save_validation_locates_an_invalid_internal_cell_in_its_sheet(self):
        book = writer.Workbook()
        sheet = book.add_sheet("SYNTHETIC_INVALID_CELL")
        # Deliberately bypass the public guard to exercise save-time validation.
        sheet.cells[(writer.MAX_ROW + 1, 1)] = ("str", "synthetic bad cell", 1, None)
        with self.assertRaises(ValueError) as caught:
            book.save(self.base / "invalid.xlsx")
        self.assertIn("SYNTHETIC_INVALID_CELL", str(caught.exception))
        self.assertIn("row=" + str(writer.MAX_ROW + 1), str(caught.exception))

    def test_cli_prints_loaded_identity_and_full_traceback_for_real_failure(self):
        data = fixture()
        data["cases"][0]["projection"]["layer_mean_methods"][0]["assumptions"] = ["\x01"]
        source = self.base / "saved core report.json"
        source.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        out = self.base / "failure.xlsx"
        result = subprocess.run(
            [sys.executable, "-S", "-u", str(Path(renderer.__file__).resolve()),
             "--report", str(source), "--out", str(out)],
            cwd=self.base, capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("Traceback (most recent call last)", result.stderr)
        self.assertIn("invalid XML character", result.stderr)
        self.assertIn("阶段=公式与假设", result.stderr)
        self.assertIn("Excel阶段：公式与假设", result.stderr)
        self.assertIn("direct cause", result.stderr)
        self.assertIn(str(Path(renderer.__file__).resolve()), result.stderr)
        self.assertIn(str(Path(writer.__file__).resolve()), result.stderr)
        self.assertIn(renderer.RENDERER_VERSION, result.stderr)
        self.assertIn(writer.WRITER_VERSION, result.stderr)
        self.assertIn("实际导入写入器", result.stderr)
        self.assertIn("坐标校验代码", result.stderr)
        self.assertFalse(out.exists())

    def test_cli_version_is_available_without_a_report(self):
        result = subprocess.run(
            [sys.executable, "-S", "-u", str(Path(renderer.__file__).resolve()), "--version"],
            cwd=self.base, capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(renderer.RENDERER_VERSION, result.stdout)


if __name__ == "__main__":
    unittest.main()
