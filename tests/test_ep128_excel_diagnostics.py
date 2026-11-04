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

    def test_swa_drilldown_preserves_unknown_zero_and_attention_scope(self):
        diagnostic = dict(status='PARTIAL', scope='SYNTHETIC_HC_MARKER_INTERVAL', stages_csv='synthetic-swa.csv',
            layer_summary=[dict(role='main', layer=0, attention='HCA', reference_eligible=False, rounds=3,
                                wall_ms_mean=1.234, wall_ms_p50=None, no_recorded_ms_mean=0.)],
            stage_summary=[dict(role='draft', layer=1, attention='SWA', stage_key='attention',
                                label='SYNTHETIC_SWA_ATTENTION', rounds=3, matched_windows=3, status='MATCHED',
                                duration_ms_mean=.4321, recorded_busy_ms_mean=None, no_recorded_ms_mean=0.),
                           dict(role='draft', layer=1, attention='SWA', stage_key='missing_projection',
                                label='SYNTHETIC_PARTIAL_SWA', rounds=3, matched_windows=2, status='PARTIAL',
                                duration_ms_mean=99999.),
                           dict(role='main', layer=2, attention='CSA', stage_key='attention',
                                label='SYNTHETIC_CSA_NOT_SWA', rounds=3, matched_windows=3, status='MATCHED',
                                duration_ms_mean=88888.)])
        rows = renderer._swa_stage_rows(diagnostic)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0][4], .4321)
        self.assertEqual(rows[0][7:9], [renderer.NA, 0.])
        self.assertEqual(rows[1][4:], [renderer.NA] * 7)
        report = fixture()
        report['cases'][0]['measurements'][0]['swa_diagnostics'] = diagnostic
        out = self.base / 'swa-diagnostic.xlsx'
        renderer.build_workbook(out, report=self.base / 'source.json', report_data=report)
        cells = read_book(out)[0]['层与组件实测'][1]
        self.assertIn('SYNTHETIC_SWA_ATTENTION', cells.values())
        self.assertIn('SYNTHETIC_PARTIAL_SWA', cells.values())
        self.assertNotIn('SYNTHETIC_CSA_NOT_SWA', cells.values())
        self.assertNotIn(99999., cells.values())
        self.assertNotIn(88888., cells.values())
        preview = out.with_suffix('.html').read_text()
        self.assertIn('SYNTHETIC_SWA_ATTENTION', preview)
        self.assertNotIn('SYNTHETIC_CSA_NOT_SWA', preview)
        self.assertIn('未知值显示—，不补0', preview)

    def test_swa_details_are_display_only_and_never_change_cycle_or_tpot(self):
        report = fixture()
        capture = report['cases'][0]['measurements'][0]
        capture['swa_diagnostics'] = dict(status='MATCHED', scope='SYNTHETIC_ONLY', stages_csv='swa.csv',
            layer_summary=[dict(role='draft', layer=0, attention='SWA', reference_eligible=True,
                                rounds=2, wall_ms_mean=1234567.)],
            stage_summary=[dict(role='draft', layer=0, attention='SWA', stage_key='attention',
                                label='SYNTHETIC_LARGE_DETAIL_ALREADY_INCLUDED', status='MATCHED',
                                rounds=2, matched_windows=2, duration_ms_mean=1234567.,
                                kernel_sum_ms_mean=2345678., total_cost_ms_mean=3456789.)])
        before = copy.deepcopy(report)
        out = self.base / 'swa-unchanged-estimate.xlsx'
        renderer.build_workbook(out, report=self.base / 'source.json', report_data=report)
        self.assertEqual(report, before)
        book = read_book(out)[0]
        self.assertIn(1234567., book['层与组件实测'][1].values())
        for method, name in zip(report['cases'][0]['projection']['layer_mean_methods'], ('方法1估算', '方法2估算')):
            for field in ('cycle_ms', 'tpot_ms', 'per_device_tokens_s'):
                self.assertIn(method[field], book[name][1].values())
            self.assertNotIn(1234567., book[name][1].values())
        self.assertIn('不加到层链、U、完整周期或TPOT', out.with_suffix('.html').read_text())

    def test_u_children_stay_with_their_parents_without_changing_parent_partition(self):
        from test_ep128_layer_mean_excel import residual_report
        report = residual_report()
        case = report['cases'][0]
        diagnostic = case['measurements'][0]['step_residuals']
        for parent, label in (('draft_to_nextmain_1', 'SYNTHETIC_METADATA_BODY'),
                              ('draft_to_nextmain_2', 'SYNTHETIC_KV_INPUTS')):
            diagnostic['substage_summary'].append(dict(window_key='draft_to_nextmain', parent_stage_key=parent,
                stage_key=label, stage_order=1, label=label, level=1, rounds=2, matched_windows=2,
                status='MATCHED', duration_ms_mean=456789., source_status='PARTIAL',
                source_cpu_tasks_total=1, source_device_tasks_total=3, source_cpu_union_ms_mean=.12345))
        groups = renderer._residual_substage_groups(diagnostic, 'draft_to_nextmain')
        self.assertEqual({parent for parent, _, _ in groups},
                         {'draft_to_nextmain_1', 'draft_to_nextmain_2', 'draft_to_nextmain_3'})
        for parent, _, children in groups:
            self.assertTrue(all(item['parent_stage_key'] == parent for item in children))
        method = case['projection']['layer_mean_methods'][0]
        rows, bars, _ = renderer._method_residual_data(case, method)
        self.assertEqual(len(rows), 15)
        self.assertEqual(len(bars), 15)
        self.assertAlmostEqual(sum(item['value_ms'] for item in bars), diagnostic['residual_ms_mean'])
        preview = renderer._residual_html(diagnostic, 'dp7')
        self.assertIn('父阶段细分', preview)
        self.assertNotIn('Markov顺序选词子阶段', preview)
        self.assertIn('SYNTHETIC_METADATA_BODY', preview)
        self.assertIn('SYNTHETIC_KV_INPUTS', preview)
        self.assertIn('本组子阶段的关联Host提交来源', preview)
        out = self.base / 'u-parent-groups.xlsx'
        renderer.build_workbook(out, report=self.base / 'source.json', report_data=report)
        book = read_book(out)[0]
        self.assertIn('SYNTHETIC_METADATA_BODY', book['时间账本'][1].values())
        self.assertNotIn(456789., book['方法1估算'][1].values())

    def test_dma_coverage_stays_separate_from_u_and_does_not_fill_missing_trace_with_zero(self):
        from test_ep128_layer_mean_excel import residual_report
        report = residual_report()
        diagnostic = report['cases'][0]['measurements'][0]['step_residuals']
        stage = diagnostic['stage_summary'][-1]
        stage.update(label='SYNTHETIC_MAIN_INPUTS_DMA', trace_dma_union_ms_mean=.7654321,
                     trace_dma_in_csv_uncovered_ms_mean=.4567891, uncovered_after_trace_dma_ms_mean=0.)
        missing = diagnostic['stage_summary'][-2]
        missing.update(label='SYNTHETIC_NO_ALIGNED_TRACE', trace_dma_union_ms_mean=None,
                       trace_dma_in_csv_uncovered_ms_mean=None, uncovered_after_trace_dma_ms_mean=None)
        rows = renderer._residual_dma_rows([stage, missing])
        self.assertEqual(rows[0][1:], [.7654321, .4567891, 0.])
        self.assertEqual(rows[1][1:], [renderer.NA] * 3)
        expected = diagnostic['residual_ms_mean']
        out = self.base / 'dma-coverage.xlsx'
        renderer.build_workbook(out, report=self.base / 'source.json', report_data=report)
        book = read_book(out)[0]
        self.assertIn(.7654321, book['时间账本'][1].values())
        self.assertIn(.4567891, book['时间账本'][1].values())
        self.assertNotIn(.7654321, book['方法1估算'][1].values())
        self.assertEqual(diagnostic['residual_ms_mean'], expected)
        preview = out.with_suffix('.html').read_text()
        self.assertIn('DMA对CSV未记录区间的补充覆盖', preview)
        self.assertIn('也不代表已证实的NPU硬件空闲', preview)
        self.assertIn('无trace或条件不足显示—', preview)


if __name__ == "__main__":
    unittest.main()
