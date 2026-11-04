"""Atomic exports and compact default reports; synthetic data, no hardware claims."""
import contextlib
import copy
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import profile_ep128 as core
from test_ep128_v2_excel import fixture as renderer_fixture, renderer as real_renderer


def fixture():
    report = renderer_fixture()
    case = report["cases"][0]
    case["projection"].pop("methods", None)
    case["source"]["class_stats"] = []
    cycle = case["measurements"][0]["rounds"][0]
    cycle["start_ns"], cycle["end_ns"] = "1793103275541889538", "1793103275642012994"
    cycle["duration_ms"] = 100.123456
    cycle.pop("nonlayer_components", None)
    cycle["ledger"].pop("intervals", None)
    for method in case["projection"]["layer_mean_methods"]:
        method["comm_model"] = dict(method["communication"])
        method["compute_evidence"] = [{"rank": "dp0", "round_id": "hc-3", "role": "main", "layer": 1,
            "operator_names": ["GroupedMatmulV4Mxfp"], "row_indices": [123], "routed_compute_ms": .123456789}]
        for rank in method["per_rank"]:
            rank["rounds"] = [{"round_id": "hc-3", "source_cycle_ms": 100.123456,
                "cycle_ms": rank["cycle_ms"], "per_device_tokens_s": method["per_device_tokens_s"],
                "global_tokens_s": method["global_tokens_s"], "tpot_ms": method["tpot_ms"],
                "class_means": copy.deepcopy(rank["class_means"]),
                "role_details": copy.deepcopy(rank["role_details"]),
                "source_partition": copy.deepcopy(rank["source_partition"]),
                "breakdown": copy.deepcopy(method["breakdown"]), "detail_rows": []}]
    return report


class ExportProgressTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="synthetic-ep128-export-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_atomic_json_roundtrips_unicode_newlines_and_exact_integers_without_full_dumps(self):
        payload = {"文本": "主模型，草稿\n第二行\r\n引号\"与emoji🙂",
                   "exact_ns": 1793103275541889538,
                   "nested": [{"nullable": None, "enabled": False, "number": 1.23456789}]}
        before = copy.deepcopy(payload)
        original_dumps = json.dumps

        def forbid_full_dumps(value, *args, **kwargs):
            if value is payload:
                self.fail("atomic_json must stream the document, not json.dumps the full payload")
            return original_dumps(value, *args, **kwargs)

        target = self.root / "统计报告.json"
        with mock.patch.object(core.json, "dumps", side_effect=forbid_full_dumps):
            core.atomic_json(target, payload)
        self.assertEqual(json.loads(target.read_text(encoding="utf-8")), before)
        self.assertEqual(payload, before)
        self.assertIn("主模型", target.read_text(encoding="utf-8"))

    def test_csv_callable_scans_twice_preserves_late_columns_bom_and_nested_values(self):
        rows = [{"文本": '中文,逗号\n下一行"引号"', "nullable": None,
                 "nested": {"z": ["逗号,换行\n", "汉字"], "a": 1793103275541889538}},
                {"文本": "第二条", "later": "最后才出现的列", "flag": True}]
        before = copy.deepcopy(rows)
        traversals = []

        def row_factory():
            traversals.append("scan")
            yield from rows

        target = self.root / "流式明细.csv"
        core.write_csv(target, row_factory)
        self.assertEqual(len(traversals), 2, "discover all columns, then write rows without materializing the table")
        self.assertTrue(target.read_bytes().startswith(b"\xef\xbb\xbf"))
        with target.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            actual = list(reader)
            self.assertEqual(reader.fieldnames, ["文本", "nullable", "nested", "later", "flag"])
        self.assertEqual(len(actual), 2)
        self.assertEqual(actual[0]["文本"], rows[0]["文本"])
        self.assertEqual(actual[0]["nullable"], "")
        self.assertEqual(json.loads(actual[0]["nested"]), rows[0]["nested"])
        self.assertEqual(actual[0]["nested"], json.dumps(rows[0]["nested"], ensure_ascii=False, sort_keys=True))
        self.assertEqual(actual[0]["later"], "")
        self.assertEqual(actual[1]["later"], "最后才出现的列")
        list_target = self.root / "列表明细.csv"
        core.write_csv(list_target, rows)
        self.assertEqual(list_target.read_bytes(), target.read_bytes())
        self.assertEqual(rows, before)

    def test_empty_csv_keeps_status_header_and_bom(self):
        target = self.root / "empty.csv"
        core.write_csv(target, lambda: iter(()))
        self.assertTrue(target.read_bytes().startswith(b"\xef\xbb\xbf"))
        with target.open(encoding="utf-8-sig", newline="") as stream:
            self.assertEqual(list(csv.reader(stream)), [["status"]])

    def assert_failed_json_preserves_destination(self, target, action, error_type, pattern):
        original = b'{"previous_complete_report":true}\n'
        target.write_bytes(original)
        original_names = {path.name for path in self.root.iterdir()}
        with self.assertRaisesRegex(error_type, pattern):
            action()
        self.assertEqual(target.read_bytes(), original)
        self.assertEqual(json.loads(target.read_text()), {"previous_complete_report": True})
        self.assertEqual({path.name for path in self.root.iterdir()}, original_names,
                         "failed writes must close and remove their temporary file")

    def test_json_encoder_failure_after_partial_write_is_atomic_and_cleans_temp(self):
        target = self.root / "existing.json"

        def broken_chunks(encoder, payload, *args, **kwargs):
            yield '{"partially_written":'
            raise OSError("synthetic disk write failure")

        with mock.patch.object(core.json.JSONEncoder, "iterencode", broken_chunks):
            self.assert_failed_json_preserves_destination(
                target, lambda: core.atomic_json(target, {"new": "report"}),
                OSError, "synthetic disk write failure")

    def test_json_nonfinite_value_cannot_replace_complete_report(self):
        target = self.root / "existing.json"
        self.assert_failed_json_preserves_destination(
            target, lambda: core.atomic_json(target, {"valid_prefix": [1, 2], "invalid": float("nan")}),
            ValueError, "JSON|compliant|range|nan")

    def test_progress_compact_json_round_csv_and_input_object_are_consistent(self):
        report = fixture()
        case = report["cases"][0]
        case["name"] = "合成BS32,含逗号\n第二行"
        report["contract"]["long_unicode_evidence"] = "证据🙂\n" * 10000
        before = copy.deepcopy(report)
        events, out = [], self.root / "完整导出"
        original_dumps = json.dumps
        def forbid_report_dumps(value, *args, **kwargs):
            if value is report:
                self.fail("export must stream JSON rather than materialize a full JSON string")
            return original_dumps(value, *args, **kwargs)
        with mock.patch.object(core.json, "dumps", side_effect=forbid_report_dumps):
            core.export_report(report, out, [], progress=lambda message: events.append(str(message)))
        self.assertEqual(report, before)
        payload = json.loads((out / "model_architecture_report.json").read_text())
        self.assertEqual(payload["contract"]["long_unicode_evidence"], before["contract"]["long_unicode_evidence"])
        self.assertEqual(payload["cases"][0]["measurements"][0]["rounds"][0]["start_ns"], "1793103275541889538")
        self.assertTrue(payload.get("detail_tables"))
        self.assertIn("round_estimates.csv", json.dumps(payload["detail_tables"]))
        self.assertIn("compute_evidence.csv", json.dumps(payload["detail_tables"]))
        methods = payload["cases"][0]["projection"]["layer_mean_methods"]
        for actual, expected in zip(methods, case["projection"]["layer_mean_methods"]):
            for field in ("cycle_ms", "per_device_tokens_s", "tpot_ms", "selected_rank", "class_means", "role_details", "source_partition"):
                self.assertEqual(actual[field], expected[field])
            self.assertNotIn("compute_evidence", actual)
            self.assertNotIn("communication", actual)
            self.assertEqual(actual["comm_model"], expected["comm_model"])
            self.assertTrue(all("rounds" not in rank for rank in actual["per_rank"]))
        with (out / "round_estimates.csv").open(encoding="utf-8-sig", newline="") as stream:
            rounds = list(csv.DictReader(stream))
        self.assertEqual(len(rounds), sum(len(rank["rounds"]) for method in case["projection"]["layer_mean_methods"] for rank in method["per_rank"]))
        with (out / "compute_evidence.csv").open(encoding="utf-8-sig", newline="") as stream:
            evidence = list(csv.DictReader(stream))
        self.assertEqual(len(evidence), 1, "shared source GMM evidence must be stored once across both methods")
        self.assertNotIn("method", evidence[0])
        with (out / "最终结果.csv").open(encoding="utf-8-sig", newline="") as stream:
            final = list(csv.DictReader(stream))
        self.assertEqual(len(final), 2)
        for row, method in zip(final, methods):
            self.assertEqual(float(row["单卡吞吐(token/s/卡)"]), method["per_device_tokens_s"])
        first_json = next(i for i, message in enumerate(events) if "model_architecture_report.json" in message)
        for filename in ("最终结果.csv", "report.html"):
            done = next(i for i, message in enumerate(events) if filename in message and "完成" in message)
            self.assertLess(done, first_json)
        for filename in ("model_architecture_report.json", "round_estimates.csv", "time_ledger.csv"):
            self.assertTrue(any(filename in message and "完成" in message for message in events))
        # A compact saved JSON remains sufficient to render the complete workbook.
        saved_workbook = self.root / "从JSON重生成.xlsx"
        real_renderer.build_workbook(saved_workbook, report=out / "model_architecture_report.json",
                                     expected_method_set="layer_mean_methods")
        self.assertTrue(saved_workbook.is_file())

    def test_export_without_callback_is_silent(self):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            core.export_report(fixture(), self.root / "quiet", [])
        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(stderr.getvalue(), "")

    def test_summary_iterators_are_scalar_and_ledger_keeps_category_totals(self):
        report = fixture()
        before = copy.deepcopy(report)
        ledger_rows = list(core.iter_report_rows(report, "time_ledger"))
        ledger = report["cases"][0]["measurements"][0]["rounds"][0]["ledger"]
        self.assertEqual(len(ledger_rows), len(ledger["rows"]))
        self.assertEqual([row["duration_ms"] for row in ledger_rows],
                         [row["duration_ms"] for row in ledger["rows"]])
        self.assertTrue(all("intervals" not in row for row in ledger_rows))
        summary = list(core.iter_report_rows(report, "summary"))
        self.assertEqual(len(summary), 3)
        for row in summary[1:]:
            self.assertTrue(all(not isinstance(value, (dict, list)) for value in row.values()), row)
        for row in core.iter_report_rows(report, "rank_estimates"):
            self.assertTrue(all(not isinstance(value, (dict, list)) for value in row.values()), row)
        self.assertEqual(report, before)

    def test_final_rows_match_every_canonical_layer_mean_result(self):
        report = fixture()
        rows = core.result_rows(report)
        core.verify_result_rows(report, rows)
        methods = report["cases"][0]["projection"]["layer_mean_methods"]
        for row, method in zip(rows, methods):
            self.assertEqual(row["状态"], "条件估算")
            for column, field in (("周期(ms)", "cycle_ms"), ("单卡吞吐(token/s/卡)", "per_device_tokens_s"), ("TPOT(ms)", "tpot_ms")):
                self.assertEqual(row[column], method[field])
            self.assertNotIn("全局吞吐(token/s)", row)

    def test_result_verification_rejects_unavailable_or_corrupted_published_values(self):
        report = fixture()
        original = core.result_rows(report)
        mutations = []
        unavailable = copy.deepcopy(original)
        unavailable[0].update({"状态": "缺少实际输入", "周期(ms)": None, "单卡吞吐(token/s/卡)": None, "TPOT(ms)": None})
        mutations.append(unavailable)
        for field in ("周期(ms)", "单卡吞吐(token/s/卡)", "TPOT(ms)"):
            rows = copy.deepcopy(original)
            rows[0][field] += .125
            mutations.append(rows)
        mutations.append(original[1:])
        for index, rows in enumerate(mutations):
            with self.subTest(mutation=index), self.assertRaisesRegex(ValueError, "结果一致性"):
                core.verify_result_rows(report, rows)

    def test_export_rejects_inconsistent_final_rows_before_any_file_write(self):
        report = fixture()
        rows = core.result_rows(report)
        rows[0]["周期(ms)"] += .125
        out = self.root / "rejected"
        out.mkdir()
        old = out / "previous_complete_export.txt"
        old.write_bytes(b"preserve previous complete export")
        with mock.patch.object(core, "result_rows", return_value=rows), \
                mock.patch.object(core, "write_csv", side_effect=AssertionError("write before validation")), \
                mock.patch.object(core, "atomic_json", side_effect=AssertionError("write before validation")), \
                mock.patch.object(core, "atomic_text", side_effect=AssertionError("write before validation")), \
                self.assertRaisesRegex(ValueError, "结果一致性"):
            core.export_report(report, out, [])
        self.assertEqual(list(out.iterdir()), [old])

    def test_unavailable_method_cannot_leak_archived_numeric_candidates(self):
        report = fixture()
        method = report['cases'][0]['projection']['layer_mean_methods'][0]
        method.update(status='UNAVAILABLE', reasons=['missing reference'])
        rows = core.result_rows(report)
        core.verify_result_rows(report, rows)
        self.assertIsNone(rows[0]['周期(ms)'])
        rows[0]['周期(ms)'] = method['cycle_ms']
        with self.assertRaisesRegex(ValueError, '缺少输入却显示'):
            core.verify_result_rows(report, rows)

    def test_source_compute_evidence_survives_when_only_method2_has_a_result(self):
        report = fixture()
        first, second = report['cases'][0]['projection']['layer_mean_methods']
        first.update(status='UNAVAILABLE', reasons=['method1 topology unavailable'], compute_evidence=[])
        second['compute_evidence'] = [{'rank': 'dp0', 'round_id': 'hc-1', 'routed_compute_ms': .2}]
        rows = list(core.iter_report_rows(report, 'compute_evidence'))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['routed_compute_ms'], .2)
        self.assertNotIn('method', rows[0])

    def test_cli_excel_passes_same_in_memory_report_to_renderer(self):
        report = fixture()
        before = copy.deepcopy(report)
        out = self.root / "output"
        runtime = {"renderer_path": str(self.root / "renderer.py"), "renderer_version": "synthetic",
                   "writer_path": str(self.root / "writer.py"), "writer_version": "synthetic"}
        build = mock.Mock()
        renderer = SimpleNamespace(runtime_diagnostics=lambda: runtime, build_workbook=build)
        stdout = io.StringIO()
        with mock.patch.dict(sys.modules, {"build_ep128_leadership_excel": renderer}), contextlib.redirect_stdout(stdout):
            core.build_cli_excel(report, out, "layer_mean_methods")
        arguments, keywords = build.call_args
        self.assertEqual(arguments, (out / "10T模型推理估算.xlsx",))
        self.assertEqual(keywords["report"], out / "model_architecture_report.json")
        self.assertIs(keywords["report_data"], report)
        self.assertEqual(keywords["expected_method_set"], "layer_mean_methods")
        self.assertTrue(callable(keywords["progress"]))
        self.assertIn(runtime["renderer_path"], stdout.getvalue())
        self.assertEqual(report, before)

    def test_cli_excel_reports_incompatible_renderer_and_saved_csv(self):
        out = self.root / "already predicted"
        runtime = {"renderer_path": str(self.root / "old-renderer.py"), "renderer_version": "old",
                   "writer_path": str(self.root / "writer.py"), "writer_version": "synthetic"}
        def old_build_workbook(destination, *, report=None, report_data=None, progress=None):
            self.fail("an incompatible interface must not be retried silently")
        renderer = SimpleNamespace(runtime_diagnostics=lambda: runtime, build_workbook=old_build_workbook)
        with mock.patch.dict(sys.modules, {"build_ep128_leadership_excel": renderer}), \
                contextlib.redirect_stdout(io.StringIO()), \
                self.assertRaisesRegex(ValueError, "接口不匹配|同步更新") as caught:
            core.build_cli_excel(fixture(), out, "layer_mean_methods")
        self.assertIn(runtime["renderer_path"], str(caught.exception))
        self.assertIn(str((out / "最终结果.csv").resolve()), str(caught.exception))
        self.assertIsInstance(caught.exception.__cause__, TypeError)


if __name__ == "__main__":
    unittest.main()
