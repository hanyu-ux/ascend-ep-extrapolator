"""Third-method display contracts; synthetic inputs, no NPU performance claims."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import profile_ep128 as core
import test_ep128_layer_mean as means
import test_ep128_layer_mean_excel as layer_excel
from test_ep128_consolidated_trends import grid_report
from test_ep128_focus_excel import CHART_NS, focus_fixture
from test_ep128_report_navigation import HTMLTargets, location_parts
from test_ep128_source_rank import source_rank_report
from test_ep128_v2_excel import read_book, renderer


ORIGINAL_SHEETS = ("估算总览", "源采样基线", "层与组件实测", "时间账本", "方法1估算",
                   "方法2估算", "通信参考原表", "公式与假设", "复核记录")


def add_synthetic_third(report):
    """An explicit renderer fixture: independent input values, not estimator code."""
    for case in report["cases"]:
        third = copy.deepcopy(renderer._method(case, "method1"))
        third.update(id="method3", name="SYNTHETIC_METHOD3_INPUT", cycle_ms=third["cycle_ms"] + 3.125,
                     per_device_tokens_s=third["per_device_tokens_s"] + 37.875,
                     tpot_ms=third["tpot_ms"] + .375,
                     formula="SYNTHETIC_CORE_METHOD3_FORMULA; mD=mC=1; extra_ms_per_call=0")
        third["breakdown"][-1]["value_ms"] += 3.125
        def update_explicit_fixture_shares(value):
            if isinstance(value, dict):
                if "cycle_share" in value and "target_total_ms" in value:
                    value["cycle_share"] = value["target_total_ms"] / third["cycle_ms"]
                for child in value.values():
                    update_explicit_fixture_shares(child)
            elif isinstance(value, list):
                for child in value:
                    update_explicit_fixture_shares(child)
        update_explicit_fixture_shares(third.get("target_stage_breakdowns") or [])
        third["communication"] = dict(extra_ms_per_call=0., remote_ratio=1.,
            multipliers={"main": {"dispatch": 1., "combine": 1.}, "draft": {"dispatch": 1., "combine": 1.}})
        case["projection"]["layer_mean_methods"].append(third)
    return report


class MethodThreeRendererTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="synthetic-method3-display-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.tools = layer_excel.LayerMeanRendererTests()

    def build(self, report):
        before = copy.deepcopy(report)
        saved = []
        native_save = renderer.Workbook.save

        def save(book, out):
            saved.append(book)
            return native_save(book, out)

        out = self.root / "method3.xlsx"
        with mock.patch.object(core, "read_capture", side_effect=AssertionError("renderer must not read CSV")), \
                mock.patch.object(renderer.Workbook, "save", new=save):
            renderer.build_workbook(out, report=self.root / "synthetic.json", report_data=report)
        self.assertEqual(report, before)
        book = read_book(out)[0]
        return out, book, saved[0], out.with_suffix(".html").read_text(encoding="utf-8")

    def test_actual_core_method_three_values_are_preserved_in_cards_and_formula(self):
        report, native = means.synthetic_report({"dp0": [1.], "dp1": [2.]})
        methods = means.attach(report, native)
        third = next(method for method in methods if method["id"] == "method3")
        self.assertEqual(third["status"], "LAYER_MEAN_ESTIMATE", third["reasons"])
        self.assertNotEqual(third["cycle_ms"], methods[0]["cycle_ms"])
        out, book, saved, page = self.build(report)
        self.assertEqual(tuple(book)[:9], ORIGINAL_SHEETS)
        self.assertEqual(tuple(book), (*ORIGINAL_SHEETS, "方法3估算"))
        for name in ("估算总览", "方法3估算"):
            sheet = next(sheet for sheet in saved.sheets if sheet.name == name)
            for field in ("cycle_ms", "per_device_tokens_s", "tpot_ms"):
                row, column = sheet.result_cells[(0, "method3")][field]
                self.assertEqual(float(sheet.cells[(row, column)][1]), third[field])
        formula = "\n".join(map(str, book["公式与假设"][1].values()))
        self.assertIn(third["formula"], formula)
        self.assertIn("方法三每层通信保持源参考值", formula)
        self.assertIn("multipliers.main.dispatch", formula)
        self.assertIn("不表示整轮通信总量必定与源模型相同", page)
        self.assertIn("10个工作表", page)

    def test_old_json_missing_method_three_is_explicitly_unavailable_without_recalculation(self):
        report = focus_fixture()
        self.assertNotIn("method3", [method["id"] for method in report["cases"][0]["projection"]["layer_mean_methods"]])
        out, book, saved, page = self.build(report)
        values = "\n".join(map(str, book["方法3估算"][1].values()))
        self.assertIn("核心未提供本方法", values)
        self.assertIn("--add-method3", values)
        third_sheet = next(sheet for sheet in saved.sheets if sheet.name == "方法3估算")
        for row, column in third_sheet.result_cells[(0, "method3")].values():
            self.assertEqual(third_sheet.cells[(row, column)][1], renderer.PENDING)
        charts = self.tools.chart_parts_by_sheet(out, book)["方法3估算"]
        self.assertFalse(charts, "absent method3 must not copy another method's charts/results")
        self.assertIn("核心报告未输出方法三", page)
        self.assertEqual(len(renderer._focus_source_entries(report["cases"][0])), 2)
        self.assertEqual(list(book["源采样基线"][1].values()).count("保留序号"), 2)

    def test_configuration_preview_includes_third_method_without_performance_values(self):
        config = self.root / "synthetic-config.json"
        config.write_text(json.dumps(dict(schema_version=2, cases=[dict(name="SYNTHETIC_ONLY", bs=4)],
                                         sample_model={}, target_model={})), encoding="utf-8")
        with mock.patch.object(core, "read_capture", side_effect=AssertionError("preview must not read CSV")):
            preview = renderer._preview_report(config)
            renderer._validate_report(preview)
            out = renderer.build_workbook(self.root / "preview.xlsx", cases=config, preview=True)
        self.assertEqual([method["id"] for method in preview["cases"][0]["projection"]["layer_mean_methods"]],
                         ["method1", "method2", "method3"])
        book = read_book(out)[0]
        self.assertIn("方法3估算", book)
        values = book["方法3估算"][1].values()
        self.assertIn("仅配置版式预览；未读取kernel CSV，未产生性能数值", values)
        self.assertFalse(self.tools.chart_parts_by_sheet(out, book)["方法3估算"])

    def test_third_method_cannot_silently_use_different_shared_source_rank(self):
        report, native = means.synthetic_report({"dp0": [1.], "dp1": [2.]})
        methods = means.attach(report, native)
        third = next(method for method in methods if method["id"] == "method3")
        self.assertEqual(third["selected_rank"], "dp1")
        third["selected_rank"] = "dp0"
        with self.assertRaisesRegex(ValueError, "共享源rank口径不一致"):
            renderer._validate_report(report)

    def test_old_core_missing_offline_interface_has_actionable_update_error(self):
        with mock.patch.object(core, "add_method3_report", new=None):
            with self.assertRaisesRegex(ValueError, "同步更新profile_ep128.py和build_ep128_leadership_excel.py") as raised:
                renderer._add_method3({})
        self.assertIn(str(Path(core.__file__).resolve()), str(raised.exception))

    def legacy_source_report(self):
        report, native = source_rank_report()
        means.attach(report, native)
        payload = core.report_json_payload(report)
        for case in payload["cases"]:
            case["projection"]["layer_mean_methods"] = [method for method in
                case["projection"]["layer_mean_methods"] if method["id"] != "method3"]
        return payload

    def run_cli(self, *arguments):
        return subprocess.run([sys.executable, "-B", "-S", renderer.__file__, *map(str, arguments)],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"})

    def test_cli_combines_method_three_alignment_bound_and_independent_sidecars(self):
        original = self.legacy_source_report()
        source = self.root / "synthetic-source.json"
        source.write_text(json.dumps(original, ensure_ascii=False), encoding="utf-8")
        before = source.read_bytes()
        out = self.root / "combined.xlsx"
        result = self.run_cli("--report", source, "--out", out, "--add-method3",
                              "--align-source-rank", "--refresh-bound")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(source.read_bytes(), before)
        self.assertTrue(out.exists())
        self.assertTrue(out.with_suffix(".html").exists())
        sidecars = [out.with_suffix(suffix) for suffix in (".method3.json", ".source_rank.json", ".bound.json")]
        self.assertTrue(all(path.exists() for path in sidecars))
        derived = [json.loads(path.read_text(encoding="utf-8")) for path in sidecars]
        self.assertEqual(derived[0], derived[1])
        self.assertEqual(derived[0], derived[2])
        methods = derived[0]["cases"][0]["projection"]["layer_mean_methods"]
        self.assertEqual([method["id"] for method in methods], ["method1", "method2", "method3"])
        self.assertEqual([method["selected_rank"] for method in methods], ["dp0"] * 3)
        self.assertEqual(methods[-1]["communication"]["extra_ms_per_call"], 0)
        self.assertFalse(methods[-1]["offline_derivation"]["source_csv_read"])
        for before_method, after_method in zip(original["cases"][0]["projection"]["layer_mean_methods"], methods):
            for field in ("cycle_ms", "per_device_tokens_s", "tpot_ms", "selected_rank"):
                self.assertEqual(after_method[field], before_method[field])
        self.assertIn("HTML写入完成", result.stderr)

    def test_cli_combined_output_paths_cannot_overwrite_input_json(self):
        payload = json.dumps(self.legacy_source_report(), ensure_ascii=False).encode("utf-8")
        for suffix in (".method3.json", ".source_rank.json", ".bound.json"):
            with self.subTest(suffix=suffix):
                out = self.root / "protected.xlsx"
                source = out.with_suffix(suffix)
                source.write_bytes(payload)
                result = self.run_cli("--report", source, "--out", out, "--add-method3",
                                      "--align-source-rank", "--refresh-bound")
                self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                self.assertEqual(source.read_bytes(), payload)
                self.assertFalse(out.exists())
                self.assertIn("不能覆盖输入报告", result.stderr)

    def test_third_method_has_bs_and_speculation_trends_using_its_own_values(self):
        report = add_synthetic_third(grid_report())
        out, book, saved, page = self.build(report)
        charts = self.tools.chart_parts_by_sheet(out, book)
        for axis in ("BS", "S"):
            method_charts = [chart for chart in charts["方法3估算"] if "随" + axis + "变化" in self.tools.chart_title(chart)]
            self.assertEqual(len(method_charts), 4)
            overview = [chart for chart in charts["估算总览"] if "随" + axis + "变化" in self.tools.chart_title(chart)]
            self.assertEqual(len(overview), 9)
            family, = renderer._trend_families(report, axis, "method3")
            for field, label, _ in (*renderer.SWEEP_METRICS, ("parts", "整轮五项耗时构成", "ms/完整周期")):
                chart, = [chart for chart in method_charts if self.tools.chart_title(chart).startswith(label + "随")]
                series = self.tools.assert_line_cache_matches_sheet(chart, book, family["categories"])
                observed = [{int(point.get("idx")): float(point.findtext("c:v", namespaces=CHART_NS))
                             for point in item.findall("c:val/c:numRef/c:numCache/c:pt", CHART_NS)} for item in series]
                matrix = renderer._trend_matrix(family, field)
                expected = [{index: value for index, value in enumerate(item["values"]) if value is not None}
                            for item in matrix["series"]]
                self.assertEqual(observed, expected)
                self.assertIn(renderer._layer_mean_name({"id": "method3"}), self.tools.chart_title(chart))
        for group in renderer._bs_comparison_groups(report):
            for row, case in zip(renderer._extra_overview_rows(group, "BS"), group["cases"]):
                third = renderer._method(case, "method3")
                self.assertEqual(row[3:6], [third[field] for field, _, _ in renderer.BS_METRICS])

    def test_third_method_reuses_source_focus_and_retains_pmu_drilldown_and_navigation(self):
        report = add_synthetic_third(focus_fixture())
        out, book, saved, page = self.build(report)
        values = "\n".join(map(str, book["方法3估算"][1].values()))
        self.assertIn("AscendBlockTables.compute_slot_mappings", values)
        self.assertIn("97.5%", values)
        self.assertIn("SYNTHETIC_AIC_EVIDENCE", values)
        self.assertNotIn("保留序号", book["方法3估算"][1].values())
        self.assertEqual(list(book["源采样基线"][1].values()).count("保留序号"), 2)
        charts = self.tools.chart_parts_by_sheet(out, book)
        refs = []
        for name in ("方法1估算", "方法3估算"):
            chart, = [chart for chart in charts[name] if "两项U逐轮耗时变化" in self.tools.chart_title(chart)]
            refs.append([node.text for node in chart.findall(".//c:f", CHART_NS)])
        self.assertEqual(refs[0], refs[1])
        self.assertTrue(all(reference.startswith("'源采样基线'!") for reference in refs[1]))
        links = book["方法3估算"][0].findall(".//s:hyperlink", {"s": renderer.xlsx_writer.MAIN})
        self.assertGreater(len(links), 10)
        for link in links:
            sheet, reference = location_parts(link.get("location"))
            self.assertIn(reference, book[sheet][1])
        targets = HTMLTargets(page)
        self.assertEqual(len(targets.ids), len(set(targets.ids)))
        self.assertTrue(set(targets.fragments).issubset(targets.ids))


if __name__ == "__main__":
    unittest.main()
