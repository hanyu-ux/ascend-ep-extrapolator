"""Fixed-rank U detail and native PMU display contracts using synthetic data only."""
import copy
from pathlib import Path
import re
import tempfile
import unittest
from unittest import mock
from test_ep128_v2_excel import fixture, read_book, renderer
import test_ep128_layer_mean_excel as layer_excel


CHART_NS = {"c": "http://schemas.openxmlformats.org/drawingml/2006/chart",
            "a": "http://schemas.openxmlformats.org/drawingml/2006/main"}


def focus_fixture():
    report = layer_excel.operator_drilldown_report()
    for capture in report["cases"][0]["measurements"]:
        factor = 1 if capture["rank"] == "dp7" else 10
        mapping = dict(path="vllm_ascend/worker/v2/block_table.py", function="AscendBlockTables.compute_slot_mappings",
            line=82, version_basis="SYNTHETIC_FIXED_COMMIT", evidence="SYNTHETIC_STATIC_ONLY")
        operator = dict(name="SYNTHETIC_FULL_NATIVE_NAME_" + "component_" * 12,
            op_type="SYNTHETIC_SLOT_KERNEL", stream_id="23", input_shapes="128;8,1024", output_shapes="128",
            source_function_mapping=[mapping], mapping_status="STATIC_SOURCE_MAPPING", example_row_indices=[99, 100])
        stages = []
        for key, label, base in (("main_inputs", "下一主模型KV与输入准备", 1.),
                                  ("main_metadata", "下一主metadata及输入映射准备", 4.)):
            values = [factor * base * value for value in (.5, 1., 1.5)]
            summary = []
            for index, (fraction, kind) in enumerate(((.5, "single_operator"), (.3, "concurrent_operators"), (.2, "no_device_record")), 1):
                summary.append(dict(component_key=key + str(index), label="SYNTHETIC_" + key + "_" + str(index),
                    kind=kind, operator_ids=["op1"] if index != 3 else [], source_function_mapping=[mapping] if index == 1 else [],
                    mapping_status="STATIC_SOURCE_MAPPING" if index != 3 else "UNMAPPED", observed_steps=3,
                    duration_ms_mean=factor * base * fraction, duration_ms_p50=factor * base * fraction,
                    duration_ms_p90=factor * base * 1.4 * fraction, duration_ms_max=max(values) * fraction,
                    share_of_parent=fraction))
            steps = [dict(step_index=index, round_id=round_id, status="MATCHED", start_ns=str(index * 100000000),
                end_ns=str(index * 100000000 + int(value * 1e6)), duration_ms=value,
                recorded_busy_ms=value * .8, no_recorded_ms=value * .2, kernel_sum_ms=value * 1.1,
                task_count=4, component_durations_ms={item["component_key"]:value * item["share_of_parent"] for item in summary})
                for index, (round_id, value) in enumerate(zip(("hc-10", "hc-13", "hc-18"), values), 1)]
            stages.append(dict(stage_key=key, label=label, rounds=3, observed_steps=3, status="MATCHED", summary=summary,
                per_step=steps, duration_ms_mean=factor * base, duration_ms_p50=factor * base,
                duration_ms_p90=factor * base * 1.4, duration_ms_max=max(values)))
        capture["step_residuals"]["focus_diagnostics"] = dict(version="1", rank=capture["rank"],
            scope="SYNTHETIC_SAME_RANK_ONLY", stages=stages, operators={"op1": operator},
            segments_csv="SYNTHETIC-focus-segments.csv", steps_csv="SYNTHETIC-focus-steps.csv")
    for method in report["cases"][0]["projection"]["layer_mean_methods"]:
        for group in method["target_stage_breakdowns"]:
            for stage in group["stage_rows"]:
                for kernel in stage["operator_breakdown"]["kernels"]:
                    metrics = {key:dict(value=value, unit="ratio", known_calls=2, total_calls=2,
                        known_duration_ms=2., total_duration_ms=2., duration_coverage=1.)
                        for key, value in (("aic_mac_ratio", .975), ("aic_mte2_ratio", .771), ("aic_mte3_ratio", 0.),
                                           ("aiv_vec_ratio", .0), ("aiv_mte2_ratio", .2), ("aiv_mte3_ratio", .3),
                                           ("aic_scalar_ratio", .1), ("aiv_scalar_ratio", .4), ("aic_icache_miss_rate", .002))}
                    kernel["pmu"] = dict(metrics=metrics, coverage=dict(known_calls=2, total_calls=2,
                        known_duration_ms=2., total_duration_ms=2., duration_coverage=1., clipped_calls=1),
                        bound=dict(code="MATRIX_ACTIVITY", label="偏矩阵计算限制", confidence="中", evidence=["SYNTHETIC_AIC_EVIDENCE"]),
                        scope_note="SYNTHETIC_NATIVE_WHOLE_CALL_ONLY")
    return report


class FocusExcelTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="synthetic-u-focus-")
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def export(self, report):
        before = copy.deepcopy(report)
        out = self.root / "focus.xlsx"
        renderer.build_workbook(out, report=self.root / "synthetic-core.json", report_data=report)
        self.assertEqual(report, before)
        return out, read_book(out)[0], out.with_suffix(".html").read_text(encoding="utf-8")

    def chart_roots(self, out, book, name):
        parts = layer_excel.LayerMeanRendererTests.chart_parts_by_sheet(self, out, book)[name]
        return parts

    def test_both_methods_show_their_fixed_rank_original_ids_and_unscaled_pmu(self):
        report = focus_fixture()
        line_positions = {}
        native_line = renderer.xlsx_writer.Worksheet.add_line_chart

        def capture_line(sheet, title, *args, **kwargs):
            if title.startswith("两项U逐轮耗时变化"):
                line_positions[sheet.name] = (kwargs["anchor"], kwargs["series"][0]["values_ref"])
            return native_line(sheet, title, *args, **kwargs)

        with mock.patch.object(renderer.xlsx_writer.Worksheet, "add_line_chart", new=capture_line):
            out, book, page = self.export(report)
        for name, rank, factor in (("方法1估算", "dp7", 1.), ("方法2估算", "dp3", 10.)):
            cells = book[name][1]
            text = "\n".join(str(value) for value in cells.values())
            self.assertIn("两项U重点诊断", text)
            self.assertIn("rank=" + rank, text)
            self.assertIn("不是CPU函数实测时间", text)
            self.assertIn("AscendBlockTables.compute_slot_mappings", text)
            self.assertIn("SYNTHETIC_FIXED_COMMIT", text)
            self.assertIn("97.5%", text)
            self.assertIn("77.1%", text)
            self.assertIn("0.0%", text)  # A measured zero is retained.
            self.assertIn("aic_icache_miss_rate=0.2%", text)
            self.assertIn("SYNTHETIC_AIC_EVIDENCE", text)
            self.assertIn("不是MFU", text)
            roots = self.chart_roots(out, book, name)
            lines = [root for root in roots if root.find(".//c:lineChart", CHART_NS) is not None
                     and "两项U逐轮耗时变化" in "".join(root.itertext())]
            self.assertEqual(len(lines), 1)
            _, reference = line_positions[name]
            self.assertTrue(reference.startswith("'源采样基线'!"))
            self.assertNotIn("保留序号", cells.values())
            series = lines[0].findall(".//c:lineChart/c:ser", CHART_NS)
            for index, base in enumerate((1., 4.)):
                categories = [point.findtext("c:v", namespaces=CHART_NS) for point in
                    series[index].findall("c:cat/c:strRef/c:strCache/c:pt", CHART_NS)]
                self.assertEqual(categories, ["hc-10", "hc-13", "hc-18"])
                values = [float(point.findtext("c:v", namespaces=CHART_NS)) for point in
                    series[index].findall("c:val/c:numRef/c:numCache/c:pt", CHART_NS)]
                self.assertEqual(values, [factor * base * value for value in (.5, 1., 1.5)])
            bars = [root for root in roots if "互斥细项均值" in "".join(root.itertext())]
            self.assertEqual(len(bars), 2)
        focus_headings = re.findall(r"<h4\b[^>]*>([^<]*)</h4>", page)
        self.assertEqual(focus_headings.count("两项U重点诊断 · 固定rank逐轮变化与函数入口对照"), 2)
        self.assertIn("SYNTHETIC_FULL_NATIVE_NAME_", page)
        self.assertIn("SYNTHETIC_NATIVE_WHOLE_CALL_ONLY", page)
        self.assertIn("原HC轮次ID（等距；非ProfilerStep）", page)

    def test_missing_step_remains_blank_and_line_cache_has_gap_at_original_index(self):
        report = focus_fixture()
        source = report["cases"][0]["measurements"][0]["step_residuals"]["focus_diagnostics"]["stages"][0]
        source.update(status="PARTIAL", observed_steps=2, duration_ms_mean=None,
                      duration_ms_p50=None, duration_ms_p90=None, duration_ms_max=None)
        source["per_step"][1].update(status="UNAVAILABLE", duration_ms=None, recorded_busy_ms=None, no_recorded_ms=None,
                                     reason="SYNTHETIC_MISSING_UNIQUE_PARENT")
        for item in source["summary"]:
            item.update(duration_ms_mean=None, duration_ms_p50=None, duration_ms_p90=None, duration_ms_max=None, share_of_parent=None)
        out, book, page = self.export(report)
        chart = next(root for root in self.chart_roots(out, book, "方法1估算") if
                     root.find(".//c:lineChart", CHART_NS) is not None and "两项U逐轮耗时变化" in "".join(root.itertext()))
        points = chart.findall(".//c:lineChart/c:ser/c:val/c:numRef/c:numCache", CHART_NS)[0]
        self.assertEqual(points.find("c:ptCount", CHART_NS).get("val"), "3")
        self.assertEqual([point.get("idx") for point in points.findall("c:pt", CHART_NS)], ["0", "2"])
        self.assertEqual(chart.find(".//c:dispBlanksAs", CHART_NS).get("val"), "gap")
        data = renderer._method_focus_data(report["cases"][0], report["cases"][0]["projection"]["layer_mean_methods"][0])
        self.assertIsNone(data["step_rows"][1][2])
        self.assertEqual(data["step_rows"][1][1], "hc-13")
        self.assertIn("原因：SYNTHETIC_MISSING_UNIQUE_PARENT", data["step_rows"][1][-1])
        self.assertIn("原因：SYNTHETIC_MISSING_UNIQUE_PARENT", "\n".join(map(str, book["源采样基线"][1].values())))
        self.assertIn("原因：SYNTHETIC_MISSING_UNIQUE_PARENT", page)
        self.assertFalse(data["stages"][0]["closed"])
        self.assertIn("缺少或未识别结果时断线", page)

    def test_corrupt_component_closure_disables_only_that_component_chart(self):
        report = focus_fixture()
        source = report["cases"][0]["measurements"][0]["step_residuals"]["focus_diagnostics"]["stages"][0]
        source["summary"][0]["duration_ms_mean"] += .25
        case = report["cases"][0]
        data = renderer._method_focus_data(case, case["projection"]["layer_mean_methods"][0])
        self.assertTrue(data["available"])
        self.assertFalse(data["stages"][0]["closed"])
        self.assertTrue(data["stages"][1]["closed"])
        out, book, _ = self.export(report)
        bars = [root for root in self.chart_roots(out, book, "方法1估算") if "互斥细项均值" in "".join(root.itertext())]
        self.assertEqual(len(bars), 1)
        self.assertIn("metadata", "".join(bars[0].itertext()))

    def test_wrong_rank_and_duplicate_step_are_not_silently_mixed(self):
        for mutation in ("rank", "duplicate"):
            report = focus_fixture()
            focus = report["cases"][0]["measurements"][0]["step_residuals"]["focus_diagnostics"]
            if mutation == "rank":
                focus["rank"] = "dp3"
            else:
                focus["stages"][0]["per_step"].append(copy.deepcopy(focus["stages"][0]["per_step"][0]))
            case = report["cases"][0]
            data = renderer._method_focus_data(case, case["projection"]["layer_mean_methods"][0])
            self.assertFalse(data["available"], mutation)

    def test_legacy_json_keeps_results_and_explains_missing_new_evidence(self):
        out, book, page = self.export(fixture())
        self.assertIn(519.4321, book["方法1估算"][1].values())
        self.assertIn("逐step与额外下钻未采集", page)
        rows, details = renderer._operator_pmu_data([dict(key="K", op_type="MatMul", name="SYNTHETIC", stage_label="parent")])
        self.assertEqual(rows[0][1:7], ["未采集"] * 6)
        self.assertIn("PMU未采集", details[0][1])
        self.assertNotIn("0.0%", str(rows))


if __name__ == "__main__":
    unittest.main()
