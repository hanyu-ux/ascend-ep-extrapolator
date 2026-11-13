"""Exclusive HCPre rows stay in the original layer-stage table (synthetic only)."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import test_ep128_layer_mean_excel as layer_excel
from test_ep128_v2_excel import read_book, renderer


LEGACY_KEYS = ("attention_prepare", "attention", "attention_output", "attention_post", "moe_prepare",
               "dispatch", "experts", "combine", "ffn_output", "ffn_post")
EXCLUSIVE_KEYS = ("attention_hcpre",) + LEGACY_KEYS


def split_preparation_fixture(report, fraction=.25):
    """Supply an independent 11-row core contract without estimating any totals."""
    seen = set()
    for case in report["cases"]:
        for method in case["projection"]["layer_mean_methods"]:
            group_lists = [method.get("target_stage_breakdowns") or []]
            group_lists.extend(rank.get("target_stage_breakdowns") or [] for rank in method.get("per_rank") or [])
            for groups in group_lists:
                for group in groups:
                    if id(group) in seen or group.get("status") != "MATCHED":
                        continue
                    seen.add(id(group))
                    original = group["stage_rows"][0]
                    assert original["stage_key"] == "attention_prepare"
                    hcpre = copy.deepcopy(original)
                    value = original["source_ms"] * fraction
                    hcpre.update(stage_key="attention_hcpre", label=group["attention"] + " Attention HCPre时段",
                                 source_ms=value, target_ms=value, delta_ms=0.,
                                 target_total_ms=value * group["target_layers"],
                                 cycle_share=value * group["target_layers"] / method["cycle_ms"])
                    original["source_ms"] -= value
                    original["target_ms"] -= value
                    original["target_total_ms"] = original["target_ms"] * group["target_layers"]
                    original["cycle_share"] = original["target_total_ms"] / method["cycle_ms"]
                    original["label"] = group["attention"] + " Q/KV、压缩/索引准备（HCPre结束后）"
                    group["stage_rows"].insert(0, hcpre)
                    for number, stage in enumerate(group["stage_rows"], 1):
                        stage["stage_order"] = number
    return report


class ExclusiveHCPreExcelTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="synthetic-hcpre-stage-excel-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.tools = layer_excel.LayerMeanRendererTests()

    def build(self, report, *, stem="hcpre"):
        before = copy.deepcopy(report)
        out = self.root / (stem + ".xlsx")
        # The public report no longer appends the independent business-group panel.
        with mock.patch.object(renderer, "_attention_prepare_section",
                               side_effect=AssertionError("extra Excel preparation panel")), \
             mock.patch.object(renderer, "_attention_prepare_html",
                               side_effect=AssertionError("extra HTML preparation panel")):
            renderer.build_workbook(out, report=self.root / "synthetic-core.json", report_data=report)
        self.assertEqual(report, before)
        return out, read_book(out)[0], out.with_suffix(".html").read_text(encoding="utf-8")

    def test_new_contract_places_both_exclusive_rows_in_each_original_stage_table_and_chart(self):
        report = split_preparation_fixture(layer_excel.attention_prepare_report())
        out, book, document = self.build(report)
        charts = self.tools.chart_parts_by_sheet(out, book)
        for number, method in enumerate(report["cases"][0]["projection"]["layer_mean_methods"], 1):
            name = "方法" + str(number) + "估算"
            for group in method["target_stage_breakdowns"]:
                data = renderer._target_layer_expansion_data(method, group["role"], group["attention"],
                                                             group["attention"] + "层内阶段展开")
                self.assertEqual(tuple(stage["stage_key"] for stage in group["stage_rows"]), EXCLUSIVE_KEYS)
                self.assertEqual(len(data["rows"]), 11)
                self.assertIsNotNone(data["series"], data["note"])
                self.assertIn("互不重叠", data["note"])
                self.assertIn("跨流活动", data["note"])
                self.assertIn("不是按kernel耗时累计相减", data["note"])
                self.assertEqual([row[0] for row in data["rows"]], [row["label"] for row in group["stage_rows"]])
                self.assertAlmostEqual(sum(row[1] for row in data["rows"]), group["source_wall_ms"])
                self.assertAlmostEqual(sum(row[2] for row in data["rows"]), group["target_wall_ms"])
                chart = next(item for item in charts[name]
                             if self.tools.chart_title(item).startswith(group["attention"] + "层内阶段展开"))
                series = self.tools.assert_bar_cache_matches_own_sheet(chart, book, name)
                self.assertEqual(len(series), 1)
                self.assertEqual(series[0][0], [row["label"] for row in group["stage_rows"]])
                self.assertEqual(series[0][1], [row["target_total_ms"] for row in group["stage_rows"]])
                first_row = next(int(cell[1:]) for cell, value in book[name][1].items()
                                 if cell.startswith("A") and value == group["stage_rows"][0]["label"])
                for offset, stage in enumerate(group["stage_rows"]):
                    for column, field in (("D", "source_ms"), ("E", "target_ms"),
                                          ("F", "target_total_ms"), ("H", "cycle_share")):
                        self.assertEqual(book[name][1][column + str(first_row + offset)], stage[field])
            self.assertNotIn("Attention准备细分（源实测）", "\n".join(map(str, book[name][1].values())))
            # These reports must still show the independent whole-layer Top8 sections.
            for attention in ("CSA", "HCA", "SWA"):
                self.assertIn(attention + "源实测Top8算子统计", book[name][1].values())
        self.assertNotIn('class="attention-prepare-expansion"', document)
        self.assertIn("HCPre结束后", document)

    def test_legacy_ten_rows_keep_original_values_without_inventing_hcpre(self):
        report = layer_excel.full_layer_drilldown_report()
        out, book, document = self.build(report, stem="legacy")
        charts = self.tools.chart_parts_by_sheet(out, book)
        for number, method in enumerate(report["cases"][0]["projection"]["layer_mean_methods"], 1):
            name = "方法" + str(number) + "估算"
            for group in method["target_stage_breakdowns"]:
                data = renderer._target_layer_expansion_data(method, group["role"], group["attention"], "阶段")
                self.assertEqual(len(data["rows"]), 10)
                self.assertIsNotNone(data["series"], data["note"])
                self.assertEqual(data["rows"][0][1], group["stage_rows"][0]["source_ms"])
                self.assertIn("HCPre未拆分", data["note"])
                self.assertIn("重算已有原始CSV", data["note"])
                self.assertNotIn("attention_hcpre", [row["stage_key"] for row in group["stage_rows"]])
                chart = next(item for item in charts[name]
                             if self.tools.chart_title(item).startswith(group["attention"] + "层内阶段展开"))
                series = self.tools.assert_bar_cache_matches_own_sheet(chart, book, name)
                self.assertEqual(len(series[0][0]), 10)
                self.assertEqual(series[0][1], [row["target_total_ms"] for row in group["stage_rows"]])
        self.assertIn("HCPre未拆分", document)
        self.assertNotIn("HCPre结束后", document)

    def test_split_preserves_parent_cycle_tpot_throughput_and_closed_component_cards(self):
        original = layer_excel.full_layer_drilldown_report()
        split = split_preparation_fixture(copy.deepcopy(original))
        _, previous, _ = self.build(original, stem="before")
        _, current, _ = self.build(split, stem="after")
        for before, after in zip(original["cases"][0]["projection"]["layer_mean_methods"],
                                 split["cases"][0]["projection"]["layer_mean_methods"]):
            for field in ("cycle_ms", "tpot_ms", "per_device_tokens_s", "global_tokens_s", "breakdown",
                          "class_means", "role_details", "source_partition"):
                self.assertEqual(before[field], after[field], field)
        for name in ("估算总览", "方法1估算", "方法2估算"):
            def cards(cells):
                return [[cells.get(column + str(int(cell[1:]) + 1)) for column in ("A", "E", "I")]
                        for cell, label in cells.items() if cell.startswith("A") and label == "完整周期 · ms"]
            self.assertEqual(cards(previous[name][1]), cards(current[name][1]), name)

    def test_nonclosing_or_incomplete_eleven_row_contract_suppresses_only_affected_plot(self):
        for invalid in ("closure", "missing", "duplicate_key", "duplicate_order", "bool_order"):
            report = split_preparation_fixture(layer_excel.full_layer_drilldown_report())
            case = report["cases"][0]
            method = case["projection"]["layer_mean_methods"][0]
            group = method["target_stage_breakdowns"][0]
            first = group["stage_rows"][0]
            if invalid == "closure":
                first["source_ms"] += .25
                first["target_ms"] += .25
                first["target_total_ms"] = first["target_ms"] * group["target_layers"]
                first["cycle_share"] = first["target_total_ms"] / method["cycle_ms"]
            elif invalid == "missing":
                group["stage_rows"].pop(1)
            elif invalid == "duplicate_key":
                first["stage_key"] = "attention_prepare"
            elif invalid == "duplicate_order":
                group["stage_rows"][1]["stage_order"] = 1
            else:
                first["stage_order"] = True
            with self.subTest(invalid=invalid):
                rendered = renderer._method_expansion_data(case, method)
                self.assertIsNone(rendered[0]["series"])
                self.assertIn("未绘图", rendered[0]["note"])
                self.assertIsNotNone(rendered[1]["series"], rendered[1]["note"])
                self.assertIsNotNone(rendered[2]["series"], rendered[2]["note"])

    def test_measured_zero_hcpre_or_remaining_preparation_stays_a_valid_row(self):
        for fraction, zero_index in ((0., 0), (1., 1)):
            report = split_preparation_fixture(layer_excel.full_layer_drilldown_report(), fraction=fraction)
            for method in report["cases"][0]["projection"]["layer_mean_methods"]:
                for group in method["target_stage_breakdowns"]:
                    data = renderer._target_layer_expansion_data(method, group["role"], group["attention"], "阶段")
                    self.assertIsNotNone(data["series"], data["note"])
                    self.assertEqual(data["rows"][zero_index][1:5], [0., 0., 0., 0.])
                    self.assertEqual(len(data["series"][0][1]), 11)

    def test_retained_business_helper_compares_full_preparation_window_as_first_two_rows(self):
        report = split_preparation_fixture(layer_excel.attention_prepare_report())
        method = report["cases"][0]["projection"]["layer_mean_methods"][0]
        data = renderer._attention_prepare_expansion_data(method, "main", "CSA")
        self.assertTrue(data["available"], data["note"])
        parent = method["attention_prepare_breakdowns"][0]
        parent["parent_wall_ms"] += .25
        parent["uncovered_ms"] += .25
        invalid = renderer._attention_prepare_expansion_data(method, "main", "CSA")
        self.assertFalse(invalid["available"])
        self.assertIn("源Attention完整准备窗不一致", invalid["note"])


if __name__ == "__main__":
    unittest.main()
