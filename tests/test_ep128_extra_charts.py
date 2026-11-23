"""Extra U comparison charts use synthetic projected results and stored stages."""
import copy
import re
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest import mock
from zipfile import ZipFile

import test_ep128_layer_mean_excel as layer_excel
from test_ep128_layer_mean_excel import residual_report, unresolved_window_report
from test_ep128_v2_excel import read_book, renderer

import profile_ep128 as core

NS = {"c": "http://schemas.openxmlformats.org/drawingml/2006/chart",
      "s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
      "a": "http://schemas.openxmlformats.org/drawingml/2006/main"}
EXTRA_SHEET = "额外图表统计"


def extra_report():
    """Four BS values, two stored ranks and three independently closed methods."""
    report = residual_report()
    template = report["cases"][0]
    report["cases"] = []
    for bs in (16, 2, 8, 4):
        case = copy.deepcopy(template)
        case["name"] = "SYNTHETIC_BS" + str(bs) + "_S7"
        case["config"].update(bs=bs, name=case["name"], draft_tokens=7)
        scale = 1 + bs / 16
        residuals = {}
        for capture in case["measurements"]:
            diagnostic = capture["step_residuals"]
            diagnostic["residual_ms_mean"] *= scale
            residuals[capture["rank"]] = diagnostic["residual_ms_mean"]
            for field in ("stage_summary", "substage_summary", "summary"):
                for item in diagnostic[field]:
                    item["duration_ms_mean"] *= scale
        third = copy.deepcopy(renderer._method(case, "method1"))
        third.update(id="method3", name="SYNTHETIC_METHOD3_TARGET")
        case["projection"]["layer_mean_methods"].append(third)
        for number, method in enumerate(case["projection"]["layer_mean_methods"], 1):
            # Target method differences belong in a non-U component. U remains
            # this method's selected rank's stored source residual, once only.
            main_extra = bs * number + (30 if number == 3 else 0)
            for rank in method["per_rank"]:
                value = residuals[rank["rank"]]
                rank["source_partition"].update(residual_ms=value,
                    cross_role_tail_residual_ms=value)
                rank["cycle_ms"] = 27 + main_extra + value
                main = next(role for role in rank["role_details"] if role["role"] == "main")
                main.update(target_layer_envelopes_ms=20 + main_extra,
                            target_chain_ms=21 + main_extra)
            selected = next(rank for rank in method["per_rank"]
                            if rank["rank"] == method["selected_rank"])
            method.update(cycle_ms=selected["cycle_ms"],
                source_partition=copy.deepcopy(selected["source_partition"]),
                role_details=copy.deepcopy(selected["role_details"]))
            for item in method["breakdown"]:
                if item["key"] == "main_layer_envelopes":
                    item["value_ms"] = 20 + main_extra
                elif item["key"] == "residual_once":
                    item["value_ms"] = residuals[method["selected_rank"]]
            if bs == 8:
                method["breakdown"].reverse()
        report["cases"].append(case)
    return report


class ExtraResidualStatisticsTests(unittest.TestCase):
    def point(self, case, identity="method1"):
        before = copy.deepcopy(case)
        result = renderer._extra_residual_point(case, identity)
        self.assertEqual(case, before)
        return result

    def test_three_methods_use_own_target_denominators_and_selected_rank_u(self):
        report = extra_report()
        original = copy.deepcopy(report)
        groups = renderer._extra_residual_groups(report)
        self.assertEqual(report, original)
        self.assertEqual(len(groups), 3)
        self.assertEqual({group["identity"] for group in groups}, set(renderer.METHOD_IDS))
        for group in groups:
            self.assertEqual(group["s"], 7)
            self.assertFalse(group["reason"])
            self.assertEqual([point["bs"] for point in group["points"]], [2, 4, 8, 16])
            for point in group["points"]:
                case = next(case for case in report["cases"] if case["config"]["bs"] == point["bs"])
                method = renderer._method(case, group["identity"])
                expected = method["source_partition"]["residual_ms"]
                self.assertEqual(point["rank"], method["selected_rank"])
                self.assertEqual(point["u_ms"], expected)
                self.assertEqual(point["cycle_ms"], method["cycle_ms"])
                self.assertAlmostEqual(point["share"], expected / method["cycle_ms"])
                self.assertIn("%", point["label"])
                self.assertIn("ms", point["label"])
        case = report["cases"][0]
        points = [self.point(case, identity) for identity in renderer.METHOD_IDS]
        self.assertEqual(points[0]["u_ms"], points[2]["u_ms"])
        self.assertNotEqual(points[0]["share"], points[2]["share"])
        self.assertNotEqual(points[0]["u_ms"], points[1]["u_ms"])

    def test_parent_stages_close_to_u_without_adding_markov_children(self):
        for case in extra_report()["cases"]:
            for identity in renderer.METHOD_IDS:
                point = self.point(case, identity)
                self.assertEqual(len(point["items"]), 15)
                self.assertAlmostEqual(sum(item["value_ms"] for item in point["items"]), point["u_ms"])
                self.assertFalse(any("CHILD_NOT_ADDED" in item["label"] for item in point["items"]))
                self.assertAlmostEqual(sum(item["value_ms"] / point["u_ms"] for item in point["items"]), 1)

    def test_upper_aggregate_survives_old_json_without_stage_diagnostics(self):
        case = extra_report()["cases"][0]
        expected = self.point(case)
        for capture in case["measurements"]:
            capture.pop("step_residuals")
        point = self.point(case)
        self.assertEqual((point["u_ms"], point["cycle_ms"], point["share"]),
                         (expected["u_ms"], expected["cycle_ms"], expected["share"]))
        self.assertIsNone(point["items"])
        self.assertTrue(point["stage_note"])

    def test_unique_selected_per_rank_partition_is_valid_fallback(self):
        case = extra_report()["cases"][0]
        method = renderer._method(case, "method1")
        expected = self.point(case)
        method.pop("source_partition")
        method["breakdown"] = []
        point = self.point(case)
        self.assertEqual(point["u_ms"], expected["u_ms"])
        self.assertEqual(point["share"], expected["share"])
        # An unselected rank's deliberately different evidence is never copied.
        self.assertNotEqual(point["u_ms"], next(rank for rank in method["per_rank"]
            if rank["rank"] != method["selected_rank"])["source_partition"]["residual_ms"])

    def test_closed_canonical_breakdown_is_valid_old_report_fallback(self):
        case = extra_report()["cases"][0]
        method = renderer._method(case, "method1")
        expected = self.point(case)
        method.pop("source_partition")
        for rank in method["per_rank"]:
            rank.pop("source_partition")
        point = self.point(case)
        self.assertEqual(point["u_ms"], expected["u_ms"])
        self.assertEqual(point["share"], expected["share"])
        method["breakdown"][0]["value_ms"] += 3
        broken = self.point(case)
        self.assertIsNone(broken["share"])
        self.assertIsNone(broken["u_ms"])

    def test_conflicting_u_aliases_or_selected_rank_records_never_fallback_silently(self):
        for variant in ("partition_alias", "selected_rank", "breakdown", "selected_cycle"):
            with self.subTest(variant=variant):
                case = extra_report()["cases"][0]
                method = renderer._method(case, "method1")
                selected = next(rank for rank in method["per_rank"] if rank["rank"] == method["selected_rank"])
                if variant == "partition_alias":
                    method["source_partition"]["cross_role_tail_residual_ms"] += 1
                elif variant == "selected_rank":
                    selected["source_partition"]["residual_ms"] += 1
                    selected["source_partition"]["cross_role_tail_residual_ms"] += 1
                elif variant == "breakdown":
                    next(item for item in method["breakdown"] if item["key"] == "residual_once")["value_ms"] += 1
                else:
                    selected["cycle_ms"] += 1
                point = self.point(case)
                self.assertIsNone(point["share"])
                self.assertTrue(point["status"])

    def test_missing_unavailable_or_impossible_u_is_not_zero_or_another_methods_result(self):
        for variant in ("missing", "missing_method3", "unavailable", "negative", "above_cycle"):
            with self.subTest(variant=variant):
                case = extra_report()["cases"][0]
                method = renderer._method(case, "method1")
                identity = "method1"
                if variant == "missing":
                    method.pop("source_partition")
                    method["breakdown"] = []
                    for rank in method["per_rank"]:
                        rank.pop("source_partition")
                elif variant == "missing_method3":
                    case["projection"]["layer_mean_methods"] = [item for item in case["projection"]["layer_mean_methods"]
                                                               if item["id"] != "method3"]
                    identity = "method3"
                elif variant == "unavailable":
                    method.update(status="UNAVAILABLE", reasons=["SYNTHETIC_MISSING"])
                else:
                    value = -1 if variant == "negative" else method["cycle_ms"] + 1
                    method["source_partition"].update(residual_ms=value, cross_role_tail_residual_ms=value)
                point = self.point(case, identity)
                self.assertIsNone(point["share"])
                self.assertTrue(point["status"])

    def test_explicit_zero_u_is_distinct_from_missing_evidence(self):
        case = extra_report()["cases"][0]
        method = renderer._method(case, "method1")
        selected = next(rank for rank in method["per_rank"] if rank["rank"] == method["selected_rank"])
        old = method["source_partition"]["residual_ms"]
        for record in (method, selected):
            record["source_partition"].update(residual_ms=0., cross_role_tail_residual_ms=0.)
            record["cycle_ms"] -= old
        next(item for item in method["breakdown"] if item["key"] == "residual_once")["value_ms"] = 0.
        point = self.point(case)
        self.assertEqual(point["u_ms"], 0.)
        self.assertEqual(point["share"], 0.)
        self.assertIsNone(point["items"], "nonzero stored stages cannot be normalized by zero U")

    def test_incomplete_parent_preserves_upper_without_fabricating_lower_zeros(self):
        case = extra_report()["cases"][0]
        expected = self.point(case)
        selected = next(capture for capture in case["measurements"] if capture["rank"] == expected["rank"])
        selected["step_residuals"]["stage_summary"].pop(0)
        point = self.point(case)
        self.assertEqual(point["share"], expected["share"])
        self.assertIsNone(point["items"])
        self.assertTrue(point["stage_rows"])
        self.assertTrue(point["stage_note"])

    def test_unresolved_whole_window_is_preserved_as_total_not_eight_fake_stages(self):
        case = unresolved_window_report()["cases"][0]
        point = self.point(case)
        self.assertIsNotNone(point["share"])
        self.assertEqual(len(point["items"]), 8)
        unknown, = [item for item in point["items"] if item.get("unresolved_window")]
        self.assertEqual(unknown["value_ms"], 3.25)
        self.assertAlmostEqual(sum(item["value_ms"] for item in point["items"]), point["u_ms"])
        self.assertIn("未识别", point["stage_note"])

    def test_grouping_separates_fixed_s_and_conditions_and_preserves_duplicate_reason(self):
        report = extra_report()
        others = copy.deepcopy(report["cases"])
        for case in others:
            case["name"] += "_OTHER_EP"
            case["config"]["target_ep"] = 256
        report["cases"].extend(others)
        self.assertEqual(len(renderer._extra_residual_groups(report)), 6)
        duplicate = copy.deepcopy(report["cases"][0])
        duplicate["name"] += "_DUPLICATE"
        report["cases"].append(duplicate)
        groups = renderer._extra_residual_groups(report)
        invalid = [group for group in groups if group["reason"]]
        self.assertEqual(len(invalid), 3)
        self.assertTrue(all("重复" in group["reason"] for group in invalid))
        self.assertEqual(len(report["cases"]), 9)
        for bad in (None, True, "7"):
            one = extra_report()
            for case in one["cases"]:
                case["config"]["draft_tokens"] = bad
            self.assertTrue(all(group["reason"] for group in renderer._extra_residual_groups(one)))
        for bad in (None, True, 0, "2"):
            one = extra_report()
            one["cases"][0]["config"]["bs"] = bad
            self.assertTrue(all(group["reason"] for group in renderer._extra_residual_groups(one)))


class ExtraResidualWorkbookTests(unittest.TestCase):
    def test_native_combo_uses_editable_percent_cells_equal_top_points_and_ms_labels(self):
        report = extra_report()
        original = copy.deepcopy(report)
        with tempfile.TemporaryDirectory(prefix="ep128-extra-charts-") as directory:
            out = Path(directory) / "extra.xlsx"
            with mock.patch.object(core, "read_capture", side_effect=AssertionError("redraw must not read CSV")):
                renderer.build_workbook(out, report=Path(directory) / "stored.json", report_data=report)
            self.assertEqual(report, original)
            book, _ = read_book(out)
            self.assertEqual(list(book)[-1], EXTRA_SHEET)
            tools = layer_excel.LayerMeanRendererTests()
            charts = tools.chart_parts_by_sheet(out, book)[EXTRA_SHEET]
            combos = [chart for chart in charts if chart.find(".//c:barChart", NS) is not None
                      and chart.find(".//c:lineChart", NS) is not None]
            self.assertEqual(len(combos), 3)
            expected_groups = renderer._extra_residual_groups(report)
            for chart, group in zip(combos, expected_groups):
                bar = chart.find(".//c:barChart", NS)
                line = chart.find(".//c:lineChart", NS)
                self.assertEqual(bar.find("c:barDir", NS).get("val"), "col")
                self.assertEqual([node.get("val") for node in bar.findall("c:axId", NS)],
                                 [node.get("val") for node in line.findall("c:axId", NS)])
                self.assertEqual(len(chart.findall(".//c:valAx", NS)), 1)
                expected = [point["share"] for point in group["points"]]
                for plot in (bar, line):
                    series, = plot.findall("c:ser", NS)
                    values = series.find("c:val/c:numRef", NS)
                    self.assertIsNotNone(values)
                    reference = values.findtext("c:f", namespaces=NS)
                    self.assertTrue(reference.startswith("'" + EXTRA_SHEET + "'!"))
                    self.assertEqual(tools.chart_reference_cells(book, reference), expected)
                    self.assertEqual([float(node.text) for node in values.findall("c:numCache/c:pt/c:v", NS)], expected)
                    categories = series.find("c:cat/c:strRef", NS)
                    self.assertIsNotNone(categories)
                    self.assertEqual([int(node.text) for node in categories.findall("c:strCache/c:pt/c:v", NS)],
                                     [2, 4, 8, 16])
                # One label per shared top point avoids duplicate text from
                # the column and line, whose numeric values are identical.
                custom = line.findall("c:ser/c:dLbls/c:dLbl", NS)
                self.assertEqual(len(custom), 4)
                for label in custom:
                    text = "".join(label.find("c:tx", NS).itertext())
                    self.assertIn("%", text)
                    self.assertIn("ms", text)
                    linked = label.find("c:tx/c:strRef/c:f", NS)
                    self.assertIsNotNone(linked)
                    self.assertTrue(linked.text.startswith("'" + EXTRA_SHEET + "'!"))
            lower = [chart for chart in charts if chart not in combos
                     and 'U主要耗时 ·' not in ''.join(chart.find('c:chart/c:title', NS).itertext())]
            self.assertEqual(len(lower), 12)
            for chart in lower:
                series, = chart.findall(".//c:barChart/c:ser", NS)
                cache = series.find("c:val/c:numRef/c:numCache", NS)
                values = [float(node.text) for node in cache.findall("c:pt/c:v", NS)]
                self.assertEqual(len(values), 15)
                self.assertAlmostEqual(sum(values), 1)
                labels = series.findall("c:dLbls/c:dLbl/c:tx", NS)
                self.assertEqual(len(labels), 15)
                self.assertTrue(all("ms" in "".join(label.itertext()) and "%" in "".join(label.itertext())
                                    for label in labels))
            links = book[EXTRA_SHEET][0].findall("s:hyperlinks/s:hyperlink", NS)
            self.assertTrue(any("源采样基线" in link.get("location", "") for link in links))
            self.assertTrue(any("估算总览" in link.get("location", "") for link in links))
            preview = out.with_suffix(".html").read_text(encoding="utf-8")
            self.assertIn(EXTRA_SHEET, preview)
            self.assertIn("%", renderer._extra_charts_html(report))
            self.assertIn("ms", renderer._extra_charts_html(report))
            svgs = [ET.fromstring(svg) for svg in re.findall(r"<svg\b.*?</svg>",
                                                          renderer._extra_charts_html(report), re.DOTALL)]
            self.assertGreaterEqual(len(svgs), 3)
            with ZipFile(out) as archive:
                self.assertIsNone(archive.testzip())


if __name__ == "__main__":
    unittest.main()
