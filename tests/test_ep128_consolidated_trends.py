"""Synthetic combined sweep curves: comparison scope, gaps and editable evidence."""
import copy
from pathlib import Path
import re
import tempfile
import unittest
from unittest import mock
import xml.etree.ElementTree as ET

import profile_ep128 as core
import test_ep128_layer_mean_excel as excel_tests
import test_ep128_sweep_views as sweep_tests

renderer = excel_tests.renderer
NS = sweep_tests.NS


def grid_report():
    report = sweep_tests.sweep_report()
    cases = report["cases"]
    report["cases"] = []
    for bs in (32, 4, 16):
        for original in cases:
            case = copy.deepcopy(original)
            draft = case["config"]["draft_tokens"]
            case["name"] = f"SYNTHETIC_BS{bs}_S{draft}"
            case["config"].update(name=case["name"], bs=bs,
                token_basis=f"configured synthetic Neff={1 + draft * .55}, BS={bs}, S={draft}, bonus included",
                synthetic_acceptance_length=1 + draft * .55,
                effective_tokens_per_round_provenance=f"configured BS={bs}; S={draft}; SAL={1 + draft * .55}")
            for number, method in enumerate(case["projection"]["layer_mean_methods"], 1):
                method["per_device_tokens_s"] += bs * number * 12.25
                method["tpot_ms"] += bs / 17.
            sweep_tests.add_source_evidence(case)
            report["cases"].append(case)
    return report


class ConsolidatedTrendTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="ep128-combined-trends-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.tools = excel_tests.LayerMeanRendererTests()

    def build(self, report):
        original = copy.deepcopy(report)
        out = self.root / "combined.xlsx"
        with mock.patch.object(core, "read_capture", side_effect=AssertionError("redraw must not reread captures")):
            renderer.build_workbook(out, report=self.root / "report.json", report_data=report,
                                    expected_method_set="layer_mean_methods")
        self.assertEqual(report, original)
        book, _ = excel_tests.read_book(out)
        return out, book, self.tools.chart_parts_by_sheet(out, book)

    def trends(self, charts, axis):
        return [chart for chart in charts if chart.find(".//c:lineChart", NS) is not None
                and "随" + axis + "变化" in self.tools.chart_title(chart)]

    def test_grid_has_one_family_per_axis_and_method_despite_derived_annotation_numbers(self):
        report = grid_report()
        before = copy.deepcopy(report)
        for axis in ("BS", "S"):
            for identity in ("method1", "method2"):
                family, = renderer._trend_families(report, axis, identity)
                self.assertEqual(family["categories"], [4, 16, 32] if axis == "BS" else [3, 7, 9])
                self.assertEqual(len(family["series"]), 3)
                matrix = renderer._trend_matrix(family, "per_device_tokens_s")
                self.assertEqual(len(matrix["series"]), 3)
                for group, series in zip(family["series"], matrix["series"]):
                    self.assertEqual(series["values"], [renderer._method(group["case_by_x"][x], identity)["per_device_tokens_s"]
                                                      for x in family["categories"]])
        self.assertEqual(report, before)

    def test_metric_units_methods_and_complete_cycle_are_separate_with_matching_physical_refs(self):
        report = grid_report()
        out, book, charts = self.build(report)
        preview = out.with_suffix(".html").read_text(encoding="utf-8")
        svgs = [ET.fromstring(svg) for svg in re.findall(r"<svg\b.*?</svg>", preview, re.DOTALL)]
        for axis, categories in (("BS", [4, 16, 32]), ("S", [3, 7, 9])):
            overview = self.trends(charts["估算总览"], axis)
            self.assertEqual(len(overview), 6)
            for number, sheet in enumerate(("方法1估算", "方法2估算"), 1):
                identity = "method" + str(number)
                family, = renderer._trend_families(report, axis, identity)
                method_charts = self.trends(charts[sheet], axis)
                self.assertEqual(len(method_charts), 4)
                for field, label, _ in (*renderer.SWEEP_METRICS, ("parts", "整轮五项耗时构成", "ms/完整周期")):
                    chart, = [chart for chart in method_charts if self.tools.chart_title(chart).startswith(label + "随")]
                    title = self.tools.chart_title(chart)
                    self.assertIn(renderer._layer_mean_name({"id": identity}), title)
                    self.assertIn("条件组", title)
                    actual_series = self.tools.assert_line_cache_matches_sheet(chart, book, categories)
                    expected = renderer._trend_matrix(family, field)
                    self.assertEqual(len(actual_series), 15 if field == "parts" else 3)
                    actual = [{int(point.get("idx")): float(point.findtext("c:v", namespaces=NS))
                               for point in series.findall("c:val/c:numRef/c:numCache/c:pt", NS)}
                              for series in actual_series]
                    self.assertEqual(actual, [{i: value for i, value in enumerate(series["values"]) if value is not None}
                                              for series in expected["series"]])
                    matches = [svg for svg in svgs if svg.findtext("title") == title]
                    self.assertTrue(matches, title)
                    for svg in matches:
                        observed = []
                        for circle in svg.findall("circle"):
                            match = re.search(r"；" + axis + r"=(\d+)；([-+\d.eE]+) ", circle.findtext("title") or "")
                            self.assertIsNotNone(match)
                            observed.append((int(match[1]), float(match[2])))
                        self.assertEqual(observed, [(categories[i], value) for series in expected["series"]
                                                    for i, value in enumerate(series["values"]) if value is not None])
                        if field == "parts":
                            self.assertGreater(float(svg.get("viewBox").split()[-1]), 420)
            for chart in overview:
                title = self.tools.chart_title(chart)
                method_markers = [renderer._layer_mean_name({"id": identity}) in title for identity in ("method1", "method2")]
                self.assertEqual(sum(method_markers), 1)
                self.assertEqual(len(self.tools.assert_line_cache_matches_sheet(chart, book, categories)), 3)

    def test_union_coordinates_have_explicit_gaps_and_unavailable_method_is_independent(self):
        report = grid_report()
        report["cases"] = [case for case in report["cases"] if not
                           (case["config"]["bs"] == 16 and case["config"]["draft_tokens"] == 7)]
        missing = next(case for case in report["cases"] if case["config"]["bs"] == 16 and case["config"]["draft_tokens"] == 3)
        renderer._method(missing, "method2").update(status="UNAVAILABLE", selected_rank=None,
                                                    reasons=["SYNTHETIC_METHOD2_POINT_MISSING"])
        for identity in ("method1", "method2"):
            family, = renderer._trend_families(report, "BS", identity)
            matrix = renderer._trend_matrix(family, "cycle_ms")
            self.assertEqual(matrix["series"][1]["values"][1], None)
            self.assertEqual(matrix["series"][0]["values"][1] is None, identity == "method2")
        out, book, charts = self.build(report)
        for sheet in ("估算总览", "方法1估算", "方法2估算"):
            for chart in self.trends(charts[sheet], "BS"):
                self.tools.assert_line_cache_matches_sheet(chart, book, [4, 16, 32])
        svg = ET.fromstring(renderer._bs_svg("test", [4, 16, 32], [("gap", [1., None, 3.], "168B8D")], "ms")).find("svg")
        self.assertEqual(len(svg.findall("circle")), 2)
        self.assertEqual([len(line.get("points").split()) for line in svg.findall("polyline")], [1, 1])

    def test_duplicates_exclude_only_ambiguous_curve_without_deleting_original_case(self):
        report = grid_report()
        duplicate = copy.deepcopy(next(case for case in report["cases"]
            if case["config"]["bs"] == 16 and case["config"]["draft_tokens"] == 7))
        duplicate["name"] += "_DUPLICATE"
        report["cases"].append(duplicate)
        for axis in ("BS", "S"):
            family, = renderer._trend_families(report, axis, "method1")
            self.assertEqual(len(family["series"]), 2)
            self.assertEqual(len(family["excluded_groups"]), 1)
            self.assertIn("重复坐标", family["excluded_groups"][0]["reason"])
        _, book, _ = self.build(report)
        for name in ("估算总览", "方法1估算", "方法2估算"):
            self.assertIn(duplicate["name"], book[name][1].values())

    def test_real_scope_changes_stay_isolated_and_numeric_token_provenance_does_not(self):
        for change in ("source_model", "target_model", "target_ep", "source_devices", "source_chips_per_card",
                       "observed_ranks", "coverage", "acceptance", "token_semantics"):
            report = grid_report()
            others = copy.deepcopy(report["cases"])
            for case in others:
                case["name"] += "_OTHER"
                if change == "source_model":
                    case["source_model"]["expert_dtype"] = "SYNTHETIC_OTHER_DTYPE"
                elif change == "target_model":
                    case["target_model"]["main"]["num_hidden_layers"] += 1
                elif change == "observed_ranks":
                    case["captures"].append({"rank": "dp_extra"})
                elif change == "coverage":
                    renderer._method(case, "method1").setdefault("coverage", {})["complete"] = False
                elif change == "acceptance":
                    case["config"].update(acceptance=.55, acceptance_semantics="mean_accepted_draft_fraction")
                elif change == "token_semantics":
                    case["config"]["token_basis"] = "observed accepted output, bonus excluded"
                else:
                    case["config"][change] = 256
            report["cases"].extend(others)
            for axis in ("BS", "S"):
                with self.subTest(change=change, axis=axis):
                    self.assertEqual(len(renderer._trend_families(report, axis, "method1")), 2)

    def test_same_fixed_s_with_different_neff_is_a_separate_curve(self):
        report = grid_report()
        others = copy.deepcopy([case for case in report["cases"] if case["config"]["draft_tokens"] == 7])
        for case in others:
            case["name"] += "_OTHER_NEFF"
            case["config"].update(effective_tokens_per_round=2.5, synthetic_acceptance_length=2.5)
            case["config"]["token_basis"] = (f"configured synthetic Neff=2.5, BS={case['config']['bs']}, S=7, bonus included")
            case["config"]["effective_tokens_per_round_provenance"] = (f"configured BS={case['config']['bs']}; S=7; SAL=2.5")
        report["cases"].extend(others)
        family, = renderer._trend_families(report, "BS", "method1")
        self.assertEqual(len(family["series"]), 4)
        self.assertEqual(len({series["label"] for series in family["series"]}), 4)

    def test_provenance_window_revision_and_mismatched_axis_values_never_normalize_away(self):
        for key, before, after in (
                ("token_basis", "configured Neff={neff}; capture window=128", "configured Neff={neff}; capture window=256"),
                ("effective_tokens_per_round_provenance", "model revision 12; S={s}", "model revision 13; S={s}"),
                ("sal_provenance", "exclude first 2 warmup steps", "exclude first 3 warmup steps"),
                ("sal_source", "recorded BS={bs}", "recorded BS=4096")):
            report = grid_report()
            others = copy.deepcopy(report["cases"])
            for cases, text in ((report["cases"], before), (others, after)):
                for case in cases:
                    config = case["config"]
                    config[key] = text.format(neff=config["effective_tokens_per_round"],
                                              s=config["draft_tokens"], bs=config["bs"])
            report["cases"].extend(others)
            for axis in ("BS", "S"):
                with self.subTest(key=key, axis=axis):
                    self.assertEqual(len(renderer._trend_families(report, axis, "method1")), 2)

    def test_nearby_neff_curves_have_distinct_legends_without_rounding_numeric_values(self):
        report = grid_report()
        original = [case for case in report["cases"] if case["config"]["draft_tokens"] == 7]
        report["cases"] = []
        for neff in (5.2000001, 5.2000002):
            for case in copy.deepcopy(original):
                config = case["config"]
                config.update(effective_tokens_per_round=neff, synthetic_acceptance_length=neff,
                              token_basis="configured synthetic output including bonus",
                              effective_tokens_per_round_provenance="configured synthetic output")
                report["cases"].append(case)
        family, = renderer._trend_families(report, "BS", "method1")
        self.assertEqual([group["fixed"][1] for group in family["series"]], [5.2000001, 5.2000002])
        self.assertEqual(len({group["label"] for group in family["series"]}), 2)
        self.assertTrue(all("固定参数原值" in group["label"] for group in family["series"]))

    def test_invalid_coordinate_never_guessed_and_singleton_curves_retain_union_marker(self):
        report = grid_report()
        report["cases"] = [case for case in report["cases"] if case["config"]["bs"] ==
                           {3: 4, 7: 16, 9: 32}[case["config"]["draft_tokens"]]]
        family, = renderer._trend_families(report, "BS", "method1")
        matrix = renderer._trend_matrix(family, "cycle_ms")
        self.assertEqual(family["categories"], [4, 16, 32])
        self.assertEqual([[i for i, value in enumerate(series["values"]) if value is not None]
                          for series in matrix["series"]], [[0], [1], [2]])
        report["cases"][0]["config"]["num_speculative_tokens"] = 1000
        families = renderer._trend_families(report, "BS", "method1")
        self.assertEqual(sum(len(family["series"]) for family in families), 2)
        self.assertTrue(any("别名冲突" in item["reason"] for family in families for item in family["excluded_groups"]))


if __name__ == "__main__":
    unittest.main()
