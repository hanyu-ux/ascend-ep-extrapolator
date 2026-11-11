"""Synthetic BS/S comparison views preserve source and target evidence separately."""
import copy
import math
from pathlib import Path
import re
import tempfile
import unittest
from unittest import mock
import xml.etree.ElementTree as ET

import profile_ep128 as core
import test_ep128_layer_mean_excel as excel_tests

renderer = excel_tests.renderer
NS = {"c": "http://schemas.openxmlformats.org/drawingml/2006/chart"}
PARTS = ("main_layer_envelopes", "main_internal_gaps", "draft_layer_envelopes",
         "draft_internal_gaps", "residual_once")


def source_parts_ns(bs, draft):
    # Independent hand-defined source timeline; never borrowed from target
    # role_details/source_partition, which deliberately contain other values.
    values = (10000000 + draft * 250000, 1120000, 3000000 + draft * 125000,
              380000, 2250000 + draft * 62500)
    return [value * bs // 8 for value in values]


def _divide(total, count):
    quotient, remainder = divmod(total, count)
    return [quotient + (index < remainder) for index in range(count)]


def add_source_evidence(case):
    bs, draft = case["config"]["bs"], case["config"]["draft_tokens"]
    effective = case["config"]["effective_tokens_per_round"]
    selected = "dp7" if (draft in (3, 9)) == (bs in (8, 32)) else "dp3"
    base = source_parts_ns(bs, draft)
    case["measurements"], case["captures"] = [], []
    per_rank = []
    for rank in ("dp7", "dp3"):
        rounds = []
        for index, factor in enumerate((1., 2.) if rank == selected else (.25, .5)):
            parts = [round(value * factor) for value in base]
            start = cursor = 1793103275000000000 + index * 1000000000
            layers = []
            for role, total, gap, count in (("main", parts[0], parts[1], 8),
                                             ("draft", parts[2], parts[3], 3)):
                if role == "draft":
                    cursor += parts[4] // 3
                gaps = _divide(gap, count - 1)
                for number, duration in enumerate(_divide(total, count)):
                    attention = "SWA" if role == "draft" else "CSA" if number in (2, 4, 6) else "HCA"
                    layers.append({"role": role, "layer": number, "attention": attention,
                        "class_key": role + "_" + attention + "_normal", "scope": "HC_marker_interval",
                        "start_ns": str(cursor), "end_ns": str(cursor + duration), "wall_ms": duration / 1e6})
                    cursor += duration
                    if number < len(gaps):
                        cursor += gaps[number]
            cursor += parts[4] - parts[4] // 3
            rounds.append({"round_id": "hc-" + str(index + 3), "start_ns": str(start),
                "end_ns": str(cursor), "duration_ms": (cursor - start) / 1e6, "layers": layers,
                "output_tokens": bs * effective, "token_basis": "configured synthetic Neff including bonus",
                "scope": "HC_cycle_not_verified_request_iteration", "ledger": {"closed": True}})
        mean = math.fsum(item["duration_ms"] for item in rounds) / len(rounds)
        rate = 1000 * bs * effective * case["config"]["source_dp"] / case["config"]["source_devices"] / mean
        case["measurements"].append({"rank": rank, "rounds": rounds})
        case["captures"].append({"rank": rank, "sha256": "SYNTHETIC_" + rank + "_" + str(bs) + "_" + str(draft)})
        per_rank.append({"rank": rank, "rounds": len(rounds), "cycle_ms_mean": mean,
                         "conditional_per_device_tokens_s": rate})
    chosen = next(item for item in per_rank if item["rank"] == selected)
    case["source"] = {"status": "SOURCE_CONDITIONAL_RATE", "per_rank": per_rank, "class_stats": [],
        "aggregate": {"selected_rank": selected, "cycle_ms": chosen["cycle_ms_mean"],
            "per_device_tokens_s": chosen["conditional_per_device_tokens_s"],
            "rank_coverage_complete": False, "captured_ranks": 2, "expected_ranks": 16,
            "aggregation": "max_of_rank_mean_proxy", "scope": "configured synthetic conditional source"}}


def sweep_report(*, axis="S"):
    report = excel_tests.layer_mean_fixture()
    template = report["cases"][0]
    report["cases"] = []
    for value in ((9, 3, 7) if axis == "S" else (32, 4, 16)):
        draft, bs = (value, 8) if axis == "S" else (7, value)
        case = copy.deepcopy(template)
        case["name"] = "SYNTHETIC_LABEL_S" + str(100 - value)
        case["config"].update(name=case["name"], bs=bs, draft_tokens=draft,
            effective_tokens_per_round=1 + draft * .55, token_basis="configured synthetic output including bonus",
            draft_layers=3, draft_passes=1, source_dp=8, source_devices=16, source_chips_per_card=1,
            expected_ranks=16, inputs=[{"rank": "dp7", "path": "synthetic/" + str(value) + ".csv"}])
        for key in ("acceptance", "acceptance_semantics", "synthetic_acceptance_length",
                    "synthetic_acception_length", "num_speculative_tokens"):
            case["config"].pop(key, None)
        for number, method in enumerate(case["projection"]["layer_mean_methods"], 1):
            selected = "dp7" if (draft == 7) == (number == 1) else "dp3"
            components = [20.125 + value + number * 2, 1 + value / 16.,
                          5 + number + value / 4., .5 + value / 32., 2 + number + value / 8.]
            values = dict(zip(PARTS, components))
            for part in method["breakdown"]:
                part["value_ms"] = values[part["key"]]
            if value == 7:
                method["breakdown"].reverse()
            method.update(selected_rank=selected, cycle_ms=math.fsum(components),
                          tpot_ms=number * 20. + value / 7., per_device_tokens_s=number * 1000. + value * 11.125)
            method["role_details"][0].update(target_layer_envelopes_ms=components[0],
                target_gap_ms=components[1], target_chain_ms=math.fsum(components[:2]))
            method["role_details"][1].update(target_layer_envelopes_ms=components[2],
                target_gap_ms=components[3], target_chain_ms=math.fsum(components[2:4]))
            method["source_partition"].update(source_cycle_ms=998877., residual_ms=components[4],
                                               cross_role_tail_residual_ms=components[4])
            for rank in method["per_rank"]:
                rank["cycle_ms"] = method["cycle_ms"] if rank["rank"] == selected else 900000. + value
        add_source_evidence(case)
        report["cases"].append(case)
    return report


def source_expected(case):
    values = [value * 1.5 / 1e6 for value in source_parts_ns(case["config"]["bs"], case["config"]["draft_tokens"])]
    cycle = math.fsum(values)
    return {"parts": values, "cycle_ms": cycle, "itl_proxy_ms": cycle,
            "tpot_ms": cycle / case["config"]["effective_tokens_per_round"],
            "per_device_tokens_s": case["source"]["aggregate"]["per_device_tokens_s"]}


class SweepViewTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="ep128-sweep-view-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.tools = excel_tests.LayerMeanRendererTests()

    def build(self, report):
        before = copy.deepcopy(report)
        out = self.root / "sweeps.xlsx"
        with mock.patch.object(core, "read_capture", side_effect=AssertionError("old JSON redraw must not rescan CSV")):
            renderer.build_workbook(out, report=self.root / "old-report.json", report_data=report,
                                    expected_method_set="layer_mean_methods")
        self.assertEqual(report, before)
        book, _ = excel_tests.read_book(out)
        self.assertEqual(len(book), 10)
        self.assertIn("核心未提供本方法", book["方法3估算"][1].values())
        self.assertIn("核心报告未输出方法三", " ".join(map(str, book["方法3估算"][1].values())))
        return out, book, self.tools.chart_parts_by_sheet(out, book)

    def title(self, chart):
        return self.tools.chart_title(chart)

    def line_values(self, chart, book, categories):
        series = self.tools.assert_line_cache_matches_sheet(chart, book, categories)
        return [{int(point.get("idx")): float(point.findtext("c:v", namespaces=NS))
                 for point in item.findall("c:val/c:numRef/c:numCache/c:pt", NS)} for item in series]

    def trend(self, charts, prefix, axis, identity=None):
        selected = [chart for chart in charts if self.title(chart).removeprefix("源采样基线 · ").startswith(prefix)
                    and "随" + axis + "变化" in self.title(chart)
                    and (identity is None or renderer._layer_mean_name({"id": identity}) in self.title(chart))]
        self.assertEqual(len(selected), 1, [self.title(chart) for chart in charts])
        return selected[0]

    def proxy_trend(self, charts, axis, identity=None):
        selected = [chart for chart in charts if "ITL代理" in self.title(chart) and "随" + axis + "变化" in self.title(chart)
                    and (identity is None or renderer._layer_mean_name({"id": identity}) in self.title(chart))]
        self.assertEqual(len(selected), 1, [self.title(chart) for chart in charts])
        return selected[0]

    def assert_html_points_match_excel(self, preview, charts, book, categories, axis):
        svgs = [ET.fromstring(svg) for svg in re.findall(r"<svg\b.*?</svg>", preview, re.DOTALL)]
        for sheet, by_sheet in charts.items():
            for chart in by_sheet:
                title = self.title(chart)
                if "随" + axis + "变化" not in title:
                    continue
                expected = [(categories[index], value) for values in self.line_values(chart, book, categories)
                            for index, value in values.items()]
                expected_title = " · ".join(title.split(" · ")[:2]) if sheet == "源采样基线" else title
                found = [svg for svg in svgs if svg.findtext("title") == expected_title]
                self.assertTrue(found, title)
                for svg in found:
                    points = []
                    for circle in svg.findall("circle"):
                        match = re.search(r"；" + axis + r"=(\d+)；([-+\d.eE]+) ", circle.findtext("title") or "")
                        self.assertIsNotNone(match)
                        points.append((int(match[1]), float(match[2])))
                    self.assertEqual(points, expected, title)

    def test_s_group_uses_numeric_config_order_and_allows_neff_to_change(self):
        report = sweep_report()
        before = copy.deepcopy(report)
        groups = renderer._s_comparison_groups(report)
        self.assertEqual(report, before)
        self.assertEqual(len(groups), 1)
        self.assertEqual([case["config"]["draft_tokens"] for case in groups[0]["cases"]], [3, 7, 9])
        self.assertEqual(len({case["config"]["effective_tokens_per_round"] for case in groups[0]["cases"]}), 3)
        self.assertFalse(renderer._s_chart_reason(groups[0]))

    def test_s_accepts_only_explicit_valid_integer_aliases_and_never_guesses_names(self):
        case = sweep_report()["cases"][0]
        self.assertEqual(renderer._s_value(case), 9)
        case["config"]["num_speculative_tokens"] = case["config"].pop("draft_tokens")
        self.assertEqual(renderer._s_value(case), 9)
        case["config"]["draft_tokens"] = 9
        self.assertEqual(renderer._s_value(case), 9)
        case["config"]["num_speculative_tokens"] = 7
        self.assertIsNone(renderer._s_value(case))
        for value in (True, -1, 7.0, "7", None):
            with self.subTest(value=value):
                case["config"] = {"draft_tokens": value}
                self.assertIsNone(renderer._s_value(case))
        case["config"] = {"draft_tokens": 0}
        self.assertEqual(renderer._s_value(case), 0)
        case["config"] = {}
        case["name"] = "SYNTHETIC_NAME_ONLY_S7"
        self.assertIsNone(renderer._s_value(case))

    def test_s_views_copy_two_methods_and_independent_source_values_into_editable_charts(self):
        report = sweep_report()
        out, book, charts = self.build(report)
        ordered = sorted(report["cases"], key=lambda case: case["config"]["draft_tokens"])
        categories = [3, 7, 9]
        for number in (0, 1):
            identity = "method" + str(number + 1)
            for prefix, field in (("单卡吞吐随S变化", "per_device_tokens_s"), ("TPOT随S变化", "tpot_ms")):
                values = self.line_values(self.trend(charts["估算总览"], prefix, "S", identity), book, categories)
                self.assertEqual(values, [{i: case["projection"]["layer_mean_methods"][number][field]
                                          for i, case in enumerate(ordered)}])
            values = self.line_values(self.proxy_trend(charts["估算总览"], "S", identity), book, categories)
            self.assertEqual(values, [{i: case["projection"]["layer_mean_methods"][number]["cycle_ms"]
                                      for i, case in enumerate(ordered)}])
        for number, name in enumerate(("方法1估算", "方法2估算")):
            chart = self.trend(charts[name], "整轮五项耗时构成随S变化", "S")
            actual = self.line_values(chart, book, categories)
            self.assertEqual(actual, [{i: next(part["value_ms"] for part in
                case["projection"]["layer_mean_methods"][number]["breakdown"] if part["key"] == key)
                for i, case in enumerate(ordered)} for key in PARTS])
        for prefix, field in (("单卡吞吐随S变化", "per_device_tokens_s"), ("TPOT随S变化", "tpot_ms")):
            actual = self.line_values(self.trend(charts["源采样基线"], prefix, "S"), book, categories)
            self.assertEqual(actual, [{i: source_expected(case)[field] for i, case in enumerate(ordered)}])
        actual = self.line_values(self.proxy_trend(charts["源采样基线"], "S"), book, categories)
        self.assertEqual(actual, [{i: source_expected(case)["cycle_ms"] for i, case in enumerate(ordered)}])
        actual = self.line_values(self.trend(charts["源采样基线"], "整轮五项耗时构成随S变化", "S"), book, categories)
        self.assertEqual(actual, [{i: source_expected(case)["parts"][component] for i, case in enumerate(ordered)}
                                  for component in range(5)])
        preview = out.with_suffix(".html").read_text(encoding="utf-8")
        self.assertIn("ITL代理", preview)
        self.assertIn("Neff", preview)
        self.assertIn("非固定接受率", preview)
        self.assertNotIn("998877", " ".join(map(str, book["源采样基线"][1].values())))
        self.assert_html_points_match_excel(preview, charts, book, categories, "S")

    def test_source_bs_views_are_added_while_original_target_bs_values_remain(self):
        report = sweep_report(axis="BS")
        out, book, charts = self.build(report)
        ordered = sorted(report["cases"], key=lambda case: case["config"]["bs"])
        categories = [4, 16, 32]
        for prefix, field in (("单卡吞吐随BS变化", "per_device_tokens_s"), ("TPOT随BS变化", "tpot_ms")):
            actual = self.line_values(self.trend(charts["源采样基线"], prefix, "BS"), book, categories)
            self.assertEqual(actual, [{i: source_expected(case)[field] for i, case in enumerate(ordered)}])
            for number in (0, 1):
                targets = self.line_values(self.trend(charts["估算总览"], prefix, "BS", "method" + str(number + 1)), book, categories)
                self.assertEqual(targets, [{i: case["projection"]["layer_mean_methods"][number][field]
                                           for i, case in enumerate(ordered)}])
        proxy = self.line_values(self.proxy_trend(charts["源采样基线"], "BS"), book, categories)
        self.assertEqual(proxy, [{i: source_expected(case)["cycle_ms"] for i, case in enumerate(ordered)}])
        chart = self.trend(charts["源采样基线"], "整轮五项耗时构成随BS变化", "BS")
        self.assertEqual(self.line_values(chart, book, categories),
                         [{i: source_expected(case)["parts"][component] for i, case in enumerate(ordered)}
                          for component in range(5)])
        self.assert_html_points_match_excel(out.with_suffix(".html").read_text(encoding="utf-8"),
                                            charts, book, categories, "BS")

    def test_missing_middle_method_breaks_only_its_s_series(self):
        report = sweep_report()
        middle = next(case for case in report["cases"] if case["config"]["draft_tokens"] == 7)
        middle["projection"]["layer_mean_methods"][1].update(status="UNAVAILABLE", selected_rank=None,
                                                                reasons=["SYNTHETIC_S7_MISSING"])
        out, book, charts = self.build(report)
        for prefix in ("单卡吞吐随S变化", "TPOT随S变化"):
            for identity, expected in (("method1", [0, 1, 2]), ("method2", [0, 2])):
                values = self.line_values(self.trend(charts["估算总览"], prefix, "S", identity), book, [3, 7, 9])
                self.assertEqual([sorted(item) for item in values], [expected])
        values = self.line_values(self.trend(charts["方法2估算"], "整轮五项耗时构成随S变化", "S"), book, [3, 7, 9])
        self.assertEqual([sorted(item) for item in values], [[0, 2]] * 5)
        source = self.line_values(self.trend(charts["源采样基线"], "整轮五项耗时构成随S变化", "S"), book, [3, 7, 9])
        self.assertEqual([sorted(item) for item in source], [[0, 1, 2]] * 5)
        svgs = [ET.fromstring(svg) for svg in re.findall(r"<svg\b.*?</svg>", out.with_suffix(".html").read_text(), re.DOTALL)]
        target = next(svg for svg in svgs if (svg.findtext("title") or "").startswith("单卡吞吐随S变化")
                      and renderer._layer_mean_name({"id": "method2"}) in svg.findtext("title"))
        self.assertEqual([len(line.get("points").split()) for line in target.findall("polyline")], [1, 1])
        self.assertEqual(len(target.findall("circle")), 2)
        self.assert_html_points_match_excel(out.with_suffix(".html").read_text(encoding="utf-8"),
                                            charts, book, [3, 7, 9], "S")

    def test_invalid_source_partition_blanks_only_source_five_parts(self):
        report = sweep_report()
        middle = next(case for case in report["cases"] if case["config"]["draft_tokens"] == 7)
        chosen = middle["source"]["aggregate"]["selected_rank"]
        measured = next(item for item in middle["measurements"] if item["rank"] == chosen)
        measured["rounds"][0]["layers"][0]["wall_ms"] += .125
        summary = core.source_sweep_summary(middle)
        self.assertTrue(summary["breakdown_reason"])
        self.assertTrue(all(item["value_ms"] is None for item in summary["breakdown"]))
        out, book, charts = self.build(report)
        values = self.line_values(self.trend(charts["源采样基线"], "整轮五项耗时构成随S变化", "S"), book, [3, 7, 9])
        self.assertEqual([sorted(item) for item in values], [[0, 2]] * 5)
        for prefix in ("单卡吞吐随S变化", "TPOT随S变化"):
            values = self.line_values(self.trend(charts["源采样基线"], prefix, "S"), book, [3, 7, 9])
            self.assertEqual([sorted(item) for item in values], [[0, 1, 2]])
        for name in ("方法1估算", "方法2估算"):
            values = self.line_values(self.trend(charts[name], "整轮五项耗时构成随S变化", "S"), book, [3, 7, 9])
            self.assertEqual([sorted(item) for item in values], [[0, 1, 2]] * 5)
        self.assert_html_points_match_excel(out.with_suffix(".html").read_text(encoding="utf-8"),
                                            charts, book, [3, 7, 9], "S")

    def test_duplicate_or_missing_s_keeps_all_rows_and_suppresses_only_ambiguous_trends(self):
        for invalid in ("duplicate", "missing"):
            report = sweep_report()
            if invalid == "duplicate":
                duplicate = copy.deepcopy(report["cases"][1])
                duplicate["name"] = "SYNTHETIC_REPEATED_S3"
                report["cases"].append(duplicate)
            else:
                report["cases"][1]["config"].pop("draft_tokens")
                report["cases"][1]["name"] = "SYNTHETIC_NAME_ONLY_S3"
            with self.subTest(invalid=invalid):
                groups = renderer._s_comparison_groups(report)
                self.assertEqual(len(groups), 1)
                self.assertTrue(renderer._s_chart_reason(groups[0]))
                _, book, charts = self.build(report)
                for name in ("估算总览", "方法1估算", "方法2估算", "源采样基线"):
                    self.assertFalse(any("随S变化" in self.title(chart) for chart in charts[name]))
                self.assertTrue(all(case["name"] in book["估算总览"][1].values() for case in report["cases"]))

    def test_different_bs_model_topology_or_explicit_acceptance_policy_splits_s_groups(self):
        for changed in ("bs", "target_model", "target_ep", "observed_ranks", "acceptance"):
            report = sweep_report()
            other = copy.deepcopy(report["cases"])
            for case in other:
                case["name"] += "_OTHER"
                if changed == "bs":
                    case["config"]["bs"] = 16
                elif changed == "target_model":
                    case["target_model"]["main"]["num_hidden_layers"] += 1
                elif changed == "target_ep":
                    case["config"]["target_ep"] = 256
                elif changed == "observed_ranks":
                    case["captures"].append({"rank": "dp_extra", "sha256": "SYNTHETIC_EXTRA"})
                else:
                    case["config"].update(acceptance=.5, acceptance_semantics="mean_accepted_draft_fraction")
            report["cases"].extend(other)
            before = copy.deepcopy(report)
            with self.subTest(changed=changed):
                groups = renderer._s_comparison_groups(report)
                self.assertEqual(report, before)
                self.assertEqual(len(groups), 2)
                self.assertEqual([[case["config"]["draft_tokens"] for case in group["cases"]] for group in groups],
                                 [[3, 7, 9], [3, 7, 9]])
