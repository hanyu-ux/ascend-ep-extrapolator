"""Acceptance definitions in renderer output; synthetic, no device sampling."""
import copy
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import profile_ep128 as core
from test_ep128_consolidated_trends import grid_report
from test_ep128_layer_mean_excel import layer_mean_fixture
from test_ep128_method3_excel import add_synthetic_third
from test_ep128_v2_excel import read_book, renderer


NEW = "mean_output_fraction_including_bonus"
OLD = "mean_accepted_draft_fraction"


def declare(case, rate, semantics=NEW):
    config = case["config"]
    draft = config["draft_tokens"]
    effective = (1 + draft) * rate if semantics == NEW else 1 + draft * rate
    config.update(acceptance=rate, acceptance_semantics=semantics,
                  effective_tokens_per_round=effective, synthetic_acceptance_length=effective,
                  token_basis="configured synthetic output including bonus",
                  effective_tokens_per_round_provenance="configured synthetic output")
    return case


class AcceptanceDisplayTests(unittest.TestCase):
    def test_preview_case_rate_or_sal_overrides_historical_defaults_without_reinterpreting_them(self):
        document = {"schema_version": 2, "defaults": {"draft_tokens": 7, "acceptance": .6,
                    "acceptance_semantics": OLD, "effective_tokens_per_round": 5.2},
                    "cases": [{"name": "NEW_RATE", "bs": 8, "acceptance": .65},
                              {"name": "DIRECT_SAL", "bs": 8, "synthetic_acceptance_length": 3.3}]}
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "cases.json"
            source.write_text(json.dumps(document), encoding="utf-8")
            before = source.read_bytes()
            report = renderer._preview_report(source)
            self.assertEqual(source.read_bytes(), before)
        new, sal = report["cases"]
        self.assertNotIn("effective_tokens_per_round", new["config"])
        self.assertNotIn("acceptance_semantics", new["config"])
        self.assertEqual(renderer._case_parameter_values(new), ["8", "7", "未声明", "65%"])
        self.assertIn("SAL=(1+S)×r", renderer._case_parameters(new))
        self.assertNotIn("历史SAL", renderer._case_parameters(new))
        self.assertNotIn("acceptance", sal["config"])
        self.assertNotIn("acceptance_semantics", sal["config"])
        self.assertEqual(renderer._case_parameter_values(sal)[-1], "未声明")

    def test_new_formula_is_distinct_from_historical_formula_without_changing_neff(self):
        case = layer_mean_fixture()["cases"][0]
        case["config"].update(bs=8, draft_tokens=7)
        new = declare(copy.deepcopy(case), .65)
        legacy = declare(copy.deepcopy(case), .6, OLD)
        self.assertEqual(renderer._case_parameter_values(new), ["8", "7", "5.2", "65%"])
        self.assertEqual(renderer._case_parameter_values(legacy), ["8", "7", "5.2", "60%"])
        self.assertIn("含bonus有效输出比例=65%", renderer._case_parameters(new))
        self.assertIn("SAL=(1+S)×r", renderer._case_parameters(new))
        self.assertIn("平均草稿接受比例=60%", renderer._case_parameters(legacy))
        self.assertIn("历史SAL=1+S×r", renderer._case_parameters(legacy))
        self.assertEqual(new["config"]["effective_tokens_per_round"], legacy["config"]["effective_tokens_per_round"])

    def test_acceptance_is_never_inferred_from_sal_or_an_inconsistent_declaration(self):
        case = layer_mean_fixture()["cases"][0]
        case["config"].update(bs=8, draft_tokens=7)
        valid = declare(case, .65)
        for change in ({"acceptance": None}, {"acceptance_semantics": None},
                       {"acceptance_semantics": "unknown"}, {"acceptance": .6},
                       {"acceptance": True}, {"acceptance": "0.65"},
                       {"effective_tokens_per_round": .5}):
            with self.subTest(change=change):
                candidate = copy.deepcopy(valid)
                candidate["config"].update(change)
                before = copy.deepcopy(candidate)
                self.assertIsNone(renderer._declared_acceptance(candidate))
                self.assertEqual(renderer._case_parameter_values(candidate)[-1], "未声明")
                self.assertNotIn("声明的", renderer._case_parameters(candidate))
                self.assertEqual(candidate, before)
        for rate, effective in ((.125, 1.), (1., 8.)):
            candidate = declare(copy.deepcopy(valid), rate)
            self.assertEqual(candidate["config"]["effective_tokens_per_round"], effective)
            self.assertIsNotNone(renderer._declared_acceptance(candidate))

    def test_strict_new_rate_lower_endpoint_does_not_relabel_a_lower_float_as_legal(self):
        case = layer_mean_fixture()["cases"][0]
        for draft in (0, 2, 7):
            endpoint = 1 / (1 + draft)
            for rate, valid in ((endpoint, True), (math.nextafter(endpoint, 0.), False)):
                with self.subTest(draft=draft, rate=rate):
                    candidate = copy.deepcopy(case)
                    candidate["config"].update(draft_tokens=draft, acceptance=rate,
                        acceptance_semantics=NEW, effective_tokens_per_round=1.)
                    self.assertEqual(renderer._declared_acceptance(candidate) is not None, valid)
        legacy = copy.deepcopy(case)
        legacy["config"].update(draft_tokens=7, acceptance=0., acceptance_semantics=OLD,
                                effective_tokens_per_round=1.)
        self.assertEqual(renderer._case_parameter_values(legacy)[-1], "0%")
        self.assertIsNotNone(renderer._declared_acceptance(legacy))

    def test_new_and_old_definitions_have_separate_bs_and_s_comparison_groups(self):
        report = grid_report()
        new = [declare(copy.deepcopy(case), .65) for case in report["cases"]]
        old = [declare(copy.deepcopy(case), .6, OLD) for case in report["cases"]]
        for case in new:
            case["name"] += "_NEW"
        for case in old:
            case["name"] += "_LEGACY"
        report["cases"] = new + old
        for group in renderer._bs_comparison_groups(report) + renderer._s_comparison_groups(report):
            self.assertEqual(len({case["config"]["acceptance_semantics"] for case in group["cases"]}), 1)
        for axis in ("BS", "S"):
            for identity in renderer.METHOD_IDS[:2]:
                families = renderer._trend_families(report, axis, identity)
                self.assertEqual(len(families), 2)
                self.assertEqual({family["scope"]["output_policy"] for family in families},
                                 {"fixed_output_fraction_including_bonus", "fixed_legacy_draft_acceptance_fraction"})
                self.assertTrue(any("SAL=(1+S)×r" in family["note"] for family in families))
                self.assertTrue(any("历史SAL=1+S×r" in family["note"] for family in families))

    def test_different_rates_at_the_same_bs_and_s_remain_separate_visible_conditions(self):
        report = grid_report()
        report["cases"] = [declare(copy.deepcopy(case), rate)
            for rate in (.6, .75) for case in report["cases"]]
        for number, case in enumerate(report["cases"]):
            case["name"] += "_" + str(number)
        for group in renderer._bs_comparison_groups(report) + renderer._s_comparison_groups(report):
            self.assertEqual(len({case["config"]["acceptance"] for case in group["cases"]}), 1)
        for axis in ("BS", "S"):
            families = renderer._trend_families(report, axis, "method1")
            self.assertEqual(len(families), 2)
            self.assertEqual({family["scope"]["conditions"]["acceptance"] for family in families}, {.6, .75})
            represented = [case["name"] for family in families for group in family["series"] for case in group["cases"]]
            self.assertCountEqual(represented, [case["name"] for case in report["cases"]])
            self.assertTrue(all(not family["excluded_groups"] for family in families))

    def test_sal_only_is_not_presented_as_a_fixed_or_measured_acceptance_rate(self):
        report = grid_report()
        for case in report["cases"]:
            case["config"].pop("acceptance", None)
            case["config"].pop("acceptance_semantics", None)
        for axis in ("BS", "S"):
            for family in renderer._trend_families(report, axis, "method1"):
                self.assertEqual(family["scope"]["output_policy"], "per_case_configured_neff")
                self.assertIn("非固定接受率", family["note"])
        headers, rows, _ = renderer._case_parameter_table(report)
        self.assertEqual(len(headers), 4)
        self.assertTrue(all(len(row) == 4 for row in rows))

    def test_rate_only_preview_displays_input_rate_without_deriving_neff_or_estimates(self):
        case = {"name": "SYNTHETIC_PREVIEW_RATE_ONLY", "config": {"bs": 8, "draft_tokens": 7,
            "acceptance": .65}, "source": {"status": "PREVIEW"}}
        before = copy.deepcopy(case)
        self.assertEqual(renderer._case_parameter_values(case), ["8", "7", "未声明", "65%"])
        self.assertIn("SAL=(1+S)×r", renderer._case_parameters(case))
        self.assertIn("配置输入，尚未产生估算", renderer._case_parameters(case))
        self.assertIn("配置输入，尚未产生估算", renderer._case_parameter_rows({"cases": [case]})[0][-1])
        self.assertIsNone(renderer._declared_acceptance(case), "preview input is not a normalized declaration")
        self.assertEqual(case, before)
        for change in ({"source": {"status": "SOURCE_CONDITIONAL_RATE"}}, {"source": {}}):
            candidate = {**case, **change}
            self.assertEqual(renderer._case_parameter_values(candidate)[-1], "未声明")
        for change in ({"effective_tokens_per_round": 4.8}, {"acceptance": .05},
                       {"acceptance": True}, {"acceptance_semantics": "unknown"}):
            candidate = {**case, "config": {**case["config"], **change}}
            self.assertEqual(renderer._case_parameter_values(candidate)[-1], "未声明")
        alias_case = copy.deepcopy(case)
        alias_case["config"]["num_speculative_tokens"] = alias_case["config"].pop("draft_tokens")
        self.assertEqual(renderer._case_parameter_values(alias_case), ["8", "7", "未声明", "65%"])

    def test_excel_html_show_definitions_but_preserve_all_three_methods_stored_results(self):
        report = add_synthetic_third(layer_mean_fixture())
        template = report["cases"][0]
        template["config"].update(bs=8, draft_tokens=7)
        new, legacy = declare(copy.deepcopy(template), .65), declare(copy.deepcopy(template), .6, OLD)
        new["name"], legacy["name"] = "SYNTHETIC_NEW_DEFINITION", "SYNTHETIC_OLD_DEFINITION"
        report["cases"] = [new, legacy]
        before = copy.deepcopy(report)
        captured = []
        original_save = renderer.Workbook.save

        def save(book, destination):
            captured.append(book)
            return original_save(book, destination)

        with tempfile.TemporaryDirectory(prefix="ep128-acceptance-definitions-") as temporary:
            out = Path(temporary) / "definitions.xlsx"
            with mock.patch.object(core, "read_capture", side_effect=AssertionError("rendering must not resample")), \
                    mock.patch.object(renderer.Workbook, "save", new=save):
                renderer.build_workbook(out, report=Path(temporary) / "stored.json", report_data=report)
            book = read_book(out)[0]
            page = out.with_suffix(".html").read_text(encoding="utf-8")
        self.assertEqual(report, before)
        for identity, name in zip(renderer.METHOD_IDS, ("方法1估算", "方法2估算", "方法3估算")):
            sheet = next(sheet for sheet in captured[0].sheets if sheet.name == name)
            for index, case in enumerate(report["cases"]):
                method = renderer._method(case, identity)
                for field in ("cycle_ms", "per_device_tokens_s", "tpot_ms"):
                    row, column = sheet.result_cells[(index, identity)][field]
                    self.assertEqual(float(sheet.cells[(row, column)][1]), method[field])
            values = "\n".join(map(str, book[name][1].values()))
            self.assertIn("含bonus有效输出比例=65%", values)
            self.assertIn("历史SAL=1+S×r", values)
        overview = "\n".join(map(str, book["估算总览"][1].values()))
        formula = "\n".join(map(str, book["公式与假设"][1].values()))
        for text in ("65%；含bonus有效输出比例；SAL=(1+S)×r", "60%；平均草稿接受比例；历史SAL=1+S×r"):
            self.assertIn(text, overview)
            self.assertIn(text, page)
        self.assertIn("acceptance_semantics", formula)
        self.assertIn(NEW, formula)
        self.assertIn(OLD, formula)
        self.assertIn("不修正每轮耗时或ITL代理值", formula)
        self.assertIn("不修正每轮耗时或ITL代理值", page)


if __name__ == "__main__":
    unittest.main()
