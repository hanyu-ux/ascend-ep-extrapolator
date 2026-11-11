"""Configured output-fraction acceptance arithmetic, never measured acceptance."""
import copy
import math
from pathlib import Path
import unittest

import profile_ep128 as core
from tests.test_ep128_layer_mean import attach
from tests.test_ep128_source_rank import source_rank_report


def configuration(**values):
    return dict(name="configured-acceptance", bs=4, draft_tokens=7, with_stack=False,
                base_ep=16, target_ep=128, expected_ranks=16, source_dp=16,
                source_devices=16, target_dp=128, target_devices=128, **values)


def normalized(defaults=None, **values):
    item = configuration()
    item.update(values)
    return core.normalize_case(defaults or {}, item, Path.cwd())


class AcceptanceOutputFractionTests(unittest.TestCase):
    def test_rate_alone_uses_bonus_inclusive_fraction_and_exact_decimal_arithmetic(self):
        for steps, expected in ((3, 2.4), (5, 3.6), (7, 4.8), (9, 6.0)):
            with self.subTest(steps=steps):
                value = normalized(draft_tokens=steps, acceptance=.6)
                self.assertEqual(value["effective_tokens_per_round"], expected)
                self.assertEqual(value["acceptance_semantics"], core.ACCEPTANCE_SEMANTICS)
                self.assertIn("SAL=(1+S)*r", value["token_basis"])
                self.assertIn("not measured", value["token_basis"])
        self.assertEqual(normalized(acceptance=.65)["effective_tokens_per_round"], 5.2)

    def test_per_case_rates_override_default_policy_and_leave_inputs_unchanged(self):
        defaults = {"acceptance": .6, "acceptance_semantics": core.ACCEPTANCE_SEMANTICS,
                    "synthetic_acceptance_length": 4.8}
        before = copy.deepcopy(defaults)
        for rate, expected in ((.5, 4), (.65, 5.2), (.9, 7.2)):
            with self.subTest(rate=rate):
                result = normalized(defaults, acceptance=rate)
                self.assertEqual(result["effective_tokens_per_round"], expected)
                self.assertNotIn("synthetic_acceptance_length", result)
        self.assertEqual(defaults, before)

    def test_changed_s_recomputes_inherited_rate_instead_of_reusing_normalized_length(self):
        defaults = normalized(acceptance=.6)
        original = copy.deepcopy(defaults)
        result = normalized(defaults, draft_tokens=9)
        self.assertEqual(result["effective_tokens_per_round"], 6)
        self.assertEqual(result["acceptance"], .6)
        self.assertEqual(defaults, original)
        result = normalized({"acceptance": .6, "synthetic_acceptance_length": 4.8},
                            draft_tokens=9)
        self.assertEqual(result["effective_tokens_per_round"], 6)
        self.assertNotIn("synthetic_acceptance_length", result)

    def test_s_alias_inheritance_recomputes_and_case_rate_uses_new_definition(self):
        defaults = normalized(acceptance=.6)
        item = {"name": "S9", "bs": 4, "num_speculative_tokens": 9}
        value = core.normalize_case(defaults, item, Path.cwd())
        self.assertEqual(value["draft_tokens"], 9)
        self.assertEqual(value["effective_tokens_per_round"], 6)
        self.assertNotIn("num_speculative_tokens", value)
        defaults["acceptance_semantics"] = "mean_accepted_draft_fraction"
        value = normalized(defaults, acceptance=.65)
        self.assertEqual(value["effective_tokens_per_round"], 5.2)
        self.assertEqual(value["acceptance_semantics"], core.ACCEPTANCE_SEMANTICS)

    def test_explicit_sal_and_neff_must_agree_with_rate_without_adding_bonus(self):
        consistent = dict(acceptance=.65, synthetic_acceptance_length=5.2,
                          effective_tokens_per_round=5.2, token_basis="Configured SAL including bonus")
        self.assertEqual(normalized(**consistent)["effective_tokens_per_round"], 5.2)
        for change in ({"acceptance": .6}, {"synthetic_acceptance_length": 4.8},
                       {"effective_tokens_per_round": 4.8}):
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "conflicts"):
                normalized(**{**consistent, **change})
        for field in ("synthetic_acceptance_length", "synthetic_acception_length"):
            self.assertEqual(normalized(acceptance=.65, **{field: 5.2})["effective_tokens_per_round"], 5.2)

    def test_independent_sal_overrides_default_rate_including_historical_defaults(self):
        for semantics in (core.ACCEPTANCE_SEMANTICS, "mean_accepted_draft_fraction"):
            defaults = {"acceptance": .6, "acceptance_semantics": semantics,
                        "effective_tokens_per_round": 5.2}
            value = normalized(defaults, synthetic_acceptance_length=3.3)
            self.assertEqual(value["effective_tokens_per_round"], 3.3)
            self.assertNotIn("acceptance", value)
            self.assertNotIn("acceptance_semantics", value)
        with self.assertRaisesRegex(ValueError, "conflicts with acceptance"):
            normalized({"acceptance": .6}, effective_tokens_per_round=5.2,
                       token_basis="Explicit Neff retains default rate consistency check")

    def test_explicit_lengths_without_rate_remain_usable_and_do_not_invent_acceptance(self):
        for values in ({"synthetic_acceptance_length": 5.2},
                       {"effective_tokens_per_round": 5.2, "token_basis": "Configured SAL including bonus"}):
            value = normalized(**values)
            self.assertEqual(value["effective_tokens_per_round"], 5.2)
            self.assertNotIn("acceptance", value)
            self.assertNotIn("acceptance_semantics", value)

    def test_historical_semantics_rejected_with_specific_preserving_sal_migration(self):
        for defaults, values in (({}, {"acceptance": .6}), ({"acceptance": .6}, {})):
            defaults = dict(defaults)
            if defaults:
                defaults["acceptance_semantics"] = "mean_accepted_draft_fraction"
            else:
                values = {**values, "acceptance_semantics": "mean_accepted_draft_fraction"}
            with self.subTest(defaults=defaults), self.assertRaisesRegex(
                    ValueError, r"acceptance=SAL/\(1\+S\).+S=7, SAL=5.2 requires acceptance=0.65"):
                normalized(defaults, **values)

    def test_unknown_or_incomplete_rate_semantics_are_not_silently_ignored(self):
        for semantics in ("request_success_fraction", "", False, 1):
            with self.subTest(semantics=semantics), self.assertRaisesRegex(ValueError, "acceptance_semantics"):
                normalized(acceptance=.65, acceptance_semantics=semantics, synthetic_acceptance_length=5.2)
        with self.assertRaisesRegex(ValueError, "requires acceptance"):
            normalized(acceptance_semantics=core.ACCEPTANCE_SEMANTICS, synthetic_acceptance_length=5.2)

    def test_rate_and_output_length_bounds_are_strict_without_clamping(self):
        for steps, rate, expected in ((0, 1., 1.), (2, 1 / 3, 1.), (3, .25, 1.),
                                      (7, .125, 1.), (7, 1., 8.)):
            with self.subTest(steps=steps, rate=rate):
                self.assertEqual(normalized(draft_tokens=steps, acceptance=rate)["effective_tokens_per_round"], expected)
        for steps, rate in ((0, .5), (2, math.nextafter(1 / 3, 0.)),
                            (7, 0), (7, .05), (7, .124999999)):
            with self.subTest(steps=steps, rate=rate), self.assertRaisesRegex(ValueError, r"must be >= 1"):
                normalized(draft_tokens=steps, acceptance=rate)
        for rate in (None, True, False, "0.65", -1, 1.00001, float("nan"), float("inf")):
            with self.subTest(rate=rate), self.assertRaises(ValueError):
                normalized(acceptance=rate)

    def test_rate_change_only_rescales_output_metrics_not_any_method_cycle_or_partition(self):
        methods_by_rate = {}
        for rate in (.6, .65):
            report, native = source_rank_report()
            case = report["cases"][0]
            config = dict(case["config"], name=case["name"], with_stack=False, acceptance=rate)
            config.pop("effective_tokens_per_round")
            case["config"] = core.normalize_case({}, config, Path.cwd())
            methods_by_rate[rate] = attach(report, native)
        for low, high in zip(methods_by_rate[.6], methods_by_rate[.65]):
            with self.subTest(method=low["id"]):
                self.assertEqual(low["status"], core.LAYER_MEAN_STATUS)
                for key in ("cycle_ms", "source_cycle_ms", "selected_rank", "class_means",
                            "role_details", "source_partition", "breakdown", "target_stage_breakdowns"):
                    self.assertEqual(low[key], high[key], key)
                self.assertAlmostEqual(low["tpot_ms"], low["cycle_ms"] / 4.8)
                self.assertAlmostEqual(high["tpot_ms"], high["cycle_ms"] / 5.2)
                for key in ("per_device_tokens_s", "global_tokens_s"):
                    self.assertAlmostEqual(high[key] / low[key], .65 / .6)

    def test_result_csv_labels_the_configured_fraction_and_explicit_semantics(self):
        report, native = source_rank_report()
        report["cases"][0]["config"].update(acceptance=.65,
            acceptance_semantics=core.ACCEPTANCE_SEMANTICS)
        attach(report, native)
        rows = core.result_rows(report)
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(row["配置接受率"] == .65 for row in rows))
        self.assertTrue(all(row["接受率口径"] == core.ACCEPTANCE_SEMANTICS for row in rows))
        self.assertTrue(all("配置平均草稿接受比例" not in row for row in rows))

    def test_offline_method3_preserves_old_report_neff_and_existing_method_results(self):
        report, native = source_rank_report()
        attach(report, native)
        case = report["cases"][0]
        case["config"].update(acceptance=.6, acceptance_semantics="mean_accepted_draft_fraction")
        case["projection"]["layer_mean_methods"] = case["projection"]["layer_mean_methods"][:2]
        stored = copy.deepcopy(core.report_json_payload(report))
        before = copy.deepcopy(stored)
        updated = core.add_method3_report(stored)
        self.assertEqual(stored, before)
        case = updated["cases"][0]
        self.assertEqual(case["config"]["effective_tokens_per_round"], 5.2)
        self.assertEqual(case["config"]["acceptance_semantics"], "mean_accepted_draft_fraction")
        self.assertEqual(case["projection"]["layer_mean_methods"][:2],
                         before["cases"][0]["projection"]["layer_mean_methods"])
        method = case["projection"]["layer_mean_methods"][-1]
        self.assertEqual(method["status"], core.LAYER_MEAN_STATUS, method["reasons"])
        self.assertAlmostEqual(method["tpot_ms"], method["cycle_ms"] / 5.2)


if __name__ == "__main__":
    unittest.main()
