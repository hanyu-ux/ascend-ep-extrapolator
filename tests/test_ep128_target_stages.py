"""Target ten-stage explanations, using synthetic native anchor timelines."""
import copy
from pathlib import Path
import statistics
from types import SimpleNamespace
import unittest

import profile_ep128 as core
from tests.test_ep128_layer_mean import attach, synthetic_report


def native_stage_report(*args, **kwargs):
    """Add the missing Attention and second HC pair to the native mean fixture."""
    report, native = synthetic_report(*args, **kwargs)
    for measured in report["cases"][0]["measurements"]:
        tasks = native[measured["rank"]]["tasks"]
        for cycle in measured["rounds"]:
            for layer in cycle["layers"]:
                left = int(layer["start_ns"])
                first = next(task for task in tasks if task.start_ns == left and task.name == "HcPre")
                factor = (first.end_ns - first.start_ns) / 20000
                for name, offset, duration in (("SparseAttnSharedkv", .1, .1),
                                                ("HcPost", .3, .02), ("HcPre", .4, .02)):
                    start = left + round(offset * factor * 1e6)
                    tasks.append(SimpleNamespace(name=name, op_type=name, label=name + " " + name,
                        start_ns=start, end_ns=start + round(duration * factor * 1e6),
                        input_shapes="", row_index=len(tasks) + 1))
        tasks.sort(key=lambda task: task.start_ns)
    return report, native


def groups(method):
    return {(row["role"], row["attention"]): row for row in method["target_stage_breakdowns"]}


def stages(group):
    return {row["stage_key"]: row for row in group["stage_rows"]}


class TargetStageTests(unittest.TestCase):
    def assert_closes(self, method):
        classes = {(row["role"], row["attention"]): row for row in method["class_means"]}
        weighted = {"main": 0., "draft": 0.}
        for key, group in groups(method).items():
            parent = classes[key]
            self.assertEqual(group["status"], "MATCHED", group["reason"])
            self.assertEqual(group["reference_layers"], parent["reference_layers"])
            self.assertEqual(group["target_layers"], parent["target_layers"])
            self.assertEqual(len(group["stage_rows"]), 10)
            self.assertEqual([row["stage_order"] for row in group["stage_rows"]], list(range(1, 11)))
            self.assertAlmostEqual(sum(row["source_ms"] for row in group["stage_rows"]), parent["wall_ms"])
            self.assertAlmostEqual(sum(row["target_ms"] for row in group["stage_rows"]), parent["target_wall_ms"])
            total = parent["target_layers"] * parent["target_wall_ms"]
            self.assertAlmostEqual(group["target_total_ms"], total)
            self.assertAlmostEqual(sum(row["target_total_ms"] for row in group["stage_rows"]), total)
            for row in group["stage_rows"]:
                self.assertAlmostEqual(row["target_total_ms"], parent["target_layers"] * row["target_ms"])
                self.assertAlmostEqual(row["cycle_share"], row["target_total_ms"] / method["cycle_ms"])
                self.assertGreaterEqual(row["cycle_share"], 0)
                self.assertLessEqual(row["cycle_share"], 1)
            weighted[group["role"]] += total
        for role in method["role_details"]:
            self.assertAlmostEqual(weighted[role["role"]], role["target_layer_envelopes_ms"])

    def test_method1_native_ten_stages_only_apply_parent_component_deltas(self):
        report, native = native_stage_report()
        method = attach(report, native)[0]
        self.assert_closes(method)
        self.assertAlmostEqual(method["cycle_ms"], 1262.592)
        self.assertAlmostEqual(method["tpot_ms"], 1262.592 / 5.2)
        self.assertAlmostEqual(method["per_device_tokens_s"], 20800 / 1262.592)
        for group in groups(method).values():
            rows = stages(group)
            self.assertAlmostEqual(rows["dispatch"]["source_ms"], .5)
            self.assertAlmostEqual(rows["dispatch"]["delta_ms"], .5 * (2.8125 - 1) + .014)
            self.assertAlmostEqual(rows["dispatch"]["target_ms"], .5 * 2.8125 + .014)
            self.assertAlmostEqual(rows["combine"]["delta_ms"], .25 * (2.8125 - 1) + .014)
            self.assertAlmostEqual(rows["combine"]["target_ms"], .25 * 2.8125 + .014)
            # The phase contains activation and gaps; only the .4ms GMM union
            # changes by TopK, never the whole 2.5ms experts phase.
            self.assertAlmostEqual(rows["experts"]["source_ms"], 2.5)
            self.assertAlmostEqual(rows["experts"]["delta_ms"], .4 * .5)
            self.assertAlmostEqual(rows["experts"]["target_ms"], 2.7)
            for key, row in rows.items():
                if key not in ("dispatch", "combine", "experts"):
                    self.assertEqual(row["delta_ms"], 0)
                    self.assertEqual(row["target_ms"], row["source_ms"])

    def test_method2_dispatch_and_combine_have_independent_historical_factors(self):
        report, native = native_stage_report()
        reference = Path(__file__).resolve().parents[1] / "examples/communication-reference.synthetic.csv"
        report["communication_reference"] = {"rows": core.read_reference(reference)}
        first, second = attach(report, native)
        self.assertEqual(first["communication"]["extra_ms_per_call"], .014)
        self.assertEqual(second["communication"]["extra_ms_per_call"], 0)
        self.assert_closes(first)
        self.assert_closes(second)
        self.assertEqual(second["communication"]["multipliers"]["main"]["dispatch"], 6.)
        self.assertEqual(second["communication"]["multipliers"]["main"]["combine"], 3.75)
        for group in groups(second).values():
            rows = stages(group)
            self.assertAlmostEqual(rows["dispatch"]["target_ms"], .5 * 6.)
            self.assertAlmostEqual(rows["combine"]["target_ms"], .25 * 3.75)
            self.assertAlmostEqual(rows["experts"]["target_ms"], 2.7)

    def test_reference_layer_stage_means_use_every_eligible_layer_of_each_role_and_attention(self):
        report, native = native_stage_report()
        for cycle in report["cases"][0]["measurements"][0]["rounds"]:
            for layer in cycle["layers"]:
                attention = next(task for task in native["dp0"]["tasks"]
                    if int(layer["start_ns"]) <= task.start_ns < int(layer["end_ns"])
                    and task.name == "SparseAttnSharedkv")
                attention.end_ns = attention.start_ns + round((.02 + .005 * layer["layer"]) * 1e6)
        method = attach(report, native)[0]
        self.assert_closes(method)
        self.assertAlmostEqual(stages(groups(method)["main", "HCA"])["attention"]["source_ms"], .04)
        self.assertAlmostEqual(stages(groups(method)["main", "CSA"])["attention"]["source_ms"], .04)
        self.assertAlmostEqual(stages(groups(method)["draft", "SWA"])["attention"]["source_ms"], .025)

    def test_lower_topk_keeps_signed_deltas_and_positive_target_phases(self):
        report, native = native_stage_report(topk=8)
        method = attach(report, native)[0]
        self.assert_closes(method)
        for group in groups(method).values():
            rows = stages(group)
            self.assertAlmostEqual(rows["experts"]["delta_ms"], -.2)
            self.assertAlmostEqual(rows["experts"]["target_ms"], 2.3)
            self.assertAlmostEqual(rows["dispatch"]["delta_ms"], .5 * (.9375 - 1) + .014)
            self.assertGreater(rows["dispatch"]["target_ms"], 0)

    def test_main0_does_not_enter_reference_stages_even_if_its_attention_is_missing(self):
        report, native = native_stage_report()
        baseline = attach(copy.deepcopy(report), copy.deepcopy(native))[0]
        main0 = report["cases"][0]["measurements"][0]["rounds"][0]["layers"][0]
        native["dp0"]["tasks"] = [task for task in native["dp0"]["tasks"]
            if not (int(main0["start_ns"]) <= task.start_ns < int(main0["end_ns"])
                    and task.name == "SparseAttnSharedkv")]
        method = attach(report, native)[0]
        self.assert_closes(method)
        self.assertEqual(method["cycle_ms"], baseline["cycle_ms"])
        for (role, _), group in groups(method).items():
            if role == "main":
                self.assertNotIn(0, group["reference_layers"])
                self.assertLess(group["source_wall_ms"], 1000)

    def test_round_means_before_rank_selection_and_selected_rank_only(self):
        report, native = native_stage_report({"dp0": [1., 3.], "dp1": [2.2, 2.2]})
        method = attach(report, native)[0]
        ranks = {row["rank"]: row for row in method["per_rank"]}
        self.assertEqual(method["selected_rank"], "dp1")
        self.assertEqual(method["target_stage_breakdowns"], ranks["dp1"]["target_stage_breakdowns"])
        self.assertEqual(groups(method)["main", "HCA"]["rounds"], 2)
        for rank in ranks.values():
            self.assert_closes(rank)
            for cycle in rank["rounds"]:
                self.assert_closes(cycle)
                self.assertTrue(all(row["rounds"] == 1 for row in cycle["target_stage_breakdowns"]))
        dp0 = stages(groups(ranks["dp0"])["main", "HCA"])
        dp1 = stages(groups(ranks["dp1"])["main", "HCA"])
        self.assertAlmostEqual(dp0["dispatch"]["source_ms"], 1.)
        self.assertAlmostEqual(dp0["dispatch"]["target_ms"], 1. * 2.8125 + .014)
        self.assertAlmostEqual(dp1["dispatch"]["source_ms"], 1.1)
        self.assertAlmostEqual(dp1["dispatch"]["target_ms"], 1.1 * 2.8125 + .014)
        averaged = stages(groups(ranks["dp0"])["main", "HCA"])["attention"]
        samples = [stages(groups(cycle)["main", "HCA"])["attention"] for cycle in ranks["dp0"]["rounds"]]
        self.assertAlmostEqual(averaged["cycle_share"], statistics.fmean(row["target_total_ms"] for row in samples)
                               / ranks["dp0"]["cycle_ms"])
        self.assertNotAlmostEqual(averaged["cycle_share"], statistics.fmean(row["cycle_share"] for row in samples), places=10)

    def test_missing_one_round_anchor_only_disables_that_class_substages(self):
        report, native = native_stage_report({"dp0": [1., 3.]})
        baseline = attach(copy.deepcopy(report), copy.deepcopy(native))[0]
        layer = report["cases"][0]["measurements"][0]["rounds"][1]["layers"][1]
        native["dp0"]["tasks"] = [task for task in native["dp0"]["tasks"]
            if not (int(layer["start_ns"]) <= task.start_ns < int(layer["end_ns"])
                    and task.name == "SparseAttnSharedkv")]
        method = attach(report, native)[0]
        self.assertEqual(method["status"], "LAYER_MEAN_ESTIMATE")
        for key in ("cycle_ms", "tpot_ms", "per_device_tokens_s", "global_tokens_s", "selected_rank", "class_means", "role_details"):
            self.assertEqual(method[key], baseline[key])
        for key, group in groups(method).items():
            if key == ("main", "HCA"):
                self.assertEqual(group["status"], "UNAVAILABLE")
                self.assertEqual(group["stage_rows"], [])
                self.assertTrue(group["reason"])
                self.assertEqual(group["target_total_ms"], groups(baseline)[key]["target_total_ms"])
            else:
                self.assertEqual(group["status"], "MATCHED")
        source_rounds = method["per_rank"][0]["rounds"]
        self.assertEqual(groups(source_rounds[0])["main", "HCA"]["status"], "MATCHED")
        self.assertEqual(groups(source_rounds[1])["main", "HCA"]["status"], "UNAVAILABLE")

    def test_old_single_hc_fixtures_still_estimate_parent_without_fabricating_stages(self):
        report, native = synthetic_report()
        method = attach(report, native)[0]
        self.assertEqual(method["status"], "LAYER_MEAN_ESTIMATE")
        self.assertAlmostEqual(method["cycle_ms"], 1262.592)
        self.assertTrue(all(row["status"] == "UNAVAILABLE" and row["stage_rows"] == []
                            for row in method["target_stage_breakdowns"]))
        self.assertTrue(all(row["target_total_ms"] > 0 for row in method["target_stage_breakdowns"]))

    def test_missing_cached_stage_evidence_does_not_make_method_unavailable(self):
        report, native = native_stage_report()
        case = report["cases"][0]
        schedule = core.b_schedule(case["source_model"], case["config"], case["target_model"])[0]
        prepared = {}
        for measured in case["measurements"]:
            tasks = native[measured["rank"]]["tasks"]
            prepared[measured["rank"]] = [core._lm_source_round(cycle, tasks, [task.start_ns for task in tasks],
                case["source_model"], case["config"], schedule, {0}, {"main": True, "draft": True})
                for cycle in measured["rounds"]]
            for source in prepared[measured["rank"]]:
                source.pop("class_stage_values")
        case["_prepared_layer_means"] = prepared
        method = attach(report, native)[0]
        self.assertEqual(method["status"], "LAYER_MEAN_ESTIMATE")
        self.assertAlmostEqual(method["cycle_ms"], 1262.592)
        self.assertTrue(all(row["status"] == "UNAVAILABLE" for row in method["target_stage_breakdowns"]))

    def test_unknown_gmm_unchanged_topk_preserves_unknown_metric_and_zero_delta_policy(self):
        report, native = native_stage_report(topk=16)
        native["dp0"]["tasks"] = [task for task in native["dp0"]["tasks"] if not core._lm_gmm(task)]
        method = attach(report, native)[0]
        self.assert_closes(method)
        self.assertTrue(all(row["routed_compute_ms"] is None and row["routed_compute_delta_ms"] == 0
                            for row in method["class_means"]))
        for group in groups(method).values():
            row = stages(group)["experts"]
            self.assertEqual(row["delta_ms"], 0)
            self.assertEqual(row["source_ms"], 2.5)
            self.assertEqual(row["target_ms"], 2.5)
            self.assertIn("routed_compute_ms", group["note"])
            self.assertIn("未知", group["note"])

    def test_overlapping_gmm_uses_union_for_increment_not_sum_or_whole_experts_span(self):
        report, native = native_stage_report()
        for measured in report["cases"][0]["measurements"]:
            for cycle in measured["rounds"]:
                for layer in cycle["layers"]:
                    named = [task for task in native[measured["rank"]]["tasks"]
                             if int(layer["start_ns"]) <= task.start_ns < int(layer["end_ns"]) and core._lm_gmm(task)]
                    named[1].start_ns = named[0].start_ns + 100000
                    named[1].end_ns = named[0].end_ns
        native["dp0"]["tasks"].sort(key=lambda task: task.start_ns)
        method = attach(report, native)[0]
        self.assert_closes(method)
        for group in groups(method).values():
            self.assertAlmostEqual(stages(group)["experts"]["delta_ms"], .25 * .5)
            self.assertAlmostEqual(stages(group)["experts"]["target_ms"], 2.5 + .125)

    def test_optional_evidence_guard_rejects_gmm_larger_than_experts_without_changing_parent(self):
        report, native = native_stage_report()
        case = report["cases"][0]
        layer = case["measurements"][0]["rounds"][0]["layers"][1]
        tasks = native["dp0"]["tasks"]
        starts = [task.start_ns for task in tasks]
        values, _ = core._lm_native_layer(tasks, starts, layer, case["source_model"], case["config"]["base_ep"])
        bad = core._lm_source_layer_stages(layer, tasks, starts, {**values, "routed_compute_ms": 1000})
        self.assertEqual(bad["status"], "UNAVAILABLE")
        self.assertEqual(bad["stage_rows"], [])
        self.assertIn("GMM union", bad["reason"])
        method = attach(report, native)[0]
        self.assert_closes(method)

    def test_source_measured_diagnostics_are_not_replaced_by_target_estimates(self):
        report, native = native_stage_report()
        measured = report["cases"][0]["measurements"][0]
        measured["layer_diagnostics"] = {"status": "MATCHED", "stage_summary": [{"duration_ms_mean": 123456}]}
        measured["swa_diagnostics"] = {"status": "PARTIAL", "stage_summary": [{"duration_ms_mean": None}]}
        before = copy.deepcopy(measured)
        method = attach(report, native)[0]
        self.assert_closes(method)
        self.assertEqual(measured, before)


if __name__ == "__main__":
    unittest.main()
