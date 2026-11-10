"""A common measured source rank must anchor both target methods and evidence."""

import copy
import math
from pathlib import Path
import statistics
import unittest

import profile_ep128 as core
from tests.test_ep128_layer_mean import attach
from tests.test_ep128_target_stages import native_stage_report


REFERENCE = Path(__file__).resolve().parents[1] / "examples/communication-reference.synthetic.csv"
POLICY = "max_of_captured_rank_mean_source_cycle"


def source_rank_report(ranks=None, *, reference=True):
    # main0 affects the measured full round but is excluded from reference
    # layer means. Thus dp0 is the source slow rank while dp1 has the larger
    # target estimate, independently in both communication models.
    ranks = ranks or (("dp0", [1., 1.], 1000), ("dp1", [2., 2.], 10))
    pieces, native = [], {}
    for rank, factors, main0 in ranks:
        report, captures = native_stage_report({rank: factors}, layer0_wall=main0)
        pieces.append(report["cases"][0])
        native.update(captures)
    case = copy.deepcopy(pieces[0])
    case["captures"] = [item for piece in pieces for item in piece["captures"]]
    case["measurements"] = [item for piece in pieces for item in piece["measurements"]]
    source_rows = [{"rank": measured["rank"], "rounds": len(measured["rounds"]),
                    "cycle_ms_mean": statistics.fmean(row["duration_ms"] for row in measured["rounds"])}
                   for measured in case["measurements"]]
    chosen = max(source_rows, key=lambda item: item["cycle_ms_mean"])
    case["source"] = {"per_rank": source_rows, "aggregate": {
        "selected_rank": chosen["rank"], "cycle_ms": chosen["cycle_ms_mean"],
        "per_device_tokens_s": 20800 / chosen["cycle_ms_mean"]}}
    report = {"schema_version": 2, "cases": [case], "communication_reference": {
        "rows": core.read_reference(REFERENCE) if reference else []}}
    return report, native


class CommonSourceRankTests(unittest.TestCase):
    def assert_policy(self, method, rank, source_ms):
        selection = method["rank_selection"]
        self.assertEqual(selection["policy"], POLICY)
        self.assertEqual(selection["basis"], "retained_source_full_cycle_before_extrapolation")
        self.assertEqual(selection["selected_rank"], rank)
        self.assertEqual(selection["selected_source_cycle_ms"], source_ms)
        self.assertEqual(selection["tie_break"], "first_source_input_rank")
        self.assertEqual(method["source_rank"], rank)

    def test_source_slowest_is_fixed_even_when_both_target_maxima_are_another_rank(self):
        report, native = source_rank_report()
        # Two participating chips per physical card must still divide the
        # selected-rank rate by 64 cards rather than 128 logical devices.
        report["cases"][0]["config"].update(target_devices=64, target_chips_per_card=2)
        before_source = copy.deepcopy(report["cases"][0]["source"])
        before_measurements = copy.deepcopy(report["cases"][0]["measurements"])
        methods = attach(report, native)
        self.assertEqual([method["selected_rank"] for method in methods], ["dp0", "dp0"])
        self.assertEqual(report["cases"][0]["source"], before_source)
        self.assertEqual(report["cases"][0]["measurements"], before_measurements)
        for method in methods:
            self.assertEqual(method["status"], "LAYER_MEAN_ESTIMATE", method["reasons"])
            rows = {row["rank"]: row for row in method["per_rank"]}
            self.assertGreater(rows["dp0"]["source_cycle_ms"], rows["dp1"]["source_cycle_ms"])
            self.assertGreater(rows["dp1"]["cycle_ms"], rows["dp0"]["cycle_ms"])
            self.assert_policy(method, "dp0", 1245.)
            self.assertEqual([row["rank"] for row in method["per_rank"]], ["dp0", "dp1"])
            # Keep each rank's unmodified target formula, rather than forcing
            # another rank's result into dp0's card or discarding dp1.
            for field in ("cycle_ms", "source_cycle_ms", "tpot_ms", "per_device_tokens_s",
                          "global_tokens_s", "class_means", "role_details", "source_partition", "breakdown"):
                self.assertEqual(method[field], rows["dp0"][field], field)
            self.assertAlmostEqual(method["tpot_ms"], method["cycle_ms"] / 5.2)
            self.assertAlmostEqual(method["per_device_tokens_s"], 41600 / method["cycle_ms"])
            self.assertAlmostEqual(method["global_tokens_s"], 1000 * 4 * 128 * 5.2 / method["cycle_ms"])
            self.assertAlmostEqual(math.fsum(row["value_ms"] for row in method["breakdown"]), method["cycle_ms"])

    def test_source_selection_uses_retained_round_mean_not_largest_single_round(self):
        report, native = source_rank_report((("dp0", [1., 3.], 1000), ("dp1", [2.2, 2.2], 1000)))
        case = report["cases"][0]
        self.assertGreater(max(row["duration_ms"] for row in case["measurements"][0]["rounds"]),
                           max(row["duration_ms"] for row in case["measurements"][1]["rounds"]))
        for method in attach(report, native):
            self.assertEqual(method["selected_rank"], "dp1")
            self.assert_policy(method, "dp1", 1245 * 2.2)
            means = {row["rank"]: row for row in method["rank_selection"]["per_rank"]}
            self.assertEqual(means["dp0"]["rounds"], 2)
            self.assertEqual(means["dp0"]["source_cycle_ms"], 1245 * 2)

    def test_ties_follow_original_source_input_order_for_both_methods(self):
        for order in (("dp9", "dp1"), ("dp1", "dp9")):
            with self.subTest(order=order):
                report, native = source_rank_report(tuple((rank, [1., 2.], 1000) for rank in order))
                for method in attach(report, native):
                    self.assertEqual(method["selected_rank"], order[0])
                    self.assert_policy(method, order[0], 1245 * 1.5)
                    self.assertEqual([row["rank"] for row in method["rank_selection"]["per_rank"]], list(order))

    def test_unavailable_method_keeps_source_rank_without_claiming_a_target_selection(self):
        # Put the source slowest second, so source evidence cannot silently
        # fall back to the first captured rank when method2 has no reference.
        report, native = source_rank_report((("dp1", [2., 2.], 10), ("dp0", [1., 1.], 1000)), reference=False)
        first, second = attach(report, native)
        self.assertEqual(first["selected_rank"], "dp0")
        self.assertEqual(second["status"], "UNAVAILABLE")
        self.assertIsNone(second["selected_rank"])
        self.assert_policy(second, "dp0", 1245.)
        for field in ("cycle_ms", "tpot_ms", "per_device_tokens_s", "global_tokens_s"):
            self.assertIsNone(second[field], field)
        source = next(row for row in second["per_rank"] if row["rank"] == "dp0")
        self.assertEqual(second["source_partition"], source["source_partition"])
        self.assertEqual(second["class_means"], source["class_means"])
        self.assertEqual(second["role_details"], source["role_details"])

    def test_stage_parts_operator_examples_and_pmu_all_follow_common_source_rank(self):
        report, native = source_rank_report()
        for rank, capture in native.items():
            for task in capture["tasks"]:
                if "GroupedMatmul" in task.name:
                    task.task_type = "AI_CORE"
                    task.pmu = {"core_type": "AI_CORE", "invalid_fields": [], "metrics": {
                        "aic_mac_ratio": .96 if rank == "dp0" else .41,
                        "aic_mte2_ratio": .3 if rank == "dp0" else .95,
                        "aic_mte3_ratio": .1}}
        methods = attach(report, native)
        for method in methods:
            selected = next(row for row in method["per_rank"] if row["rank"] == "dp0")
            found_gmms = 0
            for actual, expected in zip(method["target_stage_breakdowns"], selected["target_stage_breakdowns"]):
                without_operators = copy.deepcopy(actual)
                for stage in without_operators["stage_rows"]:
                    stage.pop("operator_breakdown")
                self.assertEqual(without_operators, expected)
                self.assertEqual(actual["status"], "MATCHED", actual.get("reason"))
                self.assertAlmostEqual(math.fsum(stage["target_total_ms"] for stage in actual["stage_rows"]),
                                       actual["target_total_ms"])
                for stage in actual["stage_rows"]:
                    ledger = stage["operator_breakdown"]
                    self.assertEqual(ledger["status"], "MATCHED", ledger.get("reason"))
                    for field in ("source_ms", "delta_ms", "target_ms", "target_total_ms", "cycle_share"):
                        self.assertAlmostEqual(math.fsum(row[field] for row in ledger["rows"]), stage[field])
                    for row in ledger["rows"]:
                        if row.get("example"):
                            self.assertEqual(row["example"]["rank"], "dp0")
                    for kernel in ledger["kernels"]:
                        self.assertEqual(kernel["example"]["rank"], "dp0")
                        if "GroupedMatmul" in kernel["name"]:
                            found_gmms += 1
                            self.assertAlmostEqual(kernel["pmu"]["metrics"]["aic_mac_ratio"]["value"], .96)
                            self.assertEqual(kernel["pmu"]["coverage"]["duration_coverage"], 1.)
            self.assertGreater(found_gmms, 0)

    def test_common_source_selection_is_case_local(self):
        first, native_first = source_rank_report((("A_source", [1.], 1000), ("A_target", [2.], 10)))
        second, native_second = source_rank_report((("B_target", [2.], 10), ("B_source", [1.], 1000)))
        first["cases"].extend(second["cases"])
        attach(first, {**native_first, **native_second})
        self.assertEqual([[method["selected_rank"] for method in case["projection"]["layer_mean_methods"]]
                          for case in first["cases"]], [["A_source", "A_source"], ["B_source", "B_source"]])


if __name__ == "__main__":
    unittest.main()
