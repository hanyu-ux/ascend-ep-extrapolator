"""Method3 uses source communication identities; synthetic arithmetic only."""
import copy
import json
import math
import unittest
from unittest import mock

import profile_ep128 as core
from tests.test_ep128_layer_mean import attach, synthetic_report
from tests.test_ep128_source_rank import source_rank_report
from tests.test_ep128_target_stages import native_stage_report


def by_id(methods):
    return {method["id"]: method for method in methods}


def legacy_payload(report):
    payload = json.loads(json.dumps(core.report_json_payload(report)))
    for case in payload["cases"]:
        case["projection"]["layer_mean_methods"] = [method for method in
            case["projection"]["layer_mean_methods"] if method["id"] != "method3"]
    return payload


class Method3Tests(unittest.TestCase):
    def test_raw_default_three_methods_and_independent_hand_arithmetic(self):
        report, native = synthetic_report()
        methods = attach(report, native)
        self.assertEqual([row["id"] for row in methods], ["method1", "method2", "method3"])
        first, second, method = methods
        self.assertEqual(method["status"], core.LAYER_MEAN_STATUS, method["reasons"])
        # Existing target depth, measured gaps and residual stay intact.
        # The only per-layer addition is GMM .4 * (24/16 - 1) = .2 ms.
        expected = 31 * 10 + 30 * 20 + 3 * 5 + 217 + 1 + 18 + 64 * .2
        self.assertAlmostEqual(expected, 1173.8)
        self.assertAlmostEqual(method["cycle_ms"], expected)
        self.assertAlmostEqual(method["per_device_tokens_s"], 20800 / expected)
        self.assertAlmostEqual(method["tpot_ms"], expected / 5.2)
        self.assertAlmostEqual(first["cycle_ms"], 1262.592)
        self.assertEqual(second["status"], "UNAVAILABLE")  # no historical reference
        self.assertEqual(method["communication"]["extra_ms_per_call"], 0)
        self.assertIn("per target layer", method["formula"])
        self.assertTrue(any("NOT_CONSTANT_WHOLE_ROUND" in row for row in method["warnings"]))

    def test_every_round_rank_class_and_component_closes_without_communication_delta(self):
        report, native = native_stage_report({"dp0": [1., 3.], "dp1": [2.2, 2.2]})
        method = by_id(attach(report, native))["method3"]
        self.assertEqual(method["selected_rank"], "dp1")
        for rank in method["per_rank"]:
            for record in [rank, *rank["rounds"]]:
                self.assertAlmostEqual(math.fsum(row["value_ms"] for row in record["breakdown"]), record["cycle_ms"])
                self.assertAlmostEqual(math.fsum(row["target_chain_ms"] for row in record["role_details"])
                    + record["source_partition"]["residual_ms"], record["cycle_ms"])
                for parent in record["class_means"]:
                    for component in ("dispatch", "combine"):
                        self.assertEqual(parent[component + "_delta_ms"], 0)
                        self.assertEqual(parent[component + "_target_ms"], parent[component + "_ms"])
                    self.assertAlmostEqual(parent["target_wall_ms"] - parent["wall_ms"],
                                           parent["routed_compute_delta_ms"])
                self.assertNotIn("operator_hotspots", record)
                for group in record["target_stage_breakdowns"]:
                    self.assertEqual(group["status"], "MATCHED", group["reason"])
                    self.assertEqual(len(group["stage_rows"]), 10)
                    self.assertAlmostEqual(math.fsum(row["target_total_ms"] for row in group["stage_rows"]),
                                           group["target_total_ms"])
                    for stage in group["stage_rows"]:
                        self.assertNotIn("operator_breakdown", stage)
                        if stage["stage_key"] != "experts":
                            self.assertEqual(stage["delta_ms"], 0)
        self.assertTrue(all(group["status"] == "MATCHED" for group in method["operator_hotspots"]))
        for row in method["detail_rows"]:
            if row["component"] in ("dispatch", "combine"):
                self.assertEqual(row["multiplier"], 1)
                self.assertEqual(row["extra_ms_per_call"], 0)
                self.assertEqual(row["weighted_delta_ms"], 0)

    def test_unchanged_topk_has_no_expert_delta_and_other_methods_unchanged(self):
        report, native = synthetic_report(topk=16)
        method = by_id(attach(report, native))["method3"]
        self.assertAlmostEqual(method["cycle_ms"], 1161.)
        for row in method["class_means"]:
            self.assertEqual(row["target_wall_ms"], row["wall_ms"])
        with mock.patch.object(core, "LAYER_MEAN_METHOD_IDS", ("method1", "method2")):
            old, old_native = synthetic_report(topk=16)
            previous = copy.deepcopy(attach(old, old_native))
        self.assertEqual(report["cases"][0]["projection"]["layer_mean_methods"][:2], previous)

    def test_all_available_methods_use_source_slowest_before_target_scaling(self):
        report, native = source_rank_report()
        methods = attach(report, native)
        self.assertEqual([row["selected_rank"] for row in methods], ["dp0"] * 3)
        for method in methods:
            self.assertEqual(method["source_rank"], "dp0")
            self.assertGreater(method["per_rank"][1]["cycle_ms"], method["per_rank"][0]["cycle_ms"])

    def test_exported_csv_rows_include_method3_with_zero_communication_delta(self):
        report, native = synthetic_report()
        case = report["cases"][0]
        case["source"] = {"aggregate": {}, "status": "SYNTHETIC", "per_rank": []}
        attach(report, native)
        rows = list(core.iter_report_rows(report, "projection_details"))
        third = [row for row in rows if row["method"] == "method3"]
        self.assertEqual(len(third), 9)
        self.assertTrue(all(row["delta_ms"] == 0 for row in third if row["component"] != "routed_compute"))
        summary = list(core.iter_report_rows(report, "summary"))
        self.assertEqual(summary[-1]["method"], "method3")

    def test_offline_matches_fresh_from_compact_source_means_without_csv_reads(self):
        report, native = source_rank_report()
        fresh = by_id(attach(report, native))["method3"]
        legacy = legacy_payload(report)
        before = copy.deepcopy(legacy)
        with mock.patch.object(core, "read_capture", side_effect=AssertionError("must not read CSV")):
            changed = core.add_method3_report(legacy)
        self.assertEqual(legacy, before)
        case = changed["cases"][0]
        methods = case["projection"]["layer_mean_methods"]
        self.assertEqual(methods[:2], before["cases"][0]["projection"]["layer_mean_methods"])
        method = methods[-1]
        self.assertEqual(method["status"], core.LAYER_MEAN_STATUS, method["reasons"])
        for field in ("cycle_ms", "per_device_tokens_s", "global_tokens_s", "tpot_ms", "source_cycle_ms"):
            self.assertAlmostEqual(method[field], fresh[field], msg=field)
        self.assertEqual(method["source_partition"], fresh["source_partition"])
        self.assertEqual(method["selected_rank"], "dp0")
        self.assertFalse(method["offline_derivation"]["source_csv_read"])
        self.assertFalse(method["offline_derivation"]["per_round_results_reconstructed"])
        for rank in method["per_rank"]:
            self.assertNotIn("rounds", rank)
        for parent in method["class_means"]:
            matching = next(row for row in fresh["class_means"] if row["role"] == parent["role"]
                            and row["attention"] == parent["attention"])
            self.assertEqual(parent["samples"], matching["samples"])
        for group in method["target_stage_breakdowns"]:
            self.assertEqual(group["rounds"], 2)
            self.assertEqual(group["status"], "MATCHED", group.get("reason"))
            for stage in group["stage_rows"]:
                self.assertNotIn("operator_breakdown", stage)
            self.assertAlmostEqual(math.fsum(stage["target_ms"] for stage in group["stage_rows"]),
                                   group["target_wall_ms"])
        for group in method["operator_hotspots"]:
            self.assertEqual(group["rounds"], 2)
            self.assertEqual(group["status"], "MATCHED", group.get("reason"))
            self.assertTrue(group["kernels"])

    def test_offline_does_not_need_method2_reference_or_hotspot_evidence(self):
        report, native = source_rank_report(reference=False)
        attach(report, native)
        legacy = legacy_payload(report)
        for method in legacy["cases"][0]["projection"]["layer_mean_methods"]:
            method.pop("operator_hotspots", None)
        changed = core.add_method3_report(legacy)
        method = changed["cases"][0]["projection"]["layer_mean_methods"][-1]
        self.assertEqual(method["status"], core.LAYER_MEAN_STATUS, method["reasons"])
        self.assertEqual(method["operator_hotspots"], [])

    def test_offline_attention_preparation_reuses_only_matching_source_evidence(self):
        report, native = source_rank_report()
        fresh = by_id(attach(report, native))["method3"]
        legacy = legacy_payload(report)
        before = copy.deepcopy(legacy)
        with mock.patch.object(core, "read_capture", side_effect=AssertionError("no source CSV reread")):
            method = core.add_method3_report(legacy)["cases"][0]["projection"]["layer_mean_methods"][-1]
        self.assertEqual(method["attention_prepare_breakdowns"], fresh["attention_prepare_breakdowns"])
        self.assertEqual(legacy, before)
        self.assertTrue(method["attention_prepare_breakdowns"])
        for mode in ("missing", "wrong_rank", "wrong_reference_layers", "wrong_rounds"):
            with self.subTest(mode=mode):
                corrupted = legacy_payload(report)
                for donor in corrupted["cases"][0]["projection"]["layer_mean_methods"]:
                    if mode == "missing":
                        donor.pop("attention_prepare_breakdowns", None)
                    else:
                        for group in donor["attention_prepare_breakdowns"]:
                            if mode == "wrong_rank":
                                group["rank"] = "dp1"
                            elif mode == "wrong_reference_layers":
                                group["reference_layers"] = [999]
                            else:
                                group["rounds"] += 1
                result = core.add_method3_report(corrupted)["cases"][0]["projection"]["layer_mean_methods"][-1]
                self.assertEqual(result["status"], core.LAYER_MEAN_STATUS)
                self.assertEqual(result["cycle_ms"], fresh["cycle_ms"])
                self.assertEqual(result["attention_prepare_breakdowns"], [])

    def test_offline_missing_or_contradictory_source_closure_is_unavailable(self):
        report, native = source_rank_report()
        attach(report, native)
        for mode in ("partition", "source_means", "source_cycle", "gap"):
            legacy = legacy_payload(report)
            first = legacy["cases"][0]["projection"]["layer_mean_methods"][0]
            if mode == "partition":
                first["per_rank"][0]["source_partition"]["residual_ms"] += 1
            elif mode == "source_means":
                first["per_rank"][0]["class_means"] = []
            elif mode == "source_cycle":
                first["per_rank"][0]["source_cycle_ms"] += 1
            else:
                del first["per_rank"][0]["role_details"][0]["reference_gap_mean_ms"]
            method = core.add_method3_report(legacy)["cases"][0]["projection"]["layer_mean_methods"][-1]
            self.assertEqual(method["status"], "UNAVAILABLE", mode)
            self.assertIsNone(method["cycle_ms"])
            self.assertTrue(method["reasons"])

    def test_offline_rejects_legacy_different_rank_until_explicit_alignment(self):
        report, native = source_rank_report()
        attach(report, native)
        legacy = legacy_payload(report)
        legacy["cases"][0]["projection"]["layer_mean_methods"][1]["selected_rank"] = "dp1"
        with self.assertRaisesRegex(ValueError, "--align-source-rank"):
            core.add_method3_report(legacy)
        aligned = core.align_report_source_rank(legacy)
        method = core.add_method3_report(aligned)["cases"][0]["projection"]["layer_mean_methods"][-1]
        self.assertEqual(method["status"], core.LAYER_MEAN_STATUS, method["reasons"])
        self.assertEqual(method["selected_rank"], "dp0")

    def test_offline_idempotent_and_duplicate_method_ids_rejected(self):
        report, native = source_rank_report()
        attach(report, native)
        payload = json.loads(json.dumps(core.report_json_payload(report)))
        again = core.add_method3_report(payload)
        self.assertEqual(again["cases"], payload["cases"])
        duplicate = copy.deepcopy(payload)
        methods = duplicate["cases"][0]["projection"]["layer_mean_methods"]
        methods.append(copy.deepcopy(methods[-1]))
        with self.assertRaisesRegex(ValueError, "duplicate method3"):
            core.add_method3_report(duplicate)

    def test_offline_unequal_round_counts_reproduce_mean_not_a_synthetic_round(self):
        report, native = native_stage_report({"dp0": [.5, 2., 4.], "dp1": [1., 2.]}, topk=8)
        fresh = by_id(attach(report, native))["method3"]
        legacy = legacy_payload(report)
        method = core.add_method3_report(legacy)["cases"][0]["projection"]["layer_mean_methods"][-1]
        self.assertEqual(method["status"], core.LAYER_MEAN_STATUS, method["reasons"])
        for actual, expected in zip(method["per_rank"], fresh["per_rank"]):
            self.assertEqual(actual["rank"], expected["rank"])
            self.assertNotIn("rounds", actual)
            self.assertAlmostEqual(actual["cycle_ms"], expected["cycle_ms"])
            for parent, expected_parent in zip(actual["class_means"], expected["class_means"]):
                self.assertEqual(parent.keys(), expected_parent.keys())
                for field, value in parent.items():
                    if type(value) in (int, float):
                        self.assertAlmostEqual(value, expected_parent[field], msg=field)
                    else:
                        self.assertEqual(value, expected_parent[field], field)
            for row in actual["class_means"]:
                self.assertLess(row["routed_compute_delta_ms"], 0)
        self.assertEqual(method["selected_rank"], "dp0")

    def test_offline_never_copies_another_methods_target_deltas_or_metrics(self):
        report, native = source_rank_report()
        fresh = by_id(attach(report, native))["method3"]
        legacy = legacy_payload(report)
        for donor in legacy["cases"][0]["projection"]["layer_mean_methods"]:
            donor.update(cycle_ms=1234567, tpot_ms=99999, per_device_tokens_s=777)
            for rank in donor["per_rank"]:
                rank["cycle_ms"] = 1234567
                for row in rank["class_means"]:
                    row.update(target_wall_ms=123456, dispatch_delta_ms=9000, combine_delta_ms=8000,
                               routed_compute_delta_ms=7000)
                for row in rank["role_details"]:
                    row.update(target_chain_ms=345678, target_gap_ms=45678, dispatch_delta_ms=9999)
        original = copy.deepcopy(legacy)
        method = core.add_method3_report(legacy)["cases"][0]["projection"]["layer_mean_methods"][-1]
        self.assertEqual(method["status"], core.LAYER_MEAN_STATUS, method["reasons"])
        self.assertAlmostEqual(method["cycle_ms"], fresh["cycle_ms"])
        self.assertEqual(method["class_means"], fresh["class_means"])
        self.assertEqual(legacy, original)

    def test_offline_source_pmu_is_unchanged_and_remains_on_same_rank(self):
        report, native = source_rank_report()
        for rank, capture in native.items():
            for task in capture["tasks"]:
                if "GroupedMatmul" in task.name:
                    task.task_type = "AI_CORE"
                    task.pmu = {"core_type": "AI_CORE", "invalid_fields": [], "metrics": {
                        "aic_mac_ratio": .96 if rank == "dp0" else .13, "aic_mte2_ratio": .3}}
        attach(report, native)
        legacy = legacy_payload(report)
        original = copy.deepcopy(legacy)
        changed = core.add_method3_report(legacy)
        self.assertEqual(legacy, original)
        self.assertEqual(changed["cases"][0]["projection"]["layer_mean_methods"][:2],
                         original["cases"][0]["projection"]["layer_mean_methods"])
        third = changed["cases"][0]["projection"]["layer_mean_methods"][-1]
        count = 0
        for group in third["operator_hotspots"]:
            for kernel in group["kernels"]:
                self.assertEqual(kernel["example"]["rank"], "dp0")
                if "GroupedMatmul" in kernel["name"]:
                    count += 1
                    self.assertAlmostEqual(kernel["pmu"]["metrics"]["aic_mac_ratio"]["value"], .96)
        self.assertGreater(count, 0)

    def test_wrong_rank_operator_evidence_is_unavailable_without_losing_parent(self):
        report, native = source_rank_report()
        attach(report, native)
        legacy = legacy_payload(report)
        donor = legacy["cases"][0]["projection"]["layer_mean_methods"][0]
        donor["operator_hotspots"][0]["kernels"][0]["example"]["rank"] = "dp1"
        original = copy.deepcopy(legacy)
        method = core.add_method3_report(legacy)["cases"][0]["projection"]["layer_mean_methods"][-1]
        self.assertEqual(method["status"], core.LAYER_MEAN_STATUS, method["reasons"])
        self.assertEqual(method["operator_hotspots"], [])
        self.assertEqual(legacy, original)

    def test_conflicting_existing_method3_is_recomputed_not_silently_retained(self):
        report, native = source_rank_report()
        attach(report, native)
        payload = json.loads(json.dumps(core.report_json_payload(report)))
        for case in payload["cases"]:
            case["target_model"]["main"]["num_experts_per_tok"] = 8
            case["target_model"]["draft"]["num_experts_per_tok"] = 8
        original = copy.deepcopy(payload)
        changed = core.add_method3_report(payload)
        third = changed["cases"][0]["projection"]["layer_mean_methods"][-1]
        self.assertEqual(third["status"], core.LAYER_MEAN_STATUS, third["reasons"])
        self.assertTrue(all(row["routed_compute_delta_ms"] < 0 for row in third["class_means"]))
        self.assertNotEqual(third["cycle_ms"], original["cases"][0]["projection"]["layer_mean_methods"][-1]["cycle_ms"])
        self.assertTrue(any("RECOMPUTED" in warning for warning in third["warnings"]))
        self.assertEqual(payload, original)
        self.assertEqual(changed["cases"][0]["projection"]["layer_mean_methods"][:2],
                         original["cases"][0]["projection"]["layer_mean_methods"][:2])
        # Even an apparently valid target total cannot excuse a stale comm model.
        payload = json.loads(json.dumps(core.report_json_payload(report)))
        payload["cases"][0]["projection"]["layer_mean_methods"][-1]["comm_model"]["extra_ms_per_call"] = .014
        corrected = core.add_method3_report(payload)["cases"][0]["projection"]["layer_mean_methods"][-1]
        self.assertEqual(corrected["comm_model"]["extra_ms_per_call"], 0)
        self.assertTrue(any("RECOMPUTED" in warning for warning in corrected["warnings"]))
        payload = json.loads(json.dumps(core.report_json_payload(report)))
        existing = payload["cases"][0]["projection"]["layer_mean_methods"][-1]
        existing["class_means"][0]["dispatch_delta_ms"] = 12
        existing["target_stage_breakdowns"][0]["stage_rows"][0]["target_ms"] = 999
        corrected = core.add_method3_report(payload)["cases"][0]["projection"]["layer_mean_methods"][-1]
        self.assertEqual(corrected["class_means"][0]["dispatch_delta_ms"], 0)
        self.assertNotEqual(corrected["target_stage_breakdowns"][0]["stage_rows"][0]["target_ms"], 999)

    def test_contradictory_source_classes_counts_and_boundary_provenance_are_unavailable(self):
        report, native = source_rank_report()
        attach(report, native)
        for mode in ("wall", "samples", "gap_count", "gap_mean", "excluded_wall", "hc_wall", "routed_overlap",
                     "class_shape", "role_shape", "partition_shape"):
            with self.subTest(mode=mode):
                legacy = legacy_payload(report)
                rank = legacy["cases"][0]["projection"]["layer_mean_methods"][0]["per_rank"][0]
                if mode == "wall":
                    rank["class_means"][0]["wall_ms"] += 1
                elif mode == "samples":
                    rank["class_means"][0]["samples"] += 1
                elif mode == "gap_count":
                    rank["role_details"][0]["reference_gap_count"] = 1
                elif mode == "gap_mean":
                    rank["role_details"][0]["reference_gap_mean_ms"] += 1
                elif mode == "excluded_wall":
                    rank["source_partition"]["excluded_reference_layer_wall_ms"] -= 1
                elif mode == "hc_wall":
                    legacy["cases"][0]["measurements"][0]["rounds"][0]["layers"][1]["wall_ms"] += 1
                elif mode == "routed_overlap":
                    rank["class_means"][0]["routed_compute_ms"] = rank["class_means"][0]["wall_ms"]
                elif mode == "class_shape":
                    rank["class_means"] = ["invalid"]
                elif mode == "role_shape":
                    rank["role_details"] = ["invalid"]
                else:
                    rank["source_partition"] = "invalid"
                original = copy.deepcopy(legacy)
                method = core.add_method3_report(legacy)["cases"][0]["projection"]["layer_mean_methods"][-1]
                self.assertEqual(method["status"], "UNAVAILABLE", method["reasons"])
                self.assertIsNone(method["cycle_ms"])
                self.assertEqual(legacy, original)

    def test_complete_aggregate_can_be_reused_when_historical_json_omits_hc_layer_boundaries(self):
        report, native = source_rank_report()
        fresh = by_id(attach(report, native))["method3"]
        legacy = legacy_payload(report)
        for case in legacy["cases"]:
            for measured in case["measurements"]:
                for cycle in measured["rounds"]:
                    cycle.pop("layers")
        method = core.add_method3_report(legacy)["cases"][0]["projection"]["layer_mean_methods"][-1]
        self.assertEqual(method["status"], core.LAYER_MEAN_STATUS, method["reasons"])
        self.assertAlmostEqual(method["cycle_ms"], fresh["cycle_ms"])

    def test_overlapping_gmm_calls_use_union_not_sum_and_target_depth_retains_each_layer_comm(self):
        report, native = native_stage_report(target_depth=61)
        for measured in report["cases"][0]["measurements"]:
            for cycle in measured["rounds"]:
                for layer in cycle["layers"]:
                    left = int(layer["start_ns"])
                    calls = [task for task in native[measured["rank"]]["tasks"]
                        if left <= task.start_ns < int(layer["end_ns"]) and "GroupedMatmul" in task.name]
                    calls[1].start_ns = calls[0].start_ns + 50000
                    calls[1].end_ns = calls[1].start_ns + 150000
        for capture in native.values():
            capture["tasks"].sort(key=lambda task: task.start_ns)
        fresh = by_id(attach(report, native))["method3"]
        self.assertEqual(fresh["status"], core.LAYER_MEAN_STATUS, fresh["reasons"])
        for parent in fresh["class_means"]:
            self.assertAlmostEqual(parent["routed_compute_ms"], .25)
            self.assertAlmostEqual(parent["routed_compute_delta_ms"], .125)
            self.assertEqual(parent["dispatch_target_ms"], .5)
            self.assertEqual(parent["combine_target_ms"], .25)
        # Identity applies to all 64 target layers; round communication is
        # deliberately not clamped to the 11-layer source sum.
        self.assertAlmostEqual(sum(row["target_layers"] * (row["dispatch_target_ms"] + row["combine_target_ms"])
                                   for row in fresh["class_means"]), 64 * .75)
        legacy = legacy_payload(report)
        method = core.add_method3_report(legacy)["cases"][0]["projection"]["layer_mean_methods"][-1]
        self.assertEqual(method["status"], core.LAYER_MEAN_STATUS, method["reasons"])
        self.assertAlmostEqual(method["cycle_ms"], fresh["cycle_ms"])
        hotspot = method["operator_hotspots"][0]
        self.assertAlmostEqual(sum(row["kernel_sum_ms"] for row in hotspot["kernels"]
                                   if "GroupedMatmul" in row["name"]), .4)


if __name__ == "__main__":
    unittest.main()
