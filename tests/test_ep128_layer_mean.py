"""Independent hand arithmetic for old layer-type means; synthetic data only."""
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import profile_ep128 as core


def synthetic_report(round_factors=None, *, target_depth=61, topk=24, packed=False, layer0_wall=1000):
    factors = round_factors or {"dp0": [1.]}
    ratios = [128, 128, 4, 128, 4, 128, 4, 128]
    model = dict(num_hidden_layers=8, num_hash_layers=3, hidden_size=1024,
                 moe_intermediate_size=512, n_routed_experts=224, num_experts_per_tok=16,
                 compress_ratios=ratios + [0, 0, 0])
    target_ratios = ratios + [4 if index % 2 == 0 else 128 for index in range(8, target_depth)]
    target = {"main": dict(model, num_hidden_layers=target_depth, compress_ratios=target_ratios,
                           n_routed_experts=1792, num_experts_per_tok=topk),
              "draft": dict(model, num_hidden_layers=3, num_hash_layers=0, compress_ratios=[0, 0, 0],
                            n_routed_experts=1792, num_experts_per_tok=topk)}
    config = dict(bs=4, draft_tokens=7, draft_layers=3, draft_passes=1, base_ep=16, target_ep=128,
                  source_dp=16, source_devices=16, source_chips_per_card=1, expected_ranks=16,
                  target_dp=128, target_devices=128, target_chips_per_card=1, ranks_per_node=8,
                  traffic_scope="interhost", effective_tokens_per_round=5.2, extra_us=14,
                  data_fraction=1, _rank_coverage_complete=len(factors) == 16)
    case = dict(name="SYNTHETIC_LAYER_MEANS", config=config, source_model=model, target_model=target,
                captures=[], measurements=[], projection={"methods": [{"id": "method1", "status": "PENDING",
                  "reasons": ["missing ownership, basis and component calibration"]}]})
    native = {}
    for rank, rank_factors in factors.items():
        tasks, rounds = [], []
        cursor = 0
        for round_number, factor in enumerate(rank_factors):
            start = cursor
            layers = []
            for index, (role, number, attention, router) in enumerate(core.a_schedule(model, config)):
                if index == 8:
                    cursor += int(7 * factor * 1e6)
                wall = (layer0_wall if index == 0 else 10 if attention == "HCA" else 20 if attention == "CSA" else 5) * factor
                end = cursor + round(wall * 1e6)
                layer = dict(role=role, layer=number, attention=attention, router=router,
                             start_ns=str(cursor), end_ns=str(end), wall_ms=wall,
                             clean=False, scope="HC_marker_interval", reasons=["no complete ownership"])
                layers.append(layer)

                def add(name, offset, duration, shapes=""):
                    left = cursor + round(offset * factor * 1e6)
                    tasks.append(SimpleNamespace(name=name, op_type=name, label=name + " " + name,
                        start_ns=left, end_ns=left + round(duration * factor * 1e6),
                        input_shapes=shapes, row_index=len(tasks) + 1))

                add("HcPre", 0, .02)
                add("MoeDistributeDispatchV3", 1, .5, "4100;32,1024;32,16;;32;;;")
                add("GroupedMatmulV4Mxfp" if packed else "GroupedMatmul", 2, .25,
                    "packed_fp4_scale_storage" if packed else "32,1024;14,1024,1024")
                add("SwiGlu", 2.3, .5, "32,1024")
                add("Mxfp4GgmmWeightNz" if packed else "GroupedMatmul", 3, .15,
                    "opaque_NZ_storage" if packed else "32,512;14,512,1024")
                add("MoeDistributeCombineV3", 4, .25)
                add("HcPost", wall / factor - .02, .02)
                cursor = end
                if role == "main" and number < 7:
                    cursor += round((99 if number == 0 else 2) * factor * 1e6)
                elif role == "draft" and number < 2:
                    cursor += round(.5 * factor * 1e6)
            cursor += round(11 * factor * 1e6)
            rounds.append(dict(round_id="hc-" + str(round_number), start_ns=str(start), end_ns=str(cursor),
                               duration_ms=(cursor - start) / 1e6, layers=layers,
                               projection_ready=False, projection_reasons=["no FULL/ownership attestation"]))
        tasks.sort(key=lambda task: task.start_ns)
        path = rank + ".csv"
        digest = "synthetic-native-hash-" + rank
        case["captures"].append(dict(rank=rank, path=path, sha256=digest))
        case["measurements"].append(dict(rank=rank, rounds=rounds, ownership_basis=None))
        native[rank] = dict(rank=rank, tasks=tasks, sha256=digest)
    return {"schema_version": 2, "cases": [case]}, native


def attach(report, native):
    def read_capture(path, rank):
        return native[rank]
    core._attach_layer_mean_estimates(report, read_capture, core.b_schedule, core.b_shapes, core.b_comm)
    return report["cases"][0]["projection"]["layer_mean_methods"]


class LayerMeanTests(unittest.TestCase):
    def test_main0_anomaly_excluded_from_means_not_hidden_in_residual_and_31_30_counts(self):
        report, native = synthetic_report()
        strict_before = copy.deepcopy(report["cases"][0]["projection"]["methods"])
        method = attach(report, native)[0]
        self.assertEqual(method["status"], "LAYER_MEAN_ESTIMATE", method["reasons"])
        classes = {(row["role"], row["attention"]): row for row in method["class_means"]}
        self.assertEqual(classes[("main", "HCA")]["reference_layers"], [1, 3, 5, 7])
        self.assertEqual(classes[("main", "CSA")]["reference_layers"], [2, 4, 6])
        self.assertEqual(classes[("main", "HCA")]["wall_ms"], 10)
        self.assertEqual(classes[("main", "CSA")]["wall_ms"], 20)
        self.assertEqual(classes[("main", "HCA")]["target_layers"], 31)
        self.assertEqual(classes[("main", "CSA")]["target_layers"], 30)
        self.assertEqual(classes[("draft", "SWA")]["target_layers"], 3)
        partition = method["source_partition"]
        self.assertEqual(partition["source_main_chain_ms"], 1211)
        self.assertEqual(partition["source_draft_chain_ms"], 16)
        self.assertEqual(partition["source_cycle_ms"], 1245)
        self.assertEqual(partition["excluded_reference_layer_wall_ms"], 1000)
        self.assertEqual(partition["residual_ms"], 18)
        self.assertEqual(partition["partition_total_ms"], partition["source_cycle_ms"])
        self.assertEqual(report["cases"][0]["projection"]["methods"], strict_before)
        smaller, smaller_native = synthetic_report(layer0_wall=10)
        smaller_method = attach(smaller, smaller_native)[0]
        self.assertEqual(smaller_method["cycle_ms"], method["cycle_ms"])
        self.assertEqual(smaller_method["source_partition"]["residual_ms"], 18)

    def test_original_gap_sum_plus_53_reference_gaps_and_cross_tail_once_close_target(self):
        report, native = synthetic_report()
        method = attach(report, native)[0]
        roles = {row["role"]: row for row in method["role_details"]}
        self.assertEqual(roles["main"]["source_gap_ms"], 111)
        self.assertEqual(roles["main"]["source_gap_count"], 7)
        self.assertEqual(roles["main"]["reference_gap_count"], 6)
        self.assertEqual(roles["main"]["reference_gap_mean_ms"], 2)
        self.assertEqual(roles["main"]["added_gap_ms"], 53 * 2)
        self.assertEqual(roles["main"]["target_gap_ms"], 217)
        self.assertEqual(roles["draft"]["source_gap_ms"], 1)
        self.assertEqual(roles["draft"]["added_gap_ms"], 0)
        self.assertEqual(roles["draft"]["target_gap_ms"], 1)
        # 31HCA*10+30CSA*20+3SWA*5 + all target internal gaps + U18,
        # plus64*(D.5+C.25)*(2.8125-1)+128*.014+64*GMM.4*(1.5-1).
        expected = (31 * 10 + 30 * 20 + 3 * 5 + 217 + 1 + 18
                    + 64 * .75 * 1.8125 + 128 * .014 + 64 * .4 * .5)
        self.assertAlmostEqual(method["cycle_ms"], expected)
        self.assertAlmostEqual(method["cycle_ms"], 1262.592)
        self.assertAlmostEqual(sum(row["value_ms"] for row in method["breakdown"]), method["cycle_ms"])
        self.assertAlmostEqual(sum(row["target_chain_ms"] for row in roles.values()) + 18, method["cycle_ms"])
        self.assertAlmostEqual(method["per_device_tokens_s"], 1000 * 4 * 5.2 / expected)

    def test_rank_means_are_computed_before_slowest_rank_selection(self):
        report, native = synthetic_report({"dp0": [1., 3.], "dp1": [2.2, 2.2]})
        method = attach(report, native)[0]
        self.assertEqual(method["status"], "LAYER_MEAN_ESTIMATE", method["reasons"])
        self.assertEqual(method["selected_rank"], "dp1")
        ranks = {row["rank"]: row for row in method["per_rank"]}
        self.assertAlmostEqual(ranks["dp0"]["cycle_ms"], 1260.8 * 2 + 1.792)
        self.assertAlmostEqual(ranks["dp1"]["cycle_ms"], 1260.8 * 2.2 + 1.792)
        self.assertAlmostEqual(method["cycle_ms"], 1260.8 * 2.2 + 1.792)
        self.assertEqual(method["class_means"], ranks["dp1"]["class_means"])
        self.assertEqual(method["role_details"], ranks["dp1"]["role_details"])
        self.assertEqual(method["source_partition"], ranks["dp1"]["source_partition"])
        self.assertTrue(any("PARTIAL_SOURCE_RANK_COVERAGE" in warning for warning in method["warnings"]))

    def test_named_packed_fp4_gmm_matches_without_calibration_and_excludes_independent_swiglu(self):
        report, native = synthetic_report(packed=True)
        method = attach(report, native)[0]
        self.assertEqual(method["status"], "LAYER_MEAN_ESTIMATE", method["reasons"])
        self.assertAlmostEqual(method["cycle_ms"], 1262.592)
        self.assertTrue(any("PACKED_OR_OPAQUE" in warning for warning in method["warnings"]))
        for row in method["class_means"]:
            self.assertAlmostEqual(row["routed_compute_ms"], .4)
            self.assertAlmostEqual(row["routed_compute_delta_ms"], .2)
        self.assertEqual(len(method["compute_evidence"]), 10)
        for proof in method["compute_evidence"]:
            self.assertEqual(proof["matched_calls"], 2)
            self.assertEqual(proof["logical_expert_axis_matches"], 0)
            self.assertTrue(any("Mxfp" in name for name in proof["operator_names"]))
            self.assertFalse(any("SwiGlu" in name for name in proof["operator_names"]))

    def test_legacy_fields_cannot_disable_or_multiply_fixed_14us(self):
        report, native = synthetic_report()
        baseline = attach(report, native)[0]
        report["cases"][0]["config"].update(extra_us=999999, data_fraction=0)
        changed = attach(report, native)[0]
        self.assertEqual(changed["cycle_ms"], baseline["cycle_ms"])
        self.assertEqual(changed["communication"]["extra_ms_per_call"], .014)
        self.assertEqual(changed["communication"]["remote_ratio"], 1.875)
        for row in changed["class_means"]:
            self.assertAlmostEqual(row["dispatch_target_ms"], .5 * 2.8125 + .014)
            self.assertAlmostEqual(row["combine_target_ms"], .25 * 2.8125 + .014)
            self.assertAlmostEqual(row["dispatch_delta_ms"], .90625 + .014)
            self.assertAlmostEqual(row["combine_delta_ms"], .453125 + .014)
            self.assertAlmostEqual(row["target_wall_ms"] - row["wall_ms"],
                                   .90625 + .453125 + .2 + .028)

    def test_fixed_14us_is_absolute_once_per_target_comm_call_in_method1_only(self):
        report, native = synthetic_report({"dp0": [1., 3.], "dp1": [2.2, 2.2]})
        reference = Path(__file__).resolve().parents[1] / "examples/communication-reference.synthetic.csv"
        report["communication_reference"] = {"rows": core.read_reference(reference)}
        with patch.object(core, "METHOD1_EXTRA_US", 0.):
            plain_first, plain_second = attach(report, native)[:2]
        fixed_first, fixed_second = attach(report, native)[:2]

        self.assertEqual(fixed_first["communication"]["extra_ms_per_call"], .014)
        self.assertEqual(fixed_second["communication"]["extra_ms_per_call"], 0.)
        for plain, fixed, added in ((plain_first, fixed_first, 128 * .014),
                                    (plain_second, fixed_second, 0.)):
            self.assertEqual(fixed["status"], "LAYER_MEAN_ESTIMATE", fixed["reasons"])
            self.assertEqual(fixed["selected_rank"], plain["selected_rank"])
            self.assertEqual(fixed["source_partition"], plain["source_partition"])
            self.assertAlmostEqual(fixed["cycle_ms"] - plain["cycle_ms"], added)
            for old_rank, rank in zip(plain["per_rank"], fixed["per_rank"]):
                self.assertEqual(rank["rank"], old_rank["rank"])
                self.assertAlmostEqual(rank["cycle_ms"] - old_rank["cycle_ms"], added)
                for old_round, cycle in zip(old_rank["rounds"], rank["rounds"]):
                    # The same fixed 1.792ms is added even when source time
                    # scales by 1x, 3x or 2.2x. It is not a payload multiplier.
                    self.assertAlmostEqual(cycle["cycle_ms"] - old_round["cycle_ms"], added)
                    self.assertEqual(cycle["source_partition"], old_round["source_partition"])
                    for old_role, role in zip(old_round["role_details"], cycle["role_details"]):
                        self.assertEqual(role["target_gap_ms"], old_role["target_gap_ms"])
                        role_added = role["target_layers"] * 2 * .014 if added else 0.
                        self.assertAlmostEqual(role["target_chain_ms"] - old_role["target_chain_ms"], role_added)
                    for old_class, row in zip(old_round["class_means"], cycle["class_means"]):
                        self.assertEqual(row["wall_ms"], old_class["wall_ms"])
                        self.assertEqual(row["routed_compute_target_ms"], old_class["routed_compute_target_ms"])
                        for component in ("dispatch", "combine"):
                            extra = .014 if added else 0.
                            self.assertEqual(row[component + "_ms"], old_class[component + "_ms"])
                            self.assertAlmostEqual(row[component + "_target_ms"] - old_class[component + "_target_ms"], extra)
                            self.assertAlmostEqual(row[component + "_delta_ms"] - old_class[component + "_delta_ms"], extra)
                        self.assertAlmostEqual(row["target_wall_ms"] - old_class["target_wall_ms"], .028 if added else 0.)
            self.assertAlmostEqual(fixed["per_device_tokens_s"], 20800 / fixed["cycle_ms"])

    def test_missing_routed_gmm_is_unavailable_not_zero_or_calibration_pending(self):
        report, native = synthetic_report()
        native["dp0"]["tasks"] = [task for task in native["dp0"]["tasks"] if "Grouped" not in task.name]
        methods = attach(report, native)
        self.assertTrue(all(method["status"] == "UNAVAILABLE" for method in methods))
        self.assertTrue(all("ROUTED_GMM_UNAVAILABLE" in method["reasons"][0] for method in methods))
        self.assertTrue(all(method["cycle_ms"] is None for method in methods))

    def test_unchanged_topk_needs_no_missing_gmm_increment_but_source_metric_stays_unknown(self):
        report, native = synthetic_report(topk=16)
        native["dp0"]["tasks"] = [task for task in native["dp0"]["tasks"] if "Grouped" not in task.name]
        method = attach(report, native)[0]
        self.assertEqual(method["status"], "LAYER_MEAN_ESTIMATE", method["reasons"])
        self.assertAlmostEqual(method["cycle_ms"], 1161 + 64 * .75 * (1.875 - 1) + 128 * .014)
        for row in method["class_means"]:
            self.assertIsNone(row["routed_compute_ms"])
            self.assertIsNone(row["routed_compute_target_ms"])
            self.assertEqual(row["routed_compute_delta_ms"], 0)
        self.assertEqual(method["compute_coverage"]["matched_reference_layers"], 0)
        self.assertEqual(method["compute_coverage"]["unmatched_reference_layers"], 10)
        self.assertTrue(any("ROUTED_GMM_UNMEASURED_UNCHANGED_TOPK" in warning for warning in method["warnings"]))
        self.assertTrue(all(proof["routed_compute_ms"] is None for proof in method["compute_evidence"]))

    def test_method_details_are_rank_class_means_with_original_round_details_retained(self):
        report, native = synthetic_report({"dp0": [1., 3.], "dp1": [2., 2.]})
        method = attach(report, native)[0]
        self.assertEqual(len(method["detail_rows"]), 2 * 3 * 3)
        for row in method["detail_rows"]:
            self.assertNotIn("round_id", row)
            self.assertEqual(row["rounds_averaged"], 2)
        rank = next(row for row in method["per_rank"] if row["rank"] == "dp0")
        self.assertEqual(len(rank["rounds"]), 2)
        self.assertEqual(len(rank["rounds"][0]["detail_rows"]), 9)
        self.assertEqual(len(rank["rounds"][1]["detail_rows"]), 9)
        dispatch = next(row for row in rank["detail_rows"] if row["role"] == "main"
                        and row["attention"] == "HCA" and row["component"] == "dispatch")
        self.assertAlmostEqual(dispatch["source_ms"], 1.)
        self.assertAlmostEqual(dispatch["weighted_delta_ms"], 31 * (1.8125 + .014))

    def test_source_schedule_and_partition_contradictions_remain_real_errors(self):
        for mutation in ("missing_layer", "wrong_wall", "overlap", "negative_residual"):
            report, native = synthetic_report()
            cycle = report["cases"][0]["measurements"][0]["rounds"][0]
            if mutation == "missing_layer":
                cycle["layers"].pop(3)
            elif mutation == "wrong_wall":
                cycle["layers"][3]["wall_ms"] += 1
            elif mutation == "overlap":
                cycle["layers"][2]["start_ns"] = cycle["layers"][1]["start_ns"]
            else:
                cycle["end_ns"] = cycle["layers"][-1]["start_ns"]
                cycle["duration_ms"] = (int(cycle["end_ns"]) - int(cycle["start_ns"])) / 1e6
            with self.subTest(mutation=mutation):
                methods = attach(report, native)
                self.assertEqual(methods[0]["status"], "UNAVAILABLE")
                self.assertTrue(methods[0]["reasons"])

    def test_no_historical_reference_only_disables_method2_preserving_source_evidence(self):
        report, native = synthetic_report()
        first, second = attach(report, native)[:2]
        self.assertEqual(first["status"], "LAYER_MEAN_ESTIMATE", first["reasons"])
        self.assertEqual(second["status"], "UNAVAILABLE")
        self.assertIsNone(second["cycle_ms"])
        self.assertEqual(second["comm_model"]["reference_bs"], 4)
        self.assertIn("fits", second["comm_model"])
        self.assertTrue(second["class_means"])
        self.assertTrue(second["role_details"])
        self.assertEqual(second["source_partition"]["residual_ms"], 18)
        self.assertTrue(all(row["cycle_ms"] is None for row in second["per_rank"]))
        self.assertTrue(all("target_wall_ms" not in row for row in second["class_means"]))

    def test_synthetic_reference_method2_needs_no_manual_basis_or_component_flags(self):
        report, native = synthetic_report()
        reference = Path(__file__).resolve().parents[1] / "examples/communication-reference.synthetic.csv"
        report["communication_reference"] = {"rows": core.read_reference(reference)}
        first, second = attach(report, native)[:2]
        self.assertEqual(second["status"], "LAYER_MEAN_ESTIMATE", second["reasons"])
        # DEMO reference: send EP128 / fitted EP16 = 4, return = 2.5;
        # the configured TopK increase contributes another factor of 1.5.
        self.assertAlmostEqual(second["comm_model"]["multipliers"]["main"]["dispatch"], 6.)
        self.assertAlmostEqual(second["comm_model"]["multipliers"]["main"]["combine"], 3.75)
        expected = 1161 + 64 * (.5 * (6. - 1) + .25 * (3.75 - 1) + .4 * .5)
        self.assertAlmostEqual(second["cycle_ms"], expected)
        self.assertEqual(first["status"], "LAYER_MEAN_ESTIMATE")


class OperatorMeanTests(unittest.TestCase):
    def report(self, *args, **kwargs):
        from tests.test_ep128_target_stages import native_stage_report
        return native_stage_report(*args, **kwargs)

    def operators(self, method, role="main", attention="HCA"):
        group = next(row for row in method["operator_hotspots"]
                     if (row["role"], row["attention"]) == (role, attention))
        self.assertEqual(group["status"], "MATCHED", group["reason"])
        return group

    def test_kernel_costs_remain_native_when_model_increment_changes_sign(self):
        actual = []
        for topk, expected_delta in ((24, .2), (8, -.2)):
            report, native = self.report(topk=topk)
            method = attach(report, native)[0]
            parent = next(row for row in method["class_means"]
                          if row["role"] == "main" and row["attention"] == "HCA")
            self.assertAlmostEqual(parent["routed_compute_delta_ms"], expected_delta)
            result = self.operators(method)
            self.assertAlmostEqual(sum(row["kernel_sum_ms"] for row in result["kernels"]
                if "GroupedMatmul" in row["name"]), .4)
            self.assertTrue(all(row["kernel_sum_ms"] >= 0 for row in result["kernels"]))
            self.assertNotIn("rows", result)
            self.assertTrue(all("delta_ms" not in row for row in result["kernels"]))
            actual.append(result)
        self.assertEqual(actual[0], actual[1])

    def test_rare_operator_uses_all_reference_layers_and_rounds_as_denominator(self):
        report, native = self.report({"dp0": [1., 3.]})
        layer = report["cases"][0]["measurements"][0]["rounds"][0]["layers"][1]
        start = int(layer["start_ns"]) + 3_400_000
        native["dp0"]["tasks"].append(SimpleNamespace(name="RareOnly", op_type="Copy", label="Copy RareOnly",
            start_ns=start, end_ns=start + 200_000, input_shapes="4,16", row_index=99999))
        native["dp0"]["tasks"].sort(key=lambda task: task.start_ns)
        result = self.operators(attach(report, native)[0])
        rare = next(row for row in result["kernels"] if row["name"] == "RareOnly")
        self.assertEqual(result["layer_round_samples"], 8)
        self.assertEqual(rare["calls_mean"], 1 / 8)
        self.assertAlmostEqual(rare["kernel_sum_ms"], .2 / 8)
        self.assertAlmostEqual(rare["duration_mean_ms"], .2)
        self.assertAlmostEqual(rare["duration_p90_ms"], .2)
        self.assertEqual(rare["example"]["layer"], 1)
        self.assertEqual(rare["example"]["round_id"], "hc-0")
        self.assertTrue(all(row["example"]["layer"] != 0 for row in result["kernels"]))

    def test_unidentified_gmm_remains_unknown_without_fabricated_zero_kernel_rows(self):
        report, native = self.report(topk=16)
        native["dp0"]["tasks"] = [task for task in native["dp0"]["tasks"] if not core._lm_gmm(task)]
        method = attach(report, native)[0]
        result = self.operators(method)
        parent = next(row for row in method["class_means"] if row["role"] == "main" and row["attention"] == "HCA")
        self.assertIsNone(parent["routed_compute_ms"])
        self.assertEqual(parent["routed_compute_delta_ms"], 0)
        self.assertTrue(all("GroupedMatmul" not in row["name"] for row in result["kernels"]))

    def test_operator_evidence_failure_does_not_change_any_parent_number(self):
        report, native = self.report()
        baseline = attach(copy.deepcopy(report), native)[0]
        with patch.object(core, "_lm_operator_member", side_effect=ValueError("synthetic missing operator evidence")):
            method = attach(report, native)[0]
        for field in ("cycle_ms", "tpot_ms", "per_device_tokens_s", "selected_rank", "class_means", "breakdown"):
            self.assertEqual(method[field], baseline[field])
        self.assertTrue(method["operator_hotspots"])
        for group in method["operator_hotspots"]:
            self.assertEqual(group["status"], "UNAVAILABLE")
            self.assertEqual(group["kernels"], [])
            self.assertTrue(group["reason"])

    def test_one_time_index_and_source_hotspot_means_shared_by_all_methods(self):
        report, native = self.report({"dp0": [1., 2.], "dp1": [1.5, 2.5]})
        reference = Path(__file__).resolve().parents[1] / "examples/communication-reference.synthetic.csv"
        report["communication_reference"] = {"rows": core.read_reference(reference)}
        with patch.object(core, "ATimeIndex", wraps=core.ATimeIndex) as indices, \
             patch.object(core, "_lm_operator_source_means", wraps=core._lm_operator_source_means) as means, \
             patch.object(core, "_lm_operator_layer", wraps=core._lm_operator_layer) as layers:
            methods = attach(report, native)
        self.assertEqual([row["selected_rank"] for row in methods], ["dp1"] * 3)
        self.assertGreaterEqual(indices.call_count, 1)
        self.assertLessEqual(indices.call_count, 2)  # At most one index per capture.
        self.assertEqual(means.call_count, 1)
        # Each rank worker scans its ten eligible layers once per round before
        # releasing native tasks; this avoids retaining/pickling whole traces
        # or rereading CSV after global rank selection. Only selected-rank PMU
        # and quantiles are aggregated, once, for all three target methods.
        self.assertEqual(layers.call_count, 40)
        self.assertEqual({call.args[2]["rank"] for call in layers.call_args_list}, {"dp0", "dp1"})
        self.assertTrue(all(self.operators(method)["kernels"] == self.operators(methods[0])["kernels"]
                            for method in methods))


if __name__ == "__main__":
    unittest.main()
