"""Source sweep values stay on the recorded source rank and exact HC bounds."""

import copy
import math
import unittest

import profile_ep128 as core


PARTS = ("main_layer_envelopes", "main_internal_gaps", "draft_layer_envelopes",
         "draft_internal_gaps", "residual_once")
EPOCH = 1793591950807238775


def source_round(index, scale=1):
    start = EPOCH + index * 1000000000
    spans = (("main", 0, 0, 20), ("main", 1, 25, 50),
             ("draft", 0, 60, 70), ("draft", 1, 75, 90))
    return {"round_id": "hc-" + str(index), "start_ns": str(start),
            "end_ns": str(start + int(100000000 * scale)), "duration_ms": 100 * scale,
            "layers": [{"role": role, "layer": layer,
                        "start_ns": str(start + int(left * 1000000 * scale)),
                        "end_ns": str(start + int(right * 1000000 * scale)),
                        "wall_ms": (right - left) * scale}
                       for role, layer, left, right in spans]}


def source_case():
    return {"name": "SOURCE_FIXTURE", "config": {"draft_tokens": 3,
                "effective_tokens_per_round": 2.5, "exclude_target_layers": [0]},
            "source": {"aggregate": {"selected_rank": "dp1", "cycle_ms": 150.,
                "per_device_tokens_s": 1234.5}},
            "measurements": [{"rank": "dp0", "rounds": [source_round(0, .5)]},
                {"rank": "dp1", "rounds": [source_round(0), source_round(1, 2)]}],
            "projection": {"layer_mean_methods": [{"selected_rank": "dp0",
                "cycle_ms": 999., "source_partition": {"residual_ms": 500.}}]}}


class SourceSweepTests(unittest.TestCase):
    def assert_unknown_parts(self, result):
        self.assertEqual([part["key"] for part in result["breakdown"]], list(PARTS))
        self.assertTrue(all(part["value_ms"] is None for part in result["breakdown"]))
        self.assertTrue(result["breakdown_reason"])

    def test_selected_source_rank_headlines_and_five_parts_close(self):
        result = core.source_sweep_summary(source_case())
        self.assertEqual(result["status"], "AVAILABLE")
        self.assertIsNone(result["reason"])
        self.assertIsNone(result["breakdown_reason"])
        self.assertEqual(result["selected_rank"], "dp1")
        self.assertEqual(result["per_device_tokens_s"], 1234.5)
        self.assertEqual(result["cycle_ms"], 150.)
        self.assertEqual(result["itl_proxy_ms"], 150.)
        self.assertEqual(result["tpot_ms"], 60.)
        self.assertEqual([row["key"] for row in result["breakdown"]], list(PARTS))
        self.assertEqual([row["value_ms"] for row in result["breakdown"]],
                         [67.5, 7.5, 37.5, 7.5, 30.])
        self.assertEqual(math.fsum(row["value_ms"] for row in result["breakdown"]), result["cycle_ms"])
        # Main0's measured 30ms mean stays included, although target references exclude it.
        self.assertEqual(result["breakdown"][0]["value_ms"], (45 + 90) / 2)

    def test_no_report_fields_or_source_objects_are_modified(self):
        case = source_case()
        before = copy.deepcopy(case)
        first = core.source_sweep_summary(case)
        second = core.source_sweep_summary(case)
        self.assertEqual(case, before)
        self.assertEqual(first, second)
        first["breakdown"][0]["value_ms"] = -123
        self.assertEqual(case, before)
        self.assertEqual(second["breakdown"][0]["value_ms"], 67.5)

    def test_no_fallback_to_target_or_another_rank_when_selected_measurement_missing(self):
        case = source_case()
        case["measurements"] = case["measurements"][:1]
        result = core.source_sweep_summary(case)
        self.assertEqual(result["status"], "PARTIAL")
        self.assertEqual(result["cycle_ms"], 150.)
        self.assertEqual(result["tpot_ms"], 60.)
        self.assertEqual(result["per_device_tokens_s"], 1234.5)
        self.assert_unknown_parts(result)

    def test_duplicate_selected_measurement_is_not_silently_picked(self):
        case = source_case()
        case["measurements"].append(copy.deepcopy(case["measurements"][1]))
        self.assert_unknown_parts(core.source_sweep_summary(case))

    def test_nonselected_rank_and_target_evidence_do_not_affect_baseline(self):
        case = source_case()
        expected = core.source_sweep_summary(case)
        case["measurements"][0] = {"rank": "dp0", "rounds": "malformed unrelated evidence"}
        case["projection"] = {"layer_mean_methods": [{"selected_rank": "dp0", "cycle_ms": 1000000}]}
        self.assertEqual(core.source_sweep_summary(case), expected)

    def test_aggregate_cycle_disagreement_keeps_headline_but_invalidates_parts(self):
        case = source_case()
        case["source"]["aggregate"]["cycle_ms"] = 151.
        result = core.source_sweep_summary(case)
        self.assertEqual(result["cycle_ms"], 151.)
        self.assertEqual(result["itl_proxy_ms"], 151.)
        self.assertEqual(result["tpot_ms"], 60.4)
        self.assert_unknown_parts(result)

    def test_missing_noninteger_and_inconsistent_boundaries_are_unknown(self):
        for mode in ("missing", "float", "round_duration", "layer_wall", "overlap", "missing_role", "prefix"):
            with self.subTest(mode=mode):
                case = source_case()
                row = case["measurements"][1]["rounds"][0]
                layer = row["layers"][0]
                if mode == "missing":
                    del layer["start_ns"]
                elif mode == "float":
                    layer["start_ns"] = float(layer["start_ns"])
                elif mode == "round_duration":
                    row["duration_ms"] += .001
                elif mode == "layer_wall":
                    layer["wall_ms"] += .001
                elif mode == "overlap":
                    row["layers"][1]["start_ns"] = layer["start_ns"]
                elif mode == "missing_role":
                    row["layers"] = [item for item in row["layers"] if item["role"] == "main"]
                else:
                    row["start_ns"] = str(int(row["start_ns"]) - 1)
                result = core.source_sweep_summary(case)
                self.assertEqual(result["cycle_ms"], 150.)
                self.assert_unknown_parts(result)

    def test_missing_rounds_or_layers_do_not_become_zero(self):
        for value in ([], None, "not a list", [{"layers": []}]):
            with self.subTest(value=value):
                case = source_case()
                case["measurements"][1]["rounds"] = value
                self.assert_unknown_parts(core.source_sweep_summary(case))

    def test_invalid_neff_only_removes_tpot_and_keeps_valid_source_parts(self):
        for value in (None, True, 0, -1, math.nan, math.inf, 4.1, "2.5"):
            with self.subTest(neff=value):
                case = source_case()
                case["config"]["effective_tokens_per_round"] = value
                result = core.source_sweep_summary(case)
                self.assertIsNone(result["tpot_ms"])
                self.assertEqual(result["itl_proxy_ms"], 150.)
                self.assertEqual(result["status"], "PARTIAL")
                self.assertIsNone(result["breakdown_reason"])

    def test_old_json_without_optional_s_still_uses_valid_neff(self):
        case = source_case()
        del case["config"]["draft_tokens"]
        self.assertEqual(core.source_sweep_summary(case)["tpot_ms"], 60.)

    def test_valid_s_alias_and_matching_dual_aliases_preserve_source_values(self):
        case = source_case()
        expected = core.source_sweep_summary(case)
        case["config"]["num_speculative_tokens"] = case["config"].pop("draft_tokens")
        self.assertEqual(core.source_sweep_summary(case), expected)
        case["config"]["draft_tokens"] = case["config"]["num_speculative_tokens"]
        self.assertEqual(core.source_sweep_summary(case), expected)

    def test_neff_above_alias_limit_removes_only_tpot(self):
        case = source_case()
        expected = core.source_sweep_summary(case)
        case["config"].pop("draft_tokens")
        case["config"]["num_speculative_tokens"] = 1
        result = core.source_sweep_summary(case)
        self.assertIsNone(result["tpot_ms"])
        self.assertEqual(result["status"], "PARTIAL")
        self.assertTrue(result["reason"])
        self.assertIsNone(result["breakdown_reason"])
        for field in ("selected_rank", "cycle_ms", "itl_proxy_ms", "per_device_tokens_s", "breakdown"):
            self.assertEqual(result[field], expected[field], field)

    def test_conflicting_or_invalid_s_aliases_remove_only_tpot(self):
        for primary, alias in ((3, 7), (True, 1), (3, "3"), (3, 3.0), (3, -1), (None, 3)):
            with self.subTest(primary=primary, alias=alias):
                case = source_case()
                expected = core.source_sweep_summary(case)
                case["config"].update(draft_tokens=primary, num_speculative_tokens=alias)
                result = core.source_sweep_summary(case)
                self.assertIsNone(result["tpot_ms"])
                self.assertEqual(result["status"], "PARTIAL")
                self.assertTrue(result["reason"])
                self.assertIsNone(result["breakdown_reason"])
                for field in ("selected_rank", "cycle_ms", "itl_proxy_ms", "per_device_tokens_s", "breakdown"):
                    self.assertEqual(result[field], expected[field], field)

    def test_source_headlines_validate_independently_and_zero_throughput_is_known(self):
        case = source_case()
        case["source"]["aggregate"]["per_device_tokens_s"] = 0
        self.assertEqual(core.source_sweep_summary(case)["per_device_tokens_s"], 0)
        for value in (math.nan, math.inf, -1, True, "123", 10 ** 1000):
            with self.subTest(value=repr(value)):
                case["source"]["aggregate"]["per_device_tokens_s"] = value
                result = core.source_sweep_summary(case)
                self.assertIsNone(result["per_device_tokens_s"])
                self.assertEqual(result["cycle_ms"], 150.)
                self.assertIsNone(result["breakdown_reason"])

    def test_missing_source_has_no_target_substitution_or_fabricated_itl(self):
        for case in ({}, None, {"source": None}, {"source": {"aggregate": []}}):
            with self.subTest(case=case):
                result = core.source_sweep_summary(case)
                self.assertEqual(result["status"], "UNAVAILABLE")
                self.assertTrue(result["reason"])
                self.assertIsNone(result["cycle_ms"])
                self.assertIsNone(result["itl_proxy_ms"])
                self.assert_unknown_parts(result)


if __name__ == "__main__":
    unittest.main()
