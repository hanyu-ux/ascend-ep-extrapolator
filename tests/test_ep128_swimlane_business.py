"""Worker business order is derived from saved same-round Host ownership."""

from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import patch

from ep128_swimlane_business import build_worker_business
import test_ep128_swimlane_share as fixtures


EPOCH = fixtures.EPOCH
ORIGIN = EPOCH + 100


def extra_event(number, kind, name, start, end, py, parent, *, tid="7"):
    return {"event_index": number, "name": "vllm/v1/worker/gpu/model_runner.py(999): " + name,
            "kind": kind, "start_ns": EPOCH+start, "end_ns": EPOCH+end, "duration_ns": end-start,
            "pid": "10", "tid": tid, "python_id": str(py), "python_parent_id": str(parent),
            "cat": "python_function", "event_type": "python"}


def case_fixture():
    source = fixtures.case_fixture()
    source["host_events"].extend([
        extra_event(110, "sample_sample", "sample", 25, 40, 110, 2),
        extra_event(111, "sample_postprocess_sampled", "postprocess_sampled", 42, 48, 111, 2),
        extra_event(112, "sample__copy_num_computed_tokens_to_cpu", "_copy_num_computed_tokens_to_cpu", 48, 49, 112, 2),
    ])
    return fixtures.build(fixtures.payload([source]))["cases"][0]


def change_event(case, number, **fields):
    for collection in (case.get("host_events", []), case.get("anchor_links", {}).get("source_events", [])):
        for event in collection:
            if event["event_index"] == number:
                event.update(fields)


def pieces_for(result, kind=None, event_id=None):
    return [piece for piece in result["pieces"]
            if (kind is None or piece["kind"] == kind)
            and (event_id is None or piece.get("event_index") == event_id)]


def spans(result, kind=None, event_id=None):
    return [(int(piece["start_ns"])-EPOCH, int(piece["end_ns"])-EPOCH)
            for piece in pieces_for(result, kind, event_id)]


def assert_closed(test, result, start=-200, end=450):
    pieces = result["pieces"]
    test.assertTrue(result["closed"])
    test.assertEqual(int(result["start_ns"]), EPOCH+start)
    test.assertEqual(int(result["end_ns"]), EPOCH+end)
    test.assertEqual(int(result["duration_ns"]), end-start)
    test.assertEqual(sum(int(piece["duration_ns"]) for piece in pieces), end-start)
    test.assertTrue(all(int(piece["start_ns"]) < int(piece["end_ns"]) for piece in pieces))
    test.assertTrue(all(int(a["end_ns"]) == int(b["start_ns"]) for a, b in zip(pieces, pieces[1:])))


class WorkerBusinessTests(unittest.TestCase):
    def test_three_outer_scopes_use_saved_round_context_and_keep_negative_start(self):
        case = case_fixture()
        before = deepcopy(case)
        with patch("builtins.open", side_effect=AssertionError("No capture reads")), \
                patch.object(Path, "open", side_effect=AssertionError("No original trace reads")):
            result = build_worker_business(case, ORIGIN)
        self.assertEqual(case, before)
        scopes = result["outer_scopes"]
        self.assertEqual([scope["role"] for scope in scopes], ["current_execute_n", "sample_tokens_n", "next_execute_n1"])
        self.assertEqual([scope["event_index"] for scope in scopes], [1, 2, 4])
        self.assertEqual([(int(scope["start_ns"])-EPOCH, int(scope["end_ns"])-EPOCH) for scope in scopes],
                         [(-200, 10), (20, 150), (160, 450)])
        self.assertAlmostEqual(scopes[0]["start_ms"], -300/1e6)
        assert_closed(self, result)

    def test_unknown_gaps_between_methods_are_recorded_and_are_not_idle(self):
        result = build_worker_business(case_fixture(), ORIGIN)
        self.assertEqual(spans(result, "UNKNOWN_GAP"), [(10, 20), (150, 160)])
        self.assertFalse(any("idle" in str(piece.get("name", "")).lower() for piece in pieces_for(result, "UNKNOWN_GAP")))
        assert_closed(self, result)

    def test_sample_subcalls_and_propose_partition_only_the_submission_wall(self):
        result = build_worker_business(case_fixture(), ORIGIN)
        self.assertEqual(spans(result, "sample_submit", 110), [(25, 40)])
        self.assertEqual(spans(result, "sample_postprocess", 111), [(42, 48)])
        self.assertEqual(spans(result, "sample_copy", 112), [(48, 49)])
        self.assertEqual(spans(result, "propose", 3), [(50, 140)])
        sample = [piece for piece in result["pieces"] if piece["role"] == "sample_tokens_n"]
        self.assertEqual(sum(int(piece["duration_ns"]) for piece in sample), 130)
        self.assertEqual([(int(piece["start_ns"])-EPOCH, int(piece["end_ns"])-EPOCH)
                          for piece in sample if piece["kind"] == "sample_remainder"],
                         [(20, 25), (40, 42), (49, 50), (140, 150)])

    def test_same_time_wrong_parent_cannot_become_sample_submission(self):
        case = case_fixture()
        change_event(case, 110, python_parent_id="999999")
        result = build_worker_business(case, ORIGIN)
        self.assertEqual(pieces_for(result, "sample_submit", 110), [])
        self.assertTrue(any(int(piece["start_ns"]) <= EPOCH+25 and int(piece["end_ns"]) >= EPOCH+40
                            and piece["kind"] == "sample_remainder" for piece in result["pieces"]))
        assert_closed(self, result)

    def test_same_named_submission_on_another_thread_is_not_borrowed(self):
        case = case_fixture()
        change_event(case, 110, tid="99")
        result = build_worker_business(case, ORIGIN)
        self.assertEqual(pieces_for(result, "sample_submit", 110), [])
        assert_closed(self, result)

    def test_absent_thread_ids_do_not_become_proof_of_one_worker_thread(self):
        case = case_fixture()
        for event in case["host_events"]:
            event.pop("pid", None)
            event.pop("tid", None)
        result = build_worker_business(case, ORIGIN)
        self.assertNotEqual(result["status"], "READY")
        self.assertFalse(any(piece["kind"] not in ("UNKNOWN_GAP", "UNKNOWN_OVERLAP")
                             for piece in result["pieces"]))
        self.assertEqual(result["trace_observations"], [])

    def test_explicit_sample_wrapper_parent_chain_is_allowed(self):
        case = case_fixture()
        wrapper = extra_event(120, "context_wrapper", "decorate_context", 22, 41, 120, 2)
        case["host_events"].append(wrapper)
        change_event(case, 110, python_parent_id="120")
        result = build_worker_business(case, ORIGIN)
        self.assertEqual(spans(result, "sample_submit", 110), [(25, 40)])
        assert_closed(self, result)

    def test_incomparable_sample_siblings_do_not_win_by_shorter_duration(self):
        case = case_fixture()
        change_event(case, 110, end_ns=str(EPOCH+80), duration_ns=55)
        result = build_worker_business(case, ORIGIN)
        unknown = spans(result, "UNKNOWN_OVERLAP")
        self.assertTrue(any(a <= 50 and b >= 80 for a, b in unknown))
        self.assertFalse(any(a < 80 and b > 50 for a, b in spans(result, "propose", 3)))
        assert_closed(self, result)

    def test_outer_method_overlap_stays_unknown_in_the_intersection(self):
        case = case_fixture()
        change_event(case, 2, start_ns=str(EPOCH), duration_ns=150)
        result = build_worker_business(case, ORIGIN)
        self.assertEqual([(int(scope["start_ns"])-EPOCH, int(scope["end_ns"])-EPOCH)
                          for scope in result["outer_scopes"]], [(-200, 10), (0, 150), (160, 450)])
        self.assertIn((0, 10), spans(result, "UNKNOWN_OVERLAP"))
        assert_closed(self, result)

    def test_clock_failure_has_no_worker_host_scopes_or_pieces(self):
        case = case_fixture()
        case.update(status="ERROR", error="CLOCK_MISMATCH")
        case["trace"]["status"] = "CLOCK_MISMATCH"
        result = build_worker_business(case, ORIGIN)
        self.assertEqual(result["status"], "UNAVAILABLE")
        self.assertEqual(result["outer_scopes"], [])
        self.assertEqual(result["pieces"], [])

    def test_next_graph_and_execute_return_are_not_clipped_to_the_NPU_cycle(self):
        case = case_fixture()
        result = build_worker_business(case, ORIGIN)
        self.assertEqual(spans(result, "next_graph", 7), [(390, 430)])
        self.assertEqual(int(result["end_ns"]), EPOCH+450)
        self.assertGreater(int(result["end_ns"]), int(case["rounds"][0]["end_ns"]))
        self.assertTrue(any(int(piece["end_ns"]) == EPOCH+450 for piece in result["pieces"]))

    def test_saved_graph_id_does_not_replace_a_missing_parent_chain(self):
        case = case_fixture()
        change_event(case, 7, python_parent_id="999999")
        result = build_worker_business(case, ORIGIN)
        self.assertEqual(pieces_for(result, "next_graph", 7), [])
        self.assertTrue(any(piece["kind"] == "execute_remainder"
                            and int(piece["start_ns"]) <= EPOCH+390
                            and int(piece["end_ns"]) >= EPOCH+430 for piece in result["pieces"]))
        assert_closed(self, result)

    def test_next_execute_has_distinct_input_slot_and_metadata_functions(self):
        result = build_worker_business(case_fixture(), ORIGIN)
        self.assertEqual(spans(result, "runner_inputs", 5), [(200, 230)])
        self.assertEqual(spans(result, "runner_attn", 9), [(232, 238)])
        self.assertEqual(spans(result, "state_attn", 6), [(240, 380)])
        assert_closed(self, result)

    def test_preparation_prefix_is_an_explicit_synthetic_aggregate_not_a_function(self):
        result = build_worker_business(case_fixture(), ORIGIN)
        prefix, = pieces_for(result, "prefix")
        self.assertEqual((int(prefix["start_ns"])-EPOCH, int(prefix["end_ns"])-EPOCH), (160, 200))
        self.assertTrue(prefix["synthetic"])
        self.assertNotIn(".py(", prefix["name"])
        self.assertTrue(all(not piece.get("synthetic", False) for piece in pieces_for(result, "runner_attn")))

    def test_missing_selected_sample_context_cannot_fall_back_to_a_nearby_call(self):
        case = case_fixture()
        case["rounds"][0]["host_pairing"]["context"]["current_sample"] = 777777
        result = build_worker_business(case, ORIGIN)
        self.assertFalse(any(scope.get("event_index") == 2 for scope in result["outer_scopes"]))
        self.assertEqual(pieces_for(result, "propose", 3), [])

    def test_identical_event_ids_in_another_case_never_reuse_previous_times(self):
        first = case_fixture()
        second = deepcopy(first)
        second.update(case="another-case", trace={**second["trace"], "sha256": "another-trace"})
        def shift(value):
            if isinstance(value, dict):
                for key, item in list(value.items()):
                    if key in ("start_ns", "end_ns", "origin_ns", "timestamp_ns") and item is not None:
                        value[key] = str(int(item)+1000)
                    else:
                        shift(item)
            elif isinstance(value, list):
                for item in value:
                    shift(item)
        shift(second)
        a = build_worker_business(first, ORIGIN)
        b = build_worker_business(second, ORIGIN+1000)
        self.assertEqual([int(scope["start_ns"]) for scope in b["outer_scopes"]],
                         [int(scope["start_ns"])+1000 for scope in a["outer_scopes"]])
        self.assertEqual([scope["start_ms"] for scope in b["outer_scopes"]],
                         [scope["start_ms"] for scope in a["outer_scopes"]])

    def test_only_the_recorded_worker_methods_get_outer_scopes(self):
        result = build_worker_business(case_fixture(), ORIGIN)
        self.assertEqual({scope["role"] for scope in result["outer_scopes"]},
                         {"current_execute_n", "sample_tokens_n", "next_execute_n1"})
        self.assertFalse(any("EngineCore" in scope["name"] or "output_thread" in scope["name"]
                             for scope in result["outer_scopes"]))

    def test_trace_facts_distinguish_main_and_draft_HC_and_only_claim_temporal_order(self):
        result = build_worker_business(case_fixture(), ORIGIN)
        facts = {(fact["scope"], fact["device_role"]): fact for fact in result["trace_observations"]}
        expected = {("current_execute_n", "main_hc"): 240,
                    ("sample_tokens_n", "draft_hc"): 200,
                    ("propose_n", "main_hc"): 110,
                    ("propose_n", "draft_hc"): 210}
        self.assertEqual(set(facts), set(expected))
        for key, delta in expected.items():
            with self.subTest(scope=key):
                fact = facts[key]
                self.assertEqual(fact["kind"], "TRACE_OBSERVATION")
                self.assertEqual(fact["host_return_before_hc_end_ns"], delta)
                self.assertAlmostEqual(fact["host_return_before_hc_end_ms"], delta/1e6)
                self.assertEqual(fact["relation"], "HOST_RETURN_PRECEDES_HC_ENVELOPE_END")
                self.assertIn("not proof", fact["limitation"])
                self.assertNotIn("depends_on", fact)

    def test_rejected_propose_identity_or_parent_does_not_reappear_as_a_round_fact(self):
        for fields in ({"python_parent_id": "999999"}, {"kind": "state_attn"}):
            with self.subTest(fields=fields):
                case = case_fixture()
                change_event(case, 3, **fields)
                result = build_worker_business(case, ORIGIN)
                self.assertEqual(pieces_for(result, "propose", 3), [])
                self.assertFalse(any(fact["scope"] == "propose_n" for fact in result["trace_observations"]))

    def test_HC_ending_before_host_return_is_not_reported_as_hidden_forward_time(self):
        case = case_fixture()
        case["rounds"][0]["regions"]["main_hc"]["end_ns"] = str(EPOCH+130)
        result = build_worker_business(case, ORIGIN)
        fact = next(item for item in result["trace_observations"]
                    if item["scope"] == "propose_n" and item["device_role"] == "main_hc")
        self.assertEqual(fact["host_return_before_hc_end_ns"], -10)
        self.assertEqual(fact["relation"], "HOST_RETURN_DOES_NOT_PRECEDE_HC_ENVELOPE_END")


if __name__ == "__main__":
    unittest.main()
