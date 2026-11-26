"""Exclusive Host display pieces require recorded ancestor ownership."""

from copy import deepcopy
import json
import unittest

from ep128_swimlane_partition import partition_host_lane


def parent(start=0, end=100, number=1):
    return {"event_index": number, "name": "runner.prepare_attn", "kind": "runner_attn",
            "start_ns": str(start), "end_ns": str(end)}


def child(number, start, end, kind="metadata_builder", ancestors=None, *, verified=True, name=None):
    chain = list([1] if ancestors is None else ancestors)
    return {"event_index": number, "name": name or kind, "kind": kind,
            "start_ns": str(start), "end_ns": str(end),
            "parent_chain_verified": verified, "ancestor_event_indexes": chain,
            "relation": "python_parent_id" if verified else "time_containment_only"}


def spans(result, kind=None):
    return [(int(piece["start_ns"]), int(piece["end_ns"])) for piece in result["pieces"]
            if kind is None or piece["kind"] == kind]


def event_spans(result, number):
    return [(int(piece["start_ns"]), int(piece["end_ns"])) for piece in result["pieces"]
            if piece.get("event_index") == number and piece["kind"] not in ("remainder", "ambiguous")]


def check_closed(test, result, start, end):
    pieces = result["pieces"]
    test.assertTrue(result["closed"])
    test.assertEqual(int(result["total_duration_ns"]), end-start)
    test.assertEqual(int(result["exclusive_duration_ns"]), end-start)
    test.assertEqual(int(pieces[0]["start_ns"]), start)
    test.assertEqual(int(pieces[-1]["end_ns"]), end)
    test.assertEqual(sum(int(piece["duration_ns"]) for piece in pieces), end-start)
    test.assertTrue(all(int(piece["end_ns"]) > int(piece["start_ns"]) for piece in pieces))
    test.assertTrue(all(int(left["end_ns"]) == int(right["start_ns"])
                        for left, right in zip(pieces, pieces[1:])))


class ExclusiveHostPartitionTests(unittest.TestCase):
    def test_nested_graph_metadata_uses_leaf_and_subtracts_from_inclusive_parent(self):
        calls = [child(2, 10, 90, "draft_graph", name="run_fullgraph"),
                 child(3, 20, 60, "draft_build_metadata", [2, 1], name="build_draft_attn_metadatas")]
        original = deepcopy(calls)
        result = partition_host_lane(parent(), calls, "outer_propose")
        check_closed(self, result, 0, 100)
        self.assertEqual(event_spans(result, 2), [(10, 20), (60, 90)])
        self.assertEqual(event_spans(result, 3), [(20, 60)])
        graph = [piece for piece in result["pieces"] if piece.get("event_index") == 2]
        self.assertTrue(all(piece["is_remainder_of_inclusive"] for piece in graph))
        self.assertTrue(all("其余wall" in piece["name"] for piece in graph))
        self.assertEqual(calls, original)
        self.assertEqual(result["source_children"], original)
        result["source_children"][0]["name"] = "changed only in output"
        self.assertEqual(calls, original)

    def test_DP_sync_all_reduce_and_wait_are_exclusive_pieces(self):
        calls = [child(2, 10, 90, "dp_sync"), child(3, 20, 80, "all_reduce", [2, 1]),
                 child(4, 30, 70, "cpu_wait", [3, 2, 1])]
        result = partition_host_lane(parent(), calls, "preparation_dp_aggregate")
        check_closed(self, result, 0, 100)
        self.assertEqual(event_spans(result, 2), [(10, 20), (80, 90)])
        self.assertEqual(event_spans(result, 3), [(20, 30), (70, 80)])
        self.assertEqual(event_spans(result, 4), [(30, 70)])
        self.assertEqual(result["wait_exclusive_ns"], 40)

    def test_unrecorded_wait_stays_unknown_not_zero(self):
        calls = [child(2, 10, 90, "dp_sync"), child(3, 20, 80, "all_reduce", [2, 1])]
        result = partition_host_lane(parent(), calls, "preparation_dp_aggregate")
        check_closed(self, result, 0, 100)
        self.assertIsNone(result["wait_exclusive_ns"])
        self.assertFalse(any(piece["kind"] == "cpu_wait" for piece in result["pieces"]))

    def test_crossing_siblings_are_ambiguous_where_they_overlap(self):
        calls = [child(2, 10, 70, "metadata_builder"), child(3, 40, 90, "slot_mappings")]
        result = partition_host_lane(parent(), calls, "runner_attn")
        check_closed(self, result, 0, 100)
        self.assertEqual(event_spans(result, 2), [(10, 40)])
        self.assertEqual(spans(result, "ambiguous"), [(40, 70)])
        self.assertEqual(event_spans(result, 3), [(70, 90)])
        ambiguous, = [piece for piece in result["pieces"] if piece["kind"] == "ambiguous"]
        self.assertEqual(set(ambiguous["candidate_event_indexes"]), {2, 3})

    def test_shorter_contained_sibling_does_not_win_by_duration(self):
        calls = [child(2, 10, 90, "metadata_builder"), child(3, 40, 60, "slot_mappings")]
        result = partition_host_lane(parent(), calls, "runner_attn")
        self.assertEqual(spans(result, "ambiguous"), [(40, 60)])
        self.assertEqual(event_spans(result, 3), [])
        self.assertEqual(event_spans(result, 2), [(10, 40), (60, 90)])

    def test_identical_siblings_are_ambiguous_without_ancestor_proof(self):
        result = partition_host_lane(parent(), [child(2, 10, 90), child(3, 10, 90)], "state_attn")
        check_closed(self, result, 0, 100)
        self.assertEqual(spans(result, "ambiguous"), [(10, 90)])
        self.assertEqual(event_spans(result, 2), [])
        self.assertEqual(event_spans(result, 3), [])

    def test_identical_recorded_descendant_can_own_the_leaf_interval(self):
        calls = [child(2, 10, 90, "metadata_builder"), child(3, 10, 90, "request_metadata", [2, 1])]
        result = partition_host_lane(parent(), calls, "state_attn")
        self.assertEqual(event_spans(result, 3), [(10, 90)])
        self.assertEqual(event_spans(result, 2), [])
        self.assertEqual(spans(result, "ambiguous"), [])

    def test_child_cannot_erase_sibling_ambiguity_with_a_contradictory_saved_chain(self):
        calls = [child(2, 10, 90, "draft_graph"),
                 child(3, 20, 80, "metadata_builder"),
                 child(4, 30, 70, "request_metadata", [3, 2, 1])]
        # Call 3 records a direct parent of 1, while call 4 claims its parent
        # path from call 3 continues through call 2. Those saved paths conflict.
        result = partition_host_lane(parent(), calls, "state_attn")
        check_closed(self, result, 0, 100)
        self.assertEqual(event_spans(result, 4), [])
        self.assertEqual(spans(result, "ambiguous"), [(20, 80)])
        self.assertTrue(any(item.get("event_index") == 4 for item in result["diagnostics"]))

    def test_time_containment_only_does_not_take_an_exclusive_slice(self):
        calls = [child(2, 10, 90, "metadata_builder"),
                 child(3, 20, 60, "cpu_wait", [2, 1], verified=False)]
        result = partition_host_lane(parent(), calls, "state_attn")
        check_closed(self, result, 0, 100)
        self.assertEqual(event_spans(result, 2), [(10, 90)])
        self.assertEqual(event_spans(result, 3), [])
        self.assertIsNone(result["wait_exclusive_ns"])
        self.assertIn(calls[1], result["source_children"])

    def test_time_only_cross_parent_candidate_cannot_steal_a_verified_owner(self):
        calls = [child(2, 20, 80, "metadata_builder"),
                 child(3, -10, 110, "cpu_wait", verified=False)]
        result = partition_host_lane(parent(), calls, "state_attn")
        check_closed(self, result, 0, 100)
        self.assertEqual(event_spans(result, 2), [(20, 80)])
        self.assertEqual(event_spans(result, 3), [])
        self.assertEqual(spans(result, "ambiguous"), [(0, 20), (80, 100)])
        self.assertIsNone(result["wait_exclusive_ns"])

    def test_all_intermediate_ancestor_intervals_must_contain_their_recorded_child(self):
        calls = [child(2, 10, 50, "draft_graph"),
                 child(3, 20, 70, "metadata_builder", [2, 1]),
                 child(4, 30, 40, "request_metadata", [3, 2, 1])]
        # Call 2 contains the tiny leaf 4 but does not contain its recorded
        # intermediate parent 3. The full saved call chain is contradictory.
        result = partition_host_lane(parent(), calls, "state_attn")
        self.assertEqual(event_spans(result, 4), [])
        self.assertEqual(event_spans(result, 3), [])
        self.assertEqual(event_spans(result, 2), [(10, 50)])
        self.assertTrue({3, 4}.issubset({item.get("event_index") for item in result["diagnostics"]}))

    def test_transparent_wrapper_and_generic_calls_are_folded_without_losing_details(self):
        calls = [child(2, 10, 90, "profiler_wrapper"),
                 child(3, 20, 80, "metadata_builder", [2, 1]),
                 child(4, 30, 40, "host_copy", [3, 2, 1]),
                 child(5, 50, 60, "native_workspace", [3, 2, 1])]
        result = partition_host_lane(parent(), calls, "state_attn")
        check_closed(self, result, 0, 100)
        self.assertEqual(event_spans(result, 3), [(20, 80)])
        self.assertFalse(any(piece["kind"] in ("profiler_wrapper", "host_copy", "native_workspace")
                             for piece in result["pieces"]))
        self.assertEqual(result["source_children"], calls)

    def test_cross_parent_call_is_unknown_in_overlap_and_retains_original_interval(self):
        calls = [child(2, 0, 30, "slot_mappings")]
        result = partition_host_lane(parent(10, 90), calls, "runner_attn")
        check_closed(self, result, 10, 90)
        self.assertEqual(event_spans(result, 2), [])
        unknown = [piece for piece in result["pieces"] if piece["kind"] == "ambiguous"]
        self.assertEqual([(int(piece["start_ns"]), int(piece["end_ns"])) for piece in unknown], [(10, 30)])
        diagnostic = next(item for item in result["diagnostics"] if item.get("event_index") == 2)
        self.assertEqual(int(diagnostic["start_ns"]), 0)
        self.assertEqual(int(diagnostic["end_ns"]), 30)
        self.assertEqual(result["source_children"], calls)

    def test_fully_external_call_does_not_borrow_time_from_the_parent(self):
        result = partition_host_lane(parent(), [child(2, 150, 180, "slot_mappings")], "runner_attn")
        check_closed(self, result, 0, 100)
        self.assertEqual(spans(result, "remainder"), [(0, 100)])
        self.assertEqual(event_spans(result, 2), [])

    def test_negative_nanosecond_coordinates_remain_exact(self):
        result = partition_host_lane(parent(-100, -10), [child(2, -90, -20)], "state_attn")
        check_closed(self, result, -100, -10)
        self.assertEqual(event_spans(result, 2), [(-90, -20)])
        self.assertEqual(spans(result, "remainder"), [(-100, -90), (-20, -10)])

    def test_large_epoch_does_not_round_away_single_nanosecond_pieces(self):
        base = 2**60+13
        result = partition_host_lane(parent(base, base+101), [child(2, base+1, base+100)], "state_attn")
        check_closed(self, result, base, base+101)
        self.assertEqual(spans(result, "remainder"), [(base, base+1), (base+100, base+101)])
        self.assertEqual(event_spans(result, 2), [(base+1, base+100)])

    def test_same_named_adjacent_calls_keep_their_original_event_identities(self):
        calls = [child(2, 10, 20, name="build"), child(3, 20, 30, name="build")]
        result = partition_host_lane(parent(), calls, "state_attn")
        check_closed(self, result, 0, 100)
        self.assertEqual(event_spans(result, 2), [(10, 20)])
        self.assertEqual(event_spans(result, 3), [(20, 30)])

    def test_gap_inside_parent_is_named_remainder_not_zero_cost_or_idle(self):
        result = partition_host_lane(parent(), [child(2, 10, 20), child(3, 40, 50)], "state_attn")
        self.assertIn((20, 40), spans(result, "remainder"))
        self.assertFalse(any("idle" in piece["name"].lower() for piece in result["pieces"]))

    def test_empty_children_leave_the_full_parent_wall_unpartitioned(self):
        result = partition_host_lane(parent(), [], "runner_inputs")
        check_closed(self, result, 0, 100)
        self.assertEqual(spans(result, "remainder"), [(0, 100)])
        self.assertIsNone(result["wait_exclusive_ns"])

    def test_input_order_cannot_choose_the_owner_of_an_ambiguous_overlap(self):
        calls = [child(2, 10, 70), child(3, 40, 90, "slot_mappings")]
        first = partition_host_lane(parent(), calls, "runner_attn")
        second = partition_host_lane(parent(), list(reversed(calls)), "runner_attn")
        selected = lambda result: [(int(piece["start_ns"]), int(piece["end_ns"]), piece["kind"], piece.get("event_index"))
                                   for piece in result["pieces"]]
        self.assertEqual(selected(first), selected(second))

    def test_piece_metadata_keeps_selection_reason_and_parent_ancestry(self):
        result = partition_host_lane(parent(), [child(2, 10, 90), child(3, 20, 60, "request_metadata", [2, 1])],
                                     "state_attn")
        leaf, = [piece for piece in result["pieces"] if piece.get("event_index") == 3]
        self.assertEqual(leaf["ancestor_event_indexes"], [2, 1])
        self.assertTrue(leaf["selection_reason"])
        self.assertTrue(leaf["relation"])
        self.assertAlmostEqual(leaf["duration_ms"], 40/1e6)
        self.assertIsInstance(json.dumps(result, allow_nan=False), str)


if __name__ == "__main__":
    unittest.main()
