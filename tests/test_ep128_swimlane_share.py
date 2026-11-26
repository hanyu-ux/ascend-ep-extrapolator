"""PPT sharing selects complete representatives without original capture IO."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import ep128_swimlane_share as share
import ep128_swimlane_trace as host_analysis
import profile_ep128 as core
import test_ep128_swimlane_trace as fixture


TOOL = Path(__file__).resolve().parents[1]/"tools/u_swimlane_probe.py"
SPEC = importlib.util.spec_from_file_location("swimlane_share_probe_tests", TOOL)
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)
EPOCH = 1790000000000000003


def source_event(number, name, start, end, **extra):
    return {"event_index": number, "name": name, "pid": "10", "tid": "7", "cat": "cpu_op",
            "start_ns": EPOCH+start, "end_ns": EPOCH+end, "duration_ns": end-start,
            "python_id": None, "python_parent_id": None, **extra}


def case_fixture(name="bs8-s15"):
    host, _ = fixture.fixture()
    host.extend([fixture.host(40, "dp_dispatch_patch", 170, 195, 40, 3),
                 fixture.host(41, "dp_dispatch", 171, 194, 41, 40),
                 fixture.host(42, "dp_sync", 173, 192, 42, 41),
                 fixture.host(43, "collective_wrapper", 175, 190, 43, 42),
                 fixture.host(44, "all_reduce", 176, 189, 44, 43),
                 fixture.host(45, "cpu_wait", 177, 188, 45, 44),
                 fixture.host(46, "blocktable_gather", 232, 234, 46, 34),
                 fixture.host(47, "slot_mappings", 234, 238, 47, 34),
                 fixture.host(60, "build_attn_metadata", 250, 370, 60, 31),
                 fixture.host(61, "metadata_builder", 260, 350, 61, 60),
                 fixture.host(10000, "blocktable_gather", 2000, 2008, 10000, 9999)])
    for event in host:
        event.update(start_ns=EPOCH+event["start_ns"], end_ns=EPOCH+event["end_ns"], event_type="python")
    row = {"round_id": "hc-1", "start_ns": EPOCH+100, "end_ns": EPOCH+400, "origin_ns": EPOCH+100,
           "duration_ns": 300, "u_ns": 100, "stage_status": "MATCHED",
           "regions": {"main_hc": {"start_ns": EPOCH+100, "end_ns": EPOCH+250, "duration_ns": 150},
                       "draft_hc": {"start_ns": EPOCH+300, "end_ns": EPOCH+350, "duration_ns": 50}},
           "u_windows": []}
    number = 0
    for key, left, right, count in (("main_to_draft", 250, 300, 8), ("draft_to_nextmain", 350, 400, 7)):
        stages = []
        for position in range(count):
            number += 1
            a, b = EPOCH+left+(right-left)*position//count, EPOCH+left+(right-left)*(position+1)//count
            stages.append({"round_id": "hc-1", "stage_key": "stage"+str(number), "stage_number": number,
                           "status": "MATCHED", "start_ns": a, "end_ns": b, "duration_ns": b-a,
                           "start_anchor": {"row_index": 101, "name": "slot", "timestamp_ns": a},
                           "end_anchor": {"row_index": 102, "name": "metadata", "timestamp_ns": b}})
        row["u_windows"].append({"round_id": "hc-1", "window_key": key, "start_ns": EPOCH+left,
                                 "end_ns": EPOCH+right, "stages": stages})
    original = deepcopy(row)
    original["round_id"] = "hc-other"
    original["start_ns"] = EPOCH+1000
    original["end_ns"] = EPOCH+1300
    original["origin_ns"] = EPOCH+1000
    original["host_pairing"] = {"status": "MATCHED", "functions": {"runner_attn": None}}
    # The old v2 record has no independent runner-attention lane. Rebuilding
    # uses only the full saved Host events, never any original trace path.
    row["host_pairing"] = {"status": "MATCHED", "functions": {"outer_propose": 3, "next_execute": 4,
                                                              "runner_inputs": 5, "state_attn": 6}}
    row["host_children"] = {"old_duplicate": [{"event_index": 10000}]}
    bindings = [{"round_id": "hc-other", "anchor_kind": "u_stage", "status": "EXACT_SOURCE_ROW", "device_row_indexes": [999]},
                {"round_id": "hc-1", "anchor_kind": "u_stage", "status": "EXACT_SOURCE_ROW", "device_row_indexes": [101]},
                {"round_id": "hc-other", "anchor_kind": "u_stage", "status": "UNRESOLVED_BOUNDARY", "device_row_indexes": []},
                {"round_id": "hc-1", "anchor_kind": "main_hc", "status": "AMBIGUOUS", "device_row_indexes": [102]}]
    raw = [deepcopy(event) for event in host]
    raw += [source_event(200000, "aten::slot", 233, 237, python_parent_id="34"),
            source_event(200001, "Node@slot", 234, 236), source_event(200002, "Enqueue@slot", 232, 233),
            source_event(200003, "Dequeue@slot", 233, 234), source_event(200004, "npu::slot", 233, 236),
            source_event(205000, "native-parent", 230, 239), source_event(300000, "unrelated", 2000, 3000)]
    linked = {"row_index": 101, "status": "LINKED", "anchor_binding_indexes": [1],
              "cpu_event_index": 200000, "node_event_index": 200001, "enqueue_event_index": 200002,
              "dequeue_event_index": 200003, "cpu_leaf_event_index": 200004,
              "source_event_indexes": [200000, 200001, 200002, 200003, 200004, 205000, 9, 4],
              "missing_source_event_indexes": [],
              "cpu_ancestors": [{"id": 9, "start_ns": EPOCH+232, "end_ns": EPOCH+238},
                                {"id": 205000, "start_ns": EPOCH+230, "end_ns": EPOCH+239}],
              "host_associations": [{"host_event_index": 9, "source_event_index": 200000,
                  "line_style": "solid", "runtime_submission_confirmed": True,
                  "relation": "EXPLICIT_PYTHON_PARENT_ID", "explicit_parent_event_indexes": [200000, 9]}]}
    unknown = {"row_index": 102, "status": "GRAPH_ORIGIN_MISSING", "anchor_binding_indexes": [3],
               "source_event_indexes": [], "missing_source_event_indexes": [], "cpu_ancestors": [], "host_associations": []}
    old_link = {"row_index": 999, "status": "API_ONLY", "anchor_binding_indexes": [0], "source_event_indexes": [300000]}
    edges = [{"round_id": "hc-1", "row_index": 101, "anchor_binding_index": 1,
              "host_event_index": 9, "source_event_index": 200000, "source_status": "LINKED",
              "line_style": "solid", "runtime_submission_confirmed": True},
             {"round_id": "hc-1", "row_index": 102, "anchor_binding_index": 3,
              "host_event_index": 6, "source_event_index": 200001, "source_status": "GRAPH_ORIGIN_MISSING",
              "line_style": "dashed", "runtime_submission_confirmed": False}]
    return {"case": name, "rank": "dp4", "bs": 8, "S": 15, "status": "READY", "detail_status": "READY",
            "error": None, "suggested_round_id": "hc-1", "source_link_round_ids": ["hc-1", "hc-other"],
            "trace": {"status": "ALIGNED_X_EVENTS", "sha256": name+"-sha", "clock_offset_ns": "0"},
            "trace_path": "must-not-be-opened/trace.json", "kernel_path": "must-not-be-opened/kernel.csv",
            "rounds": [original, row], "host_events": host,
            "device_anchors": [{"row_index": number} for number in (101, 102, 999)],
            "anchor_links": {"status": "PARTIAL", "source_status": "PARTIAL", "anchor_bindings": bindings,
                "links": [old_link, linked, unknown], "drawable_edges": edges,
                "device_tasks": [{"row_index": number} for number in (101, 102, 999)], "source_events": raw,
                "anchor_status_counts": {"EXACT_SOURCE_ROW": 2, "UNRESOLVED_BOUNDARY": 1, "AMBIGUOUS": 1},
                "source_task_status_counts": {"LINKED": 1, "GRAPH_ORIGIN_MISSING": 1, "API_ONLY": 1},
                "missing_host_event_indexes": [123456]}}


def payload(cases=None, schema="u_swimlane_v2"):
    return probe.wire({"schema": schema, "rank": "dp4", "profile_version": "fixture",
                       "cases": cases if cases is not None else [case_fixture()]})


def build(data, **kwargs):
    return share.build_ppt_evidence(data, core, host_analysis, probe.attach_host,
        {"sha256": "a"*64, "bytes": 302304053, "schema": data["schema"]}, **kwargs)


class RepresentativeShareTests(unittest.TestCase):
    def test_v2_restores_runner_attn_from_saved_events_and_does_not_mutate_or_read_inputs(self):
        data = payload()
        before = deepcopy(data)
        with patch("builtins.open", side_effect=AssertionError("No source file reads")), \
                patch.object(Path, "open", side_effect=AssertionError("No trace/CSV reads")):
            result = build(data)
        self.assertEqual(data, before)
        case, = result["cases"]
        self.assertEqual(case["representative_status"], "READY", case.get("share_error"))
        row, = case["rounds"]
        self.assertEqual(row["host_pairing"]["functions"]["runner_attn"], 9)
        self.assertEqual(row["host_lanes"]["runner_attn"]["duration_ns"], 6)
        self.assertEqual([r["event_index"] for r in row["host_pairing"]["child_layer"]["by_role"]["runner_attn"]], [46, 47])
        self.assertNotIn("host_children", row)
        self.assertEqual(case["omitted_round_count"], 1)
        self.assertTrue(result["all_cases_retained"])
        self.assertFalse(result["intervals_clipped"])
        self.assertFalse(result["byte_budget_truncation"])
        self.assertEqual(result["source_json_sha256"], "a"*64)
        self.assertEqual(result["source_json_bytes"], 302304053)

    def test_full_negative_and_cross_boundary_ranges_and_all_15_device_stages_survive(self):
        source = payload()
        case = build(source)["cases"][0]
        row = case["rounds"][0]
        self.assertEqual(int(row["host_lanes"]["outer_propose"]["relative_start_ns"]), -50)
        self.assertEqual(int(row["host_lanes"]["outer_propose"]["start_ns"]), EPOCH+50)
        next_call = next(e for e in case["host_events"] if e["event_index"] == 4)
        self.assertEqual(int(next_call["end_ns"]), EPOCH+450)
        self.assertGreater(int(next_call["end_ns"]), int(row["end_ns"]))
        original = source["cases"][0]["rounds"][1]
        self.assertEqual(row["regions"], original["regions"])
        self.assertEqual(row["u_windows"], original["u_windows"])
        self.assertEqual([s["stage_number"] for w in row["u_windows"] for s in w["stages"]], list(range(1, 16)))

    def test_selected_event_reference_closure_removes_unrelated_whole_history(self):
        case = build(payload())["cases"][0]
        ids = {e["event_index"] for e in case["host_events"]}
        source_ids = {e["event_index"] for e in case["anchor_links"]["source_events"]}
        self.assertTrue({1, 2, 3, 4, 5, 6, 7, 8, 9, 40, 41, 42, 43, 44, 45, 46, 47, 60, 61}.issubset(ids))
        self.assertNotIn(10000, ids)
        self.assertNotIn(300000, source_ids)
        self.assertTrue({200000, 200001, 200002, 200003, 200004, 205000, 9, 4}.issubset(source_ids))
        self.assertEqual(case["missing_event_indexes"], [])
        self.assertTrue(share._round_refs(case["rounds"][0]).issubset(ids | source_ids))

    def test_host_only_ancestor_found_in_source_table_is_retained_without_confidence_upgrade(self):
        original = case_fixture()
        child = fixture.host(70, "metadata_builder", 270, 330, 70, 9998)
        child.update(start_ns=EPOCH+270, end_ns=EPOCH+330, event_type="python")
        original["host_events"].append(child)
        original["anchor_links"]["source_events"].extend([
            deepcopy(child), source_event(500000, "uncatalogued-wrapper", 260, 350,
                cat="python_function", python_id="9998", python_parent_id="31")])
        case = build(payload([original]))["cases"][0]
        raw = {e["event_index"] for e in case["anchor_links"]["source_events"]}
        self.assertIn(500000, raw)
        self.assertEqual(case["missing_event_indexes"], [])
        ref = next(r for r in case["rounds"][0]["host_pairing"]["child_layer"]["by_role"]["state_attn"]
                   if r["event_index"] == 70)
        self.assertEqual(ref["relation"], "time_containment_only")
        self.assertFalse(ref["parent_chain_verified"])

    def test_nonzero_binding_indexes_are_remapped_in_links_and_edges(self):
        links = build(payload())["cases"][0]["anchor_links"]
        self.assertEqual(links["source_binding_indexes"], [1, 3])
        self.assertEqual([b["round_id"] for b in links["anchor_bindings"]], ["hc-1", "hc-1"])
        self.assertEqual([l["anchor_binding_indexes"] for l in links["links"]], [[0], [1]])
        self.assertEqual([e["anchor_binding_index"] for e in links["drawable_edges"]], [0, 1])
        self.assertEqual({t["row_index"] for t in links["device_tasks"]}, {101, 102})

    def test_missing_and_ambiguous_evidence_remains_partial_and_cannot_be_promoted(self):
        source = payload()
        case = source["cases"][0]
        case["anchor_links"]["links"][2]["missing_source_event_indexes"] = [777777]
        result = build(source)["cases"][0]
        links = result["anchor_links"]
        self.assertEqual(links["status"], "PARTIAL")
        self.assertEqual(links["source_status"], "PARTIAL")
        self.assertEqual(links["anchor_bindings"][1]["status"], "AMBIGUOUS")
        self.assertEqual(links["links"][1]["status"], "GRAPH_ORIGIN_MISSING")
        self.assertEqual(links["drawable_edges"][1]["line_style"], "dashed")
        self.assertFalse(links["drawable_edges"][1]["runtime_submission_confirmed"])
        self.assertIn(777777, result["missing_event_indexes"])
        self.assertEqual(links["links"][1]["missing_source_event_indexes"], [777777])

    def test_clock_mismatch_keeps_device_round_and_diagnostics_but_no_host_or_edges(self):
        old = case_fixture()
        old.update(status="ERROR", error="ValueError: Trace alignment failed: CLOCK_MISMATCH")
        old["trace"].update(status="CLOCK_MISMATCH", max_clock_residual_ns=17000, anchors=[{"csv_start_ns": str(EPOCH)}])
        data = payload([old])
        case = build(data)["cases"][0]
        self.assertEqual(case["status"], "ERROR")
        self.assertEqual(case["representative_status"], "DEVICE_ONLY")
        self.assertEqual(case["trace"], data["cases"][0]["trace"])
        self.assertEqual(case["error"], old["error"])
        self.assertEqual(case["host_events"], [])
        self.assertEqual(case["rounds"][0]["host_lanes"], {})
        self.assertEqual(case["anchor_links"]["drawable_edges"], [])
        self.assertEqual(case["rounds"][0]["u_windows"], data["cases"][0]["rounds"][1]["u_windows"])

    def test_same_event_id_in_different_cases_is_not_merged_or_borrowed(self):
        a, b = case_fixture("bs8-s15"), case_fixture("bs16-s15")
        next(e for e in b["host_events"] if e["event_index"] == 9).update(end_ns=EPOCH+239, duration_ns=7)
        # Its raw source identity must describe the same trace event.
        next(e for e in b["anchor_links"]["source_events"] if e["event_index"] == 9).update(end_ns=EPOCH+239, duration_ns=7)
        result = build(payload([a, b]))
        self.assertEqual([c["rounds"][0]["host_lanes"]["runner_attn"]["duration_ns"] for c in result["cases"]], [6, 7])
        self.assertEqual([c["trace"]["sha256"] for c in result["cases"]], ["bs8-s15-sha", "bs16-s15-sha"])

    def test_deduplicated_ref_fields_reconstruct_exactly_from_event_table(self):
        data = payload()
        host, _ = share._events(data["cases"][0]["host_events"])
        raw_round = share._times(data["cases"][0]["rounds"][1])
        pair = host_analysis.pair_rounds(core, [raw_round], host)
        original = probe.attach_host([raw_round], pair, host)[0]["host_pairing"]["child_layer"]
        case = build(data)["cases"][0]
        lookup = {e["event_index"]: e for e in case["host_events"]}
        saved = case["rounds"][0]["host_pairing"]["child_layer"]
        for role, refs in original["by_role"].items():
            for before, after in zip(refs, saved["by_role"][role]):
                reconstructed = deepcopy(after)
                for key in share._DUPLICATE_REF_FIELDS:
                    if key not in reconstructed and key in before:
                        reconstructed[key] = lookup[after["event_index"]][key]
                self.assertEqual(reconstructed, probe.wire(before))
        self.assertGreater(case["deduplicated_child_fields"], 0)

    def test_missing_runner_attn_does_not_borrow_unrelated_evidence_or_claim_ready(self):
        data = payload()
        data["cases"][0]["host_events"] = [e for e in data["cases"][0]["host_events"] if e["event_index"] != 9]
        case = build(data)["cases"][0]
        self.assertEqual(case["status"], "READY")  # Original v2 status stays identifiable.
        self.assertEqual(case["representative_status"], "PARTIAL")
        self.assertNotIn("runner_attn", case["rounds"][0]["host_lanes"])
        self.assertIn("runner_attn", case["rounds"][0]["host_pairing"]["missing"])

    def test_different_rank_is_an_isolated_case_error_and_all_12_cases_survive(self):
        cases = [case_fixture("case-"+str(i)) for i in range(12)]
        cases[3]["rank"] = "dp3"
        progress = []
        result = build(payload(cases), progress=progress.append)
        self.assertEqual(len(result["cases"]), 12)
        self.assertEqual(result["cases"][3]["representative_status"], "ERROR")
        self.assertIn("rank", result["cases"][3]["share_error"])
        self.assertEqual(result["cases"][4]["representative_status"], "READY")
        self.assertEqual(len(progress), 12)

    def test_source_linked_representative_priority_is_explicit_and_does_not_recompute_sources(self):
        data = payload()
        data["cases"][0]["suggested_round_id"] = "hc-other"
        data["cases"][0]["source_link_round_ids"] = ["hc-1"]
        with patch.object(core, "residual_host_rows", side_effect=AssertionError("No trace scan")), \
                patch.object(core, "residual_trace_scan", side_effect=AssertionError("No clock scan")):
            case = build(data)["cases"][0]
        self.assertEqual(case["selected_round_id"], "hc-1")
        self.assertEqual(case["selection_reason"], "ORIGINAL_SOURCE_LINKED_ROUND_PRIORITY")

    def test_conflicting_host_source_identity_does_not_silently_confirm_a_wrong_edge(self):
        data = payload()
        event = next(e for e in data["cases"][0]["anchor_links"]["source_events"] if e["event_index"] == 9)
        event["start_ns"] = str(EPOCH+999)
        case = build(data)["cases"][0]
        self.assertEqual(case["representative_status"], "ERROR")
        self.assertIn("identity conflict", case["share_error"])

    def test_full_history_counts_and_selection_metadata_are_labeled_as_source_only(self):
        data = payload()
        data["cases"][0].update(complete_rounds=31, complete_host_rounds=31, incomplete_rounds=0,
                                ready_definition="31 original rounds", round_selection="all source rounds",
                                source_link_scope="three original source-lookup rounds")
        result = build(data)
        case = result["cases"][0]
        self.assertEqual(case["source_complete_rounds"], 31)
        self.assertEqual(case["source_complete_host_rounds"], 31)
        self.assertNotIn("complete_rounds", case)
        self.assertNotIn("round_selection", case)
        self.assertEqual(case["selected_round_count"], 1)
        self.assertEqual(case["selected_complete_rounds"], 1)
        self.assertEqual(case["source_link_round_ids_original"], ["hc-1", "hc-other"])
        self.assertEqual(case["source_link_round_ids"], ["hc-1"])
        self.assertEqual(result["diagram_contract"]["host_lane_order"], [
            "outer_propose", "preparation_dp_aggregate", "runner_inputs", "runner_attn", "state_attn"])


if __name__ == "__main__":
    unittest.main()
