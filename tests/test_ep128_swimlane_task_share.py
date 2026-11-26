"""Representative sharing retains stage-internal and asynchronous task proof."""

from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import patch

import ep128_swimlane_share as share
import test_ep128_swimlane_share as fixtures


EPOCH = fixtures.EPOCH


def task(key, row, start, end, name="Metadata"):
    return {"task_key":key,"row_index":row,"csv_row_index":row if row>=0 else None,
            "source_kind":"CSV" if row>=0 else "TRACE_ONLY", "name":name,
            "start_ns":str(EPOCH+start),"end_ns":str(EPOCH+end),
            "trace_event_index":None if row>=0 else -row,"device_id":"4","stream_id":"1","task_id":str(row)}


def extended_case(name="bs8-s15"):
    case=fixtures.case_fixture(name)
    sources=[fixtures.source_event(600001,"metadata-native",280,290,python_parent_id="6100"),
             fixtures.source_event(600002,"Node@metadata",281,285),
             fixtures.source_event(600003,"Enqueue@metadata",279,280),
             fixtures.source_event(600004,"Dequeue@metadata",280,286),
             fixtures.source_event(600005,"metadata-native-leaf",282,284),
             fixtures.source_event(614000,"metadata-wrapper",270,300,cat="python_function",python_id="6100",python_parent_id="31"),
             fixtures.source_event(700000,"other-round-native",1100,1110)]
    a=task("csv:301",301,355,370)
    late=task("csv:302",302,600,620,"DelayedMetadata")
    other=task("csv:399",399,1100,1120,"OtherRound")
    unbound=task("csv:400",400,5000,5010,"Unbound")
    zero=task("trace:900000",-900000,375,375,"TraceOnlyDMA")
    links=[]
    for t in (a,late):
        links.append({"task_key":t["task_key"],"row_index":t["row_index"],"status":"LINKED",
                      "cpu_event_index":600001,"node_event_index":600002,"enqueue_event_index":600003,
                      "dequeue_event_index":600004,"cpu_leaf_event_index":600005,
                      "source_event_indexes":[600001,600002,600003,600004,600005],"missing_source_event_indexes":[],
                      "cpu_ancestors":[{"id":614000,"start_ns":str(EPOCH+270),"end_ns":str(EPOCH+300)}],
                      "host_associations":[{"host_event_index":6,"source_event_index":600001,
                          "runtime_submission_confirmed":True,"line_style":"solid","relation":"EXPLICIT_PYTHON_PARENT_ID",
                          "explicit_parent_event_indexes":[600001,614000,6]}]})
    links.extend([{"task_key":"csv:399","row_index":399,"status":"API_ONLY","source_event_indexes":[700000]},
                  {"task_key":zero["task_key"],"row_index":zero["row_index"],"status":"ZERO_DURATION_UNRESOLVED"}])
    bindings=[{"round_id":"hc-1","task_key":a["task_key"],"row_index":301,"relation":"DEVICE_CYCLE_OVERLAP",
               "overlap_start_ns":str(EPOCH+355),"overlap_end_ns":str(EPOCH+370),"overlap_duration_ns":15},
              {"round_id":"hc-other","task_key":a["task_key"],"row_index":301,"relation":"HOST_SCOPE_TIME_OVERLAP"},
              {"round_id":"hc-1","task_key":late["task_key"],"row_index":302,"relation":"EXPLICIT_SOURCE_OWNER",
               "host_event_index":6,"source_event_index":600001},
              {"round_id":"hc-other","task_key":other["task_key"],"row_index":399,"relation":"DEVICE_CYCLE_OVERLAP"},
              {"round_id":"hc-1","task_key":zero["task_key"],"row_index":zero["row_index"],"relation":"DEVICE_CYCLE_OVERLAP"}]
    edges=[{"round_ids":["hc-1"],"task_key":late["task_key"],"row_index":302,"host_event_index":6,"source_event_index":600001,
            "runtime_submission_confirmed":True,"line_style":"solid","relation":"EXPLICIT_PYTHON_PARENT_ID"}]
    case["device_task_links"]={"schema":"u_swimlane_device_task_links_v1","status":"PARTIAL",
        "trace_sha256":case["trace"]["sha256"],"clock_alignment_status":"ALIGNED_X_EVENTS",
        "device_tasks":[a,late,other,unbound,zero],"links":links,"source_events":sources,
        "task_round_bindings":bindings,"task_edges":edges,
        "host_family_scopes":[{"round_id":"hc-1","role":"state_attn","workload_round":"n+1",
            "host_event_index":6,"name":"metadata","start_ns":str(EPOCH+260),"end_ns":str(EPOCH+400)},
            {"round_id":"hc-other","role":"state_attn","workload_round":"n+1","host_event_index":99}],
        "trace_device_observations":[{"trace_event_index":800000,"candidate_task_keys":["csv:301","csv:399"]},
            {"trace_event_index":900000,"represented_by_task_key":"trace:900000","duration_ns":0},
            {"trace_event_index":800002,"represented_by_task_key":"csv:399"}],
        "ambiguous_trace_device_observations":[{"trace_event_index":800000,"candidate_task_keys":["csv:301","csv:399"]},
                                               {"trace_event_index":800001,"candidate_task_keys":["csv:399"]}]}
    return case


def build(case):
    return fixtures.build(fixtures.payload([case]))["cases"][0]


class TaskShareTests(unittest.TestCase):
    def test_metadata_not_a_boundary_and_late_task_survive_without_capture_IO(self):
        source=extended_case();before=deepcopy(source)
        with patch("builtins.open",side_effect=AssertionError("No capture IO")), \
             patch.object(Path,"open",side_effect=AssertionError("No trace IO")):
            case=build(source)
        self.assertEqual(source,before)
        self.assertEqual(case["representative_status"],"READY",case.get("share_error"))
        saved=case["device_task_links"]
        self.assertEqual({t["task_key"] for t in saved["device_tasks"]},{"csv:301","csv:302","trace:900000"})
        self.assertEqual({l["task_key"] for l in saved["links"]},{"csv:301","csv:302","trace:900000"})
        late=next(t for t in saved["device_tasks"] if t["task_key"]=="csv:302")
        self.assertGreater(int(late["start_ns"]),int(case["rounds"][0]["end_ns"]))
        self.assertEqual(int(late["end_ns"]),EPOCH+620)
        self.assertEqual(saved["task_edges"][0]["line_style"],"solid")
        self.assertEqual(saved["selection_reference_status"],"COMPLETE")

    def test_trace_only_zero_duration_identity_is_not_a_CSV_row_or_lengthened(self):
        saved=build(extended_case())["device_task_links"]
        zero=next(t for t in saved["device_tasks"] if t["source_kind"]=="TRACE_ONLY")
        self.assertEqual(zero["task_key"],"trace:900000")
        self.assertEqual(zero["row_index"],-900000)
        self.assertIsNone(zero["csv_row_index"])
        self.assertEqual(zero["start_ns"],zero["end_ns"])
        self.assertEqual(next(l for l in saved["links"] if l["task_key"]==zero["task_key"])["status"],"ZERO_DURATION_UNRESOLVED")

    def test_every_source_API_queue_native_and_ancestor_reference_survives(self):
        case=build(extended_case());saved=case["device_task_links"]
        ids={e["event_index"] for e in saved["source_events"]}
        self.assertTrue({600001,600002,600003,600004,600005,614000,6}.issubset(ids))
        self.assertNotIn(700000,ids)
        self.assertTrue(share._refs(saved["links"]).issubset(ids|{e["event_index"] for e in case["host_events"]}))
        self.assertEqual(case["missing_event_indexes"],[])
        self.assertEqual(saved["links"][0]["host_associations"][0]["relation"],"EXPLICIT_PYTHON_PARENT_ID")

    def test_raw_flow_endpoints_survive_with_exact_identity_and_shared_flow_key(self):
        source=extended_case()
        block=source["device_task_links"]
        flow_events=[fixtures.source_event(800101,"submit-flow",280,280,ph="s",flow_key="flow-42"),
                     fixtures.source_event(800102,"submit-flow",355,355,ph="f",flow_key="flow-42"),
                     fixtures.source_event(800103,"submit-flow",600,600,ph="f",flow_key="flow-42")]
        block["source_events"].extend(flow_events)
        # Exercise the named flow references directly: exporters also include
        # them in source_event_indexes, but retaining either reference is safe.
        block["links"][0]["flow_event_indexes"]=[800101,800102,800103]
        saved=build(source)["device_task_links"]
        by_id={event["event_index"]:event for event in saved["source_events"]}
        self.assertTrue({800101,800102,800103}.issubset(by_id))
        for original in flow_events:
            event=by_id[original["event_index"]]
            self.assertEqual(event["ph"],original["ph"])
            self.assertEqual(event["flow_key"],"flow-42")
            self.assertEqual(event["start_ns"],str(original["start_ns"]))
            self.assertEqual(event["end_ns"],str(original["end_ns"]))
        self.assertEqual(saved["selection_reference_status"],"COMPLETE")

    def test_missing_raw_flow_endpoint_downgrades_that_task_source_arrow(self):
        source=extended_case()
        source["device_task_links"]["links"][0]["flow_event_indexes"]=[800104]
        saved=build(source)["device_task_links"]
        self.assertIn(800104,saved["missing_event_indexes"])
        self.assertEqual(saved["selection_reference_status"],"PARTIAL")
        self.assertFalse(saved["links"][0]["host_associations"][0]["runtime_submission_confirmed"])

    def test_membership_requires_explicit_same_round_binding_not_time_overlap(self):
        source=extended_case()
        source["device_task_links"]["device_tasks"][-2].update(start_ns=str(EPOCH+360),end_ns=str(EPOCH+365))
        case=build(source);saved=case["device_task_links"]
        self.assertNotIn("csv:400",{t["task_key"] for t in saved["device_tasks"]})
        self.assertTrue(all(b["round_id"]=="hc-1" for b in saved["task_round_bindings"]))
        self.assertEqual(saved["omitted_catalog_task_count"],2)
        self.assertEqual(saved["omitted_unbound_catalog_task_count"],1)

    def test_shared_task_is_kept_once_and_other_round_membership_is_not_copied(self):
        saved=build(extended_case())["device_task_links"]
        self.assertEqual(sum(t["task_key"]=="csv:301" for t in saved["device_tasks"]),1)
        self.assertEqual(sum(b["task_key"]=="csv:301" for b in saved["task_round_bindings"]),1)
        self.assertNotIn("hc-other",str(saved["task_round_bindings"]))

    def test_task_edges_use_their_own_explicit_round_context(self):
        source=extended_case()
        source["device_task_links"]["task_edges"].append({"round_ids":["hc-other"],
            "task_key":"csv:301","host_event_index":99,"source_event_index":700000})
        saved=build(source)["device_task_links"]
        self.assertEqual(len(saved["task_edges"]),1)
        self.assertEqual(saved["task_edges"][0]["round_ids"],["hc-1"])
        self.assertNotIn(700000,{event["event_index"] for event in saved["source_events"]})

    def test_host_scopes_and_trace_observations_are_selected_without_ambiguity_erasure(self):
        saved=build(extended_case())["device_task_links"]
        self.assertEqual(len(saved["host_family_scopes"]),1)
        self.assertEqual(saved["host_family_scopes"][0]["host_event_index"],6)
        self.assertEqual({event["trace_event_index"] for event in saved["trace_device_observations"]},{800000,900000})
        self.assertEqual(saved["trace_device_observations"][0]["candidate_task_keys"],["csv:301","csv:399"])

    def test_ambiguous_trace_observation_keeps_all_candidates_with_audit_context(self):
        saved=build(extended_case())["device_task_links"]
        observation,=saved["ambiguous_trace_device_observations"]
        self.assertEqual(observation["candidate_task_keys"],["csv:301","csv:399"])
        context,=saved["ambiguity_context_device_tasks"]
        self.assertEqual(context["task_key"],"csv:399")
        self.assertNotIn("csv:399",{t["task_key"] for t in saved["device_tasks"]})

    def test_missing_source_proof_stays_explicit_and_cannot_keep_confirmed_edges(self):
        source=extended_case()
        block=source["device_task_links"]
        block["source_events"]=[e for e in block["source_events"] if e["event_index"]!=600005]
        case=build(source);saved=case["device_task_links"]
        self.assertEqual(saved["selection_reference_status"],"PARTIAL")
        self.assertIn(600005,saved["missing_event_indexes"])
        self.assertIn(600005,case["missing_event_indexes"])
        self.assertFalse(saved["task_edges"][0]["runtime_submission_confirmed"])
        self.assertEqual(saved["task_edges"][0]["line_style"],"dashed")
        self.assertEqual(saved["links"][0]["status"],"LINKED")
        self.assertFalse(saved["links"][0]["host_associations"][0]["runtime_submission_confirmed"])
        binding=next(b for b in saved["task_round_bindings"] if b["task_key"]=="csv:302")
        self.assertEqual(binding["relation"],"UNVERIFIED_SOURCE_OWNER")
        self.assertEqual(binding["original_relation"],"EXPLICIT_SOURCE_OWNER")
        self.assertFalse(binding["runtime_submission_confirmed"])

    def test_missing_binding_source_reference_cannot_keep_confirmed_ownership(self):
        source=extended_case()
        binding=source["device_task_links"]["task_round_bindings"][2]
        binding.update(source_event_index=799998,runtime_submission_confirmed=True)
        saved=build(source)["device_task_links"]
        binding=next(b for b in saved["task_round_bindings"] if b["task_key"]=="csv:302")
        self.assertEqual(binding["relation"],"UNVERIFIED_SOURCE_OWNER")
        self.assertEqual(binding["original_relation"],"EXPLICIT_SOURCE_OWNER")
        self.assertTrue(binding["original_runtime_submission_confirmed"])
        self.assertFalse(binding["runtime_submission_confirmed"])
        self.assertIn(799998,binding["missing_source_event_indexes"])
        self.assertFalse(saved["task_edges"][0]["runtime_submission_confirmed"])
        self.assertFalse(next(link for link in saved["links"] if link["task_key"]=="csv:302")
                         ["host_associations"][0]["runtime_submission_confirmed"])

    def test_clock_failure_retains_only_pure_device_cycle_records(self):
        source=extended_case()
        source.update(status="ERROR",error="CLOCK_MISMATCH")
        source["trace"]["status"]="CLOCK_MISMATCH"
        source["device_task_links"]["task_round_bindings"][0]["host_event_index"]=6
        saved=build(source)["device_task_links"]
        self.assertEqual(saved["status"],"CLOCK_UNALIGNED")
        self.assertEqual({t["task_key"] for t in saved["device_tasks"]},{"csv:301","trace:900000"})
        self.assertEqual(saved["links"],[])
        self.assertEqual(saved["source_events"],[])
        self.assertEqual(saved["task_edges"],[])
        self.assertEqual(saved["host_family_scopes"],[])
        self.assertEqual(saved["trace_device_observations"],[])
        self.assertFalse(share._refs(saved["task_round_bindings"]))

    def test_missing_indirect_source_ancestor_downgrades_its_task_edges(self):
        source=extended_case()
        source["device_task_links"]["source_events"][0]["parent_event_index"]=799999
        saved=build(source)["device_task_links"]
        self.assertEqual(saved["selection_reference_status"],"PARTIAL")
        self.assertIn(799999,saved["missing_event_indexes"])
        self.assertFalse(saved["task_edges"][0]["runtime_submission_confirmed"])

    def test_task_block_clock_failure_cannot_leak_sources_when_case_is_aligned(self):
        source=extended_case()
        source["device_task_links"]["clock_alignment_status"]="CLOCK_MISMATCH"
        case=build(source);saved=case["device_task_links"]
        self.assertEqual(saved["status"],"CLOCK_UNALIGNED")
        self.assertEqual(saved["source_events"],[])
        self.assertEqual(saved["host_family_scopes"],[])
        self.assertEqual(saved["links"],[])

    def test_missing_catalog_identity_is_explicit_and_never_makes_a_solid_arrow(self):
        source=extended_case()
        source["device_task_links"]["device_tasks"]=[t for t in source["device_task_links"]["device_tasks"]
                                                       if t["task_key"]!="csv:302"]
        saved=build(source)["device_task_links"]
        self.assertEqual(saved["missing_device_task_keys"],["csv:302"])
        self.assertEqual(saved["selection_reference_status"],"PARTIAL")
        self.assertFalse(saved["task_edges"][0]["runtime_submission_confirmed"])
        binding=next(b for b in saved["task_round_bindings"] if b["task_key"]=="csv:302")
        self.assertEqual(binding["relation"],"UNVERIFIED_SOURCE_OWNER")
        self.assertEqual(binding["original_relation"],"EXPLICIT_SOURCE_OWNER")
        self.assertFalse(binding["runtime_submission_confirmed"])

    def test_non_clock_case_error_cannot_leak_saved_Host_associations(self):
        source=extended_case();source.update(status="ERROR",error="source analysis failed")
        saved=build(source)["device_task_links"]
        self.assertEqual(saved["status"],"SOURCE_UNTRUSTED")
        self.assertEqual(saved["source_events"],[])
        self.assertEqual(saved["links"],[])

    def test_mismatching_task_trace_SHA_is_not_borrowed(self):
        source=extended_case();source["device_task_links"]["trace_sha256"]="different-case"
        case=build(source)
        self.assertEqual(case["representative_status"],"ERROR")
        self.assertIn("different trace SHA256",case["share_error"])

    def test_older_v2_v3_exports_without_task_catalog_are_unchanged(self):
        for schema in ("u_swimlane_v2","u_swimlane_v3"):
            case=fixtures.build(fixtures.payload(schema=schema))["cases"][0]
            self.assertEqual(case["representative_status"],"READY")
            self.assertNotIn("device_task_links",case)

    def test_same_event_indexes_in_separate_cases_keep_separate_identity(self):
        a,b=extended_case("a"),extended_case("b")
        next(e for e in b["device_task_links"]["source_events"] if e["event_index"]==600001)["name"]="other-case-native"
        cases=fixtures.build(fixtures.payload([a,b]))["cases"]
        names=[next(e for e in c["device_task_links"]["source_events"] if e["event_index"]==600001)["name"] for c in cases]
        self.assertEqual(names,["metadata-native","other-case-native"])


if __name__=="__main__":
    unittest.main()
