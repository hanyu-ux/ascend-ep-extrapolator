"""Anchor submission links retain real core ambiguity and exact Host evidence."""

from copy import deepcopy
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import profile_ep128 as core
import ep128_swimlane_links as links


EPOCH = 1790000000000000003
CONNECTION = 2**60 + 101
CORRELATION = 2**60 + 303


def task(number, name, start, end, **fields):
    return SimpleNamespace(row_index=number, row_indices=[number], name=name, op_type=name,
                           start_ns=EPOCH+start, end_ns=EPOCH+end, device_id="4", model_id="12",
                           stream_id="1", task_id=str(number), context_id="7", batch_id="8",
                           subtask_id="0", task_type="AI_CORE", op_state=None, **fields)


def xevent(name, start, end, pid, tid, category, args=None):
    return {"ph": "X", "name": name, "pid": pid, "tid": tid, "cat": category,
            "ts": str(Decimal(EPOCH+start)/1000), "dur": str(Decimal(end-start)/1000), "args": args or {}}


def flow(phase, category, identity, stamp, pid, tid):
    return {"ph": phase, "name": category, "cat": category, "id": identity,
            "pid": pid, "tid": tid, "ts": str(Decimal(EPOCH+stamp)/1000)}


def fixture(directory, mode="linked", explicit=True):
    target = task(1, "Kernel", 210000, 220000)
    events = [{"ph": "M", "name": "process_name", "pid": p, "args": {"name": n}}
              for p, n in ((399, "Python"), (900, "Ascend Hardware"), (408831, "CANN"))]
    events += [xevent("runner.py(1): prepare_inputs", 0, 90000, 399, 399, "python_function",
                      {"Python id": 10, "Python parent id": 0}),
               xevent("helper.py(2): launch", 1000, 85000, 399, 399, "python_function",
                      {"Python id": 20, "Python parent id": 10}),
               xevent("aten::linear", 15000, 60000, 399, 399, "cpu_op",
                      {"Python parent id": 20} if explicit else {}),
               xevent("npu::inner", 20000, 55000, 399, 399, "cpu_op"),
               xevent("Enqueue@launch", 30000, 32000, 399, 399, "enqueue",
                      {"correlation_id": CORRELATION}),
               xevent("Dequeue@launch", 39000, 50000, 399, 1171, "dequeue",
                      {"correlation_id": CORRELATION}),
               xevent("Node@launch", 40000, 45000, 408831, 1171, "cann_api",
                      {"connection_id": CONNECTION}),
               xevent(target.name, 210000, 220000, 900, 1, "kernel",
                      {"Task Type": "AI_CORE", "Physic Stream Id": "1", "Task Id": "1",
                       "Model Id": "12", "connection_id": CONNECTION}),
               flow("s", "HostToDevice", "native-flow", 42000, 408831, 1171),
               flow("f", "HostToDevice", "native-flow", 210000, 900, 1),
               flow("s", "async_task_queue", "queue-flow", 30500, 399, 399),
               flow("f", "async_task_queue", "queue-flow", 39500, 399, 1171)]
    if mode == "ambiguous":
        events.append(xevent("Node@duplicate", 40500, 45000, 408831, 1171, "cann_api",
                             {"connection_id": CONNECTION}))
    elif mode == "bad_queue":
        next(e for e in events if e.get("ph") == "s" and e.get("cat") == "async_task_queue")["tid"] = 777
    elif mode == "graph":
        events = [e for e in events if e.get("name") not in ("Node@launch", "Dequeue@launch", "Enqueue@launch")
                  and e.get("cat") != "async_task_queue"
                  and not (e.get("ph") == "s" and e.get("cat") == "HostToDevice")]
        events.append(flow("f", "HostToDevice", "native-flow", 210001, 900, 1))
    elif mode == "api_only":
        events = [e for e in events if e.get("cat") not in ("cpu_op", "enqueue", "dequeue", "async_task_queue")]
    elif mode == "cpu_only":
        events = [e for e in events if e.get("name") not in ("Node@launch", "Dequeue@launch", "Enqueue@launch")
                  and e.get("cat") not in ("HostToDevice", "async_task_queue")]
        events += [flow("s", "async_npu", "cpu-flow", 30000, 399, 399),
                   flow("f", "async_npu", "cpu-flow", 210000, 900, 1)]
    elif mode == "wrong_task_id":
        next(e for e in events if e.get("cat") == "kernel")["args"]["Task Id"] = "999"
    elif mode == "wrong_thread_parent":
        next(e for e in events if e.get("name") == "helper.py(2): launch")["tid"] = 777
    elif mode == "source_identity":
        source = next(e for e in events if e.get("name") == "aten::linear")
        source["cat"] = "cpu_op;python_function"
        source["args"]["Python id"] = 30
    trace = directory/"trace.json"
    trace.write_text(json.dumps(events), encoding="utf-8")
    csv_path = directory/"kernel.csv"
    csv_path.write_text("original source fixture", encoding="utf-8")
    capture = {"rank": "dp4", "path": str(csv_path), "sha256": "csv-sha", "tasks": [target]}
    stage = {"stage_key": "main_metadata", "stage_number": 14, "stage_order": 6, "status": "MATCHED",
             "start_ns": str(target.start_ns), "end_ns": str(target.end_ns),
             "start_anchor": {"row_index": 1, "name": target.name, "timestamp_ns": str(target.start_ns)},
             "end_anchor": {"name": "next HCPre.start", "timestamp_ns": str(target.end_ns)}}
    rounds = [{"round_id": "hc-1", "start_ns": EPOCH+100000, "end_ns": EPOCH+300000,
               "regions": {}, "u_windows": [{"round_id": "hc-1", "window_key": "draft_to_nextmain",
                    "start_ns": EPOCH+200000, "end_ns": EPOCH+250000, "stages": [stage]}]}]
    metadata = {"status": "ALIGNED_X_EVENTS", "sha256": hashlib.sha256(trace.read_bytes()).hexdigest(),
                "clock_offset_ns": "0", "process_names": {"399": "Python", "408831": "CANN", "900": "Ascend Hardware"},
                "anchors": [{"trace_pid": "900"}]}
    host = []
    for number, raw in enumerate(events, 1):
        if "python_function" in raw.get("cat", ""):
            args = core.residual_args(raw)
            start, duration = core.a_ns(raw["ts"]), core.a_ns(raw["dur"])
            host.append({"event_index": number, "name": raw["name"], "pid": str(raw["pid"]),
                         "tid": str(raw["tid"]), "cat": raw["cat"], "start_ns": start,
                         "end_ns": start+duration, "python_id": str(args["pythonid"]),
                         "python_parent_id": str(args["pythonparentid"])})
    return capture, rounds, host, trace, metadata


class AnchorLinksTests(unittest.TestCase):
    def run_fixture(self, mode="linked", explicit=True):
        temporary = tempfile.TemporaryDirectory(prefix="swimlane-links-")
        self.addCleanup(temporary.cleanup)
        directory = Path(temporary.name)
        values = fixture(directory, mode, explicit)
        result = links.export_anchor_links(core, *values[:3], values[3], values[4], 0, directory/"evidence")
        return result, values, directory

    def test_real_core_chain_finds_early_sources_without_changing_U_or_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            capture, rounds, host, trace, metadata = fixture(directory)
            original = deepcopy(rounds), deepcopy(metadata), trace.read_bytes(), Path(capture["path"]).read_bytes()
            real = core.residual_host_rows
            with patch.object(core, "residual_host_rows", wraps=real) as source:
                result = links.export_anchor_links(core, capture, rounds, host, trace, metadata, 0, directory/"proof")
            self.assertEqual(source.call_args.args[2:4], ([], []))
            self.assertEqual(rounds, original[0]); self.assertEqual(metadata, original[1])
            self.assertEqual(trace.read_bytes(), original[2]); self.assertEqual(Path(capture["path"]).read_bytes(), original[3])
            row, = result["links"]
            self.assertEqual(row["status"], "LINKED", row)
            self.assertEqual(result["source_status"], "COMPLETE")
            self.assertEqual(row["connection_id"], str(CONNECTION))
            self.assertEqual(row["correlation_id"], str(CORRELATION))
            self.assertEqual(row["queue_wait_ns"], 7000)
            self.assertEqual(row["submit_to_device_ns"], 170000)
            self.assertEqual(row["source_before_anchor_ns"], 195000)
            self.assertLess(row["cpu_start_ns"], int(rounds[0]["u_windows"][0]["start_ns"]))
            self.assertTrue(row["flow_ids"])
            self.assertTrue(all(row[k] is not None for k in ("cpu_event_index", "node_event_index", "enqueue_event_index", "dequeue_event_index")))
            self.assertTrue(all(a["runtime_submission_confirmed"] and a["line_style"] == "solid" for a in row["host_associations"]))
            parent = next(a for a in row["host_associations"] if a["host_name"].startswith("runner.py"))
            self.assertEqual(len(parent["explicit_parent_event_indexes"]), 3)
            self.assertEqual(result["device_tasks"][0]["context_id"], "7")
            self.assertEqual(result["missing_host_event_indexes"], [])
            self.assertEqual(row["missing_source_event_indexes"], [])
            self.assertTrue(all("start_ns" in a and "end_ns" in a and "start" not in a and "end" not in a
                                for a in row["cpu_ancestors"]))
            # Absolute nanoseconds intentionally exceed JS's exact integer range.
            self.assertEqual(result["device_tasks"][0]["start_ns"], EPOCH+210000)

    def test_exact_source_event_identity_is_solid_without_a_self_parent_edge(self):
        result, *_ = self.run_fixture("source_identity")
        row, = result["links"]
        association = next(a for a in row["host_associations"] if a["host_event_index"] == row["cpu_event_index"])
        self.assertEqual(association["relation"], "EXACT_SOURCE_EVENT_IDENTITY")
        self.assertTrue(association["runtime_submission_confirmed"])
        edge = next(e for e in result["drawable_edges"] if e["host_event_index"] == row["cpu_event_index"])
        self.assertEqual(edge["source_status"], "LINKED")
        self.assertEqual(edge["anchor_status"], "EXACT_SOURCE_ROW")
        self.assertEqual(edge["line_style"], "solid")

    def test_unresolved_phase_never_gets_a_solid_boundary_edge(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            capture, rounds, host, trace, metadata = fixture(directory)
            rounds[0]["u_windows"][0]["stages"][0]["status"] = "UNRESOLVED"
            result = links.export_anchor_links(core, capture, rounds, host, trace, metadata, 0, directory/"proof")
            self.assertEqual(result["links"][0]["status"], "LINKED")
            self.assertTrue(result["links"][0]["host_associations"][0]["runtime_submission_confirmed"])
            self.assertTrue(all(e["line_style"] == "dashed" and not e["runtime_submission_confirmed"]
                                for e in result["drawable_edges"]))

    def test_real_link_to_one_of_two_HC_candidates_is_not_a_unique_HC_boundary(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            capture, rounds, host, trace, metadata = fixture(directory)
            first = capture["tasks"][0]
            first.name = first.op_type = "HcPre"
            capture["tasks"].append(task(2, "HcPre", 210000, 221000))
            rounds[0]["regions"]["main_hc"] = {"start_ns": first.start_ns, "end_ns": EPOCH+230000}
            rounds[0]["u_windows"][0]["stages"][0]["start_anchor"]["name"] = first.name
            events = json.loads(trace.read_text(encoding="utf-8"))
            next(e for e in events if e.get("cat") == "kernel")["name"] = first.name
            trace.write_text(json.dumps(events), encoding="utf-8")
            metadata["sha256"] = hashlib.sha256(trace.read_bytes()).hexdigest()
            result = links.export_anchor_links(core, capture, rounds, host, trace, metadata, 0, directory/"proof")
            self.assertEqual(next(r for r in result["links"] if r["row_index"] == 1)["status"], "LINKED")
            candidates = [e for e in result["drawable_edges"] if e["anchor_kind"] == "main_hc"]
            self.assertTrue(candidates)
            self.assertTrue(all(e["anchor_status"] == "AMBIGUOUS" and e["line_style"] == "dashed" for e in candidates))
            exact = [e for e in result["drawable_edges"] if e["anchor_kind"] == "u_stage"]
            self.assertTrue(any(e["line_style"] == "solid" for e in exact))

    def test_missing_source_reference_preserves_status_but_downgrades_edges(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            capture, rounds, host, trace, metadata = fixture(directory)
            original = links._read_evidence_events
            def without_api(*args, **kwargs):
                events = original(*args, **kwargs)
                return {i: e for i, e in events.items() if not e["name"].startswith("Node@")}
            with patch.object(links, "_read_evidence_events", side_effect=without_api):
                result = links.export_anchor_links(core, capture, rounds, host, trace, metadata, 0, directory/"proof")
            row, = result["links"]
            self.assertEqual(row["status"], "LINKED")
            self.assertEqual(row["missing_source_event_indexes"], [row["node_event_index"]])
            self.assertTrue(all(e["line_style"] == "dashed" for e in result["drawable_edges"]))

    def test_pass_four_progress_is_explicit(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            capture, rounds, host, trace, metadata = fixture(directory)
            progress = []
            links.export_anchor_links(core, capture, rounds, host, trace, metadata, 0, directory/"proof", progress.append)
            self.assertIn("Pass 4/4: verifying source/parent references", progress)

    def test_deep_recorded_parent_path_and_all_core_ancestors_survive(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            capture, rounds, host, trace, metadata = fixture(directory)
            events = json.loads(trace.read_text(encoding="utf-8"))
            next(e for e in events if e.get("name") == host[0]["name"])["args"]["Python parent id"] = 100
            host[0]["python_parent_id"] = "100"
            for number in range(32):
                events.append(xevent("outer.py(1): parent"+str(number), -(number+1)*1000, 100000+number*1000,
                                     399, 399, "python_function", {"Python id": 100+number,
                                     "Python parent id": 101+number if number < 31 else 0}))
            raw = events[-1]
            start, duration = core.a_ns(raw["ts"]), core.a_ns(raw["dur"])
            host.append({"event_index": len(events), "name": raw["name"], "pid": "399", "tid": "399",
                         "cat": "python_function", "start_ns": start, "end_ns": start+duration,
                         "python_id": "131", "python_parent_id": "0"})
            trace.write_text(json.dumps(events), encoding="utf-8")
            metadata["sha256"] = hashlib.sha256(trace.read_bytes()).hexdigest()
            result = links.export_anchor_links(core, capture, rounds, host, trace, metadata, 0, directory/"proof")
            row, = result["links"]
            outer = next(a for a in row["host_associations"] if a["host_event_index"] == len(events))
            self.assertTrue(outer["runtime_submission_confirmed"])
            self.assertEqual(len(outer["explicit_parent_event_indexes"]), 35)
            self.assertGreater(len(row["cpu_ancestors"]), 32)
            retained = {e["event_index"] for e in result["source_events"]}
            self.assertTrue({int(e["id"]) for e in row["cpu_ancestors"]} <= retained)

    def test_containment_never_promotes_even_a_LINKED_device_source(self):
        result, *_ = self.run_fixture(explicit=False)
        row, = result["links"]
        self.assertEqual(row["status"], "LINKED")
        self.assertGreater(len(row["host_associations"]), 0)
        self.assertTrue(all(a["relation"] == "TIME_CONTAINMENT_ONLY" and a["line_style"] == "dashed"
                            and not a["runtime_submission_confirmed"] for a in row["host_associations"]))

    def test_graph_shared_flow_without_origin_never_gets_a_nearby_Host(self):
        result, *_ = self.run_fixture("graph")
        row, = result["links"]
        self.assertEqual(row["status"], "GRAPH_SHARED_NO_ORIGIN", row)
        self.assertIsNone(row["cpu_event_index"])
        self.assertEqual(row["host_associations"], [])
        self.assertTrue(row["flow_ids"])

    def test_duplicate_Node_or_contradictory_queue_stays_ambiguous(self):
        for mode in ("ambiguous", "bad_queue"):
            with self.subTest(mode=mode):
                result, *_ = self.run_fixture(mode)
                row, = result["links"]
                self.assertEqual(row["status"], "AMBIGUOUS", row)
                self.assertEqual(row["host_associations"], [])
                self.assertIsNone(row["node_event_index"])
                self.assertTrue(row["reason"])

    def test_partial_sources_preserve_status_and_do_not_get_solid_arrows(self):
        for mode, present, absent in (("cpu_only", "cpu_event_index", "node_event_index"),
                                      ("api_only", "node_event_index", "cpu_event_index")):
            with self.subTest(mode=mode):
                result, *_ = self.run_fixture(mode)
                row, = result["links"]
                self.assertEqual(row["status"], mode.upper(), row)
                self.assertIsNotNone(row[present]); self.assertIsNone(row[absent])
                self.assertFalse(any(a["runtime_submission_confirmed"] for a in row["host_associations"]))

    def test_native_identity_conflict_is_not_replaced_by_same_time_or_name(self):
        result, *_ = self.run_fixture("wrong_task_id")
        row, = result["links"]
        self.assertEqual(row["status"], "NO_TRACE_DEVICE")
        self.assertEqual(row["host_associations"], [])

    def test_parent_id_on_different_thread_is_not_a_verified_parent(self):
        result, *_ = self.run_fixture("wrong_thread_parent")
        row, = result["links"]
        self.assertEqual(row["status"], "LINKED")
        self.assertTrue(all(not a["runtime_submission_confirmed"] for a in row["host_associations"]))

    def test_all_U_boundaries_and_true_HC_endpoint_tasks_are_kept(self):
        tasks = [task(1, "HcPre", 0, 1), task(2, "HcPost", 90, 100),
                 task(3, "HcPre", 180, 181), task(4, "HcPost", 290, 300),
                 task(5, "HcPre", 370, 371)]
        windows = []
        number = 6
        for key, start, count in (("main_to_draft", 100, 8), ("draft_to_nextmain", 300, 7)):
            boundaries = [None]
            for index in range(1, count):
                t = task(number, "Anchor"+str(number), start+10*index, start+10*index+1)
                number += 1; tasks.append(t); boundaries.append(t)
            boundaries.append(None)
            stages = []
            for index in range(count):
                a, b = start+10*index, start+10*(index+1)
                def anchor(t, stamp):
                    return {"row_index": t.row_index, "name": t.name, "timestamp_ns": EPOCH+stamp} if t else {}
                stages.append({"stage_key": "stage"+str(index+1), "stage_order": index+1, "status": "MATCHED",
                               "start_ns": EPOCH+a, "end_ns": EPOCH+b,
                               "start_anchor": anchor(boundaries[index], a),
                               "end_anchor": anchor(boundaries[index+1], b)})
            windows.append({"window_key": key, "start_ns": EPOCH+start, "end_ns": EPOCH+start+count*10, "stages": stages})
        rounds = [{"round_id": "hc-1", "end_ns": EPOCH+370,
                   "regions": {"main_hc": {"start_ns": EPOCH, "end_ns": EPOCH+100},
                               "draft_hc": {"start_ns": EPOCH+180, "end_ns": EPOCH+300}}, "u_windows": windows}]
        original = deepcopy(rounds)
        bindings, selected = links._device_anchors(core, {"tasks": tasks}, rounds)
        self.assertEqual(len(bindings), 35)  # 2 boundaries x 15 U stages plus five HC endpoints
        self.assertEqual(len(selected), len(tasks))
        self.assertTrue(all(b["status"] in ("EXACT_SOURCE_ROW", "HC_MARKER_TIMESTAMP_UNIQUE") for b in bindings))
        final = next(b for b in bindings if b["anchor_kind"] == "u_stage" and b["stage_number"] == 15 and b["side"] == "end")
        self.assertEqual(final["device_row_indexes"], [5])
        self.assertEqual(rounds, original)
        tasks.append(task(99, "HcPre", 370, 372))
        bindings, _ = links._device_anchors(core, {"tasks": tasks}, rounds)
        final = next(b for b in bindings if b["anchor_kind"] == "next_main_hc")
        self.assertEqual(final["status"], "AMBIGUOUS")
        self.assertEqual(final["device_row_indexes"], [5, 99])

    def test_empty_anchors_do_not_claim_COMPLETE_or_scan_trace(self):
        metadata = {"status": "ALIGNED_X_EVENTS", "sha256": "expected", "clock_offset_ns": "0"}
        with patch.object(core, "residual_host_rows", side_effect=AssertionError("empty request")):
            result = links.export_anchor_links(core, {"tasks": []}, [], [], "unused.json", metadata, 0, "unused")
        self.assertEqual(result["status"], "NO_DEVICE_ANCHORS")
        self.assertEqual(result["scan_passes"], 0)

    def test_unaligned_clock_and_offset_conflict_fail_closed(self):
        for metadata, offset in (({"status": "CLOCK_MISMATCH", "sha256": "x"}, 0),
                                 ({"status": "ALIGNED_X_EVENTS", "sha256": "x", "clock_offset_ns": "5"}, 0)):
            with self.subTest(metadata=metadata):
                with self.assertRaises(ValueError):
                    links.export_anchor_links(core, {"tasks": []}, [], [], "unused", metadata, offset, "unused")

    def test_second_pass_hash_mismatch_does_not_return_links_from_changed_trace(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            capture, rounds, host, trace, metadata = fixture(directory)
            original = core.residual_iter_trace_events
            calls = 0
            def changed(*args, **kwargs):
                nonlocal calls
                calls += 1
                yield from original(*args, **kwargs)
                if calls == 2:
                    kwargs["stats"]["sha256"] = "changed"
            with patch.object(core, "residual_iter_trace_events", side_effect=changed):
                with self.assertRaisesRegex(ValueError, "SHA256"):
                    links.export_anchor_links(core, capture, rounds, host, trace, metadata, 0, directory/"proof")


if __name__ == "__main__":
    unittest.main()
