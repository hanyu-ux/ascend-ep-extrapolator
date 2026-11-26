"""Full-evidence equivalence and a reproducible post-scan exporter benchmark.

Run directly with --benchmark to time the real exporter after mocked core/trace
I/O has returned a complete evidence table. No output records are omitted.
"""

import argparse
import cProfile
import csv
import hashlib
import io
import json
from pathlib import Path
import pstats
import random
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

import profile_ep128 as core
import ep128_swimlane_links as links


EPOCH = 1790000000000000003


class Evidence(dict):
    """Count whole-table key-view requests independently of wall-clock noise."""
    key_view_calls = 0

    def keys(self):
        self.key_view_calls += 1
        return super().keys()


def workload(directory, tasks=2000, raw_events=60000, families=24):
    """Repeated graph task sources plus a large, genuinely retained Host table."""
    raw, hosts, sources = Evidence(), [], []
    flow = lambda category, identity: json.dumps([category, category, None, identity], separators=(",", ":"))
    for family in range(families):
        start, number = EPOCH+family*10000, family*10+1
        records = [
            {"event_index": number, "name": f"runner.py(1): family_{family}", "cat": "python_function",
             "pid": "399", "tid": "399", "start_ns": start, "end_ns": start+9000,
             "python_id": str(number), "python_parent_id": "0"},
            {"event_index": number+1, "name": "helper.py(2): prepare", "cat": "python_function",
             "pid": "399", "tid": "399", "start_ns": start+100, "end_ns": start+8000,
             "python_id": str(number+1), "python_parent_id": str(number)},
            {"event_index": number+2, "name": "aten::metadata", "cat": "cpu_op", "pid": "399", "tid": "399",
             "start_ns": start+1000, "end_ns": start+6000, "python_id": None, "python_parent_id": str(number+1)},
            {"event_index": number+3, "name": "Enqueue@metadata", "cat": "enqueue", "pid": "399", "tid": "399",
             "start_ns": start+2000, "end_ns": start+2500},
            {"event_index": number+4, "name": "Dequeue@metadata", "cat": "dequeue", "pid": "399", "tid": "1171",
             "start_ns": start+3000, "end_ns": start+4000},
            {"event_index": number+5, "name": "Node@metadata", "cat": "cann_api", "pid": "408831", "tid": "1171",
             "start_ns": start+3300, "end_ns": start+3500},
        ]
        for endpoint, phase, stamp, tid in ((6,"s",start+2500,"399"), (7,"f",start+4000,"1171")):
            records.append({"event_index": number+endpoint, "name": "async_task_queue", "cat": "async_task_queue",
                "pid": "399", "tid": tid, "ph": phase, "flow_key": flow("async_task_queue",str(family)),
                "flow_id": str(family), "start_ns": stamp, "end_ns": stamp, "duration_ns": 0})
        raw.update((event["event_index"], event) for event in records)
        hosts.extend(records[:2])
        sources.append(records)
    # These Host functions belong to the exported trace, on another thread.
    # They must remain in source_events and may not be pruned for performance.
    for index in range(max(0, raw_events-len(raw))):
        number, start = families*10+1+index, EPOCH+10000000+index*100
        event = {"event_index": number, "name": "runner.py(2): other_trace_call", "cat": "python_function",
            "pid": "800", "tid": "800", "start_ns": start, "end_ns": start+50,
            "python_id": str(number), "python_parent_id": "0"}
        raw[number] = event
        hosts.append(event)
    device = [SimpleNamespace(row_index=i+1, row_indices=[i+1], name="MetadataKernel", op_type="MetadataKernel",
        device_id="4", model_id="12", stream_id="1", task_id=str(i+1), context_id="7", batch_id="8",
        subtask_id="0", task_type="AI_CORE", op_state=None, start_ns=EPOCH+1000000+i*100,
        end_ns=EPOCH+1000005+i*100) for i in range(tasks)]
    trace, kernel = directory/"trace.json", directory/"kernel.csv"
    trace.write_text("[]", encoding="utf-8"); kernel.write_text("original kernel fixture", encoding="utf-8")
    metadata = {"status":"ALIGNED_X_EVENTS", "clock_offset_ns":"0", "sha256":hashlib.sha256(trace.read_bytes()).hexdigest()}
    rounds = [{"round_id":"hc-1", "start_ns":EPOCH+900000, "end_ns":EPOCH+2000000+tasks*100,
               "regions":{}, "u_windows":[], "host_pairing":{"functions":{
                   f"family_{i}":sources[i][0]["event_index"] for i in range(families)}}}]
    capture = {"rank":"dp4", "path":str(kernel), "tasks":device}

    def resolve(_trace, _capture, _windows, _uncovered, _metadata, _offset, destination, _directory, _progress, _stages):
        destination = destination.with_name("device-tasks-source-links.csv")
        with destination.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=core.RESIDUAL_SOURCE_FIELDS)
            writer.writeheader()
            for i, task in enumerate(device):
                family = sources[i%families]
                root, helper, cpu, enq, deq, node = family[:6]
                row = {"row_index":task.row_index, "name":task.name, "device_start_ns":task.start_ns,
                    "device_end_ns":task.end_ns, "status":"CPU_ONLY" if i%7 == 0 else "LINKED",
                    "cpu_event_index":cpu["event_index"], "cpu_leaf_event_index":cpu["event_index"],
                    "cpu_name":cpu["name"], "cpu_category":cpu["cat"], "cpu_start_ns":cpu["start_ns"],
                    "cpu_end_ns":cpu["end_ns"], "enqueue_event_index":enq["event_index"],
                    "dequeue_event_index":deq["event_index"], "node_event_index":node["event_index"],
                    "enqueue_start_ns":enq["start_ns"], "enqueue_end_ns":enq["end_ns"],
                    "dequeue_start_ns":deq["start_ns"], "dequeue_end_ns":deq["end_ns"],
                    "node_start_ns":node["start_ns"], "node_end_ns":node["end_ns"], "flow_ids_json":"[]",
                    "cpu_ancestors_json":json.dumps([{ "id":e["event_index"], "name":e["name"],
                        "category":e["cat"], "start":e["start_ns"], "end":e["end_ns"]} for e in (root,helper,cpu)])}
                writer.writerow(row)
        return {"source_links_csv":str(destination), "source_status":"PARTIAL", "source_scope":"benchmark complete source table"}
    return capture, rounds, hosts, trace, metadata, raw, resolve


def run_workload(directory, progress=None, **sizes):
    capture, rounds, hosts, trace, metadata, raw, resolver = workload(directory, **sizes)
    with patch.object(core,"residual_host_rows",side_effect=resolver), patch.object(links,"_read_evidence_events",return_value=raw):
        started = time.perf_counter()
        result = links.export_device_task_links(core,capture,rounds,hosts,trace,metadata,0,directory/"out",progress)
        elapsed = time.perf_counter()-started
    digest = hashlib.sha256(json.dumps(result,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode("utf-8")).hexdigest()
    return result, raw, elapsed, digest


def interval_workload(intervals=20000, queries=2000):
    records = [{"event_index":0,"pid":"1","tid":"2","cat":"python_function",
                "start_ns":EPOCH,"end_ns":EPOCH+10*intervals+100}]
    records.extend({"event_index":i+1,"pid":"1","tid":"2","cat":"python_function",
                    "start_ns":EPOCH+10*i,"end_ns":EPOCH+10*i+5} for i in range(intervals))
    index = links._HostIntervals(records)
    started, result = time.perf_counter(), []
    for _ in range(queries):
        event = {"pid":"1","tid":"2","start_ns":EPOCH+10*intervals+10,"end_ns":EPOCH+10*intervals+20}
        result.append([row["event_index"] for row in index.containing(event)])
    elapsed = time.perf_counter()-started
    return elapsed, hashlib.sha256(json.dumps(result,separators=(",",":")).encode("ascii")).hexdigest()


def temporal_workload(tasks=16000, rounds=30):
    records = [{"task_key":f"csv:{i+1}","row_index":i+1,"csv_row_index":i+1,
                "start_ns":EPOCH+i*100,"end_ns":EPOCH+i*100+40} for i in range(tasks)]
    cycles, scopes = [], []
    width = (tasks*100+rounds-1)//rounds
    for number in range(rounds):
        start, end = EPOCH+number*width, EPOCH+(number+1)*width
        cycles.append({"round_id":f"hc-{number}","start_ns":start,"end_ns":end})
        scopes.extend({"round_id":f"hc-{number}","role":f"scope-{role}","workload_round":"n+1",
            "host_event_index":number*5+role,"start_ns":start,"end_ns":end} for role in range(5))
    started = time.perf_counter()
    result = links._temporal_bindings(records,cycles,scopes)
    return time.perf_counter()-started,len(result)


class LinksPerformanceTests(unittest.TestCase):
    def test_complete_postscan_evidence_matches_preoptimization_export(self):
        with tempfile.TemporaryDirectory() as temporary:
            result, raw, _, digest = run_workload(Path(temporary),tasks=48,raw_events=480,families=6)
        # Filled from the unmodified exporter before the performance change.
        self.assertEqual(digest,"48007c0247d8be1fe497197233a76e2bb9f4aa0801a7a67384e197c6548f406f")
        self.assertEqual(len(result["device_tasks"]),48)
        self.assertEqual(len(result["links"]),48)
        self.assertEqual(len(result["source_events"]),len(raw))

    def test_repeated_tasks_do_not_rescan_the_full_raw_key_table_or_parent_chain(self):
        with tempfile.TemporaryDirectory() as temporary:
            original = links._associations
            with patch.object(links,"_associations",wraps=original) as resolver:
                _, raw, _, _ = run_workload(Path(temporary),tasks=240,raw_events=2400,families=6)
        self.assertEqual(raw.key_view_calls,0)
        self.assertLessEqual(resolver.call_count,12)  # six source IDs, two source statuses

    def test_cached_results_still_have_independent_mutable_evidence_lists(self):
        with tempfile.TemporaryDirectory() as temporary:
            result, _, _, _ = run_workload(Path(temporary),tasks=48,raw_events=480,families=6)
        first, repeated = result["links"][1], result["links"][13]
        self.assertEqual(first["host_associations"],repeated["host_associations"])
        self.assertEqual(first["flow_event_indexes"],repeated["flow_event_indexes"])
        first["host_associations"][0]["explicit_parent_event_indexes"].append(-100)
        first["flow_event_indexes"].append(-200)
        self.assertNotEqual(first["host_associations"],repeated["host_associations"])
        self.assertNotEqual(first["flow_event_indexes"],repeated["flow_event_indexes"])

    def test_host_interval_tree_keeps_all_containing_functions_and_endpoint_semantics(self):
        rng = random.Random(29632)
        records = [{"event_index":i,"pid":"1","tid":"2","cat":"python_function",
                    "start_ns":EPOCH+rng.randrange(-100,100),"end_ns":EPOCH+rng.randrange(100,250)}
                   for i in range(200)]
        records += [{"event_index":201,"pid":"1","tid":"2","cat":"python_function",
                     "start_ns":EPOCH-1000,"end_ns":EPOCH+10000}]
        ordered = sorted(records,key=lambda e:(e["start_ns"],e["end_ns"],e["event_index"]))
        index = links._HostIntervals(records)
        for _ in range(200):
            start = EPOCH+rng.randrange(-110,300)
            end = start+rng.randrange(0,80)
            for inclusive in (False,True):
                expected = [e for e in ordered if e["start_ns"]<=start
                    and (e["end_ns"]>=start if inclusive else e["end_ns"]>start) and e["end_ns"]>=end]
                actual = index.containing({"pid":"1","tid":"2","start_ns":start,"end_ns":end},include_end=inclusive)
                self.assertEqual(actual,expected)

    def test_progress_identifies_indexing_associations_and_final_source_counts(self):
        messages = []
        with tempfile.TemporaryDirectory() as temporary:
            run_workload(Path(temporary),tasks=48,raw_events=480,families=6,progress=messages.append)
        self.assertTrue(any(text.startswith("Building device-source indexes:") for text in messages))
        self.assertIn("Device task associations: 0/48 tasks",messages)
        self.assertTrue(any("complete: 48/48 tasks" in text and "source statuses=" in text for text in messages))
        self.assertLess(len(messages),12)


def benchmark():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark",action="store_true")
    parser.add_argument("--interval-benchmark",action="store_true")
    parser.add_argument("--temporal-benchmark",action="store_true")
    parser.add_argument("--tasks",type=int,default=2000)
    parser.add_argument("--raw-events",type=int,default=60000)
    parser.add_argument("--families",type=int,default=24)
    parser.add_argument("--profile",action="store_true")
    args = parser.parse_args()
    if args.temporal_benchmark:
        elapsed, count = temporal_workload()
        print(json.dumps({"tasks":16000,"rounds":30,"host_scopes":150,"seconds":elapsed,"bindings":count},sort_keys=True))
        return
    if args.interval_benchmark:
        elapsed, digest = interval_workload()
        print(json.dumps({"intervals":20000,"queries":2000,"seconds":elapsed,"output_sha256":digest},sort_keys=True))
        return
    profiler = cProfile.Profile()
    with tempfile.TemporaryDirectory() as temporary:
        if args.profile: profiler.enable()
        result, raw, elapsed, digest = run_workload(Path(temporary),tasks=args.tasks,raw_events=args.raw_events,families=args.families)
        if args.profile: profiler.disable()
    print(json.dumps({"tasks":len(result["device_tasks"]), "raw_events":len(raw), "families":args.families,
        "seconds":elapsed, "raw_keys_calls":raw.key_view_calls, "output_sha256":digest},sort_keys=True))
    if args.profile:
        pstats.Stats(profiler,stream=sys.stdout).strip_dirs().sort_stats("cumulative").print_stats(25)


if __name__ == "__main__":
    if any(flag in sys.argv for flag in ("--benchmark","--interval-benchmark","--temporal-benchmark")):
        benchmark()
    else:
        unittest.main()
