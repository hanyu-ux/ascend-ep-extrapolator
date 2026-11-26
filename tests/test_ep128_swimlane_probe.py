"""Lossless swimlane export uses saved captures and never prunes coordinates."""

from copy import deepcopy
from concurrent.futures import Future
from decimal import Decimal
import base64
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import profile_ep128 as core
import ep128_swimlane_trace as host_analysis
import test_ep128_swimlane_trace as host_fixtures
import test_ep128_v2_measure as stage_fixtures


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "u_swimlane_probe.py"
spec = importlib.util.spec_from_file_location("swimlane_probe_tests", TOOL)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def input_fixture(directory):
    test = stage_fixtures.ResidualStageTests()
    a, tasks_a, _ = test.stage_fixture("main_to_draft")
    b, tasks_b, _ = test.stage_fixture("draft_to_nextmain")
    origin = stage_fixtures.RESIDUAL_EPOCH
    tasks = []
    for offset, group in ((250, tasks_a), (450, tasks_b)):
        for task in group:
            item = deepcopy(task)
            item.start_ns += offset
            item.end_ns += offset
            item.row_index = len(tasks)+1
            item.task_id = str(item.row_index)
            tasks.append(item)
    capture = {"rank": "dp4", "path": str(directory/"kernel.csv"), "sha256": "csv-fixture", "tasks": tasks}
    measured = {"rank": "dp4", "rounds": [{"round_id": "hc-1", "start_ns": str(origin), "end_ns": str(origin+570),
                "layers": [{"role": "main", "start_ns": str(origin), "end_ns": str(origin+250)},
                           {"role": "draft", "start_ns": str(origin+350), "end_ns": str(origin+450)}]}]}
    events, _ = host_fixtures.fixture()
    next(e for e in events if e["kind"] == "main_graph").update(start_ns=550, end_ns=590, duration_ns=40)
    next(e for e in events if e["kind"] == "state_attn").update(end_ns=540, duration_ns=300)
    next(e for e in events if e["kind"] == "core_execute" and e["start_ns"] == 160).update(end_ns=700, duration_ns=540)
    for e in events:
        e["start_ns"] += origin
        e["end_ns"] += origin
    records = [{"ph": "M", "name": "process_name", "pid": 10, "args": {"name": "Python"}}]
    for e in events:
        records.append({"ph": "X", "name": e["name"], "pid": e["pid"], "tid": e["tid"], "cat": "python_function",
                        "ts": str(Decimal(e["start_ns"])/1000), "dur": str(Decimal(e["duration_ns"])/1000),
                        "args": {"Python id": e["python_id"], "Python parent id": e["python_parent_id"]}})
    kernel = directory/"kernel.csv"; kernel.write_text("fixture", encoding="utf-8")
    trace = directory/"trace_view.json"; trace.write_text(json.dumps(records), encoding="utf-8")
    case = {"name": "synthetic", "bs": 4, "draft_tokens": 7, "draft_layers": 3,
            "with_stack": True, "trim_rounds": 2}
    clock = {"status": "ALIGNED_X_EVENTS", "sha256": hashlib.sha256(trace.read_bytes()).hexdigest(),
             "clock_offset_ns": "0", "anchor_count": 3, "process_names": {"10": "Python"}}
    return case, capture, measured, kernel, trace, clock


class SwimlaneCoordinatesTests(unittest.TestCase):
    def test_all_15_stages_close_and_early_host_calls_stay_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            case, capture, measured, kernel, trace, clock = input_fixture(Path(tmp))
            original = {p: p.read_bytes() for p in (kernel, trace)}
            with patch.object(core, "normalize_case", return_value=case), \
                    patch.object(core, "validate_models"), \
                    patch.object(core, "_fixed_overlap_input", return_value=kernel), \
                    patch.object(core, "_case_trace_path", return_value=trace), \
                    patch.object(core, "read_capture", return_value=capture), \
                    patch.object(core, "measure_capture", return_value=measured) as measure, \
                    patch.object(core, "residual_trace_scan", return_value=(clock, 0)):
                result = probe.analyze_case(case, {"sample_model": {}}, Path(tmp), "dp4", lambda _: None)
            self.assertEqual(result["status"], "READY", result.get("error"))
            self.assertEqual(measure.call_args.args[2]["trim_rounds"], 0)
            self.assertEqual(case["trim_rounds"], 2)
            row = result["rounds"][0]
            self.assertEqual(row["stage_status"], "MATCHED")
            self.assertEqual([s["stage_number"] for w in row["u_windows"] for s in w["stages"]], list(range(1, 16)))
            self.assertEqual(row["duration_ns"], 570)
            self.assertEqual(row["u_ns"], 220)
            lanes = row["host_lanes"]
            self.assertEqual(lanes["outer_propose"]["relative_start_ns"], 50)
            self.assertEqual(lanes["outer_propose"]["duration_ns"], 90)
            self.assertEqual(lanes["outer_propose"]["overlap_ns"]["main_hc"], 90)
            self.assertEqual(lanes["runner_inputs"]["relative_start_ns"], 200)
            self.assertEqual(lanes["runner_inputs"]["overlap_ns"]["main_hc"], 30)
            self.assertEqual(lanes["preparation_dp_aggregate"]["duration_ns"], 40)
            self.assertEqual(lanes["runner_attn"]["relative_start_ns"], 232)
            self.assertEqual(lanes["runner_attn"]["duration_ns"], 6)
            self.assertEqual(lanes["runner_attn"]["overlap_ns"]["main_hc"], 6)
            self.assertEqual(set(lanes), set(probe.HOST_LANE_KEYS))
            self.assertEqual(row["host_pairing"]["pairing_basis"], "HOST_CALL_ORDER_AND_GRAPH_ENTRY_INFERRED")
            for p, before in original.items():
                self.assertEqual(p.read_bytes(), before)

    def test_missing_runner_attn_keeps_U_but_cannot_claim_a_complete_host_timeline(self):
        with tempfile.TemporaryDirectory() as tmp:
            case, capture, measured, kernel, trace, clock = input_fixture(Path(tmp))
            events = [e for e in json.loads(trace.read_bytes())
                      if e.get("name") != host_fixtures.NAMES["runner_attn"]]
            trace.write_text(json.dumps(events), encoding="utf-8")
            clock["sha256"] = hashlib.sha256(trace.read_bytes()).hexdigest()
            with patch.object(core, "normalize_case", return_value=case), patch.object(core, "validate_models"), \
                    patch.object(core, "_fixed_overlap_input", return_value=kernel), \
                    patch.object(core, "_case_trace_path", return_value=trace), \
                    patch.object(core, "read_capture", return_value=capture), \
                    patch.object(core, "measure_capture", return_value=measured), \
                    patch.object(core, "residual_trace_scan", return_value=(clock, 0)):
                result = probe.analyze_case(case, {"sample_model": {}}, Path(tmp), "dp4", lambda _: None)
            self.assertEqual(result["status"], "PARTIAL")
            self.assertEqual(result["complete_host_rounds"], 0)
            self.assertEqual(result["complete_rounds"], 0)
            row = result["rounds"][0]
            self.assertEqual(row["stage_status"], "MATCHED")
            self.assertEqual(row["u_ns"], 220)
            self.assertNotIn("runner_attn", row["host_lanes"])
            self.assertIn("runner_attn", row["host_pairing"]["missing"])

    def test_clock_failure_keeps_device_boundaries_without_host_attribution(self):
        with tempfile.TemporaryDirectory() as tmp:
            case, capture, measured, kernel, trace, clock = input_fixture(Path(tmp))
            clock.update(status="CLOCK_MISMATCH")
            with patch.object(core, "normalize_case", return_value=case), patch.object(core, "validate_models"), \
                    patch.object(core, "_fixed_overlap_input", return_value=kernel), \
                    patch.object(core, "_case_trace_path", return_value=trace), \
                    patch.object(core, "read_capture", return_value=capture), \
                    patch.object(core, "measure_capture", return_value=measured), \
                    patch.object(core, "residual_trace_scan", return_value=(clock, None)), \
                    patch.object(probe, "collect_host_events", side_effect=AssertionError("Must not align failed clocks")):
                result = probe.analyze_case(case, {"sample_model": {}}, Path(tmp), "dp4", lambda _: None)
            self.assertEqual(result["status"], "ERROR")
            self.assertIn("CLOCK_MISMATCH", result["error"])
            self.assertEqual(len(result["rounds"]), 1)
            self.assertEqual(result["host_events"], [])

    def test_unresolved_stages_keep_original_window_and_reject_nonclosing_partition(self):
        with tempfile.TemporaryDirectory() as tmp:
            case, capture, measured, *_ = input_fixture(Path(tmp))
            capture["tasks"] = []
            rounds, _ = probe.round_coordinates(measured, capture, case)
            self.assertEqual(rounds[0]["stage_status"], "UNRESOLVED")
            self.assertEqual(sum(len(w["stages"]) for w in rounds[0]["u_windows"]), 2)
            self.assertEqual(rounds[0]["u_ns"], 220)
            with patch.object(core, "residual_stage_plan", return_value=[]):
                with self.assertRaisesRegex(ValueError, "close"):
                    probe.round_coordinates(measured, capture, case)


class CoordinationBreakdownTests(unittest.TestCase):
    def ref(self, name, start, end, index=1, verified=True):
        return {"event_index": index, "name": "fixture.py(1): "+name, "kind": name,
                "start_ns": start, "end_ns": end, "duration_ns": end-start,
                "parent_chain_verified": verified}

    def child(self, waits=True):
        f=self.ref
        sync=f("sync_cudagraph_and_dp_padding",250,800,10)
        refs=[f("update_requests",10,100),f("gather_batch_req_state",110,160,2),
              f("dispatch",170,200,3),f("dispatch",300,350,4),sync]
        return {"by_role":{"preparation_dp_aggregate":refs},
                "prefix_partition":{"status":"MATCHED", "dp_sync":{"events":[sync]},
                    "wait_candidates":[f("wait",400,600,11),f("wait",450,500,12)] if waits else []}}

    def test_dp_local_and_graph_wall_partition_closes_without_nested_double_count(self):
        report=probe.coordination_breakdown({"start_ns":0,"end_ns":1000},self.child())
        self.assertEqual(report["status"],"MATCHED")
        self.assertEqual(report["dp_coordination_wall_ns"],550)
        self.assertEqual(report["outside_coordination_wall_ns"],450)
        self.assertEqual(report["verified_dp_wait_union_ns"],200)
        self.assertEqual(report["dp_other_wall_ns"],350)
        self.assertEqual(report["known_local_request_batch_ns"],140)
        self.assertEqual(report["local_graph_outside_dp_ns"],30)
        self.assertEqual(report["unattributed_wall_ns"],280)
        self.assertEqual(sum(p["duration_ns"] for p in report["parts"]),1000)
        self.assertTrue(all(a["end_ns"]==b["start_ns"] for a,b in zip(report["parts"],report["parts"][1:])))

    def test_no_recorded_wait_stays_unknown_instead_of_zero(self):
        report=probe.coordination_breakdown({"start_ns":0,"end_ns":1000},self.child(waits=False))
        self.assertEqual(report["dp_coordination_wall_ns"],550)
        self.assertIsNone(report["verified_dp_wait_union_ns"])
        self.assertNotIn("dp_other_wall_ns",report)

    def test_containment_only_sync_does_not_become_verified_dp_time(self):
        child=self.child()
        child["prefix_partition"]["dp_sync"]["events"][0]["parent_chain_verified"]=False
        report=probe.coordination_breakdown({"start_ns":0,"end_ns":1000},child)
        self.assertEqual(report["status"],"PARTIAL")
        self.assertIsNone(report["dp_coordination_wall_ns"])
        self.assertIsNone(report["verified_dp_wait_union_ns"])

    def test_wait_outside_sync_is_not_used_as_dp_wait(self):
        child=self.child()
        child["prefix_partition"]["wait_candidates"][0].update(start_ns=300,end_ns=900)
        report=probe.coordination_breakdown({"start_ns":0,"end_ns":1000},child)
        self.assertEqual(report["status"],"PARTIAL")
        self.assertIn("outside",report["reason"])
        self.assertIsNone(report["verified_dp_wait_union_ns"])

    def test_old_parent_only_evidence_is_not_invented_into_a_dp_split(self):
        report=probe.coordination_breakdown({"start_ns":0,"end_ns":1000},{})
        self.assertEqual(report["status"],"PARTIAL")
        self.assertEqual(report["unattributed_wall_ns"],1000)
        self.assertIsNone(report["dp_coordination_wall_ns"])

    def test_real_child_schema_integrates_with_verified_prefix_statistics(self):
        events,rounds=host_fixtures.fixture()
        h=host_fixtures.host
        events.extend([h(40,"dp_dispatch_patch",170,195,40,3), h(41,"dp_dispatch",171,194,41,40),
                       h(42,"dp_sync",173,192,42,41), h(43,"collective_wrapper",175,190,43,42),
                       h(44,"all_reduce",176,189,44,43), h(45,"cpu_wait",177,188,45,44)])
        pair=host_analysis.pair_rounds(core,rounds,events)[0]
        report=probe.coordination_breakdown({"start_ns":160,"end_ns":200},pair["child_layer"])
        self.assertEqual(pair["child_layer"]["prefix_partition"]["status"],"VERIFIED_HOST_BOUNDARIES")
        self.assertEqual(report["status"],"MATCHED")
        self.assertEqual(report["duration_ns"],40)
        self.assertEqual(report["dp_coordination_wall_ns"],19)
        self.assertEqual(report["verified_dp_wait_union_ns"],11)

    def test_detail_selection_keeps_all_rounds_and_prefers_available_children(self):
        good={"round_id":"good","duration_ns":200,"stage_status":"MATCHED","host_lanes":dict.fromkeys(probe.HOST_LANE_KEYS),
              "coordination_breakdown":{"status":"MATCHED"},"host_pairing":{"status":"MATCHED", "child_layer":{
                "by_role":{k:[{}] for k in probe.ROLE_KEYS}}}}
        partial=deepcopy(good);partial.update(round_id="missing",duration_ns=100,coordination_breakdown={"status":"PARTIAL"})
        rounds=[partial,good];before=deepcopy(rounds)
        selected,status=probe.detail_rounds(rounds,1)
        self.assertEqual(status,"READY")
        self.assertEqual(selected[0]["round_id"],"good")
        self.assertEqual(rounds,before)


class LosslessPackageTests(unittest.TestCase):
    def test_twelve_cases_and_multipart_preserve_every_coordinate_byte(self):
        rng = random.Random(123)
        payload = {"schema": probe.SCHEMA, "cases": [{"case": f"case-{n}", "rounds": [{"start_ns": 1790000000000000007+n,
                    "end_ns": 1790000000000001007+n}], "host_events": [{"start_ns": 1790000000000000001+n,
                    "end_ns": 1790000000000001001+n}], "nonrepeating_fixture": base64.b64encode(
                    bytes(rng.randrange(256) for _ in range(10000))).decode()} for n in range(12)]}
        with tempfile.TemporaryDirectory(prefix="swim package ") as tmp:
            out = Path(tmp)
            archive, files = probe.write_packages(payload, out, "dp4")
            self.assertGreater(len(files), 1)
            package = json.loads((out/"u_swimlane_dp4_package.json").read_text(encoding="utf-8"))
            self.assertFalse(package["truncated"])
            parts = []
            for name in files:
                blob = (out/name).read_bytes()
                self.assertLessEqual(len(blob), 65536)
                parts.append(json.loads(blob))
            self.assertEqual([p["part"] for p in parts], list(range(1, len(parts)+1)))
            self.assertTrue(all(p["parts"] == len(parts) for p in parts))
            gz = base64.b64decode("".join(p["data"] for p in parts))
            raw = gzip.decompress(gz)
            self.assertEqual(hashlib.sha256(raw).hexdigest(), package["raw_sha256"])
            self.assertEqual(hashlib.sha256(gz).hexdigest(), package["gzip_sha256"])
            with zipfile.ZipFile(archive) as z:
                self.assertEqual(z.read("u_swimlane_dp4.json"), raw)
            self.assertEqual((out/"u_swimlane_dp4.json").read_bytes(), raw)
            decoded = json.loads(raw)
            self.assertEqual(len(decoded["cases"]), 12)
            self.assertEqual(decoded["cases"][-1]["host_events"][0]["start_ns"], "1790000000000000012")
            self.assertEqual(decoded, probe.wire(payload))

    def test_full_checkpoint_survives_package_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            payload = {"schema": probe.SCHEMA, "cases": [{"case": "keep-all", "rounds": list(range(500))}]}
            with patch.object(probe.gzip, "compress", side_effect=OSError("synthetic packaging failure")):
                with self.assertRaises(OSError):
                    probe.write_packages(payload, out, "dp4")
            self.assertEqual(json.loads((out/"u_swimlane_dp4.json").read_bytes()), payload)

    def test_long_output_paths_keep_json_zip_and_parts_consistent(self):
        with tempfile.TemporaryDirectory(prefix="swim long ") as tmp:
            try:
                out = Path(tmp)
                for i in range(7):
                    out = out/("long-output-path-"+str(i)+"-"+"x"*28)
                self.assertGreater(len(str(out)), 260)
                core._io_path(out).mkdir(parents=True)
                payload = {"schema": probe.SCHEMA, "cases": [{"case": "long-path", "rounds": []}]}
                archive, parts = probe.write_packages(payload, out, "dp4")
                self.assertEqual(json.loads(core._io_path(out/"u_swimlane_dp4.json").read_bytes()), payload)
                with zipfile.ZipFile(core._io_path(archive)) as z:
                    self.assertEqual(json.loads(z.read("u_swimlane_dp4.json")), payload)
                self.assertTrue(all(core._io_path(out/p).is_file() for p in parts))
            finally:
                # Verify the exact test-owned target, then use extended paths
                # for cleanup as well as the file operations under test.
                target = Path(tmp).resolve()
                self.assertEqual(target.parent, Path(tempfile.gettempdir()).resolve())
                self.assertTrue(target.name.startswith("swim long "))
                shutil.rmtree(core._io_path(target))

    def test_repeated_packages_use_distinct_part_directories_without_deleting_old_parts(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            _, first = probe.write_packages({"cases": [{"case": "old"}]}, out, "dp4")
            old = {p: (out/p).read_bytes() for p in first}
            _, second = probe.write_packages({"cases": [{"case": "new"}]}, out, "dp4")
            self.assertTrue(set(first).isdisjoint(second))
            manifest = json.loads((out/"u_swimlane_dp4_package.json").read_bytes())
            self.assertEqual(manifest["parts"], second)
            for p, raw in old.items():
                self.assertEqual((out/p).read_bytes(), raw)


class WorkerIsolationTests(unittest.TestCase):
    def test_pool_creation_failure_records_every_case(self):
        items = [{"name": "one"}, {"name": "two"}]
        with patch.object(probe, "ProcessPoolExecutor", side_effect=OSError("pool unavailable")):
            result = list(probe.run_cases(items, {}, Path.cwd(), "dp4", 2))
        self.assertEqual({i for i, _ in result}, {0, 1})
        self.assertTrue(all(e["status"] == "ERROR" for _, e in result))

    def test_one_submit_failure_does_not_drop_other_case(self):
        class Pool:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False
            def shutdown(self, **kwargs):
                pass
            def submit(self, function, index, item, config, base, rank, detail_limit):
                if index == 0:
                    raise OSError("one failed submit")
                f = Future()
                f.set_result((index, {"case": item["name"], "rank": rank, "status": "PARTIAL"}))
                return f
        with patch.object(probe, "ProcessPoolExecutor", return_value=Pool()):
            result = dict(probe.run_cases([{"name": "one"}, {"name": "two"}], {}, Path.cwd(), "dp4", 2))
        self.assertEqual(result[0]["status"], "ERROR")
        self.assertEqual(result[1]["status"], "PARTIAL")

    def test_serial_worker_failure_also_preserves_case_identity(self):
        with patch.object(probe, "worker", side_effect=RuntimeError("serial failure")):
            result = list(probe.run_cases([{"name": "one"}], {}, Path.cwd(), "dp4", 1))
        self.assertEqual((result[0][0], result[0][1]["case"], result[0][1]["rank"]), (0, "one", "dp4"))
        self.assertEqual(result[0][1]["status"], "ERROR")


class SwimlaneCLITests(unittest.TestCase):
    def run_cli(self, *args):
        return subprocess.run([sys.executable, "-B", "-S", str(TOOL), *map(str, args)],
                              cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")

    def test_windows_spawn_all_twelve_cases_and_errors_are_exported_without_aborting(self):
        with tempfile.TemporaryDirectory(prefix="swimlane cases ") as tmp:
            directory = Path(tmp)
            config = json.loads((ROOT/"profile_ep128_cases.example.json").read_text(encoding="utf-8"))
            template = config["cases"][0]
            config["cases"] = [{**deepcopy(template), "name": f"bs{bs}-s{s}", "bs": bs, "draft_tokens": s,
                                 "inputs": [{"rank": "dp4", "path": "not-present.csv"}]}
                                for bs in (2, 4, 8, 16) for s in (7, 9, 15)]
            source = directory/"cases.json"
            source.write_text(json.dumps(config), encoding="utf-8-sig")
            original = source.read_bytes()
            result = self.run_cli(source, "--workers", 2, "--expected-cases", 12, "--out", directory/"out")
            self.assertEqual(result.returncode, 1, result.stdout+result.stderr)
            report = json.loads((directory/"out/u_swimlane_dp4.json").read_bytes())
            self.assertEqual(report["completed_cases"], 12)
            self.assertEqual([c["case"] for c in report["cases"]], [c["name"] for c in config["cases"]])
            self.assertTrue(all(c["status"] == "ERROR" for c in report["cases"]))
            self.assertEqual(report["execution"]["workers"], 2)
            self.assertFalse(report["retention"]["byte_budget_truncation"])
            self.assertTrue((directory/"out/u_swimlane_dp4.zip").is_file())
            self.assertEqual(source.read_bytes(), original)

    def test_expected_case_count_rejects_wrong_manifest_before_any_analysis(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/"cases.json"
            p.write_text(json.dumps({"schema_version": core.SCHEMA_VERSION, "cases": [{"name": "only-one"}]}))
            result = self.run_cli(p, "--expected-cases", 12, "--out", Path(tmp)/"out")
            self.assertEqual(result.returncode, 2)
            self.assertIn("Expected 12 cases", result.stderr)
            self.assertFalse((Path(tmp)/"out").exists())

    def test_bad_manifest_path_has_a_clear_cli_error(self):
        result = self.run_cli("nonexistent-swimlane-case-list.json")
        self.assertEqual(result.returncode, 2)
        self.assertIn("Cannot read cases JSON", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_unselected_case_other_rank_source_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            out = directory/"out";out.mkdir()
            original = out/"u_swimlane_dp4.json"
            original.write_text("original source must survive", encoding="utf-8")
            config = json.loads((ROOT/"profile_ep128_cases.example.json").read_text(encoding="utf-8"))
            template = deepcopy(config["cases"][0])
            config["cases"] = [{**deepcopy(template), "name": "selected", "inputs": [{"rank": "dp4", "path": "missing.csv"}]},
                               {**deepcopy(template), "name": "unselected", "inputs": [{"rank": "dp3", "path": "other.csv", "trace_path": str(original)}]}]
            source = directory/"cases.json";source.write_text(json.dumps(config), encoding="utf-8")
            result = self.run_cli(source, "--case", "selected", "--workers", 1, "--out", out)
            self.assertEqual(result.returncode, 2, result.stdout+result.stderr)
            self.assertIn("overwrite", result.stderr.lower())
            self.assertEqual(original.read_text(encoding="utf-8"), "original source must survive")


if __name__ == "__main__":
    unittest.main()
