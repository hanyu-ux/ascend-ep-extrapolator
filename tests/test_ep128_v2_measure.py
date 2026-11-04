"""CPU-only adversarial fixtures for the V2 measurement fragment."""

import copy
import csv
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import profile_ep128 as M
import profile_ep128 as residual

MODEL = {"num_hidden_layers": 8, "num_hash_layers": 3, "hidden_size": 7168,
         "moe_intermediate_size": 4096, "n_routed_experts": 224,
         "num_experts_per_tok": 16, "quantization_signature": {"weights": "fp4", "activation": "fp8"},
         "compress_ratios": [128, 128, 4, 128, 4, 128, 4, 128, 0, 0, 0]}

CASE = {"name": "synthetic_cpu_fixture", "bs": 32, "draft_tokens": 7,
        "effective_tokens_per_round": 5.2, "token_basis": "configured_conditional",
        "base_ep": 16, "draft_layers": 3, "draft_passes": 1, "trim_rounds": 2}

EPOCH_NS = 1770000000000000123

def stamp_ns(value):
    return f"{value // 1000}.{value % 1000:03d}"

def fixture(directory, *, rounds=8, changes=None):
    rows, records = [], []
    def add(cycle, layer, offset, duration, name, component, stream=1, owner=None):
        role = "main" if layer < 8 else "draft"
        local = layer if layer < 8 else layer - 8
        start = EPOCH_NS + cycle * 1_100_000 + layer * 100_000 + offset * 1000
        shape = "[256,7168];[256,16]" if component == "dispatch" else ""
        row = {"Name": name, "Type": name, "Start Time(us)": stamp_ns(start),
               "Duration(us)": str(duration), "Wait Time(us)": "99999",
               "Device_id": "0", "Model ID": "47" if role == "main" else "49",
               "Stream ID": str(stream), "Task ID": str(len(rows) + 1),
               "Context ID": "3", "Batch ID": "0", "Subtask ID": "0",
               "Input Shapes": shape, "Output Shapes": "", "OP State": "GRAPH"}
        rows.append(row)
        records.append({"row_index": len(rows), "owners": [owner or {"role": role, "layer": local}],
                        "component": component, "start_ns": str(start),
                        "end_ns": str(start + duration * 1000), "task_id": row["Task ID"],
                        "model_id": row["Model ID"], "stream_id": str(stream)})
    for cycle in range(rounds):
        for layer in range(11):
            add(cycle, layer, 0, 1, "TensorClone", "clone")
            add(cycle, layer, 2, 1, "HcPre", "other")
            add(cycle, layer, 6, 4, "Attention", "attention")
            if layer in (2, 4, 6):
                add(cycle, layer, 12, 3, "LightningIndexer", "attention")
            add(cycle, layer, 25, 2, "HcPost", "other")
            add(cycle, layer, 35, 2, "HcPre", "other")
            add(cycle, layer, 38, 1, "Router", "router")
            add(cycle, layer, 40, 10, "MoeDistributeDispatchV2", "dispatch")
            add(cycle, layer, 52, 4, "GroupedMatmulGateUp", "routed_mlp")
            add(cycle, layer, 59, 4, "GroupedMatmulDown", "routed_mlp")
            add(cycle, layer, 64, 10, "MoeDistributeCombineV2", "combine")
            add(cycle, layer, 80, 2, "HcPost", "other")
        add(cycle, 10, 90, 10, "LMHead", "other", stream=3,
            owner={"role": "nonlayer", "name": "main_lmhead"})
    if changes:
        changes(rows, records)
    path = directory / "kernel_details.csv"
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    capture = M.read_capture(path, "dp0")
    return capture, records

class ParserTests(unittest.TestCase):
    def test_exact_epoch_duplicate_rows_and_missing_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            capture, _ = fixture(directory)
            task = capture["tasks"][0]
            self.assertEqual(task.start_ns, EPOCH_NS)
            with Path(capture["path"]).open() as source:
                lines = source.readlines()
            with Path(capture["path"]).open("a") as out:
                out.write(lines[1])
            reread = M.read_capture(capture["path"], "dp0")
            self.assertEqual(reread["duplicate_rows"], 1)
            self.assertEqual(len(reread["tasks"]), len(capture["tasks"]))
            self.assertEqual(len(reread["tasks"][0].row_indices), 2)
            self.assertEqual(reread["tasks"][0].context_id, "3")

    def test_device_alias_conflict_and_empty_primary(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            def conflict(rows, _):
                rows[0]["Device ID"] = "1"
            with self.assertRaisesRegex(ValueError, "conflicting CSV aliases"):
                fixture(directory, changes=conflict)
            def empty(rows, _):
                for row in rows:
                    row["Device_id"] = ""
                    row["Device ID"] = "0"
            capture, _ = fixture(directory, changes=empty)
            self.assertEqual(capture["device_id"], "0")

    def test_duplicate_header_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "bad.csv"
            path.write_text("Name,Name,Start Time(us),Duration(us)\nx,x,1,1\n")
            with self.assertRaisesRegex(ValueError, "duplicate CSV header"):
                M.read_capture(path, "dp0")

    def test_multiple_devices_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, "multiple device IDs"):
                fixture(Path(temp), changes=lambda rows, _: rows[-1].update(Device_id="1"))

    def test_source_shape_conflict_rejected_without_mapping(self):
        with tempfile.TemporaryDirectory() as temp:
            def change(rows, _):
                next(row for row in rows if "Dispatch" in row["Name"])["Input Shapes"] = "[256,7168];[256,24]"
            capture, _ = fixture(Path(temp), changes=change)
            with self.assertRaisesRegex(ValueError, "TopK contradicts") as raised:
                M.measure_capture(capture, MODEL, CASE, temp)
            message = str(raised.exception)
            for value in ('dp0', str(capture['path']), 'observed_K=24',
                          'configured_K=16', 'MoeDistributeDispatchV2',
                          'Input Shapes=', '[256,7168];[256,24]', 'Input Data Types='):
                self.assertIn(value, message)

class CycleTests(unittest.TestCase):
    def test_hc_walls_trimmed_cycles_and_summary_ledger_close(self):
        with tempfile.TemporaryDirectory() as temporary:
            capture, _ = fixture(Path(temporary))
            measured = M.measure_capture(capture, MODEL, CASE, temporary)
        self.assertEqual(len(measured["rounds"]), 3)
        for cycle in measured["rounds"]:
            self.assertAlmostEqual(cycle["duration_ms"], 1.1)
            self.assertEqual(cycle["output_tokens"], 32 * 5.2)
            self.assertEqual(len(cycle["layers"]), 11)
            self.assertEqual(cycle["layers"][0]["layer"], 0)
            self.assertEqual(cycle["layers"][0]["scope"], "HC_marker_interval")
            self.assertAlmostEqual(cycle["layers"][0]["wall_ms"], .08)
            ledger = cycle["ledger"]
            self.assertTrue(ledger["closed"])
            self.assertAlmostEqual(sum(row["duration_ms"] for row in ledger["rows"]), 1.1)
            self.assertAlmostEqual(ledger["recorded_busy_ms"] + ledger["no_recorded_ms"], 1.1)
            self.assertNotIn("intervals", ledger)
            self.assertNotIn("hc_transitions", cycle)
            self.assertTrue(all("signature" not in layer for layer in cycle["layers"]))
        json.dumps(measured)

    def test_invalid_hc_dependency_not_silently_retained(self):
        with tempfile.TemporaryDirectory() as temporary:
            def change(rows, _):
                row = next(row for row in rows if row["Name"] == "MoeDistributeCombineV2")
                row["Duration(us)"] = "30"
            capture, _ = fixture(Path(temporary), changes=change)
            with self.assertRaisesRegex(ValueError, "invalid HC dependency"):
                M.measure_capture(capture, MODEL, CASE, temporary)

    def test_no_implicit_hash_router_default(self):
        with self.assertRaisesRegex(ValueError, "num_hash_layers"):
            M.a_schedule({key: value for key, value in MODEL.items() if key != "num_hash_layers"}, CASE)


class CoverageTests(unittest.TestCase):
    def task(self, index, start, end):
        return M.ATask(index, start, end, 1_000_000, "x", "x", "0", "47", "1", str(index),
                       None, None, None, "", "", "", "", None, None, [index])

    def test_all_stream_union_clips_carry_in_and_never_adds_wait(self):
        tasks = [self.task(1, -10, 40), self.task(2, 20, 60), self.task(3, 80, 90)]
        stats = M.a_stats(tasks, 0, 100)
        self.assertEqual(stats["busy_union_ms"], .00007)
        self.assertEqual(stats["kernel_sum_ms"], .00009)
        self.assertEqual(stats["wall_ms"], .00009)
        self.assertEqual([task.row_index for task in M.ATimeIndex(tasks).overlapping(30, 50)], [1, 2])

    def test_nonzero_layer_transition_can_be_fully_covered_by_tasks(self):
        tasks = [self.task(1, 98_000, 112_000), self.task(2, 108_000, 120_000)]
        stats = M.a_stats(tasks, 100_000, 120_000)
        self.assertEqual(stats["wall_ms"], .02)
        self.assertEqual(stats["busy_union_ms"], .02)
        self.assertEqual(stats["kernel_sum_ms"], .024)



RESIDUAL_EPOCH = 1_790_000_000_000_000_007


def residual_fixture(directory):
    def task(name, start_us, end_us, row, stream="1", wait=0):
        return SimpleNamespace(name=name, op_type=name, start_ns=RESIDUAL_EPOCH + int(start_us * 1000),
            end_ns=RESIDUAL_EPOCH + int(end_us * 1000), row_index=row, row_indices=[row], stream_id=stream,
            task_id=str(row), model_id="12", device_id="0", wait_ns=wait)
    tasks = [task("MoeDistributeDispatchV2", 1, 1.25, 1),
             task("MoeDistributeCombineV2", 3, 3.35, 2),
             task("MoeDistributeDispatchV2", 21, 21.45, 3),
             task("aten::clone", 8, 12, 4, wait=1_000_000_000),
             task("head_gemm", 11, 15, 5, stream="2"),
             task("sampling", 16, 22, 6),
             task("tail_clone", 28, 33, 7),
             task("tail_copy", 32, 36, 8, stream="2"),
             task("ends_at_window_start", 9, 10, 9),
             task("starts_at_window_end", 20, 20.1, 10)]
    tasks.sort(key=lambda value: (value.start_ns, value.end_ns))
    capture = {"rank": "dp/../../0", "path": str(directory / "kernel_details.csv"),
               "sha256": "synthetic-device-only", "tasks": tasks}
    cycle = {"round_id": "hc-1", "start_ns": str(RESIDUAL_EPOCH), "end_ns": str(RESIDUAL_EPOCH + 40000),
             "duration_ms": .04, "layers": [
                 {"role": "main", "layer": 0, "start_ns": str(RESIDUAL_EPOCH), "end_ns": str(RESIDUAL_EPOCH + 10000)},
                 {"role": "draft", "layer": 0, "start_ns": str(RESIDUAL_EPOCH + 20000), "end_ns": str(RESIDUAL_EPOCH + 30000)}]}
    return capture, {"rank": capture["rank"], "rounds": [cycle]}


def _residual_us_string(value_ns):
    return format(Decimal(value_ns) / 1000, "f")


def residual_trace_events(capture, offset_ns=0):
    events = []
    for task in capture["tasks"]:
        if task.row_index <= 3:
            events.append({"ph": "X", "name": task.name, "pid": 900, "tid": 1,
                "ts": _residual_us_string(task.start_ns - offset_ns),
                "dur": _residual_us_string(task.end_ns - task.start_ns),
                "args": {"Task Type": "MIX_AIV", "Physic Stream Id": "1",
                         "Task Id": task.task_id, "Model Id": "12"}})
    for name, start, end, pid, category in (
        ("outer_python", 9, 19, 10, "cpu_op"),
        ("aten::clone", 11, 13, 10, "cpu_op"),
        ("aclrtSynchronizeStream", 14, 18, 20, None),
        ("aclrtMemcpyAsync", 16, 18, 20, None),
        ("aclmdlExecuteAsync", 31, 35, 20, None),
        ("python_tail", 35, 39, 10, "python_function"),
        ("MUST_NOT_BE_HOST", 15, 16, 900, None)):
        event = {"ph": "X", "name": name, "pid": pid, "tid": 1,
                 "ts": _residual_us_string(RESIDUAL_EPOCH + start * 1000 - offset_ns), "dur": str(end - start)}
        if category:
            event["cat"] = category
        events.append(event)
    # Real exports can put metadata after every timed event. Input ordering
    # is deliberately not chronological, and CANN APIs have no cat field.
    events += [{"ph": "M", "name": "process_name", "pid": pid, "args": {"name": name}}
               for pid, name in ((10, "Python"), (20, "CANN"), (900, "Ascend Hardware"))]
    return events


class ResidualClassifierTests(unittest.TestCase):
    def test_cached_classification_matches_original_rules_with_reused_arguments(self):
        processes = {"10": "Python", "20": "CANN", "30": "Runtime Worker",
                     "40": "Ascend Hardware", "50": "GPU kernels", "60": "Other"}
        events = [
            {"pid": 10, "cat": "cpu_op", "args": {}},
            {"pid": " 10 ", "cat": "CPU_OP;python_function, user_annotation"},
            {"pid": 60, "cat": "python_function", "args": None},
            {"pid": 20, "args": {"connection_id": 2 ** 60 + 17}},
            {"pid": 30, "cat": "unclassified", "args": []},
            {"pid": 60, "cat": "CANN_API"},
            {"pid": 60, "cat": "KERNEL", "args": {}},
            {"pid": 10, "cat": "cpu_op, device", "args": {}},
            {"pid": 10, "cat": "cpu_op", "args": {"Task Type": "AI_CORE", "Physic Stream Id": "1"}},
            {"pid": 10, "cat": "cpu_op", "args": {"TASK_type": None, "STREAM-ID": None}},
            {"pid": 10, "cat": "cpu_op", "args": {"Task Type": "AI_CORE"}},
            {"pid": 40, "cat": "cpu_op", "args": {}},
            {"pid": 50, "cat": "user_annotation", "args": {}},
            {"pid": 60, "cat": None, "args": {1: "integer key", "名 字": "unicode key"}},
            {"pid": 60, "cat": "unknown", "args": {"Task Id": "old", "Task-ID": "new"}},
            {"cat": "user_annotation"},
            {"pid": 60},
        ]
        before = copy.deepcopy(events)
        for event in events:
            # This is the original uncached rule, independent of the helpers
            # whose normalized-argument fast path is under test.
            raw = event.get("args")
            arguments = ({re.sub(r"[^a-z0-9]", "", str(key).lower()): value
                          for key, value in raw.items()} if isinstance(raw, dict) else {})
            category = str(event.get("cat", "")).lower()
            device = bool(re.search(r"kernel|device|aicore|aicpu|npu", category) or
                          "tasktype" in arguments and ("physicstreamid" in arguments or "streamid" in arguments))
            identity = None if event.get("pid") is None else str(event["pid"]).strip()
            process = processes.get(identity, "")
            host = bool(not device and not re.search(r"ascend.hardware|device|kernel|gpu|npu", process, re.I) and
                        (set(re.split(r"[,; ]+", category)).intersection(("cpu_op", "python_function", "user_annotation")) or
                         re.search(r"cann|runtime|^python$", process, re.I) or category in ("cann", "runtime", "cann_api")))
            with self.subTest(event=event):
                self.assertEqual(residual.residual_args(event), arguments)
                self.assertEqual(residual.residual_device_hint(event), device)
                self.assertEqual(residual.residual_device_hint(event, arguments), device)
                self.assertEqual(residual.residual_host(event, processes), host)
                self.assertEqual(residual.residual_host(event, processes, arguments), host)
                self.assertEqual(residual.residual_host(event, processes, arguments, device), host)
        self.assertEqual(events, before)

    def test_empty_prepared_arguments_and_false_device_hint_are_not_missing(self):
        event = {"pid": 10, "cat": "cpu_op", "args": {"Task Type": "AI_CORE", "Stream Id": "1"}}
        self.assertTrue(residual.residual_device_hint(event))
        with patch.object(residual, "residual_args", side_effect=AssertionError("prepared empty args were recomputed")):
            self.assertFalse(residual.residual_device_hint(event, {}))
            self.assertTrue(residual.residual_host(event, {"10": "Python"}, {}))
        event["cat"] = "cpu_op,kernel"
        with patch.object(residual, "residual_device_hint", side_effect=AssertionError("prepared False hint was recomputed")):
            self.assertTrue(residual.residual_host(event, {"10": "Python"}, {}, False))
            self.assertFalse(residual.residual_host(event, {"10": "Python"}, {}, True))
            self.assertFalse(residual.residual_host(event, {"10": "Ascend Hardware"}, {}, False))


class StepResidualTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="ep128-residual-test-")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.capture, self.measured = residual_fixture(self.directory)

    def write_trace(self, events=None, wrapper=False):
        path = self.directory / "trace_view.json"
        events = residual_trace_events(self.capture) if events is None else events
        path.write_text(json.dumps({"traceEvents": events} if wrapper else events, ensure_ascii=False), encoding="utf-8")
        return path

    def analyze(self, **kwargs):
        return residual.residual_analyze_capture(self.capture, self.measured, self.directory / "details", **kwargs)

    def test_device_clipping_union_wait_and_exact_hc_residual(self):
        before = copy.deepcopy(self.measured)
        result = self.analyze()
        self.assertEqual(result["status"], "DEVICE_ONLY")
        self.assertEqual(result["trace"]["status"], "NOT_FOUND")
        self.assertEqual(result["residual_ms_mean"], .02)
        first, tail = result["windows"]
        self.assertEqual([first["window_key"], tail["window_key"]], ["main_to_draft", "draft_to_nextmain"])
        self.assertEqual(first["duration_ms"], .01)
        self.assertAlmostEqual(first["recorded_busy_ms"], .009)
        self.assertAlmostEqual(first["no_recorded_ms"], .001)
        self.assertAlmostEqual(first["kernel_sum_ms"], .010)
        self.assertEqual(first["task_count"], 3)
        self.assertAlmostEqual(tail["recorded_busy_ms"], .006)
        self.assertAlmostEqual(tail["no_recorded_ms"], .004)
        self.assertAlmostEqual(tail["kernel_sum_ms"], .007)
        self.assertTrue(all(row["host_event_count"] is None for row in result["windows"]))
        self.assertEqual(self.measured, before)
        path = Path(result["device_tasks_csv"])
        self.assertEqual(path.parent, (self.directory / "details").resolve())
        with path.open(encoding="utf-8-sig", newline="") as stream:
            records = list(csv.DictReader(stream))
        self.assertEqual(len(records), 5)
        clone = next(row for row in records if row["name"] == "aten::clone")
        self.assertEqual(int(clone["clip_start_ns"]), RESIDUAL_EPOCH + 10000)
        self.assertEqual(float(clone["wait_ms_diagnostic"]), 1000.)
        self.assertFalse(any("window_" in row["name"] for row in records))

    def test_long_device_task_can_clip_into_both_windows_without_counting_draft(self):
        self.capture["tasks"].append(SimpleNamespace(name="long_clone", op_type="Clone",
            start_ns=RESIDUAL_EPOCH + 9000, end_ns=RESIDUAL_EPOCH + 41000, row_index=99, wait_ns=0))
        result = self.analyze()
        self.assertTrue(all(row["recorded_busy_ms"] == .01 for row in result["windows"]))
        self.assertTrue(all(row["no_recorded_ms"] == 0 for row in result["windows"]))
        with Path(result["device_tasks_csv"]).open(encoding="utf-8-sig", newline="") as stream:
            rows = [row for row in csv.DictReader(stream) if row["name"] == "long_clone"]
        self.assertEqual(len(rows), 2)
        self.assertEqual(sum(float(row["clip_duration_ms"]) for row in rows), .02)

    def test_zero_window_is_measured_zero_without_division_or_false_tasks(self):
        cycle = self.measured["rounds"][0]
        cycle["layers"][1]["start_ns"] = str(RESIDUAL_EPOCH + 10000)
        result = self.analyze()
        first = result["windows"][0]
        self.assertEqual(first["duration_ms"], 0)
        self.assertEqual(first["recorded_busy_ms"], 0)
        self.assertEqual(first["no_recorded_ms"], 0)
        self.assertEqual(first["task_count"], 0)
        self.assertEqual(result["residual_ms_mean"], .01)

    def test_host_nested_spans_cann_metadata_and_device_exclusion(self):
        path = self.write_trace()
        result = self.analyze()
        self.assertEqual(result["status"], "DEVICE_AND_HOST", result["trace"])
        self.assertEqual(result["trace"]["status"], "ALIGNED_X_EVENTS")
        self.assertEqual(result["trace"]["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertEqual(result["trace"]["anchor_count"], 3)
        self.assertEqual(result["trace"]["clock_offset_ns"], "0")
        first, tail = result["windows"]
        self.assertAlmostEqual(first["host_span_union_ms"], .009)
        self.assertAlmostEqual(first["host_in_device_no_recorded_ms"], .001)
        self.assertEqual(first["host_event_count"], 4)
        self.assertEqual(first["host_sync_events"], 1)
        self.assertEqual(first["host_copy_events"], 1)
        self.assertAlmostEqual(tail["host_span_union_ms"], .008)
        self.assertAlmostEqual(tail["host_in_device_no_recorded_ms"], .003)
        self.assertEqual(tail["host_graph_replay_events"], 1)
        with Path(result["host_events_csv"]).open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 6)
        self.assertFalse(any(row["name"] == "MUST_NOT_BE_HOST" for row in rows))
        self.assertTrue(any(row["process_name"] == "CANN" and not row["category"] for row in rows))

    def test_epoch_nanosecond_precision_and_nonzero_clock_offset(self):
        self.write_trace(residual_trace_events(self.capture, offset_ns=123))
        result = self.analyze()
        self.assertEqual(result["trace"]["clock_offset_ns"], "123")
        self.assertEqual(result["trace"]["max_clock_residual_ns"], 0)
        self.assertEqual(result["trace"]["anchors"][0]["csv_start_ns"], str(self.capture["tasks"][0].start_ns))
        self.assertAlmostEqual(result["windows"][0]["host_span_union_ms"], .009)

    def test_clock_mismatch_or_insufficient_anchors_preserves_device_only(self):
        for mode in ("drift", "missing", "duplicate"):
            events = residual_trace_events(self.capture)
            if mode == "drift":
                events[1]["ts"] = _residual_us_string(_residual_ns_from_us(events[1]["ts"]) + 10000)
            elif mode == "missing":
                events.pop(0)
            else:
                events.insert(0, copy.deepcopy(events[0]))
            with self.subTest(mode=mode):
                self.write_trace(events)
                result = self.analyze()
                self.assertEqual(result["status"], "DEVICE_ONLY")
                self.assertEqual(result["trace"]["status"], "CLOCK_MISMATCH" if mode == "drift" else "CLOCK_UNCONFIRMED")
                self.assertIsNone(result["host_events_csv"])
                self.assertIsNone(result["windows"][0]["host_span_union_ms"])
                self.assertAlmostEqual(result["windows"][0]["recorded_busy_ms"], .009)

    def test_be_events_are_explicitly_unsupported_not_invented_durations(self):
        events = residual_trace_events(self.capture)
        events += [{"ph": phase, "name": "unpaired_cpu", "pid": 10, "tid": 1,
                    "ts": _residual_us_string(RESIDUAL_EPOCH + stamp), "cat": "cpu_op"}
                   for phase, stamp in (("B", 10000), ("E", 20000))]
        self.write_trace(events)
        result = self.analyze()
        self.assertEqual(result["trace"]["be_events_unsupported"], 2)
        self.assertIn("B/E", result["trace"]["supported_host_phase"])
        self.assertAlmostEqual(result["windows"][0]["host_span_union_ms"], .009)

    def test_actual_cann_ri_replay_names_do_not_include_capture_queries(self):
        events = residual_trace_events(self.capture)
        for name in ("aclmdlRIExecuteAsync", "aclmdlExecuteAsync", "rtModelExecute", "aclmdlRICaptureGetInfo"):
            events.append({"ph": "X", "name": name, "pid": 20, "tid": 2,
                           "ts": _residual_us_string(RESIDUAL_EPOCH + 15000), "dur": "1"})
        self.write_trace(events)
        result = self.analyze()
        self.assertEqual(result["windows"][0]["host_graph_replay_events"], 3)
        with Path(result["host_events_csv"]).open(encoding="utf-8-sig", newline="") as stream:
            flags = {row["name"]: row["graph_replay_visible"] for row in csv.DictReader(stream)}
        self.assertEqual(flags["aclmdlRIExecuteAsync"], "True")
        self.assertEqual(flags["aclmdlRICaptureGetInfo"], "False")

    def test_profilerstep_annotation_never_changes_hc_window_boundaries(self):
        events = residual_trace_events(self.capture)
        events.append({"ph": "X", "name": "ProfilerStep#7", "pid": 10, "tid": 1,
                       "cat": "user_annotation", "ts": _residual_us_string(RESIDUAL_EPOCH), "dur": "100"})
        self.write_trace(events)
        result = self.analyze()
        self.assertEqual([row["duration_ms"] for row in result["windows"]], [.01, .01])
        self.assertIn("not_ProfilerStep", result["scope"])
        self.assertEqual(result["residual_ms_mean"], .02)

    def test_truncated_unreadable_and_gzip_trace_do_not_block_device_statistics(self):
        path = self.directory / "trace_view.json"
        path.write_text('[{"ph":"X","name":"truncated"', encoding="utf-8")
        result = self.analyze()
        self.assertEqual(result["status"], "DEVICE_ONLY")
        self.assertEqual(result["trace"]["status"], "UNAVAILABLE")
        with patch.object(residual, "residual_iter_trace_events", side_effect=PermissionError("synthetic permission denial")):
            result = self.analyze()
        self.assertEqual(result["trace"]["status"], "UNAVAILABLE")
        self.assertTrue(all(row["host_event_count"] is None for row in result["windows"]))
        result = self.analyze(trace_path=self.directory / "trace_view.json.gz")
        self.assertEqual(result["trace"]["status"], "UNSUPPORTED_GZIP")

    def test_invalid_real_boundaries_raise_instead_of_hiding_in_device_only(self):
        for kind in ("prefix", "overlap", "duration", "draft_missing", "cycle_overlap"):
            measured = copy.deepcopy(self.measured)
            cycle = measured["rounds"][0]
            if kind == "prefix":
                cycle["layers"][0]["start_ns"] = str(RESIDUAL_EPOCH + 1)
            elif kind == "overlap":
                cycle["layers"][1]["start_ns"] = str(RESIDUAL_EPOCH + 9000)
            elif kind == "duration":
                cycle["duration_ms"] = 1
            elif kind == "draft_missing":
                cycle["layers"].pop()
            else:
                measured["rounds"].append(copy.deepcopy(cycle))
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                residual.residual_analyze_capture(self.capture, measured, self.directory / "invalid")


def _residual_ns_from_us(value):
    return int(Decimal(value) * 1000)


class ResidualStageTests(unittest.TestCase):
    """Independent continuous spans; native types and Host coverage do not define stages."""
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="ep128-stage-u-")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)

    def stage_fixture(self, key="main_to_draft"):
        # Deliberately use V=64, H=16 instead of the real capture's dimensions.
        # Native times are exact ns, with gaps between almost every task.
        tasks = []
        def task(name, start, duration=1, inputs="", outputs="", family=None):
            row = len(tasks) + 1
            tasks.append(SimpleNamespace(name=name, op_type=family or name,
                start_ns=RESIDUAL_EPOCH + start, end_ns=RESIDUAL_EPOCH + start + duration,
                row_index=row, row_indices=[row], stream_id="1", task_id=str(row), model_id="12",
                device_id="0", wait_ns=0, input_shapes=inputs, output_shapes=outputs,
                input_dtypes="", output_dtypes=""))
        task("output_merge", 2)
        task("RmsNorm", 10, 2, '"4,16;16"', '"4,16;4,1"')
        task("aclnnMatmul_MatMulCommon_MatMulV2", 15, 2, '"4,16;64,16"', '"4,64"', "MatMulV2")
        if key == "main_to_draft":
            task("_compute_local_logits_stats_kernel", 20)
            task("_probabilistic_rejection_kernel", 22)
            task("_resample_kernel", 24)
            task("_categorical_finalize_kernel", 26)
            task("_post_update_kernel", 28)
            task("aclnnCat_ConcatD_ConcatD", 30, inputs='"4,16;4,16;4,16"',
                 outputs='"4,48"', family="ConcatD")
            task("auxiliary_fusion_matmul", 34, inputs='"4,48;16,48"', outputs='"4,16"', family="MatMulV2")
            for start in (40, 45, 50):
                task("_prepare_dflash_inputs_kernel", start)
            task("aclnnIndex_IndexCheck_IndexCheck", 55, family="IndexCheck")
            for start in (60, 65, 70):
                task("context_kv_matmul", start - 4, inputs='"4,16;32,16"', outputs='"4,32"', family="MatMulV2")
                task("aclnnScatterNdUpdate_ScatterNdUpdateAiCore_ScatterNdUpdateSk", start,
                     family="ScatterNdUpdateSk")
            task("aclnnGather_BroadcastToAiCore_BroadcastTo", 75, family="BroadcastTo")
            task("SparseAttnSharedkvMetadata", 80)
            embedding, end = 90, 100
            boundaries = (0, 10, 20, 30, 40, 55, 75, 90, 100)
        else:
            for step in range(7):
                start = 20 + step * 8
                task("aclnnEmbedding_GatherV2AiCore_GatherV2", start, inputs='"64,8;4;1"',
                     outputs='"4,8"', family="GatherV2")
                task("aclnnMatmul_MatMulCommon_MatMulV2", start + 1, inputs='"4,8;64,8"',
                     outputs='"4,64"', family="MatMulV2")
                task("aclnnArgMax_ArgMaxV2AiCore_ArgMaxV2", start + 2, inputs='"4,64;"',
                     outputs='"4"', family="ArgMaxV2")
                task("aclnnArgMax_CastAiCore_Cast", start + 3, family="Cast")
                task("aclnnInplaceCopy_ViewCopyAiCore_ViewCopy", start + 4, family="ViewCopy")
            task("aclnnInplaceFillScalar_FillAiCore_Fill", 76, family="Fill")
            task("_zero_kv_blocks_kernel", 80)
            task("_apply_write_kernel", 81)
            task("_expand_idx_mapping_kernel", 82)
            task("_prepare_pos_seq_lens_kernel", 83)
            task("_combine_sampled_and_draft_tokens_kernel", 84)
            task("_gather_block_tables_kernel", 86)
            task("_compute_slot_mappings_kernel", 90)
            task("aclnnGather_BroadcastToAiCore_BroadcastTo", 95, family="BroadcastTo")
            task("SparseAttnSharedkvMetadata", 100)
            embedding, end = 110, 120
            boundaries = (0, 10, 20, 76, 80, 95, 110, 120)
        task("aclnnEmbedding_GatherV2AiCore_GatherV2", embedding, 2, '"64,16;4;1"', '"4,16"', "GatherV2")
        window = {"round_id": "hc-1", "window_key": key, "start_ns": str(RESIDUAL_EPOCH),
                  "end_ns": str(RESIDUAL_EPOCH + end), "duration_ms": end / 1e6}
        return window, tasks, boundaries

    def assert_spans(self, rows, window, boundaries):
        parents = [row for row in rows if row["level"] == 0]
        self.assertEqual([int(row["start_ns"]) - RESIDUAL_EPOCH for row in parents], list(boundaries[:-1]))
        self.assertEqual([int(row["end_ns"]) - RESIDUAL_EPOCH for row in parents], list(boundaries[1:]))
        self.assertEqual([row["duration_ns"] for row in parents], [b - a for a, b in zip(boundaries, boundaries[1:])])
        self.assertEqual(sum(row["duration_ns"] for row in parents), int(window["end_ns"]) - int(window["start_ns"]))
        self.assertTrue(all(row["status"] == "MATCHED" for row in rows))
        self.assertEqual([row["stage_order"] for row in parents], sorted(row["stage_order"] for row in parents))
        self.assertTrue(all(row["boundary_basis"] and row["start_anchor"] and row["end_anchor"] for row in rows))
        return parents

    def test_main_to_draft_has_eight_ordered_continuous_stages(self):
        window, tasks, boundaries = self.stage_fixture()
        before = copy.deepcopy((window, tasks))
        rows = residual.residual_stage_plan(window, list(reversed(tasks)), {"draft_layers": 3, "draft_tokens": 7})
        parents = self.assert_spans(rows, window, boundaries)
        self.assertEqual([row["stage_key"] for row in parents], ["main_head", "main_logits", "verification", "context_fusion",
            "draft_inputs", "context_kv", "draft_metadata", "draft_embedding"])
        self.assertEqual((window, tasks), before)
        # The first span is 10ns even though its sole device task lasts 1ns.
        self.assertEqual(parents[0]["duration_ns"], 10)
        self.assertNotEqual(parents[0]["duration_ns"], residual.a_stats(tasks, RESIDUAL_EPOCH, RESIDUAL_EPOCH + 10)["kernel_sum_ms"] * 1e6)

    def test_tail_has_seven_parent_stages_and_seven_nonadditive_markov_children(self):
        window, tasks, boundaries = self.stage_fixture("draft_to_nextmain")
        rows = residual.residual_stage_plan(window, tasks, {"draft_layers": 3, "draft_tokens": 7})
        parents = self.assert_spans(rows, window, boundaries)
        children = [row for row in rows if row["level"] == 1]
        self.assertEqual(len(parents), 7)
        self.assertEqual(len(children), 7)
        markov = parents[2]
        self.assertTrue(all(row["parent_stage_key"] == markov["stage_key"] for row in children))
        self.assertEqual(int(children[0]["start_ns"]), int(markov["start_ns"]))
        self.assertEqual(int(children[-1]["end_ns"]), int(markov["end_ns"]))
        self.assertEqual(sum(row["duration_ns"] for row in children), markov["duration_ns"])
        self.assertTrue(all(int(a["end_ns"]) == int(b["start_ns"]) for a, b in zip(children, children[1:])))
        # Vocab MatMul and Markov MatMuls share a native type, but are distinct stages.
        self.assertNotEqual(parents[1]["stage_key"], markov["stage_key"])

    def test_missing_ambiguous_or_out_of_order_anchors_preserve_full_unknown_span(self):
        for mode in ("missing", "ambiguous", "out_of_order", "count_mismatch"):
            window, tasks, _ = self.stage_fixture()
            sampler = next(task for task in tasks if task.name == "_compute_local_logits_stats_kernel")
            if mode == "missing":
                tasks.remove(sampler)
            elif mode == "ambiguous":
                duplicate = copy.copy(sampler); duplicate.row_index = 999
                duplicate.start_ns += 2; duplicate.end_ns += 2
                tasks.append(duplicate)
            elif mode == "out_of_order":
                sampler.start_ns = RESIDUAL_EPOCH + 35; sampler.end_ns = sampler.start_ns + 1
            else:
                tasks.remove(next(task for task in tasks if task.op_type == "ScatterNdUpdateSk"))
            with self.subTest(mode=mode):
                rows = residual.residual_stage_plan(window, tasks, {"draft_layers": 3, "draft_tokens": 7})
                self.assertEqual(len(rows), 1)
                self.assertEqual((rows[0]["stage_key"], rows[0]["status"]), ("unresolved", "UNRESOLVED"))
                self.assertTrue(rows[0]["reason"])
                self.assertEqual((int(rows[0]["start_ns"]), int(rows[0]["end_ns"])),
                                 (int(window["start_ns"]), int(window["end_ns"])))
                self.assertEqual(rows[0]["duration_ns"], 100)

    def test_markov_count_uses_case_instead_of_silently_accepting_partial_steps(self):
        window, tasks, _ = self.stage_fixture("draft_to_nextmain")
        rows = residual.residual_stage_plan(window, tasks, {"draft_tokens": 6})
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], "UNRESOLVED")
        self.assertTrue(rows[0]["reason"])

    def test_extra_raw_types_and_gaps_do_not_change_phase_spans(self):
        window, tasks, boundaries = self.stage_fixture()
        extra = copy.copy(tasks[0])
        extra.name = extra.op_type = "arbitrary_native_type"
        extra.start_ns = RESIDUAL_EPOCH + 77; extra.end_ns = RESIDUAL_EPOCH + 89
        tasks.append(extra)
        self.assert_spans(residual.residual_stage_plan(window, tasks, {"draft_layers": 3}), window, boundaries)

    def extra_task(self, tasks, name, start, family, inputs="", outputs="", duration=1):
        task = copy.copy(tasks[0])
        task.name, task.op_type = name, family
        task.start_ns, task.end_ns = RESIDUAL_EPOCH + start, RESIDUAL_EPOCH + start + duration
        task.input_shapes, task.output_shapes = inputs, outputs
        task.row_index = max(task.row_index for task in tasks) + 1
        task.row_indices, task.task_id = [task.row_index], str(task.row_index)
        tasks.append(task)
        return task

    def test_concat_outside_fusion_scope_does_not_reject_known_transition(self):
        window, tasks, boundaries = self.stage_fixture()
        self.extra_task(tasks, "kv_side_concat", 62, "ConcatD", '"4,8;4,8"', '"4,16"')
        rows = residual.residual_stage_plan(window, tasks, {"draft_layers": 3})
        self.assert_spans(rows, window, boundaries)

    def test_in_scope_concats_are_disambiguated_by_actual_fusion_shapes(self):
        window, tasks, boundaries = self.stage_fixture()
        self.extra_task(tasks, "unrelated_feature_concat", 32, "ConcatD", '"4,8;4,8"', '"4,16"')
        rows = residual.residual_stage_plan(window, tasks, {"draft_layers": 3})
        self.assert_spans(rows, window, boundaries)
        # A second fully shape-linked concat is ambiguous even at the same time.
        self.extra_task(tasks, "second_valid_feature_concat", 32, "ConcatD", '"4,24;4,24"', '"4,48"')
        rows = residual.residual_stage_plan(window, tasks, {"draft_layers": 3})
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], "UNRESOLVED")
        self.assertTrue(rows[0]["reason"])
        self.assertEqual(rows[0]["duration_ns"], 100)

    def test_known_ascend_prepare_alias_keeps_exact_phase_boundaries(self):
        window, tasks, boundaries = self.stage_fixture()
        for task in tasks:
            if task.name == "_prepare_dflash_inputs_kernel":
                task.name = "_prepare_dflash_inputs_kernel_ascend"
        self.assert_spans(residual.residual_stage_plan(window, tasks, {"draft_layers": 3}), window, boundaries)

    def test_markov_ignores_argmax_outside_chain_or_with_unrelated_shape(self):
        window, tasks, boundaries = self.stage_fixture("draft_to_nextmain")
        self.extra_task(tasks, "aclnnArgMax_ArgMaxV2AiCore_ArgMaxV2", 5, "ArgMaxV2", '"4,64;"', '"4"')
        self.extra_task(tasks, "aclnnArgMax_ArgMaxV2AiCore_ArgMaxV2", 100, "ArgMaxV2", '"4,64;"', '"4"')
        self.extra_task(tasks, "aclnnArgMax_ArgMaxV2AiCore_ArgMaxV2", 25, "ArgMaxV2", '"2,32;"', '"2"')
        self.assert_spans(residual.residual_stage_plan(window, tasks, {"draft_tokens": 7}), window, boundaries)

    def test_missing_concat_output_requires_a_derivable_matching_input_shape(self):
        for inputs, matches in (('"4,16;4,16;4,16"', True), ('"4,8;4,8"', False), ('N/A', False)):
            window, tasks, boundaries = self.stage_fixture()
            concat = next(task for task in tasks if task.op_type == "ConcatD")
            concat.input_shapes, concat.output_shapes = inputs, ""
            with self.subTest(inputs=inputs):
                rows = residual.residual_stage_plan(window, tasks, {"draft_layers": 3})
                if matches:
                    self.assert_spans(rows, window, boundaries)
                else:
                    self.assertEqual(len(rows), 1)
                    self.assertEqual(rows[0]["status"], "UNRESOLVED")
                    self.assertEqual(rows[0]["duration_ns"], 100)
                    self.assertTrue(rows[0]["reason"])

    def test_final_norm_must_match_full_projection_activation_shape(self):
        for key in ("main_to_draft", "draft_to_nextmain"):
            window, tasks, boundaries = self.stage_fixture(key)
            self.extra_task(tasks, "RmsNorm", 13, "RmsNorm", '"9,16;16"', '"9,16;9,1"')
            with self.subTest(key=key):
                rows = residual.residual_stage_plan(window, tasks, {"draft_layers": 3, "draft_tokens": 7})
                self.assert_spans(rows, window, boundaries)
                tasks.remove(next(task for task in tasks if task.name == "RmsNorm" and task.start_ns == RESIDUAL_EPOCH + 10))
                rows = residual.residual_stage_plan(window, tasks, {"draft_layers": 3, "draft_tokens": 7})
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["status"], "UNRESOLVED")
                self.assertEqual(rows[0]["duration_ns"], boundaries[-1])

    def test_missing_lookup_output_uses_indices_shape_instead_of_bias_batch(self):
        for indices, matches in (("4", True), ("9", False), ("", False)):
            window, tasks, boundaries = self.stage_fixture("draft_to_nextmain")
            lookup = next(task for task in tasks if task.name.startswith("aclnnEmbedding"))
            lookup.input_shapes = f'"64,8;{indices};1"'
            lookup.output_shapes = ""
            with self.subTest(indices=indices):
                rows = residual.residual_stage_plan(window, tasks, {"draft_tokens": 7})
                if matches:
                    self.assert_spans(rows, window, boundaries)
                else:
                    self.assertEqual(len(rows), 1)
                    self.assertEqual(rows[0]["status"], "UNRESOLVED")
                    self.assertEqual(rows[0]["duration_ns"], 120)
                    self.assertTrue(rows[0]["reason"])

    def test_markov_positions_share_request_count_and_low_rank_width(self):
        for axis in ("request_count", "low_rank"):
            window, tasks, _ = self.stage_fixture("draft_to_nextmain")
            request_count, low_rank = (9, 8) if axis == "request_count" else (4, 9)
            lookup = next(task for task in tasks if task.start_ns == RESIDUAL_EPOCH + 28)
            lookup.input_shapes, lookup.output_shapes = f'"64,{low_rank};{request_count};1"', f'"{request_count},{low_rank}"'
            matrix = next(task for task in tasks if task.start_ns == RESIDUAL_EPOCH + 29)
            matrix.input_shapes, matrix.output_shapes = f'"{request_count},{low_rank};64,{low_rank}"', f'"{request_count},64"'
            argmax = next(task for task in tasks if task.start_ns == RESIDUAL_EPOCH + 30)
            argmax.input_shapes, argmax.output_shapes = f'"{request_count},64;"', f'"{request_count}"'
            with self.subTest(axis=axis):
                rows = residual.residual_stage_plan(window, tasks, {"draft_tokens": 7})
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["status"], "UNRESOLVED")
                self.assertEqual(rows[0]["duration_ns"], 120)
                self.assertTrue(rows[0]["reason"])

    def test_reduced_draft_vocabulary_uses_bias_projection_width_not_main_embedding_width(self):
        window, tasks, boundaries = self.stage_fixture("draft_to_nextmain")
        for task in tasks:
            if task.op_type == "MatMulV2":
                width = 16 if task.start_ns == RESIDUAL_EPOCH + 15 else 8
                task.input_shapes, task.output_shapes = f'"4,{width};48,{width}"', '"4,48"'
            elif task.op_type == "ArgMaxV2":
                task.input_shapes = '"4,48;"'
        self.assert_spans(residual.residual_stage_plan(window, tasks, {"draft_tokens": 7}), window, boundaries)

    def test_tail_head_does_not_prefer_main_vocabulary_over_ambiguous_projection(self):
        window, tasks, _ = self.stage_fixture("draft_to_nextmain")
        for task in tasks:
            if task.op_type == "MatMulV2":
                width = 16 if task.start_ns == RESIDUAL_EPOCH + 15 else 8
                task.input_shapes, task.output_shapes = f'"4,{width};48,{width}"', '"4,48"'
            elif task.op_type == "ArgMaxV2":
                task.input_shapes = '"4,48;"'
        self.extra_task(tasks, "extra_main_vocab_projection", 18, "MatMulV2", '"4,16;64,16"', '"4,64"')
        rows = residual.residual_stage_plan(window, tasks, {"draft_tokens": 7})
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], "UNRESOLVED")
        self.assertEqual(rows[0]["duration_ns"], 120)
        self.assertIn("vocabulary projection", rows[0]["reason"])
        self.assertIn("observed 2", rows[0]["reason"])

    def test_explicit_gumbel_is_unknown_only_inside_markov_scope(self):
        for stamp in (25, 5, 100, 125):
            window, tasks, boundaries = self.stage_fixture("draft_to_nextmain")
            self.extra_task(tasks, "_gumbel_sample_kernel", stamp, "_gumbel_sample_kernel")
            with self.subTest(stamp=stamp):
                rows = residual.residual_stage_plan(window, tasks, {"draft_tokens": 7})
                if stamp == 25:
                    self.assertEqual(len(rows), 1)
                    self.assertEqual(rows[0]["stage_key"], "unresolved")
                    self.assertEqual(rows[0]["status"], "UNRESOLVED")
                    self.assertEqual(rows[0]["duration_ns"], 120)
                    self.assertIn("Gumbel", rows[0]["reason"])
                    self.assertIn("Embedding=7", rows[0]["reason"])
                    self.assertIn("configured positions=7", rows[0]["reason"])
                else:
                    self.assert_spans(rows, window, boundaries)

    def test_greedy_positions_require_each_matching_argmax_and_strict_cast_copy_suffix(self):
        for invalid in ("missing_position", "wrong_cast", "wrong_copy"):
            window, tasks, _ = self.stage_fixture("draft_to_nextmain")
            if invalid == "missing_position":
                tasks.remove(next(task for task in tasks if task.op_type == "ArgMaxV2"))
                # Preserve the global ArgMax count with an unrelated head op.
                self.extra_task(tasks, "aclnnArgMax_ArgMaxV2AiCore_ArgMaxV2", 5, "ArgMaxV2", '"4,64;"', '"4"')
            else:
                pattern = "aclnnArgMax_Cast" if invalid == "wrong_cast" else "aclnnInplaceCopy_ViewCopy"
                next(task for task in tasks if task.name.startswith(pattern)).name = "unrelated_same_type_kernel"
            with self.subTest(invalid=invalid):
                rows = residual.residual_stage_plan(window, tasks, {"draft_tokens": 7})
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["status"], "UNRESOLVED")
                self.assertTrue(rows[0]["reason"])
                self.assertEqual(rows[0]["duration_ns"], 120)

    def matched_capture(self):
        capture, measured = residual_fixture(self.directory)
        capture["tasks"] = [task for task in capture["tasks"] if task.row_index <= 3]
        for key, base in (("main_to_draft", 10000), ("draft_to_nextmain", 30000)):
            _, tasks, _ = self.stage_fixture(key)
            for task in tasks:
                task.start_ns = RESIDUAL_EPOCH + base + (task.start_ns - RESIDUAL_EPOCH) * 80
                task.end_ns = RESIDUAL_EPOCH + base + (task.end_ns - RESIDUAL_EPOCH) * 80
                task.row_index += len(capture["tasks"]) + 1000
                capture["tasks"].append(task)
        capture["tasks"].sort(key=lambda task: (task.start_ns, task.end_ns))
        return capture, measured

    def test_stage_means_keep_host_overlap_nonadditive_and_children_close_within_parent(self):
        capture, measured = self.matched_capture()
        (self.directory / "trace_view.json").write_text(json.dumps(residual_trace_events(capture)), encoding="utf-8")
        result = residual.residual_analyze_capture(capture, measured, self.directory / "matched",
                                                 case={"draft_layers": 3, "draft_tokens": 7})
        self.assertEqual(result["status"], "DEVICE_AND_HOST")
        self.assertEqual(len(result["stage_summary"]), 15)
        self.assertEqual(len(result["substage_summary"]), 7)
        for key in ("main_to_draft", "draft_to_nextmain"):
            phases = [row for row in result["stage_summary"] if row["window_key"] == key]
            self.assertTrue(all(row["rounds"] == row["matched_windows"] == 1 and row["status"] == "MATCHED" for row in phases))
            self.assertAlmostEqual(sum(row["duration_ms_mean"] for row in phases), .01)
            for row in phases:
                self.assertAlmostEqual(row["recorded_busy_ms_mean"] + row["no_recorded_ms_mean"], row["duration_ms_mean"])
                self.assertLessEqual(row["host_span_union_ms_mean"], row["duration_ms_mean"])
        children = result["substage_summary"]
        parent = next(row for row in result["stage_summary"] if row["stage_key"] == children[0]["parent_stage_key"])
        self.assertAlmostEqual(sum(row["duration_ms_mean"] for row in children), parent["duration_ms_mean"])

    def test_host_gap_intersections_match_independent_endpoint_partition_for_all_stages(self):
        capture, measured = self.matched_capture()
        # Overlapping streams cover some of the many native-task gaps. Their
        # durations must be unioned before intersecting Host wall spans.
        for row, start, end, stream in ((99001, 10800, 11200, "2"), (99002, 10900, 11300, "3")):
            capture["tasks"].append(SimpleNamespace(name="extra_stream_work", op_type="SyntheticWork",
                start_ns=RESIDUAL_EPOCH + start, end_ns=RESIDUAL_EPOCH + end,
                row_index=row, row_indices=[row], stream_id=stream, task_id=str(row), model_id="12", wait_ns=0))
        host_spans = [(9000, 10000), (10100, 10900), (10700, 11600), (10800, 11000),
                      (12000, 15000), (19800, 30200), (20000, 30000), (33000, 34000), (40000, 41000)]
        events = [event for event in residual_trace_events(capture)
                  if event.get("ph") == "M" or event.get("pid") == 900]
        for index, (left, right) in enumerate(host_spans):
            events.append({"ph": "X", "name": "span_" + str(index), "pid": 10, "tid": index % 3,
                           "cat": "cpu_op", "ts": _residual_us_string(RESIDUAL_EPOCH + left),
                           "dur": _residual_us_string(right - left), "args": {}})
        (self.directory / "trace_view.json").write_text(json.dumps(events), encoding="utf-8")
        result = residual.residual_analyze_capture(capture, measured, self.directory / "intersections",
                                                 case={"draft_layers": 3, "draft_tokens": 7})
        self.assertEqual(result["status"], "DEVICE_AND_HOST", result["trace"])

        def expected(left, right):
            # Exhaustive endpoint partition, rather than the optimized gap
            # index: test every elementary interval for Host/device coverage.
            hosts = [(RESIDUAL_EPOCH + a, RESIDUAL_EPOCH + b) for a, b in host_spans]
            devices = [(task.start_ns, task.end_ns) for task in capture["tasks"]]
            points = {left, right}
            for a, b in hosts + devices:
                if a < right and b > left:
                    points.update((max(left, a), min(right, b)))
            ordered, covered, in_gap = sorted(points), 0, 0
            for a, b in zip(ordered, ordered[1:]):
                if any(start <= a and end >= b for start, end in hosts):
                    covered += b - a
                    if not any(start <= a and end >= b for start, end in devices):
                        in_gap += b - a
            return covered / 1e6, in_gap / 1e6

        with Path(result["stages_csv"]).open(encoding="utf-8-sig", newline="") as stream:
            stages = list(csv.DictReader(stream))
        self.assertEqual(len(stages), 22)  # 15 parents and seven Markov children.
        self.assertTrue(all(row["status"] == "MATCHED" for row in stages))
        for row in result["windows"] + stages:
            with self.subTest(window=row["window_key"], stage=row.get("stage_key")):
                covered, in_gap = expected(int(row["start_ns"]), int(row["end_ns"]))
                self.assertAlmostEqual(float(row["host_span_union_ms"]), covered)
                self.assertAlmostEqual(float(row["host_in_device_no_recorded_ms"]), in_gap)
        with Path(result["host_events_csv"]).open(encoding="utf-8-sig", newline="") as stream:
            clips = list(csv.DictReader(stream))
        self.assertFalse(any(row["name"] in ("span_0", "span_6", "span_8") for row in clips))
        cross_window = [row for row in clips if row["name"] == "span_5"]
        self.assertEqual(len(cross_window), 2)
        self.assertAlmostEqual(sum(float(row["clip_duration_ms"]) for row in cross_window), .0004)

    def test_partial_stage_matches_do_not_publish_observed_only_means(self):
        capture, measured = self.matched_capture()
        later = copy.deepcopy(measured["rounds"][0])
        later["round_id"] = "hc-2"
        for row in (later, *later["layers"]):
            for field in ("start_ns", "end_ns"):
                row[field] = str(int(row[field]) + 40000)
        measured["rounds"].append(later)
        for original in list(capture["tasks"]):
            if original.name == "_compute_local_logits_stats_kernel":
                continue
            task = copy.copy(original)
            task.start_ns += 40000; task.end_ns += 40000; task.row_index += 10000
            capture["tasks"].append(task)
        result = residual.residual_analyze_capture(capture, measured, self.directory / "partial",
                                                 case={"draft_layers": 3, "draft_tokens": 7})
        partial = [row for row in result["stage_summary"]
                   if row["window_key"] == "main_to_draft" and row["stage_key"] != "unresolved"]
        self.assertEqual(len(partial), 8)
        self.assertTrue(all(row["rounds"] == 2 and row["matched_windows"] == 1 for row in partial))
        self.assertTrue(all(row["status"] != "MATCHED" and row["duration_ms_mean"] is None for row in partial))

    def test_device_only_unknown_stages_keep_exact_windows_and_csv_without_mutation(self):
        capture, measured = residual_fixture(self.directory)
        before = copy.deepcopy(measured)
        result = residual.residual_analyze_capture(capture, measured, self.directory / "unknown")
        self.assertEqual(measured, before)
        self.assertEqual(result["status"], "DEVICE_ONLY")
        self.assertEqual(result["substage_summary"], [])
        self.assertTrue(all(row["status"] == "UNRESOLVED" and row["matched_windows"] == 0
                            and row["duration_ms_mean"] is None for row in result["stage_summary"]))
        self.assertTrue(all(row["host_span_union_ms"] is None for row in result["windows"]))
        with Path(result["stages_csv"]).open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 2)
        for window in result["windows"]:
            stage = next(row for row in rows if row["window_key"] == window["window_key"])
            self.assertEqual(stage["stage_key"], "unresolved")
            self.assertEqual(int(stage["duration_ns"]), int(window["end_ns"]) - int(window["start_ns"]))
        self.assertNotIn("breakdown_summary", result)
        self.assertNotIn("cycle_breakdown", result)
        self.assertNotIn("host_summary", result)


@unittest.skipUnless(
    (Path(__file__).resolve().parents[1] / "real_validation_20261102"
     / "dp9_pasted_evidence_20261103" / "stage_oracle.json").is_file(),
    "Optional private device trace fixture is not bundled; synthetic coverage remains enabled")
class Dp9PastedStageTests(unittest.TestCase):
    """Complete user-supplied device windows, with independent literal ns oracles."""

    @classmethod
    def setUpClass(cls):
        cls.fixture_dir = (Path(__file__).resolve().parents[1] / "real_validation_20261102"
                           / "dp9_pasted_evidence_20261103")
        cls.oracle = json.loads((cls.fixture_dir / "stage_oracle.json").read_text(encoding="utf-8"))
        cls.payloads = {}
        for round_id, source in cls.oracle["payloads"].items():
            raw = (cls.fixture_dir / source["file"]).read_bytes()
            if hashlib.sha256(raw).hexdigest() != source["sha256"]:
                raise AssertionError("dp9 compact fixture no longer matches its transfer SHA256")
            cls.payloads[round_id] = json.loads(raw)

    def window_fixture(self, round_id, kind):
        payload = self.payloads[round_id]
        window_index, original = next((index, window) for index, window in enumerate(payload["windows"])
                                      if window["window_key"] == kind)
        tasks = []
        for owner, row_index, signature, offset, duration in payload["tasks"]:
            if owner != window_index:
                continue
            fields = dict(zip(payload["signature_columns"], payload["signatures"][signature]))
            start = int(original["start_ns"]) + offset
            tasks.append(residual.ATask(row_index=row_index, start_ns=start, end_ns=start + duration,
                wait_ns=0, task_id=None, context_id=None, batch_id=None, subtask_id=None,
                row_indices=[row_index], **fields))
        self.assertEqual(len(tasks), original["task_count"])
        return copy.deepcopy(original), tasks, copy.deepcopy(payload["metadata"]["config"])

    def assert_unknown_window(self, window, tasks, case):
        rows = residual.residual_stage_plan(window, tasks, case)
        self.assertEqual(len(rows), 1)
        self.assertEqual((rows[0]["stage_key"], rows[0]["status"]), ("unresolved", "UNRESOLVED"))
        self.assertEqual((rows[0]["start_ns"], rows[0]["end_ns"]),
                         (window["start_ns"], window["end_ns"]))
        self.assertEqual(rows[0]["duration_ns"], int(window["end_ns"]) - int(window["start_ns"]))
        self.assertTrue(rows[0]["reason"])

    def test_two_complete_dp9_rounds_match_all_independent_parent_and_child_ns(self):
        for round_id in ("hc-3", "hc-107"):
            for kind in ("main_to_draft", "draft_to_nextmain"):
                with self.subTest(round_id=round_id, kind=kind):
                    window, tasks, case = self.window_fixture(round_id, kind)
                    original = copy.deepcopy((window, tasks, case))
                    expected = self.oracle["rounds"][round_id][kind]
                    rows = residual.residual_stage_plan(window, list(reversed(tasks)), case)
                    parents = [row for row in rows if row["level"] == 0]
                    children = [row for row in rows if row["level"] == 1]
                    boundaries = expected["boundaries_ns"]
                    self.assertEqual([row["stage_key"] for row in parents], expected["stage_keys"])
                    self.assertEqual([row["start_ns"] for row in parents], boundaries[:-1])
                    self.assertEqual([row["end_ns"] for row in parents], boundaries[1:])
                    self.assertEqual([row["duration_ns"] for row in parents], expected["duration_ns"])
                    self.assertEqual(sum(row["duration_ns"] for row in parents),
                                     int(window["end_ns"]) - int(window["start_ns"]))
                    self.assertTrue(all(row["status"] == "MATCHED" for row in rows))
                    self.assertEqual((window, tasks, case), original)
                    if kind == "draft_to_nextmain":
                        self.assertEqual([row["stage_key"] for row in children],
                                         ["markov_" + str(index) for index in range(1, 8)])
                        self.assertEqual([row["start_ns"] for row in children], expected["markov_boundaries_ns"][:-1])
                        self.assertEqual([row["end_ns"] for row in children], expected["markov_boundaries_ns"][1:])
                        self.assertEqual([row["duration_ns"] for row in children], expected["markov_duration_ns"])
                        self.assertTrue(all(row["parent_stage_key"] == "markov" for row in children))
                        self.assertEqual(sum(row["duration_ns"] for row in children), parents[2]["duration_ns"])
                    else:
                        self.assertEqual(children, [])

    def test_dp9_fixture_preserves_exceptional_optional_work_and_all_window_counts(self):
        meta = self.payloads["hc-3"]["metadata"]
        self.assertEqual(meta, self.payloads["hc-107"]["metadata"])
        self.assertEqual((meta["rank"], meta["retained_rounds"], meta["total_windows"], meta["total_task_rows"]),
                         ("dp9", 152, 304, 54108))
        patterns = meta["all_window_marker_patterns"]
        windows = [(round_id, row["window_key"]) for row in patterns for round_id in row["round_ids"]]
        self.assertEqual(len(windows), len(set(windows)))
        self.assertEqual(len(windows), 304)
        self.assertEqual(sum(len(row["round_ids"]) * row["counts"]["task_rows"] for row in patterns), 54108)
        for round_id, expected_count, has_optional in (("hc-3", 110, True), ("hc-107", 106, False)):
            window, tasks, _ = self.window_fixture(round_id, "draft_to_nextmain")
            self.assertEqual(window["task_count"], expected_count)
            names = {task.name for task in tasks}
            for name in ("_zero_kv_blocks_kernel", "_apply_write_kernel"):
                self.assertEqual(name in names, has_optional)

    def test_dp9_kv_compress_requires_matching_payload_slots_cache_and_dtypes(self):
        for round_id in ("hc-3", "hc-107"):
            for invalid in ("payload_rows", "payload_width", "slot_count", "cache_output", "payload_dtype",
                            "slot_dtype", "unknown_dtype", "cache_output_dtype"):
                with self.subTest(round_id=round_id, invalid=invalid):
                    window, tasks, case = self.window_fixture(round_id, "main_to_draft")
                    write = next(task for task in tasks if task.op_type == "KvCompressEpilog")
                    shapes = write.input_shapes.strip('"').split(";")
                    payload = [int(value) for value in shapes[1].split(",")]
                    if invalid in ("payload_rows", "payload_width"):
                        payload[0 if invalid == "payload_rows" else 1] += 1
                        shapes[1] = ",".join(map(str, payload))
                        write.input_shapes = '"' + ";".join(shapes) + '"'
                    elif invalid == "slot_count":
                        shapes[2] = str(payload[0] + 1)
                        write.input_shapes = '"' + ";".join(shapes) + '"'
                    elif invalid == "cache_output":
                        write.output_shapes = '"' + str(int(shapes[0]) + 1) + '"'
                    elif invalid == "payload_dtype":
                        write.input_dtypes = "UINT8;INT32;INT32"
                    elif invalid == "slot_dtype":
                        write.input_dtypes = "UINT8;DT_BF16;INT64"
                    elif invalid == "unknown_dtype":
                        write.input_dtypes = ""
                    else:
                        write.output_dtypes = "INT32"
                    self.assert_unknown_window(window, tasks, case)

    def test_dp9_fusion_weight_contraction_and_output_dimensions_match_actual_concat(self):
        for round_id in ("hc-3", "hc-107"):
            for invalid in ("contraction_width", "output_width"):
                with self.subTest(round_id=round_id, invalid=invalid):
                    window, tasks, case = self.window_fixture(round_id, "main_to_draft")
                    boundaries = self.oracle["rounds"][round_id]["main_to_draft"]["boundaries_ns"]
                    candidates = [task for task in tasks if int(boundaries[3]) <= task.start_ns < int(boundaries[4])
                                  and "matmul" in (task.op_type + task.name).lower()]
                    self.assertEqual(len(candidates), 1)
                    fusion = candidates[0]
                    shapes = fusion.input_shapes.strip('"').split(";")
                    weight = [int(value) for value in shapes[1].split(",")]
                    weight[1 if invalid == "contraction_width" else 0] += 1
                    shapes[1] = ",".join(map(str, weight))
                    fusion.input_shapes = '"' + ";".join(shapes) + '"'
                    self.assert_unknown_window(window, tasks, case)

    def test_dp9_direct_int64_argmax_copy_requires_known_matching_dtypes_and_source_shape(self):
        for round_id in ("hc-3", "hc-107"):
            for invalid in ("argmax_dtype_unknown", "copy_input_dtype_unknown", "copy_output_dtype_unknown",
                            "copy_input_dtype_mismatch", "copy_output_dtype_mismatch", "copy_source_rows"):
                with self.subTest(round_id=round_id, invalid=invalid):
                    window, tasks, case = self.window_fixture(round_id, "draft_to_nextmain")
                    argmax = next(task for task in tasks if task.op_type == "ArgMaxV2")
                    copied = next(task for task in tasks if task.name.startswith("aclnnInplaceCopy_ViewCopy"))
                    if invalid == "argmax_dtype_unknown":
                        argmax.output_dtypes = ""
                    elif invalid == "copy_input_dtype_unknown":
                        copied.input_dtypes = ""
                    elif invalid == "copy_output_dtype_unknown":
                        copied.output_dtypes = ""
                    elif invalid == "copy_input_dtype_mismatch":
                        copied.input_dtypes = copied.input_dtypes.replace("INT64", "INT32", 1)
                    elif invalid == "copy_output_dtype_mismatch":
                        copied.output_dtypes = "INT32"
                    else:
                        shapes = copied.input_shapes.strip('"').split(";")
                        shapes[4] = str(int(shapes[4]) + 1)
                        copied.input_shapes = '"' + ";".join(shapes) + '"'
                    self.assert_unknown_window(window, tasks, case)

    def test_dp9_required_next_main_preparation_cannot_be_missing_or_reordered(self):
        for round_id in ("hc-3", "hc-107"):
            for invalid in ("missing_expand", "missing_slots", "reordered_position_prepare"):
                with self.subTest(round_id=round_id, invalid=invalid):
                    window, tasks, case = self.window_fixture(round_id, "draft_to_nextmain")
                    if invalid.startswith("missing_"):
                        name = ("_expand_idx_mapping_kernel" if invalid == "missing_expand"
                                else "_compute_slot_mappings_kernel")
                        tasks.remove(next(task for task in tasks if task.name == name))
                    else:
                        position = next(task for task in tasks if task.name == "_prepare_pos_seq_lens_kernel")
                        slots = next(task for task in tasks if task.name == "_compute_slot_mappings_kernel")
                        duration = position.end_ns - position.start_ns
                        position.start_ns = slots.end_ns + 1
                        position.end_ns = position.start_ns + duration
                    self.assert_unknown_window(window, tasks, case)

    def test_dp9_optional_kv_work_when_present_precedes_expand_in_source_order(self):
        for invalid in ("zero_after_expand", "apply_after_expand", "zero_after_apply"):
            with self.subTest(invalid=invalid):
                window, tasks, case = self.window_fixture("hc-3", "draft_to_nextmain")
                expand = next(task for task in tasks if task.name == "_expand_idx_mapping_kernel")
                apply = next(task for task in tasks if task.name == "_apply_write_kernel")
                if invalid == "apply_after_expand":
                    moved, following = apply, expand
                else:
                    moved = next(task for task in tasks if task.name == "_zero_kv_blocks_kernel")
                    following = apply if invalid == "zero_after_apply" else expand
                duration = moved.end_ns - moved.start_ns
                moved.start_ns = following.end_ns + 100
                moved.end_ns = moved.start_ns + duration
                self.assert_unknown_window(window, tasks, case)

    def test_dp9_unknown_metadata_does_not_inherit_the_same_position_stage(self):
        for round_id in ("hc-3", "hc-107"):
            for kind in ("main_to_draft", "draft_to_nextmain"):
                with self.subTest(round_id=round_id, kind=kind):
                    window, tasks, case = self.window_fixture(round_id, kind)
                    found = 0
                    for task in tasks:
                        if "MixedQuantSparseFlashMlaMetadata" in task.name:
                            task.name = task.op_type = "unsupported_attention_metadata"
                            found += 1
                    self.assertGreater(found, 0)
                    self.assert_unknown_window(window, tasks, case)


class ResidualSourceTests(unittest.TestCase):
    """Host origins follow explicit links, including submissions before U."""
    stage_fixture = ResidualStageTests.stage_fixture
    matched_capture = ResidualStageTests.matched_capture

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="ep128-host-origin-")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)

    def source_fixture(self, *, shared=False, graph=False, ambiguous=False, origin_mode="linked"):
        capture, measured = self.matched_capture()
        targets = [task for task in capture["tasks"]
                   if task.name == "aclnnMatmul_MatMulCommon_MatMulV2"
                   and task.start_ns < RESIDUAL_EPOCH + 20000]
        if shared:
            targets.insert(0, next(task for task in capture["tasks"]
                                  if task.name == "RmsNorm" and task.start_ns < targets[0].start_ns))
        connection = 269508 if graph else 2 ** 53 + 123
        correlation, flow_id = 2 ** 53 + 321, str(2 ** 60 + 17)
        events = [event for event in residual_trace_events(capture)
                  if event.get("ph") == "M" or event.get("pid") == 900]
        def event(name, start, end, pid, tid, category=None, args=None):
            item = {"ph": "X", "name": name, "pid": pid, "tid": tid,
                    "ts": _residual_us_string(RESIDUAL_EPOCH + start),
                    "dur": _residual_us_string(end - start), "args": args or {}}
            if category:
                item["cat"] = category
            return item
        def flow(phase, category, identity, stamp, pid, tid, name):
            return {"ph": phase, "cat": category, "id": identity, "name": name,
                    "ts": _residual_us_string(stamp), "pid": pid, "tid": tid, "bp": "e"}
        for task in targets:
            events.append(event(task.name, task.start_ns - RESIDUAL_EPOCH, task.end_ns - RESIDUAL_EPOCH,
                900, 1, "kernel", {"Task Type": "AI_CORE", "Physic Stream Id": "1",
                "Task Id": task.task_id, "Model Id": "12", "connection_id": connection}))
        # Deliberately place flow finishes before starts in JSON array order.
        finishes = [flow("f", "HostToDevice", flow_id, task.start_ns, 900, 1, "acl_to_npu") for task in targets]
        if graph:
            # A nearby graph launch without a matching origin ID is not proof.
            events.append(event("aclmdlRIExecuteAsync", 10500, 11300, 408831, 1171))
            events = finishes + events
        else:
            events += [event("aten::linear", -50000, -40000, 399, 399, "cpu_op",
                             {"Sequence number": -1, "Fwd thread id": 0}),
                       event("npu::vocabulary_projection", -48000, -42000, 399, 399, "cpu_op"),
                       event("Enqueue@vocabulary_projection", -47000, -46500, 399, 399,
                             args={"correlation_id": correlation}),
                       event("Dequeue@vocabulary_projection", -45000, -39000, 399, 1171,
                             args={"correlation_id": correlation}),
                       event("CANN_PARENT", -44800, -39200, 408831, 1171,
                             args={"connection_id": connection + 999}),
                       event("Node@launch", -44000, -43500, 408831, 1171,
                             args={"connection_id": connection}),
                       event("UNRELATED_SIMULTANEOUS_CPU", 11000, 11500, 399, 400, "cpu_op"),
                       event("WRONG_REUSED_ASYNC_ORIGIN", -53000, -51000, 399, 777, "cpu_op")]
            if ambiguous:
                events.append(event("Node@launch", -44500, -43400, 408831, 1171,
                                    args={"connection_id": connection}))
            finishes += [flow("f", "async_task_queue", correlation, RESIDUAL_EPOCH - 44900,
                               399, 1171, "enqueue_to_dequeue")]
            # One reused async_npu ID has two origins and two same-time ends.
            # The explicit queue and device thread disambiguate the right one.
            async_id = targets[0].start_ns
            finishes += [flow("f", "async_npu", async_id, targets[0].start_ns, 900, tid, "torch_to_npu")
                         for tid in (37, 1)]
            starts = [flow("s", "HostToDevice", flow_id, RESIDUAL_EPOCH - 43800,
                           408831, 1171, "acl_to_npu"),
                      flow("s", "async_task_queue", correlation, RESIDUAL_EPOCH - 46900,
                           399, 399, "enqueue_to_dequeue"),
                      flow("s", "async_npu", async_id, RESIDUAL_EPOCH - 52000, 399, 777, "torch_to_npu"),
                      flow("s", "async_npu", async_id, RESIDUAL_EPOCH - 48000, 399, 399, "torch_to_npu")]
            events = finishes + events + starts
        if origin_mode.startswith("future_"):
            # Keep only the right async candidate so failure must be causal,
            # not an unrelated ambiguity from the reused-ID regression.
            events = [item for item in events if not (item.get("cat") == "async_npu"
                      and (item.get("tid") == 777 or item.get("ph") == "f" and item.get("tid") == 37))]
            for item in events:
                if origin_mode == "future_node" and item.get("name") == "Node@launch":
                    item["ts"] = _residual_us_string(targets[0].start_ns + 1000)
                elif origin_mode == "future_enqueue" and item.get("name", "").startswith("Enqueue@"):
                    item["ts"] = _residual_us_string(RESIDUAL_EPOCH - 38000)
            if origin_mode == "future_cpu":
                events = [item for item in events if item.get("name") not in ("Node@launch", "CANN_PARENT")]
                for item in events:
                    if item.get("name") == "aten::linear":
                        item["ts"] = _residual_us_string(targets[0].start_ns + 1000)
                    elif item.get("name") == "npu::vocabulary_projection":
                        item["ts"] = _residual_us_string(targets[0].start_ns + 2000)
                    elif item.get("ph") == "s" and item.get("cat") == "async_npu":
                        item["ts"] = _residual_us_string(targets[0].start_ns + 2000)
        elif origin_mode == "bad_queue_flow":
            for item in events:
                if item.get("ph") == "s" and item.get("cat") == "async_task_queue":
                    item["tid"] = 777  # Explicit flow contradicts the correlation pair.
        elif origin_mode == "no_queue_flow":
            events = [event for event in events if event.get("cat") != "async_task_queue"]
        elif origin_mode == "cpu_only":
            events = [event for event in events if event.get("name") not in ("Node@launch", "CANN_PARENT")
                      and event.get("cat") != "async_npu"]
            for task in targets:
                events.insert(0, flow("f", "async_npu", task.start_ns, task.start_ns, 900, 1, "torch_to_npu"))
                events.append(flow("s", "async_npu", task.start_ns, RESIDUAL_EPOCH - 48000,
                                   399, 399, "torch_to_npu"))
        elif origin_mode == "api_only":
            events = [event for event in events if event.get("cat") not in ("cpu_op", "async_npu", "async_task_queue")
                      and not event.get("name", "").startswith(("Enqueue@", "Dequeue@"))]
        events += [{"ph": "M", "name": "process_name", "pid": pid, "args": {"name": name}}
                   for pid, name in ((399, "Python"), (408831, "CANN"))]
        (self.directory / "trace_view.json").write_text(json.dumps(events), encoding="utf-8")
        result = residual.residual_analyze_capture(capture, measured, self.directory / "origin",
                                                 case={"draft_layers": 3, "draft_tokens": 7})
        with Path(result["source_links_csv"]).open(encoding="utf-8-sig", newline="") as stream:
            links = list(csv.DictReader(stream))
        selected = [row for row in links if int(row["row_index"]) in {task.row_index for task in targets}]
        self.assertEqual(len(selected), len(targets))
        return result, selected, targets, connection, correlation

    def test_prequeued_cpu_origin_uses_explicit_flow_and_preserves_large_ids(self):
        result, links, targets, connection, correlation = self.source_fixture()
        row = links[0]
        self.assertEqual(row["status"], "LINKED", row)
        self.assertEqual(row["connection_id"], str(connection))
        self.assertEqual(row["correlation_id"], str(correlation))
        self.assertLess(int(row["cpu_end_ns"]), RESIDUAL_EPOCH + 10000)
        self.assertEqual(row["cpu_leaf_name"], "npu::vocabulary_projection")
        self.assertIn("aten::linear", row["cpu_ancestors_json"])
        self.assertEqual(int(row["source_before_window_ns"]), 60000)
        self.assertEqual(int(row["queue_wait_ns"]), 1500)
        self.assertEqual(int(row["submit_to_device_ns"]), 55200)
        self.assertIn(str(2 ** 60 + 17), row["flow_ids_json"])
        self.assertTrue(row["enqueue_event_index"] and row["dequeue_event_index"] and row["node_event_index"])
        self.assertNotEqual(row["cpu_name"], "UNRELATED_SIMULTANEOUS_CPU")
        stage = next(row for row in result["stage_summary"] if row["stage_key"] == "main_logits")
        self.assertEqual(stage["source_status"], "PARTIAL")
        self.assertGreater(stage["source_cpu_union_ms_mean"], 0)
        self.assertGreater(stage["host_span_union_ms_mean"], 0)
        # Source duration before U and simultaneous coverage are different facts.
        self.assertNotEqual(stage["source_cpu_union_ms_mean"], stage["host_span_union_ms_mean"])

    def test_host_scan_normalizes_each_valid_x_event_once_without_losing_early_sources(self):
        original_iterator = residual.residual_iter_trace_events
        original_args = residual.residual_args
        original_host_rows = residual.residual_host_rows
        in_host_scan, events, calls = False, {}, {}

        def observed_iterator(*args, **kwargs):
            for event in original_iterator(*args, **kwargs):
                if in_host_scan:
                    # Retain each object so Python cannot recycle an identity.
                    events[id(event)] = event
                yield event

        def counted_args(event):
            if in_host_scan:
                calls[id(event)] = calls.get(id(event), 0) + 1
            return original_args(event)

        def observed_host_rows(*args, **kwargs):
            nonlocal in_host_scan
            in_host_scan = True
            try:
                return original_host_rows(*args, **kwargs)
            finally:
                in_host_scan = False

        with patch.object(residual, "residual_iter_trace_events", observed_iterator), \
             patch.object(residual, "residual_args", counted_args), \
             patch.object(residual, "residual_host_rows", observed_host_rows):
            result, links, _, _, _ = self.source_fixture()
        timed = {identity: event for identity, event in events.items()
                 if event.get("ph") == "X" and Decimal(event["dur"]) > 0}
        self.assertGreater(len(timed), 3)
        self.assertEqual({identity: calls.get(identity, 0) for identity in timed},
                         dict.fromkeys(timed, 1))
        self.assertEqual(result["trace"]["process_names"]["408831"], "CANN")
        self.assertEqual(links[0]["status"], "LINKED", links)
        self.assertLess(int(links[0]["cpu_end_ns"]), RESIDUAL_EPOCH + 10000)
        self.assertEqual(int(links[0]["queue_wait_ns"]), 1500)
        self.assertEqual(int(links[0]["submit_to_device_ns"]), 55200)

    def test_one_api_with_multiple_device_finishes_does_not_duplicate_cpu_or_api_time(self):
        _, single, _, _, _ = self.source_fixture()
        result, shared, _, _, _ = self.source_fixture(shared=True)
        self.assertTrue(all(row["status"] == "LINKED" for row in shared), shared)
        self.assertEqual(len({row["node_event_index"] for row in shared}), 1)
        self.assertEqual(len({row["cpu_event_index"] for row in shared}), 1)
        stage = next(row for row in result["stage_summary"] if row["stage_key"] == "main_logits")
        self.assertEqual(stage["source_cpu_tasks_total"], 2)
        self.assertEqual(stage["source_api_tasks_total"], 2)
        self.assertEqual(stage["source_status"], "COMPLETE")
        expected_cpu = (int(single[0]["cpu_end_ns"]) - int(single[0]["cpu_start_ns"])) / 1e6
        expected_api = (int(single[0]["node_end_ns"]) - int(single[0]["node_start_ns"])) / 1e6
        self.assertAlmostEqual(stage["source_cpu_union_ms_mean"], expected_cpu)
        self.assertAlmostEqual(stage["source_api_union_ms_mean"], expected_api)

    def test_graph_finish_without_start_preserves_unknown_origin_instead_of_sharing_fake_cpu(self):
        result, links, _, _, _ = self.source_fixture(shared=True, graph=True)
        self.assertTrue(all(row["status"] == "GRAPH_SHARED_NO_ORIGIN" for row in links), links)
        self.assertTrue(all(not row["cpu_event_index"] and row["reason"] for row in links))
        stage = next(row for row in result["stage_summary"] if row["stage_key"] == "main_logits")
        self.assertEqual(stage["source_graph_tasks_total"], 2)
        self.assertEqual(stage["source_cpu_tasks_total"], 0)
        self.assertIsNone(stage["source_cpu_union_ms_mean"])
        self.assertIsNone(stage["source_api_union_ms_mean"])

    def test_repeated_connection_without_unique_origin_is_ambiguous(self):
        result, links, _, _, _ = self.source_fixture(ambiguous=True)
        self.assertEqual(links[0]["status"], "AMBIGUOUS", links)
        self.assertTrue(links[0]["reason"])
        self.assertFalse(links[0]["cpu_event_index"])
        stage = next(row for row in result["stage_summary"] if row["stage_key"] == "main_logits")
        self.assertEqual(stage["source_ambiguous_tasks_total"], 1)
        self.assertIsNone(stage["source_cpu_union_ms_mean"])


    def test_cpu_only_and_api_only_are_partial_with_unknown_complementary_time(self):
        for mode, status in (("cpu_only", "CPU_ONLY"), ("api_only", "API_ONLY")):
            with self.subTest(mode=mode):
                result, links, _, _, _ = self.source_fixture(shared=True, origin_mode=mode)
                self.assertTrue(all(row["status"] == status for row in links), links)
                self.assertEqual(result["source_status"], "PARTIAL")
                stage = next(row for row in result["stage_summary"] if row["stage_key"] == "main_logits")
                self.assertEqual(stage["source_status"], "PARTIAL")
                present = "cpu" if mode == "cpu_only" else "api"
                absent = "api" if mode == "cpu_only" else "cpu"
                self.assertEqual(stage["source_" + present + "_tasks_total"], 2)
                self.assertEqual(stage["source_" + absent + "_tasks_total"], 0)
                self.assertGreater(stage["source_" + present + "_union_ms_mean"], 0)
                self.assertIsNone(stage["source_" + absent + "_union_ms_mean"])

    def test_queue_flow_cannot_contradict_pair_but_missing_flow_keeps_explicit_correlation(self):
        for mode, expected in (("bad_queue_flow", "AMBIGUOUS"), ("no_queue_flow", "LINKED")):
            with self.subTest(mode=mode):
                _, links, _, _, _ = self.source_fixture(origin_mode=mode)
                self.assertEqual(links[0]["status"], expected, links)
                if mode == "bad_queue_flow":
                    self.assertIn("queue flow", links[0]["reason"])
                    self.assertFalse(links[0]["cpu_event_index"])
                else:
                    self.assertEqual(links[0]["cpu_leaf_name"], "npu::vocabulary_projection")
                    self.assertEqual(int(links[0]["queue_wait_ns"]), 1500)

    def test_future_api_cpu_or_enqueue_cannot_be_a_causal_origin(self):
        for mode in ("future_node", "future_cpu", "future_enqueue"):
            with self.subTest(mode=mode):
                result, links, _, _, _ = self.source_fixture(origin_mode=mode)
                self.assertEqual(links[0]["status"], "AMBIGUOUS", links)
                self.assertIn("causal order", links[0]["reason"])
                self.assertFalse(links[0]["cpu_event_index"])
                self.assertFalse(links[0]["node_event_index"])
                stage = next(row for row in result["stage_summary"] if row["stage_key"] == "main_logits")
                self.assertIsNone(stage["source_cpu_union_ms_mean"])
                self.assertIsNone(stage["source_api_union_ms_mean"])

    def test_missing_semantic_stage_round_cannot_publish_source_observed_only_mean(self):
        window, tasks, _ = self.stage_fixture()
        stage = residual.residual_stage_plan(window, tasks, {"draft_layers": 3})[1]
        stage.update(recorded_busy_ms=stage["duration_ms"], no_recorded_ms=0,
                     kernel_sum_ms=stage["duration_ms"], task_count=2, source_status="COMPLETE")
        stage.update(dict.fromkeys(residual.RESIDUAL_HOST_NUMBERS, 0))
        stage.update(dict.fromkeys(residual.RESIDUAL_SOURCE_NUMBERS, .001))
        stage.update(source_device_tasks=2, source_cpu_tasks=2, source_api_tasks=2,
                     source_queue_tasks=2, source_graph_tasks=0, source_ambiguous_tasks=0)
        windows = [window, {**window, "round_id": "hc-2"}]
        summary = residual.residual_stage_summary([stage], windows, 0)[0]
        self.assertEqual(summary["rounds"], 2)
        self.assertEqual(summary["matched_windows"], 1)
        self.assertEqual(summary["source_status"], "PARTIAL")
        self.assertEqual(summary["source_cpu_tasks_total"], 2)  # Known subset remains visible.
        self.assertIsNone(summary["duration_ms_mean"])
        for field in residual.RESIDUAL_SOURCE_NUMBERS:
            self.assertIsNone(summary[field + "_mean"], field)


class TraceStreamTests(unittest.TestCase):
    def test_whitespace_utf8_chunk_boundaries_wrapper_skip_and_trailing_data(self):
        events = [{"ph": "X", "ts": "1790000000000000.007", "dur": 1.125,
                   "name": "中文🙂", "args": {"nested": ["escaped\\\"}]", {"a": 1}]}},
                  {"ph": "M", "name": "process_name", "pid": 20, "args": {"name": "CANN"}}]
        expected = json.loads(json.dumps(events), parse_float=Decimal)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "trace.json"
            # The old character scanner accepted all str.isspace characters;
            # preserve that behavior, including Unicode whitespace outside JSON.
            for whitespace in (" \r\n\t", "\u00a0\u2003\u2028\u2029", "\x1c\x85"):
                padding = whitespace * 11
                content = ("\ufeff" + padding + "{" + padding + '"before"' + padding + ":" + padding +
                           json.dumps({"skip": ["中文🙂" * 20, {"traceEvents": ["wrong"]}]}, ensure_ascii=False) +
                           padding + "," + padding + '"traceEvents"' + padding + ":" + padding +
                           json.dumps(events, ensure_ascii=False) + padding + "," + padding + '"after":' +
                           json.dumps([{"skip": "quoted ] } \\\" 中文🙂" * 20}], ensure_ascii=False) +
                           padding + "}" + padding)
                path.write_text(content, encoding="utf-8")
                for chunk in (1, 2, 3, 7, 64, 2048):
                    with self.subTest(whitespace=repr(whitespace), chunk=chunk):
                        stats = {}
                        self.assertEqual(list(residual.residual_iter_trace_events(path, chunk_size=chunk, stats=stats)), expected)
                        self.assertEqual(stats["bytes_read"], len(content.encode("utf-8")))
                        self.assertEqual(stats["sha256"], hashlib.sha256(content.encode("utf-8")).hexdigest())
                path.write_text(content + "{}", encoding="utf-8")
                with self.subTest(trailing=repr(whitespace)), self.assertRaisesRegex(ValueError, "trailing data"):
                    list(residual.residual_iter_trace_events(path, chunk_size=1))

    def test_chunks_unicode_escaped_wrapper_and_exact_decimal(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "trace.json"
            events = [{"ph": "X", "ts": "1790000000000000.007", "dur": 1.125,
                       "name": '中文, ] { "traceEvents"', "args": {"escaped": "\\\"}"}}]
            wrapper = {"ignored": {"traceEvents": ["wrong"]}, "traceEvents": events,
                       "stackFrames": {"large": "中文{}[]\\\"" * 2000}}
            path.write_text(json.dumps(wrapper, ensure_ascii=False), encoding="utf-8")
            stats = {}
            original_open = Path.open

            class Guarded:
                def __init__(self, stream):
                    self.stream = stream

                def read(self, size):
                    if not 0 < size <= 7:
                        raise AssertionError("unbounded trace read")
                    return self.stream.read(size)

                def close(self):
                    self.stream.close()

            def guarded_open(selected, *args, **kwargs):
                stream = original_open(selected, *args, **kwargs)
                return Guarded(stream) if selected == path else stream

            with patch.object(Path, "open", guarded_open), patch.object(residual.json, "load", side_effect=AssertionError("json.load forbidden")):
                output = list(residual.residual_iter_trace_events(path, chunk_size=7, stats=stats))
            self.assertEqual(output[0]["name"], events[0]["name"])
            self.assertEqual(output[0]["dur"], Decimal("1.125"))
            self.assertEqual(output[0]["ts"], "1790000000000000.007")
            self.assertLess(stats["parser_max_buffer_chars"], 512)
            self.assertEqual(stats["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())

    def test_truncation_malformed_or_missing_array_is_reported(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "trace.json"
            for data in ('[{"ph":"X"}', '{"traceEvents":[{"a":1}]', '{"other":[]}', '[{},]', '[{}]garbage'):
                path.write_text(data, encoding="utf-8")
                with self.subTest(data=data), self.assertRaises(ValueError):
                    list(residual.residual_iter_trace_events(path, chunk_size=3))


if __name__ == "__main__":
    unittest.main()
