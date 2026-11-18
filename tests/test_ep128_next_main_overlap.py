"""Exercise fixed-rank U diagnostics with exact clocks and explicit trace links."""

from copy import copy
import csv
from decimal import Decimal
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import profile_ep128 as core
import test_ep128_v2_measure as fixtures


ROOT = Path(__file__).resolve().parents[1]
EPOCH = fixtures.RESIDUAL_EPOCH


def us(ns):
    return str(Decimal(ns) / 1000)


def device_task(name, start, end, row, model_id, task_id=None, stream="1"):
    return SimpleNamespace(
        name=name, op_type=name, start_ns=EPOCH + start, end_ns=EPOCH + end,
        row_index=row, row_indices=[row], stream_id=stream,
        task_id=str(row if task_id is None else task_id), model_id=model_id, device_id="0",
        wait_ns=0, input_shapes="", output_shapes="", input_dtypes="", output_dtypes="",
        task_type="AI_VECTOR_CORE",
    )


def event(name, start, end, pid, tid, offset, *, args=None, kind=None):
    result = {"ph": "X", "name": name, "ts": us(EPOCH + start - offset),
              "dur": us(end - start), "pid": pid, "tid": tid, "args": args or {}}
    if kind:
        result["cat"] = kind
    return result


def trace_source(events, name, start, end, connection, tasks, offset):
    """One CPU submission may launch multiple device tasks via one Node."""
    enq = end - 30
    events.extend((
        event(name, start, end, 10, connection, offset, kind="cpu_op"),
        event("Enqueue@" + name, enq, enq + 5, 10, connection, offset,
              args={"correlation_id": connection}),
        event("Dequeue@" + name, end + 5, end + 28, 10, connection + 100, offset,
              args={"correlation_id": connection}),
        event("Node@launch", end + 10, end + 20, 20, connection + 100, offset,
              args={"connection_id": connection}),
    ))
    for task in tasks:
        events.append(event(task.name, task.start_ns - EPOCH, task.end_ns - EPOCH, 900, 1, offset,
                            args={"Task Type": "AI_CORE", "Physic Stream Id": task.stream_id,
                                  "Task Id": task.task_id, "Model Id": task.model_id,
                                  "connection_id": connection}))


class NextMainTraceTests(unittest.TestCase):
    stage_fixture = fixtures.ResidualStageTests.stage_fixture
    matched_capture = fixtures.ResidualStageTests.matched_capture

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="ep128-next-main-")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)

    def run_fixture(self, *, with_trace=True, duplicate_start=False, resolve_metadata=True):
        capture, measured = self.matched_capture()
        tasks = capture["tasks"]
        for task in tasks:
            start = task.start_ns - EPOCH
            if start == 17_200 or 30_000 <= start < 36_080:
                task.model_id = "49"
        tasks.extend((
            device_task("draft_graph_start", 16_900, 17_000, 11, "49", 0),
            device_task("HcPre", 20_000, 20_200, 12, "49"),
            device_task("draft_forward", 22_000, 22_900, 13, "49"),
            device_task("HcPost", 29_800, 30_000, 14, "49"),
            device_task("next_main_metadata_copy", 38_320, 38_400, 15, "12"),
            device_task("next_main_metadata_scatter", 38_480, 38_560, 16, "12"),
        ))
        if duplicate_start:
            tasks.append(device_task("ambiguous_graph_start", 16_910, 16_990, 17, "49", 0))
        if not resolve_metadata:
            tasks[:] = [task for task in tasks if task.start_ns != EPOCH + 38_000]
        tasks.sort(key=lambda task: (task.start_ns, task.end_ns))
        if with_trace:
            offset = 123_456
            events = [item for item in fixtures.residual_trace_events(capture, offset)
                      if item.get("ph") == "M" or item.get("pid") == 900]
            by_start = {task.start_ns - EPOCH: task for task in tasks}
            trace_source(events, "host_before_draft", 300, 600, 31, [by_start[36_400]], offset)
            trace_source(events, "host_crossing_draft_start", 16_000, 18_000, 32,
                         [by_start[37_600], by_start[38_320]], offset)
            if resolve_metadata:
                trace_source(events, "host_during_draft", 17_500, 22_300, 33, [by_start[38_000]], offset)
            trace_source(events, "host_crossing_draft_end", 35_500, 36_700, 34, [by_start[38_480]], offset)
            events.append(event("MEMCPY_ASYNC", 36_840, 37_100, 900, 7, offset,
                                args={"Task Type": "PCIE_DMA_SQE", "Physic Stream Id": 7}))
            (self.directory / "trace_view.json").write_text(json.dumps(events), encoding="utf-8")
        # Reuse the existing residual collector, including full trace alignment and source-link resolution.
        case = {"draft_layers": 3, "draft_tokens": 7}
        result = core.residual_analyze_capture(capture, measured, self.directory / "details", case=case,
                                               include_next_main_overlap=True)
        return {row["stage_key"]: row for row in result["next_main_overlap"]}

    def test_real_alignment_then_device_and_host_unions_use_exact_boundaries(self):
        rows = self.run_fixture()
        inputs, metadata = rows["main_inputs"], rows["main_metadata"]

        # Stage 13 is [36400,37600), with 7 disjoint 80ns tasks. The
        # 260ns DMA interval covers one task and contributes 180ns new busy.
        self.assertEqual(inputs["trace_status"], "ALIGNED_X_EVENTS")
        self.assertAlmostEqual(inputs["stage_wall_ms"], 1_200 / 1e6)
        self.assertAlmostEqual(inputs["device_busy_csv_ms"], 560 / 1e6)
        self.assertAlmostEqual(inputs["dma_additional_ms"], 180 / 1e6)
        self.assertAlmostEqual(inputs["device_busy_with_dma_ms"], 740 / 1e6)
        self.assertAlmostEqual(inputs["device_uncovered_after_dma_ms"], 460 / 1e6)
        self.assertEqual(inputs["host_in_draft_graph_ms"], 0)
        self.assertAlmostEqual(inputs["host_before_draft_graph_ms"], 300 / 1e6)

        self.assertEqual(metadata["draft_graph_status"], "MATCHED")
        self.assertEqual(int(metadata["draft_graph_start_ns"]) - EPOCH, 16_900)
        self.assertEqual(int(metadata["draft_graph_end_ns"]) - EPOCH, 35_840)
        self.assertEqual((metadata["linked_device_tasks"], metadata["linked_host_events"]), (4, 3))
        # The first and second CPU spans partially overlap, and the first is
        # linked to two tasks. Merge them once; then clip against each device scope.
        self.assertAlmostEqual(metadata["host_linked_ms"], 7_500 / 1e6)
        self.assertAlmostEqual(metadata["host_in_draft_hc_ms"], 2_300 / 1e6)
        self.assertAlmostEqual(metadata["host_in_draft_graph_ms"], 5_740 / 1e6)
        self.assertAlmostEqual(metadata["host_on_draft_tasks_ms"], 1_100 / 1e6)
        self.assertAlmostEqual(metadata["host_before_draft_graph_ms"], 900 / 1e6)
        self.assertAlmostEqual(metadata["host_after_draft_graph_ms"], 860 / 1e6)

    def test_ambiguous_native_graph_keeps_hc_overlap_and_unknown_full_graph_overlap(self):
        metadata = self.run_fixture(duplicate_start=True)["main_metadata"]
        self.assertEqual(metadata["draft_graph_status"], "UNRESOLVED")
        self.assertIsNone(metadata["host_in_draft_graph_ms"])
        self.assertIsNone(metadata["host_on_draft_tasks_ms"])
        self.assertAlmostEqual(metadata["host_in_draft_hc_ms"], 2_300 / 1e6)

    def test_missing_trace_leaves_host_and_dma_unknown(self):
        for row in self.run_fixture(with_trace=False).values():
            self.assertEqual(row["stage_status"], "MATCHED")
            self.assertEqual(row["host_status"], "UNAVAILABLE")
            self.assertGreater(row["device_busy_csv_ms"], 0)
            self.assertIsNone(row["dma_additional_ms"])
            self.assertIsNone(row["device_uncovered_after_dma_ms"])
            self.assertIsNone(row["host_linked_ms"])
            self.assertIsNone(row["host_in_draft_graph_ms"])

    def test_missing_semantic_metadata_anchor_never_fabricates_stage_values(self):
        for row in self.run_fixture(resolve_metadata=False).values():
            self.assertEqual(row["stage_status"], "UNRESOLVED")
            self.assertIsNone(row["stage_wall_ms"])
            self.assertIsNone(row["device_busy_csv_ms"])
            self.assertIsNone(row["host_in_draft_graph_ms"])


class FixedRankCLITests(unittest.TestCase):
    def native_capture(self, directory, bs, steps, *, with_trace=True):
        # Use the repository's full HC fixture and overlay the ordered DSpark
        # transitions. Scale the transition nanoseconds to fit in the HC gaps.
        stage_fixture = fixtures.ResidualStageTests.stage_fixture
        _, head, _ = stage_fixture(self, "main_to_draft")
        _, tail, _ = stage_fixture(self, "draft_to_nextmain")
        head.append(device_task("draft_graph_start", 88, 89, 0, "49", 0))
        if steps > 7:
            markov = [task for task in tail if 68 <= task.start_ns - EPOCH <= 72]
            for task in tail:
                if task.start_ns - EPOCH >= 76:
                    task.start_ns += (steps - 7) * 8
                    task.end_ns += (steps - 7) * 8
            for position in range(1, steps - 7 + 1):
                for task in markov:
                    extra = copy(task)
                    extra.start_ns += position * 8
                    extra.end_ns += position * 8
                    tail.append(extra)

        cycles = 8

        def add_transitions(rows, _records):
            rows[:] = [row for row in rows if row["Name"] != "LMHead"]
            for cycle in range(cycles):
                for tasks, anchor in ((head, 782_000), (tail, 1_082_000)):
                    for task in tasks:
                        start = task.start_ns - EPOCH
                        draft = (tasks is head and start >= 88
                                 or tasks is tail and start < 76 + (steps - 7) * 8)
                        rows.append({
                            "Name": task.name, "Type": task.op_type,
                            "Start Time(us)": us(fixtures.EPOCH_NS + cycle * 1_100_000 + anchor + start * 80),
                            "Duration(us)": us((task.end_ns - task.start_ns) * 80),
                            "Device_id": "0", "Model ID": "49" if draft else "47",
                            "Stream ID": task.stream_id, "Task ID": "0" if task.task_id == "0" else str(len(rows) + 100),
                            "Input Shapes": task.input_shapes, "Output Shapes": task.output_shapes,
                            "Input Data Types": task.input_dtypes, "Output Data Types": task.output_dtypes,
                            "Wait Time(us)": "0", "OP State": "GRAPH",
                        })

        capture, _ = fixtures.fixture(directory, rounds=cycles, changes=add_transitions)
        if not with_trace:
            return capture["path"]
        offset = 423
        events = [{"ph": "M", "name": "process_name", "pid": pid, "args": {"name": name}}
                  for pid, name in ((10, "Python"), (20, "CANN"), (900, "Ascend Hardware"))]
        anchors = [task for task in capture["tasks"] if task.name == "MoeDistributeDispatchV2"][:3]
        for task in anchors:
            events.append(event(task.name, task.start_ns - EPOCH, task.end_ns - EPOCH, 900, 1, offset,
                                args={"Task Type": "AI_CORE", "Physic Stream Id": task.stream_id,
                                      "Task Id": task.task_id, "Model Id": task.model_id}))
        for index, task in enumerate(t for t in capture["tasks"] if t.name == "_compute_slot_mappings_kernel"):
            start = fixtures.EPOCH_NS - EPOCH + index * 1_100_000 + 803_000
            trace_source(events, "next_main_slots", start, start + bs * 1000 + steps * 25,
                         50 + index, [task], offset)
        (directory / "trace_view.json").write_text(json.dumps(events), encoding="utf-8")
        return capture["path"]

    def test_mixed_bs_and_steps_are_separate_and_missing_rank_never_falls_back(self):
        with tempfile.TemporaryDirectory(prefix="ep128-next-main-cli-") as temporary:
            base = Path(temporary)
            config = json.loads((ROOT / "profile_ep128_cases.example.json").read_text(encoding="utf-8"))
            config["sample_model"] = fixtures.MODEL
            config["cases"] = []
            for bs, steps, rank in ((4, 7, "dp0"), (8, 7, "dp0"), (8, 15, "dp0"), (16, 15, "dp1")):
                case = f"bs{bs}-s{steps}"
                directory = base / case / rank
                directory.mkdir(parents=True)
                capture_path = self.native_capture(directory, bs, steps, with_trace=bs != 4)
                config["cases"].append({"name": case, "bs": bs, "draft_tokens": steps,
                                        "synthetic_acceptance_length": 5.2,
                                        "inputs": [{"rank": rank, "path": capture_path}]})
            manifest, out = base / "cases.json", base / "out"
            manifest.write_text(json.dumps(config), encoding="utf-8")
            result = subprocess.run(
                [sys.executable, "-S", str(ROOT / "profile_ep128.py"), str(manifest),
                 "--out", str(out), "--u-overlap-rank", "dp0"],
                capture_output=True, text=True, encoding="utf-8", timeout=45,
                env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            )
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
            report = json.loads((out / "u_next_main_overlap.json").read_text(encoding="utf-8"))
            self.assertEqual(report["rank"], "dp0")
            summary = {(entry["bs"], entry["S"], entry["stage_number"]): entry
                       for entry in report["summary"]}
            self.assertEqual(set(summary), {(bs, s, n) for bs, s in ((4, 7), (8, 7), (8, 15), (16, 15))
                                            for n in (13, 14)})
            for (bs, steps, stage), entry in summary.items():
                self.assertEqual(entry["rank"], "dp0")
                if (bs, steps) == (16, 15):
                    self.assertEqual(entry["rounds"], 0)
                    self.assertTrue(entry["case_error"])
                else:
                    self.assertGreater(entry["rounds"], 0)
                    self.assertIsNone(entry["case_error"])
                    self.assertEqual(entry["matched_rounds"], entry["rounds"])
                if bs == 8 and stage == 13:
                    self.assertAlmostEqual(entry["host_in_draft_graph_ms_mean"], (8_000 + steps * 25) / 1e6)
                    self.assertEqual(entry["host_in_draft_graph_ms_n"], entry["rounds"])
                else:
                    self.assertIsNone(entry["host_in_draft_graph_ms_mean"])
            with (out / "u_next_main_overlap_rounds.csv").open(encoding="utf-8-sig", newline="") as stream:
                rounds = list(csv.DictReader(stream))
            self.assertEqual({(row["case"], row["rank"]) for row in rounds},
                             {("bs4-s7", "dp0"), ("bs8-s7", "dp0"), ("bs8-s15", "dp0")})
            self.assertTrue(all(row["host_in_draft_graph_ms"] == ""
                                for row in rounds if row["case"] == "bs4-s7"))
            self.assertTrue((out / "u_next_main_overlap_summary.csv").is_file())
            self.assertFalse((out / "model_architecture_report.json").exists())


if __name__ == "__main__":
    unittest.main()
