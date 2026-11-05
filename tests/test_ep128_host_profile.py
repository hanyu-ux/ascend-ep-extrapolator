"""Independent synthetic timing checks for the model-independent Host reader."""
import csv
from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET
from zipfile import ZipFile

import profile_host as host
from host_profile_report import write_reports


def event(name, start, duration, *, pid=1, tid=10, category="cpu_op", **extra):
    return dict(name=name, ph="X", ts=start, dur=duration, pid=pid, tid=tid,
                cat=category, **extra)


def metadata(pid, name, *, tid=0, thread=False):
    return dict(ph="M", name="thread_name" if thread else "process_name",
                pid=pid, tid=tid, args={"name": name})


def flow(phase, start, *, pid, tid, token=1, category="async_npu"):
    return dict(ph=phase, ts=start, pid=pid, tid=tid, id=token, cat=category)


class HostProfileTests(unittest.TestCase):
    def analyze(self, events, **options):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace_view.json"
            path.write_text(json.dumps({"traceEvents": events}, ensure_ascii=False), encoding="utf-8")
            return host.analyze_trace(path, **options)

    def hotspot(self, report, name, pid=None):
        rows = [row for row in report["hotspots"] if row["name"] == name
                and (pid is None or str(row["pid"]) == str(pid))]
        self.assertEqual(len(rows), 1, (name, rows))
        return rows[0]

    def test_nested_inclusive_and_self_have_independent_hand_arithmetic(self):
        report = self.analyze([event("outer", 0, 100), event("inner", 20, 40)])
        outer, inner = self.hotspot(report, "outer"), self.hotspot(report, "inner")
        self.assertAlmostEqual(outer["inclusive_sum_ms"], .1)
        self.assertAlmostEqual(outer["self_sum_ms"], .06)
        self.assertAlmostEqual(inner["self_sum_ms"], .04)
        summary = report["summary"]
        self.assertAlmostEqual(summary["host_inclusive_sum_ms"], .14)
        self.assertAlmostEqual(summary["host_self_sum_ms"], .1)
        self.assertAlmostEqual(summary["host_wall_union_ms"], .1)
        self.assertIsNone(summary["cpu_running_ms"])
        self.assertIsNone(summary["cpu_utilization_percent"])

    def test_grandchildren_are_not_subtracted_twice(self):
        report = self.analyze([event("parent", 0, 100), event("child", 10, 70),
                               event("grandchild", 20, 30)])
        self.assertAlmostEqual(self.hotspot(report, "parent")["self_sum_ms"], .03)
        self.assertAlmostEqual(self.hotspot(report, "child")["self_sum_ms"], .04)
        self.assertAlmostEqual(report["summary"]["host_self_sum_ms"], .1)

    def test_other_thread_does_not_reduce_self_or_become_cpu_utilization(self):
        report = self.analyze([event("main", 0, 100, tid=10),
                               event("worker", 20, 40, tid=11)])
        self.assertAlmostEqual(self.hotspot(report, "main")["self_sum_ms"], .1)
        self.assertAlmostEqual(report["summary"]["host_wall_union_ms"], .1)
        self.assertAlmostEqual(report["summary"]["host_thread_union_sum_ms"], .14)
        self.assertAlmostEqual(report["summary"]["host_self_sum_ms"], .14)
        self.assertEqual(report["summary"]["host_threads"], 2)

    def test_identical_tid_in_distinct_processes_is_not_the_same_thread(self):
        report = self.analyze([event("parent", 0, 100, pid=1),
                               event("child", 20, 40, pid=1),
                               event("other_process", 10, 20, pid=2)])
        self.assertEqual(report["summary"]["host_threads"], 2)
        self.assertAlmostEqual(self.hotspot(report, "parent")["self_sum_ms"], .06)
        self.assertAlmostEqual(self.hotspot(report, "other_process")["self_sum_ms"], .02)
        self.assertAlmostEqual(report["summary"]["host_thread_union_sum_ms"], .12)

    def test_crossing_intervals_have_unavailable_self_not_guessed_parent(self):
        report = self.analyze([event("cross_a", 0, 70), event("cross_b", 40, 60)])
        for name in ("cross_a", "cross_b"):
            row = self.hotspot(report, name)
            self.assertIsNone(row["self_sum_ms"])
            self.assertEqual(row["self_unavailable_calls"], 1)
        self.assertEqual(report["summary"]["self_unavailable_events"], 2)
        self.assertAlmostEqual(report["summary"]["host_wall_union_ms"], .1)

    def test_coextensive_and_duplicate_intervals_do_not_fabricate_self(self):
        for second_name in ("second", "first"):
            with self.subTest(second_name=second_name):
                report = self.analyze([event("first", 0, 100), event(second_name, 0, 100)])
                self.assertEqual(report["summary"]["self_unavailable_events"], 2)
                self.assertIsNone(self.hotspot(report, "first")["self_sum_ms"])

    def test_a_group_containing_an_ambiguous_call_does_not_show_partial_self_as_total(self):
        report = self.analyze([event("repeat", 0, 60), event("cross", 40, 60),
                               event("repeat", 200, 20)])
        row = self.hotspot(report, "repeat")
        self.assertEqual(row["calls"], 2)
        self.assertAlmostEqual(row["inclusive_sum_ms"], .08)
        self.assertIsNone(row["self_sum_ms"])
        self.assertEqual(row["self_unavailable_calls"], 1)

    def test_touching_siblings_are_valid(self):
        report = self.analyze([event("outer", 0, 100), event("first", 10, 20),
                               event("second", 30, 30)])
        self.assertAlmostEqual(self.hotspot(report, "outer")["self_sum_ms"], .05)
        self.assertEqual(report["summary"]["self_unavailable_events"], 0)

    def test_window_clips_full_tree_instead_of_reassigning_child_time(self):
        report = self.analyze([event("outer", 0, 100), event("inner", 20, 60)],
                              start_us=10, end_us=50)
        self.assertAlmostEqual(report["scope"]["wall_ms"], .04)
        self.assertAlmostEqual(self.hotspot(report, "outer")["inclusive_sum_ms"], .04)
        self.assertAlmostEqual(self.hotspot(report, "outer")["self_sum_ms"], .01)
        self.assertAlmostEqual(self.hotspot(report, "inner")["self_sum_ms"], .03)
        self.assertEqual(int(report["scope"]["start_ns"]), 10000)
        self.assertEqual(int(report["scope"]["end_ns"]), 50000)

    def test_window_inside_child_does_not_make_original_tree_ambiguous(self):
        report = self.analyze([event("outer", 0, 100), event("inner", 20, 60)],
                              start_us=30, end_us=60)
        self.assertAlmostEqual(self.hotspot(report, "outer")["self_sum_ms"], 0)
        self.assertAlmostEqual(self.hotspot(report, "inner")["self_sum_ms"], .03)
        self.assertEqual(report["summary"]["self_unavailable_events"], 0)

    def test_invalid_window_arguments_are_rejected(self):
        for options in ({"start_us": 1}, {"end_us": 2},
                        {"start_us": 2, "end_us": 1}, {"start_us": 1, "end_us": 1}):
            with self.subTest(options=options):
                with self.assertRaises((ValueError, TypeError)):
                    self.analyze([event("work", 0, 10)], **options)

    def test_begin_end_pairs_are_nested_per_process_and_thread(self):
        events = [dict(ph="B", ts=0, pid=1, tid=10, cat="cpu_op", name="outer"),
                  dict(ph="B", ts=20, pid=1, tid=10, cat="cpu_op", name="inner"),
                  dict(ph="B", ts=0, pid=2, tid=10, cat="cpu_op", name="independent"),
                  dict(ph="E", ts=30, pid=2, tid=10),
                  dict(ph="E", ts=60, pid=1, tid=10),
                  dict(ph="E", ts=100, pid=1, tid=10)]
        report = self.analyze(events)
        self.assertAlmostEqual(self.hotspot(report, "outer")["self_sum_ms"], .06)
        self.assertAlmostEqual(self.hotspot(report, "inner")["self_sum_ms"], .04)
        self.assertAlmostEqual(self.hotspot(report, "independent")["self_sum_ms"], .03)

    def test_unmatched_begin_end_is_reported_without_creating_unknown_duration(self):
        events = [event("valid", 0, 100),
                  dict(ph="E", ts=10, pid=2, tid=10),
                  dict(ph="B", ts=20, pid=2, tid=10, cat="cpu_op", name="unfinished")]
        report = self.analyze(events)
        self.assertEqual(report["summary"]["host_events"], 1)
        self.assertTrue(any("unmatched" in (row["code"] + " " + row["message"]).lower()
                            for row in report["diagnostics"]))

    def test_process_and_thread_metadata_can_follow_timed_events(self):
        report = self.analyze([event("aten::linear", 0, 100),
                               metadata(1, "Customer worker"),
                               metadata(1, "Main thread", tid=10, thread=True)])
        row = self.hotspot(report, "aten::linear")
        self.assertEqual(row["process"], "Customer worker")
        self.assertEqual(row["thread"], "Main thread")

    def test_explicit_host_pid_accepts_an_unclassified_named_annotation(self):
        report = self.analyze([event("customer_prepare", 0, 100, pid=77, category="custom")],
                              host_pids=[77])
        self.assertEqual(report["summary"]["host_events"], 1)
        self.assertEqual(str(self.hotspot(report, "customer_prepare")["pid"]), "77")

    def test_device_and_aicpu_are_kept_out_of_host_cpu(self):
        report = self.analyze([event("python", 0, 100),
                               event("MatMul", 20, 30, pid=100, category="kernel"),
                               event("DevicePrepare", 60, 20, pid=100, category="aicpu")])
        self.assertEqual(report["summary"]["host_events"], 1)
        self.assertEqual(report["summary"]["device_events"], 2)
        self.assertAlmostEqual(report["devices"][0]["recorded_union_ms"], .05)
        self.assertAlmostEqual(report["devices"][0]["aicpu_union_ms"], .02)

    def test_device_task_type_argument_identifies_aicpu_and_dma(self):
        report = self.analyze([event("host", 0, 100),
                               event("TaskOne", 10, 20, pid=100, category="kernel",
                                     args={"Task Type": "AI_CPU", "Stream Id": 1}),
                               event("TaskTwo", 50, 30, pid=100, category="kernel",
                                     args={"Task Type": "MEMCPY_ASYNC", "Stream Id": 2})])
        self.assertEqual(report["summary"]["host_events"], 1)
        self.assertAlmostEqual(report["devices"][0]["aicpu_union_ms"], .02)
        self.assertAlmostEqual(report["devices"][0]["dma_union_ms"], .03)

    def test_generated_overlap_analysis_is_not_an_extra_device_or_host_task(self):
        report = self.analyze([event("host", 0, 100),
                               event("real_kernel", 10, 20, pid=100, category="kernel"),
                               event("Computing", 10, 20, pid=200, category=""),
                               event("Free", 30, 70, pid=200, category=""),
                               metadata(200, "Overlap Analysis")])
        self.assertEqual(report["summary"]["host_events"], 1)
        self.assertEqual(report["summary"]["device_events"], 1)

    def test_all_device_streams_are_unioned_before_no_recorded_gap_detection(self):
        report = self.analyze([event("host", 0, 100),
                               event("kernel_a", 10, 40, pid=100, tid=1, category="kernel"),
                               event("kernel_b", 40, 40, pid=100, tid=2, category="kernel")])
        device = report["devices"][0]
        self.assertAlmostEqual(device["recorded_union_ms"], .07)
        self.assertAlmostEqual(device["recorded_sum_ms"], .08)
        self.assertAlmostEqual(device["no_recorded_ms"], .03)
        self.assertAlmostEqual(device["host_overlap_in_gaps_ms"], .03)
        self.assertAlmostEqual(sum(row["duration_ms"] for row in report["gaps"]), .03)
        self.assertFalse(any("idle" == str(row.get("kind", "")).lower() for row in report["gaps"]))

    def test_distinct_device_processes_have_distinct_gap_summaries(self):
        report = self.analyze([event("host", 0, 100),
                               event("device_one", 0, 100, pid=100, category="kernel"),
                               event("device_two", 0, 50, pid=200, category="kernel")])
        devices = {str(row["pid"]): row for row in report["devices"]}
        self.assertEqual(set(devices), {"100", "200"})
        self.assertAlmostEqual(devices["100"]["no_recorded_ms"], 0)
        self.assertAlmostEqual(devices["200"]["no_recorded_ms"], .05)

    def test_sync_wall_remains_host_wall_and_does_not_prove_cpu_busy(self):
        report = self.analyze([event("aclrtSynchronizeStream", 0, 100, category="runtime"),
                               event("kernel", 0, 90, pid=100, category="kernel")])
        row = self.hotspot(report, "aclrtSynchronizeStream")
        self.assertEqual(row["kind"], "sync")
        self.assertAlmostEqual(row["inclusive_sum_ms"], .1)
        self.assertIsNone(report["summary"]["cpu_running_ms"])

    def test_absent_side_is_unavailable_instead_of_inventing_device_or_host_tasks(self):
        for events, host_count, device_count in (([event("host_only", 0, 100)], 1, 0),
                ([event("device_only", 0, 100, pid=100, category="kernel")], 0, 1)):
            with self.subTest(host_count=host_count):
                report = self.analyze(events)
                self.assertEqual(report["summary"]["host_events"], host_count)
                self.assertEqual(report["summary"]["device_events"], device_count)
                self.assertFalse(report["devices"] if not device_count else report["threads"])
                self.assertIn("host_status", report["coverage"])
                self.assertIn("device_status", report["coverage"])

    def test_single_main_model_needs_no_hc_draft_or_extrapolation_parameters(self):
        report = self.analyze([event("GQA_prepare", 0, 100),
                               event("SparseAttention", 50, 40, pid=100, category="kernel")],
                              model="Step5", hardware="950", main_only=True)
        self.assertEqual(report["summary"]["host_events"], 1)
        self.assertEqual(report["summary"]["device_events"], 1)
        self.assertNotIn("projection", report)
        self.assertNotIn("effective_tokens_per_round", report["summary"])

    def test_trace_time_units_are_converted_without_changing_explicit_window_units(self):
        for unit, multiplier in (("us", 1), ("ns", 1000), ("ms", .001)):
            with self.subTest(unit=unit):
                report = self.analyze([event("work", 10 * multiplier, 100 * multiplier)],
                                      time_unit=unit, start_us=20, end_us=80)
                self.assertAlmostEqual(self.hotspot(report, "work")["inclusive_sum_ms"], .06)
                self.assertEqual(int(report["scope"]["start_ns"]), 20000)

    def test_epoch_timestamp_fraction_is_not_lost_through_float(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "precise.json"
            path.write_text('{"traceEvents":[{"ph":"X","pid":1,"tid":1,"cat":"cpu_op",'
                            '"name":"one_ns","ts":1793591950793583.620,"dur":0.001}]}',
                            encoding="utf-8")
            report = host.analyze_trace(path)
        self.assertEqual(int(report["scope"]["start_ns"]), 1793591950793583620)
        self.assertEqual(int(report["scope"]["end_ns"]), 1793591950793583621)
        self.assertAlmostEqual(self.hotspot(report, "one_ns")["inclusive_sum_ms"], .000001)

    def test_malformed_events_are_diagnosed_and_not_counted_as_zero(self):
        report = self.analyze([event("good", 0, 100), event("bad_ts", "oops", 10),
                               event("negative_duration", 0, -1), event("bool_ts", True, 5)])
        self.assertEqual(report["summary"]["host_events"], 1)
        self.assertTrue(any(row["code"] == "INVALID_EVENT" and row["count"] >= 3
                            for row in report["diagnostics"]))

    def test_incomplete_json_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "broken.json"
            path.write_text('{"traceEvents":[{"ph":"X","ts":0', encoding="utf-8")
            with self.assertRaises(ValueError):
                host.analyze_trace(path)

    def test_no_usable_timed_events_is_rejected(self):
        with self.assertRaises(ValueError):
            self.analyze([metadata(1, "python")])

    def test_unique_connection_and_cpu_parent_provide_a_measured_source(self):
        report = self.analyze([event("aten::linear", 0, 30),
                               event("Node@launch", 10, 5, category="runtime",
                                     args={"connectionId": 42}),
                               event("MatMul", 50, 20, pid=100, category="kernel",
                                     args={"connectionId": 42})])
        link = report["links"][0]
        self.assertEqual(link["status"], "LINKED")
        self.assertEqual(link["cpu_name"], "aten::linear")
        self.assertEqual(link["api_name"], "Node@launch")
        self.assertAlmostEqual(link["submit_to_device_ms"], .04)
        self.assertEqual(report["coverage"]["source_status_counts"], {"LINKED": 1})

    def test_tid_match_alone_cannot_connect_distinct_process_cpu_and_api(self):
        report = self.analyze([event("unrelated_cpu", 0, 30, pid=1, tid=10),
                               event("Node@launch", 10, 5, pid=2, tid=10, category="runtime",
                                     args={"connectionId": 42}),
                               event("MatMul", 50, 20, pid=100, category="kernel",
                                     args={"connectionId": 42})])
        link = report["links"][0]
        self.assertIsNone(link["cpu_event_id"])
        self.assertIsNone(link["cpu_name"])
        self.assertNotEqual(link["status"], "LINKED")

    def test_explicit_async_flow_can_link_cpu_across_process_tracks(self):
        report = self.analyze([event("aten::linear", 0, 30, pid=1, tid=10),
                               event("MatMul", 50, 20, pid=100, tid=1, category="kernel"),
                               flow("s", 10, pid=1, tid=10),
                               flow("f", 50, pid=100, tid=1)])
        link = report["links"][0]
        self.assertEqual(link["cpu_name"], "aten::linear")
        self.assertIn(link["status"], ("LINKED", "CPU_ONLY"))
        self.assertIsNone(link["api_event_id"])

    def test_reused_connection_without_disambiguating_flow_cannot_pick_an_api(self):
        report = self.analyze([event("Node@first", 10, 5, pid=1, category="runtime",
                                     args={"connectionId": 42}),
                               event("Node@other_process", 20, 5, pid=2, category="runtime",
                                     args={"connectionId": 42}),
                               event("MatMul", 50, 20, pid=100, category="kernel",
                                     args={"connectionId": 42})])
        link = report["links"][0]
        self.assertEqual(link["status"], "AMBIGUOUS")
        self.assertIsNone(link["api_event_id"])
        self.assertIsNone(link["cpu_event_id"])
        self.assertIsNone(link["submit_to_device_ms"])

    def test_invalid_connection_sentinel_cannot_create_an_api_link(self):
        for sentinel in (-1, "18446744073709551615"):
            with self.subTest(sentinel=sentinel):
                report = self.analyze([event("Node@no_id", 0, 10, category="runtime",
                                             args={"connectionId": sentinel}),
                                       event("MatMul", 20, 10, pid=100, category="kernel",
                                             args={"connectionId": sentinel})])
                self.assertIsNone(report["links"][0]["api_event_id"])
                self.assertNotEqual(report["links"][0]["status"], "LINKED")

    def test_one_api_graph_fanout_does_not_invent_per_kernel_cpu_work(self):
        report = self.analyze([event("broad_python_scope", 0, 100),
                               event("Node@GraphLaunch", 10, 5, category="runtime",
                                     args={"connectionId": 42}),
                               event("MatMul", 20, 10, pid=100, category="kernel",
                                     args={"connectionId": 42}),
                               event("Attention", 30, 10, pid=100, category="kernel",
                                     args={"connectionId": 42})])
        self.assertEqual(report["summary"]["host_events"], 2)
        self.assertAlmostEqual(report["summary"]["host_inclusive_sum_ms"], .105)
        self.assertEqual(len(report["links"]), 2)
        self.assertTrue(all(row["cpu_event_id"] is None for row in report["links"]))
        self.assertTrue(all(row["status"] == "GRAPH_SHARED_NO_ORIGIN" for row in report["links"]))
        self.assertTrue(all(row["queue_wait_ms"] is None for row in report["links"]))
        self.assertTrue(all(row["submit_to_device_ms"] is None for row in report["links"]))

    def test_shared_async_npu_flow_is_not_a_per_kernel_graph_cpu_origin(self):
        report = self.analyze([event("broad_python_scope", 0, 100),
                               event("Node@GraphLaunch", 10, 5, category="runtime",
                                     args={"connectionId": 42}),
                               event("MatMul", 20, 10, pid=100, tid=1, category="kernel",
                                     args={"connectionId": 42}),
                               event("Attention", 40, 10, pid=100, tid=1, category="kernel",
                                     args={"connectionId": 42}),
                               flow("s", 5, pid=1, tid=10),
                               flow("f", 20, pid=100, tid=1),
                               flow("f", 40, pid=100, tid=1)])
        self.assertTrue(all(row["cpu_event_id"] is None for row in report["links"]))
        self.assertTrue(all(row["status"] == "GRAPH_SHARED_NO_ORIGIN" for row in report["links"]))

    def test_shared_graph_api_flow_does_not_become_a_per_kernel_cpu_origin(self):
        report = self.analyze([event("framework_forward", 0, 100, pid=1, tid=10),
                               event("Node@GraphLaunch", 10, 5, pid=1, tid=10, category="runtime",
                                     args={"connectionId": 42}),
                               event("MatMul", 30, 10, pid=100, tid=1, category="kernel",
                                     args={"connectionId": 42}),
                               event("Attention", 70, 10, pid=100, tid=1, category="kernel",
                                     args={"connectionId": 42}),
                               flow("s", 10, pid=1, tid=10, token=8, category="HostToDevice"),
                               flow("f", 30, pid=100, tid=1, token=8, category="HostToDevice"),
                               flow("f", 70, pid=100, tid=1, token=8, category="HostToDevice")])
        self.assertEqual(len(report["links"]), 2)
        for row in report["links"]:
            self.assertEqual(row["status"], "GRAPH_SHARED_NO_ORIGIN")
            self.assertIsNone(row["cpu_event_id"])
            self.assertIsNone(row["cpu_name"])
            self.assertIsNone(row["queue_wait_ms"])
            self.assertIsNone(row["submit_to_device_ms"])

    def test_flow_to_another_process_cpu_does_not_disambiguate_an_unrelated_api(self):
        report = self.analyze([event("unrelated_Node@launch", 10, 5, pid=1, tid=10,
                                     category="runtime", args={"connectionId": 42}),
                               event("actual_cpu", 0, 30, pid=2, tid=20),
                               event("first_device", 30, 10, pid=100, tid=1, category="kernel",
                                     args={"connectionId": 42}),
                               event("second_device", 50, 10, pid=200, tid=1, category="kernel",
                                     args={"connectionId": 42}),
                               flow("s", 20, pid=2, tid=20, token=9),
                               flow("f", 50, pid=200, tid=1, token=9)])
        target = next(row for row in report["links"] if str(row["device_pid"]) == "200")
        self.assertNotEqual(target["status"], "LINKED")
        self.assertIsNone(target["api_event_id"])
        self.assertIsNone(target["api_pid"])
        self.assertIsNone(target["submit_to_device_ms"])

    def test_absent_host_overlap_is_unknown_in_device_gaps_not_measured_zero(self):
        report = self.analyze([event("device_only", 20, 20, pid=100, category="kernel")],
                              start_us=0, end_us=100)
        self.assertEqual(report["summary"]["host_events"], 0)
        self.assertAlmostEqual(report["devices"][0]["no_recorded_ms"], .08)
        self.assertIsNone(report["devices"][0]["host_overlap_in_gaps_ms"])
        self.assertEqual(len(report["gaps"]), 2)
        for row in report["gaps"]:
            for field in ("host_overlap_ms", "sync_overlap_ms", "copy_overlap_ms", "api_overlap_ms"):
                self.assertIsNone(row[field], field)

    def queue_events(self, *, second_enqueue=False, cross_process=False, overlap=False):
        worker_pid = 2 if cross_process else 1
        events = [event("prepare_inputs", 0, 30, pid=1, tid=10),
                  event("Enqueue@launch", 10, 15 if overlap else 2, pid=1, tid=10,
                        category="runtime", args={"correlationId": 7}),
                  event("Dequeue@launch", 20, 20, pid=worker_pid, tid=11,
                        category="runtime", args={"correlationId": 7}),
                  event("Node@launch", 25, 2, pid=worker_pid, tid=11,
                        category="runtime", args={"connectionId": 42}),
                  event("MatMul", 50, 20, pid=100, tid=1, category="kernel",
                        args={"connectionId": 42})]
        if second_enqueue:
            events.append(event("Enqueue@reused", 13, 1, pid=1, tid=12,
                                category="runtime", args={"correlationId": 7}))
        if cross_process:
            events.extend([flow("s", 10, pid=1, tid=10, token=7, category="async_task_queue"),
                           flow("f", 20, pid=2, tid=11, token=7, category="async_task_queue")])
        return events

    def test_same_process_queue_correlation_gives_wait_and_cpu_origin(self):
        report = self.analyze(self.queue_events())
        link = report["links"][0]
        self.assertEqual(link["status"], "LINKED")
        self.assertEqual(link["cpu_name"], "prepare_inputs")
        self.assertAlmostEqual(link["queue_wait_ms"], .008)
        self.assertAlmostEqual(link["submit_to_device_ms"], .025)

    def test_reused_queue_correlation_in_same_process_is_ambiguous(self):
        report = self.analyze(self.queue_events(second_enqueue=True))
        self.assertEqual(report["links"][0]["status"], "AMBIGUOUS")
        self.assertIsNone(report["links"][0]["queue_wait_ms"])
        self.assertIsNone(report["links"][0]["cpu_event_id"])

    def test_overlapping_enqueue_dequeue_does_not_become_negative_wait(self):
        report = self.analyze(self.queue_events(overlap=True))
        self.assertIsNone(report["links"][0]["queue_wait_ms"])

    def test_queue_flow_resolves_origin_before_explicit_window_across_synthetic_pids(self):
        report = self.analyze(self.queue_events(cross_process=True), start_us=45, end_us=75)
        link = report["links"][0]
        self.assertEqual(link["status"], "LINKED")
        self.assertEqual(link["cpu_name"], "prepare_inputs")
        self.assertAlmostEqual(link["queue_wait_ms"], .008)
        self.assertIsNotNone(link["enqueue_event_id"])
        self.assertIsNotNone(link["dequeue_event_id"])

    def test_only_real_step_annotations_are_exposed_as_step_windows(self):
        report = self.analyze([event("ProfilerStep#3", 0, 100, category="user_annotation"),
                               event("Iteration#4", 150, 100, category="user_annotation"),
                               event("looks_like_step", 300, 100, category="user_annotation"),
                               event("ProfilerStep#5", 450, 100, category="cpu_op")])
        self.assertEqual([row["name"] for row in report["steps"]], ["ProfilerStep#3", "Iteration#4"])

    def test_detail_cap_does_not_cap_source_coverage_counts(self):
        events = [event("host", 0, 5000)]
        events.extend(event("device", 2 * index, 1, pid=100, category="kernel")
                      for index in range(1100))
        report = self.analyze(events)
        self.assertEqual(report["summary"]["device_events"], 1100)
        self.assertEqual(sum(report["coverage"]["source_status_counts"].values()), 1100)
        self.assertEqual(report["detail_limits"]["link_rows_total"], 1100)
        self.assertTrue(report["detail_limits"]["links_truncated"])
        self.assertLess(len(report["links"]), 1100)
        self.assertEqual(report["detail_limits"]["gap_rows_total"], 1100)
        self.assertTrue(report["detail_limits"]["gaps_truncated"])
        self.assertLess(len(report["gaps"]), 1100)
        self.assertAlmostEqual(report["devices"][0]["no_recorded_ms"], 3.9)
        self.assertAlmostEqual(report["devices"][0]["host_overlap_in_gaps_ms"], 3.9)

    def test_cli_writes_portable_main_only_report_and_optional_excel(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace_view.json"
            path.write_text(json.dumps([event("custom_prepare", 0, 100, pid=77, category="custom")]),
                            encoding="utf-8")
            out = Path(directory) / "reports"
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                result = host.main([str(path), "--out", str(out), "--host-pid", "77",
                                    "--model", "Step5", "--hardware", "950", "--main-only", "--excel"])
            self.assertEqual(result, 0)
            report = json.loads((out / "host_profile_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["context"]["model"], "Step5")
            self.assertEqual(report["context"]["hardware"], "950")
            self.assertTrue(report["context"]["main_only"])
            self.assertEqual(report["summary"]["host_events"], 1)
            self.assertTrue((out / "host_profile_report.html").is_file())
            self.assertTrue((out / "host_profile_report.xlsx").is_file())
            self.assertTrue((out / "host_hotspots.csv").is_file())

    def test_cli_invalid_input_returns_failure_without_report_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            path.write_text("{", encoding="utf-8")
            out = Path(directory) / "reports"
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                result = host.main([str(path), "--out", str(out)])
            self.assertEqual(result, 2)
            self.assertFalse(out.exists())


class HostReportTests(unittest.TestCase):
    def report(self):
        return {"summary": {"cpu_running_ms": None, "cpu_utilization_percent": None},
                "context": {"model": "Step5", "hardware": "950", "main_only": True},
                "hotspots": [dict(pid="1", tid="10", name='<script>alert("unsafe")</script>',
                    kind="cpu", category="cpu_op", calls=1, inclusive_sum_ms=.1,
                    self_sum_ms=.06, self_unavailable_calls=0, status="COMPLETE"),
                    dict(pid="1", tid="10", name='=HYPERLINK("https://invalid.example")',
                    kind="cpu", category="cpu_op", calls=1, inclusive_sum_ms=.04,
                    self_sum_ms=None, self_unavailable_calls=1, status="UNAVAILABLE")],
                "gaps": [dict(device_pid="100", start_ns=1760000000000000000,
                              end_ns=1760000000000010000, duration_ms=.01)],
                "links": [dict(device_event_id=1, status="UNLINKED", queue_wait_ms=None)],
                "coverage": {"source_status_counts": {"UNLINKED": 10}},
                "detail_limits": {"gaps_truncated": True, "gap_rows_total": 20, "gaps": 1,
                                  "links_truncated": True, "link_rows_total": 10, "links": 1}}

    def test_exports_preserve_unknown_and_escape_external_trace_text(self):
        report = self.report()
        with tempfile.TemporaryDirectory() as directory:
            paths = write_reports(report, directory, excel=True)
            text = Path(paths["html"]).read_text(encoding="utf-8")
            self.assertNotIn('<script>alert("unsafe")</script>', text)
            self.assertIn("&lt;script&gt;", text)
            self.assertIn("20", text)
            self.assertEqual(json.loads(Path(paths["json"]).read_text(encoding="utf-8")), report)
            with Path(paths["host_hotspots_csv"]).open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[1]["self_sum_ms"], "")
            self.assertEqual(rows[1]["name"], "'" + report["hotspots"][1]["name"])
            with Path(paths["device_gaps_csv"]).open(encoding="utf-8-sig", newline="") as handle:
                gap = next(csv.DictReader(handle))
            self.assertEqual(gap["detail_rows_total"], "20")
            self.assertEqual(gap["detail_rows_exported"], "1")
            self.assertEqual(gap["detail_sampled"], "True")
            with Path(paths["source_links_csv"]).open(encoding="utf-8-sig", newline="") as handle:
                link = next(csv.DictReader(handle))
            self.assertEqual(link["queue_wait_ms"], "")

    def test_xlsx_contains_native_horizontal_chart_and_preserves_literal_text(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = write_reports(self.report(), directory, excel=True)
            with ZipFile(paths["xlsx"]) as archive:
                self.assertIsNone(archive.testzip())
                charts = [name for name in archive.namelist()
                          if name.startswith("xl/charts/chart") and name.endswith(".xml")]
                self.assertTrue(charts)
                ns = {"c": "http://schemas.openxmlformats.org/drawingml/2006/chart"}
                root = ET.fromstring(archive.read(charts[0]))
                bar = root.find(".//c:barChart", ns)
                self.assertEqual(bar.find("c:barDir", ns).get("val"), "bar")
                values = [float(node.text) for node in bar.findall("c:ser/c:val/c:numRef/c:numCache/c:pt/c:v", ns)]
                self.assertEqual(values, [.06])
                sheet_ns = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
                all_text = []
                for name in archive.namelist():
                    if name.startswith("xl/worksheets/sheet") and name.endswith(".xml"):
                        sheet = ET.fromstring(archive.read(name))
                        self.assertFalse(sheet.findall(".//s:f", sheet_ns))
                        all_text.extend(node.text for node in sheet.findall(".//s:t", sheet_ns))
                self.assertIn('=HYPERLINK("https://invalid.example")', all_text)
                self.assertIn("1760000000000000000", all_text)

    def test_csv_keeps_full_aggregate_rows_when_html_preview_is_bounded(self):
        report = self.report()
        report["hotspots"] = [dict(report["hotspots"][0], name="call_" + str(index))
                              for index in range(75)]
        with tempfile.TemporaryDirectory() as directory:
            paths = write_reports(report, directory)
            with Path(paths["host_hotspots_csv"]).open(encoding="utf-8-sig", newline="") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 75)
            self.assertNotIn("xlsx", paths)

    def test_export_cannot_overwrite_a_source_named_like_an_output(self):
        for filename in ("host_profile_report.json", "threads.csv"):
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as directory:
                out = Path(directory)
                source = out / filename
                source.write_text("ORIGINAL TRACE MUST SURVIVE", encoding="utf-8")
                report = self.report()
                report["source"] = {"path": str(source)}
                with self.assertRaises(ValueError):
                    write_reports(report, out, excel=True)
                self.assertEqual(source.read_text(encoding="utf-8"), "ORIGINAL TRACE MUST SURVIVE")
                self.assertEqual(list(out.iterdir()), [source])

    def test_export_rejects_symlink_alias_of_source_before_writing_any_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "trace_view.json"
            source.write_text("ORIGINAL TRACE", encoding="utf-8")
            out = root / "reports"
            out.mkdir()
            alias = out / "threads.csv"
            try:
                alias.symlink_to(source)
            except (OSError, NotImplementedError):
                self.skipTest("symlink creation is unavailable on this platform")
            report = self.report()
            report["source"] = {"path": str(source)}
            with self.assertRaises(ValueError):
                write_reports(report, out)
            self.assertEqual(source.read_text(encoding="utf-8"), "ORIGINAL TRACE")
            self.assertEqual(list(out.iterdir()), [alias])

    def test_export_rejects_hardlink_alias_of_source_before_writing_any_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "trace_view.json"
            source.write_text("ORIGINAL TRACE", encoding="utf-8")
            out = root / "reports"
            out.mkdir()
            alias = out / "host_profile_report.json"
            try:
                os.link(source, alias)
            except (OSError, NotImplementedError):
                self.skipTest("hardlink creation is unavailable on this platform")
            report = self.report()
            report["source"] = {"path": str(source)}
            with self.assertRaises(ValueError):
                write_reports(report, out)
            self.assertEqual(source.read_text(encoding="utf-8"), "ORIGINAL TRACE")
            self.assertEqual(list(out.iterdir()), [alias])


if __name__ == "__main__":
    unittest.main()
