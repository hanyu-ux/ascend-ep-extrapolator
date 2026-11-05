"""Exact whole-window accounting checked against an independent integer grid."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import sqlite3
import tempfile
import unittest

import profile_host as host


def event(name, start, stop, *, device=False, pid=None, tid=1, tasktype=""):
    return {"name": name, "ph": "X", "ts": start, "dur": stop - start,
            "pid": pid if pid is not None else 100 if device else 1,
            "tid": tid, "cat": "kernel" if device else "cpu_op",
            "args": {"Task Type": tasktype}}


def grid_oracle(spans, start, end, device_pid="100"):
    """Visit each ns independently; no sweep, event heap or production classifier."""
    primary, details, states = Counter(), Counter(), Counter()
    for stamp in range(start, end):
        device_classes = {category for side, category, left, right, pid in spans
                          if side == "device" and pid == device_pid and left <= stamp < right}
        host_classes = {category for side, category, left, right, _ in spans
                        if side == "host" and left <= stamp < right}
        if device_classes:
            key = next(iter(device_classes)) if len(device_classes) == 1 else "device_overlap"
            components = tuple(sorted(device_classes))
        elif host_classes:
            key = next(iter(host_classes)) if len(host_classes) == 1 else "host_overlap"
            components = tuple(sorted(host_classes))
        else:
            key, components = "unobserved", ()
        primary[key] += 1
        details[key, components] += 1
        states["both" if device_classes and host_classes else "device_only" if device_classes
               else "host_only" if host_classes else "neither"] += 1
    return primary, details, states


class ExclusiveLedgerTests(unittest.TestCase):
    def analyze(self, events, **options):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.json"
            path.write_text(json.dumps({"traceEvents": events}), encoding="utf-8")
            return host.analyze_trace(path, time_unit="ns", **options)

    def assert_closed(self, ledger):
        self.assertTrue(ledger["closed"])
        self.assertEqual(ledger["closure_error_ns"], 0)
        self.assertEqual(ledger["end_ns"] - ledger["start_ns"], ledger["total_ns"])
        for collection in ("breakdown", "details", "coverage_states"):
            self.assertEqual(sum(row["duration_ns"] for row in ledger[collection]), ledger["total_ns"])
            self.assertAlmostEqual(sum(row["share"] for row in ledger[collection]), 1)
            for row in ledger[collection]:
                self.assertIsInstance(row["duration_ns"], int)
                self.assertEqual(row["duration_ms"], row["duration_ns"] / 1e6)
        parents = defaultdict(int)
        for row in ledger["details"]:
            parents[row["parent_key"]] += row["duration_ns"]
        self.assertEqual(dict(parents), {row["key"]: row["duration_ns"] for row in ledger["breakdown"]})

    def assert_oracle(self, ledger, spans):
        self.assert_closed(ledger)
        primary, details, states = grid_oracle(spans, ledger["start_ns"], ledger["end_ns"], ledger["device_pid"])
        self.assertEqual({row["key"]: row["duration_ns"] for row in ledger["breakdown"]}, dict(primary))
        self.assertEqual({(row["parent_key"], tuple(row["components"])): row["duration_ns"]
                          for row in ledger["details"]}, dict(details))
        self.assertEqual({row["key"]: row["duration_ns"] for row in ledger["coverage_states"]},
                         {key: states[key] for key in ("both", "device_only", "host_only", "neither")})

    def test_overlapping_classes_keep_exact_combinations_and_host_is_not_added_twice(self):
        events = [event("host", 0, 20), event("FlashAttention", 2, 10, device=True),
                  event("HcclAllReduce", 6, 14, device=True, tid=2),
                  event("MatMul", 8, 12, device=True, tid=3)]
        spans = [("host", "host_only_other", 0, 20, "1"),
                 ("device", "device_attention", 2, 10, "100"),
                 ("device", "device_communication", 6, 14, "100"),
                 ("device", "device_matmul", 8, 12, "100")]
        report = self.analyze(events)
        ledger = report["time_ledgers"][0]
        self.assert_oracle(ledger, spans)
        rows = {row["key"]: row["duration_ns"] for row in ledger["breakdown"]}
        self.assertEqual(rows, {"host_only_other": 8, "device_attention": 4,
                                "device_communication": 2, "device_overlap": 6})
        self.assertEqual(sum(row["duration_ns"] for row in ledger["coverage_states"]
                             if row["key"] in ("both", "host_only")), 20)
        self.assertEqual(report["summary"]["host_wall_union_ms"], .000020)
        self.assertEqual(report["devices"][0]["recorded_union_ms"], .000012)

    def test_touching_nested_and_duplicated_same_class_is_one_continuous_segment(self):
        events = [event("host", 0, 8), event("FlashAttention", 2, 4, device=True),
                  event("FAInfer", 4, 6, device=True), event("FlashAttention", 3, 5, device=True),
                  event("FlashAttention", 3, 5, device=True, tid=2)]
        ledger = self.analyze(events)["time_ledgers"][0]
        self.assert_closed(ledger)
        rows = {row["key"]: row for row in ledger["breakdown"]}
        self.assertEqual(set(rows), {"device_attention", "host_only_other"})
        self.assertEqual(rows["device_attention"]["duration_ns"], 4)
        self.assertEqual(rows["device_attention"]["segment_count"], 1)
        self.assertEqual(rows["host_only_other"]["segment_count"], 2)

    def test_host_nested_different_categories_have_overlap_instead_of_arbitrary_priority(self):
        ledger = self.analyze([event("outer", 0, 12), event("aten::copy_", 2, 10),
                               event("aclrtSynchronizeStream", 4, 6)])["time_ledgers"][0]
        self.assert_oracle(ledger, [("host", "host_only_other", 0, 12, "1"),
                                   ("host", "host_only_copy", 2, 10, "1"),
                                   ("host", "host_only_sync", 4, 6, "1")])
        self.assertIsNone(ledger["device_pid"])
        self.assertEqual({row["key"]: row["duration_ns"] for row in ledger["breakdown"]},
                         {"host_only_other": 4, "host_overlap": 8})

    def test_explicit_window_is_clipped_and_retains_unobserved_edges(self):
        events = [event("FlashAttention", 0, 10, device=True),
                  event("HcclAllReduce", 20, 40, device=True), event("gloo:recv", 8, 15)]
        ledger = self.analyze(events, start_us="0.005", end_us="0.035")["time_ledgers"][0]
        self.assert_oracle(ledger, [("device", "device_attention", 0, 10, "100"),
                                   ("device", "device_communication", 20, 40, "100"),
                                   ("host", "host_only_communication", 8, 15, "1")])
        self.assertEqual(next(row["duration_ns"] for row in ledger["breakdown"] if row["key"] == "unobserved"), 5)

    def test_empty_raw_event_window_is_wholly_unobserved(self):
        with sqlite3.connect(":memory:") as db:
            db.execute("CREATE TABLE events(id INTEGER PRIMARY KEY,name TEXT,tasktype TEXT,kind TEXT,start INTEGER,end INTEGER,pid TEXT)")
            ledger = host._exclusive_time_ledgers(db, 3, 17, ["100"])[0]
        self.assert_closed(ledger)
        self.assertEqual([(row["key"], row["duration_ns"]) for row in ledger["breakdown"]], [("unobserved", 14)])

    def test_explicit_window_between_events_is_closed_even_when_both_sides_are_absent(self):
        report = self.analyze([event("known", 0, 1), event("MatMul", 30, 31, device=True)],
                              start_us="0.010", end_us="0.020")
        ledger = report["time_ledgers"][0]
        self.assertIsNone(ledger["device_pid"])
        self.assert_closed(ledger)
        self.assertEqual([(row["key"], row["duration_ns"]) for row in ledger["breakdown"]], [("unobserved", 10)])

    def test_unknown_trace_events_are_retained_as_unobserved_time_not_guessed_host(self):
        unknown = event("unknown", 2, 18, pid=3)
        unknown["cat"] = "custom"
        ledger = self.analyze([event("known", 0, 1), unknown, event("known", 19, 20)])["time_ledgers"][0]
        self.assert_closed(ledger)
        self.assertEqual({row["key"]: row["duration_ns"] for row in ledger["breakdown"]},
                         {"host_only_other": 2, "unobserved": 18})

    def test_distinct_device_ledgers_each_cover_full_window_without_combining_devices(self):
        spans = [("host", "host_only_other", 0, 20, "1"),
                 ("device", "device_attention", 0, 10, "100"),
                 ("device", "device_matmul", 5, 20, "200")]
        report = self.analyze([event("host", 0, 20), event("FlashAttention", 0, 10, device=True),
                               event("MatMul", 5, 20, device=True, pid=200)])
        self.assertEqual({row["device_pid"] for row in report["time_ledgers"]}, {"100", "200"})
        for ledger in report["time_ledgers"]:
            self.assertEqual(ledger["total_ns"], 20)
            self.assert_oracle(ledger, spans)

    def test_explicit_pid_filters_apply_before_accounting(self):
        report = self.analyze([event("host", 0, 20), event("gloo:send", 0, 20, pid=2),
                               event("MatMul", 2, 8, device=True),
                               event("FlashAttention", 0, 20, device=True, pid=200)],
                              host_pids=[1], device_pids=[100])
        self.assertEqual(len(report["time_ledgers"]), 1)
        self.assert_oracle(report["time_ledgers"][0], [("host", "host_only_other", 0, 20, "1"),
                                                      ("device", "device_matmul", 2, 8, "100")])

    def test_begin_end_events_participate_in_full_window_ledger(self):
        events = [event("host", 0, 12),
                  {"ph": "B", "ts": 2, "pid": 100, "tid": 1, "cat": "kernel", "name": "MatMul"},
                  {"ph": "E", "ts": 10, "pid": 100, "tid": 1}]
        ledger = self.analyze(events)["time_ledgers"][0]
        self.assert_oracle(ledger, [("host", "host_only_other", 0, 12, "1"),
                                   ("device", "device_matmul", 2, 10, "100")])

    def test_zero_duration_tasks_do_not_make_false_overlaps_or_segments(self):
        events = [event("host", 0, 10), event("MatMul", 2, 8, device=True),
                  event("FlashAttention", 4, 4, device=True)]
        ledger = self.analyze(events)["time_ledgers"][0]
        self.assert_oracle(ledger, [("host", "host_only_other", 0, 10, "1"),
                                   ("device", "device_matmul", 2, 8, "100")])

    def test_mc2_duplicate_aicpu_and_comm_views_are_one_fused_class(self):
        ledger = self.analyze([event("host", 0, 10),
                               event("MatmulReduceScatter", 2, 8, device=True, tasktype="AI_CPU"),
                               event("MatmulReduceScatter", 2, 8, device=True, tid=2, tasktype="COMMUNICATION")])["time_ledgers"][0]
        self.assert_closed(ledger)
        self.assertEqual({row["key"]: row["duration_ns"] for row in ledger["breakdown"]},
                         {"host_only_other": 4, "device_fused_matmul_communication": 6})

    def test_subnanosecond_endpoint_quantization_is_kept_exact_in_ledger(self):
        raw = '''{"traceEvents":[
          {"name":"host","ph":"X","pid":1,"tid":1,"cat":"cpu_op","ts":1000000000000000,"dur":0.01},
          {"name":"FlashAttention","ph":"X","pid":100,"tid":1,"cat":"kernel","ts":1000000000000000.00025,"dur":0.00125}
        ]}'''
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.json"
            path.write_text(raw, encoding="utf-8")
            report = host.analyze_trace(path)
        ledger = report["time_ledgers"][0]
        self.assert_closed(ledger)
        self.assertEqual(ledger["total_ns"], 10)
        self.assertEqual({row["key"]: row["duration_ns"] for row in ledger["breakdown"]},
                         {"host_only_other": 8, "device_attention": 2})
        self.assertTrue(any(row["code"] == "SUB_NS_QUANTIZATION" for row in report["diagnostics"]))

    def test_ledger_uses_raw_events_beyond_truncated_gap_and_link_tables(self):
        count = host.DETAIL_LIMIT + 7
        events = [event("host", 0, count * 2)] + [event("MatMul", index * 2, index * 2 + 1, device=True)
                                                   for index in range(count)]
        report = self.analyze(events)
        self.assertTrue(report["detail_limits"]["gaps_truncated"])
        self.assertTrue(report["detail_limits"]["links_truncated"])
        ledger = report["time_ledgers"][0]
        self.assert_closed(ledger)
        self.assertEqual({row["key"]: row["duration_ns"] for row in ledger["breakdown"]},
                         {"host_only_other": count, "device_matmul": count})

    def test_deterministic_random_intervals_match_per_ns_independent_oracle(self):
        rng = random.Random(705)
        mappings = [("FlashAttention", "device_attention", True), ("MatMul", "device_matmul", True),
                    ("HcclAllReduce", "device_communication", True), ("host", "host_only_other", False),
                    ("aten::copy_", "host_only_copy", False), ("gloo:send", "host_only_communication", False)]
        for _ in range(8):
            events, spans = [event("host", 0, 40)], [("host", "host_only_other", 0, 40, "1")]
            for index in range(25):
                name, category, device = rng.choice(mappings)
                left, right = sorted((rng.randrange(41), rng.randrange(41)))
                events.append(event(name, left, right, device=device, tid=index + 2))
                spans.append(("device" if device else "host", category, left, right, "100" if device else "1"))
            self.assert_oracle(self.analyze(events)["time_ledgers"][0], spans)

    def test_classification_retains_ownership_limits_and_distinguishes_control(self):
        expected = {"QkvRopeCache": "device_qkv_rope_cache", "FAInfer": "device_attention",
                    "GenericBlockSparseAttention": "device_sparse_attention",
                    "GenericBlockSparseAttentionMetadata": "device_indexer_metadata",
                    "IndexerRegionScore": "device_sparse_indexer", "UnionSparseMeta": "device_indexer_metadata",
                    "_csa_prefill_update_kernel": "device_indexer_metadata",
                    "_append_current_region_kernel": "device_indexer_metadata",
                    "_decode_summary_layout_kernel": "device_indexer_metadata",
                    "MoeGatingTopK": "device_moe_route", "GroupedMatmul": "device_expert_matmul",
                    "AddRmsNormDynamicQuant": "device_norm_activation", "SwiGlu": "device_norm_activation",
                    "MatMul": "device_matmul", "NotifyWait": "device_control", "EventRecord": "device_control",
                    "CONDITION_SQE": "device_control",
                    "MemcpyAsync": "device_dma", "HcclAllGather": "device_communication",
                    "UnrecognizedKernel": "device_other"}
        for name, category in expected.items():
            with self.subTest(name=name):
                self.assertEqual(host._ledger_class(name, "", "device"), category)
        self.assertEqual(host._ledger_class("DevicePrepare", "AI_CPU", "aicpu"), "device_aicpu_other")
        self.assertEqual(host._ledger_class("MatMul", "AI_CPU", "aicpu"), "device_matmul")
        self.assertNotIn("SWA", host._LEDGER_CLASSES["device_sparse_attention"])


if __name__ == "__main__":
    unittest.main()
