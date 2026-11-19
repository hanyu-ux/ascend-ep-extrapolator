"""Attribute saved Host windows without crossing process or event identities."""

from decimal import Decimal
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import profile_ep128 as core
import ep128_host_function_trace as probe


EPOCH = 1790000000000000007


def event(name, start, end, pid=10, tid=7, category="cpu_op"):
    return {"ph": "X", "name": name, "pid": pid, "tid": tid, "cat": category,
            "ts": str(Decimal(EPOCH + start) / 1000),
            "dur": str(Decimal(end - start) / 1000)}


class HostFunctionTraceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="ep128 host function ")
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "trace view.json"
        self.events = [
            {"ph": "M", "name": "process_name", "pid": pid, "args": {"name": name}}
            for pid, name in ((10, "Python"), (20, "CANN"), (11, "Python"))
        ]

    def write_trace(self):
        payload = json.dumps({"traceEvents": self.events}).encode("utf-8")
        self.path.write_bytes(payload)
        return {"status": "ALIGNED_X_EVENTS", "clock_offset_ns": "0",
                "sha256": hashlib.sha256(payload).hexdigest()}

    def window(self, start=10, end=90, *, anchor_index=4, **changes):
        anchor = self.events[anchor_index - 1]
        result = {"id": "metadata-1", "kind": "metadata_pre_enqueue",
                  "start_ns": str(EPOCH + start), "end_ns": str(EPOCH + end),
                  "anchor_event_index": anchor_index, "anchor_name": anchor["name"],
                  "anchor_start_ns": str(core.a_ns(anchor["ts"])),
                  "anchor_end_ns": str(core.a_ns(anchor["ts"]) + core.a_ns(anchor["dur"]))}
        result.update(changes)
        return result

    def analyze(self, window=None):
        trace = self.write_trace()
        result = probe.analyze_trace(core, self.path, trace, [self.window() if window is None else window])
        return result, result["windows"][0]

    def test_nested_cpu_and_cann_calls_have_exclusive_integer_nanoseconds(self):
        self.events.extend([
            event("metadata_wrapper", 0, 100),
            event("aten::empty", 20, 60),
            event("aclnnGetWorkspaceSize", 30, 50, 20, category="cann_api"),
            event("unrelated_python_call", 0, 100, 11),
            event("device_work", 0, 100, 900, category="kernel"),
        ])
        _, row = self.analyze()
        self.assertEqual(row["status"], "MATCHED")
        self.assertEqual((row["window_ns"], row["attributed_ns"], row["ambiguous_ns"], row["unrecorded_ns"]),
                         (80, 80, 0, 0))
        observed = {(item["name"], item["lane"]): item["ns"] for item in row["exclusive_functions"]}
        self.assertEqual(observed, {("metadata_wrapper", "cpu_thread"): 40,
                                    ("aten::empty", "cpu_thread"): 20,
                                    ("aclnnGetWorkspaceSize", "cann_thread"): 20})
        self.assertEqual(sum(observed.values()), row["attributed_ns"])

    def test_pre_launch_without_recorded_events_remains_three_milliseconds(self):
        self.events.append(event("_expand_idx_mapping_kernel", 3_000_000, 3_010_000))
        _, row = self.analyze(self.window(0, 3_000_000))
        self.assertEqual(row["status"], "MATCHED")
        self.assertEqual((row["window_ns"], row["attributed_ns"], row["ambiguous_ns"], row["unrecorded_ns"]),
                         (3_000_000, 0, 0, 3_000_000))
        self.assertEqual(row["longest_unrecorded"][0]["ns"], 3_000_000)

    def test_crossing_and_equal_intervals_retain_ambiguity(self):
        for a, b, ambiguous in (((10, 60), (40, 80), 20), ((20, 60), (20, 60), 40)):
            with self.subTest(a=a, b=b):
                self.events = self.events[:3] + [event("wrapper", 0, 100),
                                                event("first_child", *a), event("second_child", *b)]
                _, row = self.analyze()
                self.assertEqual(row["ambiguous_ns"], ambiguous)
                self.assertEqual(row["attributed_ns"] + row["ambiguous_ns"] + row["unrecorded_ns"], 80)

    def test_anchor_name_time_and_index_mismatches_are_explicit(self):
        self.events.append(event("metadata_wrapper", 0, 100))
        for changes, status in (({"anchor_name": "different"}, "ANCHOR_MISMATCH"),
                                ({"anchor_start_ns": str(EPOCH + 1)}, "ANCHOR_MISMATCH"),
                                ({"anchor_event_index": 999}, "ANCHOR_NOT_FOUND")):
            with self.subTest(changes=changes):
                _, row = self.analyze(self.window(**changes))
                self.assertEqual(row["status"], status)

    def test_anchor_without_process_or_thread_identity_cannot_select_host_events(self):
        for missing in ("pid", "tid"):
            with self.subTest(missing=missing):
                anchor = event("metadata_wrapper", 0, 100)
                del anchor[missing]
                self.events = self.events[:3] + [anchor]
                _, row = self.analyze()
                self.assertEqual(row["status"], "ANCHOR_THREAD_UNAVAILABLE")

    def test_saved_hash_mismatch_rejects_different_trace(self):
        self.events.append(event("metadata_wrapper", 0, 100))
        trace = self.write_trace()
        trace["sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            probe.analyze_trace(core, self.path, trace, [self.window()])

    def test_previous_call_on_another_process_or_thread_cannot_bound_the_window(self):
        self.events.append(event("_expand_idx_mapping_kernel", 80, 100))
        for pid, tid in ((11, 7), (10, 8)):
            with self.subTest(pid=pid, tid=tid):
                self.events = self.events[:4] + [event("previous_launch", 0, 10, pid, tid)]
                start_anchor = {"anchor_event_index": 5, "anchor_name": "previous_launch",
                                "anchor_start_ns": str(EPOCH), "anchor_end_ns": str(EPOCH + 10)}
                _, row = self.analyze(self.window(10, 80, start_anchor=start_anchor))
                self.assertEqual(row["status"], "START_ANCHOR_MISMATCH")

    def test_begin_end_events_are_counted_and_leave_unrecorded_time(self):
        self.events.append(event("_expand_idx_mapping_kernel", 100, 110))
        self.events.extend([
            {"ph": "B", "name": "python_prepare", "pid": 10, "tid": 7, "ts": str(Decimal(EPOCH) / 1000)},
            {"ph": "E", "name": "python_prepare", "pid": 10, "tid": 7,
             "ts": str(Decimal(EPOCH + 90) / 1000)},
        ])
        result, row = self.analyze(self.window(0, 90))
        self.assertEqual(result["counts"]["be_events_unsupported"], 2)
        self.assertEqual(row["unrecorded_ns"], 90)


if __name__ == "__main__":
    unittest.main()
