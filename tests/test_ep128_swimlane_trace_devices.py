"""Full trace-device retention is opt-in, exact, and independent of U anchors."""

from decimal import Decimal
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import ep128_swimlane_trace as swim
import profile_ep128 as core


def device(name="UnlistedMetadataKernel", **overrides):
    result = {"ph": "X", "name": name, "cat": "kernel", "pid": 30,
              "tid": 7, "ts": "1.001", "dur": "0.125", "args": {}}
    result.update(overrides)
    return result


def process(pid, name):
    return {"ph": "M", "name": "process_name", "pid": pid, "args": {"name": name}}


class DeviceCollectionTests(unittest.TestCase):
    def collect(self, records, *, offset=0, enabled=True, metadata=None):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.json"
            path.write_text(json.dumps({"traceEvents": records}), encoding="utf-8")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            return swim.collect_host_events(core, path, offset, digest,
                retain_device_events=enabled, device_trace_metadata=metadata)

    def test_default_result_and_host_records_are_unchanged(self):
        host = device("vllm_ascend/worker/v2/model_runner.py(789): prepare_inputs",
                      cat="python_function", pid=10, args={"Python id": 1})
        records = [host, device(), process(10, "Python"), process(30, "Ascend Hardware")]
        default = self.collect(records, enabled=False)
        opted_in = self.collect(records)
        self.assertEqual(set(default), {"events", "sha256", "process_names", "counts"})
        self.assertEqual(default, {key: opted_in[key] for key in default})
        self.assertEqual(len(opted_in["trace_device_events"]), 1)

    def test_arbitrary_metadata_dma_and_zero_duration_tasks_are_retained(self):
        records = [device(), device("MEMCPY_ASYNC", cat="dma", dur="0.2"),
                   device("EVENT_RECORD", cat="runtime", dur=0),
                   process(30, "Ascend Hardware")]
        result = self.collect(records)
        saved = result["trace_device_events"]
        self.assertEqual([event["name"] for event in saved],
                         ["UnlistedMetadataKernel", "MEMCPY_ASYNC", "EVENT_RECORD"])
        self.assertEqual([event["trace_event_index"] for event in saved], [1, 2, 3])
        self.assertEqual(saved[-1]["start_ns"], saved[-1]["end_ns"])
        self.assertEqual(result["device_event_scan"]["counts"]["zero_duration_device_events"], 1)
        self.assertEqual(result["device_event_scan"]["category_counts"],
                         {"kernel": 1, "dma": 1, "runtime": 1})
        self.assertFalse(result["device_event_scan"]["intervals_clipped"])
        self.assertFalse(result["device_event_scan"]["topn_filter"])

    def test_late_process_metadata_classifies_preceding_unhinted_device_task(self):
        result = self.collect([device("unknown", cat=""), process(30, "Ascend Hardware")])
        saved = result["trace_device_events"][0]
        self.assertFalse(saved["device_hint"])
        self.assertTrue(saved["device_process_verified"])
        self.assertEqual(saved["device_classification"], "TRACE_PROCESS_METADATA")
        self.assertEqual(saved["process_name"], "Ascend Hardware")

    def test_clock_process_names_classify_without_in_pass_metadata(self):
        result = self.collect([device("MEMCPY_ASYNC", cat="dma")],
                              metadata={"process_names": {30: "Ascend Hardware"}})
        self.assertEqual(len(result["trace_device_events"]), 1)
        self.assertEqual(result["trace_device_events"][0]["device_classification"], "TRACE_PROCESS_METADATA")

    def test_host_runtime_lookalikes_and_unknown_processes_are_not_tasks(self):
        records = [process(10, "CANN"), device("AscendCL@aclrtMemcpyAsync", cat="runtime", pid=10),
                   device("unknown", cat="cpu_op", pid=11),
                   device("unknown too", cat="", pid=12), process(12, "Python")]
        result = self.collect(records)
        self.assertEqual(result["trace_device_events"], [])
        self.assertEqual(result["device_event_scan"]["counts"]["scanned_x_events"], 3)

    def test_native_hint_alone_is_device_evidence_not_host_causality(self):
        result = self.collect([device(cat="", args={"Task Type": "MEMCPY_ASYNC", "Stream Id": 4})])
        saved = result["trace_device_events"][0]
        self.assertEqual(saved["device_classification"], "NATIVE_HINT")
        self.assertFalse(saved["device_process_verified"])
        self.assertEqual(saved["task_type"], "MEMCPY_ASYNC")
        self.assertNotIn("host_event_index", saved)
        self.assertNotIn("round_id", saved)

    def test_exact_epoch_intervals_and_native_identifiers_survive(self):
        epoch = 1790000000000000000
        huge = 9007199254740993123
        args = {"Device Id": 0, "Model Id": 42, "Stream Id": 3, "Physic Stream Id": 61,
                "Task Id": huge, "Context Id": 9, "Batch Id": 13, "Subtask Id": 2,
                "Task Type": "AI_CORE", "Connection Id": huge + 1, "Correlation Id": "c-7",
                "Tensor Content": [1, 2], "Payload": {"secret": "not an id"}}
        result = self.collect([process(30, "NPU"), device(ts=str(Decimal(epoch + 17) / 1000),
            dur="0.023", args=args, id=huge + 2, id2={"local": "local-flow"})], offset=11)
        saved = result["trace_device_events"][0]
        self.assertEqual((saved["start_ns"], saved["end_ns"], saved["duration_ns"]),
                         (epoch + 28, epoch + 51, 23))
        self.assertEqual(saved["event_index"], 2)
        self.assertEqual(saved["trace_event_index"], 2)
        self.assertEqual(saved["stream_id"], "61")
        self.assertEqual(saved["task_id"], str(huge))
        self.assertEqual(saved["connection_id"], str(huge + 1))
        self.assertEqual(saved["correlation_id"], "c-7")
        self.assertEqual(saved["source_ids"]["raw_id"], str(huge + 2))
        self.assertEqual(saved["source_ids"]["raw_id2_local"], "local-flow")
        self.assertEqual(saved["source_ids"]["streamid"], "3")
        self.assertNotIn("tensor", json.dumps(saved).lower())
        self.assertNotIn("secret", json.dumps(saved).lower())

    def test_identifier_objects_are_not_stringified_into_native_ids(self):
        saved = self.collect([device(args={"Task Id": {"value": 8}, "Model Id": [3]})])["trace_device_events"][0]
        self.assertIsNone(saved["task_id"])
        self.assertIsNone(saved["model_id"])
        self.assertNotIn("taskid", saved["source_ids"])
        self.assertNotIn("modelid", saved["source_ids"])

    def test_invalid_and_negative_device_intervals_are_diagnostic_only(self):
        invalid = device("bad", ts="invalid", cat="")
        missing = device("missing", cat="")
        del missing["dur"]
        records = [device("negative", dur=-1, cat=""), invalid, missing,
                   device("zero", dur=0, cat=""), process(30, "Ascend Hardware")]
        result = self.collect(records)
        self.assertEqual([event["name"] for event in result["trace_device_events"]], ["zero"])
        self.assertEqual(result["device_event_scan"]["counts"]["negative_device_duration_events"], 1)
        self.assertEqual(result["device_event_scan"]["counts"]["invalid_device_timed_events"], 2)

    def test_non_x_device_records_are_not_fabricated_as_timed_tasks(self):
        records = [device(ph="B"), device(ph="E"), device(ph="i")]
        result = self.collect(records)
        self.assertEqual(result["trace_device_events"], [])
        self.assertIn("B/E durations are not reconstructed", result["device_event_scan"]["supported_device_phase"])

    def test_all_tasks_retained_with_one_scan_and_no_topn(self):
        records = [device(str(index), ts=index, args={"Task Type": "AI_CORE"}) for index in range(300)]
        with patch.object(core, "residual_iter_trace_events", wraps=core.residual_iter_trace_events) as scanner:
            result = self.collect(records)
        self.assertEqual(scanner.call_count, 1)
        self.assertEqual(len(result["trace_device_events"]), 300)
        self.assertEqual(result["trace_device_events"][-1]["name"], "299")
        self.assertEqual(result["device_event_scan"]["task_type_counts"], {"AI_CORE": 300})

    def test_clock_and_collected_trace_hashes_must_match(self):
        with self.assertRaisesRegex(ValueError, "metadata SHA256"):
            self.collect([], metadata={"sha256": "0" * 64, "process_names": {}})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.json"
            path.write_text("[]", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                swim.collect_host_events(core, path, 0, "0" * 64, retain_device_events=True)

    def test_source_stamp_must_stay_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.json"
            path.write_text("[]", encoding="utf-8")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            original = core.residual_iter_trace_events

            def changing_trace(*args, **kwargs):
                yield from original(*args, **kwargs)
                path.write_text("[]\n", encoding="utf-8")

            with patch.object(core, "residual_iter_trace_events", changing_trace):
                with self.assertRaisesRegex(ValueError, "Trace changed"):
                    swim.collect_host_events(core, path, 0, digest, retain_device_events=True)


if __name__ == "__main__":
    unittest.main()
