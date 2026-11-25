"""Same-round Host swimlanes retain early submission and reject hard guesses."""

from copy import deepcopy
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import ep128_swimlane_trace as swim
import profile_ep128 as core


NAMES = {
    "core_execute": "vllm/v1/worker/gpu/model_runner.py(1630): execute_model",
    "core_sample": "vllm/v1/worker/gpu/model_runner.py(1995): sample_tokens",
    "outer_propose": "vllm_ascend/worker/v2/spec_decode/dspark/speculator.py(410): propose",
    "runner_inputs": "vllm_ascend/worker/v2/model_runner.py(789): prepare_inputs",
    "state_attn": "vllm_ascend/worker/v2/model_states/default.py(178): prepare_attn",
    "runner_attn": "vllm/v1/worker/gpu/model_runner.py(1472): prepare_attn",
    "main_graph": "vllm_ascend/worker/v2/aclgraph_utils.py(174): run_fullgraph",
    "seq_lens_cpu": "vllm_ascend/worker/v2/model_runner.py(1306): _update_seq_lens_cpu",
}


def host(number, kind, start, end, py, parent, pid="10", tid="7"):
    return {"event_index": number, "name": NAMES[kind], "kind": kind,
            "start_ns": start, "end_ns": end, "duration_ns": end - start,
            "pid": pid, "tid": tid, "python_id": str(py), "python_parent_id": str(parent),
            "cat": "python_function"}


def fixture():
    # The next inputs and metadata start while the current main HC is executing.
    events = [host(1, "core_execute", -200, 10, 1, 0),
              host(2, "core_sample", 20, 150, 2, 0),
              host(3, "outer_propose", 50, 140, 20, 2),
              host(4, "core_execute", 160, 450, 3, 0),
              host(5, "runner_inputs", 200, 230, 30, 3),
              host(6, "state_attn", 240, 380, 31, 3),
              host(7, "main_graph", 390, 430, 32, 3),
              host(8, "seq_lens_cpu", 201, 210, 33, 30),
              host(9, "runner_attn", 232, 238, 34, 3)]
    rounds = [{"round_id": "hc-1", "start_ns": "0", "end_ns": "400", "layers": [
        {"role": "main", "start_ns": "0", "end_ns": "250"},
        {"role": "draft", "start_ns": "300", "end_ns": "350"}]}]
    return events, rounds


class CollectionTests(unittest.TestCase):
    def test_exact_integer_times_full_ranges_and_original_indexes(self):
        epoch = 1790000000000000000
        records = [{"ph": "M", "name": "thread_name", "pid": 10, "tid": 7, "args": {"name": "worker"}}]
        for name in (NAMES["runner_inputs"], "vllm/v1/worker/gpu/other.py(1): prepare_inputs"):
            records.append({"ph": "X", "name": name, "cat": "python_function",
                            "pid": 10, "tid": 7, "ts": str(Decimal(epoch + 17) / 1000),
                            "dur": "0.023", "args": {"Python id": 8, "Python parent id": 4}})
        records.append({"ph": "M", "name": "process_name", "pid": 10, "args": {"name": "Python"}})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.json"
            path.write_text(json.dumps({"traceEvents": records}), encoding="utf-8")
            expected = hashlib.sha256(path.read_bytes()).hexdigest()
            result = swim.collect_host_events(core, path, 11, expected)
        self.assertEqual(result["sha256"], expected)
        self.assertEqual(result["counts"]["trace_events"], 4)
        self.assertEqual(result["process_names"], {"10": "Python"})
        self.assertEqual(len(result["events"]), 1)
        event = result["events"][0]
        self.assertEqual((event["event_index"], event["start_ns"], event["end_ns"]), (2, epoch + 28, epoch + 51))
        self.assertEqual((event["python_id"], event["python_parent_id"]), ("8", "4"))
        self.assertEqual(event["process_name"], "Python")

    def test_keeps_every_matching_call_without_topn(self):
        records = [{"ph": "X", "name": NAMES["core_execute"], "cat": "python_function",
                    "pid": 10, "tid": 7, "ts": n, "dur": 1,
                    "args": {"Python id": n + 1, "Python parent id": 0}} for n in range(30)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.json"
            path.write_text(json.dumps(records), encoding="utf-8")
            result = swim.collect_host_events(core, path, 0, hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertEqual(len(result["events"]), 30)
        self.assertEqual(result["events"][-1]["event_index"], 30)

    def test_hash_mismatch_rejects_entire_collection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.json"
            path.write_text("[]", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                swim.collect_host_events(core, path, 0, "0" * 64)

    def test_source_changed_after_scan_is_rejected_even_with_original_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.json"
            path.write_text("[]", encoding="utf-8")
            expected = hashlib.sha256(path.read_bytes()).hexdigest()
            original = core.residual_iter_trace_events

            def changing_trace(*args, **kwargs):
                yield from original(*args, **kwargs)
                path.write_text("[]\n", encoding="utf-8")

            with patch.object(core, "residual_iter_trace_events", changing_trace):
                with self.assertRaisesRegex(ValueError, "Trace changed"):
                    swim.collect_host_events(core, path, 0, expected)


class PairingTests(unittest.TestCase):
    def test_early_inputs_are_retained_in_current_main_hc(self):
        events, rounds = fixture()
        original = deepcopy(events)
        result = swim.pair_rounds(core, rounds, events)[0]
        self.assertEqual(result["status"], "MATCHED")
        self.assertEqual(result["functions"], {"outer_propose": 3, "next_execute": 4,
                                               "runner_inputs": 5, "state_attn": 6})
        self.assertEqual(result["context"], {"current_execute": 1, "current_sample": 2, "next_main_graph": 7})
        self.assertEqual(result["missing"], [])
        self.assertIn("INFERRED", result["pairing_basis"])
        self.assertEqual(result["host_ownership_status"], "VERIFIED")
        self.assertEqual(result["round_identity_status"], "INFERRED_BY_HOST_CALL_ORDER")
        self.assertTrue(result["order_verified"])
        self.assertEqual(events, original)
        self.assertEqual(next(e for e in result["small_functions"] if e["event_index"] == 8)["relation"], "python_parent_id")
        # Current main Host entry precedes this device cycle. Never clip it away.
        self.assertEqual(events[0]["start_ns"], -200)
        self.assertEqual(events[6]["end_ns"], 430)

    def test_two_worker_lanes_are_ambiguous_even_if_one_has_more_calls(self):
        events, rounds = fixture()
        events.append(host(99, "core_execute", 1000, 1100, 99, 0, pid="20", tid="8"))
        result = swim.pair_rounds(core, rounds, events)[0]
        self.assertEqual(result["status"], "AMBIGUOUS")
        self.assertIn("MULTIPLE_WORKER_LANES", result["reasons"])
        self.assertTrue(all(v is None for v in result["functions"].values()))

    def test_wrong_inputs_parent_cannot_be_replaced_by_time_containment(self):
        events, rounds = fixture()
        events[4]["python_parent_id"] = "999"
        result = swim.pair_rounds(core, rounds, events)[0]
        self.assertEqual(result["status"], "PARTIAL")
        self.assertIsNone(result["functions"]["runner_inputs"])
        self.assertIn("runner_inputs", result["missing"])

    def test_wrong_propose_parent_does_not_attach_to_nearby_sample(self):
        events, rounds = fixture()
        events[2]["python_parent_id"] = "999"
        result = swim.pair_rounds(core, rounds, events)[0]
        self.assertEqual(result["status"], "PARTIAL")
        self.assertIsNone(result["functions"]["outer_propose"])

    def test_multiple_graph_entries_are_ambiguous_not_longest_selected(self):
        events, rounds = fixture()
        events.append(host(88, "main_graph", 385, 435, 88, 3))
        result = swim.pair_rounds(core, rounds, events)[0]
        self.assertEqual(result["status"], "AMBIGUOUS")
        self.assertEqual(result["candidates"]["next_main_graph"], [88, 7])
        self.assertIsNone(result["functions"]["next_execute"])

    def test_missing_graph_keeps_round_without_guessed_core(self):
        events, rounds = fixture()
        events = [e for e in events if e["kind"] != "main_graph"]
        result = swim.pair_rounds(core, rounds, events)[0]
        self.assertEqual(result["status"], "PARTIAL")
        self.assertEqual(result["round_id"], "hc-1")
        self.assertIsNone(result["functions"]["next_execute"])

    def test_missing_first_round_inputs_are_not_borrowed_from_second_round(self):
        events, rounds = fixture()
        events = [e for e in events if e["event_index"] != 5]
        events.extend([host(12, "core_sample", 460, 550, 12, 0),
                       host(13, "outer_propose", 480, 540, 120, 12),
                       host(14, "core_execute", 560, 850, 13, 0),
                       host(15, "runner_inputs", 600, 630, 130, 13),
                       host(16, "state_attn", 640, 780, 131, 13),
                       host(17, "main_graph", 790, 830, 132, 13)])
        rounds.append({"round_id": "hc-2", "start_ns": "400", "end_ns": "800", "layers": []})
        first, second = swim.pair_rounds(core, rounds, events)
        self.assertEqual(first["status"], "PARTIAL")
        self.assertIsNone(first["functions"]["runner_inputs"])
        self.assertEqual(second["status"], "MATCHED")
        self.assertEqual(second["functions"]["runner_inputs"], 15)

    def test_no_older_sample_fallback_across_previous_execute(self):
        events, rounds = fixture()
        events[1].update(start_ns=-300, end_ns=-250)
        events[2].update(start_ns=-290, end_ns=-260)
        result = swim.pair_rounds(core, rounds, events)[0]
        self.assertEqual(result["status"], "PARTIAL")
        self.assertIsNone(result["context"]["current_sample"])
        self.assertIsNone(result["functions"]["outer_propose"])

    def test_bad_parent_containment_is_not_a_match(self):
        events, rounds = fixture()
        events[4]["end_ns"] = 500
        result = swim.pair_rounds(core, rounds, events)[0]
        self.assertEqual(result["status"], "PARTIAL")
        self.assertIsNone(result["functions"]["runner_inputs"])

    def test_reversed_inputs_and_metadata_do_not_match_despite_valid_parent_ids(self):
        events, rounds = fixture()
        events[4].update(start_ns=300, end_ns=320)
        events[5].update(start_ns=240, end_ns=290)
        result = swim.pair_rounds(core, rounds, events)[0]
        self.assertEqual(result["status"], "PARTIAL")
        self.assertEqual(result["round_identity_status"], "ORDER_CONFLICT")
        self.assertFalse(result["order_verified"])
        self.assertEqual(result["host_ownership_status"], "VERIFIED")
        self.assertTrue(all(result["ownership_verified"].values()))

    def test_graph_before_metadata_returns_is_not_a_valid_sibling_order(self):
        events, rounds = fixture()
        events[6].update(start_ns=370, end_ns=430)
        result = swim.pair_rounds(core, rounds, events)[0]
        self.assertEqual(result["status"], "PARTIAL")
        self.assertIn("INPUT_METADATA_GRAPH_ORDER_CONFLICT", result["reasons"])

    def test_multiple_samples_between_core_executes_are_ambiguous(self):
        events, rounds = fixture()
        events.append(host(66, "core_sample", 151, 159, 66, 0))
        result = swim.pair_rounds(core, rounds, events)[0]
        self.assertEqual(result["status"], "AMBIGUOUS")
        self.assertIsNone(result["functions"]["outer_propose"])


if __name__ == "__main__":
    unittest.main()
