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
    "dp_dispatch_patch": "vllm_ascend/patch/worker/patch_v2/patch_model_runner.py(16): dispatch_cg_and_sync_dp",
    "dp_dispatch": "vllm/v1/worker/gpu/dp_utils.py(214): dispatch_cg_and_sync_dp",
    "dp_sync": "vllm/v1/worker/gpu/dp_utils.py(44): sync_cudagraph_and_dp_padding",
    "all_reduce": "torch/distributed/distributed_c10d.py(2917): all_reduce",
    "collective_wrapper": "torch/distributed/c10d_logger.py(80): wrapper",
    "cpu_wait": "<built-in method wait of pybind11_builtins.Work object at 0x123>",
    "stream_dependency": "torch_npu/npu/streams.py(38): wait_stream",
    "graph_dispatch": "vllm/v1/worker/gpu/cudagraph_utils.py(193): dispatch",
    "build_attn_metadata": "vllm_ascend/worker/v2/attn_utils.py(370): build_attn_metadata",
    "metadata_builder": "vllm_ascend/attention/dsa_v1.py(846): build",
    "blocktable_gather": "vllm/v1/worker/gpu/block_table.py(188): gather_block_tables",
    "slot_mappings": "vllm/v1/worker/gpu/block_table.py(213): compute_slot_mappings",
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
    def collect(self, records):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.json"
            path.write_text(json.dumps(records), encoding="utf-8")
            return swim.collect_host_events(core, path, 0, hashlib.sha256(path.read_bytes()).hexdigest())

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

    def test_native_cpu_apis_and_source_ids_survive_but_device_duplicates_do_not(self):
        name = "AscendCL@aclnnMixedQuantSparseFlashMlaMetadataGetWorkspaceSize"
        raw = {"ph": "X", "name": name, "cat": "cann_api", "pid": 20, "tid": 7,
               "ts": "1.000", "dur": "0.125", "id": 55, "id2": {"global": "g1"},
               "args": {"Connection Id": 100, "Correlation Id": "corr", "Stream Id": 61,
                        "thread_id": 7, "Tensor Content": [1, 2, 3]}}
        device = {**raw, "pid": 30, "cat": "kernel"}
        late_device = {**raw, "pid": 31}
        result = self.collect([raw, device, late_device,
            {"ph": "M", "name": "process_name", "pid": 20, "args": {"name": "CANN"}},
            {"ph": "M", "name": "process_name", "pid": 31, "args": {"name": "Ascend Hardware"}}])
        self.assertEqual(len(result["events"]), 1)
        saved = result["events"][0]
        self.assertEqual(saved["kind"], "native_workspace")
        self.assertEqual(saved["event_type"], "native_api")
        self.assertEqual(saved["source_ids"], {"connectionid": "100", "correlationid": "corr", "streamid": "61",
                                              "raw_id": "55", "raw_id2_global": "g1"})
        self.assertEqual(saved["thread_id"], "7")
        self.assertEqual(result["counts"]["device_events_rejected"], 2)
        self.assertNotIn("tensorcontent", json.dumps(saved).lower())

    def test_stream_dependencies_are_distinct_from_blocking_waits_and_copies(self):
        names = [NAMES["stream_dependency"], "AscendCL@aclrtStreamWaitEvent",
                 NAMES["cpu_wait"], "Event::synchronize", "aten::copy_"]
        records = [{"ph": "X", "name": name, "cat": "python_function" if i in (0, 2) else "cpu_op",
                    "pid": 10, "tid": 7, "ts": i, "dur": 1} for i, name in enumerate(names)]
        result = self.collect(records)
        self.assertEqual([e["kind"] for e in result["events"]],
                         ["stream_dependency", "stream_dependency", "cpu_wait", "cpu_sync", "host_copy"])

    def test_capture_builder_is_a_recorded_call_not_synthesized_per_replay(self):
        names = ["vllm_ascend/attention/dsa_v1.py(835): build_for_cudagraph_capture", NAMES["main_graph"], NAMES["main_graph"]]
        result = self.collect([{"ph": "X", "name": name, "cat": "python_function", "pid": 10,
                                "tid": 7, "ts": i*100, "dur": 5} for i, name in enumerate(names)])
        self.assertEqual(sum(e["kind"] == "metadata_capture_builder" for e in result["events"]), 1)
        self.assertEqual(sum(e["kind"] == "main_graph" for e in result["events"]), 2)

    def test_lifecycle_gather_and_input_wrappers_are_selected_by_path(self):
        names = ["vllm/v1/worker/gpu/model_runner.py(100): update_requests",
                 "vllm_ascend/worker/v2/model_runner.py(751): gather_batch_req_state",
                 "vllm/v1/worker/gpu/block_table.py(135): apply_staged_writes",
                 "vllm/v1/worker/gpu/input_batch.py(300): prepare_pos_seq_lens",
                 "vllm/v1/worker/gpu/input_batch.py(400): combine_sampled_and_draft_tokens",
                 "vllm/utils/torch_utils.py(600): async_tensor_h2d",
                 "other_package/model_runner.py(100): update_requests"]
        result = self.collect([{"ph": "X", "name": name, "cat": "python_function", "pid": 10,
                                "tid": 7, "ts": i, "dur": 1} for i, name in enumerate(names)])
        self.assertEqual([e["kind"] for e in result["events"]],
                         ["request_update_requests", "request_gather_batch_req_state", "blocktable_apply",
                          "input_prepare_pos_seq_lens", "input_combine_sampled_and_draft_tokens", "input_h2d"])


class PairingTests(unittest.TestCase):
    def test_early_inputs_are_retained_in_current_main_hc(self):
        events, rounds = fixture()
        original = deepcopy(events)
        result = swim.pair_rounds(core, rounds, events)[0]
        self.assertEqual(result["status"], "MATCHED")
        self.assertEqual(result["functions"], {"outer_propose": 3, "next_execute": 4,
                                               "runner_inputs": 5, "runner_attn": 9, "state_attn": 6})
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
                       host(17, "main_graph", 790, 830, 132, 13),
                       host(18, "runner_attn", 632, 638, 133, 13)])
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

    def test_runner_attn_has_its_own_verified_blocktable_and_slot_children(self):
        events, rounds = fixture()
        events.extend([host(60, "blocktable_gather", 233, 234, 60, 34),
                       host(61, "slot_mappings", 234, 237, 61, 34)])
        original = deepcopy(events)
        result = swim.pair_rounds(core, rounds, events)[0]
        self.assertEqual(result["status"], "MATCHED")
        self.assertEqual(result["functions"]["runner_attn"], 9)
        self.assertTrue(result["ownership_verified"]["runner_attn"])
        children = result["child_layer"]["by_role"]["runner_attn"]
        self.assertEqual([c["event_index"] for c in children], [60, 61])
        self.assertEqual([c["kind"] for c in children], ["blocktable_gather", "slot_mappings"])
        self.assertTrue(all(c["parent_chain_verified"] and c["depth"] == 1 for c in children))
        self.assertEqual([(c["start_ns"], c["end_ns"]) for c in children], [(233, 234), (234, 237)])
        self.assertEqual(result["child_layer"]["by_role"]["runner_inputs"][0]["event_index"], 8)
        self.assertEqual(events, original)

    def test_runner_attn_in_propose_cannot_fill_missing_next_runner_attn(self):
        events, rounds = fixture()
        events = [e for e in events if e["kind"] != "runner_attn"]
        events.append(host(60, "runner_attn", 60, 70, 60, 20))
        result = swim.pair_rounds(core, rounds, events)[0]
        self.assertEqual(result["status"], "PARTIAL")
        self.assertIsNone(result["functions"]["runner_attn"])
        self.assertIn("runner_attn", result["missing"])
        self.assertFalse(result["order_verified"])
        self.assertEqual(result["child_layer"]["by_role"]["runner_attn"], [])
        self.assertIn(60, [c["event_index"] for c in result["child_layer"]["by_role"]["outer_propose"]])

    def test_runner_attn_requires_same_parent_and_worker_thread(self):
        for change in ({"python_parent_id": "999"}, {"tid": "999"}):
            with self.subTest(change=change):
                events, rounds = fixture()
                events[-1].update(change)
                result = swim.pair_rounds(core, rounds, events)[0]
                self.assertEqual(result["status"], "PARTIAL")
                self.assertIsNone(result["functions"]["runner_attn"])
                self.assertFalse(result["ownership_verified"]["runner_attn"])

    def test_runner_attn_order_must_follow_inputs_and_precede_state_metadata(self):
        for start, end in ((220, 235), (241, 245)):
            with self.subTest(start=start, end=end):
                events, rounds = fixture()
                events[-1].update(start_ns=start, end_ns=end, duration_ns=end-start)
                result = swim.pair_rounds(core, rounds, events)[0]
                self.assertEqual(result["status"], "PARTIAL")
                self.assertEqual(result["round_identity_status"], "ORDER_CONFLICT")
                self.assertIn("RUNNER_ATTN_ORDER_CONFLICT", result["reasons"])
                self.assertTrue(result["ownership_verified"]["runner_attn"])
                self.assertFalse(result["order_verified"])

    def test_two_runner_attn_calls_are_ambiguous_not_selected_by_duration(self):
        events, rounds = fixture()
        events.append(host(60, "runner_attn", 231, 239, 60, 3))
        result = swim.pair_rounds(core, rounds, events)[0]
        self.assertEqual(result["status"], "AMBIGUOUS")
        self.assertEqual(result["candidates"]["runner_attn"], [60, 9])
        self.assertIsNone(result["functions"]["runner_attn"])

    def test_runner_attn_is_not_borrowed_from_the_next_hc_round(self):
        events, rounds = fixture()
        events = [e for e in events if e["kind"] != "runner_attn"]
        events.extend([host(12, "core_sample", 460, 550, 12, 0),
                       host(13, "outer_propose", 480, 540, 120, 12),
                       host(14, "core_execute", 560, 850, 13, 0),
                       host(15, "runner_inputs", 600, 630, 130, 13),
                       host(16, "state_attn", 640, 780, 131, 13),
                       host(17, "main_graph", 790, 830, 132, 13),
                       host(18, "runner_attn", 632, 638, 133, 13)])
        rounds.append({"round_id": "hc-2", "start_ns": "400", "end_ns": "800", "layers": []})
        first, second = swim.pair_rounds(core, rounds, events)
        self.assertEqual(first["status"], "PARTIAL")
        self.assertIsNone(first["functions"]["runner_attn"])
        self.assertEqual(second["status"], "MATCHED")
        self.assertEqual(second["functions"]["runner_attn"], 18)

    def test_nested_dp_wait_proof_and_prefix_partition_do_not_sum_inclusive_times(self):
        events, rounds = fixture()
        events.extend([host(40, "dp_dispatch_patch", 170, 195, 40, 3),
                       host(41, "dp_dispatch", 171, 194, 41, 40),
                       host(42, "dp_sync", 173, 192, 42, 41),
                       host(43, "collective_wrapper", 175, 190, 43, 42),
                       host(44, "all_reduce", 176, 189, 44, 43),
                       host(45, "cpu_wait", 177, 188, 45, 44),
                       host(46, "stream_dependency", 178, 180, 46, 44)])
        result = swim.pair_rounds(core, rounds, events)[0]
        block = result["child_layer"]["prefix_partition"]
        self.assertEqual(block["status"], "VERIFIED_HOST_BOUNDARIES")
        self.assertEqual([(p["key"], p["duration_ns"]) for p in block["parts"]],
                         [("pre_dispatch", 10), ("dispatch_envelope", 25), ("post_dispatch", 5)])
        self.assertEqual(sum(p["duration_ns"] for p in block["parts"]), 40)
        self.assertEqual([r["event_index"] for r in block["dp_sync"]["events"]], [42])
        self.assertEqual([r["event_index"] for r in block["dp_sync"]["all_reduce_events"]], [44])
        self.assertEqual([r["event_index"] for r in block["wait_candidates"]], [45])
        wait = block["wait_candidates"][0]
        self.assertTrue(wait["parent_chain_verified"])
        self.assertEqual(wait["owning_all_reduce_event_index"], 44)
        self.assertEqual(wait["duration_ns"], 11)
        self.assertGreater(wait["depth"], 1)

    def test_same_named_collective_in_propose_is_not_next_prefix_dp_wait(self):
        events, rounds = fixture()
        events.extend([host(50, "all_reduce", 70, 100, 50, 20), host(51, "cpu_wait", 75, 95, 51, 50)])
        result = swim.pair_rounds(core, rounds, events)[0]
        layer = result["child_layer"]
        self.assertIn(51, [r["event_index"] for r in layer["by_role"]["outer_propose"]])
        self.assertEqual(layer["prefix_partition"]["wait_candidates"], [])
        self.assertEqual(layer["prefix_partition"]["dp_sync"]["events"], [])

    def test_inferred_wait_is_preserved_but_not_counted_as_verified_dp_wait(self):
        events, rounds = fixture()
        events.extend([host(40, "dp_dispatch", 170, 195, 40, 3),
                       host(42, "dp_sync", 173, 192, 42, 40),
                       host(44, "all_reduce", 176, 189, 44, 42),
                       host(45, "cpu_wait", 177, 188, 45, 999)])
        result = swim.pair_rounds(core, rounds, events)[0]
        layer = result["child_layer"]
        wait = next(r for r in layer["by_role"]["preparation_dp_aggregate"] if r["event_index"] == 45)
        self.assertEqual(wait["relation"], "time_containment_only")
        self.assertFalse(wait["parent_chain_verified"])
        self.assertEqual(layer["prefix_partition"]["wait_candidates"], [])

    def test_missing_child_calls_remain_unknown_instead_of_fabricated_dp(self):
        events, rounds = fixture()
        result = swim.pair_rounds(core, rounds, events)[0]
        block = result["child_layer"]["prefix_partition"]
        self.assertEqual(block["status"], "NOT_RECORDED")
        self.assertEqual(block["parts"][0]["key"], "preparation_dp_aggregate")
        self.assertEqual(block["parts"][0]["duration_ns"], 40)
        self.assertEqual(block["dp_sync"]["events"], [])
        self.assertEqual(block["wait_candidates"], [])

    def test_two_outer_dispatch_calls_do_not_choose_one_for_a_complete_partition(self):
        events, rounds = fixture()
        events.extend([host(40, "dp_dispatch", 170, 180, 40, 3),
                       host(41, "dp_dispatch", 185, 195, 41, 3)])
        block = swim.pair_rounds(core, rounds, events)[0]["child_layer"]["prefix_partition"]
        self.assertEqual(block["status"], "AMBIGUOUS")
        self.assertEqual([p["key"] for p in block["parts"]], ["preparation_dp_aggregate"])

    def test_native_metadata_is_containment_only_and_direct_python_child_has_depth_one(self):
        events, rounds = fixture()
        events.extend([host(60, "build_attn_metadata", 250, 370, 60, 31),
                       host(61, "metadata_builder", 260, 350, 61, 60),
                       {"event_index": 62, "kind": "native_workspace", "name": "AscendCL@aclnnMetadataGetWorkspaceSize",
                        "pid": "20", "tid": "7", "event_type": "native_api", "cat": "cann_api",
                        "start_ns": 270, "end_ns": 340, "duration_ns": 70}])
        layer = swim.pair_rounds(core, rounds, events)[0]["child_layer"]["by_role"]["state_attn"]
        rows = {r["event_index"]: r for r in layer}
        self.assertEqual(rows[60]["depth"], 1)
        self.assertEqual(rows[61]["depth"], 2)
        self.assertIsNone(rows[62]["depth"])
        self.assertEqual(rows[62]["relation"], "native_thread_time_containment_only")
        self.assertFalse(rows[62]["parent_chain_verified"])


if __name__ == "__main__":
    unittest.main()
