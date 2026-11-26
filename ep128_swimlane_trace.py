"""Collect complete selected Host calls and conservatively pair HC rounds.

Python parent IDs establish Host ownership.  Graph-entry ordering only infers
the corresponding device round; it is not a kernel submission/flow proof.
"""

from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path
import re
import time


_FUNCTIONS = (
    ("core_execute", r"vllm/v1/worker/gpu/model_runner\.py", "execute_model"),
    ("core_sample", r"vllm/v1/worker/gpu/model_runner\.py", "sample_tokens"),
    ("outer_propose", r"vllm_ascend/worker/v2/spec_decode/dspark/speculator\.py", "propose"),
    ("dflash_propose", r"vllm/v1/worker/gpu/spec_decode/dflash/speculator\.py", "propose"),
    ("runner_inputs", r"vllm_ascend/worker/v2/model_runner\.py", "prepare_inputs"),
    ("seq_lens_cpu", r"vllm_ascend/worker/v2/model_runner\.py", "_update_seq_lens_cpu"),
    ("runner_attn", r"vllm/v1/worker/gpu/model_runner\.py", "prepare_attn"),
    ("state_attn", r"vllm_ascend/worker/v2/model_states/default\.py", "prepare_attn"),
    ("state_inputs", r"vllm_ascend/worker/v2/model_states/default\.py", "prepare_inputs"),
    ("upstream_state_inputs", r"vllm/v1/worker/gpu/model_states/default\.py", "prepare_inputs"),
    ("build_attn_metadata", r"vllm_ascend/worker/v2/attn_utils\.py", "build_attn_metadata"),
    ("draft_build_metadata", r"vllm_ascend/worker/v2/spec_decode/dspark/speculator\.py", "build_draft_attn_metadatas"),
    ("main_graph", r"vllm_ascend/worker/v2/aclgraph_utils\.py", "run_fullgraph"),
    ("main_graph_relay", r"vllm_ascend/worker/v2/aclgraph_utils\.py", "_graph_relay"),
    ("main_graph_replay", r"vllm_ascend/worker/v2/aclgraph_utils\.py", "_graph_replay"),
    ("draft_graph", r"vllm_ascend/worker/v2/spec_decode/dflash/aclgraph\.py", "run_fullgraph"),
    ("draft_graph_replay", r"vllm_ascend/worker/v2/spec_decode/dflash/aclgraph\.py", "_graph_replay"),
    ("draft_graph_relay", r"vllm_ascend/worker/v2/spec_decode/dflash/aclgraph\.py", "_graph_relay"),
    ("upstream_graph", r"vllm/v1/worker/gpu/cudagraph_utils\.py", "run_fullgraph"),
    ("dp_dispatch", r"vllm/v1/worker/gpu/dp_utils\.py", "dispatch_cg_and_sync_dp"),
    ("dp_dispatch_patch", r"vllm_ascend/patch/worker/patch_v2/patch_model_runner\.py", "dispatch_cg_and_sync_dp"),
    ("dp_sync", r"vllm/v1/worker/gpu/dp_utils\.py", "sync_cudagraph_and_dp_padding"),
    ("graph_dispatch", r"vllm/v1/worker/gpu/cudagraph_utils\.py", "dispatch"),
    ("all_reduce", r"torch/distributed/distributed_c10d\.py", "all_reduce"),
    ("collective_wrapper", r"torch/distributed/c10d_logger\.py", "wrapper"),
    ("context_wrapper", r"torch/utils/_contextlib\.py", "decorate_context"),
    ("profiler_wrapper", r"torch_npu/profiler/profiler_interface\.py", "wrapper"),
    ("input_h2d", r"vllm/utils/torch_utils\.py", "async_tensor_h2d"),
    ("uniform_metadata", r"vllm/v1/worker/gpu/spec_decode/speculator\.py", "_build_uniform_attn_metadata"),
    ("speculator_metadata", r"vllm/v1/worker/gpu/spec_decode/speculator\.py", "_build_attn_metadata"),
    ("speculator_metadata", r"vllm_ascend/worker/v2/spec_decode/dspark/speculator\.py", "_build_attn_metadata"),
    ("dflash_inputs", r"vllm_ascend/ops/triton/(?:v2/)?spec_decode/prepare_dflash_inputs\.py", "prepare_dflash_inputs_triton"),
    ("metadata_native_wrapper", r"torch/_ops\.py", "__call__"),
    ("metadata_argument_check", r"cann_ops_transformer/utils/arg_check\.py", "wrapper"),
    ("metadata_native_wrapper", r"cann_ops_transformer/ops/attention/mixed_quant_sparse_flash_mla/mixed_quant_sparse_flash_mla\.py", "mixed_quant_sparse_flash_mla_metadata"),
)
_RUNNERS = r"(?:vllm/v1/worker/gpu|vllm_ascend/worker/v2)/model_runner\.py"
_FUNCTIONS += tuple(("request_" + name, _RUNNERS, name) for name in (
    "update_pp_decode_requests", "finish_requests", "free_states", "add_requests", "update_requests", "gather_batch_req_state"))
_FUNCTIONS += tuple((kind, r"vllm/v1/worker/gpu/block_table\.py", name) for kind, name in (
    ("blocktable_apply", "apply_staged_writes"), ("blocktable_gather", "gather_block_tables"),
    ("slot_mappings", "compute_slot_mappings")))
_FUNCTIONS += tuple(("input_" + name, r"vllm/v1/worker/gpu/input_batch\.py", name) for name in (
    "prepare_pos_seq_lens", "combine_sampled_and_draft_tokens", "expand_idx_mapping", "prepare_prefill_inputs"))
_FUNCTIONS += tuple(("sample_" + name, _RUNNERS, name) for name in (
    "sample", "postprocess_sampled", "postprocess_num_computed_tokens", "_copy_num_computed_tokens_to_cpu"))
_FUNCTIONS += tuple(("dflash_" + name, r"vllm/v1/worker/gpu/spec_decode/dflash/speculator\.py", name) for name in (
    "prepare_inputs", "_prepare_inputs", "fuse_hidden_states", "_fuse_hidden_states", "precompute_kv", "precompute_context_kv"))
_BUILDERS = r"vllm_ascend/attention/(?:dsa_v1|dsa_v41|sfa_v1|context_parallel/dsa_cp|context_parallel/sfa_cp)\.py"
_FUNCTIONS += tuple((kind, _BUILDERS, name) for kind, name in (
    ("metadata_builder", "build"), ("metadata_builder", "_build"),
    ("request_metadata", "build_req_metadata"), ("sas_metadata", "_build_sas_metadata"),
    ("qli_metadata", "_build_qli_metadata"), ("draft_metadata_builder", "build_for_drafting"),
    ("draft_request_metadata", "build_req_metadata_for_drafting"),
    ("metadata_capture_builder", "build_for_cudagraph_capture"),
    ("metadata_capture_builder", "build_for_graph_capture")))
_FUNCTIONS += tuple(("metadata_native_wrapper", r"vllm_ascend/attention/(?:mixed_quant_sparse_flash_mla|sparse_flash_mla)\.py", name)
                    for name in ("mixed_quant_sparse_flash_mla_metadata_fp8", "mixed_quant_sparse_flash_mla_metadata_turboquant",
                                 "sparse_flash_mla_metadata", "mq_meta_adapter"))
_FUNCTIONS += tuple(("sfa_metadata_prepare", r"vllm_ascend/attention/sfa_v1\.py", name) for name in (
    "build_smla_metadata", "generate_smla_plan", "_prepare_parallel_metadata", "_prepare_nope_metadata", "_build_with_metadata_view"))
_FUNCTIONS += tuple((kind, r"torch_npu/npu/streams\.py", name) for kind, name in (
    ("stream_dependency", "wait_stream"), ("stream_dependency", "wait_event"),
    ("cpu_sync", "synchronize"), ("stream_event_record", "record_event")))
_PATTERNS_BY_NAME = defaultdict(list)
for _function_kind, _function_path, _function_name in _FUNCTIONS:
    _PATTERNS_BY_NAME[_function_name].append((_function_kind,
        re.compile(r"(?:^|/)" + _function_path + r"\(\d+\): " + _function_name + r"$")))
_ROLES = ("outer_propose", "next_execute", "runner_inputs", "runner_attn", "state_attn")
_BASIS = "HOST_CALL_ORDER_AND_GRAPH_ENTRY_INFERRED"
_LIMITATIONS = (
    "Python parent IDs verify Host call ownership, not Host-to-kernel causality.",
    "Device-round identity is inferred from unique graph entry and Host call order; no kernel flow proof.",
    "Full Host call intervals may start before the HC cycle or finish after it; they are not clipped to U.",
    "Elapsed nested calls include waits and descheduling and must not be added as independent costs.",
)


def _kind(name):
    normalized = str(name).replace("\\", "/")
    # Most trace records are device tasks or unrelated Python functions. Index
    # by the exact function suffix before testing package-path expressions.
    function_name = normalized.rpartition(": ")[2]
    found = next((kind for kind, pattern in _PATTERNS_BY_NAME.get(function_name, ()) if pattern.search(normalized)), None)
    if found:
        return found
    # Stream dependency APIs enqueue ordering. They are never CPU-blocking wait.
    if re.search(r"(?:aclrtStreamWaitEvent|(?:Stream|Event)::wait(?:_event|_stream)?)$", normalized, re.I):
        return "stream_dependency"
    if re.search(r"^<built-in method wait of ", normalized):
        return "cpu_wait"
    if re.search(r"^(?:Event|Stream)::synchronize$|^AscendCL@aclrtSynchronize(?:Event|Stream|Device)$|^<built-in method synchronize of ", normalized):
        return "cpu_sync"
    if normalized.startswith("AscendCL@") and "GetWorkspaceSize" in normalized:
        return "native_workspace"
    if re.match(r"^<built-in method (?:mixed_quant_sparse_flash_mla_metadata|sparse_flash_mla_metadata) of ", normalized):
        return "metadata_native_call"
    if normalized in ("aten::copy_", "aten::_to_copy", "aten::to") or re.search(r"^AscendCL@aclrtMemcpy(?:Async)?$", normalized):
        return "host_copy"
    return None


_SOURCE_IDS = ("connectionid", "correlationid", "streamid", "physicstreamid", "taskid", "modelid",
               "deviceid", "contextid", "batchid", "subtaskid", "flowid", "externalid", "sequencenumber")
_DEVICE_IDS = {
    "device_id": ("deviceid",), "model_id": ("modelid",),
    "stream_id": ("physicstreamid", "streamid"), "task_id": ("taskid",),
    "context_id": ("contextid",), "batch_id": ("batchid",),
    "subtask_id": ("subtaskid",), "task_type": ("tasktype",),
    "connection_id": ("connectionid",), "correlation_id": ("correlationid",),
}


def _source_ids(args, raw):
    """Retain only identifier scalars from JSON, never arbitrary repr/objects."""
    def scalar(value):
        if isinstance(value, (str, int, float, bool, Decimal)):
            return str(value)
        return None
    result = {key: scalar(args[key]) for key in _SOURCE_IDS if key in args and scalar(args[key]) is not None}
    for key in ("id", "bind_id"):
        if scalar(raw.get(key)) is not None:
            result["raw_" + key] = scalar(raw[key])
    if isinstance(raw.get("id2"), dict):
        for key in ("global", "local"):
            if scalar(raw["id2"].get(key)) is not None:
                result["raw_id2_" + key] = scalar(raw["id2"][key])
    return result


def _stamp(path):
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns


def collect_host_events(core, trace_path, offset, expected_sha256, progress=None, *,
                        retain_device_events=False, device_trace_metadata=None):
    """Stream every record, retaining supported Python and CPU/native calls.

    Event indexes remain the original one-based trace indexes, including metadata
    and non-retained events.  The source must stay stable and match its expected
    hash.  Process metadata is applied after scanning, including late M records.

    Optional device retention is an independent, unfiltered trace-device
    catalog, not inferred Host ownership. It includes DMA and unknown task
    names even when absent from kernel CSV. Clock-scan process names classify
    late metadata without a second pass. Source IDs and complete original
    intervals survive; zero duration is retained and never enlarged.
    """
    if not isinstance(expected_sha256, str) or not expected_sha256:
        raise ValueError("An expected trace SHA256 is required")
    path = Path(trace_path)
    access = core._io_path(path)
    before = _stamp(access)
    offset = int(offset)
    events, processes, counts, scan = [], {}, Counter(), {}
    known_processes = ({core.residual_id(pid): str(name) for pid, name in
                        ((device_trace_metadata or {}).get("process_names") or {}).items()}
                       if retain_device_events else {})
    if retain_device_events and (device_trace_metadata or {}).get("sha256") not in (None, expected_sha256):
        raise ValueError("Device process metadata SHA256 differs from the verified trace")
    device_candidates, device_counts, device_bad_intervals = [], Counter(), Counter()
    last = time.monotonic()
    for number, raw in enumerate(core.residual_iter_trace_events(path, stats=scan), 1):
        counts["trace_events"] += 1
        if raw.get("ph") == "M" and raw.get("name") == "process_name":
            processes[core.residual_id(raw.get("pid"))] = str(core.residual_args(raw).get("name", ""))
        if retain_device_events and raw.get("ph") == "X":
            device_counts["scanned_x_events"] += 1
            args = core.residual_args(raw)
            pid, tid = core.residual_id(raw.get("pid")), core.residual_id(raw.get("tid"))
            hint = core.residual_device_hint(raw, args)
            process = processes.get(pid, known_processes.get(pid))
            # Unknown processes must survive until a possible late M record.
            # The normal caller supplies the clock scan's complete names.
            if hint or process is None or core.residual_device_process(process):
                try:
                    start = core.a_ns(raw["ts"]) + offset
                    duration = core.a_ns(raw["dur"])
                except (KeyError, TypeError, ValueError, ArithmeticError):
                    device_bad_intervals[(pid, hint, "invalid")] += 1
                else:
                    if duration < 0:
                        device_bad_intervals[(pid, hint, "negative")] += 1
                    else:
                        item = {"trace_event_index": number, "event_index": number,
                                "name": str(raw.get("name", "")), "cat": str(raw.get("cat", "")),
                                "pid": pid, "tid": tid, "ph": "X", "ts": str(raw["ts"]), "dur": str(raw["dur"]),
                                "start_ns": start, "end_ns": start+duration, "duration_ns": duration,
                                "device_hint": hint, "source_ids": _source_ids(args, raw),
                                "source_kind": "trace_device_event", "op_type": str(args.get("optype", ""))}
                        for key, choices in _DEVICE_IDS.items():
                            item[key] = next((core.residual_id(args[name]) for name in choices
                                              if isinstance(args.get(name), (str, int, float, bool, Decimal))), None)
                        device_candidates.append(item)
        categories = re.split(r"[,; ]+", str(raw.get("cat", "")))
        kind = _kind(raw.get("name", ""))
        python = "python_function" in categories
        host_category = bool(set(categories).intersection(("cpu_op", "cann", "cann_api", "runtime")))
        native_name = str(raw.get("name", "")).startswith("AscendCL@")
        if raw.get("ph") == "X" and kind and (python or host_category or native_name):
            try:
                start = core.a_ns(raw["ts"]) + offset
                duration = core.a_ns(raw["dur"])
            except (KeyError, TypeError, ValueError, ArithmeticError):
                counts["invalid_supported_events"] += 1
            else:
                if duration > 0:
                    args = core.residual_args(raw)
                    events.append({"event_index": number, "name": str(raw["name"]), "kind": kind,
                                   "pid": core.residual_id(raw.get("pid")),
                                   "tid": core.residual_id(raw.get("tid")), "cat": str(raw.get("cat", "")),
                                   "ts": str(raw["ts"]), "dur": str(raw["dur"]),
                                   "start_ns": start, "end_ns": start + duration, "duration_ns": duration,
                                   "python_id": core.residual_id(args.get("pythonid")),
                                   "python_parent_id": core.residual_id(args.get("pythonparentid")),
                                   "thread_id": core.residual_id(args.get("threadid")),
                                   "source_ids": _source_ids(args, raw), "ph": "X",
                                   "event_type": "python" if python else "native_api" if native_name else "cpu_api",
                                   "device_hint": core.residual_device_hint(raw, args)})
                else:
                    counts["nonpositive_supported_events"] += 1
        now = time.monotonic()
        if progress and (number % 100000 == 0 or now - last >= 5):
            progress("Host function scan: " + format(number, ",") + " events")
            last = now
    if _stamp(access) != before or scan.get("sha256") != expected_sha256:
        raise ValueError("Trace changed or SHA256 differs during Host-function collection")
    retained = []
    for event in events:
        event["process_name"] = processes.get(event["pid"], "")
        if event["device_hint"] or core.residual_device_process(event["process_name"]):
            counts["device_events_rejected"] += 1
            continue
        retained.append(event)
        counts[event["kind"]] += 1
    counts["retained_events"] = len(retained)
    result = {"events": retained, "sha256": scan["sha256"],
              "process_names": processes, "counts": dict(counts)}
    if retain_device_events:
        device_events = []
        task_types, categories = Counter(), Counter()
        for (pid, hint, reason), count in device_bad_intervals.items():
            process = processes.get(pid, known_processes.get(pid, ""))
            if hint or core.residual_device_process(process):
                key = "negative_device_duration_events" if reason == "negative" else "invalid_device_timed_events"
            else:
                key = "rejected_nondevice_invalid_intervals"
            device_counts[key] += count
        for event in device_candidates:
            process = processes.get(event["pid"], known_processes.get(event["pid"], ""))
            verified_process = core.residual_device_process(process)
            if not event["device_hint"] and not verified_process:
                device_counts["nondevice_unknown_process_candidates_rejected"] += 1
                continue
            event.update(process_name=process, device_process_verified=verified_process,
                         device_classification="NATIVE_HINT_AND_PROCESS" if event["device_hint"] and verified_process else
                                               "NATIVE_HINT" if event["device_hint"] else "TRACE_PROCESS_METADATA")
            device_events.append(event)
            device_counts["retained_device_x_events"] += 1
            device_counts["zero_duration_device_events"] += int(event["duration_ns"] == 0)
            task_types[event.get("task_type") or "unspecified"] += 1
            categories[event.get("cat") or "unspecified"] += 1
        result.update(trace_device_events=device_events, device_event_scan={
            "schema": "u_trace_device_event_scan_v1", "counts": dict(device_counts),
            "task_type_counts": dict(task_types), "category_counts": dict(categories),
            "supported_device_phase": "X; B/E durations are not reconstructed",
            "intervals_clipped": False, "topn_filter": False,
            "scope": "All nonnegative-duration trace X device events, independent of Host name whitelists or CSV membership. "
                     "Device process/native hints identify catalog entries, not their Python submitter or round. "
                     "Negative/invalid intervals are diagnostic counts, never fabricated tasks."})
    return result


def _lane(event):
    return str(event.get("pid")), str(event.get("tid"))


def _contains(parent, child):
    return parent["start_ns"] <= child["start_ns"] <= child["end_ns"] <= parent["end_ns"]


def _child(parent, event):
    return (parent.get("python_id") is not None
            and event.get("python_parent_id") == parent["python_id"]
            and _lane(parent) == _lane(event) and _contains(parent, event))


def _python_relation(owner, event, index):
    """Return only an explicit, unique, contained Python ancestor chain."""
    category = str(event.get("cat", ""))
    if event.get("event_type") != "python" and "python_function" not in re.split(r"[,; ]+", category):
        return None
    current, chain, used = event, [], {event["event_index"]}
    while current.get("python_parent_id") is not None:
        key = (*_lane(current), str(current["python_parent_id"]))
        parents = [p for p in index.get(key, []) if p["event_index"] not in used and _contains(p, current)]
        if len(parents) != 1:
            return None
        parent = parents[0]
        chain.append(parent["event_index"])
        if parent["event_index"] == owner["event_index"]:
            return {"relation": "python_parent_id" if len(chain) == 1 else "python_ancestor_chain",
                    "depth": len(chain), "parent_event_index": chain[0], "ancestor_event_indexes": chain}
        used.add(parent["event_index"])
        current = parent
    return None


def _child_ref(owner, event, index):
    if event["event_index"] == owner["event_index"] or not _contains(owner, event):
        return None
    proof = _python_relation(owner, event, index)
    if proof is None:
        native = event.get("event_type") == "native_api"
        native_thread = native and (str(event.get("thread_id")) == str(owner.get("tid"))
                                    or str(event.get("tid")) == str(owner.get("tid")))
        if _lane(owner) != _lane(event) and not native_thread:
            return None
        proof = {"relation": "native_thread_time_containment_only" if native else "time_containment_only",
                 "depth": None, "parent_event_index": None, "ancestor_event_indexes": []}
    ref = {key: event.get(key) for key in ("event_index", "name", "kind", "start_ns", "end_ns", "duration_ns", "event_type")}
    ref.update(proof, owner_event_index=owner["event_index"])
    ref["parent_chain_verified"] = proof["relation"] in ("python_parent_id", "python_ancestor_chain")
    if event.get("kind") == "stream_dependency":
        ref["scope_note"] = "Enqueues stream/event ordering; not a CPU-blocking wait duration"
    elif event.get("kind") == "metadata_capture_builder":
        ref["scope_note"] = "An actual capture-builder Python call, not inferred Python execution for every graph replay"
    elif event.get("kind") == "host_copy":
        ref["scope_note"] = "Host copy/conversion API wall range; not proof of DMA duration or a copy on every to() call"
    return ref


def _empty_child_layer():
    return {"by_role": {role: [] for role in (
                "outer_propose", "preparation_dp_aggregate", "runner_inputs", "runner_attn", "state_attn")},
            "prefix_partition": {"status": "UNCONFIRMED", "parts": [],
                                 "dp_sync": {"events": [], "all_reduce_events": [],
                                             "scope": "Nested observed sync/collective envelopes; inclusive durations are not additive"},
                                 "wait_candidates": []},
            "missing": [], "reasons": [],
            "scope": "References preserve full recorded calls. depth=1 requires an explicit Python parent ID; "
                     "time containment is separate, and missing children are never fabricated."}


def _child_layer(pair, events, index):
    result = _empty_child_layer()
    lookup = {e["event_index"]: e for e in events}
    roots = {role: lookup.get(pair["functions"].get(role)) for role in (
        "outer_propose", "runner_inputs", "runner_attn", "state_attn")}
    owner = lookup.get(pair["functions"].get("next_execute"))
    inputs = roots["runner_inputs"]
    prefix = (owner["start_ns"], inputs["start_ns"]) if owner and inputs else None
    roots["preparation_dp_aggregate"] = owner
    for role, scope in roots.items():
        if scope is None or role == "preparation_dp_aggregate" and prefix is None:
            result["missing"].append(role)
            continue
        for event in events:
            if role == "preparation_dp_aggregate" and not (event["start_ns"] < prefix[1] and event["end_ns"] > prefix[0]):
                continue
            ref = _child_ref(scope, event, index)
            if ref:
                result["by_role"][role].append(ref)
        result["by_role"][role].sort(key=lambda r: (r["start_ns"], r["end_ns"], r["event_index"]))
    if prefix is None:
        result["reasons"].append("PREFIX_ENDPOINTS_NOT_PAIRED")
        return result
    left, right = prefix
    block = result["prefix_partition"]
    # Unknown work remains one honest aggregate until a unique, explicitly
    # owned outer dispatch is observed. Never label the whole aggregate DP wait.
    block["parts"] = [{"key": "preparation_dp_aggregate", "start_ns": left, "end_ns": right,
                        "duration_ns": right-left, "event_indexes": [],
                        "scope": "Execute entry to inputs entry; unpartitioned local/graph/coordination work and gaps"}]
    refs = result["by_role"]["preparation_dp_aggregate"]
    proven = [r for r in refs if r["relation"] in ("python_parent_id", "python_ancestor_chain")]
    dispatches = [lookup[r["event_index"]] for r in proven if r["kind"] in ("dp_dispatch", "dp_dispatch_patch")]
    outer = [e for e in dispatches if not any(other["event_index"] != e["event_index"]
             and _python_relation(other, e, index) is not None for other in dispatches)]
    selected = outer[0] if len(outer) == 1 else None
    if len(outer) > 1:
        block["status"] = "AMBIGUOUS"
        result["reasons"].append("MULTIPLE_OUTER_DISPATCHES")
    elif selected and left <= selected["start_ns"] <= selected["end_ns"] <= right:
        block["status"] = "VERIFIED_HOST_BOUNDARIES"
        block["parts"] = []
        for key, start, end, ids, scope in (
            ("pre_dispatch", left, selected["start_ns"], [], "Local entry work and gaps before dispatch; not pure CPU busy"),
            ("dispatch_envelope", selected["start_ns"], selected["end_ns"], [selected["event_index"]],
             "Inclusive graph-selection and DP-coordination envelope; not pure DP synchronization or wait"),
            ("post_dispatch", selected["end_ns"], right, [], "Post-dispatch preparation and gaps before runner inputs")):
            block["parts"].append({"key": key, "start_ns": start, "end_ns": end,
                                   "duration_ns": end-start, "event_indexes": ids, "scope": scope})
    elif selected:
        block["status"] = "ORDER_CONFLICT"
        result["reasons"].append("DISPATCH_CROSSES_PREFIX_BOUNDARY")
    else:
        block["status"] = "NOT_RECORDED"
        result["reasons"].append("NO_VERIFIED_OUTER_DISPATCH")
    syncs = [r for r in proven if r["kind"] == "dp_sync"]
    block["dp_sync"]["events"] = syncs
    dp_roots = [lookup[r["event_index"]] for r in syncs] + ([selected] if selected else [])
    collectives = [r for r in proven if r["kind"] == "all_reduce"
                   and any(_python_relation(root, lookup[r["event_index"]], index) is not None for root in dp_roots)]
    block["dp_sync"]["all_reduce_events"] = collectives
    for ref in proven:
        if ref["kind"] != "cpu_wait":
            continue
        owners = [r for r in collectives if _python_relation(lookup[r["event_index"]], lookup[ref["event_index"]], index)]
        if len(owners) == 1:
            block["wait_candidates"].append({**ref, "owning_all_reduce_event_index": owners[0]["event_index"],
                "parent_chain_verified": True,
                "wait_proof": "EXPLICIT_PYTHON_ANCESTOR_ALL_REDUCE_WITHIN_DP_SCOPE",
                "scope_note": "Observed blocking Work/pybind wait candidate; nested wrappers must be unioned, not summed"})
    return result


def pair_rounds(core, rounds, events):
    """Return all HC rounds, preserving missing or ambiguous Host associations.

    The unique Ascend main-graph entry inside a cycle identifies the candidate
    next execute call.  Its preceding core execute/sample family identifies the
    current propose.  This sequence convention is deliberately labeled inferred.
    """
    ordered = sorted(({**e, "kind": e.get("kind") or _kind(e.get("name", "")),
                       "event_type": e.get("event_type") or ("python" if "python_function" in
                           re.split(r"[,; ]+", str(e.get("cat", ""))) else "native_api" if
                           str(e.get("name", "")).startswith("AscendCL@") else "cpu_api")} for e in events),
                     key=lambda e: (e["start_ns"], e["end_ns"], e["event_index"]))
    valid = [e for e in ordered if not e.get("device_hint")
             and not core.residual_device_process(e.get("process_name", ""))]
    by_kind = defaultdict(list)
    python_index = defaultdict(list)
    for event in valid:
        by_kind[event.get("kind") or _kind(event.get("name", ""))].append(event)
        if event.get("python_id") is not None and "python_function" in re.split(r"[,; ]+", str(event.get("cat", ""))):
            python_index[(*_lane(event), str(event["python_id"]))].append(event)
    cores = by_kind["core_execute"]
    lanes = sorted({_lane(e) for e in cores if e.get("pid") is not None and e.get("tid") is not None})
    results = []
    for number, raw_round in enumerate(rounds):
        start, end = int(raw_round["start_ns"]), int(raw_round["end_ns"])
        result = {"round_id": raw_round.get("round_id", "round-" + str(number)),
                  "start_ns": start, "end_ns": end, "status": "PARTIAL",
                  "functions": dict.fromkeys(_ROLES),
                  "context": dict.fromkeys(("current_execute", "current_sample", "next_main_graph")),
                  "small_functions": [], "missing": [], "reasons": [], "candidates": {},
                  "pairing_basis": _BASIS, "limitations": list(_LIMITATIONS),
                  "ownership_verified": dict.fromkeys(_ROLES, False),
                  "host_ownership_status": "UNCONFIRMED", "round_identity_status": "UNCONFIRMED",
                  "ownership_basis": "SAME_PID_TID_PYTHON_PARENT_ID_AND_TIME_CONTAINMENT",
                  "order_verified": False,
                  "child_layer": _empty_child_layer(),
                  "worker_lanes": [{"pid": p, "tid": t} for p, t in lanes]}
        results.append(result)
        if end <= start:
            result["reasons"].append("NONPOSITIVE_HC_CYCLE")
            result["missing"] = list(_ROLES)
            continue
        ambiguous = []

        def select(role, candidates):
            result["candidates"][role] = [e["event_index"] for e in candidates]
            if len(candidates) > 1:
                ambiguous.append(role)
                result["reasons"].append("AMBIGUOUS_" + role.upper())
                return None
            if not candidates:
                result["reasons"].append("MISSING_" + role.upper())
                return None
            return candidates[0]

        if len(lanes) != 1:
            result["status"] = "AMBIGUOUS" if lanes else "PARTIAL"
            result["round_identity_status"] = "AMBIGUOUS" if lanes else "UNCONFIRMED"
            result["reasons"].append("MULTIPLE_WORKER_LANES" if lanes else "NO_WORKER_LANE")
            result["missing"] = list(_ROLES)
            continue
        graph = select("next_main_graph", [e for e in by_kind["main_graph"]
                                          if _lane(e) == lanes[0] and start <= e["start_ns"] < end])
        owner = None
        if graph:
            result["context"]["next_main_graph"] = graph["event_index"]
            owner = select("next_execute", [e for e in cores if _child(e, graph)])
        if owner:
            result["functions"]["next_execute"] = owner["event_index"]
            selected_children = {}
            for role in ("runner_inputs", "runner_attn", "state_attn"):
                event = select(role, [e for e in by_kind[role] if _child(owner, e)])
                if event:
                    result["functions"][role] = event["event_index"]
                    selected_children[role] = event
            inputs, runner_attn, metadata = (selected_children.get(role) for role in (
                "runner_inputs", "runner_attn", "state_attn"))
            if inputs and metadata:
                parent_order = inputs["end_ns"] <= metadata["start_ns"] and metadata["end_ns"] <= graph["start_ns"]
                attn_order = (runner_attn is not None and inputs["end_ns"] <= runner_attn["start_ns"]
                              and runner_attn["end_ns"] <= metadata["start_ns"])
                result["order_verified"] = parent_order and attn_order
                if not parent_order or runner_attn is not None and not attn_order:
                    result["reasons"].append("INPUT_METADATA_GRAPH_ORDER_CONFLICT")
                if runner_attn is not None and not attn_order:
                    result["reasons"].append("RUNNER_ATTN_ORDER_CONFLICT")
            prior = [e for e in cores if _lane(e) == _lane(owner) and e["start_ns"] < owner["start_ns"]]
            current = None
            if prior:
                last_start = max(e["start_ns"] for e in prior)
                current = select("current_execute", [e for e in prior if e["start_ns"] == last_start])
            else:
                result["reasons"].append("MISSING_CURRENT_EXECUTE")
            if current:
                result["context"]["current_execute"] = current["event_index"]
                sample = select("current_sample", [e for e in by_kind["core_sample"]
                                if _lane(e) == _lane(owner)
                                and current["end_ns"] <= e["start_ns"] < e["end_ns"] <= owner["start_ns"]])
                if sample:
                    result["context"]["current_sample"] = sample["event_index"]
                    propose = select("outer_propose", [e for e in by_kind["outer_propose"] if _child(sample, e)])
                    if propose:
                        result["functions"]["outer_propose"] = propose["event_index"]
            families = [e for e in valid if e["event_index"] in
                        set(v for v in (*result["context"].values(), *result["functions"].values()) if v is not None)]
            primary = {e["event_index"] for e in families}
            for event in valid:
                if event["event_index"] in primary or event.get("kind") in ("core_execute", "core_sample", "outer_propose"):
                    continue
                containers = [p for p in families if _lane(p) == _lane(event) and _contains(p, event)]
                if containers:
                    direct = [p for p in containers if _child(p, event)]
                    result["small_functions"].append({"kind": event.get("kind") or _kind(event["name"]),
                        "event_index": event["event_index"],
                        "relation": "python_parent_id" if direct else "time_containment_only",
                        "parent_event_indexes": [p["event_index"] for p in direct or containers]})
        result["missing"] = [role for role, value in result["functions"].items() if value is None]
        result["ownership_verified"] = {role: value is not None for role, value in result["functions"].items()}
        result["host_ownership_status"] = ("VERIFIED" if not result["missing"] else
                                            "PARTIAL" if any(result["ownership_verified"].values()) else "UNCONFIRMED")
        if ambiguous:
            result["status"] = "AMBIGUOUS"
            result["round_identity_status"] = "AMBIGUOUS"
        elif "INPUT_METADATA_GRAPH_ORDER_CONFLICT" in result["reasons"]:
            result["round_identity_status"] = "ORDER_CONFLICT"
        elif not result["missing"] and result["context"]["current_execute"] is not None and result["order_verified"]:
            result["status"] = "MATCHED"
            result["round_identity_status"] = "INFERRED_BY_HOST_CALL_ORDER"
        result["child_layer"] = _child_layer(result, valid, python_index)
    return results
