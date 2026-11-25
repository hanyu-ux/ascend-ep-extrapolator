"""Collect complete selected Host calls and conservatively pair HC rounds.

Python parent IDs establish Host ownership.  Graph-entry ordering only infers
the corresponding device round; it is not a kernel submission/flow proof.
"""

from collections import Counter, defaultdict
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
)
_PATTERNS = [(kind, re.compile(r"(?:^|/)" + path + r"\(\d+\): " + name + r"$"))
             for kind, path, name in _FUNCTIONS]
_ROLES = ("outer_propose", "next_execute", "runner_inputs", "state_attn")
_BASIS = "HOST_CALL_ORDER_AND_GRAPH_ENTRY_INFERRED"
_LIMITATIONS = (
    "Python parent IDs verify Host call ownership, not Host-to-kernel causality.",
    "Device-round identity is inferred from unique graph entry and Host call order; no kernel flow proof.",
    "Full Host call intervals may start before the HC cycle or finish after it; they are not clipped to U.",
    "Elapsed nested calls include waits and descheduling and must not be added as independent costs.",
)


def _kind(name):
    normalized = str(name).replace("\\", "/")
    return next((kind for kind, pattern in _PATTERNS if pattern.search(normalized)), None)


def _stamp(path):
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns


def collect_host_events(core, trace_path, offset, expected_sha256, progress=None):
    """Stream all trace records, retaining every supported timed Python call.

    Event indexes remain the original one-based trace indexes, including metadata
    and non-retained events.  The source must stay stable and match its expected
    hash.  Process metadata is applied after scanning, including late M records.
    """
    if not isinstance(expected_sha256, str) or not expected_sha256:
        raise ValueError("An expected trace SHA256 is required")
    path = Path(trace_path)
    access = core._io_path(path)
    before = _stamp(access)
    offset = int(offset)
    events, processes, counts, scan = [], {}, Counter(), {}
    last = time.monotonic()
    for number, raw in enumerate(core.residual_iter_trace_events(path, stats=scan), 1):
        counts["trace_events"] += 1
        if raw.get("ph") == "M" and raw.get("name") == "process_name":
            processes[core.residual_id(raw.get("pid"))] = str(core.residual_args(raw).get("name", ""))
        categories = re.split(r"[,; ]+", str(raw.get("cat", "")))
        kind = _kind(raw.get("name", ""))
        if raw.get("ph") == "X" and "python_function" in categories and kind:
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
                                   "device_hint": core.residual_device_hint(raw, args)})
                    counts[kind] += 1
                else:
                    counts["nonpositive_supported_events"] += 1
        now = time.monotonic()
        if progress and (number % 100000 == 0 or now - last >= 5):
            progress("Host function scan: " + format(number, ",") + " events")
            last = now
    if _stamp(access) != before or scan.get("sha256") != expected_sha256:
        raise ValueError("Trace changed or SHA256 differs during Host-function collection")
    for event in events:
        event["process_name"] = processes.get(event["pid"], "")
    counts["retained_events"] = len(events)
    return {"events": events, "sha256": scan["sha256"],
            "process_names": processes, "counts": dict(counts)}


def _lane(event):
    return str(event.get("pid")), str(event.get("tid"))


def _contains(parent, child):
    return parent["start_ns"] <= child["start_ns"] <= child["end_ns"] <= parent["end_ns"]


def _child(parent, event):
    return (parent.get("python_id") is not None
            and event.get("python_parent_id") == parent["python_id"]
            and _lane(parent) == _lane(event) and _contains(parent, event))


def pair_rounds(core, rounds, events):
    """Return all HC rounds, preserving missing or ambiguous Host associations.

    The unique Ascend main-graph entry inside a cycle identifies the candidate
    next execute call.  Its preceding core execute/sample family identifies the
    current propose.  This sequence convention is deliberately labeled inferred.
    """
    ordered = sorted(({**e, "kind": e.get("kind") or _kind(e.get("name", ""))} for e in events),
                     key=lambda e: (e["start_ns"], e["end_ns"], e["event_index"]))
    valid = [e for e in ordered if not e.get("device_hint")
             and not core.residual_device_process(e.get("process_name", ""))]
    by_kind = defaultdict(list)
    for event in valid:
        by_kind[event.get("kind") or _kind(event.get("name", ""))].append(event)
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
            for role in ("runner_inputs", "state_attn"):
                event = select(role, [e for e in by_kind[role] if _child(owner, e)])
                if event:
                    result["functions"][role] = event["event_index"]
                    selected_children[role] = event
            inputs, metadata = selected_children.get("runner_inputs"), selected_children.get("state_attn")
            if inputs and metadata:
                result["order_verified"] = inputs["end_ns"] <= metadata["start_ns"] and metadata["end_ns"] <= graph["start_ns"]
                if not result["order_verified"]:
                    result["reasons"].append("INPUT_METADATA_GRAPH_ORDER_CONFLICT")
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
    return results
