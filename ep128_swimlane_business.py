"""Saved Worker business scopes and a continuous, exclusive Host submission line.

This is not an EngineCore/RPC/output-thread model. Only exact paired Worker
event IDs and explicit Python parent chains may assign a recorded interval.
"""

from collections import defaultdict
from copy import deepcopy
import re


SCHEMA = "u_worker_business_v1"
OUTER_ROLES = ("current_execute_n", "sample_tokens_n", "next_execute_n1")
OUTER_LABELS = {"current_execute_n": "execute_model[n] · Host提交", "sample_tokens_n": "sample_tokens[n] · Host提交",
                "next_execute_n1": "execute_model[n+1] · Host提交"}
SAMPLE_KINDS = {"sample_sample": "sample_submit", "sample_postprocess_sampled": "sample_postprocess",
                "sample_postprocess_num_computed_tokens": "sample_postprocess",
                "sample__copy_num_computed_tokens_to_cpu": "sample_copy"}
LABELS = {"sample_submit": "采样/拒绝采样 · Host提交", "sample_postprocess": "采样后处理 · Host范围",
          "sample_copy": "采样计数拷贝 · Host提交范围", "propose": "propose · 候选Host提交",
          "prefix": "下一轮前处理/DP范围（非函数）", "runner_inputs": "runner.prepare_inputs",
          "runner_attn": "runner.prepare_attn · 块表/slot", "state_attn": "model_state.prepare_attn · metadata",
          "next_graph": "下一主run_fullgraph · Host提交", "sample_remainder": "sample_tokens其余wall",
          "execute_remainder": "execute_model其余wall", "current_execute": "当前主execute_model · Host范围",
          "UNKNOWN_GAP": "未记录业务区间（UNKNOWN_GAP）", "UNKNOWN_OVERLAP": "归属不明（重叠scope）"}
SCOPE = (
    "One saved Worker thread: current execute_model[n], sample_tokens[n], then next execute_model[n+1]. "
    "Complete original outer intervals are retained, including negative starts and ends beyond this device HC cycle. "
    "The exclusive business line is wall time, not CPU busy or a sum of inclusive functions. "
    "Gaps are UNKNOWN_GAP, not idle, RPC duration or device waiting. Overlapping incomparable scopes stay unknown. "
    "Host submission/return is distinct from accelerator completion. No missing EngineCore or output-thread timing is invented."
)
MECHANISM_NOTES = (
    {"kind": "SOURCE_MECHANISM", "text": "Worker主线程按execute_model → sample_tokens → 下一execute_model推进；这里仅显示已配对Worker方法，未测得的EngineCore/RPC和输出线程不补时间。"},
    {"kind": "SOURCE_MECHANISM", "text": "这些Host方法可在设备任务完成前返回；依赖可能由设备stream/event或明确的Host等待保证，不能从函数返回推断NPU完成。"},
    {"kind": "SOURCE_MECHANISM", "text": "sample/propose条表示采样和候选生成相关工作的Host提交范围，不是相应NPU算子的执行历时。"},
    {"kind": "SOURCE_MECHANISM", "text": "Host提交范围可包含参数准备、入队和已有的显式Host等待；方法可异步返回，不代表每个子调用都不等待。未记录的缺口不能指认为等NPU。"},
    {"kind": "SOURCE_MECHANISM", "text": "前处理/DP是下一execute入口至runner.prepare_inputs入口的合成范围，不是一个名为前处理/DP的函数，也不等于纯DP等待。"},
)


def _ns(value):
    if type(value) is int:
        return value
    if isinstance(value, str) and re.fullmatch(r"[+-]?\d+", value):
        return int(value)
    raise ValueError("Exact integer nanosecond coordinates are required")


def _id(value):
    if type(value) is int:
        return value
    if isinstance(value, str) and value:
        return int(value) if re.fullmatch(r"[+-]?\d+", value) else value
    return None


def _lane(event):
    if event.get("pid") in (None, "") or event.get("tid") in (None, ""):
        return None
    return str(event["pid"]), str(event["tid"])


def _contains(parent, child):
    return _ns(parent["start_ns"]) <= _ns(child["start_ns"]) <= _ns(child["end_ns"]) <= _ns(parent["end_ns"])


def _timing(event, origin):
    start, end = _ns(event["start_ns"]), _ns(event["end_ns"])
    if end < start:
        raise ValueError("Reversed saved Worker event")
    return {"start_ns": str(start), "end_ns": str(end), "duration_ns": end-start,
            "start_ms": (start-origin)/1e6, "end_ms": (end-origin)/1e6, "duration_ms": (end-start)/1e6}


def _python(event):
    return event.get("event_type") == "python" or "python_function" in re.split(r"[,; ]+", str(event.get("cat", "")))


def _path(owner, event, parents):
    """Return the actual unique same-thread parent path, never containment alone."""
    if (not _python(event) or owner.get("python_id") is None or _lane(owner) is None
            or _lane(owner) != _lane(event) or not _contains(owner, event)):
        return None
    owner_id = _id(owner.get("event_index"))
    current, used, result = event, {_id(event.get("event_index"))}, []
    while current.get("python_parent_id") is not None:
        key = (*_lane(current), str(current["python_parent_id"]))
        candidates = [parent for parent in parents.get(key, []) if _id(parent.get("event_index")) not in used and _contains(parent, current)]
        if len(candidates) != 1:
            return None
        current = candidates[0]
        number = _id(current.get("event_index"))
        result.append(number)
        if number == owner_id:
            return result
        used.add(number)
    return None


def _piece(start, end, origin, role, kind, event=None, synthetic=False, source_ids=(), reason=None, **extra):
    return {"role": role, "kind": kind, "name": LABELS.get(kind, kind), "label": LABELS.get(kind, kind),
            "event_index": _id(event.get("event_index")) if event is not None else None,
            "source_name": event.get("name") if event is not None else None,
            "source_event_indexes": list(dict.fromkeys(source_ids)), "synthetic": synthetic,
            "selection_reason": reason or "EXPLICIT_PAIRED_WORKER_SCOPE",
            **_timing({"start_ns": start, "end_ns": end}, origin), **extra}


def _merge(pieces):
    result = []
    for item in pieces:
        key = (item["role"], item["kind"], item["synthetic"],
               tuple(item.get("candidate_event_indexes", [])) if item["kind"] == "UNKNOWN_OVERLAP" else ())
        if result:
            last = result[-1]
            old_key = (last["role"], last["kind"], last["synthetic"],
                       tuple(last.get("candidate_event_indexes", [])) if last["kind"] == "UNKNOWN_OVERLAP" else ())
            if key == old_key and last["end_ns"] == item["start_ns"]:
                last["end_ns"], last["end_ms"] = item["end_ns"], item["end_ms"]
                last["duration_ns"] += item["duration_ns"]
                last["duration_ms"] = last["duration_ns"]/1e6
                last["source_event_indexes"] = list(dict.fromkeys(last["source_event_indexes"] + item["source_event_indexes"]))
                if last["event_index"] != item["event_index"]:
                    last["event_index"] = None
                continue
        result.append(item)
    return result


def _exclusive_inside(owner, candidates, origin, role, remainder):
    start, end = _ns(owner["start_ns"]), _ns(owner["end_ns"])
    cuts = sorted({start, end, *[stamp for c in candidates for stamp in (c["start"], c["end"]) if start <= stamp <= end]})
    pieces = []
    for a, b in zip(cuts, cuts[1:]):
        if b <= a:
            continue
        active = [c for c in candidates if c["start"] <= a and c["end"] >= b]
        leaves = [c for c in active if not any(c["id"] in other["ancestors"] for other in active)]
        if not active:
            pieces.append(_piece(a, b, origin, role, remainder, owner, source_ids=[_id(owner.get("event_index"))],
                                 reason="PARENT_REMAINING_WALL_NOT_PURE_CPU"))
        elif len(leaves) != 1 or any(c is not leaves[0] and c["id"] not in leaves[0]["ancestors"] for c in active):
            pieces.append(_piece(a, b, origin, role, "UNKNOWN_OVERLAP", synthetic=True,
                source_ids=[c["id"] for c in active], candidate_event_indexes=sorted([c["id"] for c in active], key=str),
                reason="INCOMPARABLE_RECORDED_INNER_SCOPES"))
        else:
            leaf = leaves[0]
            pieces.append(_piece(a, b, origin, role, leaf["kind"], leaf["event"], synthetic=leaf.get("synthetic", False),
                source_ids=leaf.get("source_ids", [leaf["id"]]), reason=leaf.get("reason", "EXPLICIT_PYTHON_PARENT_CHAIN"),
                ancestor_event_indexes=list(leaf["ancestors"])))
    return _merge(pieces)


def build_worker_business(case, origin_ns):
    """Recover paired Worker scopes and one exclusive, gap-explicit business line."""
    origin = _ns(origin_ns)
    output = {"schema": SCHEMA, "case": case.get("case"), "rank": case.get("rank"), "origin_ns": str(origin),
              "status": "UNAVAILABLE", "outer_scopes": [], "pieces": [], "gaps": [], "diagnostics": [],
              "inner_scopes": [], "source_events": [], "trace_observations": [],
              "mechanism_notes": deepcopy(list(MECHANISM_NOTES)), "scope": SCOPE,
              "closed": False, "worker_lane_confirmed": False, "duration_ns": None, "total_duration_ns": None}
    if (case.get("trace") or {}).get("status") != "ALIGNED_X_EVENTS" or case.get("status") == "ERROR":
        output["diagnostics"].append({"code": "UNTRUSTED_CLOCK_OR_SOURCE", "trace_status": (case.get("trace") or {}).get("status"),
                                       "error": case.get("error")})
        return output
    rounds = case.get("rounds") or []
    identity = case.get("selected_round_id") or case.get("suggested_round_id")
    selected = [row for row in rounds if row.get("round_id") == identity] if identity is not None else list(rounds) if len(rounds) == 1 else []
    if len(selected) != 1:
        output["diagnostics"].append({"code": "NO_UNIQUE_SAVED_SELECTED_ROUND", "selected_round_id": identity})
        return output
    row = selected[0]
    output["round_id"] = row.get("round_id")
    pair = row.get("host_pairing") or {}
    context, functions = pair.get("context") or {}, pair.get("functions") or {}
    output.update(pairing_basis=pair.get("pairing_basis"),
                  round_identity_status=pair.get("round_identity_status"),
                  host_ownership_status=pair.get("host_ownership_status"),
                  pairing_limitations=deepcopy(pair.get("limitations") or []))
    raw_lookup, parents = defaultdict(list), defaultdict(list)
    for event in case.get("host_events") or []:
        raw_lookup[_id(event.get("event_index"))].append(event)
        if _python(event) and event.get("python_id") is not None and _lane(event) is not None:
            parents[(*_lane(event), str(event["python_id"]))].append(event)
    used = set()
    def get(number, label):
        number = _id(number)
        events = raw_lookup.get(number, [])
        if number is None or len(events) != 1:
            output["diagnostics"].append({"code": "MISSING_OR_AMBIGUOUS_PAIRED_EVENT", "role": label, "event_index": number})
            return None
        event = events[0]
        try:
            _timing(event, origin)
        except (KeyError, ValueError) as error:
            output["diagnostics"].append({"code": "INVALID_PAIRED_INTERVAL", "role": label, "event_index": number, "error": str(error)})
            return None
        used.add(number)
        return event
    outer_specs = (("current_execute_n", context.get("current_execute"), "current_execute", "core_execute"),
                   ("sample_tokens_n", context.get("current_sample"), "current_sample", "core_sample"),
                   ("next_execute_n1", functions.get("next_execute"), "next_execute", "core_execute"))
    owners = {}
    for role, number, key, expected_kind in outer_specs:
        event = get(number, role)
        if event is not None and event.get("kind") != expected_kind:
            output["diagnostics"].append({"code": "PAIRED_EVENT_KIND_MISMATCH", "role": role,
                "event_index": _id(event.get("event_index")), "expected_kind": expected_kind, "actual_kind": event.get("kind")})
            event = None
        if event is not None:
            owners[role] = event
            output["outer_scopes"].append({"role": role, "name": event.get("name"), "label": OUTER_LABELS[role],
                "event_index": _id(event.get("event_index")), "pid": event.get("pid"), "tid": event.get("tid"),
                "source_context_key": key, "round_scope": "n+1" if role == "next_execute_n1" else "n",
                **_timing(event, origin)})
    if not owners:
        return output
    same_lane = all(_lane(event) is not None for event in owners.values()) and len({_lane(event) for event in owners.values()}) == 1
    output["worker_lane_confirmed"] = same_lane
    for scope in output["outer_scopes"]:
        scope["worker_lane_confirmed"] = same_lane
    if not same_lane:
        output["diagnostics"].append({"code": "OUTER_WORKER_LANE_MISMATCH", "reason": "Not one verified Worker thread; no continuous function ownership is asserted"})
    owner_order = [owners[role] for role in OUTER_ROLES if role in owners]
    if any(_ns(a["end_ns"]) > _ns(b["start_ns"]) for a, b in zip(owner_order, owner_order[1:])):
        output["diagnostics"].append({"code": "OUTER_METHOD_ORDER_OR_OVERLAP_CONFLICT"})
    nested = {}
    sample = owners.get("sample_tokens_n")
    if sample is not None:
        candidates = []
        propose = get(functions.get("outer_propose"), "outer_propose")
        if propose is not None and propose.get("kind") != "outer_propose":
            output["diagnostics"].append({"code": "PAIRED_EVENT_KIND_MISMATCH", "role": "outer_propose",
                "event_index": _id(propose.get("event_index")), "expected_kind": "outer_propose", "actual_kind": propose.get("kind")})
            propose = None
        propose_path = _path(sample, propose, parents) if propose is not None else None
        if propose is not None and propose_path is None:
            output["diagnostics"].append({"code": "PROPOSE_PARENT_CHAIN_NOT_CONFIRMED", "event_index": _id(propose.get("event_index"))})
        if propose_path is not None:
            used.update([_id(propose["event_index"]), *propose_path])
            candidates.append({"event": propose, "id": _id(propose["event_index"]), "kind": "propose", "ancestors": propose_path,
                               "start": _ns(propose["start_ns"]), "end": _ns(propose["end_ns"])})
        for event in case.get("host_events") or []:
            if event.get("kind") not in SAMPLE_KINDS or not _contains(sample, event) or _lane(sample) != _lane(event):
                continue
            path = _path(sample, event, parents)
            if path is None:
                output["diagnostics"].append({"code": "SAMPLE_CHILD_PARENT_CHAIN_NOT_CONFIRMED", "event_index": _id(event.get("event_index"))})
                continue
            if propose is not None and _id(propose.get("event_index")) in path:
                continue  # The business line keeps propose as one submission phase.
            used.update([_id(event["event_index"]), *path])
            candidates.append({"event": event, "id": _id(event["event_index"]), "kind": SAMPLE_KINDS[event["kind"]],
                               "ancestors": path, "start": _ns(event["start_ns"]), "end": _ns(event["end_ns"])})
        nested["sample_tokens_n"] = _exclusive_inside(sample, candidates, origin, "sample_tokens_n", "sample_remainder")
        output["inner_scopes"].extend({"role": "sample_tokens_n", "kind": c["kind"], "event_index": c["id"],
            "name": c["event"].get("name"), "ancestor_event_indexes": c["ancestors"], **_timing(c["event"], origin)} for c in candidates)
    next_execute = owners.get("next_execute_n1")
    if next_execute is not None:
        candidates, selected_events = [], {}
        for role, kind, number in (("runner_inputs", "runner_inputs", functions.get("runner_inputs")),
                                   ("runner_attn", "runner_attn", functions.get("runner_attn")),
                                   ("state_attn", "state_attn", functions.get("state_attn")),
                                   ("next_graph", "main_graph", context.get("next_main_graph"))):
            if number is None and role == "runner_attn":
                recovered = [e for e in case.get("host_events") or [] if e.get("kind") == kind
                             and _lane(e) == _lane(next_execute) and _contains(next_execute, e)
                             and next_execute.get("python_id") is not None
                             and str(e.get("python_parent_id")) == str(next_execute.get("python_id"))]
                if len(recovered) == 1:
                    number = recovered[0].get("event_index")
            event = get(number, role)
            if event is not None and event.get("kind") != kind:
                output["diagnostics"].append({"code": "PAIRED_EVENT_KIND_MISMATCH", "role": role,
                    "event_index": _id(event.get("event_index")), "expected_kind": kind, "actual_kind": event.get("kind")})
                event = None
            path = _path(next_execute, event, parents) if event is not None else None
            if event is None or path is None:
                if event is not None:
                    output["diagnostics"].append({"code": "NEXT_CHILD_PARENT_CHAIN_NOT_CONFIRMED", "role": role, "event_index": number})
                continue
            used.update([_id(event["event_index"]), *path])
            selected_events[role] = event
            candidates.append({"event": event, "id": _id(event["event_index"]), "kind": role, "ancestors": path,
                               "start": _ns(event["start_ns"]), "end": _ns(event["end_ns"])})
        ordered_roles = [selected_events[role] for role in ("runner_inputs", "runner_attn", "state_attn", "next_graph") if role in selected_events]
        if any(_ns(a["end_ns"]) > _ns(b["start_ns"]) for a, b in zip(ordered_roles, ordered_roles[1:])):
            output["diagnostics"].append({"code": "NEXT_PREPARATION_ORDER_OR_OVERLAP_CONFLICT"})
        if "runner_inputs" in selected_events:
            inputs = selected_events["runner_inputs"]
            a, b = _ns(next_execute["start_ns"]), _ns(inputs["start_ns"])
            if b > a:
                synthetic = {"event_index": None, "name": LABELS["prefix"], "start_ns": str(a), "end_ns": str(b)}
                candidates.append({"event": synthetic, "id": "synthetic_prefix", "kind": "prefix", "ancestors": [],
                    "start": a, "end": b, "synthetic": True,
                    "source_ids": [_id(next_execute["event_index"]), _id(inputs["event_index"])],
                    "reason": "SYNTHETIC_EXECUTE_ENTRY_TO_RUNNER_INPUTS_ENTRY_NOT_A_FUNCTION"})
        nested["next_execute_n1"] = _exclusive_inside(next_execute, candidates, origin, "next_execute_n1", "execute_remainder")
        output["inner_scopes"].extend({"role": "next_execute_n1", "kind": c["kind"], "event_index": c["id"],
            "name": c["event"].get("name"), "synthetic": c.get("synthetic", False),
            "ancestor_event_indexes": c["ancestors"], **_timing(c["event"], origin)} for c in candidates)
    start = min(_ns(e["start_ns"]) for e in owners.values())
    end = max(_ns(e["end_ns"]) for e in owners.values())
    cuts = {start, end}
    for event in owners.values():
        cuts.update((_ns(event["start_ns"]), _ns(event["end_ns"])))
    for pieces in nested.values():
        cuts.update(_ns(p[side]) for p in pieces for side in ("start_ns", "end_ns"))
    pieces = []
    ordered = sorted(cuts)
    for a, b in zip(ordered, ordered[1:]):
        if b <= a:
            continue
        active = [(role, event) for role, event in owners.items() if _ns(event["start_ns"]) <= a and _ns(event["end_ns"]) >= b]
        if not active:
            pieces.append(_piece(a, b, origin, "worker", "UNKNOWN_GAP", synthetic=True, reason="NO_RECORDED_PAIRED_WORKER_SCOPE"))
        elif len(active) > 1 or not same_lane:
            pieces.append(_piece(a, b, origin, "worker", "UNKNOWN_OVERLAP", synthetic=True,
                source_ids=[_id(e["event_index"]) for _, e in active],
                candidate_event_indexes=sorted([_id(e["event_index"]) for _, e in active], key=str),
                reason="OVERLAPPING_OUTER_SCOPES_ON_ONE_WORKER_THREAD" if same_lane else "WORKER_THREAD_MISMATCH",
                **({} if same_lane else {"name": "归属不明（线程证据不足）", "label": "归属不明（线程证据不足）"})))
        else:
            role, event = active[0]
            found = next((p for p in nested.get(role, []) if _ns(p["start_ns"]) <= a and _ns(p["end_ns"]) >= b), None)
            if found:
                value = deepcopy(found)
                value.update(_timing({"start_ns": a, "end_ns": b}, origin))
                pieces.append(value)
            else:
                pieces.append(_piece(a, b, origin, role, "current_execute", event, source_ids=[_id(event["event_index"])]))
    pieces = _merge(pieces)
    total = sum(piece["duration_ns"] for piece in pieces)
    closed = total == end-start and (not pieces and end == start or bool(pieces)
        and _ns(pieces[0]["start_ns"]) == start and _ns(pieces[-1]["end_ns"]) == end
        and all(a["end_ns"] == b["start_ns"] for a, b in zip(pieces, pieces[1:])))
    if not closed:
        raise ValueError("Worker business line does not close to the original outer span")
    output.update(status="READY" if not output["diagnostics"] and len(owners) == 3 else "PARTIAL", pieces=pieces,
                  gaps=[deepcopy(p) for p in pieces if p["kind"] == "UNKNOWN_GAP"], closed=closed,
                  total_duration_ns=total, **_timing({"start_ns": start, "end_ns": end}, origin))
    output["source_events"] = [deepcopy(event) for event in case.get("host_events") or [] if _id(event.get("event_index")) in used]
    confirmed_propose = next((scope["event_index"] for scope in output["inner_scopes"]
                             if scope.get("role") == "sample_tokens_n" and scope.get("kind") == "propose"), None)
    saved_propose = next((e for e in output["source_events"]
                          if confirmed_propose is not None and _id(e.get("event_index")) == confirmed_propose), None)
    regions = row.get("regions") or {}
    comparisons = (("current_execute_n", owners.get("current_execute_n"), "main_hc"),
                   ("sample_tokens_n", owners.get("sample_tokens_n"), "draft_hc"),
                   ("propose_n", saved_propose, "main_hc"),
                   ("propose_n", saved_propose, "draft_hc"))
    for name, event, device_role in comparisons:
        region = regions.get(device_role)
        if same_lane and event is not None and region is not None:
            delta = _ns(region["end_ns"])-_ns(event["end_ns"])
            output["trace_observations"].append({"kind": "TRACE_OBSERVATION", "scope": name, "device_role": device_role,
                "host_event_index": _id(event.get("event_index")), "host_return_ns": str(_ns(event["end_ns"])),
                "hc_envelope_end_ns": str(_ns(region["end_ns"])), "host_return_before_hc_end_ns": delta,
                "host_return_before_hc_end_ms": delta/1e6,
                "relation": "HOST_RETURN_PRECEDES_HC_ENVELOPE_END" if delta > 0 else "HOST_RETURN_DOES_NOT_PRECEDE_HC_ENVELOPE_END",
                "limitation": "HC is an envelope, not the complete FULL graph or all device work; this is temporal evidence, not proof of a specific wait dependency"})
    return output
