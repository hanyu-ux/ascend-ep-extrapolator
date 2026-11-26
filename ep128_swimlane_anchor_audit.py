"""Audit all U-stage boundaries using saved, exact source identities only.

This module does not scan traces, repair clocks, or infer callers from timing.
Adjacent stages can share one device-task endpoint and therefore one submission.
"""

from collections import Counter, defaultdict
from copy import deepcopy
import re


SCHEMA = "u_swimlane_anchor_audit_v1"
GRADE_LABELS = {
    "VERIFIED_PYTHON_SUBMISSION": "Python父链已证实",
    "VERIFIED_CPU_API": "CPU op/API已证实；Python归属未证实",
    "TIME_CONTAINMENT_ONLY": "仅时间包含候选",
    "GRAPH_SHARED_NO_ORIGIN": "图共享来源；未记录原始提交",
    "NO_TRACE_DEVICE": "未匹配trace设备任务",
    "AMBIGUOUS": "来源或边界不唯一",
    "MISSING": "缺少来源证据",
    "CLOCK_UNALIGNED": "时钟未对齐；Host留空",
}
_SOURCE_KEYS = ("cpu_event_index", "cpu_leaf_event_index", "enqueue_event_index",
                "dequeue_event_index", "node_event_index")
_UNIQUE_BINDING = {"EXACT_SOURCE_ROW", "HC_MARKER_TIMESTAMP_UNIQUE"}
_PYTHON_RELATIONS = {"EXPLICIT_PYTHON_PARENT_ID", "EXACT_SOURCE_EVENT_IDENTITY"}


def _ns(value):
    if type(value) is int:
        return value
    if isinstance(value, str) and re.fullmatch(r"[+-]?\d+", value):
        return int(value)
    raise ValueError("Missing or non-integer exact boundary timestamp")


def _timed(event):
    try:
        return _ns(event.get("end_ns")) >= _ns(event.get("start_ns"))
    except (TypeError, ValueError):
        return False


def _is_python(event):
    return "python" in str(event.get("cat", event.get("category", event.get("event_type", "")))).lower()


def _is_cpu_api(event):
    category = str(event.get("cat", event.get("category", ""))).lower()
    return not _is_python(event) and any(token in category for token in
        ("cpu_op", "runtime", "annotation", "api", "enqueue"))


def _event_table(case, links):
    result, conflicts = {}, set()
    for event in [*(case.get("host_events") or []), *(links.get("source_events") or [])]:
        key = event.get("event_index")
        if key is None:
            continue
        previous = result.get(key)
        if previous is not None and any(
                str(previous[k]) != str(event[k]) for k in ("name", "start_ns", "end_ns", "pid", "tid")
                if k in previous and k in event):
            conflicts.add(key)
        result[key] = {**(previous or {}), **event}
    return result, conflicts


def _source(link, events, conflicts, origin):
    """A LINKED core chain proves its CPU op/API, not enclosing Python calls."""
    result = {"source_status": link.get("status", "MISSING"), "device_name": link.get("name"),
              "source_name": None, "cpu_source_name": None, "source_event_index": None,
              "source_start_ns": None, "source_end_ns": None, "source_start_ms": None,
              "source_end_ms": None, "source_time_ms": None, "source_required_event_indexes": [],
              "missing_source_event_indexes": [], "verified_python_owners": [],
              "python_owner_candidates": [], "python_owner_grade": "MISSING", "reason": link.get("reason")}
    status = result["source_status"]
    if status != "LINKED":
        result["grade"] = status if status in GRADE_LABELS else "MISSING"
        return result
    wanted = set(link.get("source_event_indexes") or [])
    wanted.update(a["id"] for a in link.get("cpu_ancestors") or [] if a.get("id") is not None)
    wanted.update(link[k] for k in _SOURCE_KEYS if link.get(k) is not None)
    missing = set(link.get("missing_source_event_indexes") or []) | (wanted - events.keys()) | (wanted & conflicts)
    result["source_required_event_indexes"] = sorted(wanted, key=str)
    result["missing_source_event_indexes"] = sorted(missing, key=str)
    source_key = next((k for k in ("cpu_event_index", "enqueue_event_index", "node_event_index")
                       if link.get(k) is not None), None)
    source_id = link.get(source_key) if source_key is not None else None
    source = events.get(source_id, {})
    complete = bool(wanted) and not missing and _timed(source)
    prefix = {"cpu_event_index": "cpu", "enqueue_event_index": "enqueue", "node_event_index": "node"}.get(source_key)
    if prefix and _timed(source):
        complete = complete and all(link.get(prefix + "_" + key) is None
                                    or str(link[prefix + "_" + key]) == str(source[key])
                                    for key in ("start_ns", "end_ns"))
    # LINKED was produced by the core flow/connection resolver. Retain its IDs
    # and every referenced source event; a nearby Python interval adds no proof.
    for association in link.get("host_associations") or []:
        if association.get("source_event_index") != source_id:
            continue
        owner_id = association.get("host_event_index")
        owner = events.get(owner_id, {})
        if not _is_python(owner):
            continue
        path = association.get("explicit_parent_event_indexes") or []
        verified = (complete and association.get("runtime_submission_confirmed") is True
                    and association.get("line_style") == "solid"
                    and association.get("relation") in _PYTHON_RELATIONS
                    and _timed(owner) and owner_id not in conflicts
                    and bool(path) and all(i in events and i not in conflicts for i in path)
                    and path[0] == source_id and path[-1] == owner_id)
        candidate = {"event_index": owner_id, "name": owner.get("name", association.get("host_name")),
                     "relation": association.get("relation"), "verified": verified,
                     "explicit_parent_event_indexes": list(path)}
        result["python_owner_candidates"].append(candidate)
        if verified:
            result["verified_python_owners"].append(candidate)
    owners = result["verified_python_owners"]
    result["python_owner_grade"] = ("VERIFIED_PYTHON_SUBMISSION" if owners else
        "TIME_CONTAINMENT_ONLY" if result["python_owner_candidates"] else "MISSING")
    # A Python event is not renamed into a CPU API when its ancestry is absent.
    cpu_api = complete and _is_cpu_api(source)
    result["grade"] = ("VERIFIED_PYTHON_SUBMISSION" if owners else "VERIFIED_CPU_API" if cpu_api else
                       "TIME_CONTAINMENT_ONLY" if result["python_owner_candidates"] else "MISSING")
    if complete:
        start, end = _ns(source["start_ns"]), _ns(source["end_ns"])
        result.update(source_event_index=source_id, source_name=source.get("name"),
                      cpu_source_name=source.get("name"), source_start_ns=str(start), source_end_ns=str(end),
                      source_start_ms=(start-origin)/1e6, source_end_ms=(end-origin)/1e6,
                      source_time_ms=(start-origin)/1e6)
    else:
        result["reason"] = "Recorded LINKED source is incomplete, conflicting, or lacks exact source timing"
    return result


def audit_stage_anchors(case, origin_ns):
    """Return 30 boundary audits; never require 30 independent submissions.

    ``physical_sources`` deduplicates device tasks by exact CSV row. The
    ``physical_anchor_id`` additionally identifies start/end of the same task.
    Clock failures intentionally retain no CPU source coordinates or arrows.
    """
    origin = _ns(origin_ns)
    rounds = case.get("rounds") or []
    selected = case.get("selected_round_id")
    chosen = next((r for r in rounds if r.get("round_id") == selected), rounds[0] if rounds else {})
    round_id = chosen.get("round_id", selected)
    stages = defaultdict(list)
    for window in chosen.get("u_windows") or []:
        for stage in window.get("stages") or []:
            number = stage.get("stage_number")
            if type(number) is int and 1 <= number <= 15:
                stages[number].append(stage)
    links = case.get("anchor_links") or {}
    bindings = links.get("anchor_bindings") or []
    events, conflicts = _event_table(case, links)
    aligned = ((case.get("trace") or {}).get("status") == "ALIGNED_X_EVENTS"
               and case.get("status") != "ERROR"
               and links.get("clock_alignment_status", "ALIGNED_X_EVENTS") == "ALIGNED_X_EVENTS")
    trace_sha = (case.get("trace") or {}).get("sha256")
    if trace_sha and links.get("trace_sha256") and trace_sha != links["trace_sha256"]:
        aligned = False
    by_row = defaultdict(list)
    for link in links.get("links") or []:
        by_row[link.get("row_index")].append(link)
    source_cache, physical, entries = {}, {}, []
    for number in range(1, 16):
        group = stages[number]
        stage = group[0] if len(group) == 1 else {}
        for side in ("start", "end"):
            anchor = stage.get(side + "_anchor") or {}
            stamp = stage.get(side + "_ns", anchor.get("timestamp_ns"))
            value = {"anchor_id": "U" + str(number) + "." + side, "round_id": round_id,
                     "stage_number": number, "stage_key": stage.get("stage_key"),
                     "window_key": stage.get("window_key"), "side": side,
                     "stage_status": stage.get("status", "MISSING"), "timestamp_ns": str(stamp) if stamp is not None else None,
                     "time_ms": (_ns(stamp)-origin)/1e6 if stamp is not None else None,
                     "declared_anchor_name": anchor.get("name"), "binding_status": "MISSING",
                     "binding_indexes": [], "device_rows": [], "physical_anchor_id": None,
                     "source_id": None, "source_status": "MISSING", "grade": "MISSING",
                     "python_owner_grade": "MISSING", "arrow_allowed": False,
                     "source_name": None, "cpu_source_name": None, "source_event_index": None,
                     "source_time_ms": None, "source_start_ns": None, "source_end_ns": None,
                     "source_start_ms": None, "source_end_ms": None, "reason": None}
            matches = [(i, b) for i, b in enumerate(bindings)
                       if b.get("round_id") == round_id and b.get("anchor_kind") == "u_stage"
                       and b.get("stage_number") == number and b.get("side") == side
                       and b.get("stage_key") == stage.get("stage_key")
                       and b.get("window_key") == stage.get("window_key")
                       and stamp is not None and b.get("timestamp_ns") is not None
                       and _ns(b["timestamp_ns"]) == _ns(stamp)]
            value["binding_indexes"] = [i for i, _ in matches]
            if len(matches) == 1:
                index, binding = matches[0]
                rows = binding.get("device_row_indexes") or []
                value.update(binding_status=binding.get("status"), device_rows=list(rows),
                             device_edge=binding.get("edge"), binding_index=index,
                             declared_anchor_name=binding.get("declared_anchor_name", anchor.get("name")))
                if len(rows) == 1:
                    row = rows[0]
                    source_id = "D" + str(row)
                    edge = binding.get("edge", side)
                    value.update(source_id=source_id, physical_anchor_id=source_id + "." + str(edge) + "@" + str(stamp))
                    value["endpoint_note"] = ("已有设备task的结束端点；不是新增任务" if edge == "end" else
                                               "已有设备task的开始端点；相邻阶段可共用")
                    candidates = by_row[row]
                    if aligned and len(candidates) == 1:
                        link = candidates[0]
                        if row not in source_cache:
                            source_cache[row] = _source(link, events, conflicts, origin)
                        value.update(deepcopy(source_cache[row]))
                        endpoint = link.get("device_" + str(edge) + "_ns")
                        exact_endpoint = endpoint is not None and _ns(endpoint) == _ns(stamp)
                        declared_row = anchor.get("row_index")
                        unique = (binding.get("status") in _UNIQUE_BINDING and binding.get("stage_status") == "MATCHED"
                                  and stage.get("status") == "MATCHED" and len(group) == 1 and exact_endpoint
                                  and stage.get("round_id", round_id) == round_id
                                  and (declared_row is None or declared_row == row))
                        if not unique:
                            value.update(grade="AMBIGUOUS" if binding.get("status") == "AMBIGUOUS" else "MISSING",
                                         reason="Boundary identity/status or exact device-task endpoint is not verified")
                        value["arrow_allowed"] = unique and value["grade"] in ("VERIFIED_PYTHON_SUBMISSION", "VERIFIED_CPU_API")
                    elif aligned and len(candidates) > 1:
                        value.update(grade="AMBIGUOUS", reason="Multiple source records for the same CSV device row")
                    task = physical.setdefault(source_id, {"source_id": source_id, "row_index": row, "anchor_ids": [],
                                                          "physical_anchor_ids": [], "arrow_allowed": False})
                    task["anchor_ids"].append(value["anchor_id"])
                    if value["physical_anchor_id"] not in task["physical_anchor_ids"]:
                        task["physical_anchor_ids"].append(value["physical_anchor_id"])
                    source_description = source_cache.get(row, value) if aligned else value
                    for key in ("source_status", "grade", "python_owner_grade", "source_name", "cpu_source_name", "device_name",
                                "source_event_index", "source_start_ns", "source_end_ns", "source_start_ms", "source_end_ms",
                                "source_time_ms", "reason", "verified_python_owners", "python_owner_candidates"):
                        if key in source_description:
                            task[key] = deepcopy(source_description[key])
                    task["arrow_allowed"] = task["arrow_allowed"] or value["arrow_allowed"]
            elif len(matches) > 1:
                value.update(grade="AMBIGUOUS", binding_status="AMBIGUOUS", reason="Multiple saved bindings for one stage boundary")
            if not aligned:
                value.update(grade="CLOCK_UNALIGNED", source_status="CLOCK_UNALIGNED", python_owner_grade="MISSING",
                             arrow_allowed=False, reason="CPU and device clocks have not passed exact alignment")
            value["grade_label"] = GRADE_LABELS[value["grade"]]
            entries.append(value)
    for task in physical.values():
        if not aligned:
            task.update(grade="CLOCK_UNALIGNED", source_status="CLOCK_UNALIGNED", arrow_allowed=False)
        task["grade_label"] = GRADE_LABELS[task.get("grade", "MISSING")]
    return {"schema": SCHEMA, "round_id": round_id, "host_clock_aligned": aligned,
            "audit_entries": entries, "physical_sources": list(physical.values()),
            "counts": dict(Counter(e["grade"] for e in entries)),
            "physical_source_counts": dict(Counter(e.get("grade", "MISSING") for e in physical.values())),
            "boundary_count": len(entries), "verified_boundary_count": sum(e["arrow_allowed"] for e in entries),
            "physical_source_count": len(physical),
            "limitations": ["30项是15阶段的起止边界审计，不是30次设备提交。",
                            "相邻阶段可共用同一CSV task端点；CPU/API已证实不代表Python业务归属已证实。",
                            "设备端点用精确CSV行号和原flow来源关联；不按最近时间补线。"]}
