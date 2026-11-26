"""Offline, exact-coordinate Host/HC/U swimlane geometry for Excel and SVG.

The input is saved evidence, never a trace reader.  Shapes are shared by the
native XLSX drawing and the HTML/SVG preview, so their time geometry is equal.
"""

from collections import Counter, defaultdict
from copy import deepcopy
import html
import math
import re


SCHEMA = "u_swimlane_plot_v1"
LANE_ORDER = ("outer_propose", "preparation_dp_aggregate", "runner_inputs", "runner_attn", "state_attn")
LANE_LABELS = {
    "outer_propose": "CPU propose（本轮候选）",
    "preparation_dp_aggregate": "CPU 下一轮前处理 / DP范围",
    "runner_inputs": "CPU runner.prepare_inputs",
    "runner_attn": "CPU runner.prepare_attn · 块表/slot",
    "state_attn": "CPU model_state.prepare_attn · metadata",
}
COLORS = {"outer_propose": "8068AA", "preparation_dp_aggregate": "C39746", "runner_inputs": "D88855",
          "runner_attn": "4A9D89", "state_attn": "547FB6", "main_hc": "173E67", "draft_hc": "188B9B",
          "dp_sync": "D4A349", "cpu_wait": "B85450", "graph_dispatch": "668C9A", "unknown": "C4CBD3"}
COLORS.update({"request_" + name: "8BA4B8" for name in (
    "update_pp_decode_requests", "finish_requests", "free_states", "add_requests", "update_requests", "gather_batch_req_state")})
COLORS.update(blocktable_apply="8BA4B8", all_reduce="D4A349")
STAGE_COLORS = ("D6E0EC", "C5D5E7", "B7CBE1", "D8E4EB", "CEDDE6", "BACFD9", "E6BC75", "E0D4B5",
                "C5DBD9", "BAD3D3", "ACCBCC", "D0DFDD", "E59B75", "B1A0CF", "CDD6E2")
WRAPPERS = {"context_wrapper", "profiler_wrapper", "metadata_native_wrapper", "metadata_argument_check",
            "collective_wrapper", "dflash_propose", "build_attn_metadata", "dp_dispatch_patch", "dp_dispatch"}
GENERIC = {"host_copy", "native_workspace", "metadata_native_call", "stream_dependency", "stream_event_record"}
NESTED_FOCUS = {"draft_build_metadata", "cpu_wait", "cpu_sync"}
SCOPE = ("各case取自身原代表轮；0为该轮主HC起点，同一毫秒刻度。Host条为完整wall范围，非CPU busy；"
         "propose来自本轮sample_tokens，其余四线属于下一execute_model准备。"
         "HC仅包络，不等于FULL图完整边界或连续设备忙碌；空隙没有自动归因为idle。"
         "U为前8段与后7段设备锚点区间，不等于Python函数划分。")


def _integer(value):
    if isinstance(value, bool) or value is None:
        raise ValueError("Missing or invalid exact nanosecond coordinate")
    return int(value)


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _bounds(value, origin):
    a, b = _integer(value["start_ns"]), _integer(value["end_ns"])
    if b < a:
        raise ValueError("Reversed interval in saved swimlane evidence")
    return {"start_ns": str(a), "end_ns": str(b), "duration_ns": b-a,
            "start_ms": (a-origin)/1e6, "end_ms": (b-origin)/1e6, "duration_ms": (b-a)/1e6}


def _name(value):
    text = str(value or "unknown")
    prefix, separator, function = text.rpartition(": ")
    if separator and function:
        path = prefix.replace("\\", "/")
        stem = path.rsplit("/", 1)[-1].split("(", 1)[0].removesuffix(".py")
        return stem + "." + function
    if text.startswith("<built-in method wait of "):
        return "Event.wait" if "Event object" in text else "pybind.wait"
    if text.startswith("<built-in method synchronize of "):
        return "Event.synchronize" if "Event object" in text else "native.synchronize"
    return text


def _sort_number(value):
    try:
        return 0, float(value)
    except (TypeError, ValueError):
        return 1, str(value)


def _resolve(ref, events, origin, lane):
    number = ref.get("event_index")
    source = events.get(number, {})
    result = {**source, **ref}
    if "start_ns" not in result or "end_ns" not in result:
        return None
    result.update(_bounds(result, origin), lane=lane, role=lane, label=_name(result.get("name")), record_type="child")
    return result


def _visible_children(role, children):
    """One business child layer; transparent wrappers are folded, not timed away.

    All calls still remain in detail_rows. Selected nested wait/metadata scopes
    are also visible because they explain the known inclusive parent envelope.
    """
    lookup = {c["event_index"]: c for c in children}
    candidates = [c for c in children if c.get("kind") not in WRAPPERS | GENERIC]
    if role == "preparation_dp_aggregate":
        selected = [c for c in candidates if str(c.get("kind", "")).startswith("request_")
                    or c.get("kind") in ("blocktable_apply", "dp_sync", "graph_dispatch", "all_reduce", "cpu_wait")]
    else:
        selected = []
        semantic_ids = {c["event_index"] for c in candidates}
        for child in candidates:
            ancestors = child.get("ancestor_event_indexes") or []
            semantic_ancestor = any(number in semantic_ids for number in ancestors)
            if not semantic_ancestor or child.get("kind") in NESTED_FOCUS:
                selected.append(child)
        # Unverified containment remains a separate visible candidate; never
        # assign a synthetic depth or claim it is a direct Python child.
        selected = [c for c in selected if c.get("parent_chain_verified") is True
                    or c.get("kind") in NESTED_FOCUS]
    groups = defaultdict(list)
    for child in selected:
        groups[(child.get("kind", "unknown"), child["label"])].append(child)
    return [{"lane": role, "kind": kind, "label": label, "events": sorted(values, key=lambda c: (c["start_ms"], c["end_ms"]))}
            for (kind, label), values in sorted(groups.items(), key=lambda pair: (min(c["start_ms"] for c in pair[1]), pair[0]))]


def _validation(row):
    issues = []
    a, b = _integer(row["start_ns"]), _integer(row["end_ns"])
    regions = row.get("regions") or {}
    if set(("main_hc", "draft_hc")).issubset(regions):
        region_ns = sum(_integer(r["end_ns"])-_integer(r["start_ns"]) for r in regions.values())
        if region_ns + _integer(row.get("u_ns", 0)) != b-a:
            issues.append("HC包络＋U未闭合到所选轮")
    stages = []
    for window in row.get("u_windows") or []:
        own = window.get("stages") or []
        stages.extend(own)
        if not own or _integer(own[0]["start_ns"]) != _integer(window["start_ns"]) or _integer(own[-1]["end_ns"]) != _integer(window["end_ns"]):
            issues.append("U窗口端点不闭合")
        if any(_integer(x["end_ns"]) != _integer(y["start_ns"]) for x, y in zip(own, own[1:])):
            issues.append("U阶段存在重叠或间隙")
    if len(stages) != 15 or any(s.get("status") != "MATCHED" for s in stages):
        issues.append("U15阶段证据不完整")
    return list(dict.fromkeys(issues))


def _has_source_proof(edge, links, events):
    """Old confirmed flags cannot survive missing source/ancestor evidence."""
    for link in links.get("links") or []:
        if link.get("row_index") != edge.get("row_index") or link.get("status") != "LINKED":
            continue
        if link.get("missing_source_event_indexes"):
            continue
        for association in link.get("host_associations") or []:
            if (association.get("host_event_index") != edge.get("host_event_index")
                    or association.get("source_event_index") != edge.get("source_event_index")
                    or association.get("runtime_submission_confirmed") is not True
                    or association.get("relation") not in ("EXPLICIT_PYTHON_PARENT_ID", "EXACT_SOURCE_EVENT_IDENTITY")):
                continue
            wanted = set(link.get("source_event_indexes") or [])
            wanted.update(a["id"] for a in link.get("cpu_ancestors") or [] if a.get("id") is not None)
            wanted.update(association.get("explicit_parent_event_indexes") or [])
            wanted.update(link.get(key) for key in ("cpu_event_index", "cpu_leaf_event_index", "node_event_index",
                                                   "enqueue_event_index", "dequeue_event_index") if link.get(key) is not None)
            wanted.update((edge.get("host_event_index"), edge.get("source_event_index")))
            if all(number in events for number in wanted):
                return True
    return False


def _panel(case, rank):
    panel = {"case": str(case.get("case", "unknown")), "rank": case.get("rank", rank),
             "bs": case.get("bs"), "S": case.get("S"), "status": case.get("representative_status", case.get("status", "UNAVAILABLE")),
             "source_status": case.get("status"), "trace_status": (case.get("trace") or {}).get("status"),
             "error": case.get("share_error") or case.get("error"), "host_lanes": {}, "child_plot_rows": [],
             "detail_rows": [], "regions": {}, "stages": [], "boundaries": [], "edges": [], "notes": [],
             "round_id": case.get("selected_round_id"), "anchor_link_status": (case.get("anchor_links") or {}).get("status"),
             "source_trace_sha256": (case.get("trace") or {}).get("sha256"), "validation_issues": []}
    if panel["rank"] != rank:
        panel.update(status="ERROR", error="固定rank不匹配；未借用其他rank")
        return panel
    rounds = case.get("rounds") or []
    if not rounds:
        panel.update(status="UNAVAILABLE", error=panel["error"] or "没有保存代表轮")
        return panel
    chosen = next((r for r in rounds if r.get("round_id") == panel["round_id"]), rounds[0])
    panel["round_id"] = chosen.get("round_id")
    origin = _integer(chosen.get("origin_ns", chosen["start_ns"]))
    panel.update(origin_ns=str(origin), **_bounds(chosen, origin), u_ns=_integer(chosen.get("u_ns", 0)))
    panel["u_ms"] = panel["u_ns"]/1e6
    panel["u_before_draft_ms"] = sum((_integer(w["end_ns"])-_integer(w["start_ns"]))/1e6
                                     for w in chosen.get("u_windows") or [] if w.get("window_key") == "main_to_draft")
    panel["u_after_draft_ms"] = sum((_integer(w["end_ns"])-_integer(w["start_ns"]))/1e6
                                    for w in chosen.get("u_windows") or [] if w.get("window_key") == "draft_to_nextmain")
    panel["regions"] = {key: {**r, **_bounds(r, origin)} for key, r in (chosen.get("regions") or {}).items()}
    for window in chosen.get("u_windows") or []:
        for stage in window.get("stages") or []:
            value = {**stage, **_bounds(stage, origin), "window_key": window["window_key"]}
            panel["stages"].append(value)
            panel["detail_rows"].append({**value, "record_type": "u_stage", "lane": "device_u", "role": "device_u",
                                         "name": value.get("label", value.get("stage_key")),
                                         "function": value.get("label", value.get("stage_key"))})
    bounds = sorted({_integer(s[side]) for s in panel["stages"] for side in ("start_ns", "end_ns")})
    panel["boundaries"] = [{"timestamp_ns": str(stamp), "time_ms": (stamp-origin)/1e6} for stamp in bounds]
    panel["validation_issues"] = _validation(chosen)
    if panel["validation_issues"]:
        panel["status"] = "PARTIAL"
    host_ok = panel["trace_status"] == "ALIGNED_X_EVENTS" and case.get("status") != "ERROR"
    events = {}
    for event in [*case.get("host_events", []), *(case.get("anchor_links") or {}).get("source_events", [])]:
        events[event["event_index"]] = {**events.get(event["event_index"], {}), **event}
    if host_ok:
        lookup = (chosen.get("host_pairing") or {}).get("functions") or {}
        for role in LANE_ORDER:
            lane = (chosen.get("host_lanes") or {}).get(role)
            if lane is None:
                continue
            event_id = lookup.get(role, lane.get("event_index", lane.get("start_event_index")))
            event = events.get(event_id, {})
            normalized = {**lane, **_bounds(lane, origin), "lane": role, "role": role, "kind": role,
                          "event_index": event_id, "name": event.get("name", role), "label": LANE_LABELS[role], "record_type": "parent"}
            panel["host_lanes"][role] = normalized
            panel["detail_rows"].append(normalized)
        child_layer = (chosen.get("host_pairing") or {}).get("child_layer") or {}
        child_groups = child_layer.get("by_role") or chosen.get("host_children") or {}
        for role in LANE_ORDER:
            children = []
            for ref in child_groups.get(role) or []:
                child = _resolve(ref, events, origin, role)
                if child is not None:
                    children.append(child)
                    panel["detail_rows"].append(child)
            panel["child_plot_rows"].extend(_visible_children(role, children))
        panel["dp_breakdown"] = deepcopy(chosen.get("coordination_breakdown") or {})
        links = case.get("anchor_links") or {}
        bindings = links.get("anchor_bindings") or []
        for edge in links.get("drawable_edges") or []:
            if edge.get("round_id") != panel["round_id"]:
                continue
            number = edge.get("anchor_binding_index")
            if not isinstance(number, int) or not 0 <= number < len(bindings):
                continue
            binding = bindings[number]
            if binding.get("timestamp_ns") is None:
                continue
            source = events.get(edge.get("source_event_index"), {})
            host_event = events.get(edge.get("host_event_index"), {})
            unique = binding.get("status") in ("EXACT_SOURCE_ROW", "HC_MARKER_TIMESTAMP_UNIQUE")
            known = binding.get("anchor_kind") != "u_stage" or binding.get("stage_status") == "MATCHED"
            confirmed = (edge.get("runtime_submission_confirmed") is True and edge.get("line_style") == "solid"
                         and edge.get("source_status") == "LINKED" and unique and known
                         and binding.get("round_id") == panel["round_id"]
                         and edge.get("row_index") in (binding.get("device_row_indexes") or [])
                         and source.get("start_ns") is not None and source.get("end_ns") is not None
                         and host_event.get("start_ns") is not None and host_event.get("end_ns") is not None
                         and _has_source_proof(edge, links, events))
            panel["edges"].append({**edge, "line_style": "solid" if confirmed else "dashed",
                "runtime_submission_confirmed": confirmed, "anchor_time_ms": (_integer(binding["timestamp_ns"])-origin)/1e6,
                "source_time_ms": (_integer(source["start_ns"])-origin)/1e6 if source.get("start_ns") is not None else None})
        panel["notes"].append("子图折叠透明wrapper并按函数归组；全部子调用与父链保留在明细表，嵌套范围不能相加。")
    else:
        panel["notes"].append("时钟校验未通过：CPU与关联箭头留空，仅保留设备HC/U。")
    panel["edge_counts"] = dict(Counter(edge["line_style"] for edge in panel["edges"]))
    panel.update(cycle_ms=panel["duration_ms"], main_hc_ms=panel["regions"].get("main_hc", {}).get("duration_ms"),
                 draft_hc_ms=panel["regions"].get("draft_hc", {}).get("duration_ms"),
                 reasons=[*panel["validation_issues"], *([str(panel["error"])] if panel.get("error") else [])])
    panel["notes"].append("来源箭头仅展示每条Host线的一例；虚线为推断/不完整来源，不能当成已证实提交；完整来源候选保留于输入JSON。")
    return panel


def summarize_swimlanes(payload, rank="dp4", source_info=None):
    """Normalize saved full v2/v3 or representative evidence; no capture IO."""
    if not isinstance(payload, dict):
        raise ValueError("Swimlane evidence must be a JSON object")
    if payload.get("rank") != rank:
        raise ValueError("Requested fixed rank differs from the saved evidence")
    if payload.get("schema") in ("u_swimlane_v2", "u_swimlane_v3"):
        import profile_ep128 as core
        import ep128_swimlane_trace as host_analysis
        from ep128_swimlane_share import build_ppt_evidence
        from tools.u_swimlane_probe import attach_host
        source = build_ppt_evidence(payload, core, host_analysis, attach_host, source_info or {}, rank=rank)
    elif payload.get("schema") == "u_swimlane_ppt_evidence_v1":
        source = payload
    else:
        raise ValueError("Use an existing full swimlane v2/v3 JSON or representative evidence JSON")
    panels = []
    for case in source.get("cases") or []:
        try:
            panels.append(_panel(case, rank))
        except (ValueError, TypeError, KeyError, ArithmeticError) as error:
            panels.append({"case": str(case.get("case", "unknown")), "rank": case.get("rank", rank), "bs": case.get("bs"),
                "S": case.get("S"), "status": "ERROR", "error": type(error).__name__ + ": " + str(error),
                "round_id": case.get("selected_round_id"), "host_lanes": {}, "regions": {}, "stages": [],
                "boundaries": [], "child_plot_rows": [], "detail_rows": [], "edges": [], "notes": []})
    panels.sort(key=lambda p: (_sort_number(p.get("bs")), _sort_number(p.get("S")), p["case"]))
    starts, ends = [0.], [.001]
    for panel in panels:
        for interval in [*panel.get("regions", {}).values(), *panel.get("host_lanes", {}).values(),
                         *[e for group in panel.get("child_plot_rows", []) for e in group["events"]]]:
            starts.append(interval["start_ms"])
            ends.append(interval["end_ms"])
        if _number(panel.get("end_ms")):
            ends.append(panel["end_ms"])
    return {"schema": SCHEMA, "rank": rank, "panels": panels, "x_min_ms": min(starts), "x_max_ms": max(ends),
            "scope": SCOPE, "source_schema": payload.get("schema"), "source_json_sha256": source.get("source_json_sha256"),
            "function_colors": dict(COLORS), "selected_rounds_per_case": 1}


def _text(elements, x, y, width, height, text, size=10, color="1C344D", bold=False, align="left", **extra):
    if width > 0 and height > 0:
        elements.append({"type": "text", "x": x, "y": y, "width": width, "height": height,
                         "text": str(text), "font_size": size, "color": color, "bold": bold, "align": align,
                         "valign": "center", **extra})


def _rect(elements, x, y, width, height, fill, **extra):
    if width > 0 and height > 0:
        elements.append({"type": "rect", "x": x, "y": y, "width": width, "height": height,
                         "fill": fill, "stroke": fill, "stroke_width": .6, **extra})


def _line(elements, x1, y1, x2, y2, color="D4DDE5", width=.7, dash="solid", **extra):
    elements.append({"type": "line", "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                     "stroke": color, "stroke_width": width, "dash": dash, **extra})


def _label(value, maximum=45):
    return value if len(value) <= maximum else value[:maximum-1] + "…"


def panel_drawing(panel, width=1600, overview=False, x_min_ms=None, x_max_ms=None):
    """Return native rect/line/text geometry. All times stay on one linear axis."""
    width = float(width)
    if not math.isfinite(width) or width < 500:
        raise ValueError("Swimlane drawing width must be at least 500")
    child_rows = [] if overview else panel.get("child_plot_rows", [])
    left, right = (165., width-18.) if overview else (340., width-32.)
    top_u, u_height, hc_y, hc_height = (104., 19., 130., 19.) if overview else (110., 26., 146., 27.)
    parent_h, child_h = (27., 0.) if overview else (37., 29.)
    y = hc_y + hc_height + (12 if overview else 20)
    rows = []
    for role in LANE_ORDER:
        rows.append((role, None, y, parent_h))
        y += parent_h
        for group in child_rows:
            if group["lane"] == role:
                rows.append((role, group, y, child_h))
                y += child_h
    axis_y, height = y+12., y+(105. if overview else 150.)
    elements = []
    _rect(elements, 0, 0, width, height, "FFFFFF", role="background")
    title = f"{panel.get('case')} · BS={panel.get('bs')} / S={panel.get('S')} · {panel.get('rank')} · {panel.get('round_id') or '无代表轮'}"
    _text(elements, 12, 5, width-24, 27, title, 10 if overview else 15, bold=True)
    cycle = panel.get("duration_ms")
    metrics = (f"整轮 {cycle:.3f} ms  |  U {panel.get('u_ms', 0):.3f} ms  |  {panel.get('status')}" if _number(cycle)
               else "未绘制时序：" + str(panel.get("error") or panel.get("status")))
    _text(elements, 12, 34, width-24, 24, metrics, 8 if overview else 10, "5A6878")
    intervals = [*panel.get("regions", {}).values(), *panel.get("host_lanes", {}).values(),
                 *[e for group in child_rows for e in group["events"]]]
    lo = float(x_min_ms) if x_min_ms is not None else min([0., *[r["start_ms"] for r in intervals]])
    hi = float(x_max_ms) if x_max_ms is not None else max([.001, panel.get("end_ms", 0), *[r["end_ms"] for r in intervals]])
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        raise ValueError("Swimlane time axis must be finite and increasing")
    if any(r["start_ms"] < lo-1e-9 or r["end_ms"] > hi+1e-9 for r in intervals):
        raise ValueError("Requested common time axis would clip a saved Host interval")
    scale = (right-left)/(hi-lo)
    x = lambda time: left + (time-lo)*scale
    tick_count = 5 if overview else 8
    for tick in range(tick_count+1):
        value = lo + (hi-lo)*tick/tick_count
        point = x(value)
        _line(elements, point, top_u-3, point, axis_y, "E5EAF0", .5)
        _text(elements, max(0, min(width-46, point-23)), axis_y+3, 46, 22, f"{value:.1f}", 7 if overview else 9, align="center")
    _text(elements, left, axis_y+26, right-left, 23, "相对本轮主HC起点 / ms（统一刻度）", 7 if overview else 9, align="center")
    if lo <= 0 <= hi:
        _line(elements, x(0), top_u-5, x(0), axis_y, "3D516B", 1.0, "dash")
    if _number(panel.get("end_ms")):
        _line(elements, x(panel["end_ms"]), top_u-5, x(panel["end_ms"]), axis_y, "3D516B", 1.0, "dash", role="cycle_end")
    # Stage dividers are behind every CPU lane. They mean device anchor time,
    # not the point at which a Python function necessarily submitted its work.
    for boundary in panel.get("boundaries", []):
        _line(elements, x(boundary["time_ms"]), top_u-2, x(boundary["time_ms"]), y,
              "A6B3C4", .65, "dash", opacity=.5, role="u_boundary", timestamp_ns=boundary["timestamp_ns"])
    _text(elements, 9, top_u, left-16, u_height, "NPU · U阶段", 8 if overview else 11, bold=True)
    _text(elements, 9, hc_y, left-16, hc_height, "NPU · HC包络", 8 if overview else 11, bold=True)
    stage_points = []
    for index, stage in enumerate(panel.get("stages", [])):
        number = stage.get("stage_number", index+1)
        color = STAGE_COLORS[(int(number)-1) % len(STAGE_COLORS)]
        start, end = x(stage["start_ms"]), x(stage["end_ms"])
        _rect(elements, start, top_u, max(.05, end-start), u_height, color, role="u_stage", stage_number=number)
        if end-start > (15 if overview else 25):
            _text(elements, start, top_u, end-start, u_height, str(number), 7 if overview else 9, align="center")
        else:
            stage_points.append((index, (start+end)/2, number))
    # Tiny stage numbers are led to three label tracks; original boundaries do
    # not move. Their complete names and durations stay in the source table.
    previous = defaultdict(lambda: left-20)
    for index, point, number in stage_points:
        track = index % 3
        label_x = min(right-17, max(point-8, previous[track]+18))
        previous[track] = label_x
        label_y = top_u-17-track*12
        _line(elements, point, top_u, label_x+8, label_y+11, "9BAABB", .5)
        _text(elements, label_x, label_y-1, 17, 13, str(number), 6 if overview else 7, align="center")
    for key in ("main_hc", "draft_hc"):
        region = panel.get("regions", {}).get(key)
        if region:
            start, end = x(region["start_ms"]), x(region["end_ms"])
            _rect(elements, start, top_u, max(.05, end-start), u_height, "EDF1F5", role="non_u", lane=key)
            if end-start > 50:
                _text(elements, start, top_u, end-start, u_height,
                      ("主HC · 非U" if key == "main_hc" else "草HC · 非U"),
                      6 if overview else 9, "6B7A8E", align="center")
            _rect(elements, start, hc_y, max(.05, end-start), hc_height, COLORS[key], role="hc", lane=key)
            text = ("主HC " if key == "main_hc" else "草HC ") + f"{region['duration_ms']:.3f}"
            if end-start > 50:
                _text(elements, start, hc_y, end-start, hc_height, text, 7 if overview else 10, "FFFFFF", True, "center")
    event_y = {}
    host_rows = {}
    for role, group, row_y, row_h in rows:
        _line(elements, left, row_y+row_h, right, row_y+row_h, "EEF1F5", .4)
        color = COLORS[role]
        if group is None:
            lane = panel.get("host_lanes", {}).get(role)
            label = LANE_LABELS[role]
            if overview:
                label = {"outer_propose": "propose", "preparation_dp_aggregate": "前处理 / DP", "runner_inputs": "runner.prepare_inputs",
                         "runner_attn": "runner.prepare_attn", "state_attn": "state.prepare_attn"}[role]
            _text(elements, 9, row_y, left-17, row_h, label, 7 if overview else 10, bold=True)
            if lane:
                start, end = x(lane["start_ms"]), x(lane["end_ms"])
                _rect(elements, start, row_y+5, max(.05, end-start), row_h-10, color,
                      role="host", lane=role, event_index=lane.get("event_index"))
                if end-start > (60 if overview else 95):
                    name = {"outer_propose": "propose", "preparation_dp_aggregate": "前处理 / DP",
                            "runner_inputs": "prepare_inputs", "runner_attn": "prepare_attn · slot", "state_attn": "prepare_attn · metadata"}[role]
                    text = f"{lane['duration_ms']:.3f} ms" if overview else name+f"  {lane['duration_ms']:.3f} ms"
                    ink = "1C344D" if role in ("preparation_dp_aggregate", "runner_inputs", "runner_attn") else "FFFFFF"
                    _text(elements, start, row_y+5, end-start, row_h-10, text, 7 if overview else 9, ink, True, "center")
                elif not overview:
                    _text(elements, left-105, row_y+17, 94, 17, f"{lane['duration_ms']:.3f} ms", 8, "536578", align="right")
                event_y[lane.get("event_index")] = row_y+row_h/2
                host_rows[role] = row_y+row_h/2
            else:
                _text(elements, left+7, row_y, right-left-14, row_h, "缺失 / 未对齐，留空", 7 if overview else 9, "9B4E37")
        else:
            label = _label(group["label"], 41)
            _text(elements, 21, row_y, left-32, row_h, "↳ " + label, 8, "52677A")
            for child in group["events"]:
                start, end = x(child["start_ms"]), x(child["end_ms"])
                proof = child.get("parent_chain_verified") is True
                fill = COLORS.get(child.get("kind"), color)
                _rect(elements, start, row_y+6, max(.05, end-start), row_h-12, fill,
                      opacity=.75 if proof else .4, dash="solid" if proof else "dash", role="child", lane=role,
                      event_index=child.get("event_index"), parent_chain_verified=proof)
                event_y[child.get("event_index")] = row_y+row_h/2
    if not overview:
        # A bounded semantic view, not a bounded evidence export. Every saved
        # candidate remains in the input JSON; no time-only edge is solid.
        examples = {}
        for edge in panel.get("edges", []):
            event_id = edge.get("host_event_index")
            if event_id not in event_y or not _number(edge.get("source_time_ms")):
                continue
            role = next((key for key, lane in panel.get("host_lanes", {}).items() if lane.get("event_index") == event_id), None)
            if role is None:
                continue
            preference = (not edge["runtime_submission_confirmed"], edge["anchor_time_ms"])
            if role not in examples or preference < examples[role][0]:
                examples[role] = preference, edge
        for role, (_, edge) in examples.items():
            source_x, target_x = x(edge["source_time_ms"]), x(edge["anchor_time_ms"])
            if not left <= source_x <= right or not left <= target_x <= right:
                continue
            _line(elements, source_x, host_rows[role], target_x, top_u+u_height,
                  COLORS[role], .9, "solid" if edge["runtime_submission_confirmed"] else "dash", arrow="end",
                  opacity=.6, role="source_edge", runtime_submission_confirmed=edge["runtime_submission_confirmed"])
    note_y = axis_y+53
    if overview:
        note = panel.get("error") or ("CPU未对齐，仅设备侧" if not panel.get("host_lanes") else "HC是包络；父子不相加；放大图与明细在下方")
        _text(elements, 10, note_y, width-20, 34, _label(str(note), 85), 7, "7C5D45" if panel.get("error") else "607186")
    else:
        dp = panel.get("dp_breakdown") or {}
        def ms(value):
            return "未知" if value is None else f"{int(value)/1e6:.3f} ms"
        dp_note = "前处理/DP分账：协调 " + ms(dp.get("dp_coordination_wall_ns")) + "；其中已证wait " + ms(dp.get("verified_dp_wait_union_ns")) + "；协调外 " + ms(dp.get("outside_coordination_wall_ns"))
        _text(elements, 10, note_y, width-20, 25, dp_note, 9, "5E526B")
        _text(elements, 10, note_y+27, width-20, 26, "透明wrapper已折叠；全部子调用与完整函数名见下方明细，完整来源链保留于输入JSON。竖线=设备锚点，虚线箭头=推断；父子范围不能相加。", 9, "607186")
        if panel.get("error"):
            _text(elements, 10, note_y+54, width-20, 23, _label(str(panel["error"]), 150), 9, "A34B35")
    return {"width": width, "height": height, "elements": elements, "title": title,
            "x_min_ms": lo, "x_max_ms": hi, "overview": overview, "axis_y": axis_y,
            "host_bottom_y": y, "scope": SCOPE}


def render_svg(drawing):
    """Render exactly the same vector elements used in the XLSX worksheet."""
    esc = lambda value: html.escape(str(value), quote=True)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {drawing["width"]:.3f} {drawing["height"]:.3f}" role="img">',
             '<title>' + esc(drawing.get("title", "Host/HC/U swimlane")) + '</title>',
             '<defs><marker id="swimlane-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="context-stroke"/></marker></defs>']
    for element in drawing["elements"]:
        kind = element["type"]
        role = ' data-role="' + esc(element.get("role", "")) + '"'
        opacity = ' opacity="' + str(element.get("opacity", 1)) + '"'
        dash = ' stroke-dasharray="5 4"' if element.get("dash") not in (None, False, "solid") else ''
        if kind == "rect":
            parts.append(f'<rect x="{element["x"]:.3f}" y="{element["y"]:.3f}" width="{element["width"]:.3f}" height="{element["height"]:.3f}" fill="#{esc(element.get("fill", "FFFFFF").lstrip("#"))}" stroke="#{esc(element.get("stroke", element.get("fill", "FFFFFF")).lstrip("#"))}" stroke-width="{element.get("stroke_width", .6)}"{role}{opacity}{dash}/>')
        elif kind == "line":
            arrow = ' marker-end="url(#swimlane-arrow)"' if element.get("arrow") in ("end", "both") else ''
            parts.append(f'<line x1="{element["x1"]:.3f}" y1="{element["y1"]:.3f}" x2="{element["x2"]:.3f}" y2="{element["y2"]:.3f}" stroke="#{esc(element.get("stroke", "A0A0A0").lstrip("#"))}" stroke-width="{element.get("stroke_width", .7)}"{role}{opacity}{dash}{arrow}/>')
        elif kind == "text":
            align = element.get("align", "left")
            anchor = "middle" if align == "center" else "end" if align == "right" else "start"
            tx = element["x"] + (element["width"]/2 if align == "center" else element["width"] if align == "right" else 0)
            ty = element["y"] + element["height"]/2
            parts.append(f'<text x="{tx:.3f}" y="{ty:.3f}" text-anchor="{anchor}" dominant-baseline="central" font-family="Microsoft YaHei,Arial,sans-serif" font-size="{element.get("font_size", 10)}pt" font-weight="{"700" if element.get("bold") else "400"}" fill="#{esc(element.get("color", "1C344D").lstrip("#"))}"{role}{opacity}>' + esc(element["text"]) + '</text>')
    parts.append('</svg>')
    return ''.join(parts)
