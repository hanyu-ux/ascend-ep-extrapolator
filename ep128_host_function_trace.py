"""Attribute saved Host windows to the functions visible in an aligned trace."""

from bisect import bisect_right
from collections import Counter, defaultdict
import re


def _anchor_matches(event, request):
    return (event["name"] == request["anchor_name"]
            and event["start"] == int(request["anchor_start_ns"])
            and event["end"] == int(request["anchor_end_ns"]))


def _function_rows(core, values, limit=5):
    ordered = sorted(values.items(), key=lambda item: (-item[1]["ns"], item[0]))
    chosen = ordered if limit is None else ordered[:limit]
    rows = []
    for (name, lane), value in chosen:
        sync, copy, graph = core.residual_flags(name)
        rows.append({"name": name if limit is None else name[:180], "lane": lane, "ns": value["ns"],
                     "event_indexes": sorted(value["event_indexes"])[:3],
                     "sync": sync, "copy": copy, "graph": graph})
    return rows, len(ordered) - len(chosen)


def _partition(core, events, left, right, *, function_limit=5):
    points, selected = defaultdict(list), {}
    inclusive = defaultdict(lambda: {"spans": [], "event_indexes": set()})
    for event in events:
        start, end = max(left, event["start"]), min(right, event["end"])
        if start >= end:
            continue
        identity = event["event_index"]
        selected[identity] = event
        points[start].append((1, identity))
        points[end].append((-1, identity))
        key = event["name"], event["lane"]
        inclusive[key]["spans"].append((start, end))
        inclusive[key]["event_indexes"].add(identity)
    points[left]
    points[right]
    active, exclusive, gaps = {}, defaultdict(lambda: {"ns": 0, "event_indexes": set()}), []
    ambiguous = attributed = 0
    cursor = left
    for point in sorted(points):
        duration = point - cursor
        if duration and not active:
            if gaps and gaps[-1][1] == cursor:
                gaps[-1] = (gaps[-1][0], point)
            else:
                gaps.append((cursor, point))
        elif duration:
            # Original bounds distinguish genuine nesting from a clipped overlap.
            deepest = [event for event in active.values() if not any(
                other["start"] >= event["start"] and other["end"] <= event["end"]
                and (other["start"], other["end"]) != (event["start"], event["end"])
                for other in active.values())]
            signatures = {(event["start"], event["end"], event["name"], event["lane"])
                          for event in deepest}
            if len(signatures) == 1:
                event = deepest[0]
                value = exclusive[event["name"], event["lane"]]
                value["ns"] += duration
                value["event_indexes"].update(item["event_index"] for item in deepest)
                attributed += duration
            else:
                ambiguous += duration
        for direction, identity in sorted(points[point]):
            if direction < 0:
                active.pop(identity, None)
            else:
                active[identity] = selected[identity]
        cursor = point
    unrecorded = sum(end - start for start, end in gaps)
    if attributed + ambiguous + unrecorded != right - left:
        raise ValueError("Host function partition does not close to its window")
    inclusive_values = {key: {"ns": sum(end - start for start, end in core.a_union(value["spans"])),
                              "event_indexes": value["event_indexes"]}
                        for key, value in inclusive.items()}
    exclusive_rows, exclusive_omitted = _function_rows(core, exclusive, function_limit)
    inclusive_rows, inclusive_omitted = _function_rows(core, inclusive_values, function_limit)
    return {"window_ns": right - left, "attributed_ns": attributed, "ambiguous_ns": ambiguous,
            "unrecorded_ns": unrecorded, "overlapping_event_count": len(selected),
            "exclusive_functions": exclusive_rows, "exclusive_functions_omitted": exclusive_omitted,
            "inclusive_functions": inclusive_rows, "inclusive_functions_omitted": inclusive_omitted,
            "longest_unrecorded": [{"start_relative_ns": start - left, "end_relative_ns": end - left,
                                    "ns": end - start}
                                   for start, end in sorted(gaps, key=lambda span: -(span[1] - span[0]))[:4]]}


def analyze_trace(core, path, trace_info, windows, *, function_limit=5):
    """Inspect existing X events without inferring unrecorded Python calls.

    Source event indexes follow the core's one-based enumeration. The same
    physical Host thread can have Python and synthetic CANN process lanes.
    Self time is elapsed time in a recorded scope, including unrecorded work
    and scheduling delays inside it. It is not a measurement of CPU busy time.
    function_limit=None retains all full function names for caller aggregation;
    the default still returns five bounded examples per window.
    """
    if function_limit is not None and (type(function_limit) is not int or function_limit < 1):
        raise ValueError("function_limit must be a positive integer or None")
    if trace_info.get("status") != "ALIGNED_X_EVENTS":
        raise ValueError("Function attribution requires an aligned source trace")
    offset = int(trace_info["clock_offset_ns"])
    requests, intervals, identities = {}, [], set()
    for window in windows:
        identity = window["id"]
        if identity in identities:
            raise ValueError("Duplicate Host window ID")
        identities.add(identity)
        left, right = int(window["start_ns"]), int(window["end_ns"])
        if right <= left:
            raise ValueError("Host windows must have positive duration")
        intervals.append((left, right))
        for request in (window, window.get("start_anchor")):
            if request:
                requests[int(request["anchor_event_index"])] = True
    union = core.a_union(intervals)
    starts, ends = [span[0] for span in union], [span[1] for span in union]
    access = core._io_path(path)
    before = access.stat()
    processes, anchors, candidates, scan = {}, {}, [], {}
    counts = Counter(x_events=0, python_function_events=0, be_events_unsupported=0, invalid_x_events=0)
    for number, raw in enumerate(core.residual_iter_trace_events(path, stats=scan), 1):
        phase = raw.get("ph")
        pid, tid = core.residual_id(raw.get("pid")), core.residual_id(raw.get("tid"))
        if phase == "M" and raw.get("name") == "process_name" and pid is not None:
            processes[pid] = str(core.residual_args(raw).get("name", ""))
        if phase in ("B", "E"):
            counts["be_events_unsupported"] += 1
        if phase != "X":
            continue
        counts["x_events"] += 1
        category = str(raw.get("cat", "")).lower()
        if "python_function" in re.split(r"[,; ]+", category):
            counts["python_function_events"] += 1
        try:
            start, duration = core.a_ns(raw["ts"]) + offset, core.a_ns(raw["dur"])
            if duration <= 0:
                continue
        except (KeyError, ValueError, ArithmeticError):
            counts["invalid_x_events"] += 1
            continue
        args = core.residual_args(raw)
        event = {"event_index": number, "name": str(raw.get("name", "")),
                 "pid": pid, "tid": tid, "start": start, "end": start + duration,
                 "category": category, "thread_id": core.residual_id(args.get("threadid")),
                 "device_hint": core.residual_device_hint(raw, args)}
        if number in requests:
            anchors[number] = event
        index = bisect_right(ends, start)
        if index < len(starts) and starts[index] < event["end"]:
            candidates.append(event)
    after = access.stat()
    stamp = lambda value: (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)
    if stamp(before) != stamp(after):
        raise ValueError("Trace changed while Host function evidence was read")
    expected = trace_info.get("sha256")
    if expected and scan.get("sha256") != expected:
        raise ValueError("Trace SHA256 differs from the saved aligned report")
    output = []
    for window in windows:
        row = {"id": window["id"], "kind": window["kind"]}
        anchor = anchors.get(int(window["anchor_event_index"]))
        if anchor is None:
            output.append({**row, "status": "ANCHOR_NOT_FOUND"})
            continue
        if not _anchor_matches(anchor, window):
            output.append({**row, "status": "ANCHOR_MISMATCH"})
            continue
        if anchor["pid"] is None or anchor["tid"] is None:
            output.append({**row, "status": "ANCHOR_THREAD_UNAVAILABLE"})
            continue
        if window.get("start_anchor"):
            request = window["start_anchor"]
            first = anchors.get(int(request["anchor_event_index"]))
            if (first is None or not _anchor_matches(first, request)
                    or (first["pid"], first["tid"]) != (anchor["pid"], anchor["tid"])):
                output.append({**row, "status": "START_ANCHOR_MISMATCH"})
                continue
        selected = []
        for event in candidates:
            process = processes.get(event["pid"], "")
            if (event["tid"] != anchor["tid"] or event["device_hint"]
                    or core.residual_device_process(process)):
                continue
            if event["pid"] == anchor["pid"] and core.residual_host_category(event["category"], process):
                lane = "cpu_thread"
            elif re.search(r"cann|runtime", process, re.I) and (
                    event["thread_id"] is None or event["thread_id"] == anchor["tid"]):
                lane = "cann_thread"
            else:
                continue
            selected.append({**event, "lane": lane})
        left, right = int(window["start_ns"]), int(window["end_ns"])
        python_events = [event for event in selected
                         if "python_function" in re.split(r"[,; ]+", event["category"].lower())
                         and max(left, event["start"]) < min(right, event["end"])]
        python_intervals = core.a_union(
            [(max(left, event["start"]), min(right, event["end"])) for event in python_events])
        row.update(status="MATCHED", anchor={key: anchor[key] for key in ("event_index", "name", "pid", "tid")},
                   python_function_events=len(python_events),
                   python_function_covered_ns=sum(end - start for start, end in python_intervals),
                   **_partition(core, selected, left, right, function_limit=function_limit))
        output.append(row)
    return {"trace_sha256": scan.get("sha256"), "sha256_matches_report": bool(expected),
            "counts": dict(counts), "windows": output,
            "scope": "Innermost visible Host scopes on the source thread; self time includes uninstrumented work and descheduling; B/E events are not reconstructed"}
