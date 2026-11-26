"""Conservative submission evidence for saved swimlane tasks and anchors.

The core owns all flow/connection/queue resolution.  The full exporter requests
every device task and projects legacy anchor evidence from the same resolution.
Python ownership still requires an explicit parent-ID path to the source.
An interval containing an API is useful context, never a verified caller.
"""

from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from copy import deepcopy
import csv
import json
from pathlib import Path
import time
from types import SimpleNamespace

from ep128_interval_index import MaxEndIntervalIndex


_IDENTITY_FIELDS = (
    "row_index", "row_indices", "name", "op_type", "device_id", "model_id",
    "stream_id", "task_id", "context_id", "batch_id", "subtask_id", "task_type",
    "op_state", "start_ns", "end_ns",
)
_INDEX_FIELDS = (
    "cpu_event_index", "cpu_leaf_event_index", "node_event_index",
    "enqueue_event_index", "dequeue_event_index",
)
_NS_FIELDS = (
    "device_start_ns", "device_end_ns", "cpu_start_ns", "cpu_end_ns",
    "node_start_ns", "node_end_ns", "enqueue_start_ns", "enqueue_end_ns",
    "dequeue_start_ns", "dequeue_end_ns", "queue_wait_ns", "submit_to_device_ns",
)
_LIMITATIONS = (
    "Device source status is resolved by the unchanged core, not by nearby Host timing.",
    "LINKED confirms the core source chain; a Python function needs exact source-event identity or an explicit parent-ID path for a solid arrow.",
    "Containment, graph-entry ordering and source-code correspondence are not device submission proof.",
    "Queue wait is dequeue.start minus enqueue.end; submit-to-device is device.start minus Node.start, not a pure queue cost.",
    "Anchor requests include HC endpoints outside U. Original U windows/stages are neither enlarged nor modified.",
    "No link is invented for CPU-only waiting; partial, ambiguous and missing graph origins remain visible.",
)


def _stamp(path):
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns


def _number(value):
    return None if value is None or value == "" else int(value)


def _task_identity(task):
    value = {key: getattr(task, key, None) for key in _IDENTITY_FIELDS}
    value["row_indices"] = list(value["row_indices"] or [task.row_index])
    return value


def _device_anchors(core, capture, rounds):
    """Return every boundary binding, including missing/ambiguous candidates."""
    tasks = list(capture["tasks"])
    by_row = defaultdict(list)
    by_edge = {"start": defaultdict(list), "end": defaultdict(list)}
    for task in tasks:
        by_row[int(task.row_index)].append(task)
        by_edge["start"][int(task.start_ns)].append(task)
        by_edge["end"][int(task.end_ns)].append(task)
    bindings, selected = [], {}

    def add(round_id, kind, side, stamp, *, declared=None, pattern=None, edge="start", **extra):
        stamp = int(stamp)
        if declared is not None:
            candidates = by_row.get(int(declared), [])
            matches = [t for t in candidates if int(getattr(t, edge + "_ns")) == stamp
                       and (not extra.get("declared_anchor_name") or t.name == extra["declared_anchor_name"])]
            status = "EXACT_SOURCE_ROW" if len(matches) == 1 else (
                "AMBIGUOUS" if len(matches) > 1 else "ANCHOR_IDENTITY_MISMATCH" if candidates else "MISSING_DEVICE_TASK")
        elif pattern is not None:
            matches = [t for t in by_edge[edge].get(stamp, []) if (
                pattern.search(t.name) or pattern.search(getattr(t, "op_type", "")))]
            status = "HC_MARKER_TIMESTAMP_UNIQUE" if len(matches) == 1 else "AMBIGUOUS" if matches else "MISSING_DEVICE_TASK"
        else:
            matches, status = [], "UNRESOLVED_BOUNDARY"
        indexes = sorted({int(t.row_index) for t in matches})
        for task in matches:
            # Duplicate CSV identities cannot be collapsed into a unique task.
            if len(by_row[int(task.row_index)]) == 1:
                selected[int(task.row_index)] = task
        bindings.append({"round_id": round_id, "anchor_kind": kind, "side": side,
                         "timestamp_ns": stamp, "edge": edge, "status": status,
                         "device_row_indexes": indexes, **extra})

    for row in rounds:
        round_id = row["round_id"]
        for role in ("main", "draft"):
            region = (row.get("regions") or {}).get(role + "_hc")
            if region:
                add(round_id, role + "_hc", "start", region["start_ns"], pattern=core.A_PRE)
                add(round_id, role + "_hc", "end", region["end_ns"], pattern=core.A_POST, edge="end")
        if row.get("end_ns") is not None:
            add(round_id, "next_main_hc", "start", row["end_ns"], pattern=core.A_PRE)
        for window in row.get("u_windows", []):
            for stage in window.get("stages", []):
                for side in ("start", "end"):
                    anchor = stage.get(side + "_anchor") or {}
                    declared = anchor.get("row_index")
                    stamp = stage[side + "_ns"]
                    boundary = int(stamp) == int(window[side + "_ns"])
                    pattern = core.A_POST if side == "start" else core.A_PRE
                    edge = "end" if side == "start" else "start"
                    add(round_id, "u_stage", side, stamp, declared=declared,
                        pattern=pattern if declared is None and boundary else None,
                        edge="start" if declared is not None else edge,
                        window_key=window["window_key"], stage_key=stage["stage_key"],
                        stage_number=stage.get("stage_number", stage.get("stage_order", 0) + (
                            8 if window["window_key"] == "draft_to_nextmain" else 0)),
                        stage_status=stage.get("status"), declared_anchor_name=anchor.get("name"))
    return bindings, [selected[key] for key in sorted(selected)]


def _parse_source(row):
    result = {key: value if value != "" else None for key, value in row.items()}
    for key in ("row_index", *_INDEX_FIELDS, *_NS_FIELDS):
        result[key] = _number(result.get(key))
    result["flow_ids"] = json.loads(result.pop("flow_ids_json", None) or "[]")
    ancestors = json.loads(result.pop("cpu_ancestors_json", None) or "[]")
    result["cpu_ancestors"] = [{**{k: v for k, v in ancestor.items() if k not in ("start", "end")},
                                "start_ns": int(ancestor["start"]), "end_ns": int(ancestor["end"])}
                               for ancestor in ancestors]
    # The synthetic source-only owner is the device task, not a real U window.
    result["source_before_anchor_ns"] = _number(result.pop("source_before_window_ns", None))
    return result


def _read_evidence_events(core, trace_path, trace_meta, wanted, progress, *, source_links=()):
    """Enrich only original source/ancestor references; never invent flow edges."""
    access = core._io_path(trace_path)
    before, scan, found = _stamp(access), {}, {}
    flow_keys = {key for link in source_links for key in link.get("flow_ids", [])}
    flow_events = []
    started = last = time.monotonic()
    number = 0
    for number, raw in enumerate(core.residual_iter_trace_events(trace_path, stats=scan), 1):
        if source_links and raw.get("ph") in ("s", "f"):
            identity = raw.get("id", raw.get("id2"))
            category = str(raw.get("cat", ""))
            key = json.dumps([category, raw.get("name"), raw.get("scope"), identity],
                             ensure_ascii=False, default=str, separators=(",", ":"))
            if identity is not None and (key in flow_keys or category == "async_task_queue"):
                try:
                    stamp = core.a_ns(raw["ts"]) + int(trace_meta["clock_offset_ns"])
                except (KeyError, TypeError, ValueError, ArithmeticError):
                    continue
                flow_events.append({"event_index": number, "name": str(raw.get("name", "")),
                    "cat": category, "ph": raw["ph"], "flow_key": key,
                    "flow_id": json.loads(json.dumps(identity, default=str)),
                    "pid": core.residual_id(raw.get("pid")), "tid": core.residual_id(raw.get("tid")),
                    "start_ns": stamp, "end_ns": stamp, "duration_ns": 0,
                    "scope": "Original trace flow endpoint; retention does not upgrade the core source status"})
        if number in wanted and raw.get("ph") == "X":
            args = core.residual_args(raw)
            try:
                start = core.a_ns(raw["ts"]) + int(trace_meta["clock_offset_ns"])
                duration = core.a_ns(raw["dur"])
            except (KeyError, TypeError, ValueError, ArithmeticError):
                continue
            if duration <= 0:
                continue
            found[number] = {"event_index": number, "name": str(raw.get("name", "")),
                             "cat": str(raw.get("cat", "")), "pid": core.residual_id(raw.get("pid")),
                             "tid": core.residual_id(raw.get("tid")), "start_ns": start,
                             "end_ns": start + duration, "duration_ns": duration,
                             "python_id": core.residual_id(args.get("pythonid")),
                             "python_parent_id": core.residual_id(args.get("pythonparentid")),
                             "connection_id": core.residual_id(args.get("connectionid")),
                             "correlation_id": core.residual_id(args.get("correlationid"))}
        if progress and number % 10000 == 0:
            now = time.monotonic()
            if now-last >= 5:
                progress("Device source references: scanned " + format(number, ",") +
                         " events; elapsed " + format(now-started, ".1f") + "s")
                last = now
    if _stamp(access) != before or scan.get("sha256") != trace_meta["sha256"]:
        raise ValueError("Trace changed or SHA256 differs during anchor-source enrichment")
    if source_links:
        queues = {link.get(key) for link in source_links for key in ("enqueue_event_index", "dequeue_event_index")}
        intervals = _HostIntervals([{**found[i], "cat": "python_function"} for i in queues if i in found])
        # A flow point can be recorded exactly at the API/queue scope end.
        # Retain such points as audit candidates; core causal checks remain
        # unchanged, so keeping endpoints does not prove a submission chain.
        queue_keys = {e["flow_key"] for e in flow_events if e["cat"] == "async_task_queue"
                      and intervals.containing(e, include_end=True)}
        selected_flow_keys = flow_keys | queue_keys
        # Retain every endpoint of a selected/shared ID, rather than one nearby
        # pair that could falsely turn reused graph flow IDs into unique proof.
        found.update({e["event_index"]: e for e in flow_events if e["flow_key"] in selected_flow_keys})
    if progress:
        progress("Device source references complete: " + format(number, ",") + " scanned / " +
                 format(len(found), ",") + " retained events; elapsed " + format(time.monotonic()-started, ".1f") + "s")
    return found


class _HostIntervals:
    def __init__(self, events):
        grouped = defaultdict(list)
        for event in events:
            if "python_function" in str(event.get("cat", "")).lower():
                grouped[(str(event.get("pid")), str(event.get("tid")))].append(event)
        self.lanes = {}
        for lane, group in grouped.items():
            group.sort(key=lambda e: (int(e["start_ns"]), int(e["end_ns"]), e["event_index"]))
            self.lanes[lane] = group, MaxEndIntervalIndex(
                [int(e["start_ns"]) for e in group], [int(e["end_ns"]) for e in group])

    def containing(self, event, *, include_end=False):
        lane = self.lanes.get((str(event.get("pid")), str(event.get("tid"))))
        if lane is None:
            return []
        group, intervals = lane
        start, end = int(event["start_ns"]), int(event["end_ns"])
        return [group[position] for position in intervals.containing(start, end, include_end=include_end)]


def _python_identities(raw_events):
    identities = defaultdict(list)
    for event in raw_events.values():
        if event.get("python_id") is not None and "python_function" in event.get("cat", "").lower():
            identities[(event["pid"], event["tid"], event["python_id"])].append(event)
    return identities


def _associations(link, raw_events, identities, index):
    source_id = next((link.get(key) for key in ("cpu_event_index", "enqueue_event_index", "node_event_index")
                      if link.get(key) is not None), None)
    source = raw_events.get(source_id)
    if source is None:
        return []
    explicit, current, used, path = {}, source, {source_id}, [source_id]
    ambiguous = False
    for _ in range(len(raw_events)):
        parent_id = current.get("python_parent_id")
        if parent_id is None:
            break
        parents = [e for e in identities.get((current["pid"], current["tid"], parent_id), [])
                   if e["event_index"] not in used and e["start_ns"] <= current["start_ns"]
                   and e["end_ns"] >= current["end_ns"]]
        if len(parents) != 1:
            ambiguous = len(parents) > 1
            break
        current = parents[0]
        used.add(current["event_index"])
        path.append(current["event_index"])
        explicit[current["event_index"]] = list(path)
    result = []
    for host in index.containing(source):
        number = host["event_index"]
        # Host references must also have been verified against this same trace.
        verified = raw_events.get(number)
        if verified is None or any(str(host.get(k)) != str(verified.get(k)) for k in (
                "name", "pid", "tid", "start_ns", "end_ns", "python_id", "python_parent_id")):
            continue
        relation = "EXACT_SOURCE_EVENT_IDENTITY" if number == source_id else (
            "EXPLICIT_PYTHON_PARENT_ID" if number in explicit else
            "AMBIGUOUS_PYTHON_PARENT_ID" if ambiguous else "TIME_CONTAINMENT_ONLY")
        solid = (relation in ("EXPLICIT_PYTHON_PARENT_ID", "EXACT_SOURCE_EVENT_IDENTITY")
                 and link["status"] == "LINKED" and not link.get("missing_source_event_indexes"))
        result.append({"host_event_index": number, "source_event_index": source_id,
                       "host_name": host["name"], "relation": relation,
                       "line_style": "solid" if solid else "dashed",
                       "runtime_submission_confirmed": solid,
                       "explicit_parent_event_indexes": [source_id] if number == source_id else explicit.get(number, []),
                       "scope": "Recorded core source with exact event identity or explicit Python parent path" if solid else
                                "Partial source chain or temporal containment; not a verified Host-to-device caller"})
    return result


def _missing_references(references, raw):
    # ``set(refs) - raw.keys()`` visits the entire large evidence dictionary on
    # every device task. Membership probes depend only on the requested refs.
    return sorted({number for number in references if number not in raw})


class _AssociationCache:
    """Case-local parent-path results; each exported link owns its own lists."""

    def __init__(self, raw, identities, intervals):
        self.raw, self.identities, self.intervals = raw, identities, intervals
        self.results = {}

    def associations(self, link):
        source = next((link.get(key) for key in ("cpu_event_index", "enqueue_event_index", "node_event_index")
                       if link.get(key) is not None), None)
        key = source, link["status"], bool(link.get("missing_source_event_indexes"))
        if key not in self.results:
            self.results[key] = _associations(link, self.raw, self.identities, self.intervals)
        return [{**row, "explicit_parent_event_indexes": list(row["explicit_parent_event_indexes"])}
                for row in self.results[key]]


class _FlowReferences:
    """Index raw endpoints once and reuse repeated graph/queue provenance."""

    def __init__(self, raw):
        self.raw, self.by_key, self.queues = raw, defaultdict(list), defaultdict(list)
        for event in raw.values():
            if event.get("ph") in ("s", "f"):
                self.by_key[event["flow_key"]].append(event["event_index"])
                if event.get("cat") == "async_task_queue":
                    self.queues[(event["pid"], event["tid"])].append(event)
        for events in self.queues.values():
            events.sort(key=lambda e: (e["start_ns"], e["event_index"]))
        self.starts = {lane: [e["start_ns"] for e in events] for lane, events in self.queues.items()}
        self.queue_keys, self.references = {}, {}

    def _queue_keys(self, event_index):
        if event_index not in self.queue_keys:
            event = self.raw.get(event_index)
            if event is None:
                keys = ()
            else:
                lane = event["pid"], event["tid"]
                events, starts = self.queues.get(lane, []), self.starts.get(lane, [])
                # Closed endpoints retain evidence candidates only. No source
                # or causality decision consumes this provenance lookup.
                first, stop = bisect_left(starts, event["start_ns"]), bisect_right(starts, event["end_ns"])
                keys = tuple(e["flow_key"] for e in events[first:stop])
            self.queue_keys[event_index] = keys
        return self.queue_keys[event_index]

    def event_indexes(self, link):
        key = tuple(link.get("flow_ids", [])), link.get("enqueue_event_index"), link.get("dequeue_event_index")
        if key not in self.references:
            selected = set(key[0])
            selected.update(self._queue_keys(key[1]))
            selected.update(self._queue_keys(key[2]))
            self.references[key] = tuple(sorted({i for flow in selected for i in self.by_key.get(flow, [])}))
        return list(self.references[key])


def _bindings_by_row(bindings):
    result = defaultdict(list)
    for index, binding in enumerate(bindings):
        for number in binding["device_row_indexes"]:
            result[number].append(index)
    return result


def _drawable_edges(links, bindings):
    result = []
    for link in links:
        for association in link["host_associations"]:
            for number in link["anchor_binding_indexes"]:
                binding = bindings[number]
                unique = binding["status"] in ("EXACT_SOURCE_ROW", "HC_MARKER_TIMESTAMP_UNIQUE")
                stage_known = binding["anchor_kind"] != "u_stage" or binding.get("stage_status") == "MATCHED"
                solid = association["runtime_submission_confirmed"] and unique and stage_known
                result.append({"host_event_index": association["host_event_index"],
                               "source_event_index": association["source_event_index"],
                               "row_index": link["row_index"], "anchor_binding_index": number,
                               "round_id": binding["round_id"], "stage_number": binding.get("stage_number"),
                               "stage_key": binding.get("stage_key"), "window_key": binding.get("window_key"),
                               "anchor_kind": binding["anchor_kind"], "side": binding["side"],
                               "source_status": link["status"], "anchor_status": binding["status"],
                               "stage_status": binding.get("stage_status"), "host_relation": association["relation"],
                               "line_style": "solid" if solid else "dashed",
                               "runtime_submission_confirmed": solid,
                               "scope": "Unique recorded source and Host identity/parent path and device boundary" if solid else
                                        "Partial or inferred relation; do not draw as confirmed submission to this boundary"})
    return result


def export_anchor_links(core, capture, rounds, host_events, trace_path, trace_meta,
                        offset, directory, progress=None):
    """Resolve every selected U/HC boundary anchor using the unchanged core.

    ``rounds`` are the full records from ``u_swimlane_probe.round_coordinates``.
    Only source-only stage copies are sent to the core. The original U ranges,
    selected Host intervals, and their pairing inference are never changed.
    The caller owns ``directory`` and may remove its intermediate CSV files.
    """
    if trace_meta.get("status") != "ALIGNED_X_EVENTS" or not trace_meta.get("sha256"):
        raise ValueError("Aligned trace metadata and SHA256 are required for device anchor links")
    if offset is None or int(offset) != int(trace_meta.get("clock_offset_ns", offset)):
        raise ValueError("Anchor source clock offset differs from the verified trace")
    metadata = deepcopy(trace_meta)
    metadata["clock_offset_ns"] = str(int(offset))
    bindings, tasks = _device_anchors(core, capture, rounds)
    result = {"schema": "u_swimlane_anchor_links_v1", "anchor_bindings": bindings,
              "device_tasks": [_task_identity(t) for t in tasks], "links": [], "source_events": [], "drawable_edges": [],
              "trace_sha256": trace_meta["sha256"], "clock_alignment_status": trace_meta["status"],
              "limitations": list(_LIMITATIONS), "evidence_omitted": False,
              "anchor_status_counts": dict(Counter(b["status"] for b in bindings))}
    if not tasks:
        result.update(status="NO_DEVICE_ANCHORS", source_task_status_counts={}, scan_passes=0)
        return result
    directory = core._io_path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / "anchors-host-events.csv"
    protected = {core._source_path(trace_path)}
    if capture.get("path"):
        protected.add(core._source_path(capture["path"]))
    for output in (destination, directory / "anchors-source-links.csv"):
        if core._source_path(output) in protected:
            raise ValueError("Anchor evidence output overlaps a source input")
    stages = [{"round_id": "anchor:" + str(t.row_index), "window_key": "source_only",
               "start_ns": str(t.start_ns), "end_ns": str(t.end_ns),
               "_task_rows": [t.row_index], "_uncovered": [], "no_recorded_ms": 0.}
              for t in tasks]
    before = _stamp(core._io_path(trace_path))
    # The core persists all source/flow events before filtering Host/U overlaps;
    # no enlarged U windows or fabricated missing flows are needed.
    summary = core.residual_host_rows(trace_path, capture, [], [], metadata, int(offset),
                                      destination, directory, progress, stages)
    if _stamp(core._io_path(trace_path)) != before:
        raise ValueError("Trace changed during core anchor-source resolution")
    complete_anchors = all(b["status"] in ("EXACT_SOURCE_ROW", "HC_MARKER_TIMESTAMP_UNIQUE") for b in bindings)
    overall = "PARTIAL" if summary["source_status"] == "COMPLETE" and not complete_anchors else summary["source_status"]
    result.update(status=overall, source_status=summary["source_status"], source_scope=summary.get("source_scope"),
                  source_reason=summary.get("source_reason"), scan_passes=1)
    if not summary.get("source_links_csv"):
        result["source_task_status_counts"] = {}
        return result
    with core._io_path(summary["source_links_csv"]).open(encoding="utf-8-sig", newline="") as stream:
        links = [_parse_source(row) for row in csv.DictReader(stream)]
    if {row["row_index"] for row in links} != {int(t.row_index) for t in tasks}:
        raise ValueError("Core source diagnostics lost a requested device anchor")
    wanted = {int(e["event_index"]) for e in host_events}
    for link in links:
        wanted.update(link[key] for key in _INDEX_FIELDS if link.get(key) is not None)
        wanted.update(int(e["id"]) for e in link["cpu_ancestors"])
    if progress:
        progress("Pass 4/4: verifying source/parent references")
    raw = _read_evidence_events(core, trace_path, metadata, wanted, progress)
    index = _HostIntervals(host_events)
    identities = _python_identities(raw)
    associations = _AssociationCache(raw, identities, index)
    binding_indexes = _bindings_by_row(bindings)
    for link in links:
        link["source_event_indexes"] = sorted({link[k] for k in _INDEX_FIELDS if link.get(k) is not None}
                                             | {int(e["id"]) for e in link["cpu_ancestors"]})
        link["missing_source_event_indexes"] = _missing_references(link["source_event_indexes"], raw)
        link["host_associations"] = associations.associations(link)
        link["anchor_binding_indexes"] = list(binding_indexes.get(link["row_index"], []))
    result.update(links=links, source_events=[raw[k] for k in sorted(raw)], scan_passes=2,
                  source_task_status_counts=dict(Counter(link["status"] for link in links)),
                  missing_host_event_indexes=_missing_references((int(e["event_index"]) for e in host_events), raw))
    result["drawable_edges"] = _drawable_edges(links, bindings)
    return result


def _round_scopes(rounds, host_events):
    """Recorded family roots, not a claim that every overlap is submission."""
    events = {int(event["event_index"]): event for event in host_events}
    result = []
    for row in rounds:
        pair = row.get("host_pairing") or {}
        selected = dict(pair.get("functions") or {})
        selected.update({"context_"+key: value for key, value in (pair.get("context") or {}).items()})
        seen = set()
        for role, number in selected.items():
            if number is None or int(number) not in events or (role, int(number)) in seen:
                continue
            seen.add((role, int(number)))
            event = events[int(number)]
            result.append({"round_id": row["round_id"], "role": role,
                           "workload_round": "n" if role in ("outer_propose", "context_current_execute", "context_current_sample") else "n+1",
                           "host_event_index": int(number), "name": event.get("name"),
                           "start_ns": int(event["start_ns"]), "end_ns": int(event["end_ns"])})
        for role, lane in (row.get("host_lanes") or {}).items():
            number = lane.get("event_index", lane.get("start_event_index"))
            if number is None:
                number = (pair.get("functions") or {}).get(role)
            if number is None or (role, int(number)) in seen:
                continue
            seen.add((role, int(number)))
            result.append({"round_id": row["round_id"], "role": role,
                           "workload_round": "n" if role == "outer_propose" else "n+1",
                           "host_event_index": int(number), "name": events.get(int(number), {}).get("name"),
                           "start_ns": int(lane["start_ns"]), "end_ns": int(lane["end_ns"])})
    return result


def _overlap(left, right, start, end):
    """Half-open spans; a zero-duration event is an observation at one point."""
    return left < end and right > start if left != right else start <= left < end


def _temporal_bindings(tasks, rounds, scopes=()):
    result = []
    for task in tasks:
        a, b = int(task["start_ns"]), int(task["end_ns"])
        identity = {"task_key": task["task_key"], "row_index": task["row_index"],
                    "csv_row_index": task.get("csv_row_index"), "trace_event_index": task.get("trace_event_index")}
        for row in rounds:
            left, right = int(row["start_ns"]), int(row["end_ns"])
            if not _overlap(a, b, left, right):
                continue
            stages = []
            for window in row.get("u_windows") or []:
                for stage in window.get("stages") or []:
                    start, end = int(stage["start_ns"]), int(stage["end_ns"])
                    if _overlap(a, b, start, end):
                        stages.append({"stage_number": stage.get("stage_number"), "stage_key": stage.get("stage_key"),
                            "window_key": window.get("window_key"), "status": stage.get("status"),
                            "overlap_start_ns": max(a, start), "overlap_end_ns": min(b, end),
                            "overlap_duration_ns": max(0, min(b, end)-max(a, start))})
            result.append({**identity, "round_id": row["round_id"], "relation": "DEVICE_CYCLE_OVERLAP",
                           "runtime_submission_confirmed": False, "overlap_start_ns": max(a, left),
                           "overlap_end_ns": min(b, right), "overlap_duration_ns": max(0, min(b, right)-max(a, left)),
                           "u_stage_overlaps": stages,
                           "hc_region_overlaps": [name for name, region in (row.get("regions") or {}).items()
                               if _overlap(a, b, int(region["start_ns"]), int(region["end_ns"]))],
                           "scope": "Device execution intersects the cycle; this does not identify its Host submitter or a global batch"})
        for scope in scopes:
            if _overlap(a, b, scope["start_ns"], scope["end_ns"]):
                result.append({**identity, "round_id": scope["round_id"], "role": scope["role"],
                    "workload_round": scope["workload_round"], "host_event_index": scope["host_event_index"],
                    "relation": "HOST_SCOPE_TIME_OVERLAP", "runtime_submission_confirmed": False,
                    "overlap_start_ns": max(a, scope["start_ns"]), "overlap_end_ns": min(b, scope["end_ns"]),
                    "overlap_duration_ns": max(0, min(b, scope["end_ns"])-max(a, scope["start_ns"])),
                    "scope": "Execution-time overlap only; asynchronous submission can precede execution and other functions can overlap"})
    return result


def _catalog(capture, trace_device_events=()):
    """Unfiltered CSV catalog plus device observations missing from that CSV.

    Negative row IDs are internal resolver keys for trace-only observations;
    ``csv_row_index`` remains None and ``task_key`` is a namespaced identity.
    Ambiguous trace/CSV observations are preserved without counting them as a
    definitely independent physical task.
    """
    tasks, resolver, counts, by_shape = [], [], Counter(), defaultdict(list)
    csv_tasks = list(capture.get("tasks") or [])
    multiplicity = Counter(int(t.row_index) for t in csv_tasks)
    for task in csv_tasks:
        row = int(task.row_index)
        counts[row] += 1
        key = "csv:"+str(row) + (":duplicate:"+str(counts[row]) if multiplicity[row] > 1 else "")
        record = {**_task_identity(task), "task_key": key, "source_kind": "KERNEL_CSV",
                  "csv_row_index": row, "trace_event_index": None, "trace_event_indexes": [],
                  "catalog_identity_status": "AMBIGUOUS_CSV_ROW" if multiplicity[row] > 1 else "CSV_ROW_IDENTITY",
                  "counts_as_independent_task": multiplicity[row] == 1}
        tasks.append(record)
        by_shape[(task.name, int(task.start_ns), int(task.end_ns))].append(record)
        if multiplicity[row] == 1:
            resolver.append(task)
    observations, ambiguous, zero_duration = [], [], 0
    used_trace_ids = set()
    for raw in trace_device_events:
        event = deepcopy(raw)
        number = int(event.get("trace_event_index", event.get("event_index")))
        if number <= 0 or number in used_trace_ids:
            raise ValueError("Trace device event indexes must be unique positive original indexes")
        used_trace_ids.add(number)
        a, b = int(event["start_ns"]), int(event["end_ns"])
        if b < a:
            raise ValueError("Reversed trace device interval")
        zero_duration += int(a == b)
        candidates = []
        for candidate in by_shape.get((event["name"], a, b), []):
            if all(candidate.get(k) is None or event.get(k) is None or str(candidate[k]) == str(event[k])
                   for k in ("stream_id", "task_id", "model_id", "device_id")):
                candidates.append(candidate)
        event.update(trace_event_index=number, task_key="trace:"+str(number),
                     candidate_task_keys=[c["task_key"] for c in candidates])
        if len(candidates) == 1:
            candidate = candidates[0]
            candidate["trace_event_indexes"].append(number)
            event.update(catalog_match_status="CSV_MATCH", represented_by_task_key=candidate["task_key"])
        elif candidates:
            event.update(catalog_match_status="AMBIGUOUS_CSV_MATCH", counts_as_independent_task=False)
            ambiguous.append(deepcopy(event))
            for candidate in candidates:
                # A duplicate CSV row remains ambiguous even before matching a
                # trace event; do not erase that original identity diagnostic.
                if candidate["catalog_identity_status"] != "AMBIGUOUS_CSV_ROW":
                    candidate["catalog_identity_status"] = "AMBIGUOUS_TRACE_CSV_MATCH"
                candidate["counts_as_independent_task"] = False
        else:
            # Core source resolution expects row_index but uses only name, exact
            # timestamps and native identities to match the original trace.
            fields = {key: event.get(key) for key in _IDENTITY_FIELDS}
            fields.update(row_index=-number, row_indices=[-number], name=event["name"],
                          op_type=event.get("op_type", event.get("task_type", "TRACE_DEVICE")),
                          start_ns=a, end_ns=b)
            resolver.append(SimpleNamespace(**fields))
            record = {**fields, "task_key": event["task_key"], "source_kind": "TRACE_ONLY",
                      "csv_row_index": None, "trace_event_index": number, "trace_event_indexes": [number],
                      "row_indices": [], "catalog_identity_status": "TRACE_EVENT_IDENTITY",
                      "counts_as_independent_task": True, "category": event.get("cat"),
                      "trace_pid": event.get("pid"), "trace_tid": event.get("tid"),
                      "connection_id": event.get("connection_id"), "correlation_id": event.get("correlation_id")}
            tasks.append(record)
            event.update(catalog_match_status="TRACE_ONLY", represented_by_task_key=record["task_key"])
        observations.append(event)
    return tasks, resolver, observations, ambiguous, zero_duration


def export_device_catalog(capture, rounds, *, trace_device_events=(), trace_meta=None, offset=None):
    """Device-only catalog, with trace observations only under a verified clock.

    The caller may retain an already SHA-checked collector snapshot before
    source resolution. This protects parsed DMA and other trace-only tasks if
    source resolution later fails, without exposing untrusted Host evidence.
    """
    trace_device_events = list(trace_device_events)
    if trace_device_events:
        if (not trace_meta or trace_meta.get("status") != "ALIGNED_X_EVENTS" or not trace_meta.get("sha256")
                or offset is None or int(offset) != int(trace_meta.get("clock_offset_ns", offset))):
            raise ValueError("Aligned trace metadata, SHA256 and matching offset required for trace device snapshot")
    tasks, _, observations, ambiguous, zeros = _catalog(capture, trace_device_events)
    result = {"schema": "u_swimlane_device_task_links_v1", "status": "DEVICE_ONLY",
            "catalog_scope": "All parsed selected-rank kernel CSV tasks plus already verified trace-only device observations; full intervals retained, no task-name or time filtering" if trace_device_events else
                             "All parsed tasks in the selected rank kernel CSV; full intervals retained, no task-name or time filtering",
            "device_tasks": tasks, "links": [], "source_events": [], "task_edges": [],
            "task_round_bindings": _temporal_bindings(tasks, rounds),
            "trace_device_observations": observations, "ambiguous_trace_device_observations": ambiguous,
            "trace_zero_duration_observations": zeros,
            "trace_only_task_status": "RETAINED" if trace_device_events else "UNAVAILABLE_CLOCK_UNALIGNED_OR_NOT_SCANNED",
            "evidence_omitted": False, "scan_passes": 0, "task_count": len(tasks),
            "limitations": ["DEVICE_CYCLE_OVERLAP describes time context only, not a Host submission.",
                            "No trace-only device coordinates or Host sources are trusted without aligned trace evidence."]}
    if trace_device_events:
        result.update(trace_sha256=trace_meta["sha256"], clock_alignment_status=trace_meta["status"])
    return result


def _source_status(links):
    statuses = Counter(link["status"] for link in links)
    return ("COMPLETE" if links and statuses["LINKED"] == len(links) else
            "PARTIAL" if any(statuses[k] for k in ("LINKED", "CPU_ONLY", "API_ONLY")) else "UNAVAILABLE")


def export_device_task_links(core, capture, rounds, host_events, trace_path, trace_meta,
                             offset, directory, progress=None, *, trace_device_events=()):
    """Resolve every catalog task, including interior kernels and trace-only DMA.

    The unchanged core joins all capture tasks, not a stage-boundary whitelist.
    Keeping the full rank catalog also retains delayed asynchronous tasks that
    execute outside a selected cycle. Round/context bindings are separately
    labeled as temporal candidates or explicit source-owner proof.
    """
    if trace_meta.get("status") != "ALIGNED_X_EVENTS" or not trace_meta.get("sha256"):
        raise ValueError("Aligned trace metadata and SHA256 are required for device task links")
    if offset is None or int(offset) != int(trace_meta.get("clock_offset_ns", offset)):
        raise ValueError("Device task source clock offset differs from the verified trace")
    trace_device_events = list(trace_device_events)
    metadata = deepcopy(trace_meta)
    metadata["clock_offset_ns"] = str(int(offset))
    catalog, resolver_tasks, observations, ambiguous, zeros = _catalog(capture, trace_device_events)
    scopes = _round_scopes(rounds, host_events)
    bindings = _temporal_bindings(catalog, rounds, scopes)
    anchor_bindings, anchor_tasks = _device_anchors(core, capture, rounds)
    result = {"schema": "u_swimlane_device_task_links_v1", "device_tasks": catalog,
              "links": [], "source_events": [], "task_edges": [], "task_round_bindings": bindings,
              "trace_device_observations": observations, "ambiguous_trace_device_observations": ambiguous,
              "trace_sha256": metadata["sha256"], "clock_alignment_status": metadata["status"],
              "trace_only_task_status": "RETAINED" if trace_device_events else "NOT_REQUESTED",
              "trace_zero_duration_observations": zeros, "evidence_omitted": False,
              "catalog_scope": "All parsed selected-rank kernel CSV tasks plus aligned trace-only device X-event observations; no task-name, time, stage or record-budget filtering",
              "source_resolution_scope": "All catalog tasks sent once to the unchanged flow/connection/queue resolver, including tasks outside selected Host and device cycles",
              "scan_passes": 0, "host_family_scopes": scopes, "task_count": len(catalog),
              "limitations": [*_LIMITATIONS,
                "Only DEVICE_CYCLE_OVERLAP and HOST_SCOPE_TIME_OVERLAP are time context; they never identify a submitter.",
                "EXPLICIT_SOURCE_OWNER requires complete recorded source and Python parent evidence; containment remains a candidate.",
                "Trace-only internal negative row IDs are not CSV rows; task_key/csv_row_index/trace_event_index distinguish their provenance.",
                "An ambiguous trace/CSV identity is a retained observation, not an additional proven independent task.",
                "Zero-duration device observations remain in the catalog, but the existing source resolver does not resolve zero-duration X events."]}
    directory = core._io_path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory/"device-tasks-host-events.csv"
    protected = {core._source_path(trace_path)}
    if capture.get("path"):
        protected.add(core._source_path(capture["path"]))
    for output in (destination, directory/"device-tasks-source-links.csv"):
        if core._source_path(output) in protected:
            raise ValueError("Device task evidence output overlaps a source input")
    all_capture = {**capture, "tasks": resolver_tasks}
    # A single synthetic owner per task requests every task without enlarging U.
    requests = [{"round_id": "device-task:"+str(t.row_index), "window_key": "source_only",
                 "start_ns": str(t.start_ns), "end_ns": str(t.end_ns), "_task_rows": [t.row_index],
                 "_uncovered": [], "no_recorded_ms": 0.} for t in resolver_tasks]
    summary = {"source_status": "UNAVAILABLE", "source_links_csv": None, "source_reason": "No resolvable catalog tasks"}
    if resolver_tasks:
        before = _stamp(core._io_path(trace_path))
        if progress:
            progress("Resolve full device catalog: " + format(len(resolver_tasks), ",") + " tasks")
        summary = core.residual_host_rows(trace_path, all_capture, [], [], metadata, int(offset),
                                          destination, directory, progress, requests)
        if _stamp(core._io_path(trace_path)) != before:
            raise ValueError("Trace changed during full device-source resolution")
        result["scan_passes"] = 1
    source_rows = []
    if summary.get("source_links_csv"):
        with core._io_path(summary["source_links_csv"]).open(encoding="utf-8-sig", newline="") as stream:
            source_rows = [_parse_source(row) for row in csv.DictReader(stream)]
        if {link["row_index"] for link in source_rows} != {int(t.row_index) for t in resolver_tasks}:
            raise ValueError("Core source diagnostics lost a requested catalog task")
    wanted = {int(e["event_index"]) for e in host_events}
    for link in source_rows:
        wanted.update(link[key] for key in _INDEX_FIELDS if link.get(key) is not None)
        wanted.update(int(e["id"]) for e in link["cpu_ancestors"])
    raw = {}
    if resolver_tasks:
        if progress:
            progress("Verify full device-catalog source/parent references")
        raw = _read_evidence_events(core, trace_path, metadata, wanted, progress, source_links=source_rows)
        result["scan_passes"] = 2
    if progress:
        progress("Building device-source indexes: " + format(len(raw), ",") + " evidence events / " +
                 format(len(host_events), ",") + " recorded Host functions")
    # Include all recorded Python ancestors recovered by core, even if the old
    # five-lane recognizer did not whitelist their names.
    python_hosts = {int(event["event_index"]): event for event in host_events}
    python_hosts.update({number: event for number, event in raw.items() if "python_function" in event.get("cat", "").lower()})
    index, identities = _HostIntervals(python_hosts.values()), _python_identities(raw)
    associations, flows = _AssociationCache(raw, identities, index), _FlowReferences(raw)
    by_row = {link["row_index"]: link for link in source_rows}
    scopes_by_id = defaultdict(list)
    for scope in scopes:
        scopes_by_id[scope["host_event_index"]].append(scope)
    task_links, edges = [], []
    temporal_rounds = defaultdict(set)
    for binding in bindings:
        temporal_rounds[binding["task_key"]].add(binding["round_id"])
    association_started = association_last = time.monotonic()
    if progress:
        progress("Device task associations: 0/" + format(len(catalog), ",") + " tasks")
    for task_number, task in enumerate(catalog, 1):
        original = by_row.get(task["row_index"]) if task["catalog_identity_status"] != "AMBIGUOUS_CSV_ROW" else None
        link = deepcopy(original) if original else {"row_index": task["row_index"], "name": task["name"],
            "device_start_ns": task["start_ns"], "device_end_ns": task["end_ns"],
            "status": "AMBIGUOUS" if task["catalog_identity_status"] == "AMBIGUOUS_CSV_ROW" else "UNAVAILABLE",
            "reason": "Duplicate CSV row identity" if task["catalog_identity_status"] == "AMBIGUOUS_CSV_ROW" else summary.get("source_reason"),
            "cpu_ancestors": [], "flow_ids": []}
        link.update(task_key=task["task_key"], csv_row_index=task.get("csv_row_index"),
                    trace_event_index=task.get("trace_event_index"), source_kind=task["source_kind"])
        invalid_identity = (task["catalog_identity_status"] == "AMBIGUOUS_TRACE_CSV_MATCH" or
                            trace_device_events and task["source_kind"] == "KERNEL_CSV" and
                            not task["trace_event_indexes"] and original and original.get("status") != "NO_TRACE_DEVICE")
        if invalid_identity:
            # The legacy resolver lacks a CSV/trace catalog cross-check on
            # device_id and can match two unknown-ID rows to one trace event.
            # Preserve its diagnostic candidate without claiming a source for
            # a catalog identity that cannot be uniquely established.
            link["rejected_core_source_candidate"] = deepcopy(original)
            link["original_core_source_status"] = link.get("status")
            link.update(status="AMBIGUOUS" if task["catalog_identity_status"] == "AMBIGUOUS_TRACE_CSV_MATCH" else "NO_TRACE_DEVICE",
                        reason="Trace/CSV native device identity is not uniquely compatible", cpu_ancestors=[], flow_ids=[])
            for key in (*_INDEX_FIELDS, *_NS_FIELDS, "cpu_name", "cpu_leaf_name", "cpu_category", "connection_id", "correlation_id"):
                if key not in ("device_start_ns", "device_end_ns"):
                    link[key] = None
        link["flow_event_indexes"] = flows.event_indexes(link)
        link["source_event_indexes"] = sorted({link[k] for k in _INDEX_FIELDS if link.get(k) is not None}
                                              | {int(e["id"]) for e in link.get("cpu_ancestors", [])}
                                              | set(link["flow_event_indexes"]))
        link["missing_source_event_indexes"] = _missing_references(link["source_event_indexes"], raw)
        link["host_associations"] = associations.associations(link) if original else []
        linked_rounds = set()
        for association in link["host_associations"]:
            for scope in scopes_by_id.get(association["host_event_index"], []):
                confirmed = association["runtime_submission_confirmed"] is True
                bindings.append({"round_id": scope["round_id"], "role": scope["role"], "workload_round": scope["workload_round"],
                    "task_key": task["task_key"], "row_index": task["row_index"], "csv_row_index": task.get("csv_row_index"),
                    "trace_event_index": task.get("trace_event_index"), "host_event_index": association["host_event_index"],
                    "source_event_index": association["source_event_index"], "host_relation": association["relation"],
                    "relation": "EXPLICIT_SOURCE_OWNER" if confirmed else "HOST_SOURCE_TIME_CONTAINMENT_ONLY",
                    "runtime_submission_confirmed": confirmed,
                    "explicit_parent_event_indexes": association.get("explicit_parent_event_indexes", []),
                    "scope": "Recorded Host/source owner" if confirmed else "Containing Host scope only; caller and round ownership unproven"})
                linked_rounds.add(scope["round_id"])
        link["task_round_ids"] = sorted(temporal_rounds[task["task_key"]] | linked_rounds)
        for association in link["host_associations"]:
            # A task can overlap several rounds and have unrelated containing
            # scopes. Only this association's family owner (or an explicitly
            # recorded ancestor of this exact source/Host path) binds its edge.
            edge_rounds = {s["round_id"] for s in scopes_by_id.get(association["host_event_index"], [])}
            if association["runtime_submission_confirmed"]:
                for owner in link["host_associations"]:
                    if (owner["runtime_submission_confirmed"]
                            and owner["source_event_index"] == association["source_event_index"]
                            and association["host_event_index"] in owner.get("explicit_parent_event_indexes", [])):
                        edge_rounds.update(s["round_id"] for s in scopes_by_id.get(owner["host_event_index"], []))
            edges.append({**deepcopy(association), "task_key": task["task_key"], "row_index": task["row_index"],
                          "csv_row_index": task.get("csv_row_index"), "trace_event_index": task.get("trace_event_index"),
                          "source_status": link["status"], "round_ids": sorted(edge_rounds),
                          "device_start_ns": task["start_ns"], "device_end_ns": task["end_ns"],
                          "edge_kind": "HOST_TO_DEVICE_TASK", "target": "whole device task, not a U boundary"})
        task_links.append(link)
        if progress:
            now = time.monotonic()
            if now-association_last >= 5:
                progress("Device task associations: " + format(task_number, ",") + "/" +
                         format(len(catalog), ",") + " tasks; elapsed " + format(now-association_started, ".1f") + "s")
                association_last = now
    result.update(links=task_links, task_edges=edges, source_events=[raw[k] for k in sorted(raw)],
                  status=_source_status(task_links), source_status=_source_status(task_links),
                  core_source_status=summary["source_status"],
                  source_scope=summary.get("source_scope"), source_reason=summary.get("source_reason"),
                  source_task_status_counts=dict(Counter(link["status"] for link in task_links)),
                  missing_host_event_indexes=_missing_references((int(e["event_index"]) for e in host_events), raw))
    if progress:
        progress("Device task associations complete: " + format(len(task_links), ",") + "/" +
                 format(len(catalog), ",") + " tasks; elapsed " + format(time.monotonic()-association_started, ".1f") +
                 "s; source statuses=" + json.dumps(result["source_task_status_counts"], sort_keys=True))
        progress("Projecting legacy U/HC anchor evidence from the same full task catalog")
    # Project old boundary-only evidence from the same resolution pass. Legacy
    # consumers keep their original endpoint/arrow semantics and do not trigger
    # a second full trace scan.
    anchor_rows = {int(t.row_index) for t in anchor_tasks}
    old_links = [deepcopy(link) for link in task_links if link.get("csv_row_index") in anchor_rows]
    binding_indexes = _bindings_by_row(anchor_bindings)
    for link in old_links:
        link["anchor_binding_indexes"] = list(binding_indexes.get(link["row_index"], []))
    old_references = {int(e["event_index"]) for e in host_events}
    for link in old_links:
        old_references.update(link.get("source_event_indexes", []))
    old_source_status = _source_status(old_links)
    anchor_status = ("PARTIAL" if old_source_status == "COMPLETE" and any(
        b["status"] not in ("EXACT_SOURCE_ROW", "HC_MARKER_TIMESTAMP_UNIQUE") for b in anchor_bindings) else old_source_status)
    result["anchor_links"] = {"schema": "u_swimlane_anchor_links_v1", "anchor_bindings": anchor_bindings,
        "device_tasks": [_task_identity(t) for t in anchor_tasks], "links": old_links,
        "source_events": [raw[k] for k in sorted(old_references) if k in raw],
        "drawable_edges": _drawable_edges(old_links, anchor_bindings),
        "trace_sha256": metadata["sha256"], "clock_alignment_status": metadata["status"],
        "status": anchor_status, "source_status": old_source_status, "scan_passes": 0,
        "source_scope": summary.get("source_scope"), "source_reason": summary.get("source_reason"),
        "anchor_status_counts": dict(Counter(b["status"] for b in anchor_bindings)),
        "source_task_status_counts": dict(Counter(link["status"] for link in old_links)),
        "evidence_omitted": False, "resolution_reused_from": "device_task_links", "limitations": list(_LIMITATIONS)}
    return result
