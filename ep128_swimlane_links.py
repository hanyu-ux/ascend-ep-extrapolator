"""Conservative submission evidence for saved swimlane device anchors.

The core owns all flow/connection/queue resolution.  This module requests only
device anchors, preserves the core's status, and separately checks whether an
exported Python function has an explicit parent-ID path to the recorded source.
An interval containing an API is useful context, never a verified caller.
"""

from bisect import bisect_right
from collections import Counter, defaultdict
from copy import deepcopy
import csv
import json
from pathlib import Path


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
    for task in tasks:
        by_row[int(task.row_index)].append(task)
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
            matches = [t for t in tasks if int(getattr(t, edge + "_ns")) == stamp and (
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


def _read_evidence_events(core, trace_path, trace_meta, wanted, progress):
    """Enrich only original source/ancestor references; never invent flow edges."""
    access = core._io_path(trace_path)
    before, scan, found = _stamp(access), {}, {}
    for number, raw in enumerate(core.residual_iter_trace_events(trace_path, stats=scan), 1):
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
        if progress and number % 100000 == 0:
            progress("Anchor source references: " + format(number, ",") + " events")
    if _stamp(access) != before or scan.get("sha256") != trace_meta["sha256"]:
        raise ValueError("Trace changed or SHA256 differs during anchor-source enrichment")
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
            maximum, prefix = -2**63, []
            for event in group:
                maximum = max(maximum, int(event["end_ns"]))
                prefix.append(maximum)
            self.lanes[lane] = group, [int(e["start_ns"]) for e in group], prefix

    def containing(self, event):
        group, starts, prefix = self.lanes.get((str(event.get("pid")), str(event.get("tid"))), ([], [], []))
        start, end = int(event["start_ns"]), int(event["end_ns"])
        first, stop = bisect_right(prefix, start), bisect_right(starts, start)
        return [e for e in group[first:stop] if int(e["end_ns"]) >= end]


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
    for link in links:
        link["source_event_indexes"] = sorted({link[k] for k in _INDEX_FIELDS if link.get(k) is not None}
                                             | {int(e["id"]) for e in link["cpu_ancestors"]})
        link["missing_source_event_indexes"] = sorted(set(link["source_event_indexes"]) - raw.keys())
        link["host_associations"] = _associations(link, raw, identities, index)
        link["anchor_binding_indexes"] = [i for i, b in enumerate(bindings) if link["row_index"] in b["device_row_indexes"]]
    result.update(links=links, source_events=[raw[k] for k in sorted(raw)], scan_passes=2,
                  source_task_status_counts=dict(Counter(link["status"] for link in links)),
                  missing_host_event_indexes=sorted({int(e["event_index"]) for e in host_events} - raw.keys()))
    result["drawable_edges"] = _drawable_edges(links, bindings)
    return result
