"""Next-main residual accounting on the source capture's aligned clock."""

from collections import defaultdict
import math
from statistics import fmean, median


PHASES = ("main_inputs", "main_metadata")
METRICS = (
    "stage_wall_ms", "device_busy_csv_ms", "device_uncovered_csv_ms",
    "dma_additional_ms", "device_busy_with_dma_ms", "device_uncovered_after_dma_ms",
    "draft_hc_ms", "draft_graph_ms", "host_linked_ms", "host_in_draft_hc_ms",
    "host_in_draft_graph_ms", "host_on_draft_tasks_ms", "host_before_draft_graph_ms",
    "host_after_draft_graph_ms",
)
ROUND_FIELDS = (
    "case", "rank", "bs", "S", "round_id", "stage_number", "stage_key", "stage_status", "stage_reason",
    "stage_start_ns", "stage_end_ns", "trace_status", "dma_status", "draft_graph_status",
    "draft_graph_reason", "draft_model_id", "draft_hc_start_ns", "draft_hc_end_ns",
    "draft_graph_start_ns", "draft_graph_end_ns", "host_status", "device_tasks", "linked_device_tasks",
    "linked_host_events", *METRICS,
)


def _union(spans):
    merged = []
    for start, end in sorted(spans):
        if start >= end:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _intersection_ns(left, right):
    """Count common time once for two sorted unions of half-open intervals."""
    i = j = total = 0
    while i < len(left) and j < len(right):
        a, b = left[i]
        c, d = right[j]
        total += max(0, min(b, d) - max(a, c))
        if b <= d:
            i += 1
        else:
            j += 1
    return total


def _draft_graph(cycle, parents, task_index, hc):
    """Identify a single recorded draft replay using its Embedding and Markov tail.

    HC timing remains usable even if the graph's native identity is incomplete.
    """
    embedding, tail, next_inputs = (parents.get(key) for key in ("draft_embedding", "markov", PHASES[0]))
    if not all(row and row["status"] == "MATCHED" for row in (embedding, tail, next_inputs)):
        return None, "Missing matched draft Embedding, Markov or next-input anchor"
    main_end = max(int(layer["end_ns"]) for layer in cycle["layers"] if layer["role"] == "main")
    next_start = int(next_inputs["start_ns"])
    tasks = task_index.overlapping(main_end, next_start)
    anchor = embedding["start_anchor"]
    matches = [t for t in tasks if t.row_index == anchor.get("row_index") and t.start_ns == int(embedding["start_ns"])]
    if len(matches) != 1 or matches[0].model_id in (None, "", "N/A", "4294967295", "-1"):
        return None, "Draft Embedding has no unique native Model ID"
    embed = matches[0]
    graph_tasks = [t for t in tasks if t.model_id == embed.model_id]
    starts = [t for t in graph_tasks if t.task_id == "0" and t.stream_id == embed.stream_id]
    starts_at = min(t.start_ns for t in graph_tasks)
    ends_at = max(t.end_ns for t in graph_tasks)
    # Verify both HC endpoints belong to this replay and the Markov anchor
    # belongs to its tail. Reject ambiguous or truncated task envelopes.
    if (len(starts) != 1 or starts[0].start_ns != starts_at
            or not main_end <= starts_at <= embed.start_ns < hc[0] < hc[1] < ends_at <= next_start
            or not any(t.start_ns == hc[0] for t in graph_tasks)
            or not any(t.end_ns == hc[1] for t in graph_tasks)
            or not any(t.row_index == tail["start_anchor"].get("row_index") for t in graph_tasks)):
        return None, "Native Task ID 0, HC, and draft tail do not delimit one recorded replay"
    return (embed.model_id, starts_at, ends_at, _union((t.start_ns, t.end_ns) for t in graph_tasks)), None


def _cpu_intervals(source_rows, wanted):
    """Use only causal source links already aligned by the core's trace scan."""
    sources, events = {}, {}
    for source in source_rows:
        row = int(source["row_index"])
        if row not in wanted or source["status"] not in ("LINKED", "CPU_ONLY"):
            continue
        event = int(source["cpu_event_index"])
        bounds = (int(source["cpu_start_ns"]), int(source["cpu_end_ns"]))
        if bounds[0] >= bounds[1] or events.setdefault(event, bounds) != bounds or row in sources:
            raise ValueError("Conflicting linked Host event identity or timestamps")
        sources[row] = event
    return sources, events


def analyze_next_main_overlap(measured, stages, task_index, source_rows, *, trace_status):
    """Return two rows per retained HC cycle, including unresolved stages."""
    parents = defaultdict(dict)
    for stage in stages:
        if stage["level"] == 0:
            parents[stage["round_id"]][stage["stage_key"]] = stage
    wanted = {row for group in parents.values() for key in PHASES
              for row in group.get(key, {}).get("_task_rows", ())}
    sources, events = _cpu_intervals(source_rows, wanted) if trace_status == "ALIGNED_X_EVENTS" else ({}, {})
    rows = []
    for cycle in measured["rounds"]:
        round_id = cycle["round_id"]
        group = parents[round_id]
        draft = [layer for layer in cycle["layers"] if layer["role"] == "draft"]
        hc = int(draft[0]["start_ns"]), int(draft[-1]["end_ns"])
        graph, graph_reason = _draft_graph(cycle, group, task_index, hc)
        for number, key in enumerate(PHASES, 13):
            stage = group.get(key, {})
            matched = stage.get("status") == "MATCHED"
            stage_status = "MATCHED" if matched else "UNRESOLVED"
            unresolved = next((s for s in group.values() if s.get("status") == "UNRESOLVED"
                               and s.get("window_key") == "draft_to_nextmain"), {})
            wall = stage.get("duration_ms") if matched else None
            busy = stage.get("recorded_busy_ms") if matched else None
            dma = stage.get("trace_dma_in_csv_uncovered_ms") if matched else None
            task_rows = stage.get("_task_rows", ()) if matched else ()
            linked = [sources[row] for row in task_rows if row in sources]
            cpu = _union(events[event] for event in set(linked))
            host_total = sum(b - a for a, b in cpu) / 1e6 if cpu else None
            host_status = ("COMPLETE" if linked and len(linked) == len(task_rows)
                           else "PARTIAL" if linked else "UNAVAILABLE")
            model, graph_start, graph_end, graph_tasks = graph if graph else (None, None, None, ())
            draft_overlap = _intersection_ns(cpu, [(graph_start, graph_end)]) if graph and cpu else None
            rows.append({
                "round_id": round_id, "stage_number": number, "stage_key": key, "stage_status": stage_status,
                "stage_reason": None if matched else stage.get("reason") or unresolved.get("reason") or "Missing stage anchor",
                "stage_start_ns": stage.get("start_ns") if matched else None,
                "stage_end_ns": stage.get("end_ns") if matched else None,
                "trace_status": trace_status, "dma_status": stage.get("trace_dma_status") if matched else None,
                "draft_graph_status": "MATCHED" if graph else "UNRESOLVED", "draft_graph_reason": graph_reason,
                "draft_model_id": model, "draft_hc_start_ns": str(hc[0]), "draft_hc_end_ns": str(hc[1]),
                "draft_graph_start_ns": str(graph_start) if graph else None,
                "draft_graph_end_ns": str(graph_end) if graph else None,
                "host_status": host_status, "device_tasks": len(task_rows) if matched else None,
                "linked_device_tasks": len(linked) if matched and trace_status == "ALIGNED_X_EVENTS" else None,
                "linked_host_events": len(set(linked)) if matched and trace_status == "ALIGNED_X_EVENTS" else None,
                "stage_wall_ms": wall, "device_busy_csv_ms": busy,
                "device_uncovered_csv_ms": stage.get("no_recorded_ms") if matched else None,
                "dma_additional_ms": dma,
                "device_busy_with_dma_ms": busy + dma if dma is not None else None,
                "device_uncovered_after_dma_ms": stage.get("uncovered_after_trace_dma_ms") if matched else None,
                "draft_hc_ms": (hc[1] - hc[0]) / 1e6,
                "draft_graph_ms": (graph_end - graph_start) / 1e6 if graph else None,
                "host_linked_ms": host_total,
                "host_in_draft_hc_ms": _intersection_ns(cpu, [hc]) / 1e6 if cpu else None,
                "host_in_draft_graph_ms": draft_overlap / 1e6 if draft_overlap is not None else None,
                "host_on_draft_tasks_ms": _intersection_ns(cpu, graph_tasks) / 1e6 if graph and cpu else None,
                "host_before_draft_graph_ms": sum(max(0, min(b, graph_start) - a) for a, b in cpu) / 1e6 if graph and cpu else None,
                "host_after_draft_graph_ms": sum(max(0, b - max(a, graph_end)) for a, b in cpu) / 1e6 if graph and cpu else None,
            })
    return rows


def summarize_next_main_overlap(rows, identity, *, case_error=None):
    result = []
    for number, key in enumerate(PHASES, 13):
        phase = [r for r in rows if r["stage_key"] == key]
        row = {**identity, "stage_number": number, "stage_key": key, "case_error": case_error,
               "rounds": len(phase), "matched_rounds": sum(r["stage_status"] == "MATCHED" for r in phase),
               "draft_graph_matched_rounds": sum(r["draft_graph_status"] == "MATCHED" for r in phase),
               "host_complete_rounds": sum(r["host_status"] == "COMPLETE" for r in phase),
               "host_partial_rounds": sum(r["host_status"] == "PARTIAL" for r in phase),
               "device_tasks_in_matched_stages": sum(r["device_tasks"] or 0 for r in phase),
               "device_tasks_in_host_observed_rounds": sum(r["device_tasks"] or 0 for r in phase if r["host_linked_ms"] is not None),
               "linked_device_tasks": sum(r["linked_device_tasks"] or 0 for r in phase),
               "trace_status": phase[0]["trace_status"] if phase else None}
        for field in METRICS:
            values = sorted(r[field] for r in phase if r[field] is not None)
            row[f"{field}_n"] = len(values)
            row[f"{field}_mean"] = fmean(values) if values else None
            row[f"{field}_p50"] = median(values) if values else None
            row[f"{field}_p90"] = values[math.ceil(0.9 * len(values)) - 1] if values else None
        comparable = [r for r in phase if r["host_in_draft_graph_ms"] is not None]
        denominator = sum(r["host_linked_ms"] for r in comparable)
        row["host_graph_overlap_share"] = (sum(r["host_in_draft_graph_ms"] for r in comparable) / denominator
                                           if denominator else None)
        result.append(row)
    return result
