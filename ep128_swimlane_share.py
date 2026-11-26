"""Select complete PPT representative evidence from an existing swimlane JSON.

This module does not open captures or traces.  Selection reduces the number of
rounds and duplicate representations, never a byte-budget subset of a selected
round's calls, coordinates or recorded source proof.
"""

from collections import Counter, defaultdict, deque
from copy import deepcopy


SCHEMA = "u_swimlane_ppt_evidence_v1"
HOST_ROLES = ("outer_propose", "runner_inputs", "runner_attn", "state_attn")
HOST_LANES = (*HOST_ROLES, "preparation_dp_aggregate")
_HISTORY_FIELDS = {
    "complete_host_rounds": "source_complete_host_rounds", "complete_rounds": "source_complete_rounds",
    "incomplete_rounds": "source_incomplete_rounds", "ready_requires_all_rounds": "source_ready_requires_all_rounds",
    "ready_definition": "source_ready_definition", "round_selection": "source_round_selection",
    "source_link_scope": "source_link_scope_original", "source_link_round_ids": "source_link_round_ids_original",
    "suggested_round_id": "source_suggested_round_id",
}
_SINGLE_REFS = {
    "event_index", "owner_event_index", "parent_event_index", "owning_all_reduce_event_index",
    "host_event_index", "source_event_index", "start_event_index", "end_event_index",
    "cpu_event_index", "cpu_leaf_event_index", "node_event_index", "enqueue_event_index", "dequeue_event_index",
}
_MANY_REFS = {
    "ancestor_event_indexes", "parent_event_indexes", "explicit_parent_event_indexes", "event_indexes",
    "source_event_indexes", "missing_source_event_indexes", "missing_host_event_indexes",
    "host_family_event_indexes", "host_scope_event_indexes", "source_required_event_indexes",
    "flow_event_indexes",
}
_STAMP_KEYS = {"start_ns", "end_ns", "origin_ns", "timestamp_ns", "clock_offset_ns", "csv_start_ns", "trace_start_ns"}
_DUPLICATE_REF_FIELDS = ("name", "kind", "start_ns", "end_ns", "duration_ns", "event_type")
_EVENT_IDENTITY = ("name", "pid", "tid", "start_ns", "end_ns", "python_id", "python_parent_id")


def _index(value):
    if value is None or value == "":
        return None
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError("Event and row indexes must be integers")
    result = int(value)
    if result < 0:
        raise ValueError("Event and row indexes must not be negative")
    return result


def _times(value, to_wire=False, key=""):
    if isinstance(value, dict):
        return {k: _times(v, to_wire, k) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_times(v, to_wire) for v in value]
    stamp = key in _STAMP_KEYS or key.endswith("_start_ns") or key.endswith("_end_ns")
    if stamp and value is not None:
        return str(int(value)) if to_wire else int(value)
    return value


def _events(values):
    ordered, lookup = [], {}
    for item in values:
        event = dict(item)
        number = _index(event.get("event_index"))
        if number is None:
            raise ValueError("Saved event has no original event_index")
        if number in lookup:
            raise ValueError("Duplicate saved event_index within one case")
        event["event_index"] = number
        for key in ("start_ns", "end_ns"):
            if key in event:
                event[key] = int(event[key])
        ordered.append(event)
        lookup[number] = event
    return ordered, lookup


def _refs(value):
    """Collect only event-index namespaces, never task/row/correlation IDs."""
    result = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key in _SINGLE_REFS:
                number = _index(item)
                if number is not None:
                    result.add(number)
            elif key in _MANY_REFS:
                for reference in item or []:
                    number = _index(reference)
                    if number is not None:
                        result.add(number)
            elif key == "cpu_ancestors":
                for ancestor in item or []:
                    number = _index(ancestor.get("id"))
                    if number is not None:
                        result.add(number)
                    result.update(_refs(ancestor))
            else:
                result.update(_refs(item))
    elif isinstance(value, (tuple, list)):
        for item in value:
            result.update(_refs(item))
    return result


def _round_refs(row):
    result = _refs(row)
    pair = row.get("host_pairing") or {}
    for key in ("functions", "context"):
        for value in (pair.get(key) or {}).values():
            number = _index(value)
            if number is not None:
                result.add(number)
    for values in (pair.get("candidates") or {}).values():
        for value in values or []:
            number = _index(value)
            if number is not None:
                result.add(number)
    return result


def _identity_equal(a, b, key):
    if key in ("start_ns", "end_ns"):
        return int(a) == int(b)
    return a is None and b is None or a is not None and b is not None and str(a) == str(b)


def _available(host, source):
    merged = dict(source)
    for number, event in host.items():
        old = merged.get(number)
        if old is not None:
            for key in _EVENT_IDENTITY:
                if key in old and key in event and not _identity_equal(old[key], event[key], key):
                    raise ValueError("Host/source event identity conflict: " + str(number) + " / " + key)
            merged[number] = {**old, **event}
        else:
            merged[number] = event
    return merged


def _closure(seeds, events):
    """Preserve explicit index references and all recorded parent candidates.

    Parent candidates are retained for audit, not resolved or promoted here.
    In particular an ambiguous parent remains ambiguous after this selection.
    """
    parents = defaultdict(list)
    for event in events.values():
        if event.get("python_id") is not None:
            parents[(str(event.get("pid")), str(event.get("tid")), str(event["python_id"]))].append(event)
    pending, wanted = deque(seeds), set(seeds)
    while pending:
        number = pending.popleft()
        event = events.get(number)
        if event is None:
            continue
        more = _refs(event)
        parent_id = event.get("python_parent_id")
        if parent_id is not None and "start_ns" in event and "end_ns" in event:
            key = (str(event.get("pid")), str(event.get("tid")), str(parent_id))
            more.update(parent["event_index"] for parent in parents.get(key, [])
                        if "start_ns" in parent and "end_ns" in parent
                        and int(parent["start_ns"]) <= int(event["start_ns"])
                        and int(parent["end_ns"]) >= int(event["end_ns"]))
        for candidate in more - wanted:
            wanted.add(candidate)
            pending.append(candidate)
    return wanted


def _selected_round(case):
    rounds = case.get("rounds") or []
    if not rounds:
        return None, "NO_SAVED_ROUND"
    by_id = {}
    for row in rounds:
        identity = row.get("round_id")
        if identity is None or identity in by_id:
            raise ValueError("Saved rounds need unique round_id within each case")
        by_id[identity] = row
    suggested = case.get("suggested_round_id")
    linked = [identity for identity in case.get("source_link_round_ids", []) if identity in by_id]
    if suggested in by_id and (not linked or suggested in linked):
        return by_id[suggested], "ORIGINAL_SUGGESTED_ROUND"
    if linked:
        ordered = sorted((by_id[identity] for identity in set(linked)),
                         key=lambda row: (int(row["duration_ns"]), str(row["round_id"])))
        return ordered[len(ordered)//2], "ORIGINAL_SOURCE_LINKED_ROUND_PRIORITY"
    candidates = [row for row in rounds if row.get("stage_status") == "MATCHED"
                  and (row.get("host_pairing") or {}).get("status") == "MATCHED"] or rounds
    ordered = sorted(candidates, key=lambda row: (int(row["duration_ns"]), str(row["round_id"])))
    return ordered[len(ordered)//2], "MEDIAN_SAVED_CYCLE_NO_ORIGINAL_SUGGESTION"


def _link_subset(original, round_id):
    """Subset recorded source evidence without re-resolving its confidence."""
    if not original:
        return {"status": "UNAVAILABLE", "reason": "No saved device-source evidence", "anchor_bindings": [],
                "links": [], "drawable_edges": [], "device_tasks": [], "source_events": []}, set(), set()
    result = deepcopy({k: v for k, v in original.items() if k not in (
        "anchor_bindings", "links", "drawable_edges", "device_tasks", "source_events", "missing_host_event_indexes")})
    old_bindings = original.get("anchor_bindings") or []
    indexes = [i for i, binding in enumerate(old_bindings) if binding.get("round_id") == round_id]
    remap = {old: new for new, old in enumerate(indexes)}
    bindings = [deepcopy(old_bindings[i]) for i in indexes]
    rows = {_index(number) for binding in bindings for number in binding.get("device_row_indexes", [])}
    rows.discard(None)
    links = []
    for link in original.get("links") or []:
        old_ids = [_index(i) for i in link.get("anchor_binding_indexes", [])]
        selected = [i for i in old_ids if i in remap]
        if not selected:
            continue
        number = _index(link.get("row_index"))
        if number not in rows or any(number not in old_bindings[i].get("device_row_indexes", []) for i in selected):
            raise ValueError("Source link/boundary device-row identity conflict")
        saved = deepcopy(link)
        saved["anchor_binding_indexes"] = [remap[i] for i in selected]
        links.append(saved)
    edges = []
    for edge in original.get("drawable_edges") or []:
        old = _index(edge.get("anchor_binding_index"))
        if old not in remap:
            continue
        if edge.get("round_id") != round_id or _index(edge.get("row_index")) not in old_bindings[old].get("device_row_indexes", []):
            raise ValueError("Drawable edge/boundary identity conflict")
        saved = deepcopy(edge)
        saved["anchor_binding_index"] = remap[old]
        edges.append(saved)
    result.update(anchor_bindings=bindings, links=links, drawable_edges=edges,
                  device_tasks=[deepcopy(task) for task in original.get("device_tasks", [])
                                if _index(task.get("row_index")) in rows],
                  source_binding_indexes=indexes,
                  selection_scope="Only selected-round boundaries and their recorded source evidence; statuses are copied, not re-resolved")
    for key, selected_counts in (
            ("anchor_status_counts", Counter(binding.get("status") for binding in bindings)),
            ("source_task_status_counts", Counter(link.get("status") for link in links))):
        if key in result:
            result["original_" + key] = result[key]
        result[key] = dict(selected_counts)
    if "missing_host_event_indexes" in original:
        result["original_missing_host_event_count"] = len(original["missing_host_event_indexes"])
    return result, _refs(links) | _refs(edges), rows


def _task_key(record):
    """Keep the task namespace separate from event indexes and CSV row IDs.

    Trace-only tasks use ``trace:<event_index>`` and may retain a negative
    internal resolver row. Neither is interpreted as a physical CSV row.
    """
    key = record.get("task_key")
    if key is not None:
        if not isinstance(key, str) or not key:
            raise ValueError("Saved device task_key must be a nonempty string")
        return key
    row = record.get("row_index")
    if isinstance(row, bool) or not isinstance(row, (int, str)):
        if row is None:
            return None
        raise ValueError("Saved task row must be an integer")
    return "row:" + str(int(row))


def _binding_relation(binding):
    return binding.get("relation", binding.get("binding_type", binding.get("type")))


def _edge_in_round(edge, round_id):
    """Read explicit edge context; task membership alone does not own an edge."""
    identities = edge.get("round_ids") or []
    if not isinstance(identities, (list, tuple)):
        raise ValueError("Saved task edge round_ids must be a list")
    return edge.get("round_id") == round_id or round_id in identities


def _observation_task_keys(observation):
    keys = set(observation.get("candidate_task_keys") or [])
    for key in ("represented_by_task_key", "task_key"):
        value = observation.get(key)
        if isinstance(value, str) and value:
            keys.add(value)
    return keys


def _device_only_binding(binding):
    # Only a recorded device-cycle relation is trusted when Host alignment
    # failed. Unexpected Host fields cannot leak through this allowlist.
    allowed = {"round_id", "task_key", "row_index", "csv_row_index", "trace_event_index", "source_kind",
               "relation", "binding_type", "type", "cycle_start_ns", "cycle_end_ns", "round_start_ns",
               "round_end_ns", "device_start_ns", "device_end_ns", "overlap_start_ns", "overlap_end_ns",
               "overlap_ns", "overlap_duration_ns", "cycle_overlap_ns", "time_relation", "scope", "status"}
    return {key: deepcopy(value) for key, value in binding.items() if key in allowed}


def _task_link_subset(original, round_id, *, trusted=True, untrusted_status="CLOCK_UNALIGNED"):
    """Keep every explicitly bound selected-round task, not only U anchors.

    A selected Host can have submitted a task that executes beyond the cycle
    or beyond its own return. Its recorded binding survives unchanged; no new
    round membership or submission confidence is inferred from timestamps.
    """
    if not isinstance(original, dict):
        return None, set(), set()
    array_keys = {"device_tasks", "links", "source_events", "task_round_bindings", "drawable_edges", "task_edges",
                  "ambiguous_trace_device_observations", "trace_device_observations", "host_family_scopes",
                  "ambiguity_context_device_tasks", "missing_host_event_indexes"}
    result = deepcopy({key: value for key, value in original.items() if key not in array_keys})
    all_bindings = original.get("task_round_bindings") or []
    selected = [deepcopy(binding) for binding in all_bindings if binding.get("round_id") == round_id
                and (trusted or _binding_relation(binding) == "DEVICE_CYCLE_OVERLAP")]
    if not trusted:
        selected = [_device_only_binding(binding) for binding in selected]
    keys = {_task_key(binding) for binding in selected}
    if None in keys:
        raise ValueError("Saved selected-round task binding has no task identity")
    tasks = [deepcopy(task) for task in original.get("device_tasks") or [] if _task_key(task) in keys]
    present = {_task_key(task) for task in tasks}
    missing_tasks = sorted(keys - present)
    all_bound = {_task_key(binding) for binding in all_bindings}
    all_bound.discard(None)
    result.update(selected_round_id=round_id, task_round_bindings=selected, device_tasks=tasks,
                  links=[], source_events=[], drawable_edges=[], task_edges=[], ambiguous_trace_device_observations=[],
                  trace_device_observations=[], host_family_scopes=[], ambiguity_context_device_tasks=[],
                  selection_scope="All recorded tasks explicitly bound to the selected round; includes stage-internal and delayed tasks, not only U boundaries",
                  original_catalog_task_count=len(original.get("device_tasks") or []),
                  selected_catalog_task_count=len(tasks),
                  omitted_catalog_task_count=len(original.get("device_tasks") or [])-len(tasks),
                  omitted_unbound_catalog_task_count=sum(_task_key(task) not in all_bound
                                                       for task in original.get("device_tasks") or []),
                  missing_device_task_keys=missing_tasks)
    if not trusted:
        result.update(status=untrusted_status, source_status="UNAVAILABLE", source_task_status_counts={},
                      selection_reference_status="UNAVAILABLE", missing_event_indexes=[],
                      reason="Original clock/source case is untrusted; only recorded device-cycle task membership is retained")
        return result, set(), keys
    selected_links = [deepcopy(link) for link in original.get("links") or [] if _task_key(link) in keys]
    edges = [deepcopy(edge) for edge in original.get("drawable_edges") or []
             if edge.get("round_id") == round_id and _task_key(edge) in keys]
    task_edges = [deepcopy(edge) for edge in original.get("task_edges") or []
                  if _edge_in_round(edge, round_id) and _task_key(edge) in keys]
    for edge in task_edges:
        if "round_ids" in edge:
            edge["original_round_ids"] = edge["round_ids"]
            edge["round_ids"] = [round_id]
    observations = [deepcopy(item) for item in original.get("ambiguous_trace_device_observations") or []
                    if _observation_task_keys(item).intersection(keys)]
    trace_observations = [deepcopy(item) for item in original.get("trace_device_observations") or []
                          if _observation_task_keys(item).intersection(keys)]
    families = [deepcopy(item) for item in original.get("host_family_scopes") or [] if item.get("round_id") == round_id]
    ambiguity_keys = {key for item in observations + trace_observations
                      for key in item.get("candidate_task_keys") or []} - keys
    context_tasks = [deepcopy(task) for task in original.get("device_tasks") or [] if _task_key(task) in ambiguity_keys]
    linked = {_task_key(link) for link in selected_links}
    result.update(links=selected_links, drawable_edges=edges, task_edges=task_edges,
                  ambiguous_trace_device_observations=observations,
                  trace_device_observations=trace_observations, host_family_scopes=families,
                  ambiguity_context_device_tasks=context_tasks,
                  missing_ambiguity_candidate_task_keys=sorted(ambiguity_keys - {_task_key(task) for task in context_tasks}),
                  missing_device_source_task_keys=sorted(present - linked),
                  source_task_status_counts=dict(Counter(link.get("status", "UNKNOWN") for link in selected_links)))
    if "source_task_status_counts" in original:
        result["original_source_task_status_counts"] = deepcopy(original["source_task_status_counts"])
    return result, _refs(selected_links) | _refs(edges) | _refs(task_edges) | _refs(selected) | _refs(families), keys


def _task_reference_status(result, available, wanted):
    """Expose missing saved proof and never keep a confirmed partial path."""
    available_ids = set(available)
    missing = set(wanted) - available_ids
    incomplete_tasks = set(result.get("missing_device_task_keys") or [])
    binding_missing = {}
    for number, binding in enumerate(result.get("task_round_bindings") or []):
        absent = _closure(_refs(binding), available) - available_ids
        binding_missing[number] = absent
        if absent:
            incomplete_tasks.add(_task_key(binding))
            missing.update(absent)
    for link in result.get("links") or []:
        if _closure(_refs(link), available) - available_ids:
            incomplete_tasks.add(_task_key(link))
    for number, binding in enumerate(result.get("task_round_bindings") or []):
        if binding_missing[number] or _task_key(binding) in incomplete_tasks:
            if (_binding_relation(binding) == "EXPLICIT_SOURCE_OWNER"
                    or binding.get("runtime_submission_confirmed") is True):
                binding["original_relation"] = _binding_relation(binding)
                binding["original_runtime_submission_confirmed"] = binding.get("runtime_submission_confirmed")
                binding.update(relation="UNVERIFIED_SOURCE_OWNER", runtime_submission_confirmed=False,
                               line_style="dashed")
            binding["selection_evidence_status"] = "MISSING_SAVED_REFERENCES_OR_TASK"
            if binding_missing[number]:
                binding["missing_source_event_indexes"] = sorted(binding_missing[number])
    for link in result.get("links") or []:
        link_missing = _closure(_refs(link), available) - available_ids
        if link_missing or _task_key(link) in incomplete_tasks:
            incomplete_tasks.add(_task_key(link))
            link["missing_source_event_indexes"] = sorted(set(link.get("missing_source_event_indexes") or []) | link_missing)
            for association in link.get("host_associations") or []:
                if association.get("runtime_submission_confirmed"):
                    association["original_runtime_submission_confirmed"] = True
                association.update(runtime_submission_confirmed=False, line_style="dashed")
                association["selection_evidence_status"] = "MISSING_SAVED_REFERENCES_OR_TASK"
        missing.update(link_missing)
    for edge in [*(result.get("drawable_edges") or []), *(result.get("task_edges") or [])]:
        if _closure(_refs(edge), available) - available_ids or _task_key(edge) in incomplete_tasks:
            if edge.get("runtime_submission_confirmed"):
                edge["original_runtime_submission_confirmed"] = True
            edge.update(runtime_submission_confirmed=False, line_style="dashed",
                        selection_evidence_status="MISSING_SAVED_REFERENCES")
    result.update(missing_event_indexes=sorted(missing), selection_reference_status="PARTIAL" if (
        missing or result.get("missing_device_task_keys") or result.get("missing_device_source_task_keys")
        or result.get("missing_ambiguity_candidate_task_keys")) else "COMPLETE")
    return missing


def _deduplicate_child_fields(row, host_lookup):
    removed = 0
    def walk(value):
        nonlocal removed
        if isinstance(value, dict):
            number = _index(value.get("event_index")) if "event_index" in value else None
            event = host_lookup.get(number)
            if event is not None:
                for key in _DUPLICATE_REF_FIELDS:
                    if key in value and key in event and _identity_equal(value[key], event[key], key):
                        del value[key]
                        removed += 1
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)
    walk((row.get("host_pairing") or {}).get("child_layer") or {})
    row.pop("host_children", None)
    return removed


def _device_rows(value):
    found = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "row_index":
                number = _index(item)
                if number is not None:
                    found.add(number)
            elif key in ("row_indices", "device_row_indexes"):
                found.update(_index(number) for number in item or [] if number is not None)
            else:
                found.update(_device_rows(item))
    elif isinstance(value, list):
        for item in value:
            found.update(_device_rows(item))
    return found


def _case_evidence(case, core, host_analysis, attach_host, rank, source_case_index):
    output = deepcopy({key: value for key, value in case.items()
                       if key not in ("rounds", "host_events", "anchor_links", "device_anchors", "device_task_links")
                       and key not in _HISTORY_FIELDS})
    for key, renamed in _HISTORY_FIELDS.items():
        if key in case:
            output[renamed] = deepcopy(case[key])
    output.update(source_case_index=source_case_index, source_case_status=case.get("status"),
                  source_detail_status=case.get("detail_status"), source_round_count=len(case.get("rounds") or []),
                  rounds=[], host_events=[], device_anchors=[], representative_status="UNAVAILABLE",
                  selected_round_count=0, selected_complete_host_rounds=0, selected_complete_rounds=0,
                  source_link_round_ids=[], source_link_scope="Only saved source evidence for the selected representative")
    if case.get("rank", rank) != rank:
        raise ValueError("Saved case rank differs from requested fixed rank")
    selected, reason = _selected_round(case)
    output.update(selection_reason=reason, selected_round_id=None if selected is None else selected["round_id"],
                  suggested_round_id=None if selected is None else selected["round_id"],
                  omitted_round_count=len(case.get("rounds") or []) - (1 if selected is not None else 0))
    if selected is None:
        return output
    output["selected_round_count"] = 1
    row = _times(deepcopy(selected))
    clock_ok = (case.get("trace") or {}).get("status") == "ALIGNED_X_EVENTS"
    trusted = clock_ok and case.get("status") != "ERROR"
    if not trusted:
        for key in ("host_pairing", "host_lanes", "host_children", "coordination_breakdown"):
            row.pop(key, None)
        row.update(host_lanes={}, host_pairing={"status": "UNAVAILABLE", "reason":
            "Original clock alignment failed or source case was ERROR; no Host association is reconstructed"})
        output.update(rounds=[row], representative_status="DEVICE_ONLY",
                      device_anchors=[deepcopy(task) for task in case.get("device_anchors", [])
                                      if _index(task.get("row_index")) in _device_rows(row)],
                      anchor_links={"status": "UNAVAILABLE", "reason": "Untrusted source case; no device-source arrows copied",
                                    "anchor_bindings": [], "links": [], "drawable_edges": [], "source_events": [], "device_tasks": []})
        if isinstance(case.get("device_task_links"), dict):
            tasks, _, _ = _task_link_subset(case["device_task_links"], row["round_id"], trusted=False,
                                           untrusted_status="CLOCK_UNALIGNED" if not clock_ok else "SOURCE_UNTRUSTED")
            output["device_task_links"] = tasks
        return output
    host_events, host_lookup = _events(case.get("host_events") or [])
    # The complete saved set is necessary here: filtering first could erase a
    # conflicting worker lane or parent candidate and manufacture uniqueness.
    pairs = host_analysis.pair_rounds(core, [row], host_events)
    row = attach_host([row], pairs, host_events)[0]
    links, source_seeds, linked_rows = _link_subset(case.get("anchor_links"), row["round_id"])
    task_original = case.get("device_task_links")
    if isinstance(task_original, dict):
        trace_sha = (case.get("trace") or {}).get("sha256")
        if trace_sha and task_original.get("trace_sha256") and trace_sha != task_original["trace_sha256"]:
            raise ValueError("Device task source evidence belongs to a different trace SHA256")
    task_clock_ok = not isinstance(task_original, dict) or task_original.get("clock_alignment_status", "ALIGNED_X_EVENTS") == "ALIGNED_X_EVENTS"
    tasks, task_seeds, _ = _task_link_subset(task_original, row["round_id"], trusted=task_clock_ok)
    source_events, source_lookup = _events((case.get("anchor_links") or {}).get("source_events") or [])
    available = _available(host_lookup, source_lookup)
    task_events, task_lookup = _events((task_original or {}).get("source_events") or [] if task_clock_ok else [])
    available = _available(available, task_lookup)
    source_wanted = _closure(source_seeds, available)
    task_wanted = _closure(task_seeds, available)
    anchor_host_wanted = _closure(_round_refs(row) | source_seeds, available)
    host_wanted = _closure(_round_refs(row) | source_seeds | task_seeds, available)
    kept_host = [deepcopy(event) for event in host_events if event["event_index"] in host_wanted]
    # Host proofs can have a recorded ancestor only in the raw source table.
    # It must survive even when it was not also a selected device-link source.
    needed_source = source_wanted | (anchor_host_wanted - host_lookup.keys())
    kept_source = [deepcopy(available[number]) for number in sorted(needed_source) if number in available]
    links["source_events"] = kept_source
    links["missing_host_event_indexes"] = sorted(
        _index(number) for number in (case.get("anchor_links") or {}).get("missing_host_event_indexes", [])
        if _index(number) in host_wanted)
    links["selected_round_id"] = row["round_id"]
    task_missing = set()
    if tasks is not None and task_clock_ok:
        tasks["source_events"] = [deepcopy(available[number]) for number in sorted(task_wanted) if number in available]
        task_missing = _task_reference_status(tasks, available, task_wanted)
        output["device_task_links"] = tasks
    elif tasks is not None:
        output["device_task_links"] = tasks
    removed = _deduplicate_child_fields(row, host_lookup)
    selected_rows = linked_rows | _device_rows(row)
    exported_ids = {event["event_index"] for event in kept_host + kept_source}
    task_exported_ids = {event["event_index"] for event in (tasks or {}).get("source_events", [])}
    complete_host = set(HOST_LANES).issubset(row["host_lanes"])
    complete_round = row["host_pairing"].get("status") == "MATCHED" and complete_host and row.get("stage_status") == "MATCHED"
    output.update(rounds=[row], host_events=kept_host, anchor_links=links,
                  device_anchors=[deepcopy(task) for task in case.get("device_anchors", [])
                                  if _index(task.get("row_index")) in selected_rows],
                  missing_event_indexes=sorted(((host_wanted | source_wanted | task_wanted)
                                               - (exported_ids | task_exported_ids)) | task_missing),
                  source_link_round_ids=[row["round_id"]] if links.get("anchor_bindings") else [],
                  selected_complete_host_rounds=int(complete_host), selected_complete_rounds=int(complete_round),
                  representative_status="READY" if complete_round else "PARTIAL",
                  deduplicated_child_fields=removed,
                  selection_counts={"source_host_events": len(host_events), "selected_host_events": len(kept_host),
                                    "source_source_events": len(source_events), "selected_source_events": len(kept_source),
                                    "source_task_source_events": len(task_events),
                                    "selected_task_source_events": len(task_exported_ids)})
    return output


def build_ppt_evidence(payload, core, host_analysis, attach_host, source_info, rank="dp4", progress=None):
    """Build one representative per case from the already-saved full JSON.

    The caller handles file reading, provenance hash, atomic writes, exact wire
    serialization, packaging and output/input protection.  This function never
    reads an original trace or kernel CSV and never mutates ``payload``.
    """
    if not isinstance(payload, dict) or payload.get("schema") not in ("u_swimlane_v2", "u_swimlane_v3"):
        raise ValueError("Provide an existing u_swimlane_v2/v3 full JSON export")
    if payload.get("rank") != rank:
        raise ValueError("Saved export rank differs from requested fixed rank")
    if not isinstance(payload.get("cases"), list):
        raise ValueError("Saved swimlane export has no cases list")
    result = {"schema": SCHEMA, "source_schema": payload["schema"], "profile_version": payload.get("profile_version"),
              "rank": rank, "source_json_sha256": source_info.get("source_json_sha256", source_info.get("sha256")),
              "source_json_bytes": source_info.get("source_json_bytes", source_info.get("bytes", source_info.get("size_bytes"))),
              "source_info": deepcopy(source_info), "planned_cases": len(payload["cases"]), "cases": [],
              "all_cases_retained": True, "selected_rounds_per_case": 1, "intervals_clipped": False,
              "byte_budget_truncation": False,
              "diagram_contract": {"host_lane_order": ["outer_propose", "preparation_dp_aggregate", "runner_inputs", "runner_attn", "state_attn"],
                  "host_color_by": "function_kind", "host_labels": "function name and wall time",
                  "show_hc_overlap_labels": False, "children_depth_to_display": 1,
                  "confirmed_device_edge": "solid only when the original recorded evidence confirms submission; "
                                           "U-boundary arrows additionally require a unique boundary",
                  "inferred_or_source_only_edge": "dashed", "unknown": "Leave missing lanes/intervals/arrows blank; never borrow another rank or round",
                  "cpu_wait": "Do not invent a directly submitted device kernel"},
              "scope": "PPT representative-round evidence, not the full run or a full-history statistical summary. "
                       "Original full export stays unchanged; every recorded selected-round child and source proof is retained, "
                       "including tasks internal to stages and tasks that execute after their owning Host scope returns. "
                       "Source confidence is copied and never strengthened by this selection.",
              "representation": {"event_namespace": "case + rank + original trace SHA256 + event_index",
                  "host_children": "Reconstruct from host_pairing.child_layer.by_role refs and this case's event tables; "
                                   "relative times are exact event.start/end_ns minus round.origin_ns",
                  "deduplicated_ref_fields": list(_DUPLICATE_REF_FIELDS),
                  "ref_resolution": "When a child ref field is absent, read it from host_events[event_index]; "
                                    "only equal duplicate fields are removed. Proof/ownership fields stay on the ref. "
                                    "Parent/source references resolve in the union of the case's host_events, "
                                    "anchor_links.source_events and device_task_links.source_events.",
                  "anchor_binding_indexes": "Remapped to the selected anchor_bindings; source_binding_indexes records old indexes",
                  "device_task_membership": "Selected only from recorded task_round_bindings, independently of U anchor bindings; "
                                            "task_key is the identity. Timestamp overlap does not prove submission ownership.",
                  "status": "case status is original source status; representative_status separately describes the rebuilt selected round"}}
    for number, case in enumerate(payload["cases"]):
        if progress:
            progress("Selecting saved PPT representative " + str(number+1) + "/" + str(len(payload["cases"]))
                     + ": " + str(case.get("case", "invalid") if isinstance(case, dict) else "invalid"))
        try:
            if not isinstance(case, dict):
                raise ValueError("Saved case must be an object")
            selected = _case_evidence(case, core, host_analysis, attach_host, rank, number)
        except Exception as error:
            selected = {"case": case.get("case", "invalid") if isinstance(case, dict) else "invalid",
                        "rank": case.get("rank", rank) if isinstance(case, dict) else rank,
                        "source_case_index": number, "source_case_status": case.get("status") if isinstance(case, dict) else None,
                        "status": "ERROR", "representative_status": "ERROR", "rounds": [], "host_events": [],
                        "error": case.get("error") if isinstance(case, dict) else None,
                        "trace": deepcopy(case.get("trace")) if isinstance(case, dict) else None,
                        "share_error": type(error).__name__ + ": " + str(error)}
        result["cases"].append(_times(selected, to_wire=True))
    result["completed_cases"] = len(result["cases"])
    result["representative_status_counts"] = dict(Counter(case["representative_status"] for case in result["cases"]))
    return result
