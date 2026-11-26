"""Mutually exclusive wall slices of one saved Host function or aggregate.

This module does not measure CPU busy, infer a missing wait or modify original
inclusive call intervals. Explicit saved ancestry, rather than interval length
or a displayed stack depth, decides whether one recorded call contains another.
"""

from collections import Counter
from copy import deepcopy
import re


SCHEMA = "u_host_exclusive_partition_v1"
TRANSPARENT_KINDS = frozenset({
    "context_wrapper", "profiler_wrapper", "metadata_native_wrapper", "metadata_argument_check",
    "collective_wrapper", "dflash_propose", "build_attn_metadata", "dp_dispatch_patch", "dp_dispatch",
})
GENERIC_KINDS = frozenset({
    "host_copy", "native_workspace", "metadata_native_call", "stream_dependency", "stream_event_record",
})
SCOPE = (
    "The pieces form one continuous, non-overlapping partition of the parent wall interval. "
    "Only saved explicit parent/ancestor evidence assigns a semantic child; temporal containment alone does not. "
    "A uniquely deepest verified semantic call owns its slice, and its inclusive ancestors are not added again. "
    "A function's remaining wall excludes its selected descendants. Parent remainder is unsegmented wall: "
    "it may contain unrecorded calls, waits, scheduling or local work and is not pure CPU computation. "
    "Incomparable overlapping verified calls and crossing-parent semantic candidates stay unknown. "
    "Original inclusive children and all diagnostics are preserved; no negative or cross-U coordinate is clipped."
)


def _ns(value):
    if type(value) is int:
        return value
    if isinstance(value, str) and re.fullmatch(r"[+-]?\d+", value):
        return int(value)
    raise ValueError("An exact integer nanosecond coordinate is required")


def _identifier(value):
    if type(value) is int:
        return value
    if isinstance(value, str) and value:
        return int(value) if re.fullmatch(r"[+-]?\d+", value) else value
    return None


def _name(event):
    return str(event.get("name") or event.get("label") or event.get("kind") or "未命名函数")


def _diagnostic(code, event, occurrence, **extra):
    return {"code": code, "event_index": event.get("event_index"), "name": _name(event),
            "occurrence_index": occurrence, "start_ns": event.get("start_ns"),
            "end_ns": event.get("end_ns"), **extra}


def _signature(record):
    return (record["event_index"], record["start"], record["end"], record["name"], record["kind"],
            tuple(record["ancestors"]), record["owner"], record["parent"], record["verified_flag"])


def _piece(left, right, parent_id, parent_name, owner=None, active=(), crossing=(), descendants=()):
    candidate_ids = sorted({e["event_index"] for e in (*active, *crossing) if e["event_index"] is not None}, key=str)
    candidate_occurrences = sorted({e["occurrence"] for e in (*active, *crossing)})
    if crossing or owner is False:
        label, kind, identity = "归属不明（重叠候选）", "ambiguous", None
        relation, reason = "unresolved_overlap", "CROSS_PARENT_BOUNDARY" if crossing else "INCOMPARABLE_VERIFIED_CALLS"
        ancestors, proof, remaining = [], [], False
        source_name = None
    elif owner is None:
        label, kind, identity = parent_name + " · 未细分wall", "remainder", parent_id
        relation, reason = "parent_remainder", "NO_VERIFIED_SEMANTIC_CHILD"
        ancestors, proof, remaining = [], [parent_id] if parent_id is not None else [], False
        source_name = parent_name
    else:
        remaining = owner["event_index"] in descendants
        label = owner["name"] + (" · 其余wall" if remaining else "")
        kind, identity, source_name = owner["kind"], owner["event_index"], owner["name"]
        relation = "verified_exclusive_remainder" if remaining else "verified_exclusive_leaf"
        reason = "DEEPEST_EXPLICIT_DESCENDANT" if len(active) > 1 else "ONLY_VERIFIED_ACTIVE_CALL"
        ancestors = list(owner["ancestors"])
        proof = [identity, *ancestors]
    return {"name": label, "label": label, "kind": kind, "event_index": identity,
            "source_name": source_name, "start_ns": str(left), "end_ns": str(right),
            "duration_ns": right-left, "duration_ms": (right-left)/1e6,
            "relation": relation, "selection_reason": reason, "selection_reasons": [reason],
            "ancestor_event_indexes": ancestors, "proof_event_indexes": proof,
            "candidate_event_indexes": candidate_ids, "candidate_occurrence_indexes": candidate_occurrences,
            "is_remainder_of_inclusive": remaining,
            "scope": "Unsegmented or unknown wall, not CPU busy" if kind in ("remainder", "ambiguous") else
                     "Exclusive wall assigned by saved ancestry; not pure CPU busy or pure communication latency"}


def _merge(pieces):
    result = []
    for piece in pieces:
        # Unknown alternatives are not merged across a candidate-set change.
        def key(item):
            return (item["kind"], item["event_index"], item["name"], item["relation"],
                    tuple(item["ancestor_event_indexes"]), item["is_remainder_of_inclusive"],
                    tuple(item["candidate_event_indexes"]) if item["kind"] == "ambiguous" else ())
        if result and result[-1]["end_ns"] == piece["start_ns"] and key(result[-1]) == key(piece):
            previous = result[-1]
            previous["end_ns"] = piece["end_ns"]
            previous["duration_ns"] += piece["duration_ns"]
            previous["duration_ms"] = previous["duration_ns"]/1e6
            for field in ("selection_reasons", "proof_event_indexes", "candidate_event_indexes", "candidate_occurrence_indexes"):
                previous[field] = list(dict.fromkeys([*previous[field], *piece[field]]))
            if len(previous["selection_reasons"]) > 1:
                previous["selection_reason"] = "SAME_OWNER_ACROSS_RECORDED_BOUNDARIES"
        else:
            result.append(piece)
    return result


def partition_host_lane(parent, children, role):
    """Partition one parent wall interval using saved explicit ancestry only.

    ``children`` is the complete original list of resolved child references,
    including folded wrappers and native/containment-only observations. The
    function returns copies and never changes these source intervals.
    """
    if not isinstance(parent, dict) or not isinstance(children, (list, tuple)):
        raise ValueError("A parent object and complete child-reference list are required")
    left, right = _ns(parent["start_ns"]), _ns(parent["end_ns"])
    if right < left:
        raise ValueError("Parent wall interval is reversed")
    parent_id = _identifier(parent.get("event_index", parent.get("start_event_index")))
    parent_name = _name(parent)
    records, diagnostics, boundaries = [], [], {left, right}
    for occurrence, raw in enumerate(children):
        if not isinstance(raw, dict):
            diagnostics.append({"code": "INVALID_CHILD_RECORD", "occurrence_index": occurrence})
            continue
        try:
            a, b = _ns(raw["start_ns"]), _ns(raw["end_ns"])
        except (ValueError, KeyError) as error:
            diagnostics.append(_diagnostic("INVALID_CHILD_COORDINATE", raw, occurrence, reason=str(error)))
            continue
        if b < a:
            diagnostics.append(_diagnostic("REVERSED_CHILD_INTERVAL", raw, occurrence))
            continue
        if a == b:
            diagnostics.append(_diagnostic("ZERO_DURATION_CHILD", raw, occurrence))
            continue
        if b > left and a < right:
            boundaries.update(stamp for stamp in (a, b) if left < stamp < right)
        identity = _identifier(raw.get("event_index"))
        ancestors = [_identifier(item) for item in raw.get("ancestor_event_indexes") or []]
        immediate = _identifier(raw.get("parent_event_index"))
        if not ancestors and immediate is not None:
            ancestors = [immediate]
        record = {"source": raw, "occurrence": occurrence, "start": a, "end": b,
                  "event_index": identity, "name": _name(raw), "kind": str(raw.get("kind") or "function"),
                  "ancestors": ancestors, "owner": _identifier(raw.get("owner_event_index")), "parent": immediate,
                  "verified_flag": raw.get("parent_chain_verified") is True,
                  "folded": raw.get("kind") in TRANSPARENT_KINDS | GENERIC_KINDS,
                  "crossing": a < left or b > right, "trusted": False}
        if record["crossing"]:
            diagnostics.append(_diagnostic("CROSS_PARENT_BOUNDARY", raw, occurrence,
                parent_start_ns=str(left), parent_end_ns=str(right),
                reason="Original child interval is retained; its intersection is not a valid whole child"))
        records.append(record)

    by_id = {}
    for record in records:
        by_id.setdefault(record["event_index"], []).append(record)
    conflicts = {identity for identity, group in by_id.items() if identity is not None
                 and len({_signature(record) for record in group}) > 1}
    for identity in sorted(conflicts, key=str):
        diagnostics.append({"code": "CONFLICTING_EVENT_ID", "event_index": identity,
                            "occurrence_indexes": [r["occurrence"] for r in by_id[identity]]})

    deduplicated = []
    seen = set()
    for record in records:
        signature = _signature(record)
        if signature in seen:
            diagnostics.append(_diagnostic("DUPLICATE_IDENTICAL_EVENT", record["source"], record["occurrence"]))
            continue
        seen.add(signature)
        deduplicated.append(record)
        source = record["source"]
        if record["folded"]:
            continue
        if record["event_index"] is None:
            diagnostics.append(_diagnostic("MISSING_EVENT_ID", source, record["occurrence"]))
            continue
        if record["event_index"] in conflicts:
            continue
        if not record["verified_flag"]:
            continue  # Kept in source_children; time containment assigns no owner.
        path = record["ancestors"]
        if (parent_id is None or not path or path[-1] != parent_id or None in path
                or record["event_index"] in path or len(path) != len(set(path))
                or record["owner"] not in (None, parent_id)
                or record["parent"] is not None and path[0] != record["parent"]):
            diagnostics.append(_diagnostic("INVALID_OR_MISSING_PARENT_PROOF", source, record["occurrence"],
                                           ancestor_event_indexes=path, expected_parent_event_index=parent_id))
            continue
        invalid_ancestors = []
        inner = record
        for position, number in enumerate(path):
            if number == parent_id:
                continue
            candidates = by_id.get(number, [])
            expected_suffix = path[position+1:]
            matching = [r for r in candidates if r["start"] <= inner["start"] and r["end"] >= inner["end"]
                        and r["ancestors"] == expected_suffix and r["verified_flag"]
                        and r["owner"] in (None, parent_id)
                        and (r["parent"] is None or expected_suffix and r["parent"] == expected_suffix[0])]
            if number in conflicts or not matching:
                invalid_ancestors.append(number)
            else:
                inner = matching[0]
        if invalid_ancestors:
            diagnostics.append(_diagnostic("MISSING_OR_CONTRADICTORY_ANCESTOR", source, record["occurrence"],
                                           missing_ancestor_event_indexes=invalid_ancestors))
            continue
        record["trusted"] = not record["crossing"]

    usable = [r for r in deduplicated if r["trusted"]]
    invalid_boundaries = [r for r in deduplicated if not r["folded"] and (r["crossing"] or r["event_index"] in conflicts)
                          and r["end"] > left and r["start"] < right]
    relationships, descendants = [], set()
    for child in usable:
        for ancestor in usable:
            if ancestor["event_index"] in child["ancestors"]:
                descendants.add(ancestor["event_index"])
                relationships.append({"parent_event_index": ancestor["event_index"], "child_event_index": child["event_index"],
                    "parent_name": ancestor["name"], "child_name": child["name"],
                    "relation": "EXPLICIT_SAVED_ANCESTOR", "ancestor_event_indexes": list(child["ancestors"]),
                    "scope": "Inclusive parent contains child; their durations are not additive"})
    atomic = []
    ordered = sorted(boundaries)
    for start, end in zip(ordered, ordered[1:]):
        if end <= start:
            continue
        active = [r for r in usable if r["start"] <= start and r["end"] >= end]
        crossing = [r for r in invalid_boundaries if r["start"] < end and r["end"] > start
                    and (r["verified_flag"] or r["event_index"] in conflicts or not active)]
        leaves = [r for r in active if not any(r["event_index"] in candidate["ancestors"] for candidate in active)]
        unique = len(leaves) == 1 and all(r is leaves[0] or r["event_index"] in leaves[0]["ancestors"] for r in active)
        owner = leaves[0] if unique else False if active else None
        atomic.append(_piece(start, end, parent_id, parent_name, owner, active, crossing, descendants))
    pieces = _merge(atomic)
    total = sum(piece["duration_ns"] for piece in pieces)
    closed = total == right-left and (not pieces and left == right or bool(pieces)
        and _ns(pieces[0]["start_ns"]) == left and _ns(pieces[-1]["end_ns"]) == right
        and all(a["end_ns"] == b["start_ns"] for a, b in zip(pieces, pieces[1:])))
    if not closed:
        raise ValueError("Exclusive Host pieces do not close to the original parent wall")
    observed_wait = any(r["kind"] == "cpu_wait" for r in usable)
    unknown = sum(piece["duration_ns"] for piece in pieces if piece["kind"] == "ambiguous")
    remainder = sum(piece["duration_ns"] for piece in pieces if piece["kind"] == "remainder")
    return {"schema": SCHEMA, "role": role, "parent": deepcopy(parent), "pieces": pieces,
            "start_ns": str(left), "end_ns": str(right), "parent_duration_ns": right-left,
            "total_duration_ns": total, "exclusive_duration_ns": total, "closed": closed,
            "status": "PARTIAL" if unknown or diagnostics else "PARTITIONED",
            "attribution_complete": not unknown and not remainder,
            "ambiguous_duration_ns": unknown, "parent_remainder_ns": remainder,
            "assigned_semantic_duration_ns": total-unknown-remainder,
            "wait_observed": observed_wait,
            "wait_exclusive_ns": sum(p["duration_ns"] for p in pieces if p["kind"] == "cpu_wait") if observed_wait else None,
            "source_children": deepcopy(list(children)), "source_child_count": len(children),
            "usable_semantic_child_count": len(usable), "folded_child_count": sum(r["folded"] for r in records),
            "containment_only_child_count": sum(not r["verified_flag"] for r in records),
            "source_boundary_ns": [str(value) for value in ordered], "atomic_piece_count": len(atomic),
            "inclusive_relationships": relationships, "diagnostics": diagnostics,
            "diagnostic_counts": dict(Counter(d["code"] for d in diagnostics)), "scope": SCOPE}
