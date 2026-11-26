"""Exact reference equivalence and bounded indexing work for large task shares.

The small legacy oracles deliberately rebuild the old index. They validate
outputs, while operation counts (not machine-dependent time limits) prevent the
full-event-table-per-task regression. No evidence is sampled or discarded.
"""

from collections import defaultdict, deque
from copy import deepcopy
import random
import unittest

import ep128_swimlane_share as share


def legacy_closure(seeds, events):
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
        more = share._refs(event)
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


def legacy_status(result, available, wanted):
    ids = set(available)
    missing = set(wanted) - ids
    incomplete = set(result.get("missing_device_task_keys") or [])
    binding_missing = {}
    for number, binding in enumerate(result.get("task_round_bindings") or []):
        absent = legacy_closure(share._refs(binding), available) - ids
        binding_missing[number] = absent
        if absent:
            incomplete.add(share._task_key(binding))
            missing.update(absent)
    for link in result.get("links") or []:
        if legacy_closure(share._refs(link), available) - ids:
            incomplete.add(share._task_key(link))
    for number, binding in enumerate(result.get("task_round_bindings") or []):
        if binding_missing[number] or share._task_key(binding) in incomplete:
            if (share._binding_relation(binding) == "EXPLICIT_SOURCE_OWNER"
                    or binding.get("runtime_submission_confirmed") is True):
                binding["original_relation"] = share._binding_relation(binding)
                binding["original_runtime_submission_confirmed"] = binding.get("runtime_submission_confirmed")
                binding.update(relation="UNVERIFIED_SOURCE_OWNER", runtime_submission_confirmed=False,
                               line_style="dashed")
            binding["selection_evidence_status"] = "MISSING_SAVED_REFERENCES_OR_TASK"
            if binding_missing[number]:
                binding["missing_source_event_indexes"] = sorted(binding_missing[number])
    for link in result.get("links") or []:
        absent = legacy_closure(share._refs(link), available) - ids
        if absent or share._task_key(link) in incomplete:
            incomplete.add(share._task_key(link))
            link["missing_source_event_indexes"] = sorted(set(link.get("missing_source_event_indexes") or []) | absent)
            for association in link.get("host_associations") or []:
                if association.get("runtime_submission_confirmed"):
                    association["original_runtime_submission_confirmed"] = True
                association.update(runtime_submission_confirmed=False, line_style="dashed")
                association["selection_evidence_status"] = "MISSING_SAVED_REFERENCES_OR_TASK"
        missing.update(absent)
    for edge in [*(result.get("drawable_edges") or []), *(result.get("task_edges") or [])]:
        if legacy_closure(share._refs(edge), available) - ids or share._task_key(edge) in incomplete:
            if edge.get("runtime_submission_confirmed"):
                edge["original_runtime_submission_confirmed"] = True
            edge.update(runtime_submission_confirmed=False, line_style="dashed",
                        selection_evidence_status="MISSING_SAVED_REFERENCES")
    result.update(missing_event_indexes=sorted(missing), selection_reference_status="PARTIAL" if (
        missing or result.get("missing_device_task_keys") or result.get("missing_device_source_task_keys")
        or result.get("missing_ambiguity_candidate_task_keys")) else "COMPLETE")
    return missing


class CountingEvents(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.values_reads = 0

    def values(self):
        self.values_reads += 1
        return super().values()


def event_graph(seed, count=80):
    rng = random.Random(seed)
    events = {}
    for number in range(count):
        events[number] = {
            "event_index": number, "name": "event-" + str(number),
            "pid": rng.randrange(2), "tid": rng.randrange(2),
            "start_ns": str(rng.randrange(10)), "end_ns": str(20 + rng.randrange(10)),
            "python_id": str(rng.randrange(15)),
            "python_parent_id": str(rng.randrange(15)),
            "parent_event_index": rng.randrange(count + 8),
            "flow_event_indexes": [rng.randrange(count + 8)],
        }
    return events


def task_result(count=30):
    return {
        "missing_device_task_keys": ["csv:4"],
        "links": [{"task_key": "csv:" + str(i), "source_event_indexes": [i],
                   "host_associations": [{"source_event_index": i, "runtime_submission_confirmed": True}]}
                  for i in range(count)],
        "task_round_bindings": [{"task_key": "csv:" + str(i), "source_event_index": i + 1,
                                 "relation": "EXPLICIT_SOURCE_OWNER", "runtime_submission_confirmed": True}
                                for i in range(count)],
        "task_edges": [{"task_key": "csv:" + str(i), "source_event_index": i,
                        "runtime_submission_confirmed": True} for i in range(count)],
    }


class ShareReferencePerformanceTests(unittest.TestCase):
    def test_closure_and_missing_are_exact_with_cycles_and_ambiguous_parent_ids(self):
        for graph_seed in range(6):
            events = event_graph(graph_seed)
            before = deepcopy(events)
            references = share._EventReferences(events)
            seeds = set(range(88))
            missing = references.missing_lookup(seeds)
            for number in seeds:
                expected = legacy_closure({number}, events)
                self.assertEqual(references.closure({number}), expected)
                self.assertEqual(missing.get(number, set()), expected - events.keys())
            self.assertEqual(events, before)

    def test_all_binding_association_edge_downgrades_equal_the_previous_algorithm(self):
        for graph_seed in range(5):
            events = event_graph(graph_seed)
            old, new = task_result(), task_result()
            wanted = legacy_closure(set(range(31)), events)
            self.assertEqual(legacy_status(old, events, wanted), share._task_reference_status(new, events, wanted))
            self.assertEqual(new, old)

    def test_large_task_set_builds_full_parent_index_once_and_keeps_every_record(self):
        count = 2000
        available = CountingEvents({i: {"event_index": i, "python_id": str(i), "pid": 1, "tid": 1,
            "python_parent_id": "0", "start_ns": 0, "end_ns": 100} for i in range(10000)})
        result = task_result(count)
        original_sizes = {key: len(result[key]) for key in ("links", "task_round_bindings", "task_edges")}
        share._task_reference_status(result, available, set(range(count + 1)))
        self.assertEqual(available.values_reads, 1)
        self.assertEqual(result["missing_event_indexes"], [])
        self.assertEqual({key: len(result[key]) for key in original_sizes}, original_sizes)

    def test_deep_complete_reference_chain_does_not_store_all_transitive_closures(self):
        count = 3000
        events = {i: {"event_index": i, "parent_event_index": i + 1} for i in range(count)}
        events[count - 1].pop("parent_event_index")
        references = share._EventReferences(events)
        self.assertEqual(references.missing_lookup(range(count)), {})
        self.assertEqual(len(references.neighbors), count)
        self.assertLessEqual(sum(len(values) for values in references.neighbors.values()), 2 * count)


if __name__ == "__main__":
    unittest.main()
