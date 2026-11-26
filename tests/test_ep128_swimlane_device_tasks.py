"""Full device catalogs keep interior kernels, DMA, and delayed submissions."""

from copy import deepcopy
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import profile_ep128 as core
import ep128_swimlane_links as links
from ep128_swimlane_trace import collect_host_events
import test_ep128_swimlane_links as base


EPOCH = base.EPOCH


def saved(directory, mode="linked", explicit=True):
    capture, rounds, hosts, trace, metadata = base.fixture(directory, mode=mode, explicit=explicit)
    root = hosts[0]["event_index"]
    rounds[0].update(host_pairing={"functions": {"state_attn": root}, "context": {}},
                     host_lanes={"state_attn": {"event_index": root, "start_ns": EPOCH, "end_ns": EPOCH+90000}})
    return capture, rounds, hosts, trace, metadata


def add_task(values, number, name, start, end, connection=base.CONNECTION, csv=True, **kwargs):
    capture, _, _, trace, metadata = values
    task = base.task(number, name, start, end, **kwargs)
    if csv:
        capture["tasks"].append(task)
    events = json.loads(trace.read_text(encoding="utf-8"))
    args = {"Task Type": task.task_type, "Physic Stream Id": task.stream_id, "Task Id": task.task_id,
            "Model Id": task.model_id}
    if connection is not None:
        args["connection_id"] = connection
    events.append(base.xevent(name, start, end, 900, 1, "kernel", args))
    index = len(events)
    trace.write_text(json.dumps(events), encoding="utf-8")
    metadata["sha256"] = hashlib.sha256(trace.read_bytes()).hexdigest()
    return task, index


def resolve(values, directory, with_trace_devices=True, progress=None):
    capture, rounds, hosts, trace, metadata = values
    devices = ()
    if with_trace_devices:
        scan = collect_host_events(core, trace, 0, metadata["sha256"], retain_device_events=True,
                                   device_trace_metadata=metadata)
        devices = scan["trace_device_events"]
    return links.export_device_task_links(core, capture, rounds, hosts, trace, metadata, 0,
                                           directory/"tasks", progress, trace_device_events=devices)


def by_key(result, key):
    return next(row for row in result["links"] if row["task_key"] == key)


class DeviceTaskExportTests(unittest.TestCase):
    def test_interior_metadata_kernel_is_retained_even_when_not_a_stage_anchor(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp); values = saved(directory)
            add_task(values, 2, "MixedQuantSparseFlashMlaMetadata", 230000, 235000)
            stage = values[1][0]["u_windows"][0]["stages"][0]
            stage["end_ns"] = str(EPOCH+245000)
            stage["end_anchor"]["timestamp_ns"] = str(EPOCH+245000)
            original = deepcopy(values[1]), values[3].read_bytes(), Path(values[0]["path"]).read_bytes()
            result = resolve(values, directory)
            self.assertEqual({row["task_key"] for row in result["device_tasks"]}, {"csv:1", "csv:2"})
            self.assertEqual(by_key(result, "csv:2")["status"], "LINKED")
            self.assertNotIn(2, {row["row_index"] for row in result["anchor_links"]["device_tasks"]})
            self.assertTrue(any(b["task_key"] == "csv:2" and b["u_stage_overlaps"]
                                for b in result["task_round_bindings"] if b["relation"] == "DEVICE_CYCLE_OVERLAP"))
            self.assertEqual(values[1], original[0]); self.assertEqual(values[3].read_bytes(), original[1])
            self.assertEqual(Path(values[0]["path"]).read_bytes(), original[2])

    def test_async_source_owner_is_retained_after_Host_and_cycle_end_without_temporal_borrowing(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp); values = saved(directory)
            add_task(values, 3, "DelayedMetadataKernel", 500000, 510000)
            result = resolve(values, directory)
            row = by_key(result, "csv:3")
            self.assertEqual(row["status"], "LINKED")
            own = [b for b in result["task_round_bindings"] if b["task_key"] == "csv:3"]
            self.assertTrue(own)
            self.assertTrue(all(b["relation"] == "EXPLICIT_SOURCE_OWNER" for b in own))
            self.assertTrue(all(b["runtime_submission_confirmed"] for b in own))
            self.assertEqual(row["task_round_ids"], ["hc-1"])

    def test_unlinked_orphan_outside_selected_round_is_not_deleted_or_assigned(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp); values = saved(directory)
            add_task(values, 4, "OrphanKernel", 700000, 710000, connection=None)
            result = resolve(values, directory)
            self.assertEqual(by_key(result, "csv:4")["status"], "UNLINKED")
            self.assertEqual(by_key(result, "csv:4")["task_round_ids"], [])
            self.assertFalse(any(b["task_key"] == "csv:4" for b in result["task_round_bindings"]))

    def test_trace_only_DMA_has_namespaced_identity_and_exact_unclipped_span(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp); values = saved(directory)
            _, number = add_task(values, 8, "MEMCPY_ASYNC", 180000, 330000, csv=False)
            result = resolve(values, directory)
            record = next(t for t in result["device_tasks"] if t["task_key"] == "trace:"+str(number))
            self.assertEqual(record["source_kind"], "TRACE_ONLY")
            self.assertIsNone(record["csv_row_index"])
            self.assertEqual(record["row_index"], -number)
            self.assertEqual(record["start_ns"], EPOCH+180000)
            self.assertEqual(record["end_ns"], EPOCH+330000)
            self.assertEqual(record["row_indices"], [])
            self.assertEqual(by_key(result, record["task_key"])["status"], "LINKED")
            overlap = next(b for b in result["task_round_bindings"] if b["task_key"] == record["task_key"]
                           and b["relation"] == "DEVICE_CYCLE_OVERLAP")
            self.assertEqual(overlap["overlap_end_ns"], EPOCH+300000)
            self.assertEqual(record["end_ns"], EPOCH+330000)  # not clipped to its context

    def test_matching_CSV_trace_tasks_are_deduplicated_but_original_observations_survive(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp); result = resolve(saved(directory), directory)
            self.assertEqual(len(result["device_tasks"]), 1)
            record, = result["device_tasks"]
            self.assertTrue(record["trace_event_indexes"])
            observation, = result["trace_device_observations"]
            self.assertEqual(observation["represented_by_task_key"], record["task_key"])

    def test_zero_duration_trace_event_is_kept_without_inventing_device_source(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp); values = saved(directory)
            _, number = add_task(values, 8, "ZeroMetadataMarker", 215000, 215000, csv=False)
            result = resolve(values, directory)
            row = by_key(result, "trace:"+str(number))
            self.assertEqual(row["status"], "NO_TRACE_DEVICE")
            self.assertFalse(row["host_associations"])
            self.assertEqual(result["trace_zero_duration_observations"], 1)

    def test_shared_graph_and_ambiguous_source_statuses_remain_visible(self):
        for mode, status in (("graph", "GRAPH_SHARED_NO_ORIGIN"), ("ambiguous", "AMBIGUOUS"),
                             ("wrong_task_id", "NO_TRACE_DEVICE")):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temp:
                directory = Path(temp); result = resolve(saved(directory, mode), directory)
                row = by_key(result, "csv:1")
                self.assertEqual(row["status"], status)
                self.assertFalse(any(e["runtime_submission_confirmed"] for e in result["task_edges"] if e["task_key"] == "csv:1"))

    def test_time_containment_does_not_become_verified_Host_or_round_owner(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp); result = resolve(saved(directory, explicit=False), directory)
            self.assertEqual(by_key(result, "csv:1")["status"], "LINKED")
            self.assertTrue(result["task_edges"])
            self.assertFalse(any(e["runtime_submission_confirmed"] for e in result["task_edges"]))
            self.assertTrue(any(b["relation"] == "HOST_SOURCE_TIME_CONTAINMENT_ONLY" for b in result["task_round_bindings"]))

    def test_duplicate_CSV_row_identity_is_not_silently_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp); values = saved(directory)
            values[0]["tasks"].append(deepcopy(values[0]["tasks"][0]))
            result = resolve(values, directory)
            self.assertEqual(len(result["device_tasks"]), 2)
            self.assertEqual(len({t["task_key"] for t in result["device_tasks"]}), 2)
            self.assertTrue(all(t["catalog_identity_status"] == "AMBIGUOUS_CSV_ROW" for t in result["device_tasks"]))
            self.assertTrue(all(row["status"] == "AMBIGUOUS" for row in result["links"]))

    def test_ambiguous_trace_to_two_CSV_identities_does_not_create_a_third_task_or_confirm_links(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp); values = saved(directory)
            values[0]["tasks"][0].task_id = None
            other = deepcopy(values[0]["tasks"][0]); other.row_index = 2; other.row_indices = [2]
            values[0]["tasks"].append(other)
            result = resolve(values, directory)
            self.assertEqual(len(result["device_tasks"]), 2)
            observation, = result["ambiguous_trace_device_observations"]
            self.assertEqual(set(observation["candidate_task_keys"]), {"csv:1", "csv:2"})
            self.assertTrue(all(row["status"] == "AMBIGUOUS" for row in result["links"]))
            self.assertFalse(any(e["runtime_submission_confirmed"] for e in result["task_edges"]))
            self.assertNotEqual(result["source_status"], "COMPLETE")

    def test_clock_failure_catalog_has_no_Host_coordinates_or_submission_edges(self):
        with tempfile.TemporaryDirectory() as temp:
            capture, rounds, *_ = saved(Path(temp))
            result = links.export_device_catalog(capture, rounds)
            self.assertEqual(len(result["device_tasks"]), 1)
            self.assertEqual(result["status"], "DEVICE_ONLY")
            self.assertTrue(all(b["relation"] == "DEVICE_CYCLE_OVERLAP" for b in result["task_round_bindings"]))
            self.assertFalse(result["source_events"] or result["links"] or result["task_edges"])
            self.assertNotIn("host_family_scopes", result)

    def test_aligned_snapshot_retains_trace_only_tasks_before_source_resolution(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp); values = saved(directory)
            _, number = add_task(values, 8, "MEMCPY_ASYNC", 180000, 330000, csv=False)
            scan = collect_host_events(core, values[3], 0, values[4]["sha256"], retain_device_events=True,
                                       device_trace_metadata=values[4])
            events = scan["trace_device_events"]
            result = links.export_device_catalog(values[0], values[1], trace_device_events=events,
                                                   trace_meta=values[4], offset=0)
            self.assertIn("trace:"+str(number), {t["task_key"] for t in result["device_tasks"]})
            self.assertFalse(result["links"] or result["source_events"] or result["task_edges"])
            self.assertTrue(all(b["relation"] == "DEVICE_CYCLE_OVERLAP" for b in result["task_round_bindings"]))
            self.assertEqual(result["trace_sha256"], values[4]["sha256"])
            for metadata, offset in ((None, 0), ({**values[4], "status":"CLOCK_MISMATCH"},0), (values[4],1)):
                with self.assertRaises(ValueError):
                    links.export_device_catalog(values[0], values[1], trace_device_events=events,
                                                 trace_meta=metadata, offset=offset)

    def test_source_ancestors_not_in_old_Host_whitelist_are_exported_and_associated(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp); values = saved(directory)
            # Keep only the root in recognized Host calls; helper.py remains in
            # raw core ancestors and must get its own explicit task edge.
            values[2].pop()
            result = resolve(values, directory)
            source_names = {e["name"] for e in result["source_events"]}
            self.assertIn("helper.py(2): launch", source_names)
            edges = [e for e in result["task_edges"] if e["host_name"] == "helper.py(2): launch"]
            self.assertTrue(edges)
            self.assertTrue(all(e["runtime_submission_confirmed"] for e in edges))

    def test_all_task_resolution_is_once_and_anchor_projection_reuses_it(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp); values = saved(directory)
            add_task(values, 2, "MetadataKernel", 230000, 235000)
            original = core.residual_host_rows
            with patch.object(core, "residual_host_rows", wraps=original) as resolver:
                result = resolve(values, directory)
            self.assertEqual(resolver.call_count, 1)
            self.assertEqual(len(resolver.call_args.args[1]["tasks"]), 2)
            self.assertEqual(result["scan_passes"], 2)
            self.assertEqual(result["anchor_links"]["scan_passes"], 0)
            self.assertEqual(result["anchor_links"]["resolution_reused_from"], "device_task_links")

    def test_original_submission_and_queue_flow_endpoints_keep_exact_provenance(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp); result = resolve(saved(directory), directory)
            link = by_key(result, "csv:1")
            raw = {e["event_index"]:e for e in result["source_events"]}
            endpoints = [raw[i] for i in link["flow_event_indexes"]]
            self.assertEqual({(e["cat"], e["ph"]) for e in endpoints}, {
                ("HostToDevice", "s"), ("HostToDevice", "f"),
                ("async_task_queue", "s"), ("async_task_queue", "f")})
            self.assertTrue(set(link["flow_event_indexes"]).issubset(link["source_event_indexes"]))
            self.assertEqual(next(e["start_ns"] for e in endpoints if e["cat"] == "HostToDevice" and e["ph"] == "s"), EPOCH+42000)
            self.assertTrue(all(e["start_ns"] == e["end_ns"] for e in endpoints))
            json.dumps(result, allow_nan=False)  # includes nanosecond/large-ID native serialization

    def test_reused_graph_flow_keeps_all_finish_endpoints_without_inventing_a_start(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp); result = resolve(saved(directory, "graph"), directory)
            link = by_key(result, "csv:1")
            raw = {e["event_index"]:e for e in result["source_events"]}
            endpoints = [raw[i] for i in link["flow_event_indexes"]]
            self.assertEqual(link["status"], "GRAPH_SHARED_NO_ORIGIN")
            self.assertEqual(len(endpoints), 2)
            self.assertTrue(all(e["ph"] == "f" for e in endpoints))
            self.assertFalse(link["host_associations"])

    def test_queue_flow_points_at_exact_scope_ends_are_retained_without_upgrading_proof(self):
        for explicit in (True, False):
            with self.subTest(explicit=explicit), tempfile.TemporaryDirectory() as temp:
                directory = Path(temp); values = saved(directory, explicit=explicit)
                trace, metadata = values[3], values[4]
                events = json.loads(trace.read_text(encoding="utf-8"))
                for event in events:
                    if event.get("cat") == "async_task_queue":
                        stamp = 32000 if event["ph"] == "s" else 50000
                        event["ts"] = str(Decimal(EPOCH+stamp)/1000)
                trace.write_text(json.dumps(events), encoding="utf-8")
                metadata["sha256"] = hashlib.sha256(trace.read_bytes()).hexdigest()
                result = resolve(values, directory)
                link = by_key(result, "csv:1")
                raw = {e["event_index"]:e for e in result["source_events"]}
                queue = [raw[i] for i in link["flow_event_indexes"] if raw[i]["cat"] == "async_task_queue"]
                self.assertEqual({(e["ph"], e["start_ns"]) for e in queue}, {
                    ("s", EPOCH+32000), ("f", EPOCH+50000)})
                self.assertTrue(set(e["event_index"] for e in queue).issubset(link["source_event_indexes"]))
                self.assertEqual(link["status"], "LINKED")
                self.assertEqual(result["core_source_status"], "COMPLETE")
                self.assertEqual(any(e["runtime_submission_confirmed"] for e in result["task_edges"]), explicit)

    def test_each_Host_edge_binds_only_its_own_family_or_recorded_parent_path(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp); values = saved(directory)
            # The root and helper are explicitly related, so both edges can
            # belong to hc-1. An overlapping device cycle alone adds no owner.
            other = deepcopy(values[1][0]); other["round_id"] = "hc-2"
            other["host_pairing"] = {}; other["host_lanes"] = {}
            values[1].append(other)
            result = resolve(values, directory)
            self.assertEqual(by_key(result, "csv:1")["task_round_ids"], ["hc-1", "hc-2"])
            self.assertTrue(result["task_edges"])
            self.assertTrue(all(e["round_ids"] == ["hc-1"] for e in result["task_edges"]))

    def test_source_references_complete_and_required_missing_reference_prevents_confirmed_edges(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp); values = saved(directory)
            real = links._read_evidence_events
            def missing(*args, **kwargs):
                raw = real(*args, **kwargs)
                return {i:e for i,e in raw.items() if not e["name"].startswith("Node@")}
            with patch.object(links,"_read_evidence_events",side_effect=missing):
                result = resolve(values,directory)
            self.assertTrue(result["links"][0]["missing_source_event_indexes"])
            self.assertFalse(any(e["runtime_submission_confirmed"] for e in result["task_edges"]))

    def test_unaligned_or_inconsistent_clock_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp); values = saved(directory)
            for status, offset in (("CLOCK_MISMATCH",0),("ALIGNED_X_EVENTS",3)):
                with self.subTest(status=status,offset=offset):
                    metadata = {**values[4],"status":status}
                    with self.assertRaises(ValueError):
                        links.export_device_task_links(core,*values[:3],values[3],metadata,offset,directory/"invalid")

    def test_output_does_not_overwrite_trace_or_kernel(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp); values = saved(directory)
            # Same filename as the evidence destination, but it is a source.
            target = directory/"device-tasks-host-events.csv"
            target.write_text(values[3].read_text(),encoding="utf-8")
            metadata={**values[4],"sha256":hashlib.sha256(target.read_bytes()).hexdigest()}
            original=target.read_bytes()
            with self.assertRaisesRegex(ValueError,"overlaps"):
                links.export_device_task_links(core,*values[:3],target,metadata,0,directory)
            self.assertEqual(target.read_bytes(),original)


if __name__ == "__main__":
    unittest.main()
