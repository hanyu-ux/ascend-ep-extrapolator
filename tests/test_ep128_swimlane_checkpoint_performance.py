"""Completed case evidence is serialized once, with bounded merge memory."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from test_ep128_swimlane_probe import probe


class CaseCheckpointTests(unittest.TestCase):
    def evidence(self, count=3):
        return [{"case": "case-"+str(i), "rank": "dp4", "status": "READY", "rounds": [
            {"round_id": "hc-1", "start_ns": 1790000000000000000+i, "end_ns": 1790000000000000010+i}],
            "device_task_links": {"device_tasks": [{"task_key": "trace:1", "source_kind": "TRACE_ONLY",
                "start_ns": 1790000000000000001+i, "end_ns": 1790000000000000002+i}],
                "source_events": [{"event_index": 2, "name": "元数据", "start_ns": 1790000000000000000+i}]}}
                for i in range(count)]

    def test_stream_assembly_equals_previous_full_payload_with_exact_times(self):
        entries = self.evidence()
        metadata = {"schema": probe.SCHEMA, "rank": "dp4", "completed_cases": 3, "planned_cases": 3,
                    "cases": entries, "scope": "complete evidence"}
        original = deepcopy(metadata)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            paths = [probe.write_case_checkpoint(entry, i, out, "dp4") for i, entry in enumerate(entries)]
            checkpoints = probe.CaseCheckpoints(paths)
            messages = []
            with patch.object(probe.json, "load", side_effect=AssertionError("Merge must not decode case trees")):
                destination = probe.write_full_checkpoint(metadata, checkpoints, out, "dp4", messages.append)
            actual = json.loads(destination.read_bytes())
            self.assertEqual(actual, probe.wire(metadata))
            self.assertEqual(list(checkpoints), probe.wire(entries))
            self.assertEqual(list(checkpoints), probe.wire(entries))  # writers can iterate independently
            self.assertEqual(len(checkpoints), 3)
            self.assertEqual(len([m for m in messages if "Assembling" in m]), 3)
        self.assertEqual(metadata, original)

    def test_each_case_is_serialized_once_and_final_merge_never_walks_it_again(self):
        entries = self.evidence(12)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            calls = []
            real_wire = probe.wire
            def counted(value, key=""):
                if isinstance(value, dict) and value.get("case"):
                    calls.append(value["case"])
                return real_wire(value, key)
            with patch.object(probe, "wire", side_effect=counted):
                files = [probe.write_case_checkpoint(entry, i, out, "dp4") for i, entry in enumerate(entries)]
                probe.write_full_checkpoint({"schema": probe.SCHEMA, "cases": entries},
                                            probe.CaseCheckpoints(files), out, "dp4")
            self.assertEqual(calls, [entry["case"] for entry in entries])
            self.assertTrue(all(path.exists() for path in files))

    def test_failed_merge_preserves_last_complete_json_and_every_case_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            path = probe.write_case_checkpoint(self.evidence(1)[0], 0, out, "dp4")
            before = path.read_bytes()
            destination = out/"u_swimlane_dp4.json"
            destination.write_text('{"previous":"complete"}', encoding="utf-8")
            with self.assertRaises(FileNotFoundError):
                probe.write_full_checkpoint({"schema": probe.SCHEMA},
                    probe.CaseCheckpoints([path, out/"missing-case.json"]), out, "dp4")
            self.assertEqual(json.loads(destination.read_bytes()), {"previous": "complete"})
            self.assertEqual(path.read_bytes(), before)

    def test_existing_csv_indexes_accept_one_case_at_a_time_without_changing_values(self):
        entries = self.evidence()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = root/"old"; old.mkdir()
            new = root/"new"; new.mkdir()
            paths = [probe.write_case_checkpoint(entry, i, root, "dp4") for i, entry in enumerate(entries)]
            old_payload = {"cases": entries}
            new_payload = {"cases": probe.CaseCheckpoints(paths)}
            probe.write_status(old_payload, old, "dp4")
            probe.write_status(new_payload, new, "dp4")
            a = probe.write_device_tables(old_payload, old, "dp4")
            b = probe.write_device_tables(new_payload, new, "dp4")
            for previous, current in zip(a, b):
                self.assertEqual(previous.read_bytes(), current.read_bytes())
            self.assertEqual((old/"u_swimlane_dp4_status.csv").read_bytes(),
                             (new/"u_swimlane_dp4_status.csv").read_bytes())


if __name__ == "__main__":
    unittest.main()
