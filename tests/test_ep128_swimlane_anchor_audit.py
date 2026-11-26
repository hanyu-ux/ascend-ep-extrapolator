"""Every device boundary is audited without inventing Python submissions."""

from copy import deepcopy
import unittest

from ep128_swimlane_anchor_audit import audit_stage_anchors


EPOCH = 1795093535000000000


def event(number, name="aten::gather", start=-100, end=-90, cat="cpu_op"):
    return {"event_index": number, "name": name, "cat": cat, "start_ns": str(EPOCH+start),
            "end_ns": str(EPOCH+end), "pid": "1", "tid": "1"}


def fixture():
    stages, bindings = [], []
    for number in range(1, 16):
        start, end = (number-1)*2, number*2
        stage = {"round_id": "hc-1", "stage_number": number, "stage_key": "stage"+str(number),
                 "window_key": "main_to_draft", "status": "MATCHED", "start_ns": str(EPOCH+start),
                 "end_ns": str(EPOCH+end), "start_anchor": {"name": "start"}, "end_anchor": {"name": "end"}}
        stages.append(stage)
        for side, stamp in (("start", start), ("end", end)):
            bindings.append({"round_id": "hc-1", "stage_number": number, "stage_key": stage["stage_key"],
                             "window_key": stage["window_key"], "stage_status": "MATCHED", "anchor_kind": "u_stage",
                             "side": side, "timestamp_ns": str(EPOCH+stamp), "edge": "start",
                             "status": "EXACT_SOURCE_ROW", "device_row_indexes": [100+stamp//2]})
    source = event(20)
    parent = event(30, "speculator.py(10): propose", -120, -80, "python_function")
    link = {"row_index": 101, "name": "Gather", "status": "LINKED",
            "device_start_ns": str(EPOCH+2), "device_end_ns": str(EPOCH+3),
            "cpu_event_index": 20, "cpu_start_ns": source["start_ns"], "cpu_end_ns": source["end_ns"],
            "source_event_indexes": [20], "missing_source_event_indexes": [],
            "flow_ids": ["exact-flow"], "host_associations": [{"host_event_index": 30, "source_event_index": 20,
                "relation": "TIME_CONTAINMENT_ONLY", "line_style": "dashed", "runtime_submission_confirmed": False}]}
    return {"selected_round_id": "hc-1", "status": "READY", "trace": {"status": "ALIGNED_X_EVENTS", "sha256": "trace"},
            "rounds": [{"round_id": "hc-1", "u_windows": [{"window_key": "main_to_draft", "stages": stages}]}],
            "host_events": [parent], "anchor_links": {"clock_alignment_status": "ALIGNED_X_EVENTS", "trace_sha256": "trace",
                "anchor_bindings": bindings, "links": [link], "source_events": [source, parent]}}


def inspect(case=None):
    return audit_stage_anchors(fixture() if case is None else case, EPOCH)


def boundary(report, name):
    return next(e for e in report["audit_entries"] if e["anchor_id"] == name)


class AnchorAuditTests(unittest.TestCase):
    def test_float_epoch_is_not_accepted_as_an_exact_boundary(self):
        with self.assertRaises(ValueError):
            audit_stage_anchors(fixture(), float(EPOCH))

    def test_all_fifteen_stages_have_both_exact_boundaries_and_no_input_mutation(self):
        case = fixture()
        before = deepcopy(case)
        report = inspect(case)
        self.assertEqual(case, before)
        self.assertEqual([e["anchor_id"] for e in report["audit_entries"]],
                         ["U"+str(i)+"."+side for i in range(1,16) for side in ("start","end")])
        self.assertEqual(report["boundary_count"], 30)
        self.assertEqual(boundary(report, "U15.end")["timestamp_ns"], str(EPOCH+30))

    def test_adjacent_boundaries_share_physical_task_and_endpoint(self):
        report = inspect()
        first, second = boundary(report, "U1.end"), boundary(report, "U2.start")
        self.assertEqual(first["physical_anchor_id"], second["physical_anchor_id"])
        source = next(s for s in report["physical_sources"] if s["source_id"] == first["source_id"])
        self.assertEqual(source["anchor_ids"], ["U1.end", "U2.start"])
        self.assertEqual(report["verified_boundary_count"], 2)
        self.assertEqual(report["physical_source_counts"]["VERIFIED_CPU_API"], 1)

    def test_CPU_op_source_is_proven_but_time_containment_does_not_prove_Python_owner(self):
        row = boundary(inspect(), "U1.end")
        self.assertEqual(row["grade"], "VERIFIED_CPU_API")
        self.assertTrue(row["arrow_allowed"])
        self.assertEqual(row["python_owner_grade"], "TIME_CONTAINMENT_ONLY")
        self.assertFalse(row["verified_python_owners"])
        self.assertEqual(row["source_start_ns"], str(EPOCH-100))
        self.assertEqual(row["source_start_ms"], -.0001)

    def test_explicit_recorded_parent_path_proves_Python_owner(self):
        case = fixture()
        assoc = case["anchor_links"]["links"][0]["host_associations"][0]
        assoc.update(relation="EXPLICIT_PYTHON_PARENT_ID", line_style="solid", runtime_submission_confirmed=True,
                     explicit_parent_event_indexes=[20,30])
        row = boundary(inspect(case), "U1.end")
        self.assertEqual(row["grade"], "VERIFIED_PYTHON_SUBMISSION")
        self.assertEqual(row["verified_python_owners"][0]["event_index"], 30)

    def test_missing_source_reference_or_required_ancestor_rejects_arrow(self):
        for modify in (lambda c: c["anchor_links"]["source_events"].clear(),
                       lambda c: c["anchor_links"]["links"][0]["source_event_indexes"].append(999),
                       lambda c: c["anchor_links"]["links"][0].update(cpu_ancestors=[{"id":999}]),
                       lambda c: c["anchor_links"]["links"][0].update(cpu_start_ns=str(EPOCH+4))):
            with self.subTest(modify=modify):
                case = fixture(); modify(case)
                row = boundary(inspect(case), "U1.end")
                self.assertFalse(row["arrow_allowed"])
                self.assertIsNone(row["source_time_ms"])

    def test_missing_python_parent_path_never_upgrades_time_containment(self):
        case = fixture()
        association = case["anchor_links"]["links"][0]["host_associations"][0]
        association.update(relation="EXPLICIT_PYTHON_PARENT_ID", line_style="solid", runtime_submission_confirmed=True,
                           explicit_parent_event_indexes=[20,999,30])
        self.assertEqual(boundary(inspect(case), "U1.end")["grade"], "VERIFIED_CPU_API")

    def test_unknown_source_category_is_not_renamed_as_a_CPU_API(self):
        case = fixture()
        case["anchor_links"]["source_events"][0].pop("cat")
        row = boundary(inspect(case), "U1.end")
        self.assertEqual(row["grade"], "TIME_CONTAINMENT_ONLY")
        self.assertFalse(row["arrow_allowed"])

    def test_clock_failure_or_different_trace_hash_keeps_device_but_no_Host_coordinates(self):
        for modify in (lambda c:c["trace"].update(status="CLOCK_MISMATCH"),
                       lambda c:c["anchor_links"].update(trace_sha256="another-trace"),
                       lambda c:c.update(status="ERROR")):
            with self.subTest(modify=modify):
                case = fixture(); modify(case)
                report = inspect(case)
                self.assertEqual(report["counts"], {"CLOCK_UNALIGNED":30})
                self.assertEqual(report["verified_boundary_count"], 0)
                for row in report["audit_entries"]:
                    self.assertIsNone(row["source_time_ms"])
                    self.assertIsNone(row["source_start_ns"])
                    self.assertIsNotNone(row["time_ms"])
                for source in report["physical_sources"]:
                    self.assertFalse(source["arrow_allowed"])
                    self.assertIsNone(source["source_start_ns"])

    def test_shared_graph_missing_trace_or_ambiguous_sources_never_gain_arrows(self):
        for status in ("GRAPH_SHARED_NO_ORIGIN", "NO_TRACE_DEVICE", "AMBIGUOUS"):
            with self.subTest(status=status):
                case = fixture(); case["anchor_links"]["links"][0]["status"] = status
                row = boundary(inspect(case), "U1.end")
                self.assertEqual(row["grade"], status)
                self.assertFalse(row["arrow_allowed"])
                self.assertIsNone(row["source_time_ms"])

    def test_stage_or_boundary_identity_failure_rejects_even_proven_source(self):
        for modify in (lambda c:c["rounds"][0]["u_windows"][0]["stages"][0].update(status="UNRESOLVED"),
                       lambda c:c["rounds"][0]["u_windows"][0]["stages"][0].update(round_id="hc-other"),
                       lambda c:c["rounds"][0]["u_windows"][0]["stages"][0]["end_anchor"].update(row_index=999),
                       lambda c:c["anchor_links"]["anchor_bindings"][1].update(status="AMBIGUOUS"),
                       lambda c:c["anchor_links"]["anchor_bindings"][1].update(stage_status="UNRESOLVED"),
                       lambda c:c["anchor_links"]["links"][0].update(device_start_ns=str(EPOCH+4)),
                       lambda c:c["anchor_links"]["anchor_bindings"].append(deepcopy(c["anchor_links"]["anchor_bindings"][1]))):
            with self.subTest(modify=modify):
                case = fixture(); modify(case)
                self.assertFalse(boundary(inspect(case), "U1.end")["arrow_allowed"])

    def test_duplicate_source_row_and_conflicting_source_event_are_not_resolved_by_timing(self):
        case = fixture()
        case["anchor_links"]["links"].append(deepcopy(case["anchor_links"]["links"][0]))
        self.assertEqual(boundary(inspect(case), "U1.end")["grade"], "AMBIGUOUS")
        case = fixture()
        case["host_events"].append(event(20, name="different-source"))
        row = boundary(inspect(case), "U1.end")
        self.assertFalse(row["arrow_allowed"])
        self.assertIn(20, row["missing_source_event_indexes"])

    def test_HCPost_end_is_existing_task_endpoint_not_new_submission(self):
        case = fixture()
        first = case["anchor_links"]["anchor_bindings"][0]
        first.update(edge="end", status="HC_MARKER_TIMESTAMP_UNIQUE", declared_anchor_name="previous HCPost.end")
        link = deepcopy(case["anchor_links"]["links"][0])
        link.update(row_index=100, device_start_ns=str(EPOCH-1), device_end_ns=str(EPOCH))
        case["anchor_links"]["links"].append(link)
        row = boundary(inspect(case), "U1.start")
        self.assertTrue(row["arrow_allowed"])
        self.assertIn("不是新增任务", row["endpoint_note"])
        self.assertIn(".end@", row["physical_anchor_id"])

    def test_no_stages_still_has_thirty_missing_audits(self):
        case = fixture(); case["rounds"].clear()
        report = inspect(case)
        self.assertEqual(report["counts"], {"MISSING":30})
        self.assertTrue(all(r["time_ms"] is None for r in report["audit_entries"]))


if __name__ == "__main__":
    unittest.main()
