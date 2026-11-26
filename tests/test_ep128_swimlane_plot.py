"""Swimlane views retain real intervals and never repair missing evidence."""

from copy import deepcopy
import json
import math
from pathlib import Path
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import ep128_swimlane_plot as plot
import test_ep128_swimlane_share as fixtures
import test_ep128_swimlane_anchor_audit as audit_fixtures
from ep128_swimlane_anchor_audit import audit_stage_anchors


def lane(panel, role):
    lanes = panel["host_lanes"]
    if isinstance(lanes, dict):
        return lanes.get(role)
    return next((item for item in lanes if item.get("role", item.get("key")) == role), None)


def summarize(data=None, **kwargs):
    return plot.summarize_swimlanes(fixtures.payload() if data is None else data, **kwargs)


class SwimlanePlotEvidenceTests(unittest.TestCase):
    def test_default_child_view_partitions_each_parent_once_without_nested_double_count(self):
        panel, = summarize()["panels"]
        self.assertEqual(len(panel["child_plot_rows"]), len(panel["host_lanes"]))
        for group in panel["child_plot_rows"]:
            self.assertTrue(group["exclusive"])
            parent = panel["host_lanes"][group["lane"]]
            pieces = group["events"]
            self.assertEqual(sum(p["duration_ns"] for p in pieces), parent["duration_ns"])
            self.assertEqual(pieces[0]["start_ns"], parent["start_ns"])
            self.assertEqual(pieces[-1]["end_ns"], parent["end_ns"])
            self.assertTrue(all(a["end_ns"] == b["start_ns"] for a,b in zip(pieces,pieces[1:])))

    def test_worker_outer_scope_sets_axis_without_changing_device_cycle(self):
        result=summarize()
        panel,=result["panels"]
        business=panel["worker_business"]
        self.assertTrue(business["closed"])
        self.assertLessEqual(result["x_min_ms"],min(s["start_ms"] for s in business["outer_scopes"]))
        self.assertGreaterEqual(result["x_max_ms"],max(s["end_ms"] for s in business["outer_scopes"]))
        self.assertEqual(panel["duration_ns"],300)
        self.assertNotEqual(business["duration_ns"],panel["duration_ns"])

    def test_full_v2_recovers_runner_attention_without_reading_trace_or_mutating_input(self):
        data = fixtures.payload()
        before = deepcopy(data)
        with patch("builtins.open", side_effect=AssertionError("No capture reads")), \
                patch.object(Path, "open", side_effect=AssertionError("No trace/CSV reads")):
            result = summarize(data)
        self.assertEqual(data, before)
        self.assertEqual(result["rank"], "dp4")
        panel, = result["panels"]
        for role in ("outer_propose", "preparation_dp_aggregate", "runner_inputs", "runner_attn", "state_attn"):
            self.assertIsNotNone(lane(panel, role), role)
        self.assertEqual(int(lane(panel, "runner_attn")["duration_ns"]), 6)

    def test_compact_references_restore_child_names_and_exact_times(self):
        compact = fixtures.build(fixtures.payload())
        before = deepcopy(compact)
        panel, = summarize(compact)["panels"]
        self.assertEqual(compact, before)
        rows = panel["detail_rows"]
        gather = next(row for row in rows if row.get("event_index") == 46)
        self.assertIn("gather_block_tables", json.dumps(gather))
        self.assertEqual(int(gather["start_ns"]), fixtures.EPOCH + 232)
        self.assertEqual(int(gather["end_ns"]), fixtures.EPOCH + 234)

    def test_negative_host_start_and_cross_boundary_call_are_not_clipped(self):
        result = summarize()
        panel, = result["panels"]
        propose = lane(panel, "outer_propose")
        self.assertEqual(int(propose["start_ns"]), fixtures.EPOCH + 50)
        self.assertEqual(int(propose["end_ns"]), fixtures.EPOCH + 140)
        self.assertLess(result["x_min_ms"], 0)
        self.assertLessEqual(result["x_min_ms"], -50 / 1e6)
        self.assertGreaterEqual(result["x_max_ms"], 300 / 1e6)

    def test_all_fifteen_device_stages_and_both_HC_ranges_remain_present(self):
        panel, = summarize()["panels"]
        stages = panel["stages"]
        self.assertEqual(sorted(stage["stage_number"] for stage in stages), list(range(1, 16)))
        self.assertEqual(sum(int(stage["duration_ns"]) for stage in stages), 100)
        regions = panel["regions"]
        self.assertEqual(int(regions["main_hc"]["duration_ns"]), 150)
        self.assertEqual(int(regions["draft_hc"]["duration_ns"]), 50)

    def test_clock_failure_keeps_device_ranges_but_no_host_lane(self):
        source = fixtures.payload()
        case = source["cases"][0]
        case.update(status="ERROR", error="ValueError: Trace alignment failed: CLOCK_MISMATCH")
        case["trace"].update(status="CLOCK_MISMATCH", max_clock_residual_ns=5000)
        panel, = summarize(source)["panels"]
        self.assertFalse(any(lane(panel, role) for role in (
            "outer_propose", "preparation_dp_aggregate", "runner_inputs", "runner_attn", "state_attn")))
        self.assertEqual(len(panel["stages"]), 15)
        self.assertIn("CLOCK_MISMATCH", json.dumps(panel))

    def test_wrong_fixed_rank_is_not_borrowed(self):
        with self.assertRaises(ValueError):
            summarize(rank="dp3")

    def test_cases_manifests_and_encoded_parts_are_not_treated_as_trace_evidence(self):
        for source in ({"schema_version": 2, "cases": [{}]},
                       {"schema": "u_swimlane_ppt_gzip_base64_v1", "part": 1, "data": "invalid"}):
            with self.subTest(schema=source.get("schema", source.get("schema_version"))):
                with self.assertRaises(ValueError):
                    summarize(source)

    def test_all_recorded_children_survive_the_detail_table_without_topn(self):
        case = fixtures.case_fixture()
        for number in range(1000, 1700):
            item = fixtures.fixture.host(number, "metadata_builder", 260, 270, number, 31)
            item.update(start_ns=fixtures.EPOCH+260, end_ns=fixtures.EPOCH+270, event_type="python")
            case["host_events"].append(item)
        panel, = summarize(fixtures.payload([case]))["panels"]
        ids = {row.get("event_index") for row in panel["detail_rows"]}
        self.assertTrue(set(range(1000, 1700)).issubset(ids))

    def test_case_order_uses_numeric_config_BS_and_S(self):
        cases = []
        for name, bs, speculative in (("first", 16, 7), ("second", 2, 15), ("third", 2, 7)):
            case = fixtures.case_fixture(name)
            case.update(bs=bs, S=speculative)
            cases.append(case)
        result = summarize(fixtures.payload(cases))
        self.assertEqual([(panel["bs"], panel["S"]) for panel in result["panels"]], [(2, 7), (2, 15), (16, 7)])

    def test_DP_sync_and_wait_are_nested_wall_scopes_not_additive_costs(self):
        panel, = summarize()["panels"]
        breakdown = panel["dp_breakdown"]
        self.assertEqual(int(breakdown["dp_coordination_wall_ns"]), 19)
        self.assertEqual(int(breakdown["verified_dp_wait_union_ns"]), 11)
        self.assertEqual(int(breakdown["outside_coordination_wall_ns"]), 21)
        self.assertEqual(sum(int(piece["duration_ns"]) for piece in breakdown["parts"]), 40)

    def test_unrecorded_wait_is_unknown_and_missing_slot_wrapper_is_not_invented(self):
        case = fixtures.case_fixture()
        removed = {45, 47}
        case["host_events"] = [event for event in case["host_events"] if event["event_index"] not in removed]
        case["anchor_links"]["source_events"] = [event for event in case["anchor_links"]["source_events"]
                                                  if event["event_index"] not in removed]
        panel, = summarize(fixtures.payload([case]))["panels"]
        self.assertIsNone(panel["dp_breakdown"]["verified_dp_wait_union_ns"])
        rows = panel["detail_rows"]
        self.assertFalse(any(row.get("event_index") == 47 for row in rows))
        self.assertIsNotNone(lane(panel, "runner_attn"))

    def test_saved_solid_edge_still_requires_unique_matched_boundary(self):
        for anchor_status, stage_status, confirmed in (
                ("EXACT_SOURCE_ROW", "MATCHED", True),
                ("AMBIGUOUS", "MATCHED", False),
                ("EXACT_SOURCE_ROW", "UNRESOLVED", False)):
            with self.subTest(anchor=anchor_status, stage=stage_status):
                compact = fixtures.build(fixtures.payload())
                bindings = compact["cases"][0]["anchor_links"]["anchor_bindings"]
                bindings[0].update(timestamp_ns=str(fixtures.EPOCH+260), anchor_kind="u_stage",
                                   status=anchor_status, stage_status=stage_status)
                panel, = summarize(compact)["panels"]
                edge = next(edge for edge in panel["edges"] if edge["row_index"] == 101)
                self.assertEqual(edge["runtime_submission_confirmed"], confirmed)
                self.assertEqual(edge["line_style"], "solid" if confirmed else "dashed")

    def test_missing_source_event_cannot_keep_a_confirmed_solid_edge(self):
        compact = fixtures.build(fixtures.payload())
        case = compact["cases"][0]
        case["anchor_links"]["anchor_bindings"][0].update(
            timestamp_ns=str(fixtures.EPOCH+260), anchor_kind="u_stage", stage_status="MATCHED")
        case["anchor_links"]["source_events"] = [event for event in case["anchor_links"]["source_events"]
                                                 if event["event_index"] != 200000]
        case["host_events"] = [event for event in case["host_events"] if event["event_index"] != 200000]
        panel, = summarize(compact)["panels"]
        edge = next(edge for edge in panel["edges"] if edge["row_index"] == 101)
        self.assertFalse(edge["runtime_submission_confirmed"])
        self.assertEqual(edge["line_style"], "dashed")

    def test_missing_required_source_ancestor_cannot_keep_a_confirmed_solid_edge(self):
        compact = fixtures.build(fixtures.payload())
        case = compact["cases"][0]
        case["anchor_links"]["anchor_bindings"][0].update(
            timestamp_ns=str(fixtures.EPOCH+260), anchor_kind="u_stage", stage_status="MATCHED")
        case["anchor_links"]["source_events"] = [event for event in case["anchor_links"]["source_events"]
                                                 if event["event_index"] != 205000]
        # The direct source and Host still exist; the required core parent is missing.
        panel, = summarize(compact)["panels"]
        edge = next(edge for edge in panel["edges"] if edge["row_index"] == 101)
        self.assertFalse(edge["runtime_submission_confirmed"])
        self.assertEqual(edge["line_style"], "dashed")


class SwimlanePlotGeometryTests(unittest.TestCase):
    def test_all_thirty_destinations_are_visible_and_shared_task_endpoint_is_drawn_once(self):
        panel, = summarize()["panels"]
        panel["anchor_audit"] = audit_stage_anchors(audit_fixtures.fixture(), audit_fixtures.EPOCH)
        drawing = plot.panel_drawing(panel)
        labels = [e for e in drawing["elements"] if e.get("role") == "anchor_destination"]
        ids = [key for e in labels for key in e["anchor_ids"]]
        self.assertEqual(sorted(ids), sorted("U"+str(i)+"."+side for i in range(1,16) for side in ("start","end")))
        self.assertEqual(len(ids), len(set(ids)))
        edges = [e for e in drawing["elements"] if e.get("role") == "source_edge"]
        self.assertEqual(len(edges), 1)
        self.assertEqual(edges[0]["anchor_ids"], ["U1.end", "U2.start"])
        self.assertTrue(edges[0]["runtime_submission_confirmed"])
        self.assertEqual(edges[0]["dash"], "solid")
        self.assertIn("无已证实提交线", plot.render_svg(drawing))

    def test_clock_failure_has_thirty_audits_but_no_source_scope_or_source_arrow(self):
        source = fixtures.payload()
        source["cases"][0].update(status="ERROR", error="CLOCK_MISMATCH")
        source["cases"][0]["trace"]["status"] = "CLOCK_MISMATCH"
        panel, = summarize(source)["panels"]
        drawing = plot.panel_drawing(panel)
        self.assertEqual(len(panel["anchor_audit"]["audit_entries"]), 30)
        self.assertFalse(any(e.get("role") in ("source_edge", "source_scope") for e in drawing["elements"]))

    def test_shared_endpoint_never_promotes_an_unmatched_stage_to_solid(self):
        panel, = summarize()["panels"]
        data = audit_fixtures.fixture()
        data["rounds"][0]["u_windows"][0]["stages"][1]["status"]="UNRESOLVED"
        panel["anchor_audit"] = audit_stage_anchors(data,audit_fixtures.EPOCH)
        drawing=plot.panel_drawing(panel)
        edges=[e for e in drawing["elements"] if e.get("role")=="source_edge"]
        self.assertEqual([e["anchor_ids"] for e in edges],[["U1.end"]])
        labels=[e for e in drawing["elements"] if e.get("role")=="anchor_destination"]
        self.assertTrue(any(e["anchor_ids"]==["U2.start"] for e in labels))

    def drawing(self, **kwargs):
        result = summarize()
        return plot.panel_drawing(result["panels"][0], width=1600,
            x_min_ms=result["x_min_ms"], x_max_ms=result["x_max_ms"], **kwargs)

    def test_geometry_is_finite_and_inside_the_complete_panel(self):
        drawing = self.drawing()
        self.assertGreater(drawing["height"], 0)
        self.assertEqual(drawing["width"], 1600)
        self.assertGreater(len(drawing["elements"]), 20)
        for element in drawing["elements"]:
            for key in ("x", "y", "x1", "x2", "y1", "y2", "width", "height"):
                if key in element:
                    self.assertTrue(math.isfinite(element[key]), (key, element))
            if "x" in element:
                self.assertGreaterEqual(element["x"], 0)
                self.assertLessEqual(element["x"] + element.get("width", 0), drawing["width"] + 1)
            if "y" in element:
                self.assertGreaterEqual(element["y"], 0)
                self.assertLessEqual(element["y"] + element.get("height", 0), drawing["height"] + 1)

    def test_small_overview_and_long_clock_error_text_boxes_stay_inside_panel(self):
        source = fixtures.payload()
        source["cases"][0].update(status="ERROR", error="CLOCK_MISMATCH：" + "未确认设备时钟；"*60)
        source["cases"][0]["trace"]["status"] = "CLOCK_MISMATCH"
        cases = [summarize()["panels"][0], summarize(source)["panels"][0]]
        for panel in cases:
            for overview, width in ((True, 620), (False, 1600)):
                with self.subTest(clock=panel.get("trace_status"), overview=overview):
                    drawing = plot.panel_drawing(panel, width=width, overview=overview)
                    for element in drawing["elements"]:
                        for start, size, bound in (("x", "width", "width"), ("y", "height", "height")):
                            if start in element:
                                self.assertGreaterEqual(element[start], 0)
                                self.assertLessEqual(element[start]+element.get(size, 0), drawing[bound]+1, element)
                        for point, bound in (("x1", "width"), ("x2", "width"), ("y1", "height"), ("y2", "height")):
                            if point in element:
                                self.assertGreaterEqual(element[point], 0)
                                self.assertLessEqual(element[point], drawing[bound]+1, element)

    def test_U_boundaries_continue_through_the_Host_rows(self):
        drawing = self.drawing()
        boundaries = [element for element in drawing["elements"] if element.get("role") == "u_boundary"]
        self.assertGreaterEqual(len(boundaries), 15)
        hosts = [element for element in drawing["elements"] if element.get("role") == "host"]
        self.assertTrue(hosts)
        last_host_y = max(element["y"] + element["height"] for element in hosts)
        self.assertTrue(all(element["y1"] < min(host["y"] for host in hosts)
                            and element["y2"] >= last_host_y for element in boundaries))

    def test_overview_uses_same_time_axis_and_does_not_change_panel_data(self):
        result = summarize()
        panel = result["panels"][0]
        before = deepcopy(panel)
        overview = plot.panel_drawing(panel, width=760, overview=True,
            x_min_ms=result["x_min_ms"], x_max_ms=result["x_max_ms"])
        detail = self.drawing()
        self.assertEqual(panel, before)
        self.assertLess(overview["height"], detail["height"])
        self.assertIn("elements", overview)

    def test_SVG_is_self_contained_and_escapes_source_text(self):
        source = fixtures.payload([fixtures.case_fixture("<unsafe>&\"case")])
        panel = summarize(source)["panels"][0]
        drawing = plot.panel_drawing(panel, width=1600)
        svg = plot.render_svg(drawing)
        root = ET.fromstring(svg)
        self.assertEqual(root.tag.rsplit("}", 1)[-1], "svg")
        self.assertIn('<unsafe>&"case', "".join(root.itertext()))
        self.assertNotIn("<unsafe>", svg)
        self.assertFalse(root.findall(".//{*}image"))
        self.assertFalse(root.findall(".//{*}script"))
        self.assertNotIn("http://", svg.replace('http://www.w3.org/2000/svg', ''))

    def test_clock_error_has_no_CPU_bars_in_rendered_geometry(self):
        source = fixtures.payload()
        source["cases"][0].update(status="ERROR", error="CLOCK_MISMATCH")
        source["cases"][0]["trace"]["status"] = "CLOCK_MISMATCH"
        panel = summarize(source)["panels"][0]
        drawing = plot.panel_drawing(panel, width=1600)
        self.assertFalse(any(element.get("role") in ("host", "child") for element in drawing["elements"]))
        self.assertIn("CLOCK_MISMATCH", plot.render_svg(drawing))


if __name__ == "__main__":
    unittest.main()
