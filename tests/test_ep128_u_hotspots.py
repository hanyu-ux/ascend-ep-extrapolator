"""Compact U charts use stored source parents, not target estimates or traces."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from zipfile import ZipFile

import ep128_u_hotspots as hotspots
from ep128_xlsx_writer import Workbook
from test_ep128_layer_mean_excel import residual_report
from test_ep128_v2_excel import renderer


HERE = Path(__file__).resolve().parents[1]
NS = {"c": "http://schemas.openxmlformats.org/drawingml/2006/chart",
      "a": "http://schemas.openxmlformats.org/drawingml/2006/main"}
WINDOWS = (("main_to_draft", 8), ("draft_to_nextmain", 7))


def hotspot_report():
    """Synthetic 120ms U with known unequal parent costs and two stored ranks."""
    captures = []
    for rank, scale in (("dp4", 1), ("dp5", 2)):
        stages, summary = [], []
        offset = 0
        for window, count in WINDOWS:
            for local in range(1, count + 1):
                number = offset + local
                stages.append(dict(window_key=window, stage_key="stage_" + str(number),
                    stage_order=local, level=0, parent_stage_key=None, label="SYNTHETIC阶段" + str(number),
                    status="MATCHED", rounds=2, matched_windows=2, duration_ms_mean=number * scale))
            summary.append(dict(window_key=window, rounds=2,
                duration_ms_mean=sum(range(offset + 1, offset + count + 1)) * scale))
            offset += count
        captures.append(dict(rank=rank, capture_path="MUST_NOT_READ_RAW_CAPTURE.csv", rounds=[],
            step_residuals=dict(rank=rank, status="DEVICE_ONLY", residual_ms_mean=120 * scale,
                trace=dict(status="NOT_FOUND", path="MUST_NOT_READ_TRACE.json"),
                stages_csv="MUST_NOT_READ_STAGES.csv", summary=summary, stage_summary=stages,
                substage_summary=[dict(window_key="draft_to_nextmain", stage_key="child_not_added",
                    level=1, parent_stage_key="stage_14", stage_order=1, duration_ms_mean=99999)])))
    case = dict(name="bs8-s15", config=dict(bs=8, draft_tokens=15), measurements=captures,
                source=dict(aggregate=dict(selected_rank="dp4")),
                projection=dict(layer_mean_methods=[dict(id="method1", status="UNAVAILABLE",
                    reasons=["SYNTHETIC_TARGET_NOT_CONFIGURED"], selected_rank="dp5")]))
    return dict(schema_version=2, profile_version="SYNTHETIC_ONLY", cases=[case], failures=[])


def chart_title(chart):
    return "".join(chart.find("c:chart/c:title", NS).itertext())


class UHotspotSummaryTests(unittest.TestCase):
    def test_top5_plus_other_close_to_original_u_with_original_denominator(self):
        report = hotspot_report()
        before = copy.deepcopy(report)
        panel, = hotspots.summarize_u_hotspots(report)
        self.assertEqual(report, before)
        self.assertEqual(panel["status"], "MATCHED")
        self.assertEqual(panel["rank"], "dp4")
        self.assertEqual(panel["u_ms"], 120)
        self.assertEqual([item["stage_number"] for item in panel["rows"][:5]], [15, 14, 13, 12, 11])
        self.assertEqual([item["value_ms"] for item in panel["rows"]], [15, 14, 13, 12, 11, 55])
        other = panel["rows"][-1]
        self.assertEqual(other["kind"], "other")
        self.assertEqual(set(other["members"]), set(range(1, 11)))
        self.assertAlmostEqual(sum(item["value_ms"] for item in panel["rows"]), 120)
        self.assertAlmostEqual(sum(item["share_u"] for item in panel["rows"]), 1)
        self.assertAlmostEqual(panel["top_share"], 65 / 120)
        self.assertAlmostEqual(panel["rows"][0]["share_u"], 15 / 120)
        self.assertIn("15.000 ms", panel["rows"][0]["label_text"])
        self.assertIn("12.5%", panel["rows"][0]["label_text"])

    def test_fixed_rank_and_source_only_work_without_publishable_target(self):
        report = hotspot_report()
        fixed, = hotspots.summarize_u_hotspots(report, rank="dp5", top=3)
        self.assertEqual(fixed["status"], "MATCHED")
        self.assertEqual(fixed["rank"], "dp5")
        self.assertEqual(fixed["u_ms"], 240)
        self.assertEqual([item["value_ms"] for item in fixed["rows"]], [30, 28, 26, 156])
        report["cases"][0].pop("projection")
        default, = hotspots.summarize_u_hotspots(report)
        self.assertEqual(default["status"], "MATCHED")
        self.assertEqual(default["rank"], "dp4")

    def test_filters_and_bs_s_order_do_not_parse_case_names(self):
        report = hotspot_report()
        first = report["cases"][0]
        a, b = copy.deepcopy(first), copy.deepcopy(first)
        a.update(name="z-not-a-bs-name", config=dict(bs=16, draft_tokens=7))
        b.update(name="a-not-a-bs-name", config=dict(bs=2, draft_tokens=15))
        report["cases"] = [first, b, a]
        panels = hotspots.summarize_u_hotspots(report)
        self.assertEqual([p["case"] for p in panels], [a["name"], b["name"], first["name"]])
        filtered = hotspots.summarize_u_hotspots(report, names=[first["name"]])
        self.assertEqual([p["case"] for p in filtered], [first["name"]])

    def test_missing_duplicate_partial_or_nonclosing_evidence_is_not_zero(self):
        for variant in ("missing_rank", "duplicate_rank", "missing_stage", "duplicate_stage",
                        "partial", "different_rounds", "different_window_rounds", "window_nonclosure", "u_nonclosure"):
            with self.subTest(variant=variant):
                report = hotspot_report()
                case = report["cases"][0]
                capture = case["measurements"][0]
                diagnostic = capture["step_residuals"]
                requested = None
                if variant == "missing_rank":
                    requested = "dp_missing"
                elif variant == "duplicate_rank":
                    case["measurements"].append(copy.deepcopy(capture))
                elif variant == "missing_stage":
                    diagnostic["stage_summary"].pop(0)
                elif variant == "duplicate_stage":
                    diagnostic["stage_summary"][1]["stage_order"] = 1
                elif variant == "partial":
                    diagnostic["stage_summary"][0].update(status="PARTIAL", matched_windows=1)
                elif variant == "different_rounds":
                    diagnostic["summary"][0]["rounds"] = 3
                elif variant == "different_window_rounds":
                    diagnostic["summary"][0]["rounds"] = 3
                    for row in diagnostic["stage_summary"]:
                        if row["window_key"] == "main_to_draft":
                            row.update(rounds=3, matched_windows=3)
                elif variant == "window_nonclosure":
                    diagnostic["summary"][0]["duration_ms_mean"] += 1
                else:
                    diagnostic["residual_ms_mean"] += 1
                panel, = hotspots.summarize_u_hotspots(report, rank=requested)
                self.assertEqual(panel["status"], "UNAVAILABLE")
                self.assertTrue(panel["reason"])
                self.assertFalse(panel["rows"])

    def test_unknown_window_and_zero_u_do_not_become_identified_stage_bars(self):
        for variant in ("unresolved", "zero"):
            report = hotspot_report()
            diagnostic = report["cases"][0]["measurements"][0]["step_residuals"]
            if variant == "unresolved":
                diagnostic["stage_summary"][0].update(status="UNRESOLVED", duration_ms_mean=None)
            else:
                diagnostic["residual_ms_mean"] = 0
            panel, = hotspots.summarize_u_hotspots(report)
            self.assertEqual(panel["status"], "UNAVAILABLE")
            self.assertFalse(panel["rows"])

    def test_top15_keeps_every_parent_without_other_or_children(self):
        panel, = hotspots.summarize_u_hotspots(hotspot_report(), top=15)
        self.assertEqual(len(panel["rows"]), 15)
        self.assertTrue(all(row["kind"] == "stage" for row in panel["rows"]))
        self.assertEqual(set(row["stage_number"] for row in panel["rows"]), set(range(1, 16)))
        self.assertAlmostEqual(panel["top_share"], 1)


class UHotspotExportsTests(unittest.TestCase):
    def test_main_offline_flag_reads_only_existing_report_and_exports_small_charts(self):
        with tempfile.TemporaryDirectory(prefix="u-hotspots-cli-") as directory:
            root = Path(directory)
            report_path = root / "model_architecture_report.json"
            report_path.write_text(json.dumps(hotspot_report(), ensure_ascii=False), encoding="utf-8")
            out = root / "重绘结果"
            result = subprocess.run([sys.executable, "-B", str(HERE / "profile_ep128.py"),
                "--u-hotspots-report", str(root), "--u-hotspots-rank", "dp4", "--u-top", "3",
                "--out", str(out)], capture_output=True, text=True, encoding="utf-8", errors="replace")
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            for name in ("u_hotspots.html", "u_hotspots.csv", "u_hotspots.xlsx"):
                self.assertTrue((out / name).is_file(), name)
            svgs = list((out / "u_hotspots_svg").glob("*.svg"))
            self.assertEqual(len(svgs), 1)
            svg = ET.fromstring(svgs[0].read_text(encoding="utf-8"))
            text = "".join(svg.itertext())
            self.assertIn("ms", text)
            self.assertIn("%", text)
            self.assertFalse((out / "step_residual_details").exists())
            with ZipFile(out / "u_hotspots.xlsx") as archive:
                self.assertIsNone(archive.testzip())
                charts = [name for name in archive.namelist() if name.startswith("xl/charts/chart") and name.endswith(".xml")]
                self.assertEqual(len(charts), 1)
                chart = ET.fromstring(archive.read(charts[0]))
                self.assertEqual(chart.find(".//c:barChart/c:barDir", NS).get("val"), "bar")
                self.assertIn("ms", "".join(chart.itertext()))
                self.assertIn("%", "".join(chart.itertext()))

    def test_full_renderer_adds_source_only_compact_charts_before_original_sections(self):
        report = residual_report()
        report["cases"][0]["source"]["aggregate"]["selected_rank"] = "dp7"
        before = copy.deepcopy(report)
        book = Workbook()
        book.detail_rows, book.report_origin = 1000, "stored.json"
        sheet = renderer._extra_charts_sheet(book, report)
        compact = [chart for chart in sheet.charts if chart["title"].startswith("U主要耗时 ·")]
        self.assertEqual(len(compact), 1)
        self.assertEqual(len(compact[0]["series"][0]["values"]), 6)
        self.assertAlmostEqual(sum(compact[0]["series"][0]["values"]), 13.125)
        self.assertEqual(len(compact[0]["series"][0]["data_labels"]), 6)
        self.assertTrue(all("ms" in label and "%" in label for label in compact[0]["series"][0]["data_labels"]))
        self.assertTrue(compact[0]["categories_ref"].startswith("'额外图表统计'!"))
        self.assertTrue(compact[0]["series"][0]["values_ref"].startswith("'额外图表统计'!"))
        self.assertTrue(all(ref.startswith("'额外图表统计'!")
                            for ref in compact[0]["series"][0]["data_label_refs"]))
        original = [chart for chart in sheet.charts if not chart["title"].startswith("U主要耗时 ·")]
        self.assertTrue(original)
        self.assertLess(compact[0]["anchor"][0], min(chart["anchor"][0] for chart in original))
        html = renderer._extra_charts_html(report)
        self.assertIn("Top5", html)
        self.assertIn("源实测", html)
        self.assertIn("ms", html)
        self.assertIn("%", html)
        self.assertEqual(report, before)


if __name__ == "__main__":
    unittest.main()
