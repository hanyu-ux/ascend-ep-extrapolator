"""Offline swimlane diagrams extend the existing extra-chart worksheet."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET
from zipfile import ZipFile

import build_ep128_leadership_excel as renderer
import profile_ep128 as core
import test_ep128_swimlane_share as fixtures
from ep128_swimlane_mechanism import mechanism_drawing
from test_ep128_layer_mean_excel import residual_report


ROOT = Path(__file__).resolve().parents[1]
XDR = {"xdr": "http://schemas.openxmlformats.org/drawingml/2006/spreadsheetDrawing"}


def sample_source(directory, twelve=False):
    cases = []
    for bs in ((2, 4, 8, 16) if twelve else (8,)):
        for steps in ((7, 9, 15) if twelve else (15,)):
            case = fixtures.case_fixture("SYNTHETIC_bs" + str(bs) + "_s" + str(steps))
            case.update(bs=bs, S=steps)
            if twelve and steps == 9 and bs in (2, 16):
                case.update(status="ERROR", error="ValueError: Trace alignment failed: CLOCK_MISMATCH")
                case["trace"]["status"] = "CLOCK_MISMATCH"
            cases.append(case)
    compact = fixtures.build(fixtures.payload(cases))
    source = directory / "u_swimlane_ppt_dp4.json"
    source.write_text(json.dumps(compact), encoding="utf-8")
    return source


def workbook_names(path):
    with ZipFile(path) as archive:
        root = ET.fromstring(archive.read("xl/workbook.xml"))
    return [item.attrib["name"] for item in root.findall(".//{*}sheet")]


class SwimlaneExcelIntegrationTests(unittest.TestCase):
    def test_async_mechanism_is_explicitly_non_temporal_and_inside_its_canvas(self):
        drawing=mechanism_drawing()
        self.assertFalse(drawing["temporal"])
        self.assertIn("非实测",drawing["description"])
        self.assertNotIn("真实时间",drawing["description"])
        for e in drawing["elements"]:
            for start,size,bound in (("x","width","width"),("y","height","height")):
                if start in e:
                    self.assertGreaterEqual(e[start],0)
                    self.assertLessEqual(e[start]+e.get(size,0),drawing[bound])
            for point,bound in (("x1","width"),("x2","width"),("y1","height"),("y2","height")):
                if point in e:
                    self.assertGreaterEqual(e[point],0)
                    self.assertLessEqual(e[point],drawing[bound])

    def test_html_is_available_before_excel_save_even_if_excel_export_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            source = sample_source(directory)
            original = source.read_bytes()
            target = directory/"early_preview.xlsx"
            messages = []
            with patch.object(renderer.Workbook, "save", side_effect=OSError("synthetic Excel write failure")):
                with self.assertRaisesRegex(OSError, "synthetic Excel"):
                    renderer.build_swimlane_workbook(target, swimlane_json=source, progress=messages.append)
            self.assertFalse(target.exists())
            document = target.with_suffix(".html").read_text(encoding="utf-8")
            self.assertIn("SYNTHETIC_bs8_s15", document)
            self.assertIn("swimlane-detail", document)
            self.assertEqual(source.read_bytes(), original)
            html_index = next(i for i, text in enumerate(messages) if "HTML泳道已生成" in text)
            excel_index = next(i for i, text in enumerate(messages) if "绘制全部case总览" in text)
            self.assertLess(html_index, excel_index)

    def test_twelve_cases_draw_locally_without_trace_reads_and_keep_error_cases(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            source = sample_source(directory, twelve=True)
            before = source.read_bytes()
            with patch.object(core, "read_capture", side_effect=AssertionError("must not read CSV")), \
                    patch.object(core, "residual_trace_scan", side_effect=AssertionError("must not read trace")):
                target = renderer.build_swimlane_workbook(directory/"local.xlsx", swimlane_json=source)
            self.assertEqual(source.read_bytes(), before)
            self.assertEqual(workbook_names(target), ["额外图表统计"])
            with ZipFile(target) as archive:
                drawing = ET.fromstring(archive.read("xl/drawings/drawing1.xml"))
                self.assertEqual(len(drawing.findall(".//xdr:grpSp", XDR)), 25)  # 24 measured panels + non-temporal mechanism
                text = " ".join(drawing.itertext())
                self.assertIn("runner.prepare_attn", text)
                self.assertIn("CLOCK_MISMATCH", text)
                self.assertIn("ms", text)
                self.assertFalse(any(name.startswith("xl/media/") for name in archive.namelist()))
            document = target.with_suffix(".html").read_text(encoding="utf-8")
            self.assertEqual(document.count('class="swimlane-detail" id='), 12)
            self.assertIn("SYNTHETIC_bs2_s9", document)
            self.assertIn("CLOCK_MISMATCH", document)
            self.assertIn("compute_slot_mappings", document)
            self.assertIn("横向不表示时间", document)
            self.assertIn("互斥子阶段汇总", document)
            self.assertIn("U15全边界来源审计", document)
            self.assertNotIn("Set-Clipboard", document)

    def test_full_report_keeps_eleven_sheets_and_old_sheet_xml_unchanged(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            source = sample_source(directory)
            report = residual_report()
            original = deepcopy(report)
            report_path = directory/"model_architecture_report.json"
            report_path.write_text(json.dumps(report), encoding="utf-8")
            base = renderer.build_workbook(directory/"base.xlsx", report=report_path, report_data=report)
            target = renderer.build_workbook(directory/"with_swimlane.xlsx", report=report_path,
                report_data=report, swimlane_json=source)
            self.assertEqual(report, original)
            self.assertEqual(workbook_names(target), list(renderer.SHEETS))
            with ZipFile(base) as first, ZipFile(target) as second:
                for index in range(1, 11):
                    self.assertEqual(first.read("xl/worksheets/sheet"+str(index)+".xml"),
                                     second.read("xl/worksheets/sheet"+str(index)+".xml"))
                graphics = [ET.fromstring(second.read(name)) for name in second.namelist()
                            if name.startswith("xl/drawings/drawing") and name.endswith(".xml")]
                self.assertTrue(any(root.findall(".//xdr:grpSp", XDR) for root in graphics))
                self.assertTrue(any(root.findall(".//xdr:graphicFrame", XDR) for root in graphics))

    def test_source_json_cannot_be_overwritten_by_xlsx_or_html(self):
        with tempfile.TemporaryDirectory() as temp:
            source = sample_source(Path(temp))
            original = hashlib.sha256(source.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError, "覆盖"):
                renderer.build_swimlane_workbook(source, swimlane_json=source)
            source_html = source.with_suffix(".html")
            source_html.write_bytes(source.read_bytes())
            with self.assertRaisesRegex(ValueError, "覆盖"):
                renderer.build_swimlane_workbook(source_html.with_suffix(".xlsx"), swimlane_json=source_html)
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), original)

    def test_rank_mismatch_does_not_borrow_another_rank(self):
        with tempfile.TemporaryDirectory() as temp:
            source = sample_source(Path(temp))
            target = Path(temp)/"wrong_rank.xlsx"
            with self.assertRaisesRegex(ValueError, "rank"):
                renderer.build_swimlane_workbook(target, swimlane_json=source, rank="dp3")
            self.assertFalse(target.exists())

    def test_main_profile_cli_has_a_json_only_fast_path(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            source = sample_source(directory)
            output = directory/"charts"
            result = subprocess.run([sys.executable, "-B", "-S", str(ROOT/"profile_ep128.py"),
                "--u-swimlane-json", str(source), "--out", str(output)], cwd=ROOT,
                capture_output=True, text=True, encoding="utf-8", errors="replace")
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            self.assertEqual(workbook_names(output/"u_swimlane.xlsx"), ["额外图表统计"])
            self.assertTrue((output/"u_swimlane.html").is_file())

    def test_profile_fast_path_rejects_collecting_and_orphan_report_options(self):
        commands = (["cases.json", "--u-swimlane-json", "saved.json"], ["--u-swimlane-report", "report.json"])
        for args in commands:
            result = subprocess.run([sys.executable, "-B", "-S", str(ROOT/"profile_ep128.py"), *args],
                cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
            self.assertEqual(result.returncode, 2)
            self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()
