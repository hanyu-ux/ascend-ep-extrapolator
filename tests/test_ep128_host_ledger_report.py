"""Export checks for complete wall-time ledgers, using synthetic data only."""
import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET
from zipfile import ZipFile

from host_profile_report import PREVIEW_ROWS, _ledger_html, _ledger_overlap_preview, write_reports


def sample_report():
    total = 100_000_000
    items = [("device_attention", "设备 Attention", 30_000_000),
             ("device_overlap", "设备多类别重叠", 20_000_000),
             ("host_only_sync", "仅 Host 同步", 20_000_000),
             ("unobserved", "双方无记录", 30_000_000)]
    rows = [{"key": key, "label": label, "duration_ns": duration,
             "duration_ms": duration / 1_000_000, "share": duration / total,
             "segment_count": 1} for key, label, duration in items]
    ledger = {"device_pid": "device-0", "start_ns": 10 ** 18,
              "end_ns": 10 ** 18 + total, "total_ns": total, "total_ms": 100,
              "closed": True, "closure_error_ns": 0, "breakdown": rows,
              "details": [{"key": "attention+communication", "parent_key": "device_overlap",
                           "label": "Attention 与通信", "components": ["attention", "communication"],
                           "duration_ns": 20_000_000, "duration_ms": 20, "share": .2,
                           "segment_count": 1}],
              "classifications": [{"key": "attention", "label": "注意力", "side": "device"},
                                  {"key": "communication", "label": "通信", "side": "device"}],
              "coverage_states": [{"key": "both", "label": "Host 与设备同时可见", "duration_ns": 10_000_000,
                                   "duration_ms": 10, "share": .1},
                                  {"key": "device_only", "label": "仅设备可见", "duration_ns": 40_000_000,
                                   "duration_ms": 40, "share": .4},
                                  {"key": "host_only", "label": "仅 Host 可见", "duration_ns": 20_000_000,
                                   "duration_ms": 20, "share": .2},
                                  {"key": "neither", "label": "双方无记录", "duration_ns": 30_000_000,
                                   "duration_ms": 30, "share": .3}]}
    hotspot = {"pid": "host", "tid": "thread", "name": "synthetic call", "self_sum_ms": 1,
               "inclusive_sum_ms": 2, "self_unavailable_calls": 0}
    return {"time_ledgers": [ledger], "hotspots": [hotspot], "apis": [hotspot]}


class HostLedgerExportTests(unittest.TestCase):
    def test_primary_exports_preserve_closure_and_device_separation(self):
        report = sample_report()
        other = copy.deepcopy(report["time_ledgers"][0])
        other.update(device_pid=None, capture_label="capture B")
        report["time_ledgers"].append(other)
        with tempfile.TemporaryDirectory() as directory:
            paths = write_reports(report, directory)
            with Path(paths["time_ledger_csv"]).open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 8)
            for pid in ("device-0", ""):
                selected = [row for row in rows if row["device_pid"] == pid]
                self.assertEqual(sum(int(row["duration_ns"]) for row in selected), 100_000_000)
                self.assertEqual(sum(float(row["share"]) for row in selected), 1)
                self.assertEqual({row["start_ns"] for row in selected}, {str(10 ** 18)})
                self.assertEqual({row["closure_error_ns"] for row in selected}, {"0"})
            self.assertEqual({row["capture_label"] for row in rows}, {"", "capture B"})
            restored = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
            self.assertEqual(restored["time_ledgers"], report["time_ledgers"])

    def test_html_primary_precedes_hotspots_and_labels_auxiliary_overlap(self):
        report = sample_report()
        with tempfile.TemporaryDirectory() as directory:
            paths = write_reports(report, directory)
            page = Path(paths["html"]).read_text(encoding="utf-8")
        self.assertLess(page.index('id="time-ledger"'), page.index("线程内热点 self"))
        self.assertIn('id="time-ledger-1"', page)
        self.assertIn("30 ms · 30.000%", page)
        self.assertIn("注意力 + 通信", page)
        self.assertIn("闭合误差 0 ns", page)
        self.assertIn("不能合并相加", page)
        self.assertIn("不是 CPU 或 NPU 利用率", page)

    def test_overlap_preview_is_bounded_but_csv_and_primary_are_complete(self):
        report = sample_report()
        ledger = report["time_ledgers"][0]
        template = ledger["details"][0]
        durations = [350_878] * 56 + [20_000_000 - 350_878 * 56]
        ledger["details"] = [dict(template, key="combination-" + str(i), label="组合 " + str(i),
                                   duration_ns=durations[i], duration_ms=durations[i] / 1_000_000,
                                   share=durations[i] / ledger["total_ns"])
                             for i in range(PREVIEW_ROWS + 7)]
        with tempfile.TemporaryDirectory() as directory:
            paths = write_reports(report, directory, excel=True)
            with Path(paths["time_ledger_details_csv"]).open(encoding="utf-8-sig", newline="") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), PREVIEW_ROWS + 7)
            page = Path(paths["html"]).read_text(encoding="utf-8")
            self.assertIn("50 / 57 项", page)
            self.assertNotIn("组合 56</td>", page)
            self.assertIn("其余7种已识别重叠组合", page)
            with Path(paths["time_ledger_csv"]).open(encoding="utf-8-sig", newline="") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 4)

    def test_truncated_overlap_previews_close_each_parent_in_integer_ns(self):
        report = sample_report()
        ledger = report["time_ledgers"][0]
        ledger["details"] = [
            {"key": parent + str(index), "parent_key": parent, "label": parent + str(index),
             "components": ["attention", "communication"], "duration_ns": 1_000_003 + index,
             "duration_ms": (1_000_003 + index) / 1_000_000,
             "share": (1_000_003 + index) / ledger["total_ns"], "segment_count": index + 1}
            for parent in ("device_overlap", "host_overlap") for index in range(30)]
        parents = ("device_overlap", "host_overlap")
        totals = {parent: sum(item["duration_ns"] for item in ledger["details"]
                              if item["parent_key"] == parent) for parent in parents}
        totals["unobserved"] = ledger["total_ns"] - sum(totals.values())
        ledger["breakdown"] = [{"key": key, "label": key, "duration_ns": duration,
                                "duration_ms": duration / 1_000_000,
                                "share": duration / ledger["total_ns"], "segment_count": 1}
                               for key, duration in totals.items()]
        before = copy.deepcopy(ledger)
        total_count, individual_count, rows = _ledger_overlap_preview(ledger)
        self.assertEqual((total_count, individual_count, len(rows)), (60, 50, 52))
        self.assertEqual(ledger, before)
        for parent in parents:
            expected = sum(item["duration_ns"] for item in ledger["details"] if item["parent_key"] == parent)
            shown = [item for item in rows if item["parent_key"] == parent]
            self.assertEqual(sum(item["duration_ns"] for item in shown), expected)
            aggregate = [item for item in shown if item.get("preview_remaining_count")]
            self.assertEqual(len(aggregate), 1)
            self.assertEqual(aggregate[0]["preview_remaining_count"], 5)
            self.assertEqual(aggregate[0]["share"], aggregate[0]["duration_ns"] / ledger["total_ns"])
        snippet = ET.fromstring(_ledger_html(report))
        visible_detail_rows = snippet.findall(".//details//tbody/tr")
        for parent in parents:
            values = [int(row.findall("td")[-1].text) for row in visible_detail_rows
                      if row.find("td").text.startswith(parent)]
            self.assertEqual(sum(values), totals[parent])
        with tempfile.TemporaryDirectory() as directory:
            paths = write_reports(report, directory, excel=True)
            self.assertEqual(Path(paths["html"]).read_text(encoding="utf-8").count("其余5种已识别重叠组合"), 2)
            with Path(paths["time_ledger_details_csv"]).open(encoding="utf-8-sig", newline="") as handle:
                exported = list(csv.DictReader(handle))
                self.assertEqual(len(exported), 60)
                self.assertFalse(any("preview_remaining" in item["key"] for item in exported))
            with ZipFile(paths["xlsx"]) as archive:
                ns = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
                detail = ET.fromstring(archive.read("xl/worksheets/sheet2.xml"))
                body = [node.text or "" for node in detail.findall(".//s:t", ns)]
                self.assertEqual(sum("其余5种已识别重叠组合" in value for value in body), 2)
                detail_rows = detail.findall("s:sheetData/s:row", ns)
                for parent in parents:
                    durations = []
                    for row in detail_rows:
                        cells = row.findall("s:c", ns)
                        first = cells[0].find("s:is/s:t", ns) if cells else None
                        if first is not None and (first.text or "").startswith(parent):
                            numeric = [cell for cell in cells if cell.get("r").startswith("F")]
                            durations.append(int(numeric[0].find("s:v", ns).text))
                    self.assertEqual(sum(durations), totals[parent])

    def test_primary_chart_never_uses_the_auxiliary_preview_limit(self):
        report = sample_report()
        ledger = report["time_ledgers"][0]
        ledger["breakdown"] = [{"key": "class-" + str(i), "label": "主项 " + str(i),
                                "duration_ns": 1_000_000, "duration_ms": 1,
                                "share": 1 / 57, "segment_count": 1}
                               for i in range(PREVIEW_ROWS + 7)]
        ledger.update(total_ns=57_000_000, total_ms=57, end_ns=ledger["start_ns"] + 57_000_000)
        with tempfile.TemporaryDirectory() as directory:
            paths = write_reports(report, directory, excel=True)
            self.assertIn("主项 56", Path(paths["html"]).read_text(encoding="utf-8"))
            with ZipFile(paths["xlsx"]) as archive:
                ns = {"c": "http://schemas.openxmlformats.org/drawingml/2006/chart"}
                chart = ET.fromstring(archive.read("xl/charts/chart1.xml"))
                values = chart.findall(".//c:numCache/c:pt/c:v", ns)
                self.assertEqual(len(values), 57)
                self.assertEqual(sum(float(node.text) for node in values), 57)

    def test_ledger_csv_output_collision_is_rejected_before_any_write(self):
        report = sample_report()
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "time_ledger.csv"
            source.write_text("original source")
            report["source"] = {"path": str(source)}
            with self.assertRaisesRegex(ValueError, "overwrite the source"):
                write_reports(report, directory)
            self.assertEqual(source.read_text(encoding="utf-8"), "original source")
            self.assertFalse((Path(directory) / "host_profile_report.json").exists())

    def test_xlsx_has_primary_chart_with_share_labels_and_original_sheets(self):
        report = sample_report()
        with tempfile.TemporaryDirectory() as directory:
            paths = write_reports(report, directory, excel=True)
            with ZipFile(paths["xlsx"]) as archive:
                ns = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
                      "c": "http://schemas.openxmlformats.org/drawingml/2006/chart"}
                workbook = ET.fromstring(archive.read("xl/workbook.xml"))
                names = [item.get("name") for item in workbook.findall("s:sheets/s:sheet", ns)]
                self.assertEqual(names[:2], ["主时间账本", "并行与重叠明细"])
                self.assertEqual(names[2:], ["总览", "Host线程", "Host热点", "API与同步", "步骤区间", "设备空档", "来源覆盖"])
                charts = [name for name in archive.namelist() if name.startswith("xl/charts/chart") and name.endswith(".xml")]
                self.assertEqual(len(charts), 3)
                chart = ET.fromstring(archive.read(charts[0]))
                self.assertEqual(chart.find(".//c:barDir", ns).get("val"), "bar")
                refs = [node.text for node in chart.findall(".//c:f", ns)]
                self.assertIn("'主时间账本'!$B$8:$B$11", refs)
                self.assertIn("'主时间账本'!$F$8:$F$11", refs)
                labels = [node.text for node in chart.findall(".//c:strCache/c:pt/c:v", ns)]
                self.assertIn("设备 Attention · 30.000%", labels)
                values = [float(node.text) for node in chart.findall(".//c:numCache/c:pt/c:v", ns)]
                self.assertEqual(values, [30, 20, 20, 30])
                sheet = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
                timestamp_text = [node.text for node in sheet.findall(".//s:t", ns)]
                self.assertTrue(any(str(10 ** 18) in value for value in timestamp_text))

    def test_external_labels_are_literal_and_legacy_reports_still_export(self):
        report = sample_report()
        report["time_ledgers"][0]["capture_label"] = "=1<script>alert(1)</script>"
        report["time_ledgers"][0]["breakdown"][0]["label"] = "<script>bad()</script>"
        with tempfile.TemporaryDirectory() as directory:
            paths = write_reports(report, directory)
            page = Path(paths["html"]).read_text(encoding="utf-8")
            self.assertNotIn("<script>", page)
            self.assertIn("&lt;script&gt;", page)
            with Path(paths["time_ledger_csv"]).open(encoding="utf-8-sig", newline="") as handle:
                self.assertTrue(next(csv.DictReader(handle))["capture_label"].startswith("'=1"))
        del report["time_ledgers"]
        self.assertEqual(_ledger_html(report), "")
        with tempfile.TemporaryDirectory() as directory:
            paths = write_reports(report, directory, excel=True)
            self.assertNotIn("time_ledger_csv", paths)
            with ZipFile(paths["xlsx"]) as archive:
                ns = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
                workbook = ET.fromstring(archive.read("xl/workbook.xml"))
                self.assertEqual(len(workbook.findall("s:sheets/s:sheet", ns)), 7)


if __name__ == "__main__":
    unittest.main()
