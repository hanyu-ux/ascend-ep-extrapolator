"""Copy/paste sharing reads only a saved full JSON and preserves packet bytes."""

import base64
from copy import deepcopy
import csv
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile
import unittest

import profile_ep128 as core
import test_ep128_swimlane_probe as fixtures


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "u_swimlane_share.py"
sys.path.insert(0, str(TOOL.parent))
spec = importlib.util.spec_from_file_location("swimlane_share_cli_tests", TOOL)
share = importlib.util.module_from_spec(spec)
spec.loader.exec_module(share)


def clock_error_payload(directory):
    case, capture, measured, _, _, _ = fixtures.input_fixture(directory)
    rounds, anchors = share.probe.round_coordinates(measured, capture, case)
    return {"schema": "u_swimlane_v2", "profile_version": core.PROFILE_VERSION,
            "rank": "dp4", "planned_cases": 1, "completed_cases": 1,
            "cases": [{"case": "bs8-s15", "rank": "dp4", "bs": 8, "S": 15,
                       "status": "ERROR", "error": "ValueError: CLOCK_MISMATCH",
                       "rounds": rounds, "host_events": [], "device_anchors": anchors,
                       "kernel_path": "not-present/kernel_details.csv",
                       "trace_path": "not-present/trace_view.json",
                       "trace": {"status": "CLOCK_MISMATCH", "anchor_count": 8,
                                 "reason": "inconsistent clock offsets", "clock_offset_ns": "7",
                                 "max_clock_residual_ns": 987654321}}]}


class SourceJSONTests(unittest.TestCase):
    def test_read_source_preserves_bytes_and_reports_exact_original_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "full.json"
            payload = {"schema": "u_swimlane_v2", "cases": [{"case": "中文路径"}]}
            p.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8-sig")
            before = p.read_bytes()
            data, info, stamp = share.read_source(p)
            self.assertEqual(data, payload)
            self.assertEqual(info, {"schema": "u_swimlane_v2", "bytes": len(before),
                                    "sha256": hashlib.sha256(before).hexdigest()})
            self.assertEqual(stamp, (p.stat().st_size, p.stat().st_mtime_ns))
            self.assertEqual(p.read_bytes(), before)

    def test_manifest_text_part_and_empty_cases_have_clear_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "input.json"
            for bad in ({"schema_version": 2, "cases": [{}]},
                        {"schema": "u_swimlane_gzip_base64_v1", "part": 1},
                        {"schema": "u_swimlane_v3", "cases": []}):
                with self.subTest(bad=bad):
                    p.write_text(json.dumps(bad), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "existing full|no cases"):
                        share.read_source(p)


class TextPacketTests(unittest.TestCase):
    def test_every_selected_byte_survives_multipart_with_exact_hashes(self):
        rng = random.Random(55)
        noise = base64.b64encode(bytes(rng.randrange(256) for _ in range(100000))).decode("ascii")
        payload = {"schema": "u_swimlane_ppt_evidence_v1", "cases": [{
            "case": "twelve-grid", "round": {"start_ns": 1790000000000000003,
            "end_ns": 1790000000000099999, "relative_start_ns": -321},
            "children": [{"event_index": i, "name": "完整子函数", "duration_ns": i + 1}
                         for i in range(700)], "nonrepeating_fixture": noise}]}
        original = deepcopy(payload)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            package = share.write_packet(payload, out, "dp4")
            self.assertGreater(package["part_count"], 1)
            parts = []
            for name in package["parts"]:
                blob = (out / name).read_bytes()
                self.assertLessEqual(len(blob), 65536)
                parts.append(json.loads(blob))
            self.assertEqual([p["part"] for p in parts], list(range(1, len(parts) + 1)))
            self.assertTrue(all(p["parts"] == len(parts) for p in parts))
            packed = base64.b64decode("".join(p["data"] for p in parts))
            raw = gzip.decompress(packed)
            self.assertEqual(hashlib.sha256(raw).hexdigest(), package["raw_sha256"])
            self.assertEqual(hashlib.sha256(packed).hexdigest(), package["gzip_sha256"])
            self.assertEqual(len(raw), package["raw_bytes"])
            self.assertEqual((out / "u_swimlane_ppt_dp4.json").read_bytes(), raw)
            self.assertEqual(json.loads(raw), share.probe.wire(payload))
            self.assertEqual(json.loads(raw)["cases"][0]["round"]["start_ns"], "1790000000000000003")
            self.assertEqual(int(json.loads(raw)["cases"][0]["round"]["relative_start_ns"]), -321)
            self.assertEqual(len(json.loads(raw)["cases"][0]["children"]), 700)
            self.assertFalse(package["truncated"])
            self.assertFalse(package["clipboard_required"])
            self.assertEqual(payload, original)

    def test_repeated_packets_do_not_delete_or_mix_old_parts(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            first = share.write_packet({"cases": [{"case": "first"}]}, out, "dp4")
            saved = {name: (out / name).read_bytes() for name in first["parts"]}
            second = share.write_packet({"cases": [{"case": "second"}]}, out, "dp4")
            self.assertTrue(set(first["parts"]).isdisjoint(second["parts"]))
            manifest = json.loads((out / "u_swimlane_ppt_dp4_package.json").read_bytes())
            self.assertEqual(manifest["parts"], second["parts"])
            for name, blob in saved.items():
                self.assertEqual((out / name).read_bytes(), blob)

    def test_long_windows_output_paths_preserve_manifest_json_and_parts(self):
        with tempfile.TemporaryDirectory(prefix="swimshare long ") as tmp:
            try:
                out = Path(tmp)
                for i in range(7):
                    out = out / ("long-share-path-" + str(i) + "-" + "x" * 28)
                self.assertGreater(len(str(out)), 260)
                core._io_path(out).mkdir(parents=True)
                payload = {"cases": [{"case": "long", "start_ns": 1790000000000000003}]}
                package = share.write_packet(payload, out, "dp4")
                self.assertEqual(json.loads(core._io_path(out / "u_swimlane_ppt_dp4.json").read_bytes()),
                                 share.probe.wire(payload))
                self.assertTrue(all(core._io_path(out / name).is_file() for name in package["parts"]))
            finally:
                # The checked absolute directory is test-owned. Extended paths
                # are needed for cleanup as well as writes on Windows.
                target = Path(tmp).resolve()
                self.assertEqual(target.parent, Path(tempfile.gettempdir()).resolve())
                self.assertTrue(target.name.startswith("swimshare long "))
                shutil.rmtree(core._io_path(target))


class ShareCLITests(unittest.TestCase):
    def cli(self, *args):
        return subprocess.run([sys.executable, "-B", "-S", str(TOOL), *map(str, args)],
                              cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")

    def test_clock_error_case_is_packaged_without_opening_missing_capture_sources(self):
        with tempfile.TemporaryDirectory(prefix="swim share cli ") as tmp:
            root = Path(tmp)
            capture_dir = root / "fixture"; capture_dir.mkdir()
            payload = clock_error_payload(capture_dir)
            # Only the full JSON survives. CSV/trace references are deliberately
            # unavailable and the fixture's files are removed before invoking CLI.
            shutil.rmtree(capture_dir)
            source = root / "full.json"
            source.write_text(json.dumps(payload), encoding="utf-8-sig")
            original = source.read_bytes()
            result = self.cli(source, "--expected-cases", 1, "--out", root / "share")
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("text parts:", result.stdout)
            report = json.loads((root / "share/u_swimlane_ppt_dp4.json").read_bytes())
            self.assertEqual(len(report["cases"]), 1)
            self.assertEqual(report["cases"][0]["status"], "ERROR")
            self.assertNotIn("share_error", report["cases"][0])
            self.assertEqual(len(report["cases"][0]["rounds"]), 1)
            self.assertEqual(report["cases"][0]["trace"], payload["cases"][0]["trace"])
            self.assertIn("CLOCK_MISMATCH", json.dumps(report))
            self.assertEqual(source.read_bytes(), original)
            self.assertTrue((root / "share/u_swimlane_ppt_dp4_package.json").is_file())
            with (root / "share/u_swimlane_ppt_dp4_status.csv").open(encoding="utf-8-sig", newline="") as handle:
                status = list(csv.DictReader(handle))
            self.assertEqual(status[0]["status"], "ERROR")
            self.assertEqual(status[0]["trace_status"], "CLOCK_MISMATCH")
            self.assertFalse(list((root / "share").glob("*.zip")))

    def test_wrong_case_count_does_not_create_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "full.json"
            source.write_text(json.dumps({"schema": "u_swimlane_v2", "cases": [{"case": "one"}]}))
            result = self.cli(source, "--expected-cases", 12, "--out", root / "out")
            self.assertEqual(result.returncode, 2)
            self.assertIn("Expected 12 cases", result.stderr)
            self.assertNotIn("Traceback", result.stderr)
            self.assertFalse((root / "out").exists())

    def test_missing_file_or_wrong_schema_has_clear_cli_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            missing = self.cli(root / "missing.json", "--out", root / "out")
            self.assertEqual(missing.returncode, 2)
            self.assertNotIn("Traceback", missing.stderr)
            source = root / "cases.json"
            source.write_text(json.dumps({"schema_version": 2, "cases": [{}]}))
            wrong = self.cli(source, "--out", root / "out")
            self.assertEqual(wrong.returncode, 2)
            self.assertIn("existing full", wrong.stderr)
            self.assertNotIn("Traceback", wrong.stderr)

    def test_output_cannot_overwrite_source_json_or_parts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidates = [root / "u_swimlane_ppt_dp4.json",
                          root / "u_swimlane_ppt_dp4_parts_abcdef" / "part001.txt"]
            for source in candidates:
                with self.subTest(source=source):
                    source.parent.mkdir(parents=True, exist_ok=True)
                    source.write_text("do not overwrite", encoding="utf-8")
                    original = source.read_bytes()
                    result = self.cli(source, "--out", root)
                    self.assertEqual(result.returncode, 2)
                    self.assertIn("overwrite", result.stderr)
                    self.assertEqual(source.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
