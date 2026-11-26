"""Clock diagnostics reuse existing inputs without repairing or sampling."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import u_swimlane_clock_probe as probe


def inputs(directory):
    cases = []
    for bs in (2,16):
        kernel, trace = directory/(str(bs)+".csv"), directory/(str(bs)+".json")
        kernel.write_text("unaltered CSV", encoding="utf-8")
        trace.write_text("unaltered trace", encoding="utf-8")
        cases.append({"name": "bs"+str(bs)+"-s9", "bs": bs, "draft_tokens": 9, "with_stack": True,
                      "acceptance": .55, "inputs": [{"rank": "dp4", "path": kernel.name, "trace_path": trace.name}]})
    config = {"schema_version": 2, "defaults":{"base_ep":16,"expected_ranks":16,"target_ep":128}, "cases": cases}
    manifest = directory/"cases.json"
    manifest.write_text(json.dumps(config), encoding="utf-8-sig")
    return manifest, config


class ClockProbeTests(unittest.TestCase):
    def test_S9_paths_use_host_stack_normalization_and_keep_fixed_rank(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            manifest, config = inputs(directory)
            sources, protected = probe._paths(config, manifest, ["bs2-s9","bs16-s9"], "dp4")
            self.assertEqual([s["S"] for s in sources], [9,9])
            self.assertEqual([s["bs"] for s in sources], [2,16])
            self.assertEqual(sources[0]["trace"], directory/"2.json")
            self.assertIn(directory/"16.csv", protected)

    def test_mismatch_writes_small_diagnostic_and_never_changes_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            manifest, _ = inputs(directory)
            original = {p: p.read_bytes() for p in directory.iterdir()}
            target = directory/"diagnostic.json"
            diagnostics = [{"status":"ALIGNED_X_EVENTS", "anchor_count":3,"max_clock_residual_ns":0},
                           {"status":"CLOCK_MISMATCH", "anchor_count":4,"max_clock_residual_ns":20000000,
                            "worst_anchors":[{"csv_row_index":1,"trace_event_index":7,"matched_key":"kind+duration+ids"}]}]
            with patch.object(probe.core,"read_capture", return_value={"sha256":"test"}) as read, \
                 patch.object(probe,"diagnose_clock",side_effect=diagnostics) as diagnose, \
                 contextlib.redirect_stdout(io.StringIO()):
                code = probe.main([str(manifest),"--out",str(target)])
            self.assertEqual(code,1)
            result = json.loads(target.read_text(encoding="ascii"))
            self.assertEqual(result["cases"][1]["max_clock_residual_ns"],20000000)
            self.assertEqual(diagnose.call_count,2)
            self.assertEqual([c.args[1] for c in read.call_args_list],["dp4","dp4"])
            self.assertLess(target.stat().st_size,65536)
            self.assertTrue(all(p.read_bytes()==data for p,data in original.items()))

    def test_case_error_does_not_hide_other_case(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            manifest, _ = inputs(directory)
            target = directory/"diagnostic.json"
            with patch.object(probe.core,"read_capture", side_effect=[OSError("missing first CSV"),{"sha256":"second"}]), \
                 patch.object(probe,"diagnose_clock", return_value={"status":"CLOCK_UNCONFIRMED"}), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(probe.main([str(manifest),"--out",str(target)]),1)
            self.assertEqual([r["status"] for r in json.loads(target.read_text())["cases"]],["ERROR","CLOCK_UNCONFIRMED"])

    def test_output_cannot_overwrite_selected_or_unselected_sources(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            manifest, _ = inputs(directory)
            for target in (manifest,directory/"2.csv",directory/"2.json",directory/"16.json"):
                with self.subTest(target=target),contextlib.redirect_stderr(io.StringIO()), \
                     patch.object(probe.core,"read_capture",side_effect=AssertionError("no IO after rejection")):
                    original=target.read_bytes()
                    with self.assertRaises(SystemExit) as error:
                        probe.main([str(manifest),"--case","bs2-s9","--out",str(target)])
                    self.assertEqual(error.exception.code,2)
                    self.assertEqual(target.read_bytes(),original)

    def test_manifest_and_case_selection_are_validated(self):
        with tempfile.TemporaryDirectory() as temp:
            directory=Path(temp);manifest,_=inputs(directory)
            for extra in (["--case","unknown"],["--case","bs2-s9","--case","bs2-s9"],["--rank","../dp4"]):
                with self.subTest(extra=extra),contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as error:
                        probe.main([str(manifest),*extra])
                    self.assertEqual(error.exception.code,2)


if __name__ == "__main__":
    unittest.main()
