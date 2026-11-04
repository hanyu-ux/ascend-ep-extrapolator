"""The public CLI has one layer-mean workflow, with no experimental modes."""
import contextlib
import io
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock

import profile_ep128 as core


HERE = Path(__file__).resolve().parent
REMOVED_OPTIONS = (
    ("--source-point",),
    ("--assumption-scenarios",),
    ("--measurement-only",),
    ("--from-report", "not-a-cases-manifest.json"),
    ("--export-ownership-template",),
    ("--router-fraction", "0.1"),
    ("--routed-mlp-fraction", "0.5"),
    ("--added-layer-extra-ms", "0.01"),
    ("--nonlayer-delta-ms", "0.2"),
)


class SingleWorkflowCLIContractTests(unittest.TestCase):
    def test_help_lists_supported_entrypoints_and_no_removed_modes(self):
        process = subprocess.run([sys.executable, "-S", str(HERE / "profile_ep128.py"), "--help"],
                                 cwd=HERE, capture_output=True, text=True, timeout=20)
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        for flag in ("--out", "--excel", "--comm-reference", "--list-inputs", "--inventory", "--version"):
            self.assertIn(flag, process.stdout)
        for arguments in REMOVED_OPTIONS:
            self.assertNotIn(arguments[0], process.stdout)

    def test_removed_options_are_rejected_before_native_csv_loading(self):
        for arguments in REMOVED_OPTIONS:
            with self.subTest(option=arguments[0]), \
                    mock.patch.object(sys, "argv", [str(HERE / "profile_ep128.py"),
                                                    "not-an-existing-cases-file.json", *arguments]), \
                    mock.patch.object(core, "read_capture", side_effect=AssertionError("must not read native CSV")) as reader, \
                    contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as stderr:
                with self.assertRaises(SystemExit) as caught:
                    core.main()
                self.assertEqual(caught.exception.code, 2)
                self.assertIn(arguments[0], stderr.getvalue())
                reader.assert_not_called()

    def test_from_report_is_not_a_second_cli_entrypoint(self):
        process = subprocess.run([sys.executable, "-S", str(HERE / "profile_ep128.py"),
                                  "--from-report", "not-an-existing-report.json"],
                                 cwd=HERE, capture_output=True, text=True, timeout=20)
        self.assertEqual(process.returncode, 2, process.stdout + process.stderr)
        self.assertIn("--from-report", process.stderr)
        self.assertNotIn("读取已有核心JSON", process.stdout)


if __name__ == "__main__":
    unittest.main()
