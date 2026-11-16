"""Long input paths: lexical Windows cases and real CPU-only file reads.

The Windows integration branch rejects unprefixed long paths even when the
machine globally enables long paths. Linux exercises actual deep files, while
the pure Windows cases verify namespace conversion without a Windows API.
"""
import hashlib
import json
import ntpath
import os
import shutil
import sys
import tempfile
import unittest
from contextlib import contextmanager, nullcontext
from pathlib import Path, PureWindowsPath
from unittest import mock

import profile_ep128 as p


class _LexicalWindowsPath(PureWindowsPath):
    """Require namespace conversion before the mocked filesystem resolution."""

    def resolve(self, strict=False):
        if not str(self).startswith(("\\\\?\\", "\\\\.\\")):
            raise AssertionError("resolved before converting the Windows I/O path")
        return self


@contextmanager
def _legacy_windows_limits():
    """Make the native Windows regression independent of machine policy."""
    original_stat, original_open = Path.stat, Path.open
    original_resolve, original_scandir = Path.resolve, os.scandir

    def check(value):
        if isinstance(value, int):
            return
        raw = os.fsdecode(value)
        if (not raw.startswith(("\\\\?\\", "\\\\.\\"))
                and len(ntpath.abspath(raw).encode("utf-16-le")) // 2 >= 260):
            raise FileNotFoundError(3, "synthetic legacy Windows path limit", raw)

    def guarded_stat(path, *args, **kwargs):
        check(path)
        return original_stat(path, *args, **kwargs)

    def guarded_open(path, *args, **kwargs):
        check(path)
        return original_open(path, *args, **kwargs)

    def guarded_resolve(path, *args, **kwargs):
        check(path)
        return original_resolve(path, *args, **kwargs)

    def guarded_scandir(path):
        check(path)
        return original_scandir(path)

    with mock.patch.object(Path, "stat", guarded_stat), \
            mock.patch.object(Path, "open", guarded_open), \
            mock.patch.object(Path, "resolve", guarded_resolve), \
            mock.patch.object(os, "scandir", guarded_scandir):
        yield


class WindowsPathConversionTests(unittest.TestCase):
    def test_drive_unc_and_existing_namespaces(self):
        examples = (
            (r"D:\技术开发部\采样\kernel_details.csv",
             r"\\?\D:\技术开发部\采样\kernel_details.csv"),
            (r"\\server\share\技术开发部\kernel_details.csv",
             r"\\?\UNC\server\share\技术开发部\kernel_details.csv"),
            (r"\\?\D:\技术开发部\kernel_details.csv",
             r"\\?\D:\技术开发部\kernel_details.csv"),
            (r"\\?\UNC\server\share\kernel_details.csv",
             r"\\?\UNC\server\share\kernel_details.csv"),
            (r"\\.\D:\技术开发部\kernel_details.csv",
             r"\\.\D:\技术开发部\kernel_details.csv"),
            (r"D:\技术开发部\采样\..\run\.\kernel_details.csv",
             r"\\?\D:\技术开发部\run\kernel_details.csv"),
        )
        with mock.patch.object(p.sys, "platform", "win32"), \
                mock.patch.object(p, "Path", PureWindowsPath):
            for supplied, expected in examples:
                with self.subTest(path=supplied):
                    converted = p._io_path(supplied)
                    self.assertEqual(str(converted), expected)
                    self.assertEqual(p._io_path(converted), converted)

    def test_relative_path_is_absolute_before_namespace_conversion(self):
        with mock.patch.object(p.sys, "platform", "win32"), \
                mock.patch.object(p, "Path", PureWindowsPath), \
                mock.patch.object(ntpath, "abspath", return_value=
                                  r"D:\工作区\采样\kernel_details.csv") as absolute:
            result = p._io_path(r".\采样\kernel_details.csv")
        self.assertEqual(str(result), r"\\?\D:\工作区\采样\kernel_details.csv")
        self.assertEqual(absolute.call_count, 1)

    def test_report_paths_keep_ordinary_drive_and_unc_names(self):
        examples = (
            (r"D:\技术开发部\kernel_details.csv", r"D:\技术开发部\kernel_details.csv"),
            (r"\\?\D:\技术开发部\kernel_details.csv", r"D:\技术开发部\kernel_details.csv"),
            (r"\\server\share\采样\kernel_details.csv",
             r"\\server\share\采样\kernel_details.csv"),
            (r"\\?\UNC\server\share\采样\kernel_details.csv",
             r"\\server\share\采样\kernel_details.csv"),
            (r"\\?\unc\server\share\采样\kernel_details.csv",
             r"\\server\share\采样\kernel_details.csv"),
        )
        with mock.patch.object(p.sys, "platform", "win32"), \
                mock.patch.object(p, "Path", _LexicalWindowsPath):
            for supplied, expected in examples:
                with self.subTest(path=supplied):
                    self.assertEqual(str(p._source_path(supplied)), expected)

    def test_non_windows_paths_keep_native_semantics(self):
        supplied = Path("采样") / ".." / "kernel_details.csv"
        with mock.patch.object(p.sys, "platform", "linux"):
            self.assertEqual(p._io_path(supplied), supplied)
            self.assertEqual(p._source_path(supplied), supplied.resolve())


class DeepInputPathTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="ep128-long-input-")).resolve()
        self.addCleanup(lambda: shutil.rmtree(p._io_path(self.root)))
        self.directory = self.root.joinpath(
            "技术开发部-" + "a" * 55,
            "10t_e224_dp16_ep16_bs2_s15_api2_accrate0.55_decode_1115",
            "10t_e224_dp16_ep16_bs2_s15_api2_accrate0.55_decode_1115",
            "dp10_pp0_tp0_dcp0_ep10_rank0_613579_20261115191035515_ascend_pt",
            "ASCEND_PROFILER_OUTPUT")
        self.csv = self.directory / "kernel_details.csv"
        self.trace = self.directory / "trace_view.json"
        self.assertLess(len(str(self.root)), 260)
        self.assertGreater(len(str(self.csv)), 260)
        p._io_path(self.directory).mkdir(parents=True)
        self.csv_bytes = ("Name,Start Time(us),Duration(us),Device_id\n"
                          "GroupedMatmul,1.001,2.5,0\n").encode("utf-8-sig")
        p._io_path(self.csv).write_bytes(self.csv_bytes)
        self.events = [{"name": "中文采样", "ph": "X", "ts": 1, "dur": 2}]
        self.trace_bytes = json.dumps({"traceEvents": self.events}, ensure_ascii=False).encode("utf-8")
        p._io_path(self.trace).write_bytes(self.trace_bytes)

    def test_short_root_discovers_and_reads_long_csv_and_trace(self):
        limits = _legacy_windows_limits() if sys.platform == "win32" else nullcontext()
        with limits:
            if sys.platform == "win32":
                with self.assertRaises(FileNotFoundError):
                    self.csv.open("rb")
            files, names = p.case_inputs({"path": str(self.root)}, self.root)
            direct = p.find_files(str(self.csv), self.root)
            capture = p.read_capture(files[0], names[0])
            reread = p.read_capture(capture["path"], names[0])
            trace_stats = {}
            events = list(p.residual_iter_trace_events(self.trace, chunk_size=7, stats=trace_stats))
        self.assertEqual(files, [self.csv])
        self.assertEqual(direct, files)
        self.assertEqual(names, ["dp10"])
        self.assertEqual(capture["path"], str(self.csv))
        self.assertEqual(capture["sha256"], hashlib.sha256(self.csv_bytes).hexdigest())
        self.assertEqual(reread["sha256"], capture["sha256"])
        self.assertEqual(len(capture["tasks"]), 1)
        self.assertEqual(capture["tasks"][0].end_ns - capture["tasks"][0].start_ns, 2500)
        self.assertEqual(events, self.events)
        self.assertEqual(trace_stats["sha256"], hashlib.sha256(self.trace_bytes).hexdigest())
        self.assertEqual(trace_stats["bytes_read"], len(self.trace_bytes))


if __name__ == "__main__":
    unittest.main()
