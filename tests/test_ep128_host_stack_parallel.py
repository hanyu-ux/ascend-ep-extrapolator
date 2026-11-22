"""Case-level stack diagnostics preserve evidence under Windows spawn."""

from copy import deepcopy
from concurrent.futures import Future
from contextlib import redirect_stdout
import io
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import pickle

import test_ep128_host_stack_probe as stack_fixtures
import test_ep128_parallel as parallel_fixtures


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "u_host_stack_probe.py"


class HostStackParallelCLITests(unittest.TestCase):
    def invoke(self, manifest, output, workers):
        return subprocess.run(
            [sys.executable, "-B", "-S", str(TOOL), str(manifest), "--rank", "dp4",
             "--samples", "1", "--workers", str(workers), "--out", str(output)],
            cwd=manifest.parent, text=True, capture_output=True, encoding="utf-8", timeout=90,
            env={**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"})

    def semantic(self, value, output):
        if isinstance(value, dict):
            return {key: self.semantic(item, output) for key, item in value.items()}
        if isinstance(value, list):
            return [self.semantic(item, output) for item in value]
        if isinstance(value, str):
            return parallel_fixtures.ParallelCaseTests().normalize_text(value, output)
        return value

    def test_spawn_matches_serial_evidence_preserves_case_order_and_source_files(self):
        helper = stack_fixtures.HostStackProbeCLITests()
        with tempfile.TemporaryDirectory(prefix="ep128 并行 stack ") as temporary:
            base = Path(temporary)
            first, _ = helper.stack_capture(base / "source first" / "dp4", 8)
            second, _ = helper.stack_capture(base / "source second" / "dp4", 16)
            manifest = helper.config(base, [("ZETA_FIRST", 8, first), ("ALPHA_SECOND", 16, second)])
            config = json.loads(manifest.read_text(encoding="utf-8-sig"))
            broken = deepcopy(config["cases"][0])
            broken.update(name="MISSING_SELECTED_RANK", inputs=[{"rank": "dp0", "path": str(first)}])
            config["cases"].insert(1, broken)
            manifest.write_text(json.dumps(config), encoding="utf-8-sig")
            sources = {path: hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in base.rglob("*") if path.is_file()}

            serial_dir, spawn_dir = base / "serial output", base / "spawn output"
            serial = self.invoke(manifest, serial_dir, 1)
            spawned = self.invoke(manifest, spawn_dir, 2)
            self.assertEqual(serial.returncode, 1, serial.stdout + serial.stderr)
            self.assertEqual(spawned.returncode, 1, spawned.stdout + spawned.stderr)
            full_name, share_name = "u_host_stack_full.json", "u_host_stack_share.json"
            for name in (full_name, share_name):
                left = json.loads((serial_dir / name).read_text(encoding="utf-8"))
                right = json.loads((spawn_dir / name).read_text(encoding="utf-8"))
                self.assertEqual(self.semantic({k: v for k, v in left.items() if k != "execution"}, serial_dir),
                                 self.semantic({k: v for k, v in right.items() if k != "execution"}, spawn_dir), name)
                self.assertEqual([row["case"] for row in right["cases"]],
                                 ["ZETA_FIRST", "MISSING_SELECTED_RANK", "ALPHA_SECOND"])
                self.assertEqual(right["completed_cases"], 3)
                failed = right["cases"][1]
                self.assertEqual(failed["status"], "ERROR")
                self.assertIn("dp4", failed["error"])
                for good in (right["cases"][0], right["cases"][2]):
                    self.assertIsNone(good["error"])
                    self.assertGreater(good["counts"]["python_function_events"], 0)
                    summary = next(row for row in good["window_summary"]
                                   if row["kind"] == "prepare_between_host_calls")
                    self.assertEqual(summary["window_mean_ns"], 8_000)
                self.assertEqual(right["execution"]["requested_workers"], 2)
                self.assertEqual(right["execution"]["workers"], 2)
                self.assertEqual(right["execution"]["parallelism"], "case")
            self.assertLessEqual((spawn_dir / share_name).stat().st_size, 65536)

            left_csvs = {p.relative_to(serial_dir): p for p in (serial_dir / "stack_details").rglob("*.csv")}
            right_csvs = {p.relative_to(spawn_dir): p for p in (spawn_dir / "stack_details").rglob("*.csv")}
            self.assertTrue(left_csvs)
            self.assertEqual(set(left_csvs), set(right_csvs))
            for relative, path in left_csvs.items():
                with self.subTest(csv=str(relative)):
                    self.assertEqual(self.semantic(path.read_text(encoding="utf-8-sig"), serial_dir),
                                     self.semantic(right_csvs[relative].read_text(encoding="utf-8-sig"), spawn_dir))
            self.assertEqual({path: hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}, sources)

    def test_zero_workers_is_rejected_before_creating_outputs(self):
        with tempfile.TemporaryDirectory(prefix="ep128 stack workers invalid ") as temporary:
            base = Path(temporary)
            output = base / "must not exist"
            result = self.invoke(base / "absent cases.json", output, 0)
            self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
            self.assertIn("workers", result.stderr)
            self.assertFalse(output.exists())


class StackWorkerTransferTests(unittest.TestCase):
    def test_worker_transfers_small_spool_identity_instead_of_full_function_evidence(self):
        tool = stack_fixtures.load_tool()
        with tempfile.TemporaryDirectory(prefix="ep128 stack IPC ") as temporary:
            base = Path(temporary)
            output = base / "diagnostic output"
            output.mkdir()
            spool = base / "private worker spool"
            spool.mkdir()
            evidence = {"case": "../case name", "rank": "dp4", "status": "PARTIAL", "error": None,
                        "all_windows": [{"evidence": "x" * 1_000_000}]}
            with patch.object(tool, "analyze_case", return_value=evidence), redirect_stdout(io.StringIO()):
                result = tool._case_worker(3, {"name": "../case name"}, {}, base, "dp4", output, 1, spool)
            self.assertEqual(result[0], 3)
            self.assertLess(len(pickle.dumps(result)), 1024)
            path = Path(result[1])
            self.assertEqual(path.parent.resolve(), spool.resolve())
            self.assertTrue(path.is_file())
            with path.open("rb") as stream:
                content = pickle.load(stream)
            self.assertEqual(content["index"], 3)
            self.assertEqual(content["entry"]["case"], evidence["case"])
            self.assertEqual(len(content["entry"]["all_windows"][0]["evidence"]), 1_000_000)

    def test_spawn_pool_yields_reversed_completion_and_does_not_retry_failed_future(self):
        tool = stack_fixtures.load_tool()
        submissions, spool_paths = [], []

        class InlinePool:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False
            def submit(self, function, *arguments):
                future = Future()
                index = arguments[0]
                submissions.append(index)
                spool_paths.append(Path(arguments[-1]))
                if index == 1:
                    future.set_exception(RuntimeError("worker failed after submission"))
                else:
                    future.set_result(function(*arguments))
                return future

        def complete_reversed(futures):
            # Both independent jobs must be in flight before collection begins.
            self.assertEqual(submissions, [0, 1])
            return iter(reversed(list(futures)))

        def result(item, config, base, rank, out, samples):
            return {"case": item["name"], "rank": rank, "status": "PARTIAL", "error": None}

        with tempfile.TemporaryDirectory(prefix="ep128 stack schedule ") as temporary:
            base = Path(temporary)
            selected = [{"name": "FIRST"}, {"name": "SECOND"}]
            with patch.object(tool, "ProcessPoolExecutor", side_effect=InlinePool) as pools, \
                    patch.object(tool, "as_completed", side_effect=complete_reversed), \
                    patch.object(tool, "analyze_case", side_effect=result) as analyzer, \
                    redirect_stdout(io.StringIO()):
                iterator = tool.run_cases(selected, {}, base, "dp4", base, 1, 2)
                first_index, first = next(iterator)
                self.assertEqual(first_index, 1)
                self.assertEqual(first["status"], "ERROR")
                self.assertIn("worker failed after submission", first["error"])
                second_index, second = next(iterator)
                self.assertEqual((second_index, second["case"]), (0, "FIRST"))
                with self.assertRaises(StopIteration):
                    next(iterator)
            self.assertEqual(pools.call_count, 1)
            self.assertEqual(pools.call_args.kwargs["max_workers"], 2)
            self.assertEqual(pools.call_args.kwargs["mp_context"].get_start_method(), "spawn")
            self.assertEqual(analyzer.call_count, 1)
            self.assertTrue(spool_paths)
            self.assertTrue(all(not path.exists() for path in spool_paths))

    def test_parent_checkpoints_each_completed_case_and_restores_selected_order(self):
        tool = stack_fixtures.load_tool()
        with tempfile.TemporaryDirectory(prefix="ep128 stack checkpoints ") as temporary:
            base = Path(temporary)
            manifest = base / "cases.json"
            manifest.write_text(json.dumps({"schema_version": 2, "defaults": {"with_stack": False},
                "cases": [{"name": "FIRST", "bs": 8, "draft_tokens": 7, "path": "missing-first"},
                          {"name": "SECOND", "bs": 16, "draft_tokens": 7, "path": "missing-second"}]}),
                encoding="utf-8")
            output = base / "output"
            snapshots = []
            original = tool.core.atomic_json

            def entry(name):
                return {"case": name, "rank": "dp4", "status": "PARTIAL", "error": None,
                        "phase_summary": [], "window_summary": [], "windows": []}

            def save(path, value, *args, **kwargs):
                if Path(path).name == "u_host_stack_full.json":
                    snapshots.append(deepcopy(value))
                return original(path, value, *args, **kwargs)

            with patch.object(tool, "run_cases", return_value=iter([(1, entry("SECOND")), (0, entry("FIRST"))])), \
                    patch.object(tool.core, "atomic_json", side_effect=save), redirect_stdout(io.StringIO()):
                code = tool.main([str(manifest), "--workers", "2", "--out", str(output)])
            self.assertEqual(code, 1)
            populated = [row for row in snapshots if row["completed_cases"]]
            self.assertEqual(populated[0]["completed_cases"], 1)
            self.assertEqual([row["case"] for row in populated[0]["cases"]], ["SECOND"])
            self.assertEqual(populated[-1]["completed_cases"], 2)
            self.assertEqual([row["case"] for row in populated[-1]["cases"]], ["FIRST", "SECOND"])
            saved = json.loads((output / "u_host_stack_share.json").read_text(encoding="utf-8"))
            self.assertEqual([row["case"] for row in saved["cases"]], ["FIRST", "SECOND"])


if __name__ == "__main__":
    unittest.main()
