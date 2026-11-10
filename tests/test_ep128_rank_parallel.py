"""Rank scheduling must preserve complete reports and use one spawn pool."""
import copy
from concurrent.futures import Future
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import textwrap
import unittest
from unittest import mock

import profile_ep128 as core
import test_ep128_parallel as case_parallel
from test_ep128_v2_measure import MODEL, fixture

HERE = Path(__file__).resolve().parents[1]


def rank_manifest(directory, *, case_count=2):
    """Each global rank has independent device metadata and native PMU rows."""
    document = json.loads((HERE / "profile_ep128_cases.example.json").read_text())
    document["sample_model"] = copy.deepcopy(MODEL)
    document["communication_reference"] = str(HERE / "examples/communication-reference.synthetic.csv")
    document["cases"] = []
    for case_number, name in enumerate(("ZETA_FIRST", "ALPHA_SECOND")[:case_count]):
        item = {"name": name, "bs": 32, "draft_tokens": 7,
                "synthetic_acceptance_length": 5.2 - case_number * .5, "inputs": []}
        for rank_number in (1, 0):
            source = directory / (name + "-dp" + str(rank_number))
            source.mkdir()
            device = str(case_number * 2 + rank_number)
            def change(rows, records):
                for row in rows:
                    row["Device_id"] = device
                    row["aic_mac_ratio"] = ".95" if "GroupedMatmul" in row["Name"] else "0"
                    row["aic_mte2_ratio"] = ".25"
                    row["aic_mte3_ratio"] = "0"
                    row["aicore_time(us)"] = row["Duration(us)"]
            capture, _ = fixture(source, changes=change)
            item["inputs"].append({"rank": "dp" + str(rank_number), "path": capture["path"]})
        document["cases"].append(item)
    path = directory / "cases.json"
    path.write_text(json.dumps(document))
    return path


class RankParallelTests(unittest.TestCase):
    def case_jobs(self, directory, *, case_count=2):
        document = json.loads(rank_manifest(directory, case_count=case_count).read_text())
        reference = core.read_reference(document["communication_reference"])
        jobs = []
        for index, item in enumerate(document["cases"]):
            case = core.normalize_case(document["defaults"], item, directory)
            case["_step_residual_dir"] = str(directory / ("details-" + str(index)))
            jobs.append((index, case, document["sample_model"], document["target_model"], directory, reference))
        return jobs

    def invoke(self, config, output, mode, workers):
        return subprocess.run([sys.executable, "-B", "-S", str(HERE / "profile_ep128.py"),
            str(config), "--out", str(output), "--parallel", mode, "--workers", str(workers)],
            text=True, capture_output=True, timeout=90)

    def assert_same_reports(self, serial_dir, rank_dir, *, exclude_failed_details=False):
        helper = case_parallel.ParallelCaseTests()
        left = json.loads((serial_dir / "model_architecture_report.json").read_text())
        right = json.loads((rank_dir / "model_architecture_report.json").read_text())
        a, b = helper.normalize_output(left, serial_dir), helper.normalize_output(right, rank_dir)
        if a != b:
            self.fail("Case/rank report fields differ:\n" + helper.report_differences(a, b))
        serial_csvs = {path.relative_to(serial_dir): path for path in serial_dir.rglob("*.csv")}
        rank_csvs = {path.relative_to(rank_dir): path for path in rank_dir.rglob("*.csv")}
        if exclude_failed_details:
            # Eager rank workers can finish diagnostics before a sibling's
            # failure is known. These retained failed-case artifacts are not
            # successful results. Keep all report fields, top-level CSVs and
            # every successful case's detail CSV under strict comparison.
            successful = {case["name"] for case in right["cases"]}
            prefixes = tuple(failure["case"] + "-" for failure in right["failures"]
                             if failure["case"] not in successful)
            def retained(relative):
                return not (prefixes and len(relative.parts) > 2
                            and relative.parts[0] == "step_residual_details"
                            and relative.parts[1].startswith(prefixes))
            serial_csvs = {key: value for key, value in serial_csvs.items() if retained(key)}
            rank_csvs = {key: value for key, value in rank_csvs.items() if retained(key)}
        self.assertTrue(serial_csvs)
        self.assertEqual(set(serial_csvs), set(rank_csvs))
        for relative, path in serial_csvs.items():
            with self.subTest(csv=str(relative)):
                self.assertEqual(helper.normalize_text(path.read_text(), serial_dir),
                                 helper.normalize_text(rank_csvs[relative].read_text(), rank_dir))
        return right

    def test_two_cases_two_ranks_match_serial_case_json_and_every_csv(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            config = rank_manifest(directory)
            serial_dir, rank_dir = directory / "serial", directory / "rank"
            serial = self.invoke(config, serial_dir, "case", 1)
            ranked = self.invoke(config, rank_dir, "rank", 2)
            self.assertEqual(serial.returncode, 0, serial.stdout + serial.stderr)
            self.assertEqual(ranked.returncode, 0, ranked.stdout + ranked.stderr)
            report = self.assert_same_reports(serial_dir, rank_dir)
            self.assertEqual([case["name"] for case in report["cases"]], ["ZETA_FIRST", "ALPHA_SECOND"])
            for case in report["cases"]:
                self.assertEqual([capture["rank"] for capture in case["captures"]], ["dp1", "dp0"])
                self.assertEqual([measured["rank"] for measured in case["measurements"]], ["dp1", "dp0"])
                self.assertEqual({method["id"]: method["status"] for method in core.displayed_methods(case)},
                                 {"method1": "LAYER_MEAN_ESTIMATE", "method2": "LAYER_MEAN_ESTIMATE"})
                self.assertNotIn("_prepared_layer_means", case)

    def test_bad_rank_and_duplicate_native_hash_fail_whole_case_in_input_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            config = rank_manifest(directory)
            document = json.loads(config.read_text())
            good, second = document["cases"]
            broken = copy.deepcopy(good)
            broken["name"] = "BAD_RANK"
            corrupt = directory / "corrupt.csv"
            corrupt.write_text("missing,required,headers\n1,2,3\n")
            broken["inputs"][1]["path"] = str(corrupt)
            duplicate_hash = copy.deepcopy(good)
            duplicate_hash["name"] = "DUPLICATED_NATIVE"
            copied = directory / "copied.csv"
            copied.write_bytes(Path(duplicate_hash["inputs"][0]["path"]).read_bytes())
            duplicate_hash["inputs"][1]["path"] = str(copied)
            document["cases"] = [good, broken, second, duplicate_hash, copy.deepcopy(good)]
            config.write_text(json.dumps(document))
            serial_dir, rank_dir = directory / "serial", directory / "rank"
            serial = self.invoke(config, serial_dir, "case", 1)
            ranked = self.invoke(config, rank_dir, "rank", 2)
            self.assertEqual(serial.returncode, 1, serial.stdout + serial.stderr)
            self.assertEqual(ranked.returncode, 1, ranked.stdout + ranked.stderr)
            report = self.assert_same_reports(serial_dir, rank_dir, exclude_failed_details=True)
            self.assertEqual([case["name"] for case in report["cases"]], ["ZETA_FIRST", "ALPHA_SECOND"])
            self.assertEqual([failure["case"] for failure in report["failures"]],
                             ["BAD_RANK", "DUPLICATED_NATIVE", "ZETA_FIRST"])
            self.assertIn("Identical CSV content", report["failures"][1]["error"])
            self.assertIn("Duplicate case name", report["failures"][2]["error"])

    def test_single_case_two_ranks_spawn_concurrently_once_and_each_csv_is_read_once(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            config = rank_manifest(directory, case_count=1)
            observations = directory / "observations"
            observations.mkdir()
            # The read barrier requires two actual children even on a fast
            # synthetic fixture. It only instruments this test launcher;
            # runtime code contains no delays or test synchronization.
            harness = directory / "observe_spawn.py"
            harness.write_text(textwrap.dedent('''\
                import concurrent.futures
                import json
                import os
                from pathlib import Path
                import sys
                import time
                sys.path.insert(0, __REPOSITORY__)
                import profile_ep128 as core
                observed = Path(__OBSERVATIONS__)
                original_pool = concurrent.futures.ProcessPoolExecutor
                class ObservedPool(original_pool):
                    def __init__(self, *args, **kwargs):
                        with (observed / ('pools-' + str(os.getpid()) + '.jsonl')).open('a') as stream:
                            stream.write(json.dumps({'pid': os.getpid(), 'workers': kwargs.get('max_workers'),
                                'start_method': kwargs['mp_context'].get_start_method()}) + '\\n')
                        super().__init__(*args, **kwargs)
                concurrent.futures.ProcessPoolExecutor = ObservedPool
                original_read = core.read_capture
                def watched_read(path, rank):
                    with (observed / ('reads-' + str(os.getpid()) + '.jsonl')).open('a') as stream:
                        stream.write(json.dumps({'pid': os.getpid(), 'rank': rank, 'path': str(path)}) + '\\n')
                    deadline = time.monotonic() + 20
                    while len(list(observed.glob('reads-*.jsonl'))) < 2:
                        if time.monotonic() >= deadline:
                            raise RuntimeError('two rank workers did not reach the read barrier')
                        time.sleep(.01)
                    return original_read(path, rank)
                core.read_capture = watched_read
                if __name__ == '__main__':
                    (observed / 'parent.json').write_text(json.dumps({'pid': os.getpid()}))
                    sys.argv = [core.__file__, __CONFIG__, '--out', __OUTPUT__,
                                '--parallel', 'rank', '--workers', '2']
                    raise SystemExit(core.main())
                ''').replace("__REPOSITORY__", repr(str(HERE)))
                .replace("__OBSERVATIONS__", repr(str(observations)))
                .replace("__CONFIG__", repr(str(config)))
                .replace("__OUTPUT__", repr(str(directory / "report"))))
            result = subprocess.run([sys.executable, "-B", "-S", str(harness)],
                                    text=True, capture_output=True, timeout=90)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            parent = json.loads((observations / "parent.json").read_text())["pid"]
            pools = [json.loads(line) for path in observations.glob("pools-*.jsonl")
                     for line in path.read_text().splitlines()]
            reads = [json.loads(line) for path in observations.glob("reads-*.jsonl")
                     for line in path.read_text().splitlines()]
            self.assertEqual(pools, [{"pid": parent, "workers": 2, "start_method": "spawn"}])
            self.assertEqual(len(reads), 2)
            self.assertEqual({row["rank"] for row in reads}, {"dp0", "dp1"})
            self.assertEqual(len({row["pid"] for row in reads}), 2)
            self.assertTrue(all(row["pid"] != parent for row in reads))

    def test_global_pool_handles_reversed_completion_without_nested_pool_or_csv_reread(self):
        class InlinePool:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False
            def submit(self, function, job, directory):
                future = Future()
                try:
                    future.set_result(function(job, directory))
                except Exception as error:
                    future.set_exception(error)
                return future
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            jobs = self.case_jobs(directory)
            with mock.patch("concurrent.futures.ProcessPoolExecutor", side_effect=InlinePool) as pools, \
                    mock.patch("concurrent.futures.as_completed", side_effect=lambda futures: iter(reversed(list(futures)))), \
                    mock.patch.object(core, "read_capture", wraps=core.read_capture) as reader, \
                    mock.patch.object(core, "_cli_run_case_jobs", side_effect=AssertionError("rank mode must not nest case pools")), \
                    redirect_stdout(io.StringIO()):
                output = core._cli_run_rank_jobs(jobs, workers=2)
            self.assertEqual(pools.call_count, 1)
            self.assertEqual(pools.call_args.kwargs["max_workers"], 2)
            self.assertEqual(pools.call_args.kwargs["mp_context"].get_start_method(), "spawn")
            self.assertEqual(reader.call_count, 4)
            self.assertEqual([item["index"] for item in output], [0, 1])
            self.assertEqual([item["result"]["name"] for item in output], ["ZETA_FIRST", "ALPHA_SECOND"])
            for item in output:
                self.assertEqual([rank["rank"] for rank in item["result"]["captures"]], ["dp1", "dp0"])
                self.assertNotIn("_prepared_layer_means", item["result"])
                self.assertEqual(core.displayed_methods(item["result"])[0]["status"], "LAYER_MEAN_ESTIMATE")

    def test_rank_worker_only_transfers_small_spool_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            job = self.case_jobs(directory, case_count=1)[0]
            plan = core._case_analysis_plan(*job[1:5])
            spool = directory / "spool"
            spool.mkdir()
            large = {"capture": {"rank": "dp0", "sha256": "synthetic"},
                     "measurement": {"large_evidence": "x" * 1000000},
                     "prepared": [], "source_errors": []}
            with mock.patch.object(core, "_analyze_case_rank", return_value=large), redirect_stdout(io.StringIO()):
                outcome = core._cli_rank_worker((3, 1, plan), spool)
            self.assertEqual(outcome[:2], (3, 1))
            self.assertEqual(Path(outcome[2]).parent, spool)
            self.assertLess(len(pickle.dumps(outcome)), 1024)
            self.assertEqual(list(spool.iterdir()), [Path(outcome[2])])
            with Path(outcome[2]).open("rb") as stream:
                payload = pickle.load(stream)
            self.assertIn(b"x" * 10000, pickle.dumps(payload))

    @staticmethod
    def assembler_plan():
        return {"case": {"name": "SYNTHETIC", "_rank_coverage_complete": True, "expected_ranks": 3},
                "model": {}, "target": {}, "names": ["dp1", "dp0", "dp2"], "source_errors": []}

    @staticmethod
    def assembler_rank(rank, digest, *, prepared=None, source_errors=(), failure=None):
        result = {"capture": {"rank": rank, "sha256": digest},
                  "measurement": {"rank": rank, "warnings": []},
                  "prepared": prepared, "source_errors": list(source_errors)}
        if failure is not None:
            result["failure"] = failure
        return result

    def test_assembler_keeps_first_soft_prepare_error_and_discards_later_prepared_evidence(self):
        plan = self.assembler_plan()
        ranks = [self.assembler_rank("dp1", "A", prepared=["earlier valid evidence"]),
                 self.assembler_rank("dp0", "B", source_errors=["FIRST_PREPARE_ERROR"]),
                 self.assembler_rank("dp2", "C", prepared=["must be ignored"],
                                     source_errors=["LATER_PREPARE_ERROR"])]
        with mock.patch.object(core, "summarize_source", return_value={}) as summary:
            result = core._assemble_case_analysis(plan, ranks)
        self.assertEqual(result["_layer_mean_source_errors"], ["FIRST_PREPARE_ERROR"])
        self.assertEqual(result["_prepared_layer_means"], {"dp1": ["earlier valid evidence"]})
        self.assertEqual([item["rank"] for item in result["captures"]], ["dp1", "dp0", "dp2"])
        self.assertEqual([item["rank"] for item in result["measurements"]], ["dp1", "dp0", "dp2"])
        self.assertEqual([item["rank"] for item in summary.call_args.args[0]], ["dp1", "dp0", "dp2"])
        plan["source_errors"] = ["PLAN_ERROR"]
        with mock.patch.object(core, "summarize_source", return_value={}):
            result = core._assemble_case_analysis(plan, ranks)
        self.assertEqual(result["_layer_mean_source_errors"], ["PLAN_ERROR"])
        self.assertEqual(result["_prepared_layer_means"], {})

    def test_assembler_duplicate_hash_precedes_same_rank_fatal_but_not_earlier_fatal(self):
        plan = self.assembler_plan()
        first = self.assembler_rank("dp1", "SAME")
        second = self.assembler_rank("dp0", "SAME", failure="SECOND_RANK_FATAL")
        with self.assertRaisesRegex(ValueError, "Identical CSV content.*dp1 and dp0"):
            core._assemble_case_analysis(plan, [first, second])
        earlier = self.assembler_rank("dp1", "SAME", failure="FIRST_RANK_FATAL")
        with self.assertRaisesRegex(ValueError, "^FIRST_RANK_FATAL$"):
            core._assemble_case_analysis(plan, [earlier, second])
        second["capture"]["sha256"] = "DIFFERENT"
        with self.assertRaisesRegex(ValueError, "^SECOND_RANK_FATAL$"):
            core._assemble_case_analysis(plan, [first, second])

    def test_rank_worker_preserves_capture_metadata_when_measurement_raises(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            job = self.case_jobs(directory, case_count=1)[0]
            plan = core._case_analysis_plan(*job[1:5])
            spool = directory / "spool"
            spool.mkdir()
            capture = {"rank": "dp0", "sha256": "native-read-before-fatal"}
            def fatal(plan, rank_index, *, context=None):
                context["capture"] = capture
                raise ValueError("measurement failed after native parse")
            with mock.patch.object(core, "_analyze_case_rank", side_effect=fatal), redirect_stdout(io.StringIO()):
                outcome = core._cli_rank_worker((0, 1, plan), spool)
            with Path(outcome[2]).open("rb") as stream:
                payload = pickle.load(stream)
            self.assertEqual(payload["result"]["capture"], capture)
            self.assertEqual(payload["result"]["failure"], "measurement failed after native parse")


if __name__ == "__main__":
    unittest.main()
