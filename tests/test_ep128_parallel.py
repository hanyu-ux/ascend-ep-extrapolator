"""Case scheduling and spawn integration; synthetic CPU data only."""
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
import unittest
from unittest import mock

import profile_ep128 as core
from test_ep128_v2_cli import manifest

HERE = Path(__file__).resolve().parents[1]


class ParallelCaseTests(unittest.TestCase):
    def invoke(self, config, output, workers):
        return subprocess.run([sys.executable, '-B', '-S', str(HERE / 'profile_ep128.py'),
            str(config), '--out', str(output), '--workers', str(workers)],
            text=True, capture_output=True, timeout=75)

    def normalize_output(self, value, output):
        if isinstance(value, dict):
            return {key: self.normalize_output(item, output) for key, item in value.items()
                    if key != 'generated_at'}
        if isinstance(value, list):
            return [self.normalize_output(item, output) for item in value]
        return value.replace(str(output), '<OUTPUT>') if isinstance(value, str) else value

    def test_spawn_and_serial_reports_and_all_csvs_are_identical_in_input_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            config = manifest(base)
            document = json.loads(config.read_text())
            first = document['cases'][0]
            first['name'] = 'ZETA_FIRST'
            second = copy.deepcopy(first)
            second['name'] = 'ALPHA_SECOND'
            second['synthetic_acceptance_length'] = 4.5
            document['cases'] = [first, second]
            config.write_text(json.dumps(document))
            serial_dir, parallel_dir = base / 'serial', base / 'parallel'
            serial = self.invoke(config, serial_dir, 1)
            parallel = self.invoke(config, parallel_dir, 2)
            self.assertEqual(serial.returncode, 0, serial.stdout + serial.stderr)
            self.assertEqual(parallel.returncode, 0, parallel.stdout + parallel.stderr)
            a = json.loads((serial_dir / 'model_architecture_report.json').read_text())
            b = json.loads((parallel_dir / 'model_architecture_report.json').read_text())
            self.assertEqual([item['name'] for item in b['cases']], ['ZETA_FIRST', 'ALPHA_SECOND'])
            self.assertEqual(self.normalize_output(a, serial_dir), self.normalize_output(b, parallel_dir))
            self.assertIn('layer_mean', b['contract'])
            self.assertTrue(all('_prepared_layer_means' not in case for case in b['cases']))
            left = {path.relative_to(serial_dir): path for path in serial_dir.rglob('*.csv')}
            right = {path.relative_to(parallel_dir): path for path in parallel_dir.rglob('*.csv')}
            self.assertTrue(left)
            self.assertEqual(set(left), set(right))
            for relative, path in left.items():
                with self.subTest(csv=str(relative)):
                    self.assertEqual(path.read_text().replace(str(serial_dir), '<OUTPUT>'),
                                     right[relative].read_text().replace(str(parallel_dir), '<OUTPUT>'))
            self.assertIn('[case ZETA_FIRST]', parallel.stdout)
            self.assertIn('读取 dp0', parallel.stdout)
            self.assertIn('统计与估算完成', parallel.stdout)

    def test_bad_case_and_duplicate_name_are_isolated_and_ordered(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            config = manifest(base)
            document = json.loads(config.read_text())
            good = document['cases'][0]
            good['name'] = 'GOOD'
            broken = copy.deepcopy(good)
            broken['name'] = 'BAD_NATIVE'
            corrupt = base / 'corrupt.csv'
            corrupt.write_text('missing,required,headers\n1,2,3\n')
            broken['inputs'] = [{'rank': 'dp0', 'path': str(corrupt)}]
            second = copy.deepcopy(good)
            second['name'] = 'GOOD_SECOND'
            duplicate = copy.deepcopy(good)
            document['cases'] = [good, broken, second, duplicate]
            config.write_text(json.dumps(document))
            output = base / 'out'
            completed = self.invoke(config, output, 2)
            self.assertEqual(completed.returncode, 1, completed.stdout + completed.stderr)
            report = json.loads((output / 'model_architecture_report.json').read_text())
            self.assertEqual([case['name'] for case in report['cases']], ['GOOD', 'GOOD_SECOND'])
            self.assertEqual([failure['case'] for failure in report['failures']], ['BAD_NATIVE', 'GOOD'])
            self.assertIn('Duplicate case name', report['failures'][1]['error'])
            self.assertEqual(len(report['failures']), 2)

    def test_complete_api_reuses_native_parse_and_discards_prepared_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            config = json.loads(manifest(base).read_text())
            case = core.normalize_case(config['defaults'], config['cases'][0], base)
            with mock.patch.object(core, 'read_capture', wraps=core.read_capture) as reader, redirect_stdout(io.StringIO()):
                completed = core._cli_case_complete(0, case, config['sample_model'], config['target_model'], base, [])
            self.assertEqual(reader.call_count, 1)
            self.assertNotIn('_prepared_layer_means', completed['result'])
            self.assertNotIn('_layer_mean_source_errors', completed['result'])
            self.assertEqual(completed['result']['projection']['layer_mean_methods'][0]['status'], 'LAYER_MEAN_ESTIMATE')
            self.assertIn('layer_mean', completed['contract'])

    def test_worker_returns_only_spool_identity_and_cleans_atomic_temporary(self):
        with tempfile.TemporaryDirectory() as temporary:
            payload = {'index': 3, 'result': {'evidence': 'x' * 1000000}, 'contract': {}}
            job = (3, {'name': 'SYNTHETIC'}, {}, None, '.', [])
            with mock.patch.object(core, '_cli_case_complete', return_value=payload):
                outcome = core._cli_case_worker(job, temporary)
            self.assertEqual(outcome, (3, str(Path(temporary) / 'case-3.pickle')))
            self.assertLess(len(pickle.dumps(outcome)), 1024)
            self.assertEqual([path.name for path in Path(temporary).iterdir()], ['case-3.pickle'])
            with Path(outcome[1]).open('rb') as stream:
                self.assertEqual(pickle.load(stream), payload)

    def test_single_case_never_creates_pool_and_unexpected_error_does_not_retry(self):
        job = (0, {'name': 'BROKEN'}, {}, None, '.', [])
        with mock.patch('concurrent.futures.ProcessPoolExecutor', side_effect=AssertionError('single case needs no pool')), \
                mock.patch.object(core.tempfile, 'TemporaryDirectory', side_effect=AssertionError('serial needs no spool')), \
                mock.patch.object(core, '_cli_case_complete', side_effect=RuntimeError('intentional worker failure')) as worker, \
                redirect_stdout(io.StringIO()):
            output = core._cli_run_case_jobs([job], workers=8)
        self.assertEqual(worker.call_count, 1)
        self.assertIn('CASE_WORKER_ERROR: RuntimeError', output[0]['failure']['error'])
        self.assertIn('intentional worker failure', output[0]['failure']['error'])

    def test_pool_future_error_is_reported_without_serial_fallback(self):
        jobs = [(index, {'name': 'CASE_' + str(index)}, {}, None, '.', []) for index in (0, 1)]
        class BrokenPool:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False
            def submit(self, function, job, directory):
                future = Future()
                future.set_exception(RuntimeError('synthetic child crash'))
                return future
        with mock.patch('concurrent.futures.ProcessPoolExecutor', side_effect=BrokenPool) as pool, \
                mock.patch.object(core, '_cli_case_worker', side_effect=AssertionError('no serial fallback')) as worker, \
                redirect_stdout(io.StringIO()):
            output = core._cli_run_case_jobs(jobs, workers=2)
        self.assertEqual(worker.call_count, 0)
        self.assertEqual([row['index'] for row in output], [0, 1])
        self.assertTrue(all('synthetic child crash' in row['failure']['error'] for row in output))
        self.assertEqual(pool.call_args.kwargs['mp_context'].get_start_method(), 'spawn')

    def test_worker_budget_conservative_defaults_and_explicit_request(self):
        with mock.patch.object(core, '_cli_case_cpu_budget', return_value=8), \
                mock.patch.object(core, '_cli_case_memory_available', return_value=3 * 1024 ** 3), \
                redirect_stdout(io.StringIO()):
            self.assertEqual(core._cli_case_workers(None, 9), 1)
            self.assertEqual(core._cli_case_workers(4, 9), 4)
        with mock.patch.object(core, '_cli_case_cpu_budget', return_value=8), \
                mock.patch.object(core, '_cli_case_memory_available', return_value=8 * 1024 ** 3), \
                redirect_stdout(io.StringIO()):
            self.assertEqual(core._cli_case_workers(None, 9), 2)
            self.assertEqual(core._cli_case_workers(None, 1), 1)
        with mock.patch.object(core, '_cli_case_cpu_budget', return_value=1), \
                mock.patch.object(core, '_cli_case_memory_available', return_value=None), \
                redirect_stdout(io.StringIO()):
            self.assertEqual(core._cli_case_workers(None, 9), 1)

    def test_memory_budget_uses_container_headroom_and_ancestor_limits(self):
        texts = {
            '/proc/meminfo': 'MemAvailable: 12000000 kB\n',
            '/proc/self/cgroup': '0::/tenant/worker\n',
            '/proc/self/mountinfo': '10 1 0:10 / /sys/fs/cgroup rw - cgroup2 cgroup rw\n',
            '/sys/fs/cgroup/memory.max': 'max', '/sys/fs/cgroup/memory.current': '10',
            '/sys/fs/cgroup/tenant/worker/memory.max': str(8 * 1024 ** 3),
            '/sys/fs/cgroup/tenant/worker/memory.current': str(1 * 1024 ** 3),
            '/sys/fs/cgroup/tenant/memory.max': str(4 * 1024 ** 3),
            '/sys/fs/cgroup/tenant/memory.current': str(2 * 1024 ** 3),
        }
        def read_text(path, *args, **kwargs):
            if str(path) not in texts:
                raise FileNotFoundError(str(path))
            return texts[str(path)]
        with mock.patch.object(Path, 'read_text', read_text):
            self.assertEqual(core._cli_case_memory_available(), 2 * 1024 ** 3)

    def test_cpu_budget_respects_cgroup_v2_and_v1_quota(self):
        import os
        for unified, fields in ((True, {'cpu.max': '100000 100000'}),
                                (False, {'cpu.cfs_quota_us': '100000', 'cpu.cfs_period_us': '100000'})):
            with self.subTest(unified=unified):
                def read_text(path, *args, **kwargs):
                    return fields[path.name]
                with mock.patch.object(os, 'cpu_count', return_value=16), \
                        mock.patch.object(core, '_cli_case_cgroups', return_value=[(Path('/synthetic-cgroup'), unified)]), \
                        mock.patch.object(Path, 'read_text', read_text):
                    self.assertEqual(core._cli_case_cpu_budget(), 1)

    def test_workers_requires_positive_integer(self):
        for workers in ('0', '-1', '1.5', 'many'):
            with self.subTest(workers=workers):
                result = subprocess.run([sys.executable, '-B', '-S', str(HERE / 'profile_ep128.py'),
                    '--workers', workers], text=True, capture_output=True, timeout=10)
                self.assertEqual(result.returncode, 2)
                self.assertIn('--workers需要正整数', result.stderr)


if __name__ == '__main__':
    unittest.main()
