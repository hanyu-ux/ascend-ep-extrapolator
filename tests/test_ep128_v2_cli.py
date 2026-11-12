"""End-to-end synthetic CPU evidence; none of these are performance measurements."""
import copy
from contextlib import redirect_stdout
import io
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest import mock
import xml.etree.ElementTree as ET
from zipfile import ZipFile

import profile_ep128 as p
from test_ep128_v2_measure import MODEL, CASE, EPOCH_NS, fixture, stamp_ns

HERE = Path(__file__).resolve().parents[1]


def manifest(base, *, full=False):
    template = json.loads((HERE / 'profile_ep128_cases.example.json').read_text())
    template['sample_model'] = copy.deepcopy(MODEL)
    template['communication_reference'] = str(HERE / 'examples/communication-reference.synthetic.csv') if full else 'intentionally_absent_optional_reference.csv'
    case = {'name': 'SYNTHETIC_CPU_ONLY', 'bs': 32, 'draft_tokens': 7,
            'synthetic_acceptance_length': 5.2, 'inputs': []}
    template['cases'] = [case]
    for rank in range(16 if full else 1):
        directory = base / ('dp' + str(rank))
        directory.mkdir()
        def change(rows, records):
            # Unique physical-device metadata prevents testing a copied CSV as
            # if it were independent evidence from another global rank.
            for row in rows:
                row['Device_id'] = str(rank)
        capture, records = fixture(directory, changes=change)
        capture['rank'] = 'dp' + str(rank)
        case['inputs'].append({'rank': capture['rank'], 'path': capture['path']})
    path = base / 'cases.json'
    path.write_text(json.dumps(template))
    return path


class CLIIntegrationTests(unittest.TestCase):
    def invoke(self, manifest_path, out, *args):
        # -S proves the runtime does not depend on site-installed packages.
        return subprocess.run([sys.executable, '-S', str(HERE / 'profile_ep128.py'), str(manifest_path),
                               '--out', str(out), *args], cwd=manifest_path.parent,
                              text=True, capture_output=True, timeout=45)

    def test_new_rate_changes_source_and_three_method_rates_without_retiming_capture(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            path = manifest(base)
            document = json.loads(path.read_text())
            item = document['cases'][0]
            item.pop('synthetic_acceptance_length')
            cases = []
            for rate, neff in ((.65, 5.2), (.8, 6.4)):
                item['acceptance'] = rate
                path.write_text(json.dumps(document))
                output = base / ('rate-' + str(rate))
                result = self.invoke(path, output, '--excel')
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                report = json.loads((output / 'model_architecture_report.json').read_text())
                case = report['cases'][0]
                self.assertEqual(case['config']['effective_tokens_per_round'], neff)
                self.assertEqual(case['config']['acceptance_semantics'], 'mean_output_fraction_including_bonus')
                with (output / '最终结果.csv').open(encoding='utf-8-sig', newline='') as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual(len(rows), 3)
                self.assertTrue(all(float(row['配置接受率']) == rate for row in rows))
                self.assertTrue(all(row['接受率口径'] == 'mean_output_fraction_including_bonus' for row in rows))
                self.assertIn('SAL=(1+S)×r', (output / '10T模型推理估算.html').read_text(encoding='utf-8'))
                cases.append(case)
            original, changed = cases
            self.assertEqual(original['captures'], changed['captures'])
            self.assertEqual(original['source']['aggregate']['cycle_ms'], changed['source']['aggregate']['cycle_ms'])
            self.assertAlmostEqual(changed['source']['aggregate']['per_device_tokens_s']
                                   / original['source']['aggregate']['per_device_tokens_s'], 6.4 / 5.2)
            for before, after in zip(p.displayed_methods(original), p.displayed_methods(changed)):
                self.assertEqual(before['id'], after['id'])
                self.assertEqual(before['status'], after['status'])
                self.assertEqual(before['cycle_ms'], after['cycle_ms'])
                self.assertEqual(before['breakdown'], after['breakdown'])
                if after['status'] == 'LAYER_MEAN_ESTIMATE':
                    self.assertAlmostEqual(after['per_device_tokens_s'] / before['per_device_tokens_s'], 6.4 / 5.2)
                    self.assertAlmostEqual(after['tpot_ms'] / before['tpot_ms'], 5.2 / 6.4)

    def test_default_estimate_with_missing_optional_reference_and_excel(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            result = self.invoke(manifest(base), base / 'report', '--excel')
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            report = json.loads((base / 'report/model_architecture_report.json').read_text())
            case = report['cases'][0]
            self.assertEqual(report['failures'], [])
            self.assertAlmostEqual(case['source']['aggregate']['per_device_tokens_s'], 166400 / 1.1)
            self.assertFalse(case['source']['aggregate']['rank_coverage_complete'])
            methods = p.displayed_methods(case)
            self.assertEqual([method['id'] for method in methods], ['method1', 'method2', 'method3'])
            self.assertEqual(methods[0]['status'], 'LAYER_MEAN_ESTIMATE', methods[0]['reasons'])
            self.assertEqual(methods[1]['status'], 'UNAVAILABLE')
            self.assertEqual(methods[2]['status'], 'LAYER_MEAN_ESTIMATE', methods[2]['reasons'])
            self.assertEqual(methods[2]['selected_rank'], methods[0]['selected_rank'])
            for row in methods[2]['class_means']:
                self.assertEqual(row['dispatch_delta_ms'], 0)
                self.assertEqual(row['combine_delta_ms'], 0)
            self.assertIsNotNone(report['communication_reference']['error'])
            self.assertFalse(list((base / 'report').glob('*_ownership_template.json')))
            with ZipFile(base / 'report/10T模型推理估算.xlsx') as archive:
                self.assertIsNone(archive.testzip())
                root = ET.fromstring(archive.read('xl/workbook.xml'))
                self.assertEqual(len(root.findall('{*}sheets/{*}sheet')), 10)
            with (base / 'report/最终结果.csv').open(encoding='utf-8-sig', newline='') as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 3)
            self.assertGreater(float(rows[0]['单卡吞吐(token/s/卡)']), 0)
            self.assertEqual(rows[1]['周期(ms)'], '')
            self.assertEqual(rows[2]['方法'], methods[2]['name'])
            self.assertAlmostEqual(float(rows[2]['周期(ms)']), methods[2]['cycle_ms'])

    def test_default_prediction_reuses_the_single_native_csv_read(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            config = json.loads(manifest(base).read_text())
            case = p.normalize_case(config['defaults'], config['cases'][0], base)
            with mock.patch.object(p, 'read_capture', wraps=p.read_capture) as reader, redirect_stdout(io.StringIO()):
                before_files = {str(path.relative_to(base)) for path in base.rglob('*')}
                before_cwd = {path.name for path in Path.cwd().iterdir()}
                result = p.analyze_case(case, config['sample_model'], config['target_model'], base, [])
                self.assertEqual(reader.call_count, 1)
                diagnostic = result['measurements'][0]['step_residuals']
                self.assertEqual(diagnostic['status'], 'DEVICE_ONLY')
                self.assertTrue(diagnostic['stage_summary'])
                for field in ('device_tasks_csv', 'host_events_csv', 'stages_csv', 'source_links_csv'):
                    self.assertIsNone(diagnostic[field], field)
                self.assertEqual({str(path.relative_to(base)) for path in base.rglob('*')}, before_files)
                self.assertEqual({path.name for path in Path.cwd().iterdir()}, before_cwd)
                # The source was already measured; no further parse is needed
                # to apply all three estimation methods to those measurements.
                Path(result['captures'][0]['path']).unlink()
                report = {'cases': [result]}
                p.attach_layer_mean_report(report)
                self.assertEqual(reader.call_count, 1)
            method = p.displayed_methods(result)[0]
            self.assertEqual(method['status'], 'LAYER_MEAN_ESTIMATE', method['reasons'])
            self.assertGreater(method['cycle_ms'], 0)

    def test_complete_rank_layer_means_do_not_need_manual_component_calibration(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            path = manifest(base, full=True)
            result = self.invoke(path, base / 'report', '--excel')
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            report = json.loads((base / 'report/model_architecture_report.json').read_text())
            case = report['cases'][0]
            self.assertEqual(len(case['source']['per_rank']), 16)
            self.assertTrue(case['source']['aggregate']['rank_coverage_complete'])
            self.assertEqual(case['projection']['display_method_set'], 'layer_mean_methods')
            methods = p.displayed_methods(case)
            self.assertEqual([method['id'] for method in methods], ['method1', 'method2', 'method3'])
            classes = {(row['role'], row['attention']): row for row in methods[0]['class_means']}
            self.assertEqual(classes['main', 'HCA']['reference_layers'], [1, 3, 5, 7])
            self.assertEqual(classes['main', 'CSA']['reference_layers'], [2, 4, 6])
            self.assertEqual(sum(row['target_layers'] for row in classes.values() if row['role'] == 'main'), 61)
            with (base / 'report/最终结果.csv').open(encoding='utf-8-sig', newline='') as stream:
                final_rows = list(csv.DictReader(stream))
            self.assertEqual(len(final_rows), 3)
            for row, method in zip(final_rows, methods):
                self.assertAlmostEqual(float(row['单卡吞吐(token/s/卡)']), method['per_device_tokens_s'])
                self.assertAlmostEqual(float(row['周期(ms)']), method['cycle_ms'])
                self.assertNotAlmostEqual(float(row['单卡吞吐(token/s/卡)']), case['source']['aggregate']['per_device_tokens_s'])
            for method in methods:
                self.assertEqual(method['status'], 'LAYER_MEAN_ESTIMATE', method['reasons'])
                self.assertAlmostEqual(method['per_device_tokens_s'], 166400 / method['cycle_ms'])
                self.assertAlmostEqual(method['global_tokens_s'], method['per_device_tokens_s'] * 128)
                self.assertAlmostEqual(sum(r['value_ms'] for r in method['breakdown']), method['cycle_ms'])
                self.assertTrue(method['source_partition']['closed'])
                self.assertGreater(method['source_partition']['residual_ms'], 0)
            self.assertNotIn('全局吞吐(token/s)', final_rows[0])
            ns = case['measurements'][0]['rounds'][0]['start_ns']
            self.assertIsInstance(ns, str)
            self.assertTrue(int(ns) > 2**53)

    def test_cli_conflicting_modes_and_float_schema_fail(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            path = manifest(base)
            result = self.invoke(path, base / 'out', '--list-inputs', '--excel')
            self.assertEqual(result.returncode, 2)
            document = json.loads(path.read_text()); document['schema_version'] = 2.0
            path.write_text(json.dumps(document))
            result = self.invoke(path, base / 'out', '--list-inputs')
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('schema_version=2', result.stderr)

    def test_list_inputs_flushes_before_scan_completes_and_never_reads_csv(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            source = base / 'dp0'; source.mkdir()
            # Listing must work even when the selected file is not valid CSV.
            (source / 'kernel_details.csv').write_text('not parsed kernel data')
            config = base / 'cases.json'
            config.write_text(json.dumps({'schema_version': 2,
                'defaults': {'expected_ranks': 1},
                'cases': [{'name': 'list-only', 'path': str(source)}]}))
            harness = '''import sys
sys.path.insert(0, sys.argv[1])
import profile_ep128 as p
original = p.Path.rglob
def paused_search(self, pattern):
    if sys.stdin.readline().strip() != 'continue':
        raise RuntimeError('test scan was not released')
    return original(self, pattern)
p.Path.rglob = paused_search
sys.argv = [p.__file__, sys.argv[2], '--list-inputs']
raise SystemExit(p.main())
'''
            # No -u: the CLI itself must flush before a slow filesystem search.
            with subprocess.Popen([sys.executable, '-S', '-c', harness, str(HERE), str(config)],
                    cwd=base, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, text=True) as process:
                lines, scanning = [], threading.Event()
                def receive():
                    for line in process.stdout:
                        lines.append(line)
                        if '查找 kernel_details.csv：' in line:
                            scanning.set()
                reader = threading.Thread(target=receive, daemon=True)
                reader.start()
                try:
                    self.assertTrue(scanning.wait(5), ''.join(lines))
                    visible = ''.join(lines)
                    self.assertIn('输入清单模式', visible)
                    self.assertIn(str(HERE / 'profile_ep128.py'), visible)
                    self.assertIn(str(config), visible)
                    self.assertIn('list-only；开始扫描输入路径', visible)
                    self.assertNotIn('输入清单完成', visible)
                    self.assertIsNone(process.poll())
                    process.stdin.write('continue\n'); process.stdin.flush()
                    self.assertEqual(process.wait(timeout=10), 0)
                finally:
                    if process.poll() is None:
                        process.kill(); process.wait(timeout=5)
                    reader.join(timeout=5)
                output = ''.join(lines)
                self.assertIn('dp0\t' + str((source / 'kernel_details.csv').resolve()), output)
                self.assertIn('Expected global-rank coverage: True', output)
                self.assertIn('已列出 1 份CSV，失败 0 个case', output)
                self.assertFalse((base / 'ep128_report_v2').exists())

    def test_list_inputs_reports_missing_paths_and_duplicate_ranks(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            duplicate = base / 'duplicates'
            for host in ('node0', 'node1'):
                directory = duplicate / host / 'rank0'; directory.mkdir(parents=True)
                (directory / 'kernel_details.csv').write_text('not parsed kernel data')
            path = base / 'cases.json'
            path.write_text(json.dumps({'schema_version': 2, 'defaults': {'expected_ranks': 16},
                'cases': [{'name': 'missing', 'path': str(base / 'absent')},
                          {'name': 'duplicates', 'path': str(duplicate)}]}))
            result = self.invoke(path, base / 'out', '--list-inputs')
            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            self.assertIn('输入清单模式', result.stdout)
            self.assertIn('missing；开始扫描输入路径', result.stdout)
            self.assertIn('Profile path not found', result.stdout)
            self.assertIn('duplicates；开始扫描输入路径', result.stdout)
            self.assertIn('Duplicate global rank/capture', result.stdout)
            self.assertIn('失败 2 个case', result.stdout)
            self.assertFalse((base / 'out').exists())

    def test_bs4_bs8_reported_v3_layout_survives_csv_parse_and_excel(self):
        # Device timings are synthetic; only the shape/dtype serialization is
        # based on the reported V3 row. This is not a hardware forecast.
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            path = manifest(base)
            document = json.loads(path.read_text())
            original = Path(document['cases'][0]['inputs'][0]['path'])
            with original.open(encoding='utf-8-sig', newline='') as stream:
                original_rows = list(csv.DictReader(stream))
            document['cases'] = []
            for bs in (4, 8):
                rows = copy.deepcopy(original_rows)
                for row in rows:
                    if 'Dispatch' in row['Name']:
                        capacity = bs * 8
                        row.update(Name='MoeDistributeDispatchV3', Type='MoeDistributeDispatchV3')
                        row['Input Shapes'] = f'"4100;{capacity},7168;{capacity},16;;{capacity};;;"'
                        row['Input Data Types'] = ('INT32;DT_BF16;INT32;DT_UNDEFINED;BOOL;'
                                                   'DT_UNDEFINED;DT_UNDEFINED;DT_UNDEFINED')
                source = base / f'bs{bs}' / 'dp0' / 'kernel_details.csv'
                source.parent.mkdir(parents=True)
                fields = list(dict.fromkeys(key for row in rows for key in row))
                with source.open('w', encoding='utf-8', newline='') as stream:
                    writer = csv.DictWriter(stream, fieldnames=fields)
                    writer.writeheader(); writer.writerows(rows)
                document['cases'].append({'name': f'layout-bs{bs}', 'bs': bs, 'draft_tokens': 7,
                    'synthetic_acceptance_length': 5.2,
                    'inputs': [{'rank': 'dp0', 'path': str(source)}]})
            path.write_text(json.dumps(document))
            result = self.invoke(path, base / 'out', '--excel')
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            report = json.loads((base / 'out/model_architecture_report.json').read_text())
            self.assertEqual(report['failures'], [])
            self.assertEqual({case['name'] for case in report['cases']}, {'layout-bs4', 'layout-bs8'})
            for case in report['cases']:
                self.assertEqual(case['source_model']['num_experts_per_tok'], 16)
                self.assertTrue(case['measurements'][0]['rounds'])
                self.assertFalse(any('DISPATCH_INPUT_LAYOUT_UNVERIFIED' in warning
                                     for warning in case['measurements'][0]['warnings']))
                methods = p.displayed_methods(case)
                self.assertEqual(case['projection']['display_method_set'], 'layer_mean_methods')
                self.assertEqual(methods[0]['status'], 'LAYER_MEAN_ESTIMATE', methods[0]['reasons'])
                self.assertEqual(methods[1]['status'], 'UNAVAILABLE')
            self.assertTrue((base / 'out/10T模型推理估算.xlsx').is_file())

    def test_identical_csv_cannot_be_renamed_into_complete_rank_coverage(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            path = manifest(base)
            document = json.loads(path.read_text())
            first = document['cases'][0]['inputs'][0]
            duplicate = base / 'copied.csv'; duplicate.write_bytes(Path(first['path']).read_bytes())
            document['cases'][0]['inputs'].append({'rank': 'dp1', 'path': str(duplicate)})
            path.write_text(json.dumps(document))
            result = self.invoke(path, base / 'out')
            self.assertEqual(result.returncode, 1)
            report = json.loads((base / 'out/model_architecture_report.json').read_text())
            self.assertEqual(report['cases'], [])
            self.assertIn('Identical CSV content', report['failures'][0]['error'])

    def test_multi_bs_uses_own_timing_and_reference_in_one_workbook(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            path = manifest(base)
            document = json.loads(path.read_text())
            document['communication_reference'] = str(HERE / 'examples/communication-reference.synthetic.csv')
            document['cases'] = []
            # Different synthetic durations ensure a cached BS32 duration
            # cannot pass this test. BS24 deliberately has no reference row.
            expected = {}
            for bs, factor in ((8, 2), (32, 1), (24, 3)):
                directory = base / ('bs' + str(bs))
                directory.mkdir()
                def scale(rows, records):
                    for row in rows:
                        start = p.a_ns(row['Start Time(us)'])
                        row['Start Time(us)'] = stamp_ns(EPOCH_NS + (start - EPOCH_NS) * factor)
                        row['Duration(us)'] = str(float(row['Duration(us)']) * factor)
                capture, _ = fixture(directory, changes=scale)
                name = 'synthetic-bs' + str(bs)
                document['cases'].append({'name': name, 'bs': bs, 'draft_tokens': 7,
                                          'synthetic_acceptance_length': 5.2,
                                          'inputs': [{'rank': 'dp0', 'path': capture['path']}]})
                expected[name] = (bs, 1.1 * factor)
            path.write_text(json.dumps(document))
            result = self.invoke(path, base / 'out', '--excel')
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            report = json.loads((base / 'out/model_architecture_report.json').read_text())
            self.assertEqual(len(report['cases']), 3)
            for item in report['cases']:
                bs, duration = expected[item['name']]
                source = item['source']['aggregate']
                self.assertAlmostEqual(source['cycle_ms'], duration)
                self.assertAlmostEqual(source['per_device_tokens_s'], 1000 * bs * 5.2 / duration)
                method2 = p.displayed_methods(item)[1]
                self.assertEqual(method2['comm_model']['reference_bs'], bs)
                fit = method2['comm_model']['fits']['dispatch']
                if bs == 24:
                    self.assertIsNone(fit)
                    self.assertEqual(method2['status'], 'UNAVAILABLE')
                else:
                    expected32 = {8: 16., 32: 64.}[bs]
                    self.assertEqual(fit['observed_ep32'], expected32)
            with (base / 'out/最终结果.csv').open(encoding='utf-8-sig', newline='') as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 9)
            self.assertEqual({row['采样组合'] for row in rows}, set(expected))
            for item in report['cases']:
                methods = p.displayed_methods(item)
                self.assertEqual([method['id'] for method in methods], ['method1', 'method2', 'method3'])
                own_rows = [row for row in rows if row['采样组合'] == item['name']]
                self.assertEqual([row['方法'] for row in own_rows], [method['name'] for method in methods])
                self.assertEqual(methods[2]['status'], 'LAYER_MEAN_ESTIMATE', methods[2]['reasons'])
            with ZipFile(base / 'out/10T模型推理估算.xlsx') as archive:
                self.assertEqual(len(ET.fromstring(archive.read('xl/workbook.xml')).findall('{*}sheets/{*}sheet')), 10)
                for index in (1, 2, 3, 4, 5, 6, 8, 9, 10):
                    xml = archive.read(f'xl/worksheets/sheet{index}.xml').decode()
                    for name in expected:
                        self.assertIn(name, xml)

    def test_same_bs_different_speculation_lengths_keep_independent_samples(self):
        # Separate synthetic captures have deliberately different timings.
        # Neither changing S on one capture nor reusing a BS-keyed result can
        # establish the two independent estimates checked here.
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            path = manifest(base)
            document = json.loads(path.read_text())
            # An explicit case length replaces an older manifest's fixed
            # Neff and acceptance policy; neither case declares a rate.
            document['defaults'].update(acceptance=.6,
                acceptance_semantics='mean_accepted_draft_fraction',
                effective_tokens_per_round=5.2,
                token_basis='Configured legacy S7 Neff including bonus; synthetic only')
            document['cases'] = []
            expected = {}
            for speculative, factor, effective in ((7, 1, 5.2), (9, 2, 6.4)):
                directory = base / f's{speculative}' / 'dp0'
                directory.mkdir(parents=True)

                def scale(rows, records):
                    for row in rows:
                        start = p.a_ns(row['Start Time(us)'])
                        row['Start Time(us)'] = stamp_ns(EPOCH_NS + (start - EPOCH_NS) * factor)
                        row['Duration(us)'] = str(float(row['Duration(us)']) * factor)
                        if 'Dispatch' in row['Name']:
                            tokens = 32 * (speculative + 1)
                            row['Input Shapes'] = f'[{tokens},7168];[{tokens},16]'

                capture, _ = fixture(directory, changes=scale)
                name = f'SYNTHETIC_bs32_s{speculative}'
                length_field = ('synthetic_acceptance_length' if speculative == 7
                                else 'synthetic_acception_length')
                document['cases'].append({'name': name, 'bs': 32,
                    'num_speculative_tokens': speculative,
                    length_field: effective,
                    'inputs': [{'rank': 'dp0', 'path': capture['path']}]})
                expected[name] = (speculative, effective, 1.1 * factor, capture['path'])
            path.write_text(json.dumps(document))
            result = self.invoke(path, base / 'out', '--excel')
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            report = json.loads((base / 'out/model_architecture_report.json').read_text())
            self.assertEqual(report['failures'], [])
            self.assertEqual([case['name'] for case in report['cases']], list(expected))
            methods = {}
            digests = set()
            for case in report['cases']:
                speculative, effective, duration, source_path = expected[case['name']]
                self.assertEqual(case['config']['bs'], 32)
                self.assertEqual(case['config']['draft_tokens'], speculative)
                self.assertAlmostEqual(case['config']['effective_tokens_per_round'], effective)
                self.assertAlmostEqual(case['config']['synthetic_acceptance_length'], effective)
                self.assertNotIn('synthetic_acception_length', case['config'])
                self.assertNotIn('acceptance', case['config'])
                self.assertNotIn('acceptance_semantics', case['config'])
                self.assertEqual([capture['path'] for capture in case['captures']], [source_path])
                digests.add(case['captures'][0]['sha256'])
                self.assertAlmostEqual(case['source']['aggregate']['cycle_ms'], duration)
                self.assertAlmostEqual(case['source']['aggregate']['per_device_tokens_s'],
                                       1000 * 32 * effective / duration)
                self.assertTrue(case['measurements'][0]['rounds'])
                for cycle in case['measurements'][0]['rounds']:
                    self.assertAlmostEqual(cycle['duration_ms'], duration)
                    self.assertAlmostEqual(cycle['output_tokens'], 32 * effective)
                method = p.displayed_methods(case)[0]
                methods[case['name']] = method
                self.assertEqual(method['status'], 'LAYER_MEAN_ESTIMATE', method['reasons'])
                self.assertAlmostEqual(method['tpot_ms'], method['cycle_ms'] / effective)
            self.assertEqual(len(digests), 2)
            self.assertNotAlmostEqual(*(method['cycle_ms'] for method in methods.values()))
            with (base / 'out/最终结果.csv').open(encoding='utf-8-sig', newline='') as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 6)
            for name, (speculative, effective, _, _) in expected.items():
                own_rows = [row for row in rows if row['采样组合'] == name]
                self.assertEqual(len(own_rows), 3)
                for row in own_rows:
                    self.assertEqual(int(row['草稿token数S']), speculative)
                    self.assertAlmostEqual(float(row['有效输出Neff(含bonus)']), effective)
                    self.assertEqual(row['配置接受率'], '')
                    self.assertEqual(row['接受率口径'], '')
                available = next(row for row in own_rows if row['周期(ms)'])
                self.assertAlmostEqual(float(available['周期(ms)']), methods[name]['cycle_ms'])
                self.assertAlmostEqual(float(available['TPOT(ms)']), methods[name]['cycle_ms'] / effective)
            html = (base / 'out/report.html').read_text(encoding='utf-8')
            for name in expected:
                self.assertIn(name, html)
            with ZipFile(base / 'out/10T模型推理估算.xlsx') as archive:
                self.assertIsNone(archive.testzip())
                for index in (1, 2, 3):
                    xml = archive.read(f'xl/worksheets/sheet{index}.xml').decode()
                    for name in expected:
                        self.assertIn(name, xml)



    def test_default_diagnostics_and_excel_preserve_estimates_and_show_device_only_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cases = manifest(root)
            baseline, enabled = root / 'baseline', root / 'diagnostics'
            for out, flags in ((baseline, ()), (enabled, ('--excel',))):
                result = self.invoke(cases, out, *flags)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            left = json.loads((baseline / 'model_architecture_report.json').read_text())
            right = json.loads((enabled / 'model_architecture_report.json').read_text())
            self.assertEqual(left['cases'][0]['projection'], right['cases'][0]['projection'])
            self.assertEqual((baseline / '最终结果.csv').read_bytes(), (enabled / '最终结果.csv').read_bytes())
            self.assertIn('step_residuals', left['cases'][0]['measurements'][0])
            self.assertEqual(left['cases'][0]['measurements'][0]['step_residuals']['stage_summary'],
                             right['cases'][0]['measurements'][0]['step_residuals']['stage_summary'])
            diagnostic = right['cases'][0]['measurements'][0]['step_residuals']
            self.assertEqual(diagnostic['status'], 'DEVICE_ONLY')
            self.assertEqual(diagnostic['source_status'], 'UNAVAILABLE')
            self.assertIsNone(diagnostic['source_links_csv'])
            self.assertTrue(all(row['source_cpu_union_ms_mean'] is None
                                for row in diagnostic['stage_summary']))
            self.assertEqual(len(diagnostic['windows']), 6)
            self.assertTrue(all(window['host_span_union_ms'] is None for window in diagnostic['windows']))
            self.assertTrue(Path(diagnostic['device_tasks_csv']).is_file())
            with (enabled / 'step_residual_windows.csv').open(encoding='utf-8-sig', newline='') as source:
                rows = list(csv.DictReader(source))
            self.assertEqual(len(rows), 6)
            for row in rows:
                self.assertEqual(row['step_kind'], 'hc_cycle')
                self.assertAlmostEqual(float(row['recorded_busy_ms']) + float(row['no_recorded_ms']),
                                       float(row['duration_ms']))
            with ZipFile(enabled / '10T模型推理估算.xlsx') as archive:
                ledger = archive.read('xl/worksheets/sheet4.xml').decode()
                self.assertIn('HC周期剩余区间实测', ledger)
                self.assertIn('主→草稿', ledger)
                self.assertIn('草稿→下一主', ledger)
                self.assertIn('阶段', ledger)
                self.assertIn('未识别', ledger)
            with Path(diagnostic['stages_csv']).open(encoding='utf-8-sig', newline='') as source:
                stages = list(csv.DictReader(source))
            self.assertEqual(len(stages), 6)
            self.assertEqual(diagnostic['substage_summary'], [])
            for window in diagnostic['windows']:
                stage = next(row for row in stages
                             if (row['round_id'], row['window_key']) == (window['round_id'], window['window_key']))
                self.assertEqual(stage['stage_key'], 'unresolved')
                self.assertEqual(stage['status'], 'UNRESOLVED')
                self.assertTrue(stage['reason'])
                self.assertEqual(int(stage['duration_ns']), int(window['end_ns']) - int(window['start_ns']))
            self.assertTrue(all(row['matched_windows'] == 0 and row['duration_ms_mean'] is None
                                for row in diagnostic['stage_summary']))
            self.assertTrue((enabled / 'step_residual_stages.csv').is_file())
            self.assertTrue((enabled / 'step_residual_substages.csv').is_file())
            self.assertNotIn('breakdown_summary', diagnostic)
            self.assertNotIn('cycle_breakdown', diagnostic)
            self.assertNotIn('host_summary', diagnostic)

    def test_native_attention_csv_adds_operator_evidence_without_changing_estimates(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            cases = manifest(base)
            document = json.loads(cases.read_text())
            document['communication_reference'] = str(HERE / 'examples/communication-reference.synthetic.csv')
            cases.write_text(json.dumps(document))
            baseline_out, expanded_out = base / 'baseline', base / 'expanded'
            baseline_run = self.invoke(cases, baseline_out)
            self.assertEqual(baseline_run.returncode, 0, baseline_run.stdout + baseline_run.stderr)
            source = Path(document['cases'][0]['inputs'][0]['path'])
            with source.open(encoding='utf-8-sig', newline='') as stream:
                rows = list(csv.DictReader(stream))
            for row in rows:
                row['Task Type'] = 'AI_CORE'
                if row['Type'] != 'Attention':
                    continue
                layer = ((p.a_ns(row['Start Time(us)']) - EPOCH_NS) % 1_100_000) // 100_000
                kind = ('SparseAttnSharedkv' if layer >= 8 else
                        'SparseFlashAttention' if MODEL['compress_ratios'][layer] == 4 else
                        'MixedQuantSparseFlashMla')
                # Change only operator metadata, preserving every timestamp,
                # duration and HC boundary from the baseline capture.
                row.update(Name='aclnn' + kind + '_' + kind, Type=kind)
            fields = list(dict.fromkeys(key for row in rows for key in row))
            with source.open('w', encoding='utf-8', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader(); writer.writerows(rows)
            expanded_run = self.invoke(cases, expanded_out, '--excel')
            self.assertEqual(expanded_run.returncode, 0, expanded_run.stdout + expanded_run.stderr)
            before = json.loads((baseline_out / 'model_architecture_report.json').read_text())
            after = json.loads((expanded_out / 'model_architecture_report.json').read_text())
            self.assertEqual(after['failures'], [])
            old_methods = p.displayed_methods(before['cases'][0])
            new_methods = p.displayed_methods(after['cases'][0])
            self.assertEqual([method['id'] for method in new_methods], ['method1', 'method2', 'method3'])
            self.assertEqual([method['id'] for method in old_methods], ['method1', 'method2', 'method3'])
            for old, new in zip(old_methods, new_methods):
                self.assertEqual(new['status'], 'LAYER_MEAN_ESTIMATE', new['reasons'])
                for field in ('cycle_ms', 'tpot_ms', 'per_device_tokens_s', 'global_tokens_s',
                              'selected_rank', 'class_means', 'role_details', 'source_partition', 'breakdown'):
                    self.assertEqual(new[field], old[field], (new['id'], field))
                self.assertEqual(len(new['operator_hotspots']), 3)
                self.assertEqual(len(new['target_stage_breakdowns']), 3)
                for group in new['target_stage_breakdowns']:
                    self.assertEqual(group['status'], 'MATCHED', group['reason'])
                    self.assertEqual(len(group['stage_rows']), 10)
                    self.assertAlmostEqual(sum(row['target_ms'] for row in group['stage_rows']), group['target_wall_ms'])
                    self.assertTrue(all('operator_breakdown' not in stage for stage in group['stage_rows']))
                for group in new['operator_hotspots']:
                    self.assertEqual(group['status'], 'MATCHED', group['reason'])
                    self.assertTrue(group['kernels'])
                    self.assertTrue(all(kernel['task_type'] == 'AI_CORE' for kernel in group['kernels']))
            self.assertEqual((baseline_out / '最终结果.csv').read_bytes(),
                             (expanded_out / '最终结果.csv').read_bytes())
            html = (expanded_out / '10T模型推理估算.html').read_text(encoding='utf-8')
            self.assertEqual(html.count('class="operator-hotspot-section"'), 9)
            self.assertTrue((expanded_out / '10T模型推理估算.xlsx').is_file())

    def test_input_listing_and_inventory_do_not_analyze_cycles_or_scan_trace(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            cases = manifest(base)
            csv_directory = Path(json.loads(cases.read_text())['cases'][0]['inputs'][0]['path']).parent
            with mock.patch.object(p, 'analyze_case', side_effect=AssertionError('must not analyze a case')), \
                    mock.patch.object(p, 'residual_analyze_capture', side_effect=AssertionError('must not scan trace')):
                for arguments in ((str(cases), '--list-inputs'), ('--inventory', str(csv_directory))):
                    with self.subTest(arguments=arguments), mock.patch.object(sys, 'argv', [p.__file__, *arguments]), \
                            redirect_stdout(io.StringIO()):
                        self.assertEqual(p.main(), 0)
            self.assertFalse((base / 'ep128_report_v2').exists())
            self.assertFalse((base / 'step_residual_details').exists())

    def test_removed_diagnostic_flag_is_rejected_before_reading_inputs(self):
        for flags in (('--step-residuals',), ('--list-inputs', '--step-residuals')):
            with self.subTest(flags=flags):
                result = subprocess.run([sys.executable, '-S', str(HERE / 'profile_ep128.py'),
                                         'not-read.json', *flags], text=True, capture_output=True)
                self.assertEqual(result.returncode, 2)
                self.assertIn('--step-residuals', result.stderr)
                self.assertNotIn('FileNotFoundError', result.stderr)
        help_result = subprocess.run([sys.executable, '-S', str(HERE / 'profile_ep128.py'), '--help'],
                                     text=True, capture_output=True)
        self.assertNotIn('--step-residuals', help_result.stdout)


if __name__ == "__main__":
    unittest.main()
