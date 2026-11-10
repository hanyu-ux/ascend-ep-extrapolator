#!/usr/bin/env python3
"""Schema-2 renderer contracts; all numeric fixtures are synthetic, never profiling claims."""
import ast
import copy
import csv
import importlib.util
import json
import os
import subprocess
from pathlib import Path
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from zipfile import ZipFile

HERE=Path(__file__).resolve().parents[1]
RENDERER=HERE/'ep128_v2_excel.py'
if not RENDERER.exists():
    RENDERER=HERE/'build_ep128_leadership_excel.py'
WRITER_DIR=HERE
sys.path.insert(0,str(WRITER_DIR))
spec=importlib.util.spec_from_file_location('renderer_v2_under_test',RENDERER)
renderer=importlib.util.module_from_spec(spec)
spec.loader.exec_module(renderer)
NS={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}


def _source_fixture():
    methods=[]
    for identity,rate,total in (('method1',987.654321,123.456789),('method2',876.543219,133.456789)):
        methods.append(dict(id=identity,name=identity+' synthetic',status='CONDITIONAL_SCENARIO',reasons=[],
            assumptions=['SYNTHETIC_ONLY: matched evidence fixture, not real performance'],
            comm_model=dict(payload_ratio=2.8125,diagnostic_ms=0.123456789),
            per_rank=[dict(rank='dp0',rounds=[dict(round_id=7,source_cycle_ms=100.123456789,
                cycle_ms=total,per_device_tokens_s=rate,formula_terms={'source':100.123456789},reasons=[])])],
            selected_rank='dp0',aggregation='max_of_rank_mean_proxy',cycle_ms=total,
            per_device_tokens_s=rate,global_tokens_s=111222.333444,tpot_ms=23.456789,
            formula='CORE_ONLY R1=R0+core_terms; do not recompute',
            breakdown=[dict(key='source',label='源周期',value_ms=100.0,formula='core.R0',basis='measured'),
                       dict(key='extra',label='新增条件贡献',value_ms=total-100.0,formula='core.delta',basis='assumed')],
            detail_rows=[dict(rank='dp0',round_id=7,class_key='main_CSA_normal',component='dispatch',
                 source_ms=.123456789,target_ms=.234567891,delta_ms=.111111102,
                 weighted_delta_ms=3.222221958,basis='assumed',status='DIAGNOSTIC',formula='CORE D formula')]))
    ledger=dict(duration_ms=100.123456789,recorded_busy_ms=80.0,no_recorded_ms=20.123456789,
        category_total_ms=100.123456789,closed=True,scope='mapped source period',
        rows=[dict(key='owned',label='已归属层',duration_ms=80.0),dict(key='unrecorded',label='无设备任务记录',duration_ms=20.123456789)],
        intervals=[dict(start_ns=1793103275541889538,end_ns=1793103275621889538,duration_ms=80.0,key='owned',
                        label='已归属层',layer_owners=['main.0'],nonlayer_owners=[],recorded=True)])
    case=dict(name='SYNTHETIC_ONLY_BS32',config=dict(bs=32,draft_tokens=7,effective_tokens_per_round=5.2,
          token_basis='configured conditional',source_dp=8,source_devices=16,target_dp=64,target_devices=128),
        source_model=dict(num_hidden_layers=8),target_model=dict(main={'num_hidden_layers':61},draft={'num_hidden_layers':3}),
        captures=[dict(rank='dp0',sha256='fixture')],source=dict(status='SOURCE_CONDITIONAL_RATE',
            per_rank=[dict(rank='dp0',rounds=6,cycle_ms_mean=100.123456789,cycle_ms_p50=99.123456,
                cycle_ms_p95=105.123456,recorded_busy_ms_mean=80,no_recorded_ms_mean=20.123456789,
                conditional_per_rank_tokens_s=2222.222,conditional_per_device_tokens_s=1111.111,
                token_basis='configured BS/Neff conditional, not client measured',scope='HC_cycle_not_verified_request_iteration')],
            aggregate=dict(selected_rank='dp0',cycle_ms=100.123456789,per_device_tokens_s=777.777777,
                global_tokens_s=12444.444432,scope='source conditional proxy',aggregation='max_of_rank_mean_proxy'),
            class_stats=[dict(class_key='main_CSA_normal',scope='HC_marker_interval',samples=12,
                wall_ms_mean=4.567891234,busy_union_ms_mean=4.123456789,dispatch_ms_mean=.123456789,
                combine_ms_mean=.234567891,router_ms_mean=None,routed_mlp_ms_mean=None)]),
        measurements=[dict(rank='dp0',rounds=[dict(round_id=7,ledger=ledger,
            layers=[dict(role='main',layer=0,class_key='main_HCA_hash',scope='HC_marker_interval',
                wall_ms=4.555555555,busy_union_ms=4.111111111,dispatch_ms=.111111111,combine_ms=.222222222,
                router_ms=None,routed_mlp_ms=None,clean=False)],nonlayer_components=[])])],
        projection=dict(structure={'added_counts':{'main_CSA_normal':27}},methods=methods),
        review=dict(reasons=[],findings=['SYNTHETIC_ONLY']))
    return dict(schema_version=2,generated_at='2026-11-01T00:00:00+00:00',source_kind='offline_kernel_csv',
                cases=[case],failures=[],contract={'renderer':'display_only'})



def fixture():
    report = _source_fixture()
    case = report['cases'][0]
    case['projection']['display_method_set'] = 'layer_mean_methods'
    case['source']['class_stats'][0]['class_key'] = 'OLD_HASH_WITH_MAIN0_ONLY'
    for strict in case['projection']['methods']:
        strict.update(status='PENDING', reasons=['OLD_STRICT_COMPONENT_CALIBRATION_ONLY'])
        strict['detail_rows'] = [dict(rank='dp0', round_id='hc-3', component='router',
            target_layers=2, status='PENDING', reasons=['OLD_STRICT_ROUTER_ONLY']),
            dict(rank='dp0', round_id='hc-3', component='dispatch',
                 formula='2 * ((2.8125 * source_ms + 0.014)-source_ms)',
                 status='COMMUNICATION_SENSITIVITY_ONLY')]
    methods = []
    for identity, selected, cycle, rate in (('method1', 'dp7', 40.125, 519.4321),
                                           ('method2', 'dp3', 50.25, 413.8765)):
        ranks = []
        for rank in ('dp7', 'dp3'):
            chosen = rank == selected
            base = 0. if chosen else 9000.
            classes = []
            for role, attention, layers, count, wall in (
                    ('main', 'CSA', [2, 4, 6], 30, 2.12345),
                    ('main', 'HCA', [1, 3, 5, 7], 31, 3.12345),
                    ('draft', 'SWA', [0, 1, 2], 3, 4.12345)):
                classes.append(dict(role=role, attention=attention, reference_layers=layers,
                    samples=7 * len(layers), target_layers=count, wall_ms=wall + base,
                    dispatch_ms=.11123 + base, combine_ms=.22234 + base,
                    routed_compute_ms=.33345 + base, dispatch_target_ms=.41123 + base,
                    combine_target_ms=.62234 + base, routed_compute_target_ms=.83345 + base,
                    dispatch_delta_ms=.3, combine_delta_ms=.4, routed_compute_delta_ms=.5,
                    target_wall_ms=wall + 1.2 + base))
            roles = [dict(role=role, source_layer_envelopes_ms=source + base,
                target_layer_envelopes_ms=target + base, source_chain_ms=source + .25 + base,
                source_gap_ms=.25, reference_gap_mean_ms=.012345, added_gap_ms=.75,
                target_gap_ms=1., target_chain_ms=target + 1. + base,
                dispatch_delta_ms=.6, combine_delta_ms=.8, routed_mlp_delta_ms=1.)
                for role, source, target in (('main', 12., 20.), ('draft', 3., 5.))]
            ranks.append(dict(rank=rank, rounds=[], cycle_ms=cycle if chosen else cycle - 5,
                class_means=classes, role_details=roles,
                source_partition=dict(source_cycle_ms=22., source_main_chain_ms=12.25, source_draft_chain_ms=3.25,
                                      partition_total_ms=22., closed=True, scope='SYNTHETIC_SOURCE_PARTITION',
                                      cross_role_tail_residual_ms=cycle - 27.,
                                      residual_ms=cycle - 27.)))
        chosen = next(item for item in ranks if item['rank'] == selected)
        methods.append(dict(id=identity, name=identity + ' 层均值估算', status='LAYER_MEAN_ESTIMATE',
            reasons=[], assumptions=['SYNTHETIC conditional layer means'], warnings=['TARGET_HARDWARE_UNVALIDATED'],
            cycle_ms=cycle, per_device_tokens_s=rate, global_tokens_s=998877.123456,
            tpot_ms=7.76543 if identity == 'method1' else 9.76543, selected_rank=selected,
            aggregation='core_selected_max_rank_mean', per_rank=ranks,
            class_means=copy.deepcopy(chosen['class_means']), role_details=copy.deepcopy(chosen['role_details']),
            source_partition=copy.deepcopy(chosen['source_partition']),
            detail_rows=[dict(rank=selected, role='main', attention='CSA',
                              reference_layers=[2, 4, 6], core_evidence='SELECTED_CORE_DETAIL_ONLY')],
            breakdown=[dict(key=key, label=label, value_ms=value, formula='SYNTHETIC_CORE_ONLY')
                for key, label, value in (('main_layer_envelopes', '主模型目标层区间', 20.),
                    ('main_internal_gaps', '主内部衔接', 1.), ('draft_layer_envelopes', '草稿目标层区间', 5.),
                    ('draft_internal_gaps', '草稿内部衔接', 1.), ('residual_once', 'U仅一次', cycle - 27.))],
            communication=dict(extra_ms_per_call=.014 if identity == 'method1' else 0.),
            formula='SYNTHETIC_CORE_LAYER_MEANS: class means + gaps + U once; no renderer math'))
    case['projection']['layer_mean_methods'] = methods
    return report



def read_book(path):
    with ZipFile(path) as archive:
        wb=ET.fromstring(archive.read('xl/workbook.xml'))
        names=[node.attrib['name'] for node in wb.findall('s:sheets/s:sheet',NS)]
        result={}
        for index,name in enumerate(names,1):
            root=ET.fromstring(archive.read(f'xl/worksheets/sheet{index}.xml'))
            cells={}
            for node in root.findall('.//s:sheetData/s:row/s:c',NS):
                if node.attrib.get('t') in ('inlineStr','str'):
                    value=''.join(node.find('s:is',NS).itertext()) if node.find('s:is',NS) is not None else node.findtext('s:v',None,NS)
                else:
                    raw=node.findtext('s:v',None,NS)
                    value=float(raw) if raw is not None else None
                cells[node.attrib['r']]=value
            result[name]=(root,cells)
        return result,archive.namelist()



class RendererV2Tests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix='synthetic-layer-mean-renderer-')
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def build(self, data=None, **kwargs):
        report = self.root / 'report.json'
        report.write_text(json.dumps(data or fixture(), ensure_ascii=False), encoding='utf-8')
        out = self.root / 'export' / 'synthetic.xlsx'
        result = renderer.build_workbook(out, report=report, **kwargs)
        self.assertEqual(result, out)
        return out, read_book(out)

    def test_nine_sheets_copy_core_numbers_without_prediction_recalculation(self):
        data = fixture()
        out, (book, files) = self.build(data)
        self.assertEqual(tuple(book), renderer.SHEETS)
        for method in data['cases'][0]['projection']['layer_mean_methods']:
            for field in ('cycle_ms', 'per_device_tokens_s', 'tpot_ms'):
                self.assertIn(method[field], book['估算总览'][1].values())
                self.assertIn(str(method[field]), out.with_suffix('.html').read_text())
        for root, _ in book.values():
            self.assertEqual(root.findall('.//s:f', NS), [])
        self.assertTrue(any(path.startswith('xl/charts/chart') for path in files))
        self.assertIn('token/s/card', out.with_suffix('.html').read_text())
        self.assertNotIn('998877.123456', out.with_suffix('.html').read_text())

    def test_unavailable_result_hides_all_target_values_but_keeps_source_means(self):
        data = fixture()
        method = data['cases'][0]['projection']['layer_mean_methods'][0]
        method.update(status='UNAVAILABLE', reasons=['ACTUAL_INPUT_MISSING'],
                      cycle_ms=998877.665544, per_device_tokens_s=887766.554433, tpot_ms=776655.443322)
        method['role_details'][0]['target_chain_ms'] = 665544.332211
        out, (book, _) = self.build(data)
        text = out.with_suffix('.html').read_text()
        for sheet in ('估算总览', '方法1估算'):
            values = list(book[sheet][1].values())
            for forbidden in (998877.665544, 887766.554433, 776655.443322, 665544.332211):
                self.assertNotIn(forbidden, values)
                self.assertNotIn(str(forbidden), text)
        self.assertIn(3.12345, book['方法1估算'][1].values())
        self.assertIn('ACTUAL_INPUT_MISSING', text)

    def test_incomplete_breakdown_does_not_draw_fake_target_contribution(self):
        data = fixture()
        for method in data['cases'][0]['projection']['layer_mean_methods']:
            method['breakdown'][0]['value_ms'] = None
        _, (_, files) = self.build(data)
        self.assertFalse(any(path.startswith('xl/charts/chart') for path in files))

    def test_obsolete_method_sets_are_rejected_before_workbook_generation(self):
        for selected in ('methods', 'assumption_methods', 'source_point_methods', None):
            data = fixture()
            data['cases'][0]['projection']['display_method_set'] = selected
            with self.subTest(selected=selected), self.assertRaisesRegex(ValueError, '仅支持layer_mean_methods'):
                self.build(data)

    def test_legacy_reports_and_nonfinite_values_are_rejected(self):
        for data in ({'summaries': []}, {'schema_version': 1, 'cases': []}, {'schema_version': 2.0, 'cases': []}):
            with self.subTest(data=data), self.assertRaisesRegex(ValueError, 'schema_version=2'):
                self.build(data)
        data = fixture()
        data['cases'][0]['projection']['layer_mean_methods'][0]['cycle_ms'] = float('nan')
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            self.build(data)

    def test_archived_reference_takes_precedence_without_changing_core_results(self):
        data = fixture()
        data['communication_reference'] = {'unit': 'unknown', 'rows': [{'bs': 32, 'ep32_send': 7.123456789}]}
        reference = self.root / 'other.csv'
        reference.write_text('bs,ep32_send\n32,999999\n', encoding='utf-8')
        _, (book, _) = self.build(data, reference=reference)
        self.assertIn(7.123456789, book['通信参考原表'][1].values())
        self.assertNotIn('999999', book['通信参考原表'][1].values())
        self.assertIn(519.4321, book['估算总览'][1].values())

    def test_preview_reuses_layer_mean_layout_without_reading_any_capture(self):
        config = self.root / 'cases.json'
        config.write_text(json.dumps({'schema_version': 2, 'sample_model': {'num_hidden_layers': 8},
            'target_model': {'main': {'num_hidden_layers': 61}}, 'cases': [{'name': 'PREVIEW_ONLY', 'bs': 32,
            'path': '/must/not/read/kernel_details.csv', 'draft_tokens': 7}]}), encoding='utf-8')
        out = self.root / 'preview.xlsx'
        renderer.build_workbook(out, cases=config, preview=True)
        book, files = read_book(out)
        self.assertEqual(tuple(book), renderer.SHEETS)
        self.assertFalse(any(path.startswith('xl/charts/chart') for path in files))
        overview = book['估算总览'][1]
        for label in ('完整周期 · ms', 'TPOT · ms/token', '预测单卡吞吐 · token/s/card'):
            cards = [cell for cell, value in overview.items() if value == label]
            self.assertEqual(len(cards), 2)
            for cell in cards:
                column = ''.join(c for c in cell if c.isalpha())
                row = int(''.join(c for c in cell if c.isdigit()))
                self.assertEqual(overview[column + str(row + 1)], renderer.PENDING)
        self.assertIn('版式预览', str(overview))
        with self.assertRaisesRegex(ValueError, 'preview'):
            renderer.build_workbook(out, cases=config)

    def test_preview_rejects_legacy_and_nonfinite_configuration(self):
        config = self.root / 'legacy.json'
        for data, error in (({'sample_model': {}, 'cases': []}, 'schema_version=2'),
                            ({'schema_version': 2, 'cases': [{'bs': float('nan')}]}, 'nonfinite')):
            config.write_text(json.dumps(data), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, error):
                renderer.build_workbook(self.root / 'invalid.xlsx', cases=config, preview=True)

    def test_cli_can_render_preview_without_capture_or_estimator(self):
        config = self.root / 'preview-cli.json'
        config.write_text(json.dumps({'schema_version': 2, 'sample_model': {},
                                     'cases': [{'name': 'ONLY_CONFIG', 'bs': 32}]}), encoding='utf-8')
        out = self.root / 'cli.xlsx'
        result = subprocess.run([sys.executable, '-S', str(RENDERER), '--cases', str(config), '--preview', '--out', str(out)],
                                capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(out.is_file())
        self.assertTrue(out.with_suffix('.html').is_file())

    def test_html_escapes_names_formulas_and_evidence(self):
        data = fixture()
        data['cases'][0]['name'] = '<script>alert("bad")</script>'
        data['cases'][0]['projection']['layer_mean_methods'][0]['formula'] = '<svg onload=bad>'
        data['cases'][0]['projection']['layer_mean_methods'][0]['warnings'] = ['<img src=x onerror=bad>']
        out, _ = self.build(data)
        text = out.with_suffix('.html').read_text()
        for forbidden in ('<script>', '<svg onload', '<img src=x'):
            self.assertNotIn(forbidden, text)
        self.assertIn('&lt;script&gt;', text)
        self.assertIn('&lt;svg', text)
        self.assertIn('&lt;img', text)

    def test_renderer_has_no_estimator_or_legacy_import(self):
        tree = ast.parse(RENDERER.read_text())
        imports = [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names]
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.module == 'profile_ep128':
                # Old JSON redraw may read the pure source-only summary helper;
                # target estimation, rank selection and CSV access stay in core.
                self.assertEqual([alias.name for alias in node.names], ['source_sweep_summary'])
            else:
                imports.append(node.module or '')
        self.assertFalse(any('profile_ep128' in name or 'source_point' in name or 'legacy' in name for name in imports))
        functions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
        self.assertFalse(any('assumption' in name or 'point_diagnostic' in name for name in functions))


if __name__ == '__main__':
    unittest.main()
