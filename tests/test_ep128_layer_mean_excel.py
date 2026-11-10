"""Synthetic old-style layer-mean presentation contracts; no NPU/data claims."""
import copy
import json
import posixpath
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from zipfile import ZipFile
from pathlib import Path
import tempfile
import unittest
import unicodedata
from unittest import mock

from test_ep128_v2_excel import fixture, read_book, renderer


def layer_mean_fixture():
    return fixture()


def residual_report():
    report = layer_mean_fixture()
    case = report['cases'][0]
    case['measurements'], case['captures'] = [], []
    for rank, residual in (('dp7', 13.125), ('dp3', 23.25)):
        parents, children, summary = [], [], []
        for window, count in (('main_to_draft', 8), ('draft_to_nextmain', 7)):
            for order in range(1, count + 1):
                parents.append(dict(window_key=window, stage_key=window + '_' + str(order),
                    stage_order=order, label=rank + '_PARENT_' + window + '_' + str(order),
                    level=0, parent_stage_key=None, duration_ms_mean=residual / 15,
                    rounds=2, matched_windows=2, status='MATCHED', boundary_basis='SYNTHETIC_NATIVE_ANCHORS'))
            summary.append(dict(window_key=window, rounds=2, duration_ms_mean=count * residual / 15))
        for order in range(1, 8):
            children.append(dict(window_key='draft_to_nextmain', stage_key='markov_' + str(order),
                stage_order=order, label=rank + '_CHILD_NOT_ADDED_' + str(order), level=1,
                parent_stage_key='draft_to_nextmain_3', duration_ms_mean=residual / 105,
                rounds=2, matched_windows=2, status='MATCHED'))
        case['measurements'].append(dict(rank=rank, rounds=[], step_residuals=dict(status='DEVICE_ONLY',
            trace=dict(status='NOT_FOUND'), residual_ms_mean=residual, stage_summary=parents,
            substage_summary=children, summary=summary)))
        case['captures'].append(dict(rank=rank, sha256='SYNTHETIC_' + rank))
    return report


def unresolved_window_report():
    """Synthetic 71-round structure: 3.25 ms unknown U1 plus seven known U2 stages."""
    report = residual_report()
    case = report['cases'][0]
    capture, method = case['measurements'][0], case['projection']['layer_mean_methods'][0]
    capture['rank'] = method['selected_rank'] = case['captures'][0]['rank'] = 'dp1'
    diagnostic = capture['step_residuals']
    diagnostic['rank'] = 'dp1'
    for item in diagnostic['summary']:
        item['rounds'] = 71
        if item['window_key'] == 'main_to_draft':
            item['duration_ms_mean'] = 3.25
    unknown = dict(window_key='main_to_draft', stage_key='unresolved', stage_order=1,
                   label='SYNTHETIC_UNRESOLVED_WINDOW', level=0, parent_stage_key=None,
                   rounds=71, matched_windows=0, status='UNRESOLVED', duration_ms_mean=None,
                   duration_ns_total=230750000, reason='SYNTHETIC_MISSING_FINAL_HIDDEN_NORM')
    diagnostic['stage_summary'] = [unknown] + [row for row in diagnostic['stage_summary']
                                               if row['window_key'] == 'draft_to_nextmain']
    for row in diagnostic['stage_summary'][1:] + diagnostic['substage_summary']:
        row.update(rounds=71, matched_windows=71)
        row['label'] = row['label'].replace('dp7_', 'dp1_', 1)
    total = sum(item['duration_ms_mean'] for item in diagnostic['summary'])
    diagnostic['residual_ms_mean'] = total
    old = method['source_partition']['residual_ms']
    method['source_partition'].update(residual_ms=total, cross_role_tail_residual_ms=total)
    method['cycle_ms'] += total - old
    method['breakdown'][-1]['value_ms'] = total
    for rank in method['per_rank']:
        if rank['rank'] == 'dp7':
            rank.update(rank='dp1', cycle_ms=method['cycle_ms'],
                        source_partition=copy.deepcopy(method['source_partition']))
    return report


DRILLDOWN_TITLES = ('SWA层内阶段展开', '下一主模型KV与输入准备 · 阶段展开',
                    '下一主metadata及输入映射准备 · 阶段展开')
MAIN_DRILLDOWN_TITLES = ('CSA层内阶段展开', 'HCA层内阶段展开')
SWA_STAGE_KEYS = ('attention_prepare', 'attention', 'attention_output', 'attention_post',
                  'moe_prepare', 'dispatch', 'experts', 'combine', 'ffn_output', 'ffn_post')


def attach_target_stage_contract(report):
    """Synthetic core output, kept independent of the display implementation."""
    case = report['cases'][0]
    for method in case['projection']['layer_mean_methods']:
        capture = next(row for row in case['measurements'] if row['rank'] == method['selected_rank'])
        diagnostic = capture.get('layer_diagnostics', capture.get('swa_diagnostics', {}))
        groups = []
        for parent in method['class_means']:
            role, attention = parent['role'], parent['attention']
            reference_layers = parent['reference_layers']
            group = dict(role=role, attention=attention, reference_layers=list(reference_layers),
                         target_layers=parent['target_layers'], rounds=2, status='UNAVAILABLE',
                         reason='SYNTHETIC_TARGET_STAGES_NOT_CAPTURED', stage_rows=[],
                         source_wall_ms=parent['wall_ms'], target_wall_ms=parent['target_wall_ms'],
                         target_total_ms=parent['target_layers'] * parent['target_wall_ms'])
            source_stages = [row for row in diagnostic.get('stage_summary', [])
                             if row['role'] == role and row['attention'] == attention
                             and row['layer'] in reference_layers]
            if len(source_stages) == 10 * len(reference_layers):
                stages = []
                for order, key in enumerate(SWA_STAGE_KEYS, 1):
                    matches = [row for row in source_stages if row['stage_key'] == key]
                    source = sum(row['duration_ms_mean'] for row in matches) / len(matches)
                    delta = {'dispatch': parent['dispatch_delta_ms'], 'combine': parent['combine_delta_ms'],
                             'experts': parent['routed_compute_delta_ms']}.get(key, 0.)
                    target = source + delta
                    stages.append(dict(stage_key=key, stage_order=order, label=matches[0]['label'],
                                       source_ms=source, delta_ms=delta, target_ms=target,
                                       target_total_ms=target * parent['target_layers']))
                parent.update(wall_ms=sum(row['source_ms'] for row in stages),
                              target_wall_ms=sum(row['target_ms'] for row in stages))
                group.update(status='MATCHED', reason=None, stage_rows=stages,
                             source_wall_ms=parent['wall_ms'], target_wall_ms=parent['target_wall_ms'],
                             target_total_ms=parent['target_layers'] * parent['target_wall_ms'])
            groups.append(group)
        for role in method['role_details']:
            role['target_layer_envelopes_ms'] = sum(parent['target_layers'] * parent['target_wall_ms']
                                                  for parent in method['class_means'] if parent['role'] == role['role'])
            role['target_chain_ms'] = role['target_layer_envelopes_ms'] + role['target_gap_ms']
        components = [method['role_details'][0]['target_layer_envelopes_ms'], method['role_details'][0]['target_gap_ms'],
                      method['role_details'][1]['target_layer_envelopes_ms'], method['role_details'][1]['target_gap_ms'],
                      method['source_partition']['residual_ms']]
        method['cycle_ms'] = sum(components)
        for item, value in zip(method['breakdown'], components):
            item['value_ms'] = value
        for group in groups:
            for stage in group['stage_rows']:
                stage['cycle_share'] = stage['target_total_ms'] / method['cycle_ms']
        method['target_stage_breakdowns'] = groups
        selected = next(rank for rank in method['per_rank'] if rank['rank'] == method['selected_rank'])
        selected.update(cycle_ms=method['cycle_ms'], class_means=copy.deepcopy(method['class_means']),
                        role_details=copy.deepcopy(method['role_details']), target_stage_breakdowns=copy.deepcopy(groups))
    return report


def drilldown_report():
    """Two independently measured ranks, with three unequal SWA layer means."""
    report = residual_report()
    for capture in report['cases'][0]['measurements']:
        rank = capture['rank']
        scale = 1. if rank == 'dp7' else 3.
        diagnostic = capture['step_residuals']
        for old, parent_key, count in (('draft_to_nextmain_5', 'main_inputs', 7),
                                       ('draft_to_nextmain_6', 'main_metadata', 3)):
            parent = next(row for row in diagnostic['stage_summary'] if row['stage_key'] == old)
            parent['stage_key'] = parent_key
            divisor = count * (count + 1) / 2
            for order in range(1, count + 1):
                diagnostic['substage_summary'].append(dict(window_key='draft_to_nextmain',
                    stage_key=parent_key + '_' + str(order), stage_order=parent['stage_order'] * 1000 + order,
                    label=rank + '_' + parent_key + '_DETAIL_' + str(order),
                    level=1, parent_stage_key=parent_key, rounds=2, matched_windows=2,
                    duration_ms_mean=parent['duration_ms_mean'] * order / divisor,
                    status='MATCHED', boundary_basis='SYNTHETIC_NATIVE_ANCHORS'))
        layer_summary, stage_summary = [], []
        for layer in range(3):
            values = [scale * (layer + 1) * (order + 1) / 100 for order in range(10)]
            layer_summary.append(dict(role='draft', attention='SWA', layer=layer, rounds=2,
                                      wall_ms_mean=sum(values), reference_eligible=True))
            for order, (stage_key, value) in enumerate(zip(SWA_STAGE_KEYS, values), 1):
                stage_summary.append(dict(role='draft', attention='SWA', layer=layer,
                    stage_key=stage_key, stage_order=order, label=rank + '_SWA_DETAIL_' + str(order),
                    rounds=2, matched_windows=2, duration_ms_mean=value,
                    status='MATCHED', reason=None))
        capture['swa_diagnostics'] = dict(status='MATCHED', layer_summary=layer_summary,
                                           stage_summary=stage_summary)
    return attach_target_stage_contract(report)


def full_layer_drilldown_report():
    """Main CSA/HCA plus draft SWA, all per-layer means kept independently."""
    report = drilldown_report()
    for capture in report['cases'][0]['measurements']:
        diagnostic = copy.deepcopy(capture['swa_diagnostics'])
        rank = capture['rank']
        rank_scale = 1. if rank == 'dp7' else 3.
        for attention, layers, multiplier in (('CSA', (2, 4, 6), 5.),
                                               ('HCA', (0, 1, 3, 5, 7), 11.)):
            for layer in layers:
                values = [rank_scale * multiplier * (layer + 1) * (order + 1) / 100
                          for order in range(10)]
                diagnostic['layer_summary'].append(dict(role='main', attention=attention,
                    layer=layer, rounds=2, wall_ms_mean=sum(values), reference_eligible=layer != 0))
                for order, (key, value) in enumerate(zip(SWA_STAGE_KEYS, values), 1):
                    diagnostic['stage_summary'].append(dict(role='main', attention=attention,
                        layer=layer, stage_key=key, stage_order=order,
                        label=rank + '_' + attention + '_DETAIL_' + str(order), rounds=2,
                        matched_windows=2, duration_ms_mean=value, status='MATCHED', reason=None))
        capture['layer_diagnostics'] = diagnostic
    return attach_target_stage_contract(report)


def bs_trend_report():
    """Unsorted BS and changing selected ranks; values are synthetic core output."""
    report = layer_mean_fixture()
    template = report['cases'][0]
    report['cases'] = []
    for bs in (32, 4, 16, 8):
        case = copy.deepcopy(template)
        # Names and paths deliberately disagree with BS and differ per capture.
        case['name'] = 'SYNTHETIC_LABEL_BS' + str(100 - bs)
        case['config'].update(bs=bs, name=case['name'], path='synthetic/bs' + str(bs),
                              inputs=[{'rank': 'dp0', 'path': 'synthetic/' + str(bs) + '.csv'}])
        for number, method in enumerate(case['projection']['layer_mean_methods'], 1):
            selected = 'dp7' if (bs in (4, 16)) == (number == 1) else 'dp3'
            components = [number * 10. + bs + .125, bs / 8.,
                          number * 5. + bs / 4., bs / 16., number * 2. + bs / 2.]
            for row, value in zip(method['breakdown'], components):
                row['value_ms'] = value
            method.update(selected_rank=selected, cycle_ms=sum(components),
                          tpot_ms=number * 20. + bs / 7.,
                          per_device_tokens_s=number * 1000. + bs * 11.125)
            method['role_details'][0].update(target_layer_envelopes_ms=components[0],
                target_gap_ms=components[1], target_chain_ms=sum(components[:2]))
            method['role_details'][1].update(target_layer_envelopes_ms=components[2],
                target_gap_ms=components[3], target_chain_ms=sum(components[2:4]))
            method['source_partition'].update(cross_role_tail_residual_ms=components[4],
                                               residual_ms=components[4])
            for rank in method['per_rank']:
                chosen = rank['rank'] == selected
                rank['cycle_ms'] = method['cycle_ms'] if chosen else 900000. + bs
                rank['role_details'] = copy.deepcopy(method['role_details'])
                rank['source_partition'] = copy.deepcopy(method['source_partition'])
                if not chosen:
                    for role in rank['role_details']:
                        role['target_chain_ms'] += 900000.
            if bs == 16:
                # Canonical component keys, rather than JSON row position,
                # determine the cross-BS series order.
                method['breakdown'].reverse()
        report['cases'].append(case)
    return report


def operator_drilldown_report():
    """Synthetic independent core contract: concurrent calls, gaps and signed models."""
    report = full_layer_drilldown_report()
    for method in report['cases'][0]['projection']['layer_mean_methods']:
        next(row for row in method['class_means'] if row['attention'] == 'SWA')['dispatch_delta_ms'] = -.03
    attach_target_stage_contract(report)
    native = 'SYNTHETIC_SAME_NATIVE_NAME_' + 'long_component_' * 12
    for method in report['cases'][0]['projection']['layer_mean_methods']:
        for group in method['target_stage_breakdowns']:
            # A measured zero-length phase remains a complete, empty partition.
            first, second = group['stage_rows'][:2]
            transfer = first['source_ms']
            for field in ('source_ms', 'target_ms'):
                first[field] = 0.
                second[field] += transfer
            first.update(target_total_ms=0., cycle_share=0.)
            second.update(target_total_ms=second['target_ms'] * group['target_layers'])
            second['cycle_share'] = second['target_total_ms'] / method['cycle_ms']
            for stage in group['stage_rows']:
                source, count = stage['source_ms'], group['target_layers']
                rows, kernels, members = [], [], []
                example = dict(rank=method['selected_rank'], round_id='SYNTHETIC_ROUND_4',
                    layer=group['reference_layers'][0], row_index=stage['stage_order'] * 10,
                    row_indices=[stage['stage_order'] * 10], start_ns='100', end_ns='200', duration_ns=100)
                for suffix, width, duration, calls in (('A', 16, .8 * source, 2.), ('B', 32, .35 * source, 1.)):
                    member = dict(key='SYNTHETIC_' + group['attention'] + '_' + stage['stage_key'] + '_' + suffix,
                        name=native, op_type='MatMul', input_shapes=f'{stage["stage_order"]},{width};{width},16',
                        output_shapes=f'{stage["stage_order"]},16', input_dtypes='BF16;BF16', output_dtypes='BF16',
                        stream_id=suffix, task_type='AI_CORE')
                    members.append(dict(member, instances=1))
                    if source:
                        kernels.append(dict(member, calls_mean=calls, kernel_sum_ms=duration,
                            duration_mean_ms=duration / calls, duration_p90_ms=duration / calls,
                            example={**example, 'row_index': example['row_index'] + (suffix == 'B')}))
                def append_row(key, kind, value, row_members, sample, delta=0.):
                    rows.append(dict(key=key, label='SYNTHETIC_' + kind + '_' + stage['stage_key'], kind=kind,
                        source_ms=value, delta_ms=delta, target_ms=value + delta,
                        target_total_ms=(value + delta) * count,
                        cycle_share=(value + delta) * count / method['cycle_ms'], members=row_members, example=sample))
                if source:
                    append_row('single', 'operator', .45 * source, members[:1], example)
                    append_row('parallel', 'parallel', .35 * source, members, example)
                    gap_duration = stage['stage_order'] * 1000
                    gap = {**example, 'start_ns': '100', 'end_ns': str(100 + gap_duration), 'duration_ns': gap_duration,
                           'before': {**members[0], 'row_index': 17}, 'after': {**members[1], 'row_index': 29}}
                    append_row('gap', 'gap', .2 * source, [], gap)
                if stage['delta_ms']:
                    append_row('model_' + stage['stage_key'], 'model_delta', 0., [], None, stage['delta_ms'])
                stage['operator_breakdown'] = dict(status='MATCHED', reason=None, rows=rows, kernels=kernels,
                    reference_layers=group['reference_layers'], rounds=group['rounds'], note='SYNTHETIC_OPERATOR_SCOPE')
    return report


class LayerMeanRendererTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='ep128-layer-mean-excel-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def build(self, report):
        before = copy.deepcopy(report)
        out = self.root / 'layer-means.xlsx'
        renderer.build_workbook(out, report=self.root / 'source.json', report_data=report,
                                expected_method_set='layer_mean_methods')
        self.assertEqual(report, before)
        return out, read_book(out)

    def chart_parts_by_sheet(self, out, book):
        """Follow actual OOXML relationships, independent of chart numbering."""
        ns = {'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships',
              'p': 'http://schemas.openxmlformats.org/package/2006/relationships',
              'c': 'http://schemas.openxmlformats.org/drawingml/2006/chart'}
        result = {name: [] for name in book}
        with ZipFile(out) as archive:
            def related(part):
                directory, filename = posixpath.split(part)
                rels = posixpath.join(directory, '_rels', filename + '.rels')
                if rels not in archive.namelist():
                    return {}
                root = ET.fromstring(archive.read(rels))
                return {row.get('Id'): posixpath.normpath(posixpath.join(directory, row.get('Target')))
                        for row in root.findall('p:Relationship', ns)}
            for index, name in enumerate(book, 1):
                part = f'xl/worksheets/sheet{index}.xml'
                links = related(part)
                for drawing in book[name][0].findall('{*}drawing'):
                    drawing_part = links[drawing.get('{' + ns['r'] + '}id')]
                    drawing_links = related(drawing_part)
                    root = ET.fromstring(archive.read(drawing_part))
                    for node in root.findall('.//c:chart', ns):
                        chart_part = drawing_links[node.get('{' + ns['r'] + '}id')]
                        result[name].append(ET.fromstring(archive.read(chart_part)))
        return result

    def chart_reference_cells(self, book, reference):
        match = re.fullmatch(r"'([^']+)'!\$([A-Z]+)\$(\d+):\$([A-Z]+)\$(\d+)", reference)
        self.assertIsNotNone(match, reference)
        sheet, first_col, first, last_col, last = match.groups()
        self.assertEqual(first_col, last_col)
        return [book[sheet][1].get(first_col + str(row)) for row in range(int(first), int(last) + 1)]

    def chart_title(self, chart):
        return ''.join(chart.find('{*}chart/{*}title').itertext())

    def assert_bar_cache_matches_own_sheet(self, chart, book, sheet_name):
        ns = {'c': 'http://schemas.openxmlformats.org/drawingml/2006/chart'}
        self.assertEqual(chart.find('.//c:barDir', ns).get('val'), 'bar')
        self.assertEqual(chart.find('.//c:grouping', ns).get('val'), 'clustered')
        result = []
        for series in chart.findall('.//c:barChart/c:ser', ns):
            labels = series.find('c:cat/c:strRef', ns)
            values = series.find('c:val/c:numRef', ns)
            self.assertIsNotNone(labels)
            self.assertIsNotNone(values)
            label_ref = labels.findtext('c:f', namespaces=ns)
            value_ref = values.findtext('c:f', namespaces=ns)
            self.assertTrue(label_ref.startswith("'" + sheet_name + "'!"), label_ref)
            self.assertTrue(value_ref.startswith("'" + sheet_name + "'!"), value_ref)
            cached_labels = [point.findtext('c:v', namespaces=ns)
                             for point in labels.findall('c:strCache/c:pt', ns)]
            cached_values = [float(point.findtext('c:v', namespaces=ns))
                             for point in values.findall('c:numCache/c:pt', ns)]
            self.assertEqual(cached_labels, self.chart_reference_cells(book, label_ref))
            self.assertEqual(cached_values, self.chart_reference_cells(book, value_ref))
            self.assertEqual(int(values.find('c:numCache/c:ptCount', ns).get('val')), len(cached_values))
            result.append((cached_labels, cached_values))
        self.assertTrue(result)
        return result

    def assert_line_cache_matches_sheet(self, chart, book, expected_bs):
        ns = {'c': 'http://schemas.openxmlformats.org/drawingml/2006/chart'}
        self.assertIsNotNone(chart.find('.//c:lineChart', ns))
        self.assertEqual(chart.find('c:chart/c:dispBlanksAs', ns).get('val'), 'gap')
        for series in chart.findall('.//c:lineChart/c:ser', ns):
            categories = series.find('c:cat/c:strRef', ns)
            self.assertIsNotNone(categories)
            labels = [row.findtext('c:v', namespaces=ns)
                      for row in categories.findall('c:strCache/c:pt', ns)]
            self.assertEqual(labels, list(map(str, expected_bs)))
            self.assertEqual([float(value) for value in self.chart_reference_cells(
                book, categories.findtext('c:f', namespaces=ns))], expected_bs)
            values = series.find('c:val/c:numRef', ns)
            self.assertIsNotNone(values)
            self.assertEqual(int(values.find('c:numCache/c:ptCount', ns).get('val')), len(expected_bs))
            actual = {int(row.get('idx')): float(row.findtext('c:v', namespaces=ns))
                      for row in values.findall('c:numCache/c:pt', ns)}
            cells = self.chart_reference_cells(book, values.findtext('c:f', namespaces=ns))
            self.assertEqual(actual, {index: value for index, value in enumerate(cells)
                                      if isinstance(value, (int, float)) and not isinstance(value, bool)})
        return chart.findall('.//c:lineChart/c:ser', ns)

    def test_case_acceptance_label_requires_an_explicit_consistent_mean_fraction(self):
        # All cases are synthetic declarations, not measured acceptance rates.
        valid = dict(bs=8, draft_tokens=9, effective_tokens_per_round=6.399999999999999,
                     acceptance=.6, acceptance_semantics='mean_accepted_draft_fraction')
        self.assertEqual(renderer._case_parameter_values({'config': valid}), ['8', '9', '6.4', '60%'])
        variants = [
            {key: value for key, value in valid.items() if key != 'acceptance'},
            {key: value for key, value in valid.items() if key != 'acceptance_semantics'},
            {key: value for key, value in valid.items() if key != 'effective_tokens_per_round'},
            {key: value for key, value in valid.items() if key != 'draft_tokens'},
            {**valid, 'acceptance_semantics': 'conditional_per_position_probability'},
            {**valid, 'effective_tokens_per_round': 5.2},
            {**valid, 'acceptance': True},
            {**valid, 'acceptance': '0.6'},
            {**valid, 'acceptance': -0.1},
            {**valid, 'acceptance': 1.1},
            {**valid, 'draft_tokens': True},
            {**valid, 'effective_tokens_per_round': True},
            # A synthetic length alone never supplies an omitted Neff/rate.
            {'bs': 8, 'draft_tokens': 9, 'synthetic_acceptance_length': 6.4},
        ]
        for config in variants:
            with self.subTest(config=config):
                before = copy.deepcopy(config)
                self.assertEqual(renderer._case_parameter_values({'config': config})[-1], '未声明')
                self.assertEqual(config, before)
        for rate, effective, label in ((0., 1., '0%'), (1., 10., '100%'), (.625, 6.625, '62.5%')):
            with self.subTest(rate=rate):
                config = {**valid, 'acceptance': rate, 'effective_tokens_per_round': effective}
                self.assertEqual(renderer._case_parameter_values({'config': config})[-1], label)
        self.assertEqual(renderer._case_parameter_values({}), ['未声明'] * 4)

    def test_mixed_speculation_parameters_are_visible_without_merging_cases_or_charts(self):
        report = bs_trend_report()
        templates = {case['config']['bs']: case for case in report['cases']}
        report['cases'] = []
        declarations = [('SYNTHETIC_Z_FIRST', 8, 9, 6.399999999999999),
                        ('SYNTHETIC_A_SECOND', 8, 7, 5.2),
                        ('SYNTHETIC_M_THIRD', 4, 9, 6.399999999999999),
                        ('SYNTHETIC_B_FOURTH', 4, 7, 5.2)]
        expected_rows, parameter_lines = [], []
        for name, bs, draft, effective in declarations:
            case = copy.deepcopy(templates[bs])
            case['name'] = name
            case['config'].update(name=name, bs=bs, draft_tokens=draft, effective_tokens_per_round=effective,
                                  synthetic_acceptance_length=effective, acceptance=.6,
                                  acceptance_semantics='mean_accepted_draft_fraction')
            report['cases'].append(case)
            neff = '6.4' if draft == 9 else '5.2'
            expected_rows.append([name, str(bs), str(draft), neff, '60%'])
            parameter_lines.append(f'BS={bs}；草稿S={draft}；Neff={neff}（含bonus）；'
                                   '声明的平均草稿接受比例=60%（配置假设，非实测）')
        groups = renderer._bs_comparison_groups(report)
        self.assertEqual([[case['config']['draft_tokens'] for case in group['cases']] for group in groups],
                         [[9, 9], [7, 7]])
        self.assertEqual([[case['config']['bs'] for case in group['cases']] for group in groups], [[4, 8], [4, 8]])
        self.assertEqual(renderer._case_parameter_rows(report), expected_rows)
        out, (book, _) = self.build(report)
        self.assertEqual(tuple(book), renderer.SHEETS)
        overview = book['估算总览'][1]
        title_row = next(int(cell[1:]) for cell, value in overview.items()
                         if re.fullmatch(r'A\d+', cell) and value == '采样配置对照')
        header_row = title_row + 2
        self.assertEqual([overview.get(column + str(header_row)) for column in ('A', 'E', 'F', 'G', 'I')],
                         renderer.CASE_PARAMETERS_HEADERS)
        for offset, expected in enumerate(expected_rows, 1):
            self.assertEqual([overview.get(column + str(header_row + offset))
                              for column in ('A', 'E', 'F', 'G', 'I')], expected)
        for sheet_name in ('估算总览', '方法1估算', '方法2估算'):
            notes = [value for cell, value in book[sheet_name][1].items()
                     if re.fullmatch(r'A\d+', cell) and value in parameter_lines]
            self.assertEqual(notes, parameter_lines)
            self.assertNotIn('Neff=6.399999', ' '.join(str(value) for value in book[sheet_name][1].values()))
        ns = {'c': 'http://schemas.openxmlformats.org/drawingml/2006/chart'}
        charts = self.chart_parts_by_sheet(out, book)
        for sheet_name, expected_count in (('估算总览', 6), ('方法1估算', 2), ('方法2估算', 2)):
            # S sweeps are additional views; this contract covers fixed-S BS curves.
            lines = [chart for chart in charts[sheet_name] if chart.find('.//c:lineChart', ns) is not None
                     and '随BS变化' in self.chart_title(chart)]
            self.assertEqual(len(lines), expected_count)
            for chart in lines:
                self.assert_line_cache_matches_sheet(chart, book, [4, 8])
        for group_index, group in enumerate(groups):
            chart = charts['估算总览'][group_index * 3]
            for series_index, identity in enumerate(('method1', 'method2')):
                actual = [float(point.findtext('c:v', namespaces=ns)) for point in
                          chart.findall('.//c:ser', ns)[series_index].findall('c:val/c:numRef/c:numCache/c:pt', ns)]
                expected = [next(method for method in case['projection']['layer_mean_methods']
                                 if method['id'] == identity)['per_device_tokens_s'] for case in group['cases']]
                self.assertEqual(actual, expected)
        preview = out.with_suffix('.html').read_text(encoding='utf-8')
        start = preview.index('class="case-parameters"')
        table = preview[start:preview.index('</table>', start)]
        self.assertEqual(sorted(expected_rows, key=lambda row: table.index(row[0])), expected_rows)
        for name, *_ in declarations:
            self.assertIn('<td>' + name + '</td>', table)
        for line in parameter_lines:
            self.assertEqual(preview.count('<p>' + line + '</p>'), 3)
        self.assertNotIn('Neff=6.399999', preview)
        self.assertIn('非实测', table)
        self.assertTrue(renderer.RENDERER_VERSION.startswith('20261110.'))

    def test_operator_preview_closes_signed_models_and_keeps_largest_remaining_gap(self):
        report = operator_drilldown_report()
        method = report['cases'][0]['projection']['layer_mean_methods'][0]
        before = copy.deepcopy(report)
        data = renderer._method_operator_data(method, 'draft', 'SWA', limit=1)
        self.assertTrue(data['available'], data['note'])
        group = next(row for row in method['target_stage_breakdowns'] if row['attention'] == 'SWA')
        rows = data['target_rows']
        self.assertAlmostEqual(sum(row[3] for row in rows), group['target_total_ms'])
        self.assertAlmostEqual(sum(row[1] + row[2] for row in rows), group['target_total_ms'])
        self.assertTrue(any(row[2] < 0 for row in rows))
        self.assertTrue(any(row[2] > 0 for row in rows))
        self.assertTrue(all(row[1] >= 0 for row in rows))
        remainder_gap = next(row for row in data['target_items'] if row.get('remainder') and row['kind'] == 'gap')
        self.assertIn('合计（非单一热点）', remainder_gap['label'])
        self.assertEqual(remainder_gap['example']['duration_ns'], 10000)
        gap_row = next(row for row in rows if '无设备记录区间' in row[0])
        self.assertIn('前=MatMul', gap_row[-1])
        self.assertIn('[s=A,row=17]', gap_row[-1])
        self.assertIn('后=MatMul', gap_row[-1])
        self.assertIn('[s=B,row=29]', gap_row[-1])
        for direction in ('before', 'after'):
            self.assertIn(remainder_gap['example'][direction]['name'], gap_row[-1])
        self.assertIn('模型增减', data['total_row'][-1])
        self.assertEqual(report, before)

    def test_operator_svg_wraps_cjk_and_mixed_labels_without_changing_signed_bars(self):
        labels = ['模型修正与无记录区间' * 20,
                  '#12 CSA输出 · QuantMatmul BF16[s18] 输入[16,7168] A<& ' * 8,
                  'MatMul FP16']
        coverage_legend = '原覆盖：' + '按目标同类层数展开的源互斥记录覆盖；' * 8
        model_legend = '模型修正：' + '独立显示有符号增减，不视为已验证优化收益；' * 8
        markup = renderer._operator_svg('SYNTHETIC_SIGNED_CHART', labels,
            [(coverage_legend, [4., 8., 2.], '168B8D'),
             (model_legend, [-2., .5, 0.], 'D4793A')], 'ms')
        svg = ET.fromstring(re.search(r'<svg\b.*?</svg>', markup, re.DOTALL).group())
        visible_texts = [''.join([node.text or '', *(line.text or '' for line in node.findall('{*}tspan'))])
                         for node in svg.findall('{*}text')]
        self.assertIn(coverage_legend, visible_texts)
        self.assertIn(model_legend, visible_texts)
        texts = [node for node in svg.findall('{*}text')
                 if node.findall('{*}tspan') and node.findtext('{*}title') in labels]
        self.assertEqual([node.findtext('{*}title') for node in texts], labels)
        for index, node in enumerate(texts):
            lines = node.findall('{*}tspan')
            if index < 2:
                self.assertGreater(len(lines), 2)
            else:
                self.assertEqual(len(lines), 1)
            self.assertEqual(''.join(line.text or '' for line in lines), labels[index])
            for line in lines:
                cells = sum(2 if unicodedata.east_asian_width(char) in ('W', 'F') else 1 for char in line.text or '')
                self.assertLessEqual(cells, 68)
                self.assertEqual(line.get('x'), '12')
            self.assertFalse(any('…' in (line.text or '') for line in lines))
            if index + 1 < len(texts):
                baseline = float(node.get('y'))
                for line in lines:
                    baseline = float(line.get('y')) if line.get('y') is not None else baseline + float(line.get('dy', 0))
                self.assertLess(baseline, float(texts[index + 1].get('y')))
                self.assertGreater(float(texts[index + 1].get('y')) - float(node.get('y')), 44)
        rectangles = [node for node in svg.findall('{*}rect') if node.find('{*}title') is not None]
        zero = float(svg.find('{*}line').get('x1'))
        self.assertEqual(len(rectangles), 6)
        self.assertAlmostEqual(float(rectangles[0].get('x')), zero)
        self.assertAlmostEqual(float(rectangles[1].get('x')) + float(rectangles[1].get('width')), zero)
        self.assertAlmostEqual(float(rectangles[1].get('width')) / float(rectangles[0].get('width')), .5)
        self.assertAlmostEqual(float(rectangles[2].get('width')) / float(rectangles[0].get('width')), 2.)
        self.assertTrue(all(float(node.get('width')) >= 0 for node in rectangles))
        self.assertLess(max(float(node.get('y')) + float(node.get('height')) for node in rectangles),
                        float(svg.get('viewBox').split()[3]))

    def test_operator_charts_use_selected_core_values_and_readable_traceable_fingerprints(self):
        report = operator_drilldown_report()
        out, (book, _) = self.build(report)
        compact = lambda value: ''.join(str(value).split())
        self.assertEqual(tuple(book), renderer.SHEETS)
        charts = self.chart_parts_by_sheet(out, book)
        for number, method in enumerate(report['cases'][0]['projection']['layer_mean_methods'], 1):
            sheet_name = '方法' + str(number) + '估算'
            cells = book[sheet_name][1]
            self.assertTrue(any(str(value).startswith('CSA/HCA/SWA算子下钻位置：CSA：第') for value in cells.values()))
            for role, attention in (('main', 'CSA'), ('main', 'HCA'), ('draft', 'SWA')):
                data = renderer._method_operator_data(method, role, attention)
                self.assertTrue(data['available'], data['note'])
                target = next(chart for chart in charts[sheet_name] if self.chart_title(chart).startswith(attention + '算子互斥覆盖与模型增减'))
                cache = self.assert_bar_cache_matches_own_sheet(target, book, sheet_name)
                self.assertEqual([values for _, values in cache], [[row[column] for row in data['target_rows']] for column in (1, 2)])
                self.assertEqual(len(cache[0][0]), len(set(cache[0][0])))
                self.assertEqual(cache[0][0], [row[0] for row in data['target_rows']])
                for label, item, record in zip(cache[0][0], data['target_items'], data['target_rows']):
                    if item.get('remainder'):
                        continue
                    self.assertIn(compact(item['stage_label']), compact(label))
                    for member in item['members']:
                        self.assertIn(compact(member['op_type'] or member['name']), compact(label))
                        self.assertIn(compact(member['input_shapes']), compact(label))
                        self.assertIn(compact(member['output_shapes']), compact(record[-1]))
                self.assertNotIn('合计（闭合目标父类）', cache[0][0])
                self.assertAlmostEqual(sum(sum(values) for _, values in cache), data['total_row'][3])
                hotspot = next(chart for chart in charts[sheet_name] if self.chart_title(chart).startswith(attention + '源实测算子热点'))
                raw_cache = self.assert_bar_cache_matches_own_sheet(hotspot, book, sheet_name)
                self.assertEqual(raw_cache[0][1], [row[2] for row in data['kernel_rows']])
                for row, kernel in zip(data['kernel_rows'], data['kernel_items']):
                    self.assertIn(compact(kernel['name']), compact(row[-1]))
                self.assertEqual(len(data['kernel_items']), 8)
                # Same native name retains distinct shape/stream fingerprints.
                self.assertEqual(len({row['name'] for row in data['kernel_items']}), 1)
                self.assertGreater(len({(row['input_shapes'], row['stream_id']) for row in data['kernel_items']}), 1)
                old = next(chart for chart in charts[sheet_name]
                           if self.chart_title(chart).startswith(attention + '层内阶段展开'))
                self.assertEqual(len(self.assert_bar_cache_matches_own_sheet(old, book, sheet_name)[0][1]), 10)
                self.assertIn(data['total_row'][3], cells.values())
            self.assertNotIn('SYNTHETIC_OPERATOR_SCOPE', ' '.join(map(str, cells.values())))
            title = next(cell for cell, value in cells.items() if value == 'CSA算子下钻 · 目标贡献')
            title_cell = book[sheet_name][0].find('.//{*}c[@r="' + title + '"]')
            self.assertEqual(int(title_cell.get('s')), renderer.xlsx_writer.STYLE_IDS['section'])
            self.assertEqual(book[sheet_name][0].find('.//{*}row[@r="' + title[1:] + '"]').get('ht'), '30')
        preview = out.with_suffix('.html').read_text(encoding='utf-8')
        self.assertEqual(preview.count('class="operator-drilldown"'), 6)
        self.assertEqual(preview.count('class="operator-chart"'), 12)
        self.assertIn('关键路径及优化收益尚未验证', preview)
        self.assertIn('SYNTHETIC_OPERATOR_SCOPE', preview)
        self.assertIn('SYNTHETIC_SAME_NATIVE_NAME_' + 'long_component_' * 12, preview)
        self.assertNotRegex(preview, r'width:-')
        svgs = [ET.fromstring(svg) for svg in re.findall(r'<svg\b.*?</svg>', preview, re.DOTALL)]
        rectangles = [node for svg in svgs for node in svg.findall('{*}rect')]
        self.assertTrue(rectangles)
        self.assertTrue(all(float(node.get('width')) >= 0 for node in rectangles))

    def test_long_operator_fields_remain_visible_in_tables_chart_categories_and_html(self):
        from html.parser import HTMLParser

        report = operator_drilldown_report()
        expected = {}
        for method in report['cases'][0]['projection']['layer_mean_methods']:
            for group in method['target_stage_breakdowns']:
                role, attention = group['role'], group['attention']
                before = renderer._method_operator_data(method, role, attention)
                expected[method['id'], attention] = (
                    [(row['key'], row['kind'], row['stage_order']) for row in before['target_items']],
                    [row[1:5] for row in before['target_rows']],
                    [row['key'] for row in before['kernel_items']], [row[1:5] for row in before['kernel_rows']])
                for stage in group['stage_rows']:
                    stage['label'] = ('完整父阶段：HCPre与层内QKV准备及输入映射、结果归并衔接；' * 5
                                      + attention + '_' + str(stage['stage_order']) + '_PARENT_END')
                    breakdown = stage['operator_breakdown']
                    replacements = {}
                    for kernel in breakdown['kernels']:
                        suffix = kernel['key'][-1]
                        fields = dict(
                            name='aclnn_FullNative_' + 'ProjectionAndPreparation_' * 10 + suffix + '_NATIVE_END',
                            op_type='FullOperatorType_' + 'QuantBatchMatmulV3_' * 7 + suffix + '_TYPE_END',
                            input_shapes=';'.join(f'{number},8192,128,16' for number in range(1, 14)),
                            output_shapes=';'.join(f'{number},7168,64,8' for number in range(1, 10)))
                        kernel.update(fields)
                        replacements[kernel['key']] = fields
                    for row in breakdown['rows']:
                        for member in row['members']:
                            member.update(replacements[member['key']])
                        example = row.get('example') or {}
                        for direction in ('before', 'after'):
                            neighbor = example.get(direction)
                            if neighbor and neighbor.get('key') in replacements:
                                neighbor.update(replacements[neighbor['key']])

        out, (book, _) = self.build(report)
        charts = self.chart_parts_by_sheet(out, book)
        compact = lambda value: ''.join(str(value).split())
        visible_fields = set()
        for number, method in enumerate(report['cases'][0]['projection']['layer_mean_methods'], 1):
            sheet_name = '方法' + str(number) + '估算'
            cells = book[sheet_name][1]
            # Only directly displayed provenance cells count; raw JSON and
            # hover titles must not be the only place with complete names.
            provenance = ''.join(str(value) for cell, value in sorted(cells.items(),
                key=lambda item: (int(re.search(r'\d+', item[0]).group()), item[0]))
                if re.fullmatch(r'(?:I|K)\d+', cell) and isinstance(value, str))
            continuations = [int(cell[1:]) for cell, value in cells.items()
                             if re.fullmatch(r'A\d+', cell) and isinstance(value, str)
                             and value.startswith('（续 #')]
            self.assertTrue(continuations)
            for row in continuations:
                self.assertTrue(all(not isinstance(cells.get(column + str(row)), (int, float))
                                    for column in ('E', 'F', 'G', 'H', 'I', 'J')))
            for role, attention in (('main', 'CSA'), ('main', 'HCA'), ('draft', 'SWA')):
                data = renderer._method_operator_data(method, role, attention)
                self.assertTrue(data['available'], data['note'])
                old_items, old_values, old_kernels, old_kernel_values = expected[method['id'], attention]
                self.assertEqual([(row['key'], row['kind'], row['stage_order']) for row in data['target_items']], old_items)
                self.assertEqual([row[1:5] for row in data['target_rows']], old_values)
                self.assertEqual([row['key'] for row in data['kernel_items']], old_kernels)
                self.assertEqual([row[1:5] for row in data['kernel_rows']], old_kernel_values)
                for prefix, records in ((attention + '算子互斥覆盖与模型增减', data['target_rows']),
                                        (attention + '源实测算子热点', data['kernel_rows'])):
                    chart = next(chart for chart in charts[sheet_name] if self.chart_title(chart).startswith(prefix))
                    cache = self.assert_bar_cache_matches_own_sheet(chart, book, sheet_name)
                    self.assertEqual(cache[0][0], [row[0] for row in records])
                    self.assertTrue(any(len(label) > 180 for label in cache[0][0]))
                    self.assertFalse(any('…' in label for label in cache[0][0]))
                for record, item in zip(data['target_rows'], data['target_items']):
                    if item.get('remainder'):
                        continue
                    self.assertIn(compact(item['stage_label']), compact(record[0]))
                    visible_fields.add(item['stage_label'])
                    for member in item['members']:
                        self.assertIn(compact(member['op_type']), compact(record[0]))
                        self.assertIn(compact(member['input_shapes']), compact(record[0]))
                        for field in ('name', 'output_shapes'):
                            self.assertIn(compact(member[field]), compact(provenance))
                        visible_fields.update(member[field] for field in ('name', 'op_type', 'input_shapes', 'output_shapes'))
            for row in book[sheet_name][0].findall('.//{*}sheetData/{*}row'):
                source_text = [cells.get(cell.get('r')) for cell in row.findall('{*}c')
                               if re.fullmatch(r'(?:I|K)\d+', cell.get('r'))]
                source_text = [value for value in source_text if isinstance(value, str)]
                if source_text:
                    self.assertGreaterEqual(float(row.get('ht', 20)),
                        10 * max(value.count('\n') + 1 for value in source_text))
                    self.assertLessEqual(float(row.get('ht', 20)), 409)

        class VisibleText(HTMLParser):
            def __init__(self):
                super().__init__()
                self.hidden, self.parts = [], []
            def handle_starttag(self, tag, attrs):
                if tag in ('title', 'details', 'script', 'style'):
                    self.hidden.append(tag)
            def handle_endtag(self, tag):
                if self.hidden and tag == self.hidden[-1]:
                    self.hidden.pop()
            def handle_data(self, data):
                if not self.hidden:
                    self.parts.append(data)

        preview = out.with_suffix('.html').read_text(encoding='utf-8')
        parser = VisibleText(); parser.feed(preview)
        visible = compact(''.join(parser.parts))
        for field in visible_fields:
            self.assertIn(compact(field), visible)
        for source in re.findall(r'<svg\b.*?</svg>', preview, re.DOTALL):
            svg = ET.fromstring(source)
            for node in svg.findall('{*}text'):
                lines = node.findall('{*}tspan')
                if lines and node.findtext('{*}title', '').startswith(('#', '其余')):
                    self.assertEqual(compact(''.join(line.text or '' for line in lines)),
                                     compact(node.findtext('{*}title')))

    def test_operator_missing_conflicting_or_partial_evidence_suppresses_only_new_class_charts(self):
        for invalid in ('old_json', 'status', 'rounds', 'references', 'source_sum', 'delta', 'total', 'share',
                        'duplicate', 'member_shape', 'member_instances', 'example', 'gap_neighbors', 'kernel_count'):
            report = operator_drilldown_report()
            method = report['cases'][0]['projection']['layer_mean_methods'][0]
            group = next(row for row in method['target_stage_breakdowns'] if row['attention'] == 'CSA')
            breakdown = group['stage_rows'][1]['operator_breakdown']
            item = breakdown['rows'][0]
            if invalid == 'old_json':
                group['stage_rows'][1].pop('operator_breakdown')
            elif invalid == 'status':
                breakdown['status'] = 'PARTIAL'
            elif invalid == 'rounds':
                breakdown['rounds'] += 1
            elif invalid == 'references':
                breakdown['reference_layers'] = [0, 1]
            elif invalid == 'source_sum':
                item.update(source_ms=item['source_ms'] + 1, target_ms=item['target_ms'] + 1,
                            target_total_ms=item['target_total_ms'] + group['target_layers'])
                item['cycle_share'] = item['target_total_ms'] / method['cycle_ms']
            elif invalid in ('delta', 'total', 'share'):
                item[{'delta': 'delta_ms', 'total': 'target_total_ms', 'share': 'cycle_share'}[invalid]] += 1
            elif invalid == 'duplicate':
                breakdown['rows'].append(copy.deepcopy(item))
            elif invalid == 'member_shape':
                item['members'][0]['input_shapes'] = 'SYNTHETIC_WRONG_SHAPE'
            elif invalid == 'member_instances':
                item['members'][0]['instances'] = 2
            elif invalid == 'example':
                item['example']['rank'] = 'SYNTHETIC_OTHER_RANK'
            elif invalid == 'gap_neighbors':
                next(row for row in breakdown['rows'] if row['kind'] == 'gap')['example'].pop('before')
            else:
                breakdown['kernels'][0]['calls_mean'] += 10
            with self.subTest(invalid=invalid):
                data = renderer._method_operator_data(method, 'main', 'CSA')
                self.assertFalse(data['available'])
                self.assertIn('未绘图', data['note'])
                self.assertTrue(renderer._method_operator_data(method, 'main', 'HCA')['available'])
                self.assertIsNotNone(renderer._target_layer_expansion_data(method, 'main', 'CSA', 'CSA')['series'])
                if invalid == 'old_json':
                    self.assertIn('请用新版核心重算', data['note'])
                    out, (book, _) = self.build(report)
                    by_sheet = self.chart_parts_by_sheet(out, book)['方法1估算']
                    self.assertFalse(any(self.chart_title(chart).startswith('CSA算子互斥覆盖') for chart in by_sheet))
                    self.assertTrue(any(self.chart_title(chart).startswith('CSA层内阶段展开') for chart in by_sheet))
                    self.assertIn('请用新版核心重算', out.with_suffix('.html').read_text())

    def test_sal_only_cases_hide_unused_rate_column_and_keep_legacy_declared_rates(self):
        # Synthetic normalized core reports: the second case never declares a rate.
        for with_legacy_rate in (False, True):
            with self.subTest(with_legacy_rate=with_legacy_rate):
                report = layer_mean_fixture()
                first = report['cases'][0]
                first['name'] = 'SYNTHETIC_FIRST_S7'
                first['config'].update(name=first['name'], bs=8, draft_tokens=7,
                                       synthetic_acceptance_length=5.2, effective_tokens_per_round=5.2)
                second = copy.deepcopy(first)
                second['name'] = 'SYNTHETIC_SECOND_S9'
                second['config'].update(name=second['name'], draft_tokens=9,
                                        synthetic_acceptance_length=6.4, effective_tokens_per_round=6.4)
                report['cases'].append(second)
                if with_legacy_rate:
                    first['config'].update(acceptance=.6, acceptance_semantics='mean_accepted_draft_fraction')
                expected_headers = renderer.CASE_PARAMETERS_HEADERS if with_legacy_rate else renderer.CASE_PARAMETERS_HEADERS[:-1]
                expected_rows = [[first['name'], '8', '7', '5.2'], [second['name'], '8', '9', '6.4']]
                if with_legacy_rate:
                    expected_rows[0].append('60%')
                    expected_rows[1].append('未声明')
                headers, rows, _ = renderer._case_parameter_table(report)
                self.assertEqual(headers, expected_headers)
                self.assertEqual(rows, expected_rows)
                first_line = 'BS=8；草稿S=7；Neff=5.2（含bonus）'
                if with_legacy_rate:
                    first_line += '；声明的平均草稿接受比例=60%（配置假设，非实测）'
                second_line = 'BS=8；草稿S=9；Neff=6.4（含bonus）'
                out, (book, _) = self.build(report)
                self.assertEqual(tuple(book), renderer.SHEETS)
                cells = book['估算总览'][1]
                title_row = next(int(cell[1:]) for cell, value in cells.items()
                                 if re.fullmatch(r'A\d+', cell) and value == '采样配置对照')
                header_row = title_row + 2
                columns = ('A', 'E', 'F', 'G', 'I') if with_legacy_rate else ('A', 'E', 'F', 'G')
                self.assertEqual([cells.get(column + str(header_row)) for column in columns], expected_headers)
                for offset, expected in enumerate(expected_rows, 1):
                    self.assertEqual([cells.get(column + str(header_row + offset)) for column in columns], expected)
                if not with_legacy_rate:
                    self.assertNotIn('I' + str(header_row), cells)
                    self.assertIn(f'G{header_row}:L{header_row}',
                                  [node.get('ref') for node in book['估算总览'][0].findall('{*}mergeCells/{*}mergeCell')])
                for sheet_name in ('估算总览', '方法1估算', '方法2估算'):
                    values = list(book[sheet_name][1].values())
                    self.assertIn(first_line, values)
                    self.assertIn(second_line, values)
                    self.assertFalse(any('声明的平均草稿接受比例=未声明' in str(value) for value in values))
                    if sheet_name != '估算总览':
                        method_id = 'method1' if sheet_name == '方法1估算' else 'method2'
                        for case in report['cases']:
                            method = next(row for row in case['projection']['layer_mean_methods'] if row['id'] == method_id)
                            for field in ('cycle_ms', 'tpot_ms', 'per_device_tokens_s'):
                                self.assertIn(method[field], values)
                preview = out.with_suffix('.html').read_text(encoding='utf-8')
                start = preview.index('class="case-parameters"')
                table = preview[start:preview.index('</table>', start)]
                self.assertEqual(table.count('<th>'), 5 if with_legacy_rate else 4)
                self.assertEqual(preview.count('<p>' + first_line + '</p>'), 3)
                self.assertEqual(preview.count('<p>' + second_line + '</p>'), 3)
                self.assertNotIn('声明的平均草稿接受比例=未声明', preview)
                # Spellings are explained once, rather than repeated in every case.
                self.assertEqual(preview.count('synthetic_acceptance_length'), 1)
                self.assertEqual(preview.count('synthetic_acception_length'), 1)

    def test_old_style_twelve_columns_cards_matrices_and_closed_charts(self):
        report = layer_mean_fixture()
        out, (book, files) = self.build(report)
        self.assertEqual(tuple(book), renderer.SHEETS)
        for sheet, _ in book.values():
            self.assertEqual(len(sheet.findall('.//{*}cols/{*}col')), 12)
            self.assertEqual(sheet.findall('.//{*}f'), [])
        overview = book['估算总览'][1]
        for method in report['cases'][0]['projection']['layer_mean_methods']:
            for field in ('cycle_ms', 'per_device_tokens_s', 'tpot_ms'):
                self.assertIn(method[field], overview.values())
        for identity, name in (('method1', '方法1估算'), ('method2', '方法2估算')):
            method = next(item for item in report['cases'][0]['projection']['layer_mean_methods'] if item['id'] == identity)
            cells = book[name][1]
            for field in ('cycle_ms', 'per_device_tokens_s', 'tpot_ms'):
                self.assertIn(method[field], cells.values())
            self.assertIn('源实测主模型', cells.values())
            self.assertIn('源实测草稿', cells.values())
            self.assertIn('目标主模型', cells.values())
            self.assertIn('目标草稿', cells.values())
            self.assertIn('CSA Dispatch', cells.values())
            self.assertIn('HCA 路由计算', cells.values())
            self.assertIn('SWA Combine', cells.values())
            self.assertIn('层间衔接区间', cells.values())
            self.assertIn('轮次剩余区间', cells.values())
            card_row = next(int(cell[1:]) for cell, value in cells.items()
                            if cell.startswith('A') and value == '完整周期 · ms')
            self.assertEqual(cells['E' + str(card_row)], 'TPOT · ms/token')
            self.assertEqual(cells['I' + str(card_row)], '预测单卡吞吐 · token/s/card')
            self.assertEqual(cells['A' + str(card_row + 1)], method['cycle_ms'])
            self.assertEqual(cells['E' + str(card_row + 1)], method['tpot_ms'])
            self.assertEqual(cells['I' + str(card_row + 1)], method['per_device_tokens_s'])
        self.assertTrue(any(name.startswith('xl/charts/chart') for name in files))
        self.assertNotIn('全局吞吐', out.with_suffix('.html').read_text())

    def test_selected_rank_class_means_are_not_combined_with_other_rank(self):
        report = layer_mean_fixture()
        _, (book, _) = self.build(report)
        for name in ('方法1估算', '方法2估算'):
            cells = book[name][1]
            row = next(int(cell[1:]) for cell, value in cells.items() if value == 'HCA 单层')
            self.assertEqual(cells['C' + str(row)], 3.12345)
            self.assertEqual(cells['E' + str(row)], 4.32345)
            self.assertEqual(cells['I' + str(row)], '[1,3,5,7]')
            self.assertEqual(cells['D' + str(row)], renderer.NA)
            self.assertNotIn(9003.12345, cells.values())

    def test_new_reference_sheet_uses_excluded_zero_attention_means_without_router_column(self):
        _, (book, _) = self.build(layer_mean_fixture())
        values = list(book['层与组件实测'][1].values())
        text = ' '.join(map(str, values))
        self.assertNotIn('OLD_HASH_WITH_MAIN0_ONLY', text)
        self.assertNotIn('Router均值(ms)', values)
        self.assertIn('源路由GMM(ms)', values)
        self.assertIn('[1,3,5,7]', values)
        self.assertIn('[2,4,6]', values)
        self.assertIn('CSA', values)
        self.assertIn('HCA', values)
        self.assertIn('SWA', values)

    def test_strict_pending_records_and_global_metric_do_not_leak_into_new_presentation(self):
        report = layer_mean_fixture()
        _, (book, _) = self.build(report)
        for name in ('估算总览', '方法1估算', '方法2估算', '公式与假设', '复核记录'):
            text = ' '.join(map(str, book[name][1].values()))
            for forbidden in ('OLD_STRICT_COMPONENT_CALIBRATION_ONLY', 'OLD_STRICT_ROUTER_ONLY',
                              'COMMUNICATION_SENSITIVITY_ONLY', '+ 0.014', '全局吞吐',
                              'LAYER_MEAN_ESTIMATE', 'layer_mean_methods', 'core_selected_max_rank_mean'):
                self.assertNotIn(forbidden, text)
            self.assertNotIn(998877.123456, book[name][1].values())
        self.assertEqual(report['cases'][0]['projection']['methods'][0]['status'], 'PENDING')

    def test_multiple_bs_cases_keep_their_own_three_card_values(self):
        report = layer_mean_fixture()
        other = copy.deepcopy(report['cases'][0])
        other['name'] = 'SYNTHETIC_OTHER_BS8'
        other['config']['bs'] = 8
        for method in other['projection']['layer_mean_methods']:
            method['per_device_tokens_s'] += 100.
            method['cycle_ms'] += 10.
            method['tpot_ms'] += 2.
            method['breakdown'][-1]['value_ms'] += 10.
        report['cases'].append(other)
        _, (book, _) = self.build(report)
        for case in report['cases']:
            for method in case['projection']['layer_mean_methods']:
                for name in ('估算总览', '方法1估算' if method['id'] == 'method1' else '方法2估算'):
                    for field in ('cycle_ms', 'per_device_tokens_s', 'tpot_ms'):
                        self.assertIn(method[field], book[name][1].values())

    def test_bs_trends_sort_real_config_and_copy_both_methods_without_rank_reaggregation(self):
        report = bs_trend_report()
        out, (book, _) = self.build(report)
        ns = {'c': 'http://schemas.openxmlformats.org/drawingml/2006/chart'}
        charts = self.chart_parts_by_sheet(out, book)
        ordered = sorted(report['cases'], key=lambda case: case['config']['bs'])
        expected_bs = [4, 8, 16, 32]
        overview = [chart for chart in charts['估算总览'] if chart.find('.//c:lineChart', ns) is not None]
        self.assertEqual(len(overview), 3)
        first_values = {tuple(case['projection']['layer_mean_methods'][0][field] for case in ordered): field
                        for field in ('cycle_ms', 'tpot_ms', 'per_device_tokens_s')}
        seen = set()
        for chart in overview:
            series = self.assert_line_cache_matches_sheet(chart, book, expected_bs)
            self.assertEqual(len(series), 2)
            vector = tuple(float(row.findtext('c:v', namespaces=ns))
                           for row in series[0].findall('c:val/c:numRef/c:numCache/c:pt', ns))
            self.assertIn(vector, first_values)
            field = first_values[vector]
            seen.add(field)
            for index, item in enumerate(series):
                values = [float(row.findtext('c:v', namespaces=ns))
                          for row in item.findall('c:val/c:numRef/c:numCache/c:pt', ns)]
                self.assertEqual(values, [case['projection']['layer_mean_methods'][index][field] for case in ordered])
                self.assertIn(renderer._layer_mean_name({'id': 'method' + str(index + 1)}),
                              ''.join(item.find('c:tx', ns).itertext()))
            first_row = int(re.search(r'\$A\$(\d+):',
                series[0].findtext('c:cat/c:strRef/c:f', namespaces=ns)).group(1))
            for row, case in enumerate(ordered, first_row):
                self.assertEqual(book['估算总览'][1]['D' + str(row)],
                                 case['projection']['layer_mean_methods'][0]['selected_rank'])
                self.assertEqual(book['估算总览'][1]['H' + str(row)],
                                 case['projection']['layer_mean_methods'][1]['selected_rank'])
        self.assertEqual(seen, {'cycle_ms', 'tpot_ms', 'per_device_tokens_s'})
        for index, name in enumerate(('方法1估算', '方法2估算')):
            line = [chart for chart in charts[name] if chart.find('.//c:lineChart', ns) is not None]
            self.assertEqual(len(line), 1)
            # Existing per-case closed bars remain, alongside one BS comparison.
            self.assertEqual(len(charts[name]), 5)
            series = self.assert_line_cache_matches_sheet(line[0], book, expected_bs)
            self.assertEqual(len(series), 5)
            methods = [case['projection']['layer_mean_methods'][index] for case in ordered]
            self.assertGreater(len({method['selected_rank'] for method in methods}), 1)
            keys = ('main_layer_envelopes', 'main_internal_gaps', 'draft_layer_envelopes',
                    'draft_internal_gaps', 'residual_once')
            labels = ('主模型层区间', '主模型层间衔接', '草稿层区间', '草稿层间衔接', '轮次剩余区间U')
            for component, label, item in zip(keys, labels, series):
                values = [float(row.findtext('c:v', namespaces=ns))
                          for row in item.findall('c:val/c:numRef/c:numCache/c:pt', ns)]
                rows = [next(row for row in method['breakdown'] if row['key'] == component) for method in methods]
                self.assertEqual(values, [row['value_ms'] for row in rows])
                self.assertIn(label, ''.join(item.find('c:tx', ns).itertext()))
            first_row = int(re.search(r'\$A\$(\d+):',
                series[0].findtext('c:cat/c:strRef/c:f', namespaces=ns)).group(1))
            for row, method in enumerate(methods, first_row):
                self.assertEqual(book[name][1]['D' + str(row)], method['selected_rank'])

    def test_unavailable_middle_bs_is_a_blank_point_with_original_category_index(self):
        report = bs_trend_report()
        missing = next(case for case in report['cases'] if case['config']['bs'] == 8)
        missing['projection']['layer_mean_methods'][1].update(
            status='UNAVAILABLE', selected_rank=None, reasons=['SYNTHETIC_BS8_REFERENCE_MISSING'])
        out, (book, _) = self.build(report)
        ns = {'c': 'http://schemas.openxmlformats.org/drawingml/2006/chart'}
        charts = self.chart_parts_by_sheet(out, book)
        for chart in charts['估算总览']:
            if chart.find('.//c:lineChart', ns) is None:
                continue
            series = self.assert_line_cache_matches_sheet(chart, book, [4, 8, 16, 32])
            for index, item in enumerate(series):
                points = item.findall('c:val/c:numRef/c:numCache/c:pt', ns)
                self.assertEqual([int(row.get('idx')) for row in points], [0, 1, 2, 3] if index == 0 else [0, 2, 3])
                values_ref = item.findtext('c:val/c:numRef/c:f', namespaces=ns)
                if index == 1:
                    self.assertIsNone(self.chart_reference_cells(book, values_ref)[1])
        line = [chart for chart in charts['方法2估算'] if chart.find('.//c:lineChart', ns) is not None]
        self.assertEqual(len(line), 1)
        for series in self.assert_line_cache_matches_sheet(line[0], book, [4, 8, 16, 32]):
            self.assertEqual([int(row.get('idx')) for row in series.findall('c:val/c:numRef/c:numCache/c:pt', ns)], [0, 2, 3])
            self.assertIsNone(self.chart_reference_cells(book, series.findtext('c:val/c:numRef/c:f', namespaces=ns))[1])

    def test_nonclosing_breakdown_blanks_all_five_parts_but_keeps_valid_headline(self):
        report = bs_trend_report()
        case = next(case for case in report['cases'] if case['config']['bs'] == 8)
        method = case['projection']['layer_mean_methods'][0]
        method['breakdown'][0]['value_ms'] += .75
        out, (book, _) = self.build(report)
        ns = {'c': 'http://schemas.openxmlformats.org/drawingml/2006/chart'}
        charts = self.chart_parts_by_sheet(out, book)
        line = [chart for chart in charts['方法1估算'] if chart.find('.//c:lineChart', ns) is not None]
        self.assertEqual(len(line), 1)
        for series in self.assert_line_cache_matches_sheet(line[0], book, [4, 8, 16, 32]):
            self.assertEqual([int(row.get('idx')) for row in series.findall('c:val/c:numRef/c:numCache/c:pt', ns)],
                             [0, 2, 3])
            self.assertIsNone(self.chart_reference_cells(book, series.findtext('c:val/c:numRef/c:f', namespaces=ns))[1])
        method_cells = book['方法1估算'][1]
        row = next(int(cell[1:]) for cell, value in method_cells.items()
                   if re.fullmatch(r'B\d+', cell) and value == case['name'])
        self.assertTrue(all(method_cells.get(column + str(row)) is None for column in 'EFGHI'))
        self.assertEqual(method_cells['J' + str(row)], method['cycle_ms'])
        self.assertIn('不闭合', method_cells['K' + str(row)])
        for chart in charts['估算总览']:
            if chart.find('.//c:lineChart', ns) is not None:
                series = self.assert_line_cache_matches_sheet(chart, book, [4, 8, 16, 32])[0]
                self.assertEqual([int(point.get('idx')) for point in
                                  series.findall('c:val/c:numRef/c:numCache/c:pt', ns)], [0, 1, 2, 3])
        overview_cells = book['估算总览'][1]
        row = next(int(cell[1:]) for cell, value in overview_cells.items()
                   if re.fullmatch(r'B\d+', cell) and value == case['name'])
        self.assertEqual([overview_cells[column + str(row)] for column in 'EFG'],
                         [method[field] for field in ('per_device_tokens_s', 'cycle_ms', 'tpot_ms')])

    def test_old_writer_missing_line_chart_interface_fails_clearly_before_saving(self):
        out = self.root / 'old-writer.xlsx'
        with mock.patch.object(renderer.xlsx_writer.Worksheet, 'add_line_chart', None), \
                mock.patch.object(renderer.Workbook, 'save') as save, \
                self.assertRaisesRegex(ValueError, 'Excel写入器缺少BS趋势图接口') as raised:
            renderer.build_workbook(out, report=self.root / 'source.json', report_data=bs_trend_report(),
                                    expected_method_set='layer_mean_methods')
        self.assertIn('ep128_xlsx_writer.py', str(raised.exception))
        self.assertIn('请同步更新', str(raised.exception))
        save.assert_not_called()
        self.assertFalse(out.exists())
        self.assertFalse(out.with_suffix('.html').exists())

    def test_offline_html_has_all_bs_trends_and_splits_missing_points(self):
        report = bs_trend_report()
        missing = next(case for case in report['cases'] if case['config']['bs'] == 8)
        missing['projection']['layer_mean_methods'][1].update(
            status='UNAVAILABLE', selected_rank=None, reasons=['SYNTHETIC_BS8_REFERENCE_MISSING'])
        out, _ = self.build(report)
        preview = out.with_suffix('.html').read_text(encoding='utf-8')
        svgs = [ET.fromstring(svg) for svg in re.findall(r'<svg\b.*?</svg>', preview, re.DOTALL)]
        svgs = [svg for svg in svgs if not (svg.findtext('title') or '').startswith('源采样基线 · ')]
        self.assertEqual(len(svgs), 5)
        self.assertEqual([svg.findtext('title') for svg in svgs[:3]],
                         ['单卡吞吐随BS变化', '完整周期（ITL代理）随BS变化', 'TPOT随BS变化'])
        for index, svg in enumerate(svgs[3:], 1):
            self.assertIn(renderer._layer_mean_name({'id': 'method' + str(index)}), svg.findtext('title'))
            self.assertIn('整轮五项耗时构成随BS变化', svg.findtext('title'))
        def xs(line):
            return [float(point.split(',')[0]) for point in line.get('points').split()]
        for svg in svgs[:3]:
            first = [line for line in svg.findall('polyline') if line.get('stroke') == '#168B8D']
            second = [line for line in svg.findall('polyline') if line.get('stroke') == '#D4793A']
            self.assertEqual([xs(line) for line in first], [[100., 420., 740., 1060.]])
            self.assertEqual([xs(line) for line in second], [[100.], [740., 1060.]])
        self.assertEqual([xs(line) for line in svgs[3].findall('polyline')],
                         [[100., 420., 740., 1060.]] * 5)
        self.assertEqual([xs(line) for line in svgs[4].findall('polyline')],
                         [[100.], [740., 1060.]] * 5)

    def test_non_bs_config_models_and_observed_rank_changes_never_share_a_trend(self):
        for changed in ('draft_tokens', 'effective_tokens_per_round', 'target_ep',
                        'source_model', 'target_model', 'observed_ranks', 'rank_coverage'):
            report = bs_trend_report()
            other = [copy.deepcopy(case) for case in report['cases'] if case['config']['bs'] in (4, 8)]
            for case in other:
                case['name'] += '_OTHER_SCENARIO'
                if changed == 'draft_tokens':
                    case['config']['draft_tokens'] += 1
                elif changed == 'effective_tokens_per_round':
                    case['config']['effective_tokens_per_round'] += 1.
                elif changed == 'target_ep':
                    case['config']['target_ep'] = 256
                elif changed == 'source_model':
                    case['source_model']['expert_dtype'] = 'SYNTHETIC_OTHER_DTYPE'
                elif changed == 'target_model':
                    case['target_model']['main']['num_hidden_layers'] += 1
                elif changed == 'observed_ranks':
                    case['captures'].append(dict(rank='dp_extra', sha256='SYNTHETIC_EXTRA'))
                    case['measurements'].append(dict(rank='dp_extra', rounds=[]))
                else:
                    for method in case['projection']['layer_mean_methods']:
                        method.setdefault('coverage', {})['complete'] = False
            report['cases'].extend(other)
            before = copy.deepcopy(report)
            with self.subTest(changed=changed):
                groups = renderer._bs_comparison_groups(report)
                self.assertEqual(report, before)
                self.assertEqual(len(groups), 2)
                self.assertCountEqual([[case['config']['bs'] for case in group['cases']]
                                       for group in groups], [[4, 8], [4, 8, 16, 32]])
                out, (book, _) = self.build(report)
                ns = {'c': 'http://schemas.openxmlformats.org/drawingml/2006/chart'}
                charts = self.chart_parts_by_sheet(out, book)
                for name, expected in (('估算总览', 6), ('方法1估算', 2), ('方法2估算', 2)):
                    line = [chart for chart in charts[name] if chart.find('.//c:lineChart', ns) is not None
                            and '随BS变化' in self.chart_title(chart)]
                    self.assertEqual(len(line), expected)
                    for chart in line:
                        categories = [node.text for node in chart.findall('.//c:ser/c:cat/c:strRef/c:strCache/c:pt/c:v', ns)]
                        # Each chart contains one intact group, not duplicate
                        # categories from unrelated source/target conditions.
                        self.assertIn(categories[:len(set(categories))], [['4', '8'], ['4', '8', '16', '32']])

    def test_duplicate_bs_keeps_each_case_row_and_suppresses_group_trends(self):
        report = bs_trend_report()
        duplicate = copy.deepcopy(next(case for case in report['cases'] if case['config']['bs'] == 8))
        duplicate['name'] = 'SYNTHETIC_REPEATED_BS8'
        for method in duplicate['projection']['layer_mean_methods']:
            method['cycle_ms'] += 100.
            method['breakdown'][-1]['value_ms'] += 100.
        report['cases'].append(duplicate)
        before = copy.deepcopy(report)
        groups = renderer._bs_comparison_groups(report)
        self.assertEqual(report, before)
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]['duplicate_bs'], [8])
        self.assertEqual([case['config']['bs'] for case in groups[0]['cases']], [4, 8, 8, 16, 32])
        out, (book, _) = self.build(report)
        ns = {'c': 'http://schemas.openxmlformats.org/drawingml/2006/chart'}
        charts = self.chart_parts_by_sheet(out, book)
        for name in ('估算总览', '方法1估算', '方法2估算'):
            self.assertFalse(any(chart.find('.//c:lineChart', ns) is not None for chart in charts[name]))
            self.assertIn('相同BS存在多条采样组合，未合并趋势', ' '.join(map(str, book[name][1].values())))
        cells = book['估算总览'][1]
        s_section = next(int(cell[1:]) for cell, value in cells.items()
                         if re.fullmatch(r'A\d+', cell) and isinstance(value, str)
                         and value.startswith('固定BS，不同投机步长S'))
        rows = [(int(cell[1:]), value) for cell, value in cells.items()
                if re.fullmatch(r'B\d+', cell) and int(cell[1:]) < s_section
                and value in {case['name'] for case in report['cases']}]
        self.assertEqual(len(rows), 5)
        cases = {case['name']: case for case in report['cases']}
        for row, name in rows:
            self.assertEqual(book['估算总览'][1]['F' + str(row)],
                             cases[name]['projection']['layer_mean_methods'][0]['cycle_ms'])

    def test_missing_bs_is_retained_without_guessing_case_name_or_drawing_trends(self):
        report = bs_trend_report()
        missing = report['cases'][0]
        missing['config'].pop('bs')
        missing['name'] = 'SYNTHETIC_NAME_ONLY_BS512'
        out, (book, _) = self.build(report)
        ns = {'c': 'http://schemas.openxmlformats.org/drawingml/2006/chart'}
        charts = self.chart_parts_by_sheet(out, book)
        for name in ('估算总览', '方法1估算', '方法2估算'):
            self.assertFalse(any(chart.find('.//c:lineChart', ns) is not None for chart in charts[name]))
            row = next(int(cell[1:]) for cell, value in book[name][1].items()
                       if re.fullmatch(r'B\d+', cell) and value == missing['name'])
            self.assertIsNone(book[name][1].get('A' + str(row)))

    def test_bs_index_keeps_singleton_with_different_target_dp_visible_before_grouped_tables(self):
        report = bs_trend_report()
        report['cases'] = [case for case in report['cases'] if case['config']['bs'] != 32]
        for case in report['cases']:
            case['config']['target_dp'] = 128 if case['config']['bs'] == 16 else 64
        groups = renderer._bs_comparison_groups(report)
        self.assertCountEqual([[case['config']['bs'] for case in group['cases']] for group in groups],
                              [[4, 8], [16]])
        out, (book, _) = self.build(report)
        ordered = sorted(report['cases'], key=lambda case: case['config']['bs'])
        cells = book['估算总览'][1]
        title = '本次报告全部采样组合（含单独分组的BS）'
        title_row = next(int(cell[1:]) for cell, value in cells.items()
                         if re.fullmatch(r'A\d+', cell) and value == title)
        table_row = next(int(cell[1:]) for cell, value in cells.items()
                         if re.fullmatch(r'B\d+', cell) and value == 'BS对比组')
        self.assertLess(title_row, table_row)
        index_rows = sorted((int(cell[1:]), value) for cell, value in cells.items()
                            if re.fullmatch(r'D\d+', cell) and value in {case['name'] for case in ordered})
        self.assertEqual([value for _, value in index_rows], [case['name'] for case in ordered])
        first_grouped_row = min(int(cell[1:]) for cell, value in cells.items()
                                if re.fullmatch(r'B\d+', cell) and value in {case['name'] for case in ordered})
        for (row, name), case in zip(index_rows, ordered):
            self.assertGreater(row, table_row)
            self.assertLess(row, first_grouped_row)
            self.assertEqual(cells['A' + str(row)], case['config']['bs'])
            self.assertIn('BS对比组', cells['B' + str(row)])
            self.assertIn('方法1', cells['G' + str(row)])
            self.assertIn('方法2', cells['G' + str(row)])
        self.assertIn('一个BS', cells['G' + str(index_rows[-1][0])])
        single = ordered[-1]
        for name in ('估算总览', '方法1估算', '方法2估算'):
            original_rows = [int(cell[1:]) for cell, value in book[name][1].items()
                             if re.fullmatch(r'B\d+', cell) and value == single['name']]
            self.assertEqual(len(original_rows), 1)
            self.assertEqual(book[name][1]['A' + str(original_rows[0])], 16.)
        ns = {'c': 'http://schemas.openxmlformats.org/drawingml/2006/chart'}
        charts = self.chart_parts_by_sheet(out, book)
        for name in ('估算总览', '方法1估算', '方法2估算'):
            line = [chart for chart in charts[name] if chart.find('.//c:lineChart', ns) is not None]
            self.assertEqual(len(line), 3 if name == '估算总览' else 1)
            for chart in line:
                self.assert_line_cache_matches_sheet(chart, book, [4, 8])
        preview = out.with_suffix('.html').read_text(encoding='utf-8')
        start = preview.index(title)
        index_html = preview[start:preview.index('</table>', start)]
        self.assertEqual(sorted(ordered, key=lambda case: index_html.index(case['name'])), ordered)

    def test_progress_reports_only_actual_disk_bs_and_silent_export_has_same_cells(self):
        report = bs_trend_report()
        report['cases'] = [case for case in report['cases'] if case['config']['bs'] in (4, 8)]
        source = self.root / 'actual-bs4-bs8.json'
        encoded = json.dumps(report, ensure_ascii=False)
        source.write_text(encoded, encoding='utf-8')
        silent, verbose = self.root / 'silent.xlsx', self.root / 'verbose.xlsx'
        with mock.patch('builtins.print') as printer:
            renderer.build_workbook(silent, report=source)
        printer.assert_not_called()
        messages = []
        with mock.patch.object(renderer, '_read_json', wraps=renderer._read_json) as reader:
            renderer.build_workbook(verbose, report=source, progress=messages.append)
        reader.assert_called_once_with(source)
        origins = [message for message in messages if message.startswith('输入报告：')]
        self.assertEqual(len(origins), 1)
        self.assertIn('磁盘JSON', origins[0])
        self.assertIn(str(source.resolve()), origins[0])
        self.assertIn('输入清单：case=2；配置BS=[4, 8]；失败case=0', messages)
        combinations = [message for message in messages if message.startswith('输入组合：')]
        self.assertEqual(len(combinations), 2)
        for case in report['cases']:
            line = next(message for message in combinations if case['name'] in message)
            self.assertIn('config.bs=' + str(case['config']['bs']), line)
            self.assertIn('方法1=条件估算', line)
            self.assertIn('方法2=条件估算', line)
        self.assertTrue(any(message.startswith('BS分组：') for message in messages))
        self.assertFalse(any('config.bs=16' in message for message in messages))
        self.assertEqual(source.read_text(encoding='utf-8'), encoded)
        self.assertEqual({name: cells for name, (_, cells) in read_book(silent)[0].items()},
                         {name: cells for name, (_, cells) in read_book(verbose)[0].items()})

    def test_in_memory_input_index_and_diagnostics_ignore_stale_disk_case(self):
        report = bs_trend_report()
        stale = copy.deepcopy(report)
        stale['cases'] = [case for case in stale['cases'] if case['config']['bs'] == 16]
        stale['cases'][0]['name'] = 'STALE_DISK_ONLY_BS16'
        report['cases'] = [case for case in report['cases'] if case['config']['bs'] in (4, 8)]
        before = copy.deepcopy(report)
        source = self.root / 'stale-source-label.json'
        encoded = json.dumps(stale, ensure_ascii=False)
        source.write_text(encoded, encoding='utf-8')
        messages = []
        out = self.root / 'memory.xlsx'
        with mock.patch.object(renderer, '_read_json', side_effect=AssertionError('must not reread source label')) as reader:
            renderer.build_workbook(out, report=source, report_data=report, progress=messages.append)
        reader.assert_not_called()
        self.assertEqual(report, before)
        self.assertEqual(source.read_text(encoding='utf-8'), encoded)
        origin = next(message for message in messages if message.startswith('输入报告：'))
        self.assertIn('复用内存报告（来源路径仅为标签）', origin)
        self.assertIn(str(source.resolve()), origin)
        self.assertIn('输入清单：case=2；配置BS=[4, 8]；失败case=0', messages)
        self.assertFalse(any('STALE_DISK_ONLY_BS16' in message or 'config.bs=16' in message for message in messages))
        book = read_book(out)[0]
        for name in ('估算总览', '方法1估算', '方法2估算'):
            self.assertNotIn('STALE_DISK_ONLY_BS16', book[name][1].values())

    def test_unavailable_method_keeps_source_means_but_does_not_publish_target_values(self):
        report = layer_mean_fixture()
        method = report['cases'][0]['projection']['layer_mean_methods'][1]
        method.update(status='UNAVAILABLE', reasons=['ACTUAL_REFERENCE_BS_ROW_MISSING'])
        method.update(cycle_ms=111222.333, per_device_tokens_s=222333.444, tpot_ms=333444.555)
        _, (book, _) = self.build(report)
        for name in ('估算总览', '方法2估算'):
            for value in (111222.333, 222333.444, 333444.555):
                self.assertNotIn(value, book[name][1].values())
        self.assertIn(3.12345, book['方法2估算'][1].values())
        self.assertIn('ACTUAL_REFERENCE_BS_ROW_MISSING', ' '.join(map(str, book['复核记录'][1].values())))

    def test_canonical_cell_audit_refuses_incorrect_layer_mean_cards(self):
        report = layer_mean_fixture()
        original = renderer._method
        def wrong(case, identity):
            method = copy.deepcopy(original(case, identity))
            method['per_device_tokens_s'] = 123456789.987
            return method
        with mock.patch.object(renderer, '_method', side_effect=wrong), \
                mock.patch.object(renderer.Workbook, 'save') as save, \
                self.assertRaisesRegex(ValueError, '阶段=核对Excel关键结果'):
            renderer.build_workbook(self.root / 'wrong.xlsx', report=self.root / 'source.json',
                                    report_data=report, expected_method_set='layer_mean_methods')
        save.assert_not_called()

    def test_layer_mean_status_and_selected_rank_are_explicit(self):
        for invalid in ('wrong_set', 'missing_rank', 'ambiguous_rank', 'missing_rate'):
            report = layer_mean_fixture()
            method = report['cases'][0]['projection']['layer_mean_methods'][0]
            if invalid == 'wrong_set':
                report['cases'][0]['projection']['display_method_set'] = 'methods'
            elif invalid == 'missing_rank':
                method['selected_rank'] = 'not sampled'
            elif invalid == 'ambiguous_rank':
                method['per_rank'].append(copy.deepcopy(method['per_rank'][0]))
            else:
                method['per_device_tokens_s'] = None
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                renderer._validate_report(report)

    def test_actual_math_helper_contract_renders_without_strict_component_evidence(self):
        from test_ep128_layer_mean import attach, synthetic_report
        report, native = synthetic_report({'dp0': [1., 3.], 'dp1': [2.2, 2.2]})
        methods = attach(report, native)
        self.assertEqual(methods[0]['status'], 'LAYER_MEAN_ESTIMATE')
        self.assertEqual(methods[0]['selected_rank'], 'dp1')
        _, (book, _) = self.build(report)
        self.assertIn(methods[0]['cycle_ms'], book['估算总览'][1].values())
        self.assertIn(methods[0]['per_device_tokens_s'], book['方法1估算'][1].values())
        self.assertIn('缺少实际输入', book['方法2估算'][1].values())
        self.assertNotIn('missing ownership, basis and component calibration',
                         ' '.join(map(str, book['复核记录'][1].values())))

    def test_fixed_communication_cost_is_shown_only_when_core_report_includes_it(self):
        report = layer_mean_fixture()
        _, (book, _) = self.build(report)
        values = list(book['公式与假设'][1].values())
        self.assertIn(.014, values)
        self.assertIn('方法一每次Dispatch/Combine附加14μs；已计入核心增量', values)
        self.assertIn('方法二按历史通信倍率换算，无额外14μs', values)
        for method in report['cases'][0]['projection']['layer_mean_methods']:
            method.pop('communication')
        _, (book, _) = self.build(report)
        self.assertNotIn('14μs', ' '.join(map(str, book['公式与假设'][1].values())))

    def test_unavailable_target_does_not_present_source_rank_as_a_selected_target(self):
        report = layer_mean_fixture()
        method = report['cases'][0]['projection']['layer_mean_methods'][1]
        method.update(status='UNAVAILABLE', reasons=['ACTUAL_INPUT_MISSING'], selected_rank=None, source_rank='dp7')
        out, (book, _) = self.build(report)
        text = ' '.join(map(str, book['方法2估算'][1].values()))
        self.assertIn('目标rank未选择；源参考rank=dp7', text)
        self.assertNotIn('所选rank=dp7', text)
        self.assertNotIn('所选rank=dp3', text)
        self.assertIn('目标rank未选择；源参考rank=dp7', out.with_suffix('.html').read_text())

    def test_ledger_uses_core_rank_partitions_without_expanding_task_intervals(self):
        report = layer_mean_fixture()
        case = report['cases'][0]
        # Old per-task evidence is irrelevant to the selected layer-mean report.
        case['measurements'][0]['rounds'][0]['ledger']['intervals'][0]['label'] = 'OLD_ATOMIC_TASK_ONLY'
        _, (book, _) = self.build(report)
        values = list(book['时间账本'][1].values())
        self.assertIn('源主模型层链(ms)', values)
        self.assertIn('源草稿层链(ms)', values)
        self.assertIn(12.25, values)
        self.assertIn(3.25, values)
        self.assertIn('SYNTHETIC_SOURCE_PARTITION', values)
        self.assertNotIn('OLD_ATOMIC_TASK_ONLY', values)

    def test_method_pages_do_not_dump_component_record_metadata(self):
        report = layer_mean_fixture()
        for method in report['cases'][0]['projection']['layer_mean_methods']:
            method['detail_rows'][0].update(status='LAYER_MEAN_ESTIMATE',
                                            basis='INTERNAL_RECORD_METADATA_ONLY')
        _, (book, _) = self.build(report)
        for name in ('方法1估算', '方法2估算'):
            text = ' '.join(map(str, book[name][1].values()))
            self.assertNotIn('LAYER_MEAN_ESTIMATE', text)
            self.assertNotIn('INTERNAL_RECORD_METADATA_ONLY', text)


    def test_selected_rank_u_tables_and_horizontal_charts_close_without_markov_double_count(self):
        report = residual_report()
        out, (book, files) = self.build(report)
        methods = report['cases'][0]['projection']['layer_mean_methods']
        chart_ns = {'c': 'http://schemas.openxmlformats.org/drawingml/2006/chart'}
        with ZipFile(out) as archive:
            charts = [ET.fromstring(archive.read(name)) for name in files if name.startswith('xl/charts/chart')]
        u_charts = [chart for chart in charts if '轮次剩余区间耗时构成' in ''.join(chart.find('c:chart/c:title', chart_ns).itertext())]
        self.assertEqual(len(u_charts), 2)
        self.assertEqual(renderer.METHOD_RESIDUAL_HEADERS,
                         ['过渡区间', '阶段', '耗时均值(ms)', '匹配轮次 / 状态'])
        for index, method in enumerate(methods):
            rank = method['selected_rank']
            other_rank = 'dp3' if rank == 'dp7' else 'dp7'
            rows, items, note = renderer._method_residual_data(report['cases'][0], method)
            self.assertEqual(len(rows), 15)
            self.assertEqual(len(items), 15)
            self.assertAlmostEqual(sum(item['value_ms'] for item in items), method['source_partition']['residual_ms'])
            self.assertTrue(all(item['label'].startswith(rank + '_PARENT_') for item in items))
            name = '方法1估算' if index == 0 else '方法2估算'
            cells = book[name][1]
            self.assertFalse(any(other_rank + '_PARENT_' in str(value) for value in cells.values()))
            self.assertFalse(any('_CHILD_NOT_ADDED_' in str(value) for value in cells.values()))
            table_rows = sorted(int(cell[1:]) for cell, value in cells.items()
                                if cell.startswith('C') and str(value).startswith(rank + '_PARENT_'))
            self.assertEqual(len(table_rows), 15)
            chart = next(chart for chart in u_charts if renderer._layer_mean_name(method)
                         in ''.join(chart.find('c:chart/c:title', chart_ns).itertext()))
            self.assertEqual(chart.find('.//c:barDir', chart_ns).get('val'), 'bar')
            series = chart.find('.//c:ser', chart_ns)
            self.assertIn('阶段耗时(ms)', ''.join(series.find('c:tx', chart_ns).itertext()))
            values = [float(node.text) for node in series.findall('c:val/c:numRef/c:numCache/c:pt/c:v', chart_ns)]
            labels = [node.text for node in series.findall('c:cat/c:strRef/c:strCache/c:pt/c:v', chart_ns)]
            self.assertEqual(values, [cells['I' + str(row)] for row in table_rows])
            self.assertEqual(labels, [cells['C' + str(row)] for row in table_rows])
            self.assertAlmostEqual(sum(values), method['source_partition']['residual_ms'])
            self.assertEqual(series.find('c:cat/c:strRef/c:f', chart_ns).text,
                             f"'{name}'!$C${table_rows[0]}:$C${table_rows[-1]}")
            self.assertEqual(series.find('c:val/c:numRef/c:f', chart_ns).text,
                             f"'{name}'!$I${table_rows[0]}:$I${table_rows[-1]}")
            method_html = renderer._method_residual_html(report['cases'][0], method)
            self.assertEqual(method_html.count('class="u-bar"'), 15)
            self.assertNotIn(other_rank + '_PARENT_', method_html)
            self.assertNotIn('_CHILD_NOT_ADDED_', method_html)
        self.assertEqual(out.with_suffix('.html').read_text().count('class="u-bar"'), 30)
        titles = [''.join(chart.find('c:chart/c:title', chart_ns).itertext()) for chart in charts]
        self.assertTrue(titles[0].startswith('完整周期分解'))
        self.assertTrue(titles[1].startswith('轮次剩余区间耗时构成'))
        self.assertTrue(titles[2].startswith('完整周期分解'))
        self.assertTrue(titles[3].startswith('轮次剩余区间耗时构成'))

    def test_missing_unselected_or_duplicate_rank_never_falls_back_to_other_capture(self):
        for invalid in ('unselected', 'missing', 'duplicate'):
            report = residual_report()
            case = report['cases'][0]
            method = case['projection']['layer_mean_methods'][0]
            if invalid == 'unselected':
                method.update(selected_rank=None, source_rank='dp3', status='UNAVAILABLE', reasons=['MISSING_TARGET'])
            elif invalid == 'missing':
                case['measurements'] = [row for row in case['measurements'] if row['rank'] != method['selected_rank']]
            else:
                case['measurements'].append(copy.deepcopy(case['measurements'][0]))
            with self.subTest(invalid=invalid):
                rows, items, note = renderer._method_residual_data(case, method)
                self.assertEqual(rows, [])
                self.assertIsNone(items)
                self.assertNotIn('dp3_PARENT_', renderer._method_residual_html(case, method))
                self.assertTrue(note)
        _, (book, _) = self.build(report)
        self.assertFalse(any('_PARENT_' in str(value) for value in book['方法1估算'][1].values()))

    def test_partial_missing_disordered_or_nonclosing_u_never_draws_numeric_bars(self):
        for invalid in ('partial', 'missing', 'order', 'sample_count', 'window_sum', 'diag_sum', 'method_sum', 'unavailable'):
            report = residual_report()
            case = report['cases'][0]
            method = case['projection']['layer_mean_methods'][0]
            diagnostic = case['measurements'][0]['step_residuals']
            if invalid == 'partial':
                diagnostic['stage_summary'][0].update(status='PARTIAL', matched_windows=1)
            elif invalid == 'missing':
                diagnostic['stage_summary'].pop(0)
            elif invalid == 'order':
                diagnostic['stage_summary'][0]['stage_order'] = 2
            elif invalid == 'sample_count':
                diagnostic['stage_summary'][0].update(rounds=1, matched_windows=1)
            elif invalid == 'window_sum':
                diagnostic['summary'][0]['duration_ms_mean'] += 1
            elif invalid == 'diag_sum':
                diagnostic['residual_ms_mean'] += 1
            elif invalid == 'method_sum':
                method['source_partition']['cross_role_tail_residual_ms'] += 1
            else:
                method.update(status='UNAVAILABLE', reasons=['SYNTHETIC_TARGET_INPUT_MISSING'])
            with self.subTest(invalid=invalid):
                rows, items, note = renderer._method_residual_data(case, method)
                self.assertTrue(rows)
                self.assertIsNone(items)
                self.assertIn('未绘图', note)
                self.assertNotIn('class="u-bar"', renderer._method_residual_html(case, method))
                if invalid == 'partial':
                    self.assertEqual(rows[0][1], '阶段未识别（原窗口实测总耗时）')
                    self.assertEqual(rows[0][2], diagnostic['summary'][0]['duration_ms_mean'])
                    self.assertIn('1/2', rows[0][3])
                    self.assertNotIn('请重跑基础统计', note)

    def test_wholly_unresolved_windows_keep_measured_totals_as_two_window_bars(self):
        report = residual_report()
        case = report['cases'][0]
        method = case['projection']['layer_mean_methods'][0]
        diagnostic = case['measurements'][0]['step_residuals']
        original = copy.deepcopy(diagnostic)
        for summary in diagnostic['summary']:
            summary.update(recorded_busy_ms_mean=summary['duration_ms_mean'] * .7,
                           no_recorded_ms_mean=summary['duration_ms_mean'] * .3,
                           host_span_union_ms_mean=summary['duration_ms_mean'] * .4)
            unknown = dict(window_key=summary['window_key'], stage_key='unresolved', stage_order=1,
                label='ORIGINAL_UNRESOLVED', level=0, parent_stage_key=None, rounds=2, matched_windows=0,
                status='UNRESOLVED', duration_ms_mean=None,
                duration_ns_total=round(summary['duration_ms_mean'] * 2e6), reason='SYNTHETIC_MISSING_STAGE_ANCHOR')
            diagnostic['stage_summary'] = [row for row in diagnostic['stage_summary']
                                            if row['window_key'] != summary['window_key']] + [unknown]
            ledger_rows = renderer._residual_stage_rows([unknown], summary)
            self.assertEqual(ledger_rows[0][0], '阶段未识别（原窗口实测总耗时）')
            self.assertEqual(ledger_rows[0][1:5], [summary[key] for key in
                ('duration_ms_mean', 'recorded_busy_ms_mean', 'no_recorded_ms_mean', 'host_span_union_ms_mean')])
            self.assertIn('0/2', ledger_rows[0][-1])
            self.assertIn('UNRESOLVED', ledger_rows[0][-1])
        before = copy.deepcopy(diagnostic)
        rows, items, note = renderer._method_residual_data(case, method)
        self.assertEqual(len(rows), 2)
        self.assertAlmostEqual(sum(row[2] for row in rows), diagnostic['residual_ms_mean'])
        self.assertTrue(all(row[1].endswith('阶段未识别（原窗口实测总耗时）') for row in rows))
        self.assertTrue(all('0/2' in row[3] and 'SYNTHETIC_MISSING_STAGE_ANCHOR' in row[3] for row in rows))
        self.assertEqual(len(items), 2)
        self.assertTrue(all(item['unresolved_window'] for item in items))
        self.assertEqual(sum(item['value_ms'] for item in items), diagnostic['residual_ms_mean'])
        self.assertIn('含2个未识别整窗和0个已匹配阶段', note)
        self.assertIn('并非15阶段全匹配', note)
        self.assertNotIn('请重跑基础统计', note)
        html = renderer._method_residual_html(case, method)
        self.assertIn('阶段未识别（原窗口实测总耗时）', html)
        self.assertEqual(html.count('class="u-bar"'), 2)
        self.assertIn('轮次剩余区间2项耗时构成，含2个未识别整窗', html)
        out, (book, _) = self.build(report)
        self.assertEqual(diagnostic, before)
        self.assertTrue(all(row['duration_ms_mean'] is None for row in diagnostic['stage_summary']))
        cells = book['方法1估算'][1]
        self.assertTrue(any(isinstance(value, str) and value.endswith('阶段未识别（原窗口实测总耗时）')
                            for value in cells.values()))
        self.assertIn(original['summary'][0]['duration_ms_mean'], cells.values())
        self.assertIn('阶段未识别（原窗口实测总耗时）', out.with_suffix('.html').read_text())
        chart = next(chart for chart in self.chart_parts_by_sheet(out, book)['方法1估算']
                     if self.chart_title(chart).startswith('轮次剩余区间耗时构成'))
        series = self.assert_bar_cache_matches_own_sheet(chart, book, '方法1估算')
        self.assertEqual(len(series[0][0]), 2)
        self.assertNotEqual(series[0][0][0], series[0][0][1])
        self.assertEqual(sum(series[0][1]), diagnostic['residual_ms_mean'])

    def test_partial_unresolved_subgroup_never_claims_full_window_stage_or_occupancy_means(self):
        report = residual_report()
        case = report['cases'][0]
        method = case['projection']['layer_mean_methods'][0]
        diagnostic = case['measurements'][0]['step_residuals']
        summary = diagnostic['summary'][0]
        stages = [row for row in diagnostic['stage_summary'] if row['window_key'] == summary['window_key']]
        for row in stages:
            row.update(status='PARTIAL', matched_windows=1, duration_ms_mean=None)
        unknown = dict(window_key=summary['window_key'], stage_key='unresolved', stage_order=1,
            label='PARTIAL_UNRESOLVED', level=0, parent_stage_key=None, rounds=2, matched_windows=0,
            status='UNRESOLVED', duration_ms_mean=None,
            duration_ns_total=round(summary['duration_ms_mean'] * 1e6), reason='ONLY_ONE_ROUND_UNRESOLVED')
        diagnostic['stage_summary'].append(unknown)
        ledger_rows = renderer._residual_stage_rows([unknown], summary)
        self.assertEqual(ledger_rows[0][1:5], [renderer.NA] * 4)
        self.assertIn('UNRESOLVED', ledger_rows[0][-1])
        rows, items, note = renderer._method_residual_data(case, method)
        self.assertEqual(len(rows), 8)  # One known whole window and the other seven matched stages.
        self.assertEqual(rows[0][1], '阶段未识别（原窗口实测总耗时）')
        self.assertEqual(rows[0][2], summary['duration_ms_mean'])
        self.assertIn('1/2', rows[0][3])
        self.assertIn('ONLY_ONE_ROUND_UNRESOLVED', rows[0][3])
        self.assertIsNone(items)
        self.assertNotIn('class="u-bar"', renderer._method_residual_html(case, method))
        self.assertNotIn('请重跑基础统计', note)

    def test_unknown_whole_window_and_known_seven_stages_form_one_closed_chart(self):
        report = unresolved_window_report()
        before = copy.deepcopy(report)
        case = report['cases'][0]
        method = case['projection']['layer_mean_methods'][0]
        diagnostic = case['measurements'][0]['step_residuals']
        rows, items, note = renderer._method_residual_data(case, method)
        self.assertEqual(len(rows), 8)
        self.assertEqual(len(items), 8)
        self.assertEqual(items[0]['value_ms'], 3.25)
        self.assertTrue(items[0]['unresolved_window'])
        self.assertEqual(items[0]['window_key'], 'main_to_draft')
        self.assertTrue(items[0]['label'].startswith('主→草稿 · 阶段未识别'))
        self.assertIn('0/71 · UNRESOLVED', rows[0][3])
        self.assertIn('SYNTHETIC_MISSING_FINAL_HIDDEN_NORM', rows[0][3])
        self.assertTrue(all(row[-1] == '71/71 · MATCHED' for row in rows[1:]))
        self.assertEqual(sum(item['value_ms'] for item in items), diagnostic['residual_ms_mean'])
        self.assertEqual(sum(item['value_ms'] for item in items), method['source_partition']['residual_ms'])
        self.assertIn('含1个未识别整窗和7个已匹配阶段', note)
        self.assertIn('并非15阶段全匹配', note)
        self.assertNotIn('_CHILD_NOT_ADDED_', repr(items))
        self.assertNotIn('dp3_PARENT_', repr(items))
        html = renderer._method_residual_html(case, method)
        self.assertEqual(html.count('class="u-bar"'), 8)
        self.assertIn('轮次剩余区间8项耗时构成，含1个未识别整窗', html)
        self.assertNotIn('aria-label="轮次剩余区间15阶段', html)
        out, (book, _) = self.build(report)
        charts = self.chart_parts_by_sheet(out, book)['方法1估算']
        chart = next(chart for chart in charts if self.chart_title(chart).startswith('轮次剩余区间耗时构成'))
        self.assertIn('含未识别整窗', self.chart_title(chart))
        series = self.assert_bar_cache_matches_own_sheet(chart, book, '方法1估算')
        self.assertEqual(len(series), 1)
        self.assertEqual(series[0][0], [row[1] for row in rows])
        self.assertEqual(series[0][1], [item['value_ms'] for item in items])
        cells = book['方法1估算'][1]
        card_row = next(int(cell[1:]) for cell, value in cells.items()
                        if cell.startswith('A') and value == '完整周期 · ms')
        self.assertEqual(cells['A' + str(card_row + 1)], method['cycle_ms'])
        self.assertEqual(cells['E' + str(card_row + 1)], method['tpot_ms'])
        self.assertEqual(cells['I' + str(card_row + 1)], method['per_device_tokens_s'])
        self.assertIsNone(diagnostic['stage_summary'][0]['duration_ms_mean'])
        self.assertEqual(report, before)

    def test_whole_unresolved_window_requires_complete_rounds_and_totals(self):
        for invalid in ('subset', 'rounds', 'status', 'matched', 'no_total', 'missing_mean', 'negative_mean',
                        'nonfinite_mean', 'duplicate_summary', 'duplicate_stage', 'extra_partial',
                        'different_window_rounds', 'diagnostic_total', 'method_total', 'summary_total', 'known_window_partial'):
            report = unresolved_window_report()
            case = report['cases'][0]
            method = case['projection']['layer_mean_methods'][0]
            diagnostic = case['measurements'][0]['step_residuals']
            unknown, summary = diagnostic['stage_summary'][0], diagnostic['summary'][0]
            if invalid == 'subset':
                unknown['duration_ns_total'] //= 2
            elif invalid == 'rounds':
                unknown['rounds'] = 70
            elif invalid == 'status':
                unknown['status'] = 'PARTIAL'
            elif invalid == 'matched':
                unknown['matched_windows'] = 1
            elif invalid == 'no_total':
                unknown.pop('duration_ns_total')
            elif invalid == 'missing_mean':
                summary.pop('duration_ms_mean')
            elif invalid == 'negative_mean':
                summary['duration_ms_mean'] = -3.25
            elif invalid == 'nonfinite_mean':
                summary['duration_ms_mean'] = float('nan')
            elif invalid == 'duplicate_summary':
                diagnostic['summary'].append(copy.deepcopy(summary))
            elif invalid == 'duplicate_stage':
                diagnostic['stage_summary'].append(copy.deepcopy(unknown))
            elif invalid == 'extra_partial':
                diagnostic['stage_summary'].append(dict(diagnostic['stage_summary'][1],
                    window_key='main_to_draft', status='PARTIAL', matched_windows=1))
            elif invalid == 'different_window_rounds':
                diagnostic['summary'][1]['rounds'] = 70
                for stage in diagnostic['stage_summary'][1:]:
                    stage.update(rounds=70, matched_windows=70)
            elif invalid == 'diagnostic_total':
                diagnostic['residual_ms_mean'] += 1
            elif invalid == 'method_total':
                method['source_partition']['cross_role_tail_residual_ms'] += 1
            elif invalid == 'summary_total':
                summary['duration_ms_mean'] += 1
            else:
                diagnostic['stage_summary'][1].update(status='PARTIAL', matched_windows=1)
            with self.subTest(invalid=invalid):
                rows, items, note = renderer._method_residual_data(case, method)
                self.assertTrue(rows)
                self.assertIsNone(items)
                self.assertIn('未绘图', note)
                self.assertNotIn('class="u-bar"', renderer._method_residual_html(case, method))

    def test_zero_duration_known_stage_is_kept_in_partial_window_chart(self):
        report = unresolved_window_report()
        case = report['cases'][0]
        method = case['projection']['layer_mean_methods'][0]
        stages = case['measurements'][0]['step_residuals']['stage_summary']
        stages[2]['duration_ms_mean'] += stages[1]['duration_ms_mean']
        stages[1]['duration_ms_mean'] = 0.
        _, items, _ = renderer._method_residual_data(case, method)
        self.assertEqual(len(items), 8)
        self.assertEqual(items[1]['value_ms'], 0.)
        self.assertEqual(sum(item['value_ms'] for item in items), method['source_partition']['residual_ms'])

    def test_method_drilldown_tables_and_horizontal_charts_use_each_selected_rank(self):
        report = drilldown_report()
        out, (book, _) = self.build(report)
        by_sheet = self.chart_parts_by_sheet(out, book)
        case = report['cases'][0]
        for index, method in enumerate(case['projection']['layer_mean_methods'], 1):
            name = '方法' + str(index) + '估算'
            capture = next(row for row in case['measurements'] if row['rank'] == method['selected_rank'])
            other_rank = 'dp3' if capture['rank'] == 'dp7' else 'dp7'
            cells = book[name][1]
            title_rows = []
            for title in DRILLDOWN_TITLES:
                matches = [int(re.search(r'\d+', cell).group()) for cell, value in cells.items()
                           if isinstance(value, str) and value.startswith(title)]
                self.assertEqual(len(matches), 1, (name, title))
                title_rows.append(matches[0])
            u_row = next(int(re.search(r'\d+', cell).group()) for cell, value in cells.items()
                         if isinstance(value, str) and value.startswith('轮次剩余区间耗时构成'))
            rank_row = next(int(re.search(r'\d+', cell).group()) for cell, value in cells.items()
                            if value == '采样rank')
            self.assertEqual([u_row, *title_rows, rank_row], sorted([u_row, *title_rows, rank_row]))
            charts = [chart for chart in by_sheet[name]
                      if any(self.chart_title(chart).startswith(title) for title in DRILLDOWN_TITLES)]
            self.assertEqual(len(charts), 3)
            self.assertEqual([next(title for title in DRILLDOWN_TITLES if self.chart_title(chart).startswith(title))
                              for chart in charts], list(DRILLDOWN_TITLES))
            for chart in charts:
                self.assertIn(renderer._layer_mean_name(method), self.chart_title(chart))
                self.assert_bar_cache_matches_own_sheet(chart, book, name)
            swa = self.assert_bar_cache_matches_own_sheet(charts[0], book, name)
            self.assertEqual(len(swa), 1)
            target = next(group for group in method['target_stage_breakdowns'] if group['attention'] == 'SWA')
            expected = sorted(target['stage_rows'], key=lambda row: row['stage_order'])
            labels, values = swa[0]
            self.assertEqual(labels, [row['label'] for row in expected])
            self.assertEqual(values, [row['target_total_ms'] for row in expected])
            self.assertEqual(len(values), 10)
            self.assertAlmostEqual(sum(values), target['target_total_ms'])
            self.assertAlmostEqual(values[5], 1.26 if capture['rank'] == 'dp7' else 1.98)
            self.assertAlmostEqual(values[6], 1.92 if capture['rank'] == 'dp7' else 2.76)
            self.assertNotEqual(values, [row['source_ms'] for row in expected])
            for chart, parent_key in zip(charts[1:], ('main_inputs', 'main_metadata')):
                series = self.assert_bar_cache_matches_own_sheet(chart, book, name)
                self.assertEqual(len(series), 1)
                expected = sorted((row for row in capture['step_residuals']['substage_summary']
                                   if row['parent_stage_key'] == parent_key), key=lambda row: row['stage_order'])
                self.assertEqual(series[0][0], [row['label'] for row in expected])
                self.assertEqual(series[0][1], [row['duration_ms_mean'] for row in expected])
            self.assertFalse(any(other_rank + '_SWA_DETAIL_' in str(value)
                                 or other_rank + '_main_inputs_DETAIL_' in str(value)
                                 or other_rank + '_main_metadata_DETAIL_' in str(value)
                                 for value in cells.values()))
            # Expanded observations are not added to the core's already closed totals.
            card_row = next(int(cell[1:]) for cell, value in cells.items()
                            if cell.startswith('A') and value == '完整周期 · ms')
            self.assertEqual(cells['A' + str(card_row + 1)], method['cycle_ms'])
            self.assertEqual(cells['E' + str(card_row + 1)], method['tpot_ms'])

    def test_method_drilldown_drawing_anchors_leave_each_next_table_uncovered(self):
        out, (book, _) = self.build(drilldown_report())
        ns = {'c': 'http://schemas.openxmlformats.org/drawingml/2006/chart',
              'xdr': 'http://schemas.openxmlformats.org/drawingml/2006/spreadsheetDrawing',
              'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'}
        with ZipFile(out) as archive:
            for sheet_name in ('方法1估算', '方法2估算'):
                sheet_index = list(book).index(sheet_name) + 1
                drawing = ET.fromstring(archive.read('xl/drawings/drawing' + str(sheet_index) + '.xml'))
                rels = ET.fromstring(archive.read('xl/drawings/_rels/drawing' + str(sheet_index) + '.xml.rels'))
                links = {node.get('Id'): posixpath.normpath(posixpath.join('xl/drawings', node.get('Target')))
                         for node in rels}
                anchors = []
                for anchor in drawing.findall('xdr:twoCellAnchor', ns):
                    chart_id = anchor.find('.//c:chart', ns).get('{' + ns['r'] + '}id')
                    title = self.chart_title(ET.fromstring(archive.read(links[chart_id])))
                    if title.startswith('轮次剩余区间耗时构成') or any(
                            title.startswith(prefix) for prefix in DRILLDOWN_TITLES):
                        anchors.append((title, int(anchor.findtext('xdr:from/xdr:row', namespaces=ns)) + 1,
                                        int(anchor.findtext('xdr:to/xdr:row', namespaces=ns)) + 1))
                self.assertEqual(len(anchors), 4)
                for previous, following in zip(anchors, anchors[1:]):
                    self.assertLess(previous[2], following[1])
                cells = book[sheet_name][1]
                for (_, _, end), next_title in zip(anchors, (*DRILLDOWN_TITLES, '采样rank')):
                    next_row = next(int(re.search(r'\d+', cell).group()) for cell, value in cells.items()
                                    if isinstance(value, str) and value.startswith(next_title))
                    self.assertLessEqual(end, next_row)

    def test_method_drilldowns_missing_rank_duplicate_capture_and_old_json_keep_placeholders(self):
        for missing in ('selected_rank', 'capture', 'duplicate_capture', 'old_json'):
            report = drilldown_report()
            case = report['cases'][0]
            method = case['projection']['layer_mean_methods'][0]
            if missing == 'selected_rank':
                method.update(selected_rank=None, source_rank='dp3', status='UNAVAILABLE',
                              reasons=['SYNTHETIC_TARGET_INPUT_MISSING'])
            elif missing == 'capture':
                case['measurements'].pop(0)
            elif missing == 'duplicate_capture':
                case['measurements'].append(copy.deepcopy(case['measurements'][0]))
            else:
                case['measurements'][0].pop('swa_diagnostics')
                case['measurements'][0]['step_residuals'].pop('substage_summary')
                method.pop('target_stage_breakdowns')
            with self.subTest(missing=missing):
                out, (book, _) = self.build(report)
                cells = book['方法1估算'][1]
                for title in DRILLDOWN_TITLES:
                    self.assertTrue(any(isinstance(value, str) and value.startswith(title)
                                        for value in cells.values()), title)
                charts = self.chart_parts_by_sheet(out, book)['方法1估算']
                for title in DRILLDOWN_TITLES:
                    expected_target = missing in ('capture', 'duplicate_capture') and title == DRILLDOWN_TITLES[0]
                    self.assertEqual(any(self.chart_title(chart).startswith(title) for chart in charts), expected_target)
                text = ' '.join(map(str, cells.values()))
                self.assertTrue(any(word in text for word in ('缺', '不足', '未绘', '不可用', '未提供')))
                if missing not in ('capture', 'duplicate_capture'):
                    self.assertNotIn('dp3_SWA_DETAIL_', text)
                self.assertNotIn('dp3_main_inputs_DETAIL_', text)
                self.assertNotIn('dp3_main_metadata_DETAIL_', text)

    def test_partial_or_nonclosing_u_child_group_suppresses_only_its_chart(self):
        for invalid in ('partial', 'count', 'missing', 'parent_sum', 'window', 'level'):
            for parent_key, title in zip(('main_inputs', 'main_metadata'), DRILLDOWN_TITLES[1:]):
                report = drilldown_report()
                diagnostic = report['cases'][0]['measurements'][0]['step_residuals']
                child = next(row for row in diagnostic['substage_summary'] if row['parent_stage_key'] == parent_key)
                if invalid == 'partial':
                    child.update(status='PARTIAL', matched_windows=1)
                elif invalid == 'count':
                    child['rounds'] = 3
                elif invalid == 'missing':
                    diagnostic['substage_summary'].remove(child)
                elif invalid == 'parent_sum':
                    child['duration_ms_mean'] += 1.
                elif invalid == 'window':
                    child['window_key'] = 'main_to_draft'
                else:
                    child['level'] = 0
                with self.subTest(invalid=invalid, parent=parent_key):
                    out, (book, _) = self.build(report)
                    charts = self.chart_parts_by_sheet(out, book)['方法1估算']
                    self.assertFalse(any(self.chart_title(chart).startswith(title) for chart in charts))
                    other_title = DRILLDOWN_TITLES[2] if parent_key == 'main_inputs' else DRILLDOWN_TITLES[1]
                    self.assertTrue(any(self.chart_title(chart).startswith(other_title) for chart in charts))
                    self.assertTrue(any(isinstance(value, str) and value.startswith(title)
                                        for value in book['方法1估算'][1].values()))

    def test_incomplete_or_nonclosing_swa_layer_cannot_publish_a_ten_stage_chart(self):
        for invalid in ('partial', 'missing_stage', 'layer_sum', 'missing_layer_wall', 'sample_count', 'order'):
            report = drilldown_report()
            method = report['cases'][0]['projection']['layer_mean_methods'][0]
            group = next(row for row in method['target_stage_breakdowns'] if row['attention'] == 'SWA')
            if invalid == 'partial':
                group.update(status='UNAVAILABLE', reason='SYNTHETIC_PARTIAL_NATIVE_STAGE')
            elif invalid == 'missing_stage':
                group['stage_rows'].pop(0)
            elif invalid == 'layer_sum':
                group['target_wall_ms'] += .5
            elif invalid == 'missing_layer_wall':
                group.pop('target_wall_ms')
            elif invalid == 'sample_count':
                group['rounds'] = 0
            else:
                group['stage_rows'][0]['stage_order'] = 2
            with self.subTest(invalid=invalid):
                out, (book, _) = self.build(report)
                charts = self.chart_parts_by_sheet(out, book)['方法1估算']
                self.assertFalse(any(self.chart_title(chart).startswith(DRILLDOWN_TITLES[0]) for chart in charts))
                for title in DRILLDOWN_TITLES[1:]:
                    self.assertTrue(any(self.chart_title(chart).startswith(title) for chart in charts))

    def test_zero_duration_is_measured_and_keeps_its_bar_point_and_table_reference(self):
        report = drilldown_report()
        for capture in report['cases'][0]['measurements']:
            for layer in range(3):
                stages = [row for row in capture['swa_diagnostics']['stage_summary'] if row['layer'] == layer]
                stages[1]['duration_ms_mean'] += stages[0]['duration_ms_mean']
                stages[0]['duration_ms_mean'] = 0.
            for parent_key in ('main_inputs', 'main_metadata'):
                stages = [row for row in capture['step_residuals']['substage_summary']
                          if row['parent_stage_key'] == parent_key]
                stages[1]['duration_ms_mean'] += stages[0]['duration_ms_mean']
                stages[0]['duration_ms_mean'] = 0.
        for method in report['cases'][0]['projection']['layer_mean_methods']:
            group = next(item for item in method['target_stage_breakdowns'] if item['attention'] == 'SWA')
            first, second = group['stage_rows'][:2]
            for field in ('source_ms', 'target_ms', 'target_total_ms', 'cycle_share'):
                second[field] += first[field]
                first[field] = 0.
        out, (book, _) = self.build(report)
        by_sheet = self.chart_parts_by_sheet(out, book)
        for name in ('方法1估算', '方法2估算'):
            charts = [chart for chart in by_sheet[name] if any(
                self.chart_title(chart).startswith(title) for title in DRILLDOWN_TITLES)]
            self.assertEqual(len(charts), 3)
            for chart in charts:
                for _, values in self.assert_bar_cache_matches_own_sheet(chart, book, name):
                    self.assertEqual(values[0], 0.)
                    self.assertGreater(values[1], 0.)

    def test_main_csa_hca_and_draft_swa_have_five_separate_method_tables_and_charts(self):
        report = full_layer_drilldown_report()
        out, (book, _) = self.build(report)
        case = report['cases'][0]
        titles = (*MAIN_DRILLDOWN_TITLES, *DRILLDOWN_TITLES)
        by_sheet = self.chart_parts_by_sheet(out, book)
        for index, method in enumerate(case['projection']['layer_mean_methods'], 1):
            name = '方法' + str(index) + '估算'
            cells = book[name][1]
            capture = next(item for item in case['measurements'] if item['rank'] == method['selected_rank'])
            charts = [chart for chart in by_sheet[name] if any(self.chart_title(chart).startswith(title)
                                                              for title in titles)]
            self.assertEqual(len(charts), 5)
            title_rows = []
            for title, chart in zip(titles, charts):
                self.assertTrue(self.chart_title(chart).startswith(title))
                self.assertIn(renderer._layer_mean_name(method), self.chart_title(chart))
                matches = [int(re.search(r'\d+', cell).group()) for cell, value in cells.items()
                           if isinstance(value, str) and value.startswith(title)]
                self.assertEqual(len(matches), 1)
                title_rows.append(matches[0])
            u_row = next(int(re.search(r'\d+', cell).group()) for cell, value in cells.items()
                         if isinstance(value, str) and value.startswith('轮次剩余区间耗时构成'))
            rank_row = next(int(re.search(r'\d+', cell).group()) for cell, value in cells.items()
                            if value == '采样rank')
            self.assertEqual([u_row, *title_rows, rank_row], sorted([u_row, *title_rows, rank_row]))
            for chart, role, attention in zip(charts[:3], ('main', 'main', 'draft'), ('CSA', 'HCA', 'SWA')):
                series = self.assert_bar_cache_matches_own_sheet(chart, book, name)
                self.assertEqual(len(series), 1)
                parent = next(row for row in method['target_stage_breakdowns']
                              if row['role'] == role and row['attention'] == attention)
                expected = sorted(parent['stage_rows'], key=lambda row: row['stage_order'])
                labels, values = series[0]
                self.assertEqual(labels, [row['label'] for row in expected])
                self.assertEqual(values, [row['target_total_ms'] for row in expected])
                self.assertEqual(len(values), 10)
                self.assertAlmostEqual(sum(values), parent['target_total_ms'])
                self.assertAlmostEqual(sum(row['cycle_share'] for row in expected),
                                       parent['target_total_ms'] / method['cycle_ms'])
                self.assertNotEqual(values, [row['source_ms'] for row in expected])
                if role == 'main':
                    self.assertNotIn(0, parent['reference_layers'])
            text = ' '.join(map(str, cells.values()))
            self.assertIn('主0', text)
            self.assertTrue(any(word in text for word in ('不参与', '排除', '不作参考', '仅展示')))
            other_rank = 'dp3' if capture['rank'] == 'dp7' else 'dp7'
            for attention in ('CSA', 'HCA', 'SWA'):
                self.assertNotIn(other_rank + '_' + attention + '_DETAIL_', text)
            # The rendering extension keeps every result card at the core value.
            card_row = next(int(cell[1:]) for cell, value in cells.items()
                            if cell.startswith('A') and value == '完整周期 · ms')
            self.assertEqual(cells['A' + str(card_row + 1)], method['cycle_ms'])
            self.assertEqual(cells['E' + str(card_row + 1)], method['tpot_ms'])
            self.assertEqual(cells['I' + str(card_row + 1)], method['per_device_tokens_s'])
        preview = out.with_suffix('.html').read_text()
        for title in titles:
            self.assertEqual(preview.count('<h4>' + title), 2, title)
        self.assertEqual(preview.count('class="method-stage-expansion"'), 10)

    def test_incomplete_main_attention_class_keeps_table_and_suppresses_only_its_chart(self):
        for attention, title in zip(('CSA', 'HCA'), MAIN_DRILLDOWN_TITLES):
            for invalid in ('missing_stage', 'partial', 'missing_wall', 'nonclosing'):
                report = full_layer_drilldown_report()
                method = report['cases'][0]['projection']['layer_mean_methods'][0]
                group = next(row for row in method['target_stage_breakdowns'] if row['attention'] == attention)
                stage = group['stage_rows'][0]
                if invalid == 'missing_stage':
                    group['stage_rows'].remove(stage)
                elif invalid == 'partial':
                    group.update(status='UNAVAILABLE', reason='SYNTHETIC_PARTIAL_NATIVE_STAGE')
                elif invalid == 'missing_wall':
                    group.pop('target_wall_ms')
                else:
                    stage['target_total_ms'] += 1.
                with self.subTest(attention=attention, invalid=invalid):
                    out, (book, _) = self.build(report)
                    charts = self.chart_parts_by_sheet(out, book)['方法1估算']
                    self.assertFalse(any(self.chart_title(chart).startswith(title) for chart in charts))
                    other_title = MAIN_DRILLDOWN_TITLES[1] if attention == 'CSA' else MAIN_DRILLDOWN_TITLES[0]
                    self.assertTrue(any(self.chart_title(chart).startswith(other_title) for chart in charts))
                    self.assertTrue(any(self.chart_title(chart).startswith(DRILLDOWN_TITLES[0]) for chart in charts))
                    cells = book['方法1估算'][1]
                    self.assertTrue(any(isinstance(value, str) and value.startswith(title) for value in cells.values()))
                    self.assertIn('未绘图', ' '.join(map(str, cells.values())))

    def test_target_swa_only_contract_keeps_u_expansions_and_main_unknown_placeholders(self):
        out, (book, _) = self.build(drilldown_report())
        by_sheet = self.chart_parts_by_sheet(out, book)
        preview = out.with_suffix('.html').read_text()
        for name in ('方法1估算', '方法2估算'):
            for title in MAIN_DRILLDOWN_TITLES:
                self.assertTrue(any(isinstance(value, str) and value.startswith(title)
                                    for value in book[name][1].values()))
                self.assertFalse(any(self.chart_title(chart).startswith(title) for chart in by_sheet[name]))
                self.assertIn('<h4>' + title, preview)
            for title in DRILLDOWN_TITLES:
                self.assertTrue(any(self.chart_title(chart).startswith(title) for chart in by_sheet[name]))

    def test_core_target_contract_takes_precedence_over_source_layer_diagnostics(self):
        report = full_layer_drilldown_report()
        for capture in report['cases'][0]['measurements']:
            for diagnostic in (capture['swa_diagnostics'], capture['layer_diagnostics']):
                for row in diagnostic['stage_summary']:
                    row.update(label='SOURCE_STAGE_MUST_NOT_REPLACE_CORE_TARGET', duration_ms_mean=99999.)
        out, (book, _) = self.build(report)
        charts = self.chart_parts_by_sheet(out, book)
        for name in ('方法1估算', '方法2估算'):
            self.assertNotIn('SOURCE_STAGE_MUST_NOT_REPLACE_CORE_TARGET', book[name][1].values())
            swa = next(chart for chart in charts[name] if self.chart_title(chart).startswith(DRILLDOWN_TITLES[0]))
            series = self.assert_bar_cache_matches_own_sheet(swa, book, name)
            self.assertEqual(len(series), 1)
            self.assertTrue(all(value < 99999. for _, values in series for value in values))
            self.assertEqual(series[0][1][5], 1.26 if name == '方法1估算' else 1.98)

    def test_target_tables_keep_core_means_totals_percentages_and_notes(self):
        report = full_layer_drilldown_report()
        for method in report['cases'][0]['projection']['layer_mean_methods']:
            for group in method['target_stage_breakdowns']:
                group['note'] = 'CORE_GMM_UNKNOWN_DELTA_ZERO_RETAINED'
        before = copy.deepcopy(report)
        out, (book, _) = self.build(report)
        ns = {'s': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
        for number, method in enumerate(report['cases'][0]['projection']['layer_mean_methods'], 1):
            sheet, cells = book['方法' + str(number) + '估算']
            for group in method['target_stage_breakdowns']:
                for stage in group['stage_rows']:
                    row = next(int(cell[1:]) for cell, value in cells.items()
                               if cell.startswith('A') and value == stage['label'])
                    for column, field in (('D', 'source_ms'), ('E', 'target_ms'),
                                          ('F', 'target_total_ms'), ('H', 'cycle_share')):
                        self.assertEqual(cells[column + str(row)], stage[field])
                    percent_cell = sheet.find('.//s:c[@r="H' + str(row) + '"]', ns)
                    self.assertEqual(int(percent_cell.get('s')), renderer.xlsx_writer.STYLE_IDS['percent'])
                self.assertLess(sum(stage['cycle_share'] for stage in group['stage_rows']), 1.)
            text = ' '.join(map(str, cells.values()))
            self.assertIn('CORE_GMM_UNKNOWN_DELTA_ZERO_RETAINED', text)
            self.assertIn('激活与衔接不整体倍增', text)
            self.assertIn('源实测U仅保留一次', text)
        preview = out.with_suffix('.html').read_text()
        self.assertIn('CORE_GMM_UNKNOWN_DELTA_ZERO_RETAINED', preview)
        self.assertIn('目标同类全部层合计(ms/完整周期)', preview)
        self.assertRegex(preview, r'\d+\.\d{3}%')
        self.assertEqual(report, before)

    def test_old_json_never_relabels_source_stage_diagnostics_as_target(self):
        report = full_layer_drilldown_report()
        for method in report['cases'][0]['projection']['layer_mean_methods']:
            method.pop('target_stage_breakdowns')
        out, (book, _) = self.build(report)
        charts = self.chart_parts_by_sheet(out, book)
        for name in ('方法1估算', '方法2估算'):
            for title in (*MAIN_DRILLDOWN_TITLES, DRILLDOWN_TITLES[0]):
                self.assertFalse(any(self.chart_title(chart).startswith(title) for chart in charts[name]))
            for title in DRILLDOWN_TITLES[1:]:
                self.assertTrue(any(self.chart_title(chart).startswith(title) for chart in charts[name]))
            self.assertIn('请用新版核心重算', ' '.join(map(str, book[name][1].values())))
        source_text = ' '.join(map(str, book['层与组件实测'][1].values()))
        self.assertIn('dp7_SWA_DETAIL_1', source_text)

    def test_target_contract_does_not_require_source_diagnostic_tables(self):
        report = full_layer_drilldown_report()
        report['cases'][0]['measurements'] = []
        out, (book, _) = self.build(report)
        by_sheet = self.chart_parts_by_sheet(out, book)
        for name in ('方法1估算', '方法2估算'):
            for title in (*MAIN_DRILLDOWN_TITLES, DRILLDOWN_TITLES[0]):
                self.assertTrue(any(self.chart_title(chart).startswith(title) for chart in by_sheet[name]))
            for title in DRILLDOWN_TITLES[1:]:
                self.assertFalse(any(self.chart_title(chart).startswith(title) for chart in by_sheet[name]))

    def test_target_group_identity_reference_closure_and_cycle_share_are_checked(self):
        for invalid in ('duplicate', 'wrong_attention', 'wrong_role', 'main0', 'references', 'count',
                        'source_parent', 'target_parent', 'total_parent', 'share', 'delta', 'negative', 'missing_value'):
            report = full_layer_drilldown_report()
            case = report['cases'][0]
            method = case['projection']['layer_mean_methods'][0]
            group = method['target_stage_breakdowns'][0]
            if invalid == 'duplicate':
                method['target_stage_breakdowns'].append(copy.deepcopy(group))
            elif invalid == 'wrong_attention':
                group['attention'] = 'HCA'
            elif invalid == 'wrong_role':
                group['role'] = 'draft'
            elif invalid == 'main0':
                group['reference_layers'] = [0, 2, 4, 6]
            elif invalid == 'references':
                group['reference_layers'] = [1, 3, 5]
            elif invalid == 'count':
                group['target_layers'] += 1
            elif invalid in ('source_parent', 'target_parent', 'total_parent'):
                group[{'source_parent': 'source_wall_ms', 'target_parent': 'target_wall_ms',
                       'total_parent': 'target_total_ms'}[invalid]] += .25
            elif invalid == 'share':
                group['stage_rows'][0]['cycle_share'] += .01
            elif invalid == 'delta':
                group['stage_rows'][0]['delta_ms'] += .25
            elif invalid == 'negative':
                group['stage_rows'][0]['target_ms'] = -.25
            else:
                group['stage_rows'][0]['target_total_ms'] = None
            with self.subTest(invalid=invalid):
                rendered = renderer._method_expansion_data(case, method)
                self.assertIsNone(rendered[0]['series'])
                self.assertIsNotNone(rendered[2]['series'])
                self.assertIn('未绘图', rendered[0]['note'])

    def test_explicit_core_reference_layers_are_displayed_without_renderer_exclusions(self):
        report = full_layer_drilldown_report()
        case = report['cases'][0]
        case['config']['exclude_target_layers'] = []
        method = case['projection']['layer_mean_methods'][0]
        parent = next(row for row in method['class_means'] if row['attention'] == 'HCA')
        group = next(row for row in method['target_stage_breakdowns'] if row['attention'] == 'HCA')
        parent['reference_layers'] = group['reference_layers'] = [0, 1, 3, 5, 7]
        rendered = renderer._method_expansion_data(case, method)[1]
        self.assertIsNotNone(rendered['series'])
        self.assertIn('[0,1,3,5,7]', rendered['note'])
        self.assertNotIn('主0排除', rendered['note'])
        self.assertEqual(rendered['series'][0][1], [row['target_total_ms'] for row in group['stage_rows']])

    def test_optional_input_anchors_and_dynamic_metadata_counts_use_observed_stage_order(self):
        report = drilldown_report()
        for capture in report['cases'][0]['measurements']:
            diagnostic = capture['step_residuals']
            for parent_key, count in (('main_inputs', 5), ('main_metadata', 7)):
                parent = next(row for row in diagnostic['stage_summary'] if row['stage_key'] == parent_key)
                children = [row for row in diagnostic['substage_summary'] if row['parent_stage_key'] == parent_key]
                diagnostic['substage_summary'] = [row for row in diagnostic['substage_summary']
                                                   if row['parent_stage_key'] != parent_key]
                for number in range(count):
                    diagnostic['substage_summary'].append(dict(children[0],
                        stage_key=parent_key + '_dynamic_' + str(number),
                        stage_order=parent['stage_order'] * 1000 + number * 2,
                        label=capture['rank'] + '_' + parent_key + '_DYNAMIC_' + str(number),
                        duration_ms_mean=parent['duration_ms_mean'] / count))
            diagnostic['substage_summary'].reverse()
            capture['swa_diagnostics']['stage_summary'].reverse()
            capture['swa_diagnostics']['layer_summary'].reverse()
        for method in report['cases'][0]['projection']['layer_mean_methods']:
            for group in method['target_stage_breakdowns']:
                group['stage_rows'].reverse()
        out, (book, _) = self.build(report)
        by_sheet = self.chart_parts_by_sheet(out, book)
        for name, rank in (('方法1估算', 'dp7'), ('方法2估算', 'dp3')):
            for title, parent_key, count in zip(DRILLDOWN_TITLES[1:], ('main_inputs', 'main_metadata'), (5, 7)):
                chart = next(chart for chart in by_sheet[name] if self.chart_title(chart).startswith(title))
                series = self.assert_bar_cache_matches_own_sheet(chart, book, name)
                self.assertEqual(series[0][0], [rank + '_' + parent_key + '_DYNAMIC_' + str(number)
                                               for number in range(count)])
                self.assertEqual(len(series[0][1]), count)
            swa = next(chart for chart in by_sheet[name] if self.chart_title(chart).startswith(DRILLDOWN_TITLES[0]))
            series = self.assert_bar_cache_matches_own_sheet(swa, book, name)
            self.assertEqual(len(series), 1)
            self.assertEqual(series[0][0], [rank + '_SWA_DETAIL_' + str(order) for order in range(1, 11)])
            self.assertEqual(series[0][1][0], .06 if rank == 'dp7' else .18)
            self.assertLess(series[0][1][0], series[0][1][1])

    def test_default_renderer_filename_overview_and_titles_use_product_name(self):
        source = self.root / 'core.json'
        source.write_text(json.dumps(layer_mean_fixture()), encoding='utf-8')
        result = subprocess.run([sys.executable, '-S', str(Path(renderer.__file__).resolve()), '--report', str(source)],
                                cwd=self.root, text=True, capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        out = self.root / '10T模型推理估算.xlsx'
        self.assertTrue(out.is_file())
        book, _ = read_book(out)
        self.assertEqual(next(iter(book)), '估算总览')
        self.assertEqual(book['估算总览'][1]['A1'], '10T模型推理估算')
        preview = out.with_suffix('.html').read_text()
        self.assertIn('<title>10T模型推理估算</title>', preview)
        self.assertIn('<h1>10T模型推理估算</h1>', preview)


if __name__ == '__main__':
    unittest.main()
