"""Meaningful V2 aggregation/input tests; synthetic timing is not hardware validation."""
import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest

import profile_ep128 as p


def case():
    return {'name': 'bs32-s7', 'bs': 32, 'draft_tokens': 7, 'with_stack': False,
            'base_ep': 16, 'target_ep': 128, 'expected_ranks': 16,
            'source_dp': 16, 'source_devices': 16, 'target_dp': 128, 'target_devices': 128,
            'effective_tokens_per_round': 5.2, 'token_basis': 'synthetic including bonus; conditional'}


def measurements():
    return [{'rank': 'dp0', 'rounds': [
        {'round_id': 'r0', 'duration_ms': 100, 'output_tokens': 32 * 5.2,
         'token_basis': 'synthetic including bonus', 'scope': 'HC_cycle_not_verified_request_iteration',
         'layers': [], 'ledger': {'recorded_busy_ms': 80, 'no_recorded_ms': 20}},
        {'round_id': 'r1', 'duration_ms': 300, 'output_tokens': 32 * 5.2,
         'token_basis': 'synthetic including bonus', 'scope': 'HC_cycle_not_verified_request_iteration',
         'layers': [], 'ledger': {'recorded_busy_ms': 250, 'no_recorded_ms': 50}}]}]


class V2DriverTests(unittest.TestCase):
    def test_ratio_of_sums_not_mean_of_rates(self):
        c = case(); c['_rank_coverage_complete'] = False
        result = p.summarize_source(measurements(), c)
        self.assertAlmostEqual(result['per_rank'][0]['conditional_per_rank_tokens_s'], 832)
        self.assertAlmostEqual(result['aggregate']['per_device_tokens_s'], 832)
        self.assertAlmostEqual(result['aggregate']['global_tokens_s'], 832 * 16)
        self.assertFalse(result['aggregate']['rank_coverage_complete'])

    def test_general_topology_not_ep_or_csv_count(self):
        c = case(); c.update(source_dp=4, source_devices=8, base_ep=8, _rank_coverage_complete=False)
        result = p.summarize_source(measurements(), c)
        self.assertAlmostEqual(result['aggregate']['per_device_tokens_s'], 416)
        self.assertAlmostEqual(result['aggregate']['global_tokens_s'], 3328)

    def test_topology_missing_means_pending_system_rates(self):
        c = case(); c.update(source_dp=None, source_devices=None, _rank_coverage_complete=False)
        result = p.summarize_source(measurements(), c)
        self.assertIsNone(result['aggregate']['per_device_tokens_s'])
        self.assertEqual(result['per_rank'][0]['conditional_per_rank_tokens_s'], 832)

    def test_max_rank_mean_and_same_rank_rate(self):
        m = measurements()
        second = copy.deepcopy(m[0]); second['rank'] = 'dp1'
        for r in second['rounds']: r['duration_ms'] *= 2
        m.append(second)
        c = case(); c['_rank_coverage_complete'] = False
        result = p.summarize_source(m, c)
        self.assertEqual(result['aggregate']['selected_rank'], 'dp1')
        self.assertEqual(result['aggregate']['cycle_ms'], 400)
        self.assertEqual(result['aggregate']['per_device_tokens_s'], 416)

    def test_effective_length_no_bonus_double_add(self):
        c = p.normalize_case({}, case(), Path.cwd())
        self.assertEqual(c['effective_tokens_per_round'], 5.2)

    def test_ambiguous_acceptance_rejected(self):
        c = case(); del c['effective_tokens_per_round']; del c['token_basis']; c['acceptance'] = .6
        with self.assertRaisesRegex(ValueError, 'ambiguous'):
            p.normalize_case({}, c, Path.cwd())
        c['acceptance_semantics'] = 'mean_accepted_draft_fraction'
        self.assertEqual(p.normalize_case({}, c, Path.cwd())['effective_tokens_per_round'], 5.2)

    def test_neff_and_integer_guards(self):
        for key, value in [('bs', True), ('draft_tokens', 7.1), ('effective_tokens_per_round', 9), ('target_devices', 0), ('source_chips_per_card', 0)]:
            c = case(); c[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                p.normalize_case({}, c, Path.cwd())

    def test_missing_dp_not_inherited_from_ep(self):
        c = case(); del c['source_dp']; del c['target_dp']
        result = p.normalize_case({}, c, Path.cwd())
        self.assertIsNone(result['source_dp'])
        self.assertIsNone(result['target_dp'])

    def test_main_zero_is_excluded_from_reference_without_reducing_depth(self):
        c = case(); c['exclude_target_layers'] = [0]
        normalized = p.normalize_case({}, c, Path.cwd())
        self.assertEqual(normalized['exclude_target_layers'], [0])
        self.assertEqual(normalized['draft_layers'], 3)
        c['exclude_target_layers'] = [0, 1]
        with self.assertRaisesRegex(ValueError, 'exclude_target_layers'):
            p.normalize_case({}, c, Path.cwd())

    def test_capture_rank_nearest_and_coverage(self):
        self.assertEqual(p.rank_name(Path('/data/parent_dp128/node_rank3/ASCEND_PROFILER_OUTPUT/kernel_details.csv'), 0), 'dp3')
        self.assertFalse(p.rank_coverage(['trace' + str(i) for i in range(16)], case()))
        self.assertTrue(p.rank_coverage(['dp' + str(i) for i in range(16)], case()))

    def test_duplicate_capture_rank_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            for host in ('a', 'b'):
                path = base / host / 'rank0' / 'kernel_details.csv'
                path.parent.mkdir(parents=True); path.write_text('x')
            with self.assertRaisesRegex(ValueError, 'Duplicate global rank'):
                p.case_inputs({'path': str(base)}, base)
            files, names = p.case_inputs({'inputs': [{'path': str(base / 'a/rank0/kernel_details.csv'), 'rank': 'dp0'}, {'path': str(base / 'b/rank0/kernel_details.csv'), 'rank': 'dp8'}]}, base)
            self.assertEqual(names, ['dp0', 'dp8'])
            self.assertEqual(len(files), 2)

    def test_pending_public_result_does_not_leak_internal_target_number(self):
        c = case(); c['_rank_coverage_complete'] = False
        source = p.summarize_source(measurements(), c)
        report = {'cases': [{'name': 'test', 'source': source, 'projection': {'display_method_set': 'layer_mean_methods', 'layer_mean_methods': [
            {'id': 'method1', 'name': 'M1', 'status': 'UNAVAILABLE', 'cycle_ms': 100,
             'per_device_tokens_s': 1664, 'global_tokens_s': 212992, 'reasons': ['missing native GMM for changed TopK']}]}}], 'failures': []}
        rows = p.result_rows(report)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertIsNone(row['周期(ms)'])
        self.assertIsNone(row['单卡吞吐(token/s/卡)'])

    def test_csv_none_and_nested_value_not_zero(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'out.csv'
            p.write_csv(path, [{'value': None, 'reasons': ['a', 'b'], 'number': 1.5}])
            with path.open(encoding='utf-8-sig', newline='') as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual(row['value'], '')
            self.assertEqual(json.loads(row['reasons']), ['a', 'b'])
            self.assertEqual(row['number'], '1.5')


if __name__ == '__main__': unittest.main()
