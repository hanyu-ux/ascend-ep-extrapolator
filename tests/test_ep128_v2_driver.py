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

    def test_acceptance_defaults_to_output_fraction_and_old_semantics_is_rejected(self):
        c = case(); del c['effective_tokens_per_round']; del c['token_basis']; c['acceptance'] = .6
        result = p.normalize_case({}, c, Path.cwd())
        self.assertEqual(result['effective_tokens_per_round'], 4.8)
        self.assertEqual(result['acceptance_semantics'], 'mean_output_fraction_including_bonus')
        c['acceptance_semantics'] = 'mean_accepted_draft_fraction'
        with self.assertRaisesRegex(ValueError, 'historical|旧'):
            p.normalize_case({}, c, Path.cwd())

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


class MixedSpeculationConfigTests(unittest.TestCase):
    """Independent samples may change S while sharing an explicit token policy."""
    def defaults(self, *, rate=True):
        values = case()
        for key in ('name', 'bs', 'draft_tokens', 'effective_tokens_per_round', 'token_basis'):
            values.pop(key)
        if rate:
            values.update(acceptance=.6, acceptance_semantics='mean_output_fraction_including_bonus')
        return values

    def normalize(self, defaults, item):
        before = copy.deepcopy((defaults, item))
        try:
            return p.normalize_case(defaults, item, Path.cwd())
        finally:
            self.assertEqual((defaults, item), before, 'normalization must not mutate shared input objects')

    def test_speculative_token_alias_is_canonical_and_case_overrides_defaults(self):
        configurations = (
            ({}, {'num_speculative_tokens': 9}, 9),
            ({'num_speculative_tokens': 7}, {}, 7),
            ({'draft_tokens': 7}, {'num_speculative_tokens': 9}, 9),
            ({'num_speculative_tokens': 7}, {'draft_tokens': 9}, 9),
            ({'draft_tokens': 7, 'num_speculative_tokens': 7},
             {'draft_tokens': 9, 'num_speculative_tokens': 9}, 9),
        )
        for common, local, expected in configurations:
            with self.subTest(defaults=common, item=local):
                result = self.normalize({**self.defaults(), **common},
                    {'name': 'independent-sample', 'bs': 8, **local})
                self.assertEqual(result['draft_tokens'], expected)
                self.assertNotIn('num_speculative_tokens', result)
                self.assertEqual(result['effective_tokens_per_round'], {7: 4.8, 9: 6.0}[expected])

    def test_same_layer_conflicting_token_names_are_rejected_even_when_overridden(self):
        for common, local in (
            ({'draft_tokens': 7, 'num_speculative_tokens': 9}, {'draft_tokens': 9}),
            ({'draft_tokens': 7}, {'draft_tokens': 9, 'num_speculative_tokens': 7}),
        ):
            with self.subTest(defaults=common, item=local), self.assertRaises(ValueError):
                self.normalize({**self.defaults(), **common}, {'name': 'conflict', 'bs': 8, **local})

    def test_both_token_names_validate_their_own_layer_without_coercion(self):
        for field in ('draft_tokens', 'num_speculative_tokens'):
            for invalid in (None, True, -1, 9.0, '9'):
                for layer in ('defaults', 'item'):
                    common = self.defaults()
                    local = {'name': 'invalid-length', 'bs': 8, 'draft_tokens': 9}
                    if layer == 'defaults':
                        common[field] = invalid  # An explicit valid case cannot hide bad defaults.
                    else:
                        common['draft_tokens'] = 7
                        local[field] = invalid
                    with self.subTest(field=field, value=invalid, layer=layer), self.assertRaises(ValueError):
                        self.normalize(common, local)

    def test_sixty_percent_policy_gives_exact_lengths_for_independent_samples(self):
        common = self.defaults()
        normalized = []
        for length, expected in ((3, 2.4), (5, 3.6), (7, 4.8), (9, 6.0)):
            local = {'name': f'bs8-s{length}', 'bs': 8, 'num_speculative_tokens': length,
                     'path': f'generic-independent-samples/s{length}'}
            with self.subTest(length=length):
                result = self.normalize(common, local)
                self.assertEqual(result['effective_tokens_per_round'], expected)
                self.assertIn('bonus', result['token_basis'])
                self.assertEqual(result['path'], local['path'])
                normalized.append(result)
        self.assertEqual([entry['draft_tokens'] for entry in normalized], [3, 5, 7, 9])
        self.assertEqual(len({entry['name'] for entry in normalized}), 4)

    def test_case_token_selectors_replace_inherited_fixed_length_and_basis(self):
        old_basis = 'Old explicit S7 assumption including bonus'
        selectors = (
            {'effective_tokens_per_round': 6.0, 'token_basis': 'New explicit S9 assumption including bonus'},
            {'synthetic_acceptance_length': 6.0},
            {'acceptance': .6},
            {'acceptance_semantics': 'mean_output_fraction_including_bonus'},
        )
        for inherited in ('effective_tokens_per_round', 'synthetic_acceptance_length'):
            for selector in selectors:
                common = {**self.defaults(), inherited: 5.2, 'token_basis': old_basis}
                local = {'name': 's9-independent', 'bs': 8, 'num_speculative_tokens': 9, **selector}
                with self.subTest(inherited=inherited, selector=selector):
                    result = self.normalize(common, local)
                    self.assertEqual(result['effective_tokens_per_round'], 6.0)
                    self.assertNotEqual(result['token_basis'], old_basis)
                    self.assertTrue(result['token_basis'])
        # Preserve the legacy explicit-Neff contract: its explanation may
        # come from defaults, while SAL/rate paths generate their own basis.
        common = {**self.defaults(rate=False), 'effective_tokens_per_round': 5.2, 'token_basis': old_basis}
        local = {'name': 'inherited-basis', 'bs': 8, 'draft_tokens': 9, 'effective_tokens_per_round': 6.4}
        result = self.normalize(common, local)
        self.assertEqual(result['effective_tokens_per_round'], 6.4)
        self.assertEqual(result['token_basis'], old_basis)
        common.pop('token_basis')
        with self.assertRaisesRegex(ValueError, 'token_basis'):
            self.normalize(common, local)

    def test_explicit_lengths_and_recognized_acceptance_rate_must_agree(self):
        valid = {'name': 'consistent-s9', 'bs': 8, 'draft_tokens': 9,
                 'effective_tokens_per_round': 6.0, 'synthetic_acceptance_length': 6.0,
                 'token_basis': 'Configured including one bonus token', 'acceptance': .6,
                 'acceptance_semantics': 'mean_output_fraction_including_bonus'}
        self.assertEqual(self.normalize(self.defaults(), valid)['effective_tokens_per_round'], 6.0)
        for altered in (
            {'synthetic_acceptance_length': 5.2},
            {'effective_tokens_per_round': 5.2},
            {'acceptance': .5},
            {'synthetic_acceptance_length': 5.2, 'effective_tokens_per_round': 5.2},
        ):
            with self.subTest(altered=altered), self.assertRaises(ValueError):
                self.normalize(self.defaults(), {**valid, **altered})
        result = self.normalize(self.defaults(), {'name': 'sal-overrides-default-rate', 'bs': 8,
                                'draft_tokens': 9, 'synthetic_acceptance_length': 5.2})
        self.assertEqual(result['effective_tokens_per_round'], 5.2)
        self.assertNotIn('acceptance', result)
        self.assertNotIn('acceptance_semantics', result)
        # Explicit Neff alone retains the existing default-rate constraint.
        with self.assertRaises(ValueError):
            self.normalize(self.defaults(), {'name': 'conflicting-s9', 'bs': 8, 'draft_tokens': 9,
                'effective_tokens_per_round': 5.2, 'token_basis': 'Including bonus'})

    def test_independent_samples_need_only_speculative_count_and_synthetic_length(self):
        common = self.defaults(rate=False)
        for spelling in ('synthetic_acceptance_length', 'synthetic_acception_length'):
            normalized = []
            for length, effective in ((3, 2.3), (5, 4.4), (7, 5.2), (9, 6.4)):
                with self.subTest(spelling=spelling, length=length):
                    local = {'name': f'bs8-s{length}', 'bs': 8, 'num_speculative_tokens': length,
                             spelling: effective, 'path': f'generic-independent-samples/s{length}'}
                    result = self.normalize(common, local)
                    self.assertEqual(result['draft_tokens'], length)
                    self.assertEqual(result['synthetic_acceptance_length'], effective)
                    self.assertEqual(result['effective_tokens_per_round'], effective)
                    self.assertIn('bonus', result['token_basis'])
                    self.assertNotIn('num_speculative_tokens', result)
                    self.assertNotIn('synthetic_acception_length', result)
                    self.assertNotIn('acceptance', result)
                    self.assertNotIn('acceptance_semantics', result)
                    normalized.append(result)
            self.assertEqual(len({entry['name'] for entry in normalized}), 4)
            self.assertEqual(len({entry['path'] for entry in normalized}), 4)

    def test_synthetic_length_spellings_share_canonical_case_override_behavior(self):
        configurations = (
            ({'synthetic_acception_length': 5.2}, {}, 5.2),
            ({'synthetic_acceptance_length': 5.2}, {'synthetic_acception_length': 6.4}, 6.4),
            ({'synthetic_acception_length': 5.2}, {'synthetic_acceptance_length': 6.4}, 6.4),
            ({'synthetic_acceptance_length': 5.2, 'synthetic_acception_length': 5.2},
             {'synthetic_acceptance_length': 6.4, 'synthetic_acception_length': 6.4}, 6.4),
        )
        for inherited, explicit, expected in configurations:
            with self.subTest(defaults=inherited, item=explicit):
                result = self.normalize({**self.defaults(rate=False), **inherited},
                    {'name': 'canonical-sal', 'bs': 8, 'num_speculative_tokens': 9, **explicit})
                self.assertEqual(result['synthetic_acceptance_length'], expected)
                self.assertEqual(result['effective_tokens_per_round'], expected)
                self.assertNotIn('synthetic_acception_length', result)
        for inherited, explicit in (
            ({'synthetic_acceptance_length': 5.2, 'synthetic_acception_length': 6.4},
             {'synthetic_acceptance_length': 6.4}),
            ({'synthetic_acceptance_length': 5.2},
             {'synthetic_acceptance_length': 5.2, 'synthetic_acception_length': 6.4}),
        ):
            with self.subTest(conflicting_defaults=inherited, conflicting_item=explicit), self.assertRaises(ValueError):
                self.normalize({**self.defaults(rate=False), **inherited},
                    {'name': 'conflicting-sal', 'bs': 8, 'draft_tokens': 9, **explicit})

    def test_synthetic_length_aliases_reject_invalid_values_before_inheritance(self):
        invalid_values = (None, True, False, '5.2', 0, .999, -1, float('nan'), float('inf'), -float('inf'))
        for spelling in ('synthetic_acceptance_length', 'synthetic_acception_length'):
            for invalid in invalid_values:
                for layer in ('defaults', 'item'):
                    common = self.defaults(rate=False)
                    local = {'name': 'invalid-sal', 'bs': 8, 'num_speculative_tokens': 9,
                             'synthetic_acceptance_length': 6.4}
                    if layer == 'defaults':
                        common[spelling] = invalid  # Explicit overrides must not hide malformed defaults.
                    else:
                        local[spelling] = invalid
                    with self.subTest(spelling=spelling, invalid=invalid, layer=layer), self.assertRaises(ValueError):
                        self.normalize(common, local)
        for spelling in ('synthetic_acceptance_length', 'synthetic_acception_length'):
            for length in (3, 9):
                for effective in (1, length + 1):
                    with self.subTest(spelling=spelling, length=length, boundary=effective):
                        result = self.normalize(self.defaults(rate=False),
                            {'name': 'sal-boundary', 'bs': 8, 'num_speculative_tokens': length, spelling: effective})
                        self.assertEqual(result['effective_tokens_per_round'], effective)
                with self.subTest(spelling=spelling, length=length, outside='S+1'), self.assertRaises(ValueError):
                    self.normalize(self.defaults(rate=False),
                        {'name': 'too-large-sal', 'bs': 8, 'num_speculative_tokens': length, spelling: length + 1.01})
                # Approximate equality to an explicit Neff must never bypass
                # the strict [1,S+1] domain of the separately declared SAL.
                with self.subTest(spelling=spelling, length=length, outside='near-S+1'), self.assertRaises(ValueError):
                    self.normalize(self.defaults(rate=False), {'name': 'near-boundary-sal', 'bs': 8,
                        'num_speculative_tokens': length, spelling: length + 1 + 1e-9,
                        'effective_tokens_per_round': length + 1, 'token_basis': 'Including one bonus token'})
                with self.subTest(spelling=spelling, length=length, inside='small-valid-difference'):
                    result = self.normalize(self.defaults(rate=False), {'name': 'in-range-sal', 'bs': 8,
                        'num_speculative_tokens': length, spelling: length + .5 + 1e-9,
                        'effective_tokens_per_round': length + .5, 'token_basis': 'Including one bonus token'})
                    self.assertEqual(result['effective_tokens_per_round'], length + .5)

    def test_explicit_synthetic_length_isolates_old_default_rate_but_preserves_explicit_rate_checks(self):
        for spelling in ('synthetic_acceptance_length', 'synthetic_acception_length'):
            common = {**self.defaults(), 'effective_tokens_per_round': 5.2,
                      'token_basis': 'Old configured S7 mean including bonus'}
            local = {'name': 'independent-s9-sal', 'bs': 8, 'num_speculative_tokens': 9, spelling: 3.3}
            result = self.normalize(common, local)
            self.assertEqual(result['effective_tokens_per_round'], 3.3)
            self.assertNotIn('acceptance', result)
            self.assertNotIn('acceptance_semantics', result)
            self.assertNotEqual(result['token_basis'], common['token_basis'])
            for explicit_rate in ({'acceptance': .6},
                                  {'acceptance_semantics': 'mean_accepted_draft_fraction'},
                                  {'acceptance': .6, 'acceptance_semantics': 'mean_accepted_draft_fraction'}):
                with self.subTest(spelling=spelling, explicit_rate=explicit_rate), self.assertRaises(ValueError):
                    self.normalize(common, {**local, **explicit_rate})

    def test_legacy_explicit_neff_is_preserved_without_inferring_acceptance(self):
        common = {**self.defaults(rate=False), 'draft_tokens': 7,
                  'effective_tokens_per_round': 5.2, 'token_basis': 'Legacy configured mean including bonus'}
        for local in ({'draft_tokens': 9}, {'num_speculative_tokens': 9}):
            with self.subTest(local=local):
                result = self.normalize(common, {'name': 'legacy-explicit', 'bs': 8, **local})
                self.assertEqual(result['draft_tokens'], 9)
                self.assertEqual(result['effective_tokens_per_round'], 5.2)
                self.assertEqual(result['token_basis'], common['token_basis'])
                self.assertNotIn('acceptance', result)
        # New acceptance values always constrain Neff; unknown semantics and
        # inconsistent rate/explicit-length declarations cannot be ignored.
        for semantics in (None, 'request_success_fraction'):
            explicit = {**case(), 'acceptance': .1}
            if semantics is not None:
                explicit['acceptance_semantics'] = semantics
            with self.subTest(unrecognized_semantics=semantics), self.assertRaises(ValueError):
                self.normalize({}, explicit)


if __name__ == '__main__': unittest.main()
