"""Independent configured numerator checks; no measured acceptance is invented."""
import copy
import math
import unittest

from tools import validate_ep128_capture as independent


class AcceptanceValidatorTests(unittest.TestCase):
    def test_rate_only_defaults_to_new_including_bonus_formula(self):
        for rate, expected in ((.6, 4.8), (.65, 5.2), (1., 8.), (.125, 1.)):
            with self.subTest(rate=rate):
                value, metadata = independent.configured_effective_output(
                    {'draft_tokens': 7, 'acceptance': rate})
                self.assertEqual(value, expected)
                self.assertEqual(metadata['acceptance_semantics'], 'mean_output_fraction_including_bonus')
                self.assertEqual(metadata['acceptance'], rate)
                self.assertIn('NOT_MEASURED', metadata['basis'])

    def test_explicit_new_semantics_and_lengths_must_agree(self):
        case = {'draft_tokens': 7, 'acceptance': .65,
                'acceptance_semantics': 'mean_output_fraction_including_bonus',
                'synthetic_acceptance_length': 5.2, 'effective_tokens_per_round': 5.2}
        original = copy.deepcopy(case)
        self.assertEqual(independent.configured_effective_output(case)[0], 5.2)
        self.assertEqual(case, original)
        for changed in ({'acceptance': .6}, {'effective_tokens_per_round': 4.8},
                        {'synthetic_acceptance_length': 4.8}):
            with self.subTest(changed=changed), self.assertRaisesRegex(ValueError, 'conflicts'):
                independent.configured_effective_output({**case, **changed})

    def test_old_or_unknown_semantics_are_rejected_even_with_explicit_sal(self):
        for semantics in ('mean_accepted_draft_fraction', 'conditional_per_position_probability', '', False):
            with self.subTest(semantics=semantics), self.assertRaises(ValueError):
                independent.configured_effective_output({'draft_tokens': 7, 'acceptance': .6,
                    'synthetic_acceptance_length': 5.2, 'acceptance_semantics': semantics})
        with self.assertRaisesRegex(ValueError, 'SAL/\(1\+S\)'):
            independent.configured_effective_output({'draft_tokens': 7, 'acceptance': .6,
                'acceptance_semantics': 'mean_accepted_draft_fraction'})

    def test_rate_and_output_length_bounds_preserve_one_output_minimum(self):
        for bad in (None, True, '0.65', float('nan'), float('inf'), -1, 0, .124999, 1.001):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                independent.configured_effective_output({'draft_tokens': 7, 'acceptance': bad})
        for field in ('effective_tokens_per_round', 'synthetic_acceptance_length'):
            for bad in (0, .5, 8.1, True, float('nan')):
                with self.subTest(field=field, bad=bad), self.assertRaises(ValueError):
                    independent.configured_effective_output({'draft_tokens': 7, field: bad})
        self.assertEqual(independent.configured_effective_output({'draft_tokens': 0, 'acceptance': 1})[0], 1)
        with self.assertRaisesRegex(ValueError, 'at least 1/\(1\+S\)'):
            independent.configured_effective_output({'draft_tokens': 0, 'acceptance': .9})
        with self.assertRaises(ValueError):
            independent.configured_effective_output({'draft_tokens': 7,
                'acceptance_semantics': 'mean_output_fraction_including_bonus'})
        self.assertEqual(independent.configured_effective_output({'draft_tokens': 2, 'acceptance': 1 / 3})[0], 1)
        with self.assertRaisesRegex(ValueError, 'at least 1/\(1\+S\)'):
            independent.configured_effective_output({'draft_tokens': 2,
                'acceptance': math.nextafter(1 / 3, 0)})

    def test_aliases_are_normalized_without_double_bonus(self):
        value, metadata = independent.configured_effective_output({
            'num_speculative_tokens': 7, 'synthetic_acception_length': 5.2})
        self.assertEqual(value, 5.2)
        self.assertIsNone(metadata['acceptance'])
        self.assertIsNone(metadata['acceptance_semantics'])
        for case in ({'draft_tokens': 7, 'num_speculative_tokens': 9, 'acceptance': .65},
                     {'draft_tokens': 7, 'synthetic_acceptance_length': 5.2,
                      'synthetic_acception_length': 4.8}):
            with self.subTest(case=case), self.assertRaisesRegex(ValueError, 'conflicts'):
                independent.configured_effective_output(case)

    def test_case_output_policy_replaces_inherited_length_or_legacy_rate(self):
        defaults = {'draft_tokens': 7, 'effective_tokens_per_round': 5.2,
                    'synthetic_acceptance_length': 5.2, 'acceptance': .6,
                    'acceptance_semantics': 'mean_accepted_draft_fraction'}
        original = copy.deepcopy(defaults)
        case = independent.merge_output_case(defaults, {'num_speculative_tokens': 9,
            'synthetic_acception_length': 6.4})
        self.assertEqual(independent.configured_effective_output(case)[0], 6.4)
        self.assertNotIn('acceptance', case)
        self.assertNotIn('acceptance_semantics', case)
        self.assertEqual(defaults, original)
        case = independent.merge_output_case({'draft_tokens': 7,
            'effective_tokens_per_round': 5.2}, {'acceptance': .6})
        self.assertNotIn('effective_tokens_per_round', case)
        self.assertEqual(independent.configured_effective_output(case)[0], 4.8)
        case = independent.merge_output_case(defaults, {'acceptance': .65})
        self.assertNotIn('acceptance_semantics', case)
        self.assertEqual(independent.configured_effective_output(case)[0], 5.2)
        case = independent.merge_output_case({'draft_tokens': 7, 'acceptance': .65,
            'synthetic_acceptance_length': 5.2, 'effective_tokens_per_round': 5.2},
            {'num_speculative_tokens': 9})
        self.assertNotIn('effective_tokens_per_round', case)
        self.assertNotIn('synthetic_acceptance_length', case)
        self.assertEqual(independent.configured_effective_output(case)[0], 6.5)
        with self.assertRaisesRegex(ValueError, 'Legacy'):
            independent.configured_effective_output(independent.merge_output_case(defaults,
                {'synthetic_acceptance_length': 5.2, 'acceptance_semantics': 'mean_accepted_draft_fraction'}))


if __name__ == '__main__':
    unittest.main()
