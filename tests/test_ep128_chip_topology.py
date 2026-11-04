"""Physical-card versus logical-chip accounting for the default layer means."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

import profile_ep128 as p
from tools import validate_ep128_capture as independent
from test_ep128_layer_mean import synthetic_report, attach
from test_ep128_v2_driver import measurements as source_measurements
from test_ep128_v2_measure import fixture as csv_fixture, MODEL as CSV_MODEL


def dual_chip_inputs():
    report, native = synthetic_report({"dp0": [1.], "dp1": [1.]}, target_depth=8)
    case = report["cases"][0]
    case["source_model"]["n_routed_experts"] = 28
    for role in ("main", "draft"):
        case["target_model"][role]["n_routed_experts"] = 28
    case["config"].update(name=case["name"], base_ep=2, target_ep=2, source_dp=2, target_dp=2, expected_ranks=2,
        source_devices=1, target_devices=1, source_chips_per_card=2, target_chips_per_card=2,
        ranks_per_node=2, traffic_scope="intranode", extra_us=14, with_stack=False,
        token_basis="SYNTHETIC configured output including bonus", _rank_coverage_complete=True)
    return report, native


class ChipTopologyTests(unittest.TestCase):
    def test_default_single_chip_budget_rejects_dp_exceeding_physical_cards(self):
        report, _ = dual_chip_inputs()
        config = report["cases"][0]["config"]
        config.pop("source_chips_per_card")
        config.pop("target_chips_per_card")
        with self.assertRaisesRegex(ValueError, "logical NPUs"):
            p.normalize_case({}, config, Path.cwd())
        topology, reasons = p.b_topology(config)
        self.assertEqual(topology["source"]["chips_per_card"], 1)
        self.assertTrue(any("DP不能" in reason for reason in reasons))

    def test_dual_chip_card_preserves_cycle_and_uses_physical_card_denominator(self):
        report, native = dual_chip_inputs()
        case = report["cases"][0]
        config = p.normalize_case({}, case["config"], Path.cwd())
        case["config"] = config
        self.assertEqual(config["source_logical_devices"], 2)
        self.assertEqual(config["target_logical_devices"], 2)
        method = attach(report, native)[0]
        self.assertEqual(method["status"], "LAYER_MEAN_ESTIMATE", method["reasons"])
        # 5 HCA + 3 CSA + 3 SWA; source gaps112 and residual18;
        # each of11 calls adds .5*(D.5+C.25+GMM.4) and two14us terms.
        expected_cycle = 5 * 10 + 3 * 20 + 3 * 5 + 112 + 18 + 11 * (.5 * 1.15 + .028)
        self.assertAlmostEqual(method["cycle_ms"], expected_cycle)
        self.assertAlmostEqual(method["per_device_tokens_s"], 1000 * 4 * 2 * 5.2 / expected_cycle)
        self.assertAlmostEqual(method["global_tokens_s"], method["per_device_tokens_s"])
        baseline_cycle = method["cycle_ms"]
        # The same two logical chips can be two physical cards instead.
        config.update(source_devices=2, target_devices=2, source_chips_per_card=1, target_chips_per_card=1)
        changed = attach(report, native)[0]
        self.assertEqual(changed["cycle_ms"], baseline_cycle)
        self.assertAlmostEqual(changed["per_device_tokens_s"], method["per_device_tokens_s"] / 2)

    def test_positive_integer_chips_required_by_normalization(self):
        report, _ = dual_chip_inputs()
        config = report["cases"][0]["config"]
        for key in ("source_chips_per_card", "target_chips_per_card"):
            for bad in (0, -1, True, 2., "2", None):
                with self.subTest(key=key, bad=bad), self.assertRaisesRegex(ValueError, key):
                    p.normalize_case({}, {**config, key: bad}, Path.cwd())

    def test_dp_is_independent_of_expert_parallel_and_source_rate_uses_physical_cards(self):
        report, _ = dual_chip_inputs()
        config = report["cases"][0]["config"]
        topology, errors = p.b_topology({**config, "source_dp": 1, "target_dp": 1})
        self.assertEqual(errors, [])
        self.assertEqual(topology["source"]["expert_parallel"], 2)
        self.assertEqual(topology["source"]["dp_request_groups"], 1)
        source = p.summarize_source(source_measurements(), config)
        self.assertAlmostEqual(source["aggregate"]["per_device_tokens_s"], 832 * 2)
        config.update(expected_ranks=1, _rank_coverage_complete=True)
        source = p.summarize_source(source_measurements(), config)
        self.assertFalse(source["aggregate"]["rank_coverage_complete"])
        self.assertTrue(any("expected_ranks" in text for text in source["topology_reasons"]))

    def test_intranode_cannot_silently_expand_to_cross_node_ep(self):
        report, native = dual_chip_inputs()
        case = report["cases"][0]
        case["config"].update(target_ep=4, target_devices=2, target_dp=2)
        for role in ("main", "draft"):
            case["target_model"][role]["n_routed_experts"] = 56
        method = attach(report, native)[0]
        self.assertEqual(method["status"], "UNAVAILABLE")
        self.assertTrue(any("intranode" in reason for reason in method["reasons"]))

    def test_independent_validator_counts_two_logical_csvs_as_one_explicit_dual_chip_card(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            inputs = []
            for rank in (0, 1):
                directory = base / f'dp{rank}'
                directory.mkdir()
                def change(rows, records, rank=rank):
                    for row in rows:
                        row['Device_id'] = str(rank)
                capture, _ = csv_fixture(directory, changes=change)
                inputs.append({'rank': f'dp{rank}', 'path': capture['path']})
            manifest = {'sample_model': {**CSV_MODEL, 'n_routed_experts': 28},
                        'defaults': {'bs': 32, 'draft_tokens': 7, 'draft_layers': 3, 'draft_passes': 1,
                            'effective_tokens_per_round': 5.2, 'trim_rounds': 2, 'base_ep': 2,
                            'expected_ranks': 2, 'source_dp': 2, 'source_devices': 1, 'source_chips_per_card': 2},
                        'cases': [{'name': 'SYNTHETIC_TOPOLOGY_ONLY', 'inputs': inputs}]}
            path = base / 'manifest.json'
            path.write_text(json.dumps(manifest))
            with redirect_stdout(io.StringIO()):
                output = independent.validate(path)
            self.assertTrue(output['all_device_accounting_passed'], output['cases'][0]['errors'])
            self.assertEqual(output['cases'][0]['source_topology']['logical_devices'], 2)
            self.assertEqual(output['cases'][0]['source_topology']['physical_cards'], 1)
            manifest['defaults'].pop('source_chips_per_card')
            path.write_text(json.dumps(manifest))
            with redirect_stdout(io.StringIO()):
                rejected = independent.validate(path)
            self.assertFalse(rejected['all_device_accounting_passed'])
            self.assertTrue(any('logical NPU budget' in text for text in rejected['cases'][0]['errors']))


if __name__ == "__main__":
    unittest.main()
