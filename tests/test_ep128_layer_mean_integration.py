"""Independent layer-mean arithmetic and offline CLI integration.

All records built here are synthetic arithmetic fixtures, including packed FP4
shapes. They are not A5 measurements or target accuracy evidence.
"""
import copy
import csv
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from zipfile import ZipFile

import profile_ep128 as core

HERE = Path(__file__).resolve().parents[1]


def synthetic_report(directory, depth=61, target_k=24):
    source_ratios = [128, 128, 4, 128, 4, 128, 4, 128]
    source = dict(hidden_size=1024, moe_intermediate_size=512, n_routed_experts=28,
                  num_experts_per_tok=16, num_hidden_layers=8, num_hash_layers=3,
                  compress_ratios=source_ratios + [0, 0, 0], expert_dtype="fp4",
                  torch_dtype="bfloat16")
    ratios = source_ratios + [4 if i % 2 == 0 else 128 for i in range(8, depth)]
    target = {"main": {**source, "num_hidden_layers": depth, "num_experts_per_tok": target_k,
                        "compress_ratios": ratios},
              "draft": {**source, "num_hidden_layers": 3, "num_hash_layers": 0,
                         "num_experts_per_tok": target_k, "compress_ratios": [0, 0, 0]}}
    config = dict(bs=4, source_dp=2, source_devices=1, source_chips_per_card=2,
                  target_dp=2, target_devices=1, target_chips_per_card=2,
                  base_ep=2, target_ep=2, draft_tokens=7, draft_layers=3,
                  draft_passes=1, effective_tokens_per_round=5.2,
                  ranks_per_node=2, traffic_scope="intranode", data_fraction=1,
                  extra_us=14, expected_ranks=2, expected_rank_names=["dp0", "dp1"],
                  _rank_coverage_complete=True, with_stack=False, trim_rounds=0,
                  token_basis="configured synthetic Neff5.2 including bonus; not measured output")
    report = dict(schema_version=2, generated_at="synthetic arithmetic only",
                  communication_reference={"rows": []}, failures=[], cases=[])
    case = dict(name="SYNTHETIC_PACKED_FP4_NOT_HARDWARE", source_model=source,
                target_model=target, config=config, captures=[], measurements=[],
                projection={"methods": [{"id": "method1", "name": "strict", "status": "PENDING",
                                          "reasons": ["synthetic: no ownership attestation"]}]},
                review={"reasons": ["SYNTHETIC_ONLY: not hardware evidence"], "findings": []},
                source={"status": "SOURCE_CONDITIONAL_RATE", "aggregate": {},
                        "per_rank": [], "class_stats": []})
    fields = ["Name", "Type", "Start Time(us)", "Duration(us)", "Stream ID",
              "Input Shapes", "Output Shapes", "Input Data Types"]
    scales = {"dp0": [1., 3.], "dp1": [4., .5]}
    walls = [100., 10., 5., 20., 15., 30., 25., 40.]
    main_gaps = [9., 1., 2., 3., 4., 5., 6.]
    for rank_index, (rank, factors) in enumerate(scales.items()):
        native, cycles = [], []
        next_start = 1790000000000000000 + rank_index * 10000000000000
        # A third native cycle closes the two retained adjacent-HC cycles.
        for round_index, factor in enumerate(factors + [1.]):
            start = next_start
            cursor, layers = start, []
            for role in ("main", "draft"):
                role_walls = walls if role == "main" else [2., 4., 6.]
                for index, wall in enumerate(role_walls):
                    attention = ("HCA" if source_ratios[index] == 128 else "CSA") if role == "main" else "SWA"
                    router = "hash" if role == "main" and index < 3 else "normal"
                    end = cursor + round(wall * factor * 1e6)
                    layers.append(dict(role=role, layer=index, attention=attention, router=router,
                        class_key="_".join((role, attention, router)), start_ns=str(cursor), end_ns=str(end),
                        wall_ms=wall * factor, scope="HC_marker_interval", clean=False))
                    # Every shape is synthetic. The independent GMM law includes
                    # Gate+Down .20ms only, never the separate SwiGLU .07ms.
                    for name, offset, duration, inputs, outputs, dtype in (
                        ("HcPre", 0., .01, "", "", ""),
                        ("HcPost", .012, .003, "", "", ""),
                        ("HcPre", .02, .01, "", "", ""),
                        ("MoeDistributeDispatchV2", .05, .04, "32,1024;32,16", "", ""),
                        ("aclnnGroupedMatmulGateUp", .12, .08, "512,1024;14,1024,512", "512,1024", "DT_BF16;DT_FLOAT4"),
                        ("SwiGlu", .22, .07, "512,1024", "512,512", "DT_BF16"),
                        ("aclnnGroupedMatmulDown", .32, .12, "512,512;14,512,512", "512,1024", "DT_BF16;DT_FLOAT4"),
                        ("MoeDistributeCombineV2", .60, .06, "512,1024;32,16", "", ""),
                        ("HcPost", wall - .01, .01, "", "", "")):
                        stamp = cursor + round(offset * factor * 1e6)
                        native.append({"Name": name, "Type": name, "Start Time(us)": str(stamp // 1000) + "." + f"{stamp % 1000:03d}",
                            "Duration(us)": format(duration * factor * 1000, ".3f"), "Stream ID": "1",
                            "Input Shapes": inputs, "Output Shapes": outputs, "Input Data Types": dtype})
                    if attention == "CSA":
                        stamp = cursor + round(.011 * factor * 1e6)
                        native.append({"Name": "LightningIndexer", "Type": "LightningIndexer",
                            "Start Time(us)": str(stamp // 1000) + "." + f"{stamp % 1000:03d}",
                            "Duration(us)": format(.0005 * factor * 1000, ".3f"), "Stream ID": "1",
                            "Input Shapes": "", "Output Shapes": "", "Input Data Types": ""})
                    cursor = end
                    if index + 1 < len(role_walls):
                        gap = main_gaps[index] if role == "main" else [7., 11.][index]
                        cursor += round(gap * factor * 1e6)
                if role == "main":
                    cursor += round(13 * factor * 1e6)
            cursor += round(17 * factor * 1e6)
            if round_index < len(factors):
                cycles.append(dict(round_id=f"r{round_index}", start_ns=str(start), end_ns=str(cursor),
                    duration_ms=(cursor - start) / 1e6, layers=layers, scope="synthetic HC cycle",
                    output_tokens=4 * 5.2, token_basis="configured synthetic only", consecutive_decode=False))
            next_start = cursor
        path = directory / (rank + ".csv")
        with path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(native)
        case["captures"].append(dict(rank=rank, path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        case["measurements"].append(dict(rank=rank, rounds=cycles))
        mean = statistics.fmean(cycle["duration_ms"] for cycle in cycles)
        case["source"]["per_rank"].append(dict(rank=rank, rounds=2, cycle_ms_mean=mean,
            conditional_per_rank_tokens_s=20800 / mean, conditional_per_device_tokens_s=41600 / mean,
            scope="synthetic arithmetic", token_basis="configured synthetic"))
    selected = max(case["source"]["per_rank"], key=lambda row: row["cycle_ms_mean"])
    case["source"]["aggregate"] = dict(selected_rank=selected["rank"], cycle_ms=selected["cycle_ms_mean"],
        per_device_tokens_s=41600 / selected["cycle_ms_mean"], global_tokens_s=41600 / selected["cycle_ms_mean"])
    report["cases"].append(case)
    return report, scales


def expected_round(factor, depth=61, target_k=24):
    hca = 5 + sum(i % 2 == 1 for i in range(8, depth))
    csa = 3 + sum(i % 2 == 0 for i in range(8, depth))
    # Independent reference formula from declared synthetic wall/task times.
    # main0=100 is replaced with mean(main1,3,5,7)=25; CSA mean=15.
    communication = (.04 + .06) * (target_k / 16 - 1)
    routed_pair = (.08 + .12) * (target_k / 16 - 1)
    per_layer_delta = communication + routed_pair
    main = hca * (25 + per_layer_delta) + csa * (15 + per_layer_delta)
    main += sum([9, 1, 2, 3, 4, 5, 6]) + (depth - 8) * statistics.fmean([1, 2, 3, 4, 5, 6])
    draft = 3 * (4 + per_layer_delta) + 7 + 11
    # Every target layer has one D and one C. The extra 14us is absolute,
    # outside the synthetic source-duration scale and outside the multiplier.
    return factor * (main + draft + 13 + 17) + (depth + 3) * 2 * .014


class LayerMeanIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="layer-mean-synthetic-")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)

    def attach(self, report):
        core.attach_layer_mean_report(report)
        return report["cases"][0]["projection"]["layer_mean_methods"][0]

    def test_packed_fp4_pair_excludes_swiglu_and_61_depth_preserves_gap_and_u_once(self):
        report, scales = synthetic_report(self.directory)
        method = self.attach(report)
        self.assertEqual(method["status"], "LAYER_MEAN_ESTIMATE", method.get("reasons"))
        self.assertEqual(method["selected_rank"], "dp1")
        for rank in method["per_rank"]:
            expected = [expected_round(factor) for factor in scales[rank["rank"]]]
            self.assertEqual(len(rank["rounds"]), 2)
            for index, row in enumerate(rank["rounds"]):
                self.assertAlmostEqual(row["cycle_ms"], expected[index], places=8)
                factor = scales[rank["rank"]][index]
                for layer in row["class_means"]:
                    self.assertAlmostEqual(layer["dispatch_target_ms"], .04 * factor * 1.5 + .014)
                    self.assertAlmostEqual(layer["combine_target_ms"], .06 * factor * 1.5 + .014)
                    self.assertAlmostEqual(layer["routed_compute_target_ms"], .20 * factor * 1.5)
        # Max of rank means, not mean of each round's slowest rank.
        expected_mean = max(statistics.fmean(expected_round(v) for v in values) for values in scales.values())
        wrong_order = statistics.fmean(max(expected_round(scales[r][i]) for r in scales) for i in (0, 1))
        self.assertNotAlmostEqual(expected_mean, wrong_order)
        self.assertAlmostEqual(method["cycle_ms"], expected_mean, places=8)
        self.assertAlmostEqual(method["per_device_tokens_s"], 41600 / expected_mean, places=8)
        self.assertAlmostEqual(method["global_tokens_s"], method["per_device_tokens_s"], places=8)
        self.assertAlmostEqual(method["tpot_ms"], expected_mean / 5.2, places=8)
        self.assertEqual(report["cases"][0]["projection"]["methods"][0]["status"], "PENDING")

    def test_unchanged_depth_and_topk_still_replaces_main0_not_exact_source_identity(self):
        report, scales = synthetic_report(self.directory, depth=8, target_k=16)
        method = self.attach(report)
        self.assertEqual(method["status"], "LAYER_MEAN_ESTIMATE", method.get("reasons"))
        self.assertAlmostEqual(method["cycle_ms"], 260 * 2.25 + 11 * 2 * .014, places=8)
        self.assertNotAlmostEqual(method["cycle_ms"], report["cases"][0]["source"]["aggregate"]["cycle_ms"])

    def test_plain_cases_cli_predicts_and_removed_modes_are_unavailable(self):
        report, _ = synthetic_report(self.directory, depth=12, target_k=24)
        original = report["cases"][0]
        # Legacy fields must not silently disable the user's fixed 14us.
        original["config"].update(extra_us=0, data_fraction=0)
        configuration = dict(schema_version=2, sample_model=original["source_model"], target_model=original["target_model"],
            defaults=original["config"], cases=[{"name": original["name"], "bs": 4, "draft_tokens": 7,
                "inputs": [{"rank": row["rank"], "path": row["path"]} for row in original["captures"]]}])
        source = self.directory / "cases.json"
        source.write_text(json.dumps(configuration), encoding="utf-8")
        out = self.directory / "ordinary"
        command = [sys.executable, "-S", str(HERE / "profile_ep128.py")]
        result = subprocess.run([*command, str(source), "--out", str(out), "--excel"],
                                text=True, capture_output=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        data = json.loads((out / "model_architecture_report.json").read_text())
        case = data["cases"][0]
        self.assertEqual(case["projection"]["display_method_set"], "layer_mean_methods")
        method = case["projection"]["layer_mean_methods"][0]
        self.assertEqual(method["status"], "LAYER_MEAN_ESTIMATE", method.get("reasons"))
        self.assertEqual(method["comm_model"]["extra_ms_per_call"], .014)
        self.assertAlmostEqual(method["cycle_ms"], expected_round(2.25, depth=12), places=8)
        self.assertTrue((out / "10T模型推理估算.xlsx").is_file())
        with ZipFile(out / "10T模型推理估算.xlsx") as archive:
            self.assertIsNone(archive.testzip())
            for filename in archive.namelist():
                if filename.endswith(".xml"):
                    ET.fromstring(archive.read(filename))
        help_result = subprocess.run([*command, "--help"], text=True, capture_output=True, timeout=20)
        for flag in ("--source-point", "--assumption-scenarios", "--from-report", "--step-residuals"):
            self.assertNotIn(flag, help_result.stdout)
            rejected = subprocess.run([*command, flag], text=True, capture_output=True, timeout=20)
            self.assertNotEqual(rejected.returncode, 0)


class SWADrilldownTests(unittest.TestCase):
    def fixture(self, directory, mutate=None):
        from test_ep128_v2_measure import fixture, MODEL, CASE
        def changes(rows, records):
            for row in rows:
                if row["Model ID"] == "49" and row["Name"] == "Attention":
                    row.update(Name="aclnnSparseAttnSharedkv_SparseAttnSharedkv", Type="SparseAttnSharedkv")
            if mutate:
                mutate(rows)
        capture, _ = fixture(directory, changes=changes)
        return capture, core.measure_capture(capture, MODEL, CASE)

    def test_swa_spans_close_to_each_layer_and_keep_main_comparisons(self):
        with tempfile.TemporaryDirectory() as temp:
            capture, measured = self.fixture(Path(temp))
            diagnostic = core.swa_analyze_capture(capture, measured, temp)
            self.assertEqual(diagnostic["status"], "MATCHED")
            self.assertEqual(len(diagnostic["layer_summary"]), 11)
            self.assertFalse(diagnostic["layer_summary"][0]["reference_eligible"])
            self.assertEqual(len(diagnostic["stage_summary"]), 30)
            for layer in (0, 1, 2):
                rows = [row for row in diagnostic["stage_summary"] if row["layer"] == layer]
                baseline = next(row for row in diagnostic["layer_summary"] if row["role"] == "draft" and row["layer"] == layer)
                self.assertAlmostEqual(sum(row["duration_ms_mean"] for row in rows), baseline["wall_ms_mean"])
                self.assertTrue(all(row["attention"] == "SWA" for row in rows))
            with Path(diagnostic["stages_csv"]).open() as source:
                rows = list(csv.DictReader(source))
            first = rows[0]
            self.assertIsInstance(json.loads(first["top_gaps"]), list)
            self.assertIn("row_index", json.loads(first["start_anchor"]))

    def test_missing_swa_anchor_is_unknown_and_preserves_whole_layer(self):
        def missing(rows):
            for row in rows:
                if row["Model ID"] == "49" and row["Type"] == "SparseAttnSharedkv":
                    row.update(Name="UnknownAttention", Type="UnknownAttention")
        with tempfile.TemporaryDirectory() as temp:
            capture, measured = self.fixture(Path(temp), missing)
            diagnostic = core.swa_analyze_capture(capture, measured, temp)
            self.assertEqual(diagnostic["status"], "PARTIAL")
            self.assertTrue(all(row["stage_key"] == "unresolved" and row["duration_ms_mean"] is None
                                for row in diagnostic["stage_summary"]))
            self.assertTrue(all(row["wall_ms_mean"] > 0 for row in diagnostic["layer_summary"]))

    def test_absent_wait_column_stays_unknown_without_losing_busy_or_wall(self):
        def missing_wait(rows):
            for row in rows:
                row.pop("Wait Time(us)", None)
        with tempfile.TemporaryDirectory() as temp:
            capture, measured = self.fixture(Path(temp), missing_wait)
            diagnostic = core.swa_analyze_capture(capture, measured, temp)
            self.assertEqual(diagnostic["status"], "MATCHED")
            self.assertTrue(all(row["total_cost_ms_mean"] is None for row in diagnostic["layer_summary"]))
            self.assertTrue(all(row["total_cost_ms_mean"] is None for row in diagnostic["stage_summary"]
                                if row["kernel_sum_ms_mean"] > 0))
            self.assertTrue(all(row["total_cost_ms_mean"] == 0 for row in diagnostic["stage_summary"]
                                if row["kernel_sum_ms_mean"] == 0))
            self.assertTrue(all(row["recorded_busy_ms_mean"] > 0 for row in diagnostic["layer_summary"]))

    def test_one_bad_round_never_becomes_a_zero_or_a_subset_mean(self):
        # Mutate a retained round directly, avoiding any guessed epoch units.
        with tempfile.TemporaryDirectory() as temp:
            capture, measured = self.fixture(Path(temp))
            layer = measured["rounds"][0]["layers"][8]
            task = next(task for task in capture["tasks"] if int(layer["start_ns"]) <= task.start_ns < int(layer["end_ns"])
                        and task.op_type == "SparseAttnSharedkv")
            task.name, task.op_type = "UnknownAttention", "UnknownAttention"
            diagnostic = core.swa_analyze_capture(capture, measured, temp)
            partial = [row for row in diagnostic["stage_summary"] if row["layer"] == 0]
            self.assertTrue(partial)
            self.assertTrue(all(row["duration_ms_mean"] is None for row in partial))
            self.assertTrue(any(row["status"] == "PARTIAL" for row in partial))
            other = [row for row in diagnostic["stage_summary"] if row["layer"] != 0]
            self.assertTrue(all(row["status"] == "MATCHED" for row in other))


class AllLayerDrilldownTests(unittest.TestCase):
    """General diagnostic boundaries are observations, never extra model cost."""

    def fixture(self, directory):
        from test_ep128_v2_measure import fixture, MODEL, CASE

        def changes(rows, records):
            for row, record in zip(rows, records):
                if row['Name'] != 'Attention':
                    continue
                owner = record['owners'][0]
                if owner['role'] == 'draft':
                    kind, duration = 'SparseAttnSharedkv', '3'
                elif MODEL['compress_ratios'][owner['layer']] == 4:
                    kind, duration = 'SparseFlashAttention', '4'
                else:
                    kind, duration = 'MixedQuantSparseFlashMla', '5'
                row.update(Name='aclnn' + kind + '_' + kind, Type=kind, **{'Duration(us)': duration})

        capture, _ = fixture(directory, changes=changes)
        return capture, core.measure_capture(capture, MODEL, CASE)

    def test_generic_diagnostics_close_each_main_and_draft_layer_without_cross_role_means(self):
        with tempfile.TemporaryDirectory() as temp:
            capture, measured = self.fixture(Path(temp))
            before = copy.deepcopy(measured)
            diagnostic = core.layer_analyze_capture(capture, measured, temp)
            self.assertEqual(measured, before)
            self.assertEqual(diagnostic['status'], 'MATCHED')
            self.assertEqual(len(diagnostic['layer_summary']), 11)
            self.assertEqual(len(diagnostic['stage_summary']), 110)
            for parent in diagnostic['layer_summary']:
                stages = [row for row in diagnostic['stage_summary']
                          if (row['role'], row['attention'], row['layer']) ==
                             (parent['role'], parent['attention'], parent['layer'])]
                self.assertEqual([row['stage_order'] for row in stages], list(range(1, 11)))
                self.assertEqual(len({row['stage_key'] for row in stages}), 10)
                self.assertTrue(all(row['status'] == 'MATCHED' for row in stages))
                self.assertTrue(all(row['rounds'] == row['matched_windows'] == len(measured['rounds'])
                                    for row in stages))
                self.assertAlmostEqual(sum(row['duration_ms_mean'] for row in stages), parent['wall_ms_mean'])
                self.assertEqual(parent['reference_eligible'], parent['role'] != 'main' or parent['layer'] != 0)
                attention = next(row for row in stages if row['stage_key'] == 'attention')
                self.assertAlmostEqual(attention['duration_ms_mean'], {'CSA': .004, 'HCA': .005, 'SWA': .003}[parent['attention']])
            with Path(diagnostic['stages_csv']).open() as stream:
                exported = list(csv.DictReader(stream))
            self.assertEqual(len(exported), 110 * len(measured['rounds']))
            self.assertEqual({row['role'] for row in exported}, {'main', 'draft'})
            self.assertEqual({row['attention'] for row in exported}, {'CSA', 'HCA', 'SWA'})
            report = {'cases': [dict(name='SYNTHETIC_ALL_LAYER_DRILLDOWN',
                                     source={}, projection={'methods': []},
                                     measurements=[dict(rank=capture['rank'], layer_diagnostics=diagnostic)])]}
            summary_rows = list(core.iter_report_rows(report, 'layer_stage_summary'))
            self.assertEqual(len(summary_rows), 110)
            self.assertEqual({row['rank'] for row in summary_rows}, {capture['rank']})
            self.assertEqual({row['attention'] for row in summary_rows}, {'CSA', 'HCA', 'SWA'})

    def test_generic_stage_plan_uses_native_attention_and_preserves_continuous_layer_bounds(self):
        with tempfile.TemporaryDirectory() as temp:
            capture, measured = self.fixture(Path(temp))
            cycle = measured['rounds'][0]
            for role, attention in (('main', 'CSA'), ('main', 'HCA'), ('draft', 'SWA')):
                layer = next(row for row in cycle['layers'] if row['role'] == role and row['attention'] == attention)
                stages = core.layer_stage_plan(layer, capture['tasks'])
                self.assertEqual(len(stages), 10)
                self.assertEqual(int(stages[0]['start_ns']), int(layer['start_ns']))
                self.assertEqual(int(stages[-1]['end_ns']), int(layer['end_ns']))
                self.assertTrue(all(left['end_ns'] == right['start_ns'] for left, right in zip(stages, stages[1:])))
                self.assertEqual(sum(row['duration_ns'] for row in stages), int(layer['end_ns']) - int(layer['start_ns']))
                self.assertTrue(all(row['status'] == 'MATCHED' for row in stages))
                self.assertIn('Q/KV', stages[0]['label'])
                if attention in ('CSA', 'HCA'):
                    self.assertIn('压缩', stages[0]['label'])
                if attention == 'CSA':
                    self.assertIn('索引', stages[0]['label'])

    def test_optional_indexer_and_metadata_do_not_replace_or_duplicate_the_native_attention_anchor(self):
        with tempfile.TemporaryDirectory() as temp:
            capture, measured = self.fixture(Path(temp))
            layer = next(row for row in measured['rounds'][0]['layers']
                         if row['role'] == 'main' and row['attention'] == 'CSA')
            tasks = [task for task in capture['tasks'] if task.op_type != 'LightningIndexer']
            attention = next(task for task in tasks
                             if int(layer['start_ns']) <= task.start_ns < int(layer['end_ns'])
                             and task.op_type == 'SparseFlashAttention')
            metadata = copy.deepcopy(attention)
            metadata.name = 'aclnnSparseFlashAttentionMetadata_SparseFlashAttentionMetadata'
            metadata.op_type = 'SparseFlashAttentionMetadata'
            metadata.start_ns, metadata.end_ns = attention.start_ns - 1000, attention.start_ns
            stages = core.layer_stage_plan(layer, [*tasks, metadata])
            self.assertEqual(len(stages), 10)
            self.assertTrue(all(row['status'] == 'MATCHED' for row in stages))
            body = next(row for row in stages if row['stage_key'] == 'attention')
            self.assertEqual(int(body['start_ns']), attention.start_ns)
            self.assertEqual(int(body['end_ns']), attention.end_ns)
            # Two actual attention kernels are ambiguous; preserve the layer,
            # rather than picking whichever one happened to occur first.
            stages = core.layer_stage_plan(layer, [*tasks, copy.deepcopy(attention)])
            self.assertEqual(len(stages), 1)
            self.assertEqual(stages[0]['status'], 'UNRESOLVED')
            self.assertEqual(stages[0]['duration_ns'], int(layer['end_ns']) - int(layer['start_ns']))

    def test_partial_main_layer_preserves_unknown_means_and_other_classes_remain_complete(self):
        with tempfile.TemporaryDirectory() as temp:
            capture, measured = self.fixture(Path(temp))
            layer = next(row for row in measured['rounds'][0]['layers']
                         if row['role'] == 'main' and row['attention'] == 'CSA')
            task = next(task for task in capture['tasks']
                        if int(layer['start_ns']) <= task.start_ns < int(layer['end_ns'])
                        and task.op_type == 'SparseFlashAttention')
            task.name, task.op_type = 'UnknownAttention', 'UnknownAttention'
            diagnostic = core.layer_analyze_capture(capture, measured, temp)
            self.assertEqual(diagnostic['status'], 'PARTIAL')
            partial = [row for row in diagnostic['stage_summary']
                       if row['role'] == 'main' and row['layer'] == layer['layer']]
            self.assertTrue(partial)
            self.assertTrue(all(row['duration_ms_mean'] is None for row in partial))
            self.assertTrue(any(row['status'] == 'PARTIAL' for row in partial))
            other = [row for row in diagnostic['stage_summary']
                     if row['role'] != 'main' or row['layer'] != layer['layer']]
            self.assertEqual(len(other), 100)
            self.assertTrue(all(row['status'] == 'MATCHED' for row in other))
            self.assertTrue(all(row['wall_ms_mean'] > 0 for row in diagnostic['layer_summary']))

    def test_one_pass_compatibility_keeps_legacy_swa_summary_draft_only(self):
        with tempfile.TemporaryDirectory() as temp:
            capture, measured = self.fixture(Path(temp))
            generic, legacy = core.layer_analyze_capture(capture, measured, temp, with_swa_compat=True)
            standalone = core.swa_analyze_capture(capture, measured, Path(temp) / 'legacy')
            self.assertEqual(len(generic['stage_summary']), 110)
            self.assertEqual(len(legacy['stage_summary']), 30)
            self.assertTrue(all(row['role'] == 'draft' and row['attention'] == 'SWA'
                                for row in legacy['stage_summary']))
            self.assertEqual(legacy['stage_summary'], standalone['stage_summary'])
            self.assertEqual(legacy['layer_summary'], standalone['layer_summary'])
            self.assertEqual(len(legacy['layer_summary']), 11)


if __name__ == "__main__":
    unittest.main()
