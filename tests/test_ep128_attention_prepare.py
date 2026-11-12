"""Business-stage identification must stay conservative and preserve overlap."""

import copy
from types import SimpleNamespace
import unittest

from ep128_attention_prepare import analyze_attention_prepare


def task(name, start, end, *, inputs="", outputs="", stream="22", **extra):
    return dict(name=name, op_type=name, start_ns=start, end_ns=end,
                input_shapes=inputs, output_shapes=outputs, stream_id=stream, **extra)


def rows(result):
    return {row["key"]: row for row in result["rows"]}


MODEL = dict(hidden_size=7168, q_lora_rank=1536, head_dim=512,
             local_num_attention_heads=8, index_head_dim=128, local_index_heads=16)


class AttentionPrepareTests(unittest.TestCase):
    def test_clip_parallel_records_and_preserve_wall_time(self):
        result = analyze_attention_prepare([
            task("HcPre", 0, 20),
            task("main.wq_a", 15, 70),
            task("main.wkv", 30, 110, stream="99"),
        ], 10, 90, attention="SWA")
        group = rows(result)
        self.assertEqual(result["parent_wall_ns"], 80)
        self.assertEqual(result["recorded_busy_ns"], 80)
        self.assertEqual(result["uncovered_ns"], 0)
        self.assertEqual(result["kernel_sum_ns"], 125)
        self.assertEqual(result["cross_group_overlap_ns"], 45)
        self.assertEqual(group["hc_pre"]["kernel_sum_ns"], 10)
        self.assertEqual(group["window_kv_projection"]["busy_union_ns"], 60)
        self.assertEqual(result["assigned_kernel_count"], 3)
        self.assertNotIn("kernels", result)
        self.assertTrue(all("kernels" not in row for row in result["rows"]))

    def test_group_sum_union_and_span_have_different_meanings(self):
        result = analyze_attention_prepare([
            task("main.q_norm", 0, 30), task("main.wq_b", 20, 50),
            task("main.q_b_quant", 70, 90),
        ], 0, 100)
        group = rows(result)["q_expand"]
        self.assertEqual(group["kernel_sum_ns"], 80)
        self.assertEqual(group["busy_union_ns"], 70)
        self.assertEqual(group["span_ns"], 90)
        self.assertEqual(result["uncovered_ns"], 30)

    def test_generic_matmul_quant_scatter_not_assigned_by_order_or_stream(self):
        source = [task("MatMul", 0, 10, stream="22"),
                  task("DynamicQuant", 10, 20, stream="21"),
                  task("KvCompressScatter", 20, 30, stream="22")]
        result = analyze_attention_prepare(source, 0, 40, attention="CSA")
        self.assertEqual(result["unassigned_kernel_count"], 3)
        self.assertEqual(result["unassigned_kernel_sum_ns"], 30)
        source[0]["stream_id"], source[1]["stream_id"] = "1", "22"
        other = analyze_attention_prepare(list(reversed(source)), 0, 40, attention="CSA")
        self.assertEqual(result["rows"], other["rows"])

    def test_indexer_enclosing_scope_disambiguates_same_leaf_names(self):
        result = analyze_attention_prepare([
            task("QuantMatmul", 0, 10, function="self.indexer.cv_wq_b.matmul"),
            task("Compressor", 10, 20, function="self.indexer.compressor.forward"),
            task("MatMul", 20, 30, function="self.indexer.weights_proj"),
        ], 0, 40, attention="CSA")
        group = rows(result)
        for key in ("indexer_query", "indexer_key_cache", "indexer_select"):
            self.assertEqual(group[key]["kernel_sum_ns"], 10)
        self.assertIsNone(group["q_expand"]["kernel_sum_ns"])

    def test_projection_shapes_use_actual_configured_dimensions(self):
        result = analyze_attention_prepare([
            task("QuantMatmul", 0, 10, inputs="64,7168;7168,1536", outputs="64,1536"),
            task("QuantMatmul", 10, 20, inputs="64,7168;512,7168", outputs="64,512"),
            task("QuantMatmul", 20, 30, inputs="64,1536;1536,4096", outputs="64,4096"),
            task("QuantMatmul", 30, 40, inputs="64,1536;1536,2048", outputs="64,2048"),
        ], 0, 50, model=MODEL, attention="CSA")
        self.assertEqual(result["assigned_kernel_count"], 4)
        for key in ("q_low_rank", "window_kv_projection", "q_expand", "indexer_query"):
            self.assertEqual(rows(result)[key]["kernel_sum_ns"], 10)

    def test_unconfigured_or_packed_projection_is_unknown(self):
        source = [task("QuantMatmul", 0, 10, inputs="64,7168;7168,1536", outputs="64,1536")]
        self.assertEqual(analyze_attention_prepare(source, 0, 20)["assigned_kernel_count"], 0)
        source[0]["input_shapes"] = "64,7168;96,448,16,16"
        self.assertEqual(analyze_attention_prepare(source, 0, 20, model=MODEL)["assigned_kernel_count"], 0)

    def test_norm_dimension_collision_remains_unassigned(self):
        source = [task("RmsNorm", 0, 10, inputs="64,512;512", outputs="64,512")]
        result = analyze_attention_prepare(source, 0, 20,
            model=dict(hidden_size=512, q_lora_rank=512, head_dim=512))
        self.assertEqual(result["assigned_kernel_count"], 0)
        self.assertIn("多个业务归属", rows(result)["unassigned"]["reason"])

    def test_norm_and_rope_require_disambiguating_shapes(self):
        result = analyze_attention_prepare([
            task("RmsNorm", 0, 10, inputs="64,7168;7168"),
            task("RmsNormDynamicQuant", 10, 20, inputs="64,1536;1536"),
            task("RmsNorm", 20, 30, inputs="64,512;512"),
            task("InplacePartialRotaryMul", 30, 40, inputs="64,1,8,512;64,64;64,64"),
            task("InplacePartialRotaryMul", 40, 50, inputs="64,1,1,512;64,64;64,64"),
            task("InplacePartialRotaryMul", 50, 60, inputs="64,16,128;64,64;64,64"),
        ], 0, 70, model=MODEL, attention="CSA")
        for key in ("input_norm", "q_expand", "window_kv_norm", "q_final", "window_kv_rope", "indexer_query"):
            self.assertEqual(rows(result)[key]["kernel_sum_ns"], 10)

    def test_unique_hca_compressor_but_not_multiple_csa_compressors(self):
        source = [task("Compressor", 0, 10)]
        result = analyze_attention_prepare(source, 0, 30, attention="HCA")
        self.assertEqual(rows(result)["attention_compress"]["kernel_sum_ns"], 10)
        source.append(task("Compressor", 10, 20))
        result = analyze_attention_prepare(source, 0, 30, attention="CSA")
        self.assertEqual(result["unassigned_kernel_count"], 2)
        self.assertIsNone(rows(result)["attention_compress"]["kernel_sum_ns"])

    def test_csa_compressor_actual_norm_weight_shape_separates_branches(self):
        source = [task("Compressor", 0, 10, inputs="64,7168;7168,1024;7168,1024;1,2048;4,1024;512"),
                  task("Compressor", 10, 20, inputs="64,7168;7168,256;7168,256;1,512;4,256;128")]
        result = analyze_attention_prepare(source, 0, 30, model=MODEL, attention="CSA")
        self.assertEqual(rows(result)["attention_compress"]["kernel_sum_ns"], 10)
        self.assertEqual(rows(result)["indexer_key_cache"]["kernel_sum_ns"], 10)

    def test_fused_indexer_records_are_not_split_into_fake_components(self):
        result = analyze_attention_prepare([
            task("IndexerQuantScatter", 0, 12),
            task("QuantLightningIndexerV2", 12, 35),
        ], 0, 40, attention="CSA")
        group = rows(result)
        self.assertEqual(group["indexer_key_cache"]["kernel_sum_ns"], 12)
        self.assertEqual(group["indexer_select"]["kernel_sum_ns"], 23)
        self.assertIsNone(group["indexer_query"]["kernel_sum_ns"])
        self.assertEqual(result["kernel_sum_ns"], 35)

    def test_shared_input_quant_only_explicit_shared_scope(self):
        result = analyze_attention_prepare([
            task("DynamicQuant", 0, 10),
            task("DynamicQuant", 10, 20, function="shared_hs_quant"),
        ], 0, 30, attention="SWA")
        self.assertEqual(rows(result)["input_quant"]["kernel_count"], 1)
        self.assertEqual(result["unassigned_kernel_count"], 1)

    def test_missing_groups_unknown_and_structural_absence_distinct(self):
        result = analyze_attention_prepare([task("HcPre", 0, 10)], 0, 20, attention="SWA")
        group = rows(result)
        self.assertEqual(group["q_low_rank"]["status"], "UNKNOWN")
        self.assertIsNone(group["q_low_rank"]["busy_union_ns"])
        self.assertEqual(group["unassigned"]["status"], "MATCHED")
        self.assertEqual(group["unassigned"]["kernel_count"], 0)
        self.assertEqual(group["unassigned"]["kernel_sum_ns"], 0)
        for key in ("attention_compress", "indexer_query", "indexer_key_cache", "indexer_select"):
            self.assertEqual(group[key]["status"], "NOT_APPLICABLE")
            self.assertEqual(group[key]["busy_union_ns"], 0)
        result = analyze_attention_prepare([task("Compressor", 0, 10)], 0, 20, attention="HCA")
        self.assertEqual(rows(result)["indexer_select"]["status"], "NOT_APPLICABLE")

    def test_structural_contradiction_is_reported_instead_of_discarded(self):
        result = analyze_attention_prepare([task("QuantLightningIndexerV2", 0, 10)], 0, 20, attention="SWA")
        self.assertEqual(result["unassigned_kernel_count"], 1)
        self.assertIn("不一致", rows(result)["unassigned"]["reason"])

    def test_exact_large_timestamps_and_half_open_parent_bounds(self):
        base = 1_793_934_945_924_000_000
        result = analyze_attention_prepare([
            task("HcPre", base, base + 5),
            task("wq_a", base + 5, base + 10),
            task("wq_b", base + 10, base + 20),
            task("wkv", base - 10, base),
        ], str(base), str(base + 10))
        self.assertEqual(result["kernel_count"], 2)
        self.assertEqual(result["kernel_sum_ns"], 10)
        self.assertEqual(result["start_ns"], str(base))
        with self.assertRaises(ValueError):
            analyze_attention_prepare([], float(base), base + 10)
        with self.assertRaises(ValueError):
            analyze_attention_prepare([], 1, 1)

    def test_empty_invalid_records_and_object_input(self):
        result = analyze_attention_prepare([], 0, 100)
        self.assertEqual(result["status"], "UNAVAILABLE")
        self.assertEqual(result["uncovered_ns"], 100)
        source = [SimpleNamespace(**task("HcPreV2", 0, 10)), task("broken", "bad", 12)]
        result = analyze_attention_prepare(source, 0, 100)
        self.assertEqual(result["status"], "PARTIAL")
        self.assertEqual(result["invalid_record_count"], 1)
        self.assertEqual(result["assigned_kernel_count"], 1)

    def test_pure_function_keeps_inputs_and_returns_function_entries(self):
        source = [task("npu_hc_pre_v2", 0, 10)]
        scope = dict(rank="4", role="main", layer=2)
        saved = copy.deepcopy((source, MODEL, scope))
        result = analyze_attention_prepare(source, 0, 20, model=MODEL, attention="CSA", scope=scope)
        self.assertEqual((source, MODEL, scope), saved)
        self.assertIsNot(result["scope"], scope)
        self.assertTrue(result["function_entries"])
        self.assertTrue(all(not entry.startswith("/") for entry in result["function_entries"]))


if __name__ == "__main__":
    unittest.main()
