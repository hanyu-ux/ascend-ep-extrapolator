"""HCPre has its own measured span; splitting explanations changes no model cost.

All timelines are synthetic.  The tests exercise observable boundaries,
cross-stream clipping and saved-report compatibility rather than copying the
stage planner's implementation.
"""

import copy
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import profile_ep128 as core
from tests.test_ep128_layer_mean import attach
from tests.test_ep128_method3 import legacy_payload
from tests.test_ep128_source_rank import source_rank_report
from tests.test_ep128_target_stages import native_stage_report


STAGE_KEYS = (
    "attention_hcpre", "attention_prepare", "attention", "attention_output",
    "attention_post", "moe_prepare", "dispatch", "experts", "combine",
    "ffn_output", "ffn_post",
)
PARENT_FIELDS = (
    "status", "selected_rank", "source_rank", "source_cycle_ms", "cycle_ms",
    "tpot_ms", "per_device_tokens_s", "global_tokens_s", "class_means",
    "role_details", "source_partition", "breakdown",
)


def first_layer(report, role="main", attention="HCA"):
    return next(layer for layer in report["cases"][0]["measurements"][0]["rounds"][0]["layers"]
                if layer["role"] == role and layer["attention"] == attention
                and (role != "main" or layer["layer"] != 0))


def layer_tasks(native, layer, rank="dp0"):
    left, right = int(layer["start_ns"]), int(layer["end_ns"])
    return [task for task in native[rank]["tasks"]
            if left <= task.start_ns < right and task.end_ns > left]


def stage_groups(method):
    return {(group["role"], group["attention"]): group
            for group in method["target_stage_breakdowns"]}


def merge_saved_preparation(rows):
    """Represent an old report that measured only the combined preparation."""
    rows = copy.deepcopy(rows)
    first, rest = rows[0], rows[1]
    if first["stage_key"] != "attention_hcpre" or rest["stage_key"] != "attention_prepare":
        raise AssertionError("fixture must supply the new measured HCPre split")
    merged = dict(rest, label="SYNTHETIC legacy HCPre and Q/KV preparation")
    for field in ("source_ms", "delta_ms", "target_ms", "target_total_ms", "cycle_share"):
        if field in first or field in rest:
            merged[field] = first[field] + rest[field]
    output = [merged, *rows[2:]]
    for number, row in enumerate(output, 1):
        row["stage_order"] = number
    return output


def convert_saved_stages_to_legacy(value):
    if isinstance(value, dict):
        rows = value.get("stage_rows")
        if isinstance(rows, list) and tuple(row.get("stage_key") for row in rows) == STAGE_KEYS:
            value["stage_rows"] = merge_saved_preparation(rows)
        for child in value.values():
            convert_saved_stages_to_legacy(child)
    elif isinstance(value, list):
        for child in value:
            convert_saved_stages_to_legacy(child)


class HCPreStageTests(unittest.TestCase):
    def assert_parents_equal(self, actual, expected):
        self.assertEqual([method["id"] for method in actual], [method["id"] for method in expected])
        for current, baseline in zip(actual, expected):
            for field in PARENT_FIELDS:
                self.assertEqual(current.get(field), baseline.get(field), (current["id"], field))

    def test_all_layer_types_partition_hcpre_at_its_actual_end(self):
        report, native = native_stage_report()
        for role, attention in (("main", "CSA"), ("main", "HCA"), ("draft", "SWA")):
            with self.subTest(attention=attention):
                layer = first_layer(report, role, attention)
                tasks = layer_tasks(native, layer)
                left, right = int(layer["start_ns"]), int(layer["end_ns"])
                pre = next(task for task in tasks if task.name == "HcPre" and task.start_ns == left)
                attn = next(task for task in tasks if task.name == "SparseAttnSharedkv")
                rows = core.layer_stage_plan(layer, tasks)
                self.assertEqual(tuple(row["stage_key"] for row in rows), STAGE_KEYS)
                self.assertTrue(all(row["status"] == "MATCHED" for row in rows))
                self.assertEqual((int(rows[0]["start_ns"]), int(rows[0]["end_ns"])), (left, pre.end_ns))
                self.assertEqual((int(rows[1]["start_ns"]), int(rows[1]["end_ns"])),
                                 (pre.end_ns, attn.start_ns))
                self.assertEqual(rows[0]["end_anchor"]["edge"], "end")
                self.assertEqual(rows[0]["end_anchor"]["row_index"], pre.row_index)
                self.assertEqual(rows[1]["start_anchor"], rows[0]["end_anchor"])
                self.assertEqual(int(rows[2]["start_ns"]), attn.start_ns)
                self.assertEqual(int(rows[2]["end_ns"]), attn.end_ns)
                # Hand-specified fixture edges: all nine stages after the old
                # preparation span retain their measured boundaries.
                old_edges = (100_000, 200_000, 300_000, 400_000, 1_000_000,
                             1_500_000, 4_000_000, 4_250_000,
                             right - left - 20_000, right - left)
                self.assertEqual([(int(row["start_ns"]) - left, int(row["end_ns"]) - left)
                                  for row in rows[2:]], list(zip(old_edges, old_edges[1:])))
                self.assertEqual(int(rows[-1]["end_ns"]), right)
                for previous, following in zip(rows, rows[1:]):
                    self.assertEqual(previous["end_ns"], following["start_ns"])
                self.assertEqual(sum(row["duration_ns"] for row in rows), right - left)
                self.assertTrue(all(row["duration_ns"] >= 0 for row in rows))

    def test_nanosecond_hcpre_boundary_is_exact_at_large_absolute_timestamp(self):
        report, native = native_stage_report()
        layer = copy.deepcopy(first_layer(report))
        tasks = copy.deepcopy(layer_tasks(native, layer))
        original_left = int(layer["start_ns"])
        absolute = 1_793_934_945_924_000_000
        shift = absolute - original_left
        layer["start_ns"] = str(absolute)
        layer["end_ns"] = str(int(layer["end_ns"]) + shift)
        for task in tasks:
            task.start_ns += shift
            task.end_ns += shift
        pre = next(task for task in tasks if task.name == "HcPre" and task.start_ns == absolute)
        pre.end_ns += 1
        rows = core.layer_stage_plan(layer, tasks)
        self.assertEqual(rows[0]["duration_ns"], 20_001)
        self.assertEqual(rows[0]["end_ns"], str(absolute + 20_001))
        self.assertEqual(rows[1]["start_ns"], str(absolute + 20_001))
        self.assertEqual(rows[1]["duration_ns"], 79_999)

    def test_cross_stream_calls_are_clipped_once_without_adding_wall_time(self):
        report, native = native_stage_report()
        layer = copy.deepcopy(first_layer(report))
        left, right = int(layer["start_ns"]), int(layer["end_ns"])
        tasks = copy.deepcopy(layer_tasks(native, layer))
        for task in tasks:
            task.wait_ns = 0
        measured = {"rounds": [{"round_id": "SYNTHETIC_hcpre_clip", "layers": [layer]}]}
        capture = dict(rank="dp0", path="SYNTHETIC_ONLY.csv", tasks=tasks)
        with tempfile.TemporaryDirectory(prefix="synthetic-hcpre-") as directory:
            baseline = core.layer_analyze_capture(capture, measured, Path(directory) / "before")
            # Starts before the layer, crosses HCPre.end, starts exactly there,
            # or ends exactly there.  All execute on another stream.
            for number, (a, b) in enumerate(((-5_000, 5_000), (15_000, 25_000),
                                            (20_000, 30_000), (10_000, 20_000)), 100_000):
                tasks.append(SimpleNamespace(name="SYNTHETIC_AUX_" + str(number), op_type="TensorMove",
                    label="SYNTHETIC_AUX", start_ns=left + a, end_ns=left + b,
                    row_index=number, stream_id="auxiliary99", wait_ns=0))
            tasks.sort(key=lambda task: task.start_ns)
            actual = core.layer_analyze_capture(capture, measured, Path(directory) / "after")
        before = {row["stage_key"]: row for row in baseline["stage_summary"]}
        after = {row["stage_key"]: row for row in actual["stage_summary"]}
        self.assertEqual(set(after), set(STAGE_KEYS))
        self.assertAlmostEqual(after["attention_hcpre"]["kernel_sum_ms_mean"]
                               - before["attention_hcpre"]["kernel_sum_ms_mean"], .020)
        self.assertAlmostEqual(after["attention_prepare"]["kernel_sum_ms_mean"]
                               - before["attention_prepare"]["kernel_sum_ms_mean"], .015)
        self.assertAlmostEqual(sum(row["kernel_sum_ms_mean"] for row in after.values())
                               - sum(row["kernel_sum_ms_mean"] for row in before.values()), .035)
        self.assertAlmostEqual(sum(row["duration_ms_mean"] for row in after.values()), (right - left) / 1e6)
        self.assertEqual(actual["layer_summary"][0]["wall_ms_mean"], baseline["layer_summary"][0]["wall_ms_mean"])
        self.assertAlmostEqual(sum(row["recorded_busy_ms_mean"] + row["no_recorded_ms_mean"]
                                   for row in after.values()), (right - left) / 1e6)

    def test_zero_length_remaining_preparation_is_valid(self):
        # An empty half-open window contains no part of a crossing task, even
        # for call/wait diagnostics; it must not manufacture a nonzero count.
        boundary = 1000000
        crossing = SimpleNamespace(start_ns=boundary - 10, end_ns=boundary + 10,
            row_index=999, name="SYNTHETIC_CROSSING", op_type="Copy", wait_ns=1000)
        metrics = core.diagnostic_metrics([crossing], boundary, boundary)
        for field in ("recorded_busy_ms", "no_recorded_ms", "kernel_sum_ms", "task_count", "task_wait_sum_ms"):
            self.assertEqual(metrics[field], 0, field)
        report, native = native_stage_report()
        baseline = attach(copy.deepcopy(report), copy.deepcopy(native))
        for measured in report["cases"][0]["measurements"]:
            for cycle in measured["rounds"]:
                for layer in cycle["layers"]:
                    tasks = layer_tasks(native, layer, measured["rank"])
                    left = int(layer["start_ns"])
                    pre = next(task for task in tasks if task.name == "HcPre" and task.start_ns == left)
                    attention = next(task for task in tasks if task.name == "SparseAttnSharedkv")
                    pre.end_ns = attention.start_ns
                    rows = core.layer_stage_plan(layer, tasks)
                    self.assertEqual(rows[1]["duration_ns"], 0)
                    self.assertEqual(rows[1]["status"], "MATCHED")
        methods = attach(report, native)
        self.assert_parents_equal(methods, baseline)
        for method in (method for method in methods if method["status"] == core.LAYER_MEAN_STATUS):
            for group in method["target_stage_breakdowns"]:
                self.assertEqual(group["status"], "MATCHED", group["reason"])
                self.assertEqual(group["stage_rows"][1]["source_ms"], 0)
                self.assertEqual(group["stage_rows"][1]["target_ms"], 0)

    def test_missing_or_overlapping_anchors_disable_only_explanations(self):
        for corruption in ("missing_attention", "overlapping_hcpre"):
            with self.subTest(corruption=corruption):
                report, native = native_stage_report()
                baseline = attach(copy.deepcopy(report), copy.deepcopy(native))
                layer = first_layer(report)
                tasks = layer_tasks(native, layer)
                attention = next(task for task in tasks if task.name == "SparseAttnSharedkv")
                if corruption == "missing_attention":
                    native["dp0"]["tasks"].remove(attention)
                else:
                    pre = next(task for task in tasks if task.name == "HcPre"
                               and task.start_ns == int(layer["start_ns"]))
                    pre.end_ns = attention.start_ns + 1
                rows = core.layer_stage_plan(layer, layer_tasks(native, layer))
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["status"], "UNRESOLVED")
                self.assertEqual(rows[0]["duration_ns"], int(layer["end_ns"]) - int(layer["start_ns"]))
                methods = attach(report, native)
                self.assert_parents_equal(methods, baseline)
                for method in (method for method in methods if method["status"] == core.LAYER_MEAN_STATUS):
                    group = stage_groups(method)["main", "HCA"]
                    self.assertEqual(group["status"], "UNAVAILABLE")
                    self.assertEqual(group["stage_rows"], [])
                    self.assertTrue(group["reason"])
                    self.assertEqual(stage_groups(method)["main", "CSA"]["status"], "MATCHED")

    def test_all_methods_keep_parent_arithmetic_and_shared_rank(self):
        report, native = source_rank_report()
        methods = attach(copy.deepcopy(report), copy.deepcopy(native))
        unavailable = dict(layer=-1, status="UNAVAILABLE", reason="SYNTHETIC disabled explanation", stage_rows=[])
        with mock.patch.object(core, "_lm_source_layer_stages", return_value=unavailable):
            without_stages = attach(copy.deepcopy(report), copy.deepcopy(native))
        self.assert_parents_equal(methods, without_stages)
        self.assertEqual([method["selected_rank"] for method in methods], ["dp0"] * 3)
        for method in methods:
            self.assertEqual(method["status"], core.LAYER_MEAN_STATUS, method["reasons"])
            for group in method["target_stage_breakdowns"]:
                self.assertEqual(group["status"], "MATCHED", group["reason"])
                self.assertEqual(tuple(row["stage_key"] for row in group["stage_rows"]), STAGE_KEYS)
                for row in group["stage_rows"][:2]:
                    self.assertEqual(row["delta_ms"], 0)
                    self.assertEqual(row["source_ms"], row["target_ms"])
                self.assertAlmostEqual(math.fsum(row["source_ms"] for row in group["stage_rows"]), group["source_wall_ms"])
                self.assertAlmostEqual(math.fsum(row["target_total_ms"] for row in group["stage_rows"]), group["target_total_ms"])
                auxiliary = next(item for item in method["attention_prepare_breakdowns"]
                                 if (item["role"], item["attention"]) == (group["role"], group["attention"]))
                self.assertAlmostEqual(auxiliary["parent_wall_ms"],
                                       sum(row["source_ms"] for row in group["stage_rows"][:2]))

    def test_mixed_saved_stage_schemas_are_not_averaged_or_guessed(self):
        report, native = native_stage_report()
        baseline = attach(copy.deepcopy(report), copy.deepcopy(native))
        original = core._lm_source_layer_stages

        def mixed(layer, *args, **kwargs):
            result = original(layer, *args, **kwargs)
            if layer["role"] == "main" and layer["layer"] == 1:
                result["stage_rows"] = merge_saved_preparation(result["stage_rows"])
            return result

        with mock.patch.object(core, "_lm_source_layer_stages", side_effect=mixed):
            methods = attach(report, native)
        self.assert_parents_equal(methods, baseline)
        for method in (method for method in methods if method["status"] == core.LAYER_MEAN_STATUS):
            group = stage_groups(method)["main", "HCA"]
            self.assertEqual(group["status"], "UNAVAILABLE")
            self.assertEqual(group["stage_rows"], [])
            self.assertIn("schemas disagree", group["reason"])
            self.assertEqual(stage_groups(method)["main", "CSA"]["status"], "MATCHED")

    def test_legacy_ten_stage_json_keeps_unsplit_preparation_in_offline_method3(self):
        report, native = source_rank_report()
        fresh_methods = attach(report, native)
        old = legacy_payload(report)
        convert_saved_stages_to_legacy(old)
        before = copy.deepcopy(old)
        with mock.patch.object(core, "read_capture", side_effect=AssertionError("No CSV reread")):
            derived = core.add_method3_report(old)
        self.assertEqual(old, before)
        methods = derived["cases"][0]["projection"]["layer_mean_methods"]
        self.assert_parents_equal(methods, fresh_methods)
        for method in methods:
            for key, group in stage_groups(method).items():
                self.assertEqual(group["status"], "MATCHED", group["reason"])
                self.assertEqual(tuple(row["stage_key"] for row in group["stage_rows"]), STAGE_KEYS[1:])
                source = stage_groups(next(item for item in fresh_methods if item["id"] == method["id"]))[key]
                self.assertAlmostEqual(group["stage_rows"][0]["source_ms"],
                                       sum(row["source_ms"] for row in source["stage_rows"][:2]))
                self.assertAlmostEqual(math.fsum(row["target_ms"] for row in group["stage_rows"]), group["target_wall_ms"])
                self.assertNotIn("attention_hcpre", {row["stage_key"] for row in group["stage_rows"]})


if __name__ == "__main__":
    unittest.main()
