"""CPU regressions for reported Dispatch V3 input slots; no timing predictions."""
from pathlib import Path
import tempfile
import unittest

import profile_ep128 as p
from test_ep128_v2_measure import CASE, MODEL, fixture


REPORTED_SHAPES = '"4100;32,7168;32,16;;32;;;"'
REPORTED_DTYPES = "INT32;DT_BF16;INT32;DT_UNDEFINED;BOOL;DT_UNDEFINED;DT_UNDEFINED;DT_UNDEFINED"
REPORTED_OUTPUTS = '"7168,7168;7168,224;917504;14;224;1;7168"'


def dispatch_task(shapes=REPORTED_SHAPES, dtypes=REPORTED_DTYPES, *, version=3):
    return p.ATask(
        row_index=89, row_indices=[89], start_ns=1000, end_ns=2000, wait_ns=0,
        name=f"MoeDistributeDispatchV{version}", op_type=f"MoeDistributeDispatchV{version}",
        device_id="0", model_id="47", stream_id="1", task_id="89",
        context_id="3", batch_id="0", subtask_id="0", input_shapes=shapes,
        output_shapes=REPORTED_OUTPUTS, input_dtypes=dtypes, output_dtypes="",
        task_type="AI_CORE", op_state="GRAPH")


class DispatchLayoutTests(unittest.TestCase):
    def assert_unknown(self, task):
        result = p.a_dispatch_inputs(task)
        self.assertEqual(result["layout"], "unknown", result)
        for key in ("x_shape", "expert_ids_shape", "x_index", "expert_ids_index"):
            self.assertIsNone(result[key], result)
        self.assertIsInstance(result["reason"], str)
        self.assertTrue(result["reason"])

    def test_reported_v3_shape_keeps_empty_slots_and_uses_x1_ids2(self):
        self.assertEqual(p.a_dims(REPORTED_SHAPES),
                         [[4100], [32, 7168], [32, 16], [], [32], [], [], []])
        result = p.a_dispatch_inputs(dispatch_task())
        self.assertEqual(result["layout"], "leading_int32_context")
        self.assertEqual(result["x_index"], 1)
        self.assertEqual(result["expert_ids_index"], 2)
        self.assertEqual(result["x_shape"], [32, 7168])
        self.assertEqual(result["expert_ids_shape"], [32, 16])
        self.assertIsNone(result["reason"])
        p.a_validate_source_shapes([dispatch_task()], MODEL)

    def test_outer_quotes_are_optional_and_not_a_different_layout(self):
        quoted = p.a_dispatch_inputs(dispatch_task())
        unquoted = p.a_dispatch_inputs(dispatch_task(REPORTED_SHAPES.strip('"')))
        self.assertEqual(quoted, unquoted)

    def test_reported_v3_with_real_k24_still_rejects_source_k16(self):
        task = dispatch_task(REPORTED_SHAPES.replace("32,16", "32,24"))
        self.assertEqual(p.a_dispatch_inputs(task)["expert_ids_shape"], [32, 24])
        with self.assertRaisesRegex(ValueError, "TopK"):
            p.a_validate_source_shapes([task], MODEL)

    def test_reported_v3_with_wrong_hidden_width_still_rejects(self):
        task = dispatch_task(REPORTED_SHAPES.replace("32,7168", "32,4096"))
        with self.assertRaisesRegex(ValueError, "hidden"):
            p.a_validate_source_shapes([task], MODEL)

    def test_reported_v3_with_mismatched_token_rows_rejects(self):
        task = dispatch_task(REPORTED_SHAPES.replace("32,16", "16,16"))
        with self.assertRaisesRegex(ValueError, "(row|token|dimension|capacity)"):
            p.a_validate_source_shapes([task], MODEL)

    def test_shifted_layout_needs_dtype_proof_in_the_exact_slots(self):
        variants = (
            "",
            REPORTED_DTYPES.replace("INT32;", "FLOAT32;", 1),
            REPORTED_DTYPES.replace("DT_BF16", "INT32"),
            REPORTED_DTYPES.replace("DT_BF16;INT32", "DT_BF16;FLOAT32"),
            "INT32;DT_BF16",
        )
        for dtypes in variants:
            with self.subTest(dtypes=dtypes):
                self.assert_unknown(dispatch_task(dtypes=dtypes))

    def test_unsupported_prefix_shape_and_v2_do_not_get_v3_slot_shift(self):
        self.assert_unknown(dispatch_task(REPORTED_SHAPES.replace("4100;", "4100,1;", 1)))
        self.assert_unknown(dispatch_task(version=2))

    def test_standard_v2_shape_with_or_without_dtype_is_unchanged(self):
        for dtypes in ("", "DT_BF16;INT32"):
            with self.subTest(dtypes=dtypes):
                task = dispatch_task("[32,7168];[32,16]", dtypes, version=2)
                result = p.a_dispatch_inputs(task)
                self.assertEqual(result["layout"], "standard")
                self.assertEqual((result["x_index"], result["expert_ids_index"]), (0, 1))
                self.assertEqual(result["x_shape"], [32, 7168])
                self.assertEqual(result["expert_ids_shape"], [32, 16])
                p.a_validate_source_shapes([task], MODEL)
        with self.assertRaisesRegex(ValueError, "TopK"):
            p.a_validate_source_shapes(
                [dispatch_task("[32,7168];[32,24]", "", version=2)], MODEL)

    def test_unknown_layout_does_not_search_later_tensors_for_width16(self):
        task = dispatch_task("4100;32,7168;32,24;;32,16;;;",
                             "FLOAT32;DT_BF16;FLOAT32;DT_UNDEFINED;INT32;DT_UNDEFINED;DT_UNDEFINED;DT_UNDEFINED")
        self.assert_unknown(task)
        p.a_validate_source_shapes([task], MODEL)

    def test_unknown_source_layout_remains_diagnostic_with_warning(self):
        def unknown_shapes(rows, records):
            for row in rows:
                if "Dispatch" in row["Name"]:
                    row.update({"Name": "MoeDistributeDispatchV3", "Type": "MoeDistributeDispatchV3",
                                "Input Shapes": REPORTED_SHAPES, "Input Data Types": ""})
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            capture, _ = fixture(directory, changes=unknown_shapes)
            result = p.measure_capture(capture, MODEL, CASE, directory)
        self.assertTrue(result["rounds"])
        self.assertTrue(any("DISPATCH_INPUT_LAYOUT_UNVERIFIED" in warning
                            for warning in result["warnings"]), result["warnings"])
        self.assertTrue(all(layer["scope"] == "HC_marker_interval"
                            for item in result["rounds"] for layer in item["layers"]))


if __name__ == "__main__":
    unittest.main()
