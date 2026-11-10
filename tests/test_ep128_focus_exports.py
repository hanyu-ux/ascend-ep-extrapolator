"""Focus exports keep each rank and HC round identity without nested ledgers."""
import unittest

import profile_ep128 as core


class FocusExportTests(unittest.TestCase):
    def report(self):
        measurements = []
        for rank, value in (("dp0", 1.5), ("dp1", 3.0)):
            stage = {"stage_key": "main_inputs", "label": "下一主模型KV与输入准备",
                     "per_step": [{"round_id": "hc-17", "step_index": 2,
                                   "duration_ms": value, "component_durations_ms": {"op-a": value}}],
                     "summary": [{"key": "op-a", "duration_ms_mean": value,
                                  "source_function_mapping": [{"function": "InputBatch.prepare_inputs"}]}]}
            measurements.append({"rank": rank, "step_residuals": {
                "focus_diagnostics": {"rank": rank, "stages": [stage]}}})
        return {"cases": [{"name": "SYNTHETIC_EXPORT", "source": {},
                           "projection": {"layer_mean_methods": []}, "measurements": measurements}]}

    def test_step_export_retains_rank_round_and_parent_and_omits_nested_components(self):
        rows = list(core.iter_report_rows(self.report(), "step_residual_focus_steps"))
        self.assertEqual([(row["rank"], row["round_id"], row["duration_ms"]) for row in rows],
                         [("dp0", "hc-17", 1.5), ("dp1", "hc-17", 3.0)])
        self.assertTrue(all(row["parent_stage_key"] == "main_inputs" for row in rows))
        self.assertTrue(all("components" not in row for row in rows))
        self.assertTrue(all("component_durations_ms" not in row for row in rows))

    def test_summary_export_keeps_function_evidence_and_old_reports_export_empty(self):
        report = self.report()
        rows = list(core.iter_report_rows(report, "step_residual_focus_summary"))
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[1]["source_function_mapping"][0]["function"], "InputBatch.prepare_inputs")
        report["cases"][0]["measurements"] = [{"rank": "dp0", "step_residuals": {"status": "DEVICE_ONLY"}}]
        self.assertEqual(list(core.iter_report_rows(report, "step_residual_focus_steps")), [])
        self.assertEqual(list(core.iter_report_rows(report, "step_residual_focus_summary")), [])


if __name__ == "__main__":
    unittest.main()
