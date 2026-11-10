"""Exact trace timestamp semantics for the Decimal parsing fast path."""

from decimal import Decimal, ROUND_DOWN, ROUND_HALF_EVEN, ROUND_UP, localcontext
from pathlib import Path
import tempfile
import unittest

import profile_ep128 as core


def original_ns(value, label="timestamp", missing=False):
    """Pre-optimization contract, including context-dependent integer rounding."""
    if value is None or str(value).strip() in ("", "N/A", "None"):
        if missing:
            return None
        raise ValueError(label + " is missing")
    try:
        number = Decimal(str(value).strip())
        if not number.is_finite():
            raise ValueError(label + " must be finite")
        return int((number * 1000).to_integral_value())
    except (ArithmeticError, ValueError) as error:
        raise ValueError(label + " must be finite microseconds: " + str(value)) from error


class TraceTimestampPerformanceTests(unittest.TestCase):
    def compare(self, value, **kwargs):
        def observed(fn):
            try:
                return ("value", fn(value, **kwargs))
            except Exception as error:
                return ("error", type(error), str(error))

        self.assertEqual(observed(core.a_ns), observed(original_ns))

    def test_decimal_epoch_fraction_negative_and_exponent_stay_exact(self):
        for text in ("1793591950807238.775", "-1793591950807238.775",
                     "0.0005", "-0.0005", "0.0015", "-0.0015",
                     "-0", "1E+35", "1E-100", "12345678901234567890123456789.123"):
            with self.subTest(text=text):
                self.compare(Decimal(text))
        self.assertEqual(core.a_ns(Decimal("1793591950807238.775")), 1793591950807238775)

    def test_context_precision_rounding_and_flags_are_preserved(self):
        for precision in (3, 7, 28, 50):
            for rounding in (ROUND_DOWN, ROUND_HALF_EVEN, ROUND_UP):
                for value in (Decimal("1793591950807238.7755"),
                              Decimal("-0.0005"), Decimal("1E+35")):
                    with self.subTest(precision=precision, rounding=rounding, value=value):
                        with localcontext() as context:
                            context.prec = precision
                            context.rounding = rounding
                            context.clear_flags()
                            expected = original_ns(value)
                            expected_flags = dict(context.flags)
                            context.clear_flags()
                            self.assertEqual(core.a_ns(value), expected)
                            self.assertEqual(dict(context.flags), expected_flags)

    def test_non_decimal_inputs_keep_original_errors_and_missing_behavior(self):
        for missing in (False, True):
            for value in (None, "", " N/A ", "None", "broken", float("nan"),
                          float("inf"), 0, -7, 12345678901234567890123456789,
                          " 1.0005 ", -0.0015, True):
                with self.subTest(value=value, missing=missing):
                    self.compare(value, label="device time", missing=missing)

    def test_integer_inputs_still_follow_decimal_context(self):
        with localcontext() as context:
            context.prec = 3
            context.rounding = ROUND_DOWN
            self.assertEqual(core.a_ns(123456789), 123000000000)
            self.compare(123456789)

    def test_nonfinite_decimals_and_decimal_subclasses_keep_contract(self):
        class TextDecimal(Decimal):
            def __str__(self):
                return " 3.0005 "

        for value in (Decimal("NaN"), Decimal("sNaN"), Decimal("Infinity"),
                      Decimal("-Infinity"), TextDecimal("2")):
            with self.subTest(value=repr(value)):
                self.compare(value, label="duration")

    def test_stream_parser_large_epoch_still_reaches_exact_ns(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace_view.json"
            path.write_text('[{"ph":"X","name":"kernel","ts":1793591950807238.775,"dur":0.0015}]')
            event, = core.residual_iter_trace_events(path, chunk_size=7)
        self.assertIs(type(event["ts"]), Decimal)
        self.assertEqual(core.a_ns(event["ts"]), 1793591950807238775)
        self.assertEqual(core.a_ns(event["dur"]), 2)


if __name__ == "__main__":
    unittest.main()
