import unittest

import pandas as pd

from scripts.backtest_benchmark_enhanced import (
    combine_benchmark_and_spread,
    combine_benchmark_and_top_excess,
)


class BenchmarkEnhancedTests(unittest.TestCase):
    def test_zero_active_exposure_equals_benchmark(self):
        top = pd.Series([0.02, -0.01])
        bottom = pd.Series([-0.01, 0.01])
        benchmark = pd.Series([0.005, 0.002])
        result = combine_benchmark_and_spread(top, bottom, benchmark, 0.0)
        pd.testing.assert_series_equal(result, benchmark)

    def test_twenty_percent_gross_uses_ten_percent_per_leg(self):
        top = pd.Series([0.03])
        bottom = pd.Series([-0.01])
        benchmark = pd.Series([0.005])
        result = combine_benchmark_and_spread(top, bottom, benchmark, 0.20)
        self.assertAlmostEqual(float(result.iloc[0]), 0.009)

    def test_invalid_exposure_is_rejected(self):
        with self.assertRaises(ValueError):
            combine_benchmark_and_spread(pd.Series([0.0]), pd.Series([0.0]), pd.Series([0.0]), 1.1)

    def test_top_excess_overlay_only_acts_when_gate_is_open(self):
        top = pd.Series([0.03, 0.03])
        benchmark = pd.Series([0.01, 0.01])
        gate = pd.Series([0.0, 1.0])
        result = combine_benchmark_and_top_excess(top, benchmark, 0.20, gate)
        self.assertAlmostEqual(float(result.iloc[0]), 0.01)
        self.assertAlmostEqual(float(result.iloc[1]), 0.014)


if __name__ == "__main__":
    unittest.main()
