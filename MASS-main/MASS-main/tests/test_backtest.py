import unittest

import pandas as pd

from stock_disagreement.backtest import BacktestConfig, run_long_only_backtest


class BacktestTests(unittest.TestCase):
    def setUp(self):
        self.prices = pd.DataFrame({
            "Stock": ["A", "B"] * 3,
            "Date": [20240101, 20240101, 20240102, 20240102, 20240103, 20240103],
            "Open": [10, 10, 10, 10, 11, 10],
            "Close": [10, 10, 11, 10, 11, 10],
            "Value": [1e9] * 6,
        })
        self.signals = pd.DataFrame({
            "Stock": ["A", "B", "A", "B"],
            "Date": [20240101, 20240101, 20240102, 20240102],
            "Signal_1d": [1.0, 0.0, 1.0, 0.0],
        })

    def test_signal_is_executed_at_next_open(self):
        result = run_long_only_backtest(
            self.signals,
            self.prices,
            "Signal_1d",
            config=BacktestConfig(top_n=1, commission_bps=0, slippage_bps=0, sell_stamp_duty_bps=0, impact_bps=0),
        )
        day_one = result.daily[result.daily.Date == 20240101].iloc[0]
        day_two = result.daily[result.daily.Date == 20240102].iloc[0]
        self.assertEqual(day_one["positions"], 0)
        self.assertEqual(day_two["positions"], 1)
        self.assertGreater(day_two["return"], 0)

    def test_costs_reduce_equity_and_are_reported(self):
        free = run_long_only_backtest(
            self.signals,
            self.prices,
            "Signal_1d",
            config=BacktestConfig(top_n=1, commission_bps=0, slippage_bps=0, sell_stamp_duty_bps=0, impact_bps=0),
        )
        costly = run_long_only_backtest(
            self.signals,
            self.prices,
            "Signal_1d",
            config=BacktestConfig(top_n=1, commission_bps=10, slippage_bps=10, sell_stamp_duty_bps=10, impact_bps=0),
        )
        self.assertLess(costly.daily.iloc[-1].equity, free.daily.iloc[-1].equity)
        self.assertGreater(costly.metrics["total_transaction_cost"], 0)

    def test_liquidity_cap_limits_trade_size(self):
        prices = self.prices.copy()
        prices["Value"] = 1000.0
        result = run_long_only_backtest(
            self.signals,
            prices,
            "Signal_1d",
            config=BacktestConfig(initial_capital=1e6, top_n=1, max_participation_rate=0.01, commission_bps=0, slippage_bps=0, sell_stamp_duty_bps=0, impact_bps=0),
        )
        self.assertLessEqual(result.trades["notional"].max(), 10.0)

    def test_industry_balancing_uses_multiple_industries(self):
        signals = pd.DataFrame({
            "Stock": ["A", "B", "C", "D"], "Date": [20240101] * 4,
            "Signal_5d": [4.0, 3.0, 2.0, 1.0], "Industry": ["Bank", "Bank", "Tech", "Tech"],
        })
        prices = pd.DataFrame({
            "Stock": ["A", "B", "C", "D"] * 2,
            "Date": [20240101] * 4 + [20240102] * 4,
            "Open": [10.0] * 8, "Close": [10.0] * 8, "Value": [1e9] * 8,
        })
        result = run_long_only_backtest(
            signals, prices, "Signal_5d",
            config=BacktestConfig(top_n=2, industry_balanced=True, commission_bps=0,
                                  slippage_bps=0, impact_bps=0, sell_stamp_duty_bps=0),
        )
        self.assertEqual(set(result.trades["Stock"]), {"00000A", "00000C"})

    def test_market_exposure_scales_target_capital(self):
        exposure = pd.DataFrame({"Date": [20240101], "TargetExposure": [0.5]})
        result = run_long_only_backtest(
            self.signals, self.prices, "Signal_1d", market_exposure=exposure,
            config=BacktestConfig(top_n=1, max_position_weight=1.0, commission_bps=0, slippage_bps=0,
                                  sell_stamp_duty_bps=0, impact_bps=0),
        )
        first_trade = result.trades.iloc[0]
        self.assertAlmostEqual(first_trade["notional"], 5_000_000.0)

    def test_residual_signal_removes_industry_level(self):
        from stock_disagreement.backtest import _residualize_cross_section
        frame = pd.DataFrame({
            "Signal_5d": [10.0, 10.0, -4.0, -4.0],
            "Industry": ["A", "A", "B", "B"],
            "Value": [100.0] * 4,
        })
        residual = _residualize_cross_section(frame, "Signal_5d")
        means = residual.groupby("Industry")["ResidualSignal"].mean()
        self.assertTrue((means.abs() < 1e-10).all())


if __name__ == "__main__":
    unittest.main()
