import tempfile
import unittest
from pathlib import Path

import pandas as pd

from stock_disagreement.agent.basic_agent import StockDisagreementAgent
from stock_disagreement.agent.stock_selector import RandomStockSelector
from stock_disagreement.llm_cache import LLMResponseCache
from stock_disagreement.signals import (
    HorizonTarget,
    RollingHorizonAlphaCalibrator,
    add_horizon_signals,
)


class LLMResponseCacheTests(unittest.TestCase):
    def test_cache_key_is_stable_and_round_trips(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = LLMResponseCache(Path(directory), "readwrite")
            payload = {"model": "deepseek-chat", "input": "hello", "temperature": 0.0}
            first_key = cache.key(payload)
            second_key = cache.key({"temperature": 0.0, "input": "hello", "model": "deepseek-chat"})
            self.assertEqual(first_key, second_key)
            self.assertIsNone(cache.get(first_key))
            cache.put(first_key, {
                "text": "response",
                "finish_reason": "stop",
                "metrics": {"total_tokens": 123, "latency_seconds": 0.5},
            })
            cached = cache.get(first_key)
            self.assertEqual(cached["text"], "response")
            self.assertEqual(cached["metrics"]["total_tokens"], 123)
            self.assertEqual(cache.stats()["hits"], 1)

    def test_read_only_cache_rejects_miss(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = LLMResponseCache(Path(directory), "read")
            with self.assertRaises(RuntimeError):
                cache.require_hit("missing")


class DeterministicSelectorTests(unittest.TestCase):
    def test_same_seed_and_date_produce_same_candidates(self):
        pool = pd.DataFrame({
            "Stock": [f"S{index}" for index in range(20)],
            "Date": [20240102] * 20,
        })
        first = RandomStockSelector(10, pool, random_seed=123)
        second = RandomStockSelector(10, pool, random_seed=123)
        third = RandomStockSelector(10, pool, random_seed=456)
        self.assertEqual(
            first.select_stock_for_llm(20240102),
            second.select_stock_for_llm(20240102),
        )
        self.assertNotEqual(
            first.select_stock_for_llm(20240102),
            third.select_stock_for_llm(20240102),
        )


class CausalPromptTests(unittest.TestCase):
    def test_strategy_prompt_does_not_depend_on_future_news_rows(self):
        def prompt(news_titles):
            agent = StockDisagreementAgent.__new__(StockDisagreementAgent)
            agent.description = {"Close": "Closing price."}
            agent.prepare_data = {
                "news": pd.DataFrame({"NewsTitle": news_titles}),
            }
            return agent.generate_strategy_input()[0]

        short_prompt = prompt(["historical headline"])
        extended_prompt = prompt([
            "different ordering",
            "historical headline",
            "future headline",
        ])
        self.assertEqual(short_prompt, extended_prompt)
        self.assertNotIn("future headline", extended_prompt)


class HorizonSignalTests(unittest.TestCase):
    def test_horizon_signals_use_independent_alphas_and_exposure(self):
        frame = pd.DataFrame({
            "Signal_mean": [0.8],
            "Signal_std": [-0.2],
            "ExposureControl": [0.5],
        })
        result = add_horizon_signals(frame, {"1d": 1.0, "5d": 0.0, "10d": 0.25})
        self.assertAlmostEqual(result.loc[0, "Signal_1d"], 0.4)
        self.assertAlmostEqual(result.loc[0, "Signal_5d"], -0.1)
        self.assertAlmostEqual(result.loc[0, "Signal_10d"], 0.025)
        self.assertNotIn("Signal_1d", frame.columns)

    @staticmethod
    def _calibration_frame(reverse_label=False):
        label = [2.0, 1.0, 0.0] if reverse_label else [0.0, 1.0, 2.0]
        return pd.DataFrame({
            "Signal_mean": [0.0, 1.0, 2.0],
            "Signal_std": [0.0, -1.0, -2.0],
            "ExposureControl": [1.0, 1.0, 1.0],
            "label": label,
        })

    def test_rolling_calibrator_uses_only_matured_observations(self):
        def make_calibrator():
            return RollingHorizonAlphaCalibrator(
                [HorizonTarget("1d", "label", maturity_days=2)],
                {"1d": 0.5},
                grid_step=0.5,
                min_observations=3,
                shrinkage=0.0,
            )

        first = make_calibrator()
        second = make_calibrator()
        for date in range(4):
            first.observe(date, self._calibration_frame(reverse_label=False))
            second.observe(date, self._calibration_frame(reverse_label=(date == 3)))
        self.assertEqual(first.select(4)["1d"], 1.0)
        self.assertEqual(second.select(4)["1d"], 1.0)
        self.assertEqual(first.audit_records[-1]["matured_observations"], 3)

    def test_rolling_calibrator_falls_back_before_minimum_history(self):
        calibrator = RollingHorizonAlphaCalibrator(
            [HorizonTarget("1d", "label", maturity_days=2)],
            {"1d": 0.8},
            min_observations=3,
        )
        calibrator.observe(1, self._calibration_frame())
        self.assertEqual(calibrator.select(2)["1d"], 0.8)
        self.assertEqual(calibrator.audit_records[-1]["status"], "fallback")


if __name__ == "__main__":
    unittest.main()
