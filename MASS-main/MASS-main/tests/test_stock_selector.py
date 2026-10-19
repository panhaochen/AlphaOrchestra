import unittest

import pandas as pd

from stock_disagreement.agent.stock_selector import MVEqualStockSelector, RandomStockSelector


class StockSelectorTests(unittest.TestCase):
    def setUp(self):
        self.pool = pd.DataFrame({
            "Date": [20240102] * 5,
            "Stock": ["000001", "000002", "000003", "000004", "000005"],
            "Industry": ["A", "A", "B", "B", "C"],
        })

    def test_random_candidates_are_unique_and_reproducible(self):
        first = RandomStockSelector(4, self.pool, random_seed=42).select_stock_for_llm(20240102)
        second = RandomStockSelector(4, self.pool, random_seed=42).select_stock_for_llm(20240102)
        self.assertEqual(first, second)
        self.assertEqual(len(first), len(set(first)))

    def test_mv_selector_falls_back_when_free_mv_is_unavailable(self):
        selected = MVEqualStockSelector(3, self.pool, random_seed=42).select_stock_for_llm(date=20240102)
        self.assertEqual(len(selected), 3)
        self.assertEqual(len(selected), len(set(selected)))
        self.assertTrue(set(selected).issubset(set(self.pool["Stock"])))


if __name__ == "__main__":
    unittest.main()
