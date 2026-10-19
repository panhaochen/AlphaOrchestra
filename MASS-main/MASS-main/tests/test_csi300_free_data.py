import unittest

import numpy as np
import pandas as pd

from stock_disagreement.data.csi300_free import (
    build_derived_tables,
    cross_sectional_label,
    expand_constituent_snapshots,
    normalize_stock_code,
)


class Csi300FreeDataTests(unittest.TestCase):
    def test_stock_code_normalization_preserves_leading_zeroes(self):
        self.assertEqual(normalize_stock_code("sz.000001"), "000001")
        self.assertEqual(normalize_stock_code("600000.SH"), "600000")
        self.assertEqual(normalize_stock_code("1"), "000001")

    def test_constituents_are_only_forward_filled(self):
        snapshots = pd.DataFrame(
            {
                "SnapshotDate": [20230102, 20230102, 20230201],
                "Stock": ["000001", "600000", "000002"],
            }
        )
        result = expand_constituent_snapshots(
            snapshots, pd.Series([20221230, 20230103, 20230131, 20230201])
        )
        self.assertNotIn(20221230, set(result["Date"]))
        self.assertEqual(set(result[result.Date == 20230131].Stock), {"000001", "600000"})
        self.assertEqual(set(result[result.Date == 20230201].Stock), {"000002"})

    def test_cross_sectional_label_is_winsorized_and_standardized(self):
        result = cross_sectional_label(pd.Series([0.0, 1.0, 2.0, 100.0]))
        self.assertAlmostEqual(float(result.mean()), 0.0, places=12)
        self.assertAlmostEqual(float(result.std()), 1.0, places=12)

    def test_derived_labels_use_future_prices_without_future_membership(self):
        dates = list(range(20230101, 20230126))
        rows = []
        for stock, slope in (("000001", 1.0), ("000002", 0.5), ("600000", -0.25)):
            for offset, date in enumerate(dates):
                price = 10 + slope * offset
                rows.append(
                    {
                        "Stock": stock,
                        "Date": date,
                        "Open": price,
                        "High": price + 1,
                        "Low": price - 1,
                        "Close": price,
                        "AdjClose": price,
                        "Volume": 100 + offset,
                        "Value": 1000 + offset,
                        "Turn": 1.0,
                        "PE_TTM": 10 + offset / 10,
                        "PB": 2 + offset / 100,
                    }
                )
        raw = pd.DataFrame(rows)
        constituents = pd.DataFrame(
            [(stock, date) for stock in ("000001", "000002", "600000") for date in dates[:20]],
            columns=["Stock", "Date"],
        )
        industries = pd.DataFrame(
            {
                "Stock": ["000001", "000002", "600000"],
                "Name": ["A", "B", "C"],
                "Industry": ["X", "X", "Y"],
            }
        )
        tables = build_derived_tables(raw, constituents, industries)
        labels = tables["labels"]
        early = labels[labels.Date == dates[0]]
        self.assertEqual(len(early), 3)
        self.assertTrue(np.isfinite(early["10_15_labelB"]).all())
        self.assertEqual(list(tables["pool"].columns), ["Stock", "Date"])


if __name__ == "__main__":
    unittest.main()
