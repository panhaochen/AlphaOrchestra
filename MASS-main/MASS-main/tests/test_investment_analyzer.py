import unittest

from stock_disagreement.agent.investment_analyzer import InvestmentAnalyzer


class InvestmentAnalyzerTests(unittest.TestCase):
    def setUp(self):
        analyzer = InvestmentAnalyzer()
        analyzer.data = {}
        analyzer.call_metrics = {}
        self.analyzer = analyzer

    def test_structured_metadata_reaches_stock_opinions(self):
        self.analyzer.record_score(
            20240102,
            investor_type=1,
            investor_id=1,
            stocks={"A": 1, "B": 0},
            decisions={
                "A": {
                    "confidence": 0.9,
                    "horizon_days": 30,
                    "evidence": ({"source_id": "report-1"},),
                    "reasoning_summary": "profit improved",
                }
            },
            call_metrics={"total_tokens": 120, "latency_seconds": 0.5},
        )
        opinions = self.analyzer.collect_stock_opinions(20240102, ["A", "B"], {1: 1.0})
        self.assertEqual(opinions["A"]["confidences"], [0.9])
        self.assertEqual(opinions["A"]["horizons"], [30.0])
        self.assertEqual(opinions["A"]["evidence"][0][0]["source_id"], "report-1")
        self.assertEqual(self.analyzer.call_metrics[20240102][1][0]["total_tokens"], 120)

    def test_unselected_stocks_receive_neutral_metadata_defaults(self):
        self.analyzer.record_score(20240102, 1, 1, {"A": 0})
        opinions = self.analyzer.collect_stock_opinions(20240102, ["A", "B"], {1: 1.0})
        self.assertEqual(opinions["A"]["confidences"], [0.5])
        self.assertEqual(opinions["A"]["horizons"], [20.0])
        self.assertEqual(opinions["B"]["confidences"], [0.5])
        self.assertEqual(opinions["B"]["horizons"], [20.0])

    def test_failed_call_is_audited_without_creating_an_opinion(self):
        self.analyzer.record_call_metrics(
            20240102,
            7,
            {"total_tokens": 99, "success": False, "failure_reason": "invalid JSON"},
        )
        self.assertNotIn(20240102, self.analyzer.data)
        self.assertEqual(self.analyzer.call_metrics[20240102][7][0]["total_tokens"], 99)
        self.assertFalse(self.analyzer.call_metrics[20240102][7][0]["success"])

    def test_contextual_weights_change_final_consensus(self):
        self.analyzer.record_score(20240102, 1, 1, {"A": 1})
        self.analyzer.record_score(20240102, 2, 1, {"A": 0})
        result = self.analyzer.calculate_stock_disagreement_score(
            20240102,
            ["A"],
            {1: 0.5, 2: 0.5},
            contextual_weights={"A": [0.9, 0.1]},
        )
        self.assertAlmostEqual(result["A"][1], 0.9)

    def test_opinions_have_stable_agent_order(self):
        self.analyzer.record_score(20240102, 2, 1, {"A": 0})
        self.analyzer.record_score(20240102, 1, 1, {"A": 1})
        opinions = self.analyzer.collect_stock_opinions(
            20240102, ["A"], {1: 0.5, 2: 0.5}
        )
        self.assertEqual(opinions["A"]["agent_types"], [1, 2])

    def test_debate_revision_updates_opinion_and_metrics(self):
        self.analyzer.record_score(20240102, 1, 1, {"A": 1})
        self.analyzer.apply_debate_revision(
            20240102,
            1,
            "A",
            {
                "prediction": -1.0,
                "confidence": 0.8,
                "horizon_days": 5,
                "evidence": ({"source_id": "debate"},),
                "reasoning_summary": "revised",
            },
            call_metrics={"total_tokens": 50},
        )
        opinions = self.analyzer.collect_stock_opinions(20240102, ["A"], {1: 1.0})
        self.assertEqual(opinions["A"]["scores"], [0.0])
        self.assertEqual(opinions["A"]["confidences"], [0.8])
        self.assertEqual(self.analyzer.call_metrics[20240102][1][0]["total_tokens"], 50)

    def test_abstention_zeroes_final_signal(self):
        self.analyzer.record_score(20240102, 1, 1, {"A": 1})
        result = self.analyzer.calculate_stock_disagreement_score(
            20240102,
            ["A"],
            {1: 1.0},
            disagreement_diagnostics={
                "A": {"type": "abstain", "severity": 1.0, "exposure_control": 0.0}
            },
        )
        self.assertEqual(result["A"][0], 0.0)
        self.assertEqual(result["A"][3], 0.0)
        self.assertEqual(result["A"][5], "abstain")

    def test_signal_alpha_prevents_equal_mean_std_cancellation(self):
        self.analyzer.record_score(20240102, 1, 1, {"A": 1})
        self.analyzer.record_score(20240102, 2, 1, {"A": 0})
        neutral = self.analyzer.calculate_stock_disagreement_score(
            20240102, ["A"], {1: 0.5, 2: 0.5}, alpha=0.5
        )
        tilted = self.analyzer.calculate_stock_disagreement_score(
            20240102, ["A"], {1: 0.5, 2: 0.5}, alpha=0.7
        )
        self.assertEqual(neutral["A"][0], 0.0)
        self.assertGreater(tilted["A"][0], 0.0)


if __name__ == "__main__":
    unittest.main()
