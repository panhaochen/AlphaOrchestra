import json
import unittest

from stock_disagreement.decision import parse_agent_decision_response, parse_debate_revision


class AgentDecisionParserTests(unittest.TestCase):
    def test_debate_revision_is_validated(self):
        revision = parse_debate_revision(
            json.dumps({
                "stock": "A",
                "prediction": -0.4,
                "confidence": 0.7,
                "horizon_days": 5,
                "evidence": ["policy notice"],
                "reasoning_summary": "Risk increased",
            }),
            "A",
        )
        self.assertEqual(revision.stock, "A")
        self.assertEqual(revision.prediction, -0.4)
        with self.assertRaises(ValueError):
            parse_debate_revision('{"stock": "B"}', "A")

    def test_structured_response_is_validated_and_normalized(self):
        response = json.dumps({
            "Stock": ["A"],
            "Decisions": [{
                "stock": "A",
                "prediction": 2,
                "confidence": 0.8,
                "horizon_days": 10,
                "evidence": [{"source_id": "report-1", "claim": "profit improved"}],
                "reasoning_summary": "Improving fundamentals",
                "risk_factors": ["valuation"],
            }],
        })
        decision = parse_agent_decision_response(response, ["A", "B"], 1)
        detail = decision.by_stock()["A"]
        self.assertFalse(decision.legacy_response)
        self.assertEqual(detail["prediction"], 1.0)
        self.assertEqual(detail["confidence"], 0.8)
        self.assertEqual(detail["evidence"][0]["source_id"], "report-1")

    def test_legacy_stock_only_response_remains_supported(self):
        decision = parse_agent_decision_response('{"Stock": ["A"]}', ["A"], 1)
        self.assertTrue(decision.legacy_response)
        self.assertEqual(decision.by_stock()["A"]["confidence"], 0.5)

    def test_illegal_or_duplicate_stocks_are_rejected(self):
        with self.assertRaises(ValueError):
            parse_agent_decision_response('{"Stock": ["X"]}', ["A"], 1)
        with self.assertRaises(ValueError):
            parse_agent_decision_response('{"Stock": ["A", "A"]}', ["A"], 2)


if __name__ == "__main__":
    unittest.main()
