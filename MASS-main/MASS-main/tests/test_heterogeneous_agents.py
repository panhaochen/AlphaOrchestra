import random
import unittest

from stock_disagreement.agent.basic_agent import Modality
from stock_disagreement.exp.trainer import StockDisagreementTrainer


class HeterogeneousAgentPoolTests(unittest.TestCase):
    def setUp(self):
        self.trainer = StockDisagreementTrainer.__new__(StockDisagreementTrainer)
        self.trainer.random = random.Random(42)

    def test_modality_portfolio_contains_distinct_nonempty_combinations(self):
        modalities = [Modality.BASE_DATA, Modality.CROSS_INDUSTRY_LABEL, Modality.PRICE_FEATURE]
        combinations = self.trainer._heterogeneous_modalities(modalities)
        self.assertEqual(len(combinations), 7)
        self.assertEqual(len({int(value) for value in combinations}), 7)
        self.assertTrue(all(int(value) > 0 for value in combinations))

    def test_personas_differ_across_types_and_replicas(self):
        first = self.trainer._agent_persona(0, 0)
        second_type = self.trainer._agent_persona(1, 0)
        second_replica = self.trainer._agent_persona(0, 1)
        self.assertNotEqual(first, second_type)
        self.assertNotEqual(first, second_replica)
        self.assertIn("independent replica 2", second_replica)


if __name__ == "__main__":
    unittest.main()
