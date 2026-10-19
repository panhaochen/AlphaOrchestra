import unittest
from types import SimpleNamespace

import pandas as pd

from stock_disagreement.exp.orchestration import (
    AgentProfileStore,
    ContextualAggregator,
    ConditionalDebateCoordinator,
    DailyInferenceBudget,
    InferenceBudgetController,
    DiversityCostAwareRouter,
    MarketState,
    MarketStateEncoder,
    StructuredDisagreementDiagnoser,
)


class MarketStateEncoderTests(unittest.TestCase):
    def setUp(self):
        self.pool = pd.DataFrame(
            {
                "Stock": ["A", "B", "A", "B"],
                "Date": [20240102, 20240102, 20240103, 20240103],
                "Industry": ["I1", "I2", "I1", "I2"],
            }
        )
        self.market = pd.DataFrame(
            {
                "Stock": ["A", "B", "A", "B", "A", "B"],
                "Date": [20240101, 20240101, 20240102, 20240102, 20240103, 20240103],
                "Close": [100.0, 100.0, 102.0, 98.0, 103.0, 99.0],
            }
        )

    def test_state_uses_only_observations_available_by_date(self):
        encoder = MarketStateEncoder(self.pool, self.market)
        before = encoder.encode(20240102)
        future_changed = self.market.copy()
        future_changed.loc[future_changed["Date"] == 20240103, "Close"] = [10000.0, 0.01]
        after = MarketStateEncoder(self.pool, future_changed).encode(20240102)
        self.assertEqual(before, after)
        self.assertEqual(before.observation_count, 2)
        self.assertAlmostEqual(before.breadth, 0.5)

    def test_missing_market_data_has_neutral_fallback(self):
        state = MarketStateEncoder(self.pool).encode(20240102)
        self.assertEqual(state.breadth, 0.5)
        self.assertEqual(state.observation_count, 0)


class RouterTests(unittest.TestCase):
    def test_dynamic_budget_is_bounded_by_min_and_max_agents(self):
        agents = [
            SimpleNamespace(agent_type=i, investor_type=i, modality=1 << (i % 3), estimated_cost=1.0)
            for i in range(64)
        ]
        router = DiversityCostAwareRouter(top_k=0, min_agents=8, max_agents=24)
        low = MarketState(20240102, 0.01, 0.5, 0.01, 0.0, 0.0)
        middle = MarketState(20240103, 0.01, 0.5, 0.01, 0.0, 0.5)
        high = MarketState(20240104, 0.01, 0.5, 0.01, 0.0, 1.0)
        self.assertEqual(len(router.select_agents(agents, low)), 8)
        self.assertEqual(len(router.select_agents(agents, middle)), 16)
        self.assertEqual(len(router.select_agents(agents, high)), 24)

    def test_agent_identity_is_distinct_from_duplicate_modalities(self):
        agents = [
            SimpleNamespace(agent_type=0, investor_type=0, modality=256, estimated_cost=1.0),
            SimpleNamespace(agent_type=3, investor_type=0, modality=256, estimated_cost=1.0),
        ]
        router = DiversityCostAwareRouter(top_k=2)
        state = MarketState(20240102, 0.01, 0.5, 0.01, 0.0, 0.5)
        selected = router.select_agents(agents, state)
        weights = router.active_distributions(selected, {0: 1.0, 3: 1.0})
        self.assertEqual(set(weights), {0, 3})
        self.assertEqual(
            [item["agent_type"] for item in router.selection_trace(agents, selected, state)],
            [0, 3],
        )
        self.assertEqual(
            [item["investor_type"] for item in router.selection_trace(agents, selected, state)],
            [0, 0],
        )

    def test_router_respects_top_k_and_normalizes_active_weights(self):
        agents = [
            SimpleNamespace(modality=1, estimated_cost=1.0),
            SimpleNamespace(modality=2, estimated_cost=1.0),
            SimpleNamespace(modality=3, estimated_cost=2.0),
        ]
        router = DiversityCostAwareRouter(top_k=2)
        state = MarketState(20240102, 0.01, 0.5, 0.01, 0.0, 0.5)
        selected = router.select_agents(agents, state)
        self.assertEqual(len(selected), 2)
        weights = router.active_distributions(selected, {1: 2.0, 2: 1.0, 3: 1.0})
        self.assertAlmostEqual(sum(weights.values()), 1.0)
        trace = router.selection_trace(agents, selected, state)
        self.assertEqual(sum(item["selected"] for item in trace), 2)

    def test_historical_performance_and_realized_cost_affect_selection(self):
        profiles = AgentProfileStore(performance_decay=1.0)
        profiles.update_performance(1, 0.4)
        profiles.update_performance(2, -0.2)
        profiles.update_call(1, {"total_tokens": 100, "latency_seconds": 0.1, "success": True})
        profiles.update_call(2, {"total_tokens": 5000, "latency_seconds": 5.0, "success": True})
        agents = [
            SimpleNamespace(modality=1, estimated_cost=1.0),
            SimpleNamespace(modality=2, estimated_cost=1.0),
        ]
        router = DiversityCostAwareRouter(top_k=1, profile_store=profiles, cost_lambda=0.1)
        state = MarketState(20240102, 0.01, 0.5, 0.01, 0.0, 0.5)
        selected = router.select_agents(agents, state)
        self.assertEqual(selected[0].modality, 1)
        trace = router.selection_trace(agents, selected, state)
        self.assertEqual(trace[0]["historical_rank_ic"], 0.4)
        self.assertGreater(trace[1]["normalized_realized_cost"], trace[0]["normalized_realized_cost"])

    def test_exploration_prefers_less_observed_agent(self):
        profiles = AgentProfileStore(performance_decay=1.0)
        profiles.update_performance(1, 0.0)
        profiles.update_performance(1, 0.0)
        agents = [
            SimpleNamespace(agent_type=1, investor_type=0, modality=32, estimated_cost=1.0),
            SimpleNamespace(agent_type=2, investor_type=1, modality=32, estimated_cost=1.0),
        ]
        router = DiversityCostAwareRouter(
            top_k=1,
            profile_store=profiles,
            exploration_lambda=0.5,
        )
        state = MarketState(20240102, 0.01, 0.5, 0.01, 0.0, 0.5)
        self.assertEqual(router.select_agents(agents, state)[0].agent_type, 2)

    def test_family_diversity_discourages_duplicate_family(self):
        agents = [
            SimpleNamespace(agent_type=0, investor_type=0, modality=32, estimated_cost=1.0),
            SimpleNamespace(agent_type=1, investor_type=0, modality=256, estimated_cost=1.0),
            SimpleNamespace(agent_type=2, investor_type=1, modality=256, estimated_cost=1.0),
        ]
        router = DiversityCostAwareRouter(
            top_k=2,
            diversity_lambda=0.0,
            family_diversity_lambda=1.0,
        )
        state = MarketState(20240102, 0.01, 0.5, 0.01, 0.0, 0.5)
        selected = router.select_agents(agents, state)
        self.assertEqual({agent.investor_type for agent in selected}, {0, 1})


class AgentProfileTests(unittest.TestCase):
    def test_label_outcome_is_hidden_until_maturity(self):
        dates = [20240101, 20240102, 20240103, 20240104]
        self.assertIsNone(AgentProfileStore.matured_prediction_date(dates, 20240103, 3))
        self.assertEqual(
            AgentProfileStore.matured_prediction_date(dates, 20240104, 3),
            20240101,
        )

    def test_profile_updates_use_ewma_and_actual_call_metrics(self):
        store = AgentProfileStore(performance_decay=0.25)
        store.update_performance(1, 0.4)
        store.update_performance(1, 0.0)
        store.update_call(1, {"total_tokens": 100, "latency_seconds": 1.0, "success": True})
        store.update_call(1, {"total_tokens": 300, "latency_seconds": 3.0, "success": False})
        profile = store.get(1)
        self.assertAlmostEqual(profile.ewma_rank_ic, 0.3)
        self.assertEqual(profile.average_tokens, 200.0)
        self.assertEqual(profile.average_latency_seconds, 2.0)
        self.assertEqual(profile.failure_rate, 0.5)

    def test_model_cost_is_invariant_to_network_latency_and_transient_failure(self):
        fast = AgentProfileStore()
        slow = AgentProfileStore()
        fast.update_call(1, {
            "total_tokens": 1200,
            "latency_seconds": 1.0,
            "monetary_cost": 0.02,
            "success": True,
        })
        slow.update_call(1, {
            "total_tokens": 1200,
            "latency_seconds": 300.0,
            "monetary_cost": 0.02,
            "success": False,
        })
        self.assertEqual(fast.get(1).normalized_cost, slow.get(1).normalized_cost)


class ContextualAggregatorTests(unittest.TestCase):
    def test_explicit_profile_snapshot_prevents_future_profile_leakage(self):
        final_store = AgentProfileStore(performance_decay=1.0)
        snapshot = AgentProfileStore(performance_decay=1.0)
        final_store.update_performance(1, 1.0)
        final_store.update_performance(2, -1.0)
        aggregator = ContextualAggregator(
            profile_store=final_store,
            confidence_lambda=0.0,
            evidence_lambda=0.0,
            agreement_lambda=0.0,
            cost_lambda=0.0,
        )
        opinions = {
            "A": {
                "scores": [1.0, 0.0],
                "weights": [1.0, 1.0],
                "agent_types": [1, 2],
            }
        }
        state = MarketState(20240102, 0.01, 0.5, 0.01, 0.0, 0.5)
        frozen, _ = aggregator.calculate_weights(opinions, state, profile_store=snapshot)
        leaked, _ = aggregator.calculate_weights(opinions, state)
        self.assertAlmostEqual(frozen["A"][0], 0.5)
        self.assertGreater(leaked["A"][0], frozen["A"][0])

    def test_weights_are_stock_specific_normalized_and_explainable(self):
        profiles = AgentProfileStore(performance_decay=1.0)
        profiles.update_performance(1, 0.5)
        profiles.update_performance(2, -0.2)
        aggregator = ContextualAggregator(profile_store=profiles)
        opinions = {
            "A": {
                "scores": [1.0, 0.0],
                "weights": [0.5, 0.5],
                "agent_types": [1, 2],
                "confidences": [0.9, 0.4],
                "evidence": [[{"source_id": "report"}], []],
            }
        }
        state = MarketState(20240102, 0.01, 0.5, 0.01, 0.0, 0.5)
        weights, trace = aggregator.calculate_weights(opinions, state)
        self.assertAlmostEqual(sum(weights["A"]), 1.0)
        self.assertGreater(weights["A"][0], weights["A"][1])
        self.assertEqual(len(trace), 2)
        self.assertIn("performance_component", trace[0])

    def test_neutral_context_falls_back_to_base_weights(self):
        aggregator = ContextualAggregator(
            performance_lambda=0.0,
            confidence_lambda=0.0,
            evidence_lambda=0.0,
            agreement_lambda=0.0,
            cost_lambda=0.0,
        )
        opinions = {
            "A": {
                "scores": [1.0, 0.0],
                "weights": [3.0, 1.0],
                "agent_types": [1, 2],
            }
        }
        state = MarketState(20240102, 0.0, 0.5, 0.0, 0.0, 0.0)
        weights, _ = aggregator.calculate_weights(opinions, state)
        self.assertAlmostEqual(weights["A"][0], 0.75)
        self.assertAlmostEqual(weights["A"][1], 0.25)


class DisagreementTests(unittest.TestCase):
    def test_diagnosis_is_weighted_and_auditable(self):
        diagnoser = StructuredDisagreementDiagnoser(
            severity_threshold=0.05,
            abstention_threshold=0.1,
        )
        opinions = {
            "A": {
                "scores": [0.0, 1.0],
                "weights": [0.9, 0.1],
                "agent_types": [1, 2],
                "confidences": [0.9, 0.2],
                "horizons": [5, 120],
                "evidence": [[{"source_id": "fundamental"}], [{"source_id": "news"}]],
                "reasoning": ["earnings are improving", "policy risk is rising"],
            }
        }
        state = MarketState(20240102, 0.0, 0.5, 0.0, 0.0, 0.0)
        result = diagnoser.diagnose(opinions, state)["A"]
        self.assertAlmostEqual(result["pred_dispersion"], 0.3)
        self.assertEqual(result["recommended_action"], result["type"])
        self.assertEqual(result["agent_types"], [1, 2])
        self.assertGreater(result["evidence_disagreement"], 0.0)
        self.assertGreater(result["horizon_gap"], 0.0)
        self.assertGreaterEqual(result["exposure_control"], 0.0)

    def test_high_severity_causes_real_abstention(self):
        diagnoser = StructuredDisagreementDiagnoser(
            severity_threshold=0.05,
            abstention_threshold=0.1,
        )
        opinions = {
            "A": {
                "scores": [0.0, 1.0],
                "weights": [0.5, 0.5],
                "agent_types": [1, 2],
            }
        }
        state = MarketState(20240102, 0.0, 0.5, 0.0, 0.0, 0.0)
        result = diagnoser.diagnose(opinions, state)["A"]
        self.assertEqual(result["recommended_action"], "abstain")
        self.assertEqual(result["exposure_control"], 0.0)

    def test_empty_metadata_produces_finite_diagnosis(self):
        diagnoser = StructuredDisagreementDiagnoser()
        opinions = {
            "A": {
                "scores": [0.0, 0.0],
                "weights": [0.5, 0.5],
                "agent_types": [1, 2],
                "confidences": [float("nan"), float("nan")],
                "horizons": [],
                "evidence": [[], []],
                "reasoning": ["", ""],
            }
        }
        state = MarketState(20240102, 0.0, 0.5, 0.0, 0.0, 0.1)
        result = diagnoser.diagnose(opinions, state)["A"]
        self.assertTrue(result["severity"] >= 0.0)
        self.assertGreater(result["exposure_control"], 0.0)


class ConditionalDebateTests(unittest.TestCase):
    def test_only_targeted_stocks_and_most_conflicting_agents_are_selected(self):
        coordinator = ConditionalDebateCoordinator(max_agents_per_stock=2, max_stocks_per_date=1)
        opinions = {
            "A": {"scores": [0.0, 0.5, 1.0], "agent_types": [1, 2, 3]},
            "B": {"scores": [0.0, 1.0], "agent_types": [1, 2]},
        }
        diagnostics = {
            "A": {"recommended_action": "targeted_debate", "severity": 0.3},
            "B": {"recommended_action": "targeted_debate", "severity": 0.2},
        }
        plan = coordinator.build_plan(opinions, diagnostics)
        self.assertEqual(len(plan), 1)
        self.assertEqual(plan[0]["stock"], "A")
        self.assertEqual(set(plan[0]["agent_types"]), {1, 3})


class InferenceBudgetTests(unittest.TestCase):
    def test_call_limits_are_hard_per_date_and_type(self):
        controller = InferenceBudgetController(DailyInferenceBudget(max_agent_calls=2, max_debate_calls=1))
        self.assertTrue(controller.admit(20240102, "agent"))
        self.assertTrue(controller.admit(20240102, "agent"))
        self.assertFalse(controller.admit(20240102, "agent"))
        self.assertTrue(controller.admit(20240102, "debate"))
        self.assertFalse(controller.admit(20240102, "debate"))
        self.assertTrue(controller.admit(20240103, "agent"))

    def test_estimated_resources_control_admission_and_actual_overrun_is_audited(self):
        controller = InferenceBudgetController(DailyInferenceBudget(max_total_tokens=100, max_monetary_cost=1.0))
        self.assertTrue(controller.admit(20240102, "agent", {"total_tokens": 60, "monetary_cost": 0.4}))
        self.assertFalse(controller.admit(20240102, "agent", {"total_tokens": 50, "monetary_cost": 0.2}))
        controller.record_actual(20240102, "agent", {"total_tokens": 120, "monetary_cost": 0.5})
        record = controller.records()[0]
        self.assertTrue(record["token_overrun"])
        self.assertFalse(record["cost_overrun"])

    def test_cached_usage_preserves_profile_cost_but_not_billed_budget(self):
        controller = InferenceBudgetController(DailyInferenceBudget(max_total_tokens=100))
        controller.record_actual(20240102, "agent", {
            "total_tokens": 120,
            "latency_seconds": 2.0,
            "billed_total_tokens": 0,
            "billed_latency_seconds": 0.0,
        })
        record = controller.records()[0]
        self.assertEqual(record["actual_tokens"], 0.0)
        self.assertEqual(record["actual_latency_seconds"], 0.0)
        self.assertFalse(record["token_overrun"])


if __name__ == "__main__":
    unittest.main()
