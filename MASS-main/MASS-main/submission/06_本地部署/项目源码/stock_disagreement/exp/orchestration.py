from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MarketState:
    date: int
    volatility: float
    breadth: float
    industry_dispersion: float
    event_intensity: float
    complexity: float
    observation_count: int = 0


class MarketStateEncoder:
    def __init__(
        self,
        stock_pool: pd.DataFrame,
        market_data: pd.DataFrame | None = None,
        news_relationship: pd.DataFrame | None = None,
        lookback_days: int = 20,
    ) -> None:
        self.stock_pool = stock_pool
        self.market_data = market_data if market_data is not None else pd.DataFrame()
        self.news_relationship = news_relationship if news_relationship is not None else pd.DataFrame()
        self.lookback_days = max(int(lookback_days), 2)

    def encode(self, date: int) -> MarketState:
        pool = self.stock_pool[self.stock_pool["Date"] == date]
        returns = self._observable_returns(date, pool)
        current_returns = returns[returns["Date"] == date] if not returns.empty else returns
        if current_returns.empty and not returns.empty:
            latest_date = returns["Date"].max()
            current_returns = returns[returns["Date"] == latest_date]

        per_stock_volatility = returns.groupby("Stock")["Return"].std() if not returns.empty else pd.Series(dtype=float)
        volatility = self._safe_float(per_stock_volatility.median())
        breadth = self._safe_float((current_returns["Return"] > 0).mean()) if len(current_returns) else 0.5

        merged = pool[[column for column in ["Stock", "Industry"] if column in pool]].drop_duplicates()
        merged = merged.merge(current_returns[["Stock", "Return"]], on="Stock", how="inner") if len(current_returns) else merged
        if "Industry" in merged and "Return" in merged:
            industry_ret = merged.groupby("Industry")["Return"].mean()
            industry_dispersion = self._safe_float(industry_ret.std())
        else:
            industry_dispersion = 0.0

        if not self.news_relationship.empty and "Date" in self.news_relationship:
            event_intensity = float(len(self.news_relationship[self.news_relationship["Date"] == date]))
            event_intensity = event_intensity / max(float(len(pool)), 1.0)
        else:
            event_intensity = 0.0

        complexity = self._normalize(volatility, scale=0.03) * 0.4
        complexity += abs(breadth - 0.5) * 0.6
        complexity += self._normalize(industry_dispersion, scale=0.03) * 0.3
        complexity += min(event_intensity, 1.0) * 0.2
        complexity = min(max(complexity, 0.0), 1.0)
        return MarketState(
            date,
            volatility,
            breadth,
            industry_dispersion,
            event_intensity,
            complexity,
            int(len(current_returns)),
        )

    def _observable_returns(self, date: int, pool: pd.DataFrame) -> pd.DataFrame:
        """Return trailing price observations available no later than ``date``."""
        required = {"Stock", "Date", "Close"}
        if self.market_data.empty or not required.issubset(self.market_data.columns):
            return pd.DataFrame(columns=["Stock", "Date", "Return"])
        active_stocks = set(pool["Stock"].tolist()) if "Stock" in pool else set()
        history = self.market_data[self.market_data["Date"] <= date]
        if active_stocks:
            history = history[history["Stock"].isin(active_stocks)]
        available_dates = sorted(history["Date"].dropna().unique())
        if len(available_dates) > self.lookback_days + 1:
            history = history[history["Date"].isin(available_dates[-(self.lookback_days + 1):])]
        history = history[["Stock", "Date", "Close"]].dropna().sort_values(["Stock", "Date"])
        history["Return"] = history.groupby("Stock")["Close"].pct_change(fill_method=None)
        return history.dropna(subset=["Return"])[["Stock", "Date", "Return"]]

    @staticmethod
    def _normalize(value: float, scale: float = 1.0) -> float:
        if not np.isfinite(value):
            return 0.0
        absolute = abs(float(value))
        return absolute / (absolute + max(scale, np.finfo(float).eps))

    @staticmethod
    def _safe_float(value: Any) -> float:
        if value is None or not np.isfinite(value):
            return 0.0
        return float(value)


@dataclass
class AgentProfile:
    agent_type: int
    ewma_rank_ic: float = 0.0
    performance_observations: int = 0
    average_tokens: float = 0.0
    average_latency_seconds: float = 0.0
    call_observations: int = 0
    failed_calls: int = 0
    average_monetary_cost: float = 0.0

    @property
    def failure_rate(self) -> float:
        return self.failed_calls / self.call_observations if self.call_observations else 0.0

    @property
    def normalized_cost(self) -> float:
        # Signal and routing decisions must be reproducible between a live call
        # and a cache replay. Wall-clock latency and transient failures depend on
        # network conditions, so retain them for operational audit but exclude
        # them from the economic score used by the model.
        return self.average_tokens / 1000.0 + self.average_monetary_cost


class AgentProfileStore:
    """Online profiles updated only when outcomes have become observable."""

    def __init__(self, performance_decay: float = 0.2) -> None:
        self.performance_decay = min(max(float(performance_decay), 0.0), 1.0)
        self.profiles: dict[int, AgentProfile] = {}
        self.evaluated_dates: set[int] = set()
        self.cost_dates: set[int] = set()

    def get(self, agent_type: int) -> AgentProfile:
        key = int(agent_type)
        if key not in self.profiles:
            self.profiles[key] = AgentProfile(agent_type=key)
        return self.profiles[key]

    def update_performance(self, agent_type: int, rank_ic: float) -> None:
        if not np.isfinite(rank_ic):
            return
        profile = self.get(agent_type)
        if profile.performance_observations == 0:
            profile.ewma_rank_ic = float(rank_ic)
        else:
            profile.ewma_rank_ic = (
                (1.0 - self.performance_decay) * profile.ewma_rank_ic
                + self.performance_decay * float(rank_ic)
            )
        profile.performance_observations += 1

    def update_call(self, agent_type: int, metrics: dict[str, Any]) -> None:
        profile = self.get(agent_type)
        count = profile.call_observations
        tokens = float(metrics.get("total_tokens", 0) or 0)
        latency = float(metrics.get("latency_seconds", 0) or 0)
        monetary_cost = float(metrics.get("monetary_cost", 0) or 0)
        profile.average_tokens = (profile.average_tokens * count + tokens) / (count + 1)
        profile.average_latency_seconds = (profile.average_latency_seconds * count + latency) / (count + 1)
        profile.average_monetary_cost = (profile.average_monetary_cost * count + monetary_cost) / (count + 1)
        profile.call_observations += 1
        if not metrics.get("success", True):
            profile.failed_calls += 1

    def records(self) -> list[dict[str, Any]]:
        return [
            {
                "agent_type": profile.agent_type,
                "ewma_rank_ic": profile.ewma_rank_ic,
                "performance_observations": profile.performance_observations,
                "average_tokens": profile.average_tokens,
                "average_latency_seconds": profile.average_latency_seconds,
                "call_observations": profile.call_observations,
                "failed_calls": profile.failed_calls,
                "failure_rate": profile.failure_rate,
                "average_monetary_cost": profile.average_monetary_cost,
                "normalized_cost": profile.normalized_cost,
            }
            for _, profile in sorted(self.profiles.items())
        ]

    @staticmethod
    def matured_prediction_date(dates: list[int], current_date: int, maturity_days: int) -> int | None:
        """Return the newest prediction date whose outcome is observable."""
        try:
            current_index = dates.index(current_date)
        except ValueError:
            return None
        mature_index = current_index - max(int(maturity_days), 1)
        return dates[mature_index] if mature_index >= 0 else None


@dataclass(frozen=True)
class DailyInferenceBudget:
    max_agent_calls: int = 0
    max_debate_calls: int = 0
    max_total_tokens: int = 0
    max_latency_seconds: float = 0.0
    max_monetary_cost: float = 0.0


class InferenceBudgetController:
    """Pre-call admission using estimates, followed by actual-usage auditing."""

    def __init__(self, policy: DailyInferenceBudget) -> None:
        self.policy = policy
        self.ledgers: dict[int, dict[str, float]] = {}

    def admit(self, date: int, call_type: str, estimate: dict[str, float] | None = None) -> bool:
        estimate = estimate or {}
        ledger = self._ledger(date)
        call_key = "reserved_agent_calls" if call_type == "agent" else "reserved_debate_calls"
        call_limit = self.policy.max_agent_calls if call_type == "agent" else self.policy.max_debate_calls
        if call_limit > 0 and ledger[call_key] + 1 > call_limit:
            return False
        tokens = max(float(estimate.get("total_tokens", 0) or 0), 0.0)
        latency = max(float(estimate.get("latency_seconds", 0) or 0), 0.0)
        cost = max(float(estimate.get("monetary_cost", 0) or 0), 0.0)
        if self.policy.max_total_tokens > 0 and ledger["reserved_tokens"] + tokens > self.policy.max_total_tokens:
            return False
        if self.policy.max_latency_seconds > 0 and ledger["reserved_latency_seconds"] + latency > self.policy.max_latency_seconds:
            return False
        if self.policy.max_monetary_cost > 0 and ledger["reserved_monetary_cost"] + cost > self.policy.max_monetary_cost:
            return False
        ledger[call_key] += 1
        ledger["reserved_tokens"] += tokens
        ledger["reserved_latency_seconds"] += latency
        ledger["reserved_monetary_cost"] += cost
        return True

    def record_actual(self, date: int, call_type: str, metrics: dict[str, Any]) -> None:
        ledger = self._ledger(date)
        key = "actual_agent_calls" if call_type == "agent" else "actual_debate_calls"
        ledger[key] += 1
        ledger["actual_tokens"] += float(metrics.get("billed_total_tokens", metrics.get("total_tokens", 0)) or 0)
        ledger["actual_latency_seconds"] += float(metrics.get("billed_latency_seconds", metrics.get("latency_seconds", 0)) or 0)
        ledger["actual_monetary_cost"] += float(metrics.get("billed_monetary_cost", metrics.get("monetary_cost", 0)) or 0)

    def estimate_for_agent(self, profile: AgentProfile, fallback_tokens: float = 4096.0) -> dict[str, float]:
        return {
            "total_tokens": profile.average_tokens if profile.call_observations else fallback_tokens,
            "latency_seconds": profile.average_latency_seconds if profile.call_observations else 0.0,
            "monetary_cost": profile.average_monetary_cost if profile.call_observations else 0.0,
        }

    def records(self) -> list[dict[str, Any]]:
        records = []
        for date, ledger in sorted(self.ledgers.items()):
            record = {"date": date, **ledger}
            record["token_overrun"] = self.policy.max_total_tokens > 0 and ledger["actual_tokens"] > self.policy.max_total_tokens
            record["latency_overrun"] = self.policy.max_latency_seconds > 0 and ledger["actual_latency_seconds"] > self.policy.max_latency_seconds
            record["cost_overrun"] = self.policy.max_monetary_cost > 0 and ledger["actual_monetary_cost"] > self.policy.max_monetary_cost
            records.append(record)
        return records

    def _ledger(self, date: int) -> dict[str, float]:
        if date not in self.ledgers:
            self.ledgers[date] = {
                "reserved_agent_calls": 0,
                "reserved_debate_calls": 0,
                "reserved_tokens": 0.0,
                "reserved_latency_seconds": 0.0,
                "reserved_monetary_cost": 0.0,
                "actual_agent_calls": 0,
                "actual_debate_calls": 0,
                "actual_tokens": 0.0,
                "actual_latency_seconds": 0.0,
                "actual_monetary_cost": 0.0,
            }
        return self.ledgers[date]


class DiversityCostAwareRouter:
    def __init__(
        self,
        top_k: int = 0,
        min_agents: int = 1,
        max_agents: int = 0,
        diversity_lambda: float = 0.35,
        cost_lambda: float = 0.05,
        profile_store: AgentProfileStore | None = None,
        performance_lambda: float = 1.0,
        exploration_lambda: float = 0.0,
        family_diversity_lambda: float = 0.0,
    ) -> None:
        self.top_k = top_k
        self.min_agents = max(1, min_agents)
        self.max_agents = max(0, int(max_agents))
        if self.max_agents and self.max_agents < self.min_agents:
            raise ValueError("max_agents must be zero or greater than or equal to min_agents")
        self.diversity_lambda = diversity_lambda
        self.cost_lambda = cost_lambda
        self.profile_store = profile_store
        self.performance_lambda = performance_lambda
        self.exploration_lambda = max(float(exploration_lambda), 0.0)
        self.family_diversity_lambda = max(float(family_diversity_lambda), 0.0)

    def select_agents(self, agents: list[Any], market_state: MarketState) -> list[Any]:
        if not agents:
            return []
        budget = self._budget(len(agents), market_state)
        selected: list[Any] = []
        remaining = list(agents)
        while remaining and len(selected) < budget:
            best_agent = max(
                remaining,
                key=lambda agent: self._selection_score(agent, selected, market_state),
            )
            selected.append(best_agent)
            remaining.remove(best_agent)
        return selected

    def selection_trace(self, agents: list[Any], selected: list[Any], market_state: MarketState) -> list[dict[str, Any]]:
        selected_ids = {id(agent) for agent in selected}
        planned_budget = self._budget(len(agents), market_state)
        trace = []
        for index, agent in enumerate(agents):
            trace.append({
                "agent_index": index,
                "agent_type": self._agent_type(agent),
                "investor_type": int(getattr(agent, "investor_type", self._agent_type(agent))),
                "modality": int(agent.modality),
                "selected": id(agent) in selected_ids,
                "candidate_count": len(agents),
                "planned_active_count": planned_budget,
                "actual_active_count": len(selected),
                "market_complexity": float(market_state.complexity),
                "estimated_cost": float(getattr(agent, "estimated_cost", 1.0)),
                "standalone_score": self._selection_score(agent, [], market_state),
                "historical_rank_ic": self._profile(agent).ewma_rank_ic if self.profile_store else 0.0,
                "historical_observations": self._profile(agent).performance_observations if self.profile_store else 0,
                "exploration_bonus": self._exploration_bonus(agent),
                "normalized_realized_cost": self._profile(agent).normalized_cost if self.profile_store else 0.0,
            })
        return trace

    def active_distributions(self, agents: list[Any], base_distributions: dict[int, float]) -> dict[int, float]:
        active_types = {self._agent_type(agent) for agent in agents}
        distributions = {
            agent_type: weight
            for agent_type, weight in base_distributions.items()
            if agent_type in active_types and weight > 0
        }
        total = sum(distributions.values())
        if total <= 0 and active_types:
            return {agent_type: 1.0 / len(active_types) for agent_type in active_types}
        return {agent_type: weight / total for agent_type, weight in distributions.items()}

    def _budget(self, agent_count: int, market_state: MarketState) -> int:
        if self.top_k > 0:
            return min(max(self.min_agents, self.top_k), agent_count)
        upper = min(self.max_agents, agent_count) if self.max_agents > 0 else agent_count
        lower = min(self.min_agents, upper)
        dynamic_budget = int(np.ceil(lower + market_state.complexity * (upper - lower)))
        return min(max(self.min_agents, dynamic_budget), agent_count)

    def _selection_score(self, agent: Any, selected: list[Any], market_state: MarketState) -> float:
        modality_count = max(int(agent.modality).bit_count(), 1)
        profile = self._profile(agent)
        utility = 1.0 + 0.15 * modality_count + 0.25 * market_state.complexity
        utility += self.performance_lambda * profile.ewma_rank_ic
        utility += self._exploration_bonus(agent)
        diversity_penalty = sum(self._similarity(agent, other) for other in selected)
        family_penalty = sum(
            self._same_family(agent, other)
            for other in selected
        )
        cost = profile.normalized_cost if profile.call_observations else float(getattr(agent, "estimated_cost", 1.0))
        return (
            utility
            - self.diversity_lambda * diversity_penalty
            - self.family_diversity_lambda * family_penalty
            - self.cost_lambda * cost
        )

    def _exploration_bonus(self, agent: Any) -> float:
        observations = self._profile(agent).performance_observations
        return self.exploration_lambda / np.sqrt(observations + 1.0)

    def _profile(self, agent: Any) -> AgentProfile:
        if self.profile_store is None:
            return AgentProfile(agent_type=self._agent_type(agent))
        return self.profile_store.get(self._agent_type(agent))

    @staticmethod
    def _agent_type(agent: Any) -> int:
        return int(getattr(agent, "agent_type", agent.modality))

    @staticmethod
    def _similarity(agent_a: Any, agent_b: Any) -> float:
        a = int(agent_a.modality)
        b = int(agent_b.modality)
        union = (a | b).bit_count()
        if union == 0:
            return 0.0
        return (a & b).bit_count() / union

    @staticmethod
    def _same_family(agent_a: Any, agent_b: Any) -> float:
        family_a = int(getattr(agent_a, "investor_type", DiversityCostAwareRouter._agent_type(agent_a)))
        family_b = int(getattr(agent_b, "investor_type", DiversityCostAwareRouter._agent_type(agent_b)))
        return 1.0 if family_a == family_b else 0.0


class ContextualAggregator:
    """Produce stock-specific trust weights from current evidence and history."""

    def __init__(
        self,
        profile_store: AgentProfileStore | None = None,
        performance_lambda: float = 1.0,
        confidence_lambda: float = 0.6,
        evidence_lambda: float = 0.15,
        agreement_lambda: float = 0.4,
        cost_lambda: float = 0.05,
    ) -> None:
        self.profile_store = profile_store
        self.performance_lambda = performance_lambda
        self.confidence_lambda = confidence_lambda
        self.evidence_lambda = evidence_lambda
        self.agreement_lambda = agreement_lambda
        self.cost_lambda = cost_lambda

    def calculate_weights(
        self,
        stock_opinions: dict[str, dict[str, list[Any]]],
        market_state: MarketState,
        profile_store: AgentProfileStore | None = None,
    ) -> tuple[dict[str, list[float]], list[dict[str, Any]]]:
        effective_profile_store = profile_store if profile_store is not None else self.profile_store
        weights_by_stock: dict[str, list[float]] = {}
        trace: list[dict[str, Any]] = []
        for stock, values in stock_opinions.items():
            scores = np.asarray(values.get("scores", []), dtype=float)
            base = np.asarray(values.get("weights", []), dtype=float)
            agent_types = [int(value) for value in values.get("agent_types", [])]
            if len(scores) == 0:
                weights_by_stock[stock] = []
                continue
            base = np.where(np.isfinite(base) & (base > 0), base, 0.0)
            base = base / base.sum() if base.sum() > 0 else np.full(len(scores), 1.0 / len(scores))
            confidences = self._aligned_numeric(values.get("confidences"), len(scores), 0.5)
            evidence_quality = np.asarray([
                min(len(items) / 3.0, 1.0) if isinstance(items, list) else 0.0
                for items in values.get("evidence", [[] for _ in scores])
            ], dtype=float)
            if len(evidence_quality) != len(scores):
                evidence_quality = np.zeros(len(scores), dtype=float)
            consensus = float(np.average(scores, weights=base))
            disagreement = np.abs(scores - consensus)
            logits = np.log(np.maximum(base, 1e-12))
            components = []
            for index, agent_type in enumerate(agent_types):
                profile = effective_profile_store.get(agent_type) if effective_profile_store else AgentProfile(agent_type)
                performance = self.performance_lambda * profile.ewma_rank_ic
                confidence = self.confidence_lambda * (confidences[index] - 0.5)
                evidence = self.evidence_lambda * evidence_quality[index]
                agreement = -self.agreement_lambda * disagreement[index] * (0.5 + market_state.complexity)
                cost = -self.cost_lambda * profile.normalized_cost
                logits[index] += performance + confidence + evidence + agreement + cost
                components.append((profile, performance, confidence, evidence, agreement, cost))
            normalized = self._softmax(logits)
            weights_by_stock[stock] = normalized.tolist()
            for index, agent_type in enumerate(agent_types):
                profile, performance, confidence, evidence, agreement, cost = components[index]
                trace.append({
                    "stock": stock,
                    "agent_type": agent_type,
                    "base_weight": float(base[index]),
                    "contextual_weight": float(normalized[index]),
                    "score": float(scores[index]),
                    "confidence": float(confidences[index]),
                    "evidence_quality": float(evidence_quality[index]),
                    "historical_rank_ic": profile.ewma_rank_ic,
                    "performance_component": performance,
                    "confidence_component": confidence,
                    "evidence_component": evidence,
                    "agreement_component": agreement,
                    "cost_component": cost,
                })
        return weights_by_stock, trace

    @staticmethod
    def _aligned_numeric(values: Any, length: int, default: float) -> np.ndarray:
        result = np.asarray(values if values is not None else [], dtype=float)
        if len(result) != length:
            return np.full(length, default, dtype=float)
        return np.where(np.isfinite(result), result, default)

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - np.max(logits)
        exponentials = np.exp(shifted)
        total = exponentials.sum()
        return exponentials / total if total > 0 else np.full(len(logits), 1.0 / len(logits))


class ConditionalDebateCoordinator:
    def __init__(self, max_agents_per_stock: int = 2, max_stocks_per_date: int = 5) -> None:
        self.max_agents_per_stock = max(int(max_agents_per_stock), 1)
        self.max_stocks_per_date = max(int(max_stocks_per_date), 0)

    def build_plan(
        self,
        opinions: dict[str, dict[str, list[Any]]],
        diagnostics: dict[str, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        candidates = []
        for stock, diagnosis in diagnostics.items():
            if diagnosis.get("recommended_action") != "targeted_debate" or stock not in opinions:
                continue
            values = opinions[stock]
            scores = np.asarray(values.get("scores", []), dtype=float)
            agent_types = [int(value) for value in values.get("agent_types", [])]
            if len(scores) < 2 or len(agent_types) != len(scores):
                continue
            center = float(np.median(scores))
            ranked = sorted(
                range(len(scores)),
                key=lambda index: abs(float(scores[index]) - center),
                reverse=True,
            )
            selected_indices = ranked[: min(self.max_agents_per_stock, len(ranked))]
            candidates.append({
                "stock": stock,
                "severity": float(diagnosis.get("severity", 0.0)),
                "agent_types": [agent_types[index] for index in selected_indices],
                "opinion_indices": selected_indices,
            })
        candidates.sort(key=lambda item: item["severity"], reverse=True)
        return candidates[: self.max_stocks_per_date]


class StructuredDisagreementDiagnoser:
    def __init__(
        self,
        severity_threshold: float = 0.35,
        abstention_threshold: float = 0.45,
    ) -> None:
        self.severity_threshold = severity_threshold
        self.abstention_threshold = abstention_threshold

    def diagnose(
        self,
        stock_opinions: dict[str, dict[str, list[float]]],
        market_state: MarketState,
    ) -> dict[str, dict[str, Any]]:
        diagnostics = {}
        for stock, values in stock_opinions.items():
            scores = np.array(values["scores"], dtype=float)
            weights = np.array(values["weights"], dtype=float)
            if len(scores) == 0:
                pred_dispersion = 0.0
                confidence_dispersion = 0.0
            else:
                weights = np.where(np.isfinite(weights) & (weights > 0), weights, 0.0)
                weights = weights / weights.sum() if weights.sum() > 0 else np.full(len(scores), 1.0 / len(scores))
                mean_score = float(np.average(scores, weights=weights))
                pred_dispersion = float(np.sqrt(np.average((scores - mean_score) ** 2, weights=weights)))
                confidences = np.array(values.get("confidences", np.abs(scores)), dtype=float)
                if len(confidences) != len(scores):
                    confidences = np.abs(scores)
                confidences = np.where(np.isfinite(confidences), confidences, 0.5)
                confidence_dispersion = float(np.sqrt(np.average((confidences - np.average(confidences, weights=weights)) ** 2, weights=weights)))
            horizons = np.array(values.get("horizons", []), dtype=float)
            horizon_gap = self._normalized_range(horizons, scale=252.0)
            evidence_disagreement = self._set_disagreement(values.get("evidence", []), self._evidence_tokens)
            semantic_disagreement = self._set_disagreement(values.get("reasoning", []), self._text_tokens)
            severity = pred_dispersion * 0.45 + confidence_dispersion * 0.15
            severity += horizon_gap * 0.1 + evidence_disagreement * 0.1
            severity += semantic_disagreement * 0.1 + market_state.complexity * 0.1
            severity = float(severity) if np.isfinite(severity) else 1.0
            exposure_control = max(0.0, 1.0 - severity)
            if severity >= self.abstention_threshold:
                action = "abstain"
                exposure_control = 0.0
            elif severity >= self.severity_threshold:
                action = "targeted_debate"
            else:
                action = "aggregate"
            diagnostics[stock] = {
                "type": action,
                "recommended_action": action,
                "severity": severity,
                "pred_dispersion": pred_dispersion,
                "confidence_dispersion": confidence_dispersion,
                "horizon_gap": horizon_gap,
                "evidence_disagreement": evidence_disagreement,
                "semantic_disagreement": semantic_disagreement,
                "market_complexity": market_state.complexity,
                "exposure_control": exposure_control,
                "agent_types": list(values.get("agent_types", [])),
            }
        return diagnostics

    @staticmethod
    def _normalized_range(values: np.ndarray, scale: float) -> float:
        finite = values[np.isfinite(values)]
        if len(finite) < 2:
            return 0.0
        return float(min((finite.max() - finite.min()) / max(scale, 1.0), 1.0))

    @classmethod
    def _set_disagreement(cls, values: list[Any], tokenizer) -> float:
        sets = [tokenizer(value) for value in values]
        pairs = []
        for left_index in range(len(sets)):
            for right_index in range(left_index + 1, len(sets)):
                union = sets[left_index] | sets[right_index]
                if union:
                    pairs.append(1.0 - len(sets[left_index] & sets[right_index]) / len(union))
        return float(np.mean(pairs)) if pairs else 0.0

    @staticmethod
    def _evidence_tokens(value: Any) -> set[str]:
        tokens = set()
        for item in value if isinstance(value, list) else []:
            if isinstance(item, dict):
                identity = item.get("source_id") or item.get("claim")
                if identity:
                    tokens.add(str(identity).strip().lower())
            elif item:
                tokens.add(str(item).strip().lower())
        return tokens

    @staticmethod
    def _text_tokens(value: Any) -> set[str]:
        return {token.lower() for token in str(value).split() if len(token) > 2}
