"""Validated, backward-compatible schema for an agent's investment decision."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from typing import Any


@dataclass(frozen=True)
class StockDecision:
    stock: str
    prediction: float = 1.0
    confidence: float = 0.5
    horizon_days: int = 20
    evidence: tuple[dict[str, str], ...] = field(default_factory=tuple)
    reasoning_summary: str = ""
    risk_factors: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AgentDecision:
    stocks: tuple[str, ...]
    decisions: tuple[StockDecision, ...]
    legacy_response: bool = False

    def by_stock(self) -> dict[str, dict[str, Any]]:
        return {decision.stock: decision.to_dict() for decision in self.decisions}


def parse_debate_revision(response: str, stock: str) -> StockDecision:
    payload = json.loads(response)
    if not isinstance(payload, dict):
        raise ValueError("Debate revision must be a JSON object.")
    returned_stock = str(payload.get("stock", stock))
    if returned_stock != str(stock):
        raise ValueError(f"Debate revision returned unexpected stock {returned_stock}.")
    evidence = tuple(_normalize_evidence(value) for value in payload.get("evidence", []) if value)
    return StockDecision(
        stock=returned_stock,
        prediction=_bounded_float(payload.get("prediction", 0.0), -1.0, 1.0, 0.0),
        confidence=_bounded_float(payload.get("confidence", 0.5), 0.0, 1.0, 0.5),
        horizon_days=int(_bounded_float(payload.get("horizon_days", 20), 1, 3650, 20)),
        evidence=evidence,
        reasoning_summary=str(payload.get("reasoning_summary", ""))[:2000],
        risk_factors=tuple(str(value)[:300] for value in payload.get("risk_factors", []) if value),
    )


def parse_agent_decision_response(
    response: str,
    legal_stocks: list[str],
    expected_count: int,
    default_horizon_days: int = 20,
) -> AgentDecision:
    payload = json.loads(response)
    if not isinstance(payload, dict) or not isinstance(payload.get("Stock"), list):
        raise ValueError('Decision response must contain a "Stock" list.')
    stocks = tuple(str(stock) for stock in payload["Stock"])
    if len(stocks) != expected_count or len(set(stocks)) != len(stocks):
        raise ValueError(f"Expected {expected_count} unique stocks, received {len(stocks)}.")
    illegal = sorted(set(stocks) - {str(stock) for stock in legal_stocks})
    if illegal:
        raise ValueError(f"Decision contains illegal stocks: {illegal}")

    raw_decisions = payload.get("Decisions")
    legacy = not isinstance(raw_decisions, list)
    detail_by_stock: dict[str, dict[str, Any]] = {}
    if not legacy:
        for item in raw_decisions:
            if isinstance(item, dict) and str(item.get("stock", "")) in stocks:
                detail_by_stock[str(item["stock"])] = item

    decisions = []
    for stock in stocks:
        item = detail_by_stock.get(stock, {})
        evidence = tuple(_normalize_evidence(value) for value in item.get("evidence", []) if value)
        decisions.append(
            StockDecision(
                stock=stock,
                prediction=_bounded_float(item.get("prediction", 1.0), -1.0, 1.0, 1.0),
                confidence=_bounded_float(item.get("confidence", 0.5), 0.0, 1.0, 0.5),
                horizon_days=int(_bounded_float(item.get("horizon_days", default_horizon_days), 1, 3650, default_horizon_days)),
                evidence=evidence,
                reasoning_summary=str(item.get("reasoning_summary", ""))[:2000],
                risk_factors=tuple(str(value)[:300] for value in item.get("risk_factors", []) if value),
            )
        )
    return AgentDecision(stocks=stocks, decisions=tuple(decisions), legacy_response=legacy)


def _normalize_evidence(value: Any) -> dict[str, str]:
    if isinstance(value, dict):
        return {
            "source_id": str(value.get("source_id", ""))[:300],
            "timestamp": str(value.get("timestamp", ""))[:100],
            "claim": str(value.get("claim", ""))[:1000],
            "direction": str(value.get("direction", "neutral"))[:50],
        }
    return {"source_id": "", "timestamp": "", "claim": str(value)[:1000], "direction": "neutral"}


def _bounded_float(value: Any, lower: float, upper: float, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return min(max(number, lower), upper)
