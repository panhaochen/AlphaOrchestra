"""Agent API with expensive LLM-backed modules loaded only when requested."""

from importlib import import_module
from typing import Any


_EXPORTS = {
    "BasicStockSelector": ("stock_disagreement.agent.stock_selector", "BasicStockSelector"),
    "RandomStockSelector": ("stock_disagreement.agent.stock_selector", "RandomStockSelector"),
    "IndustryEqualStockSelector": ("stock_disagreement.agent.stock_selector", "IndustryEqualStockSelector"),
    "IndustryBasisStockSelector": ("stock_disagreement.agent.stock_selector", "IndustryBasisStockSelector"),
    "MVEqualStockSelector": ("stock_disagreement.agent.stock_selector", "MVEqualStockSelector"),
    "StockDisagreementAgent": ("stock_disagreement.agent.basic_agent", "StockDisagreementAgent"),
    "Modality": ("stock_disagreement.agent.basic_agent", "Modality"),
    "InvestmentAnalyzer": ("stock_disagreement.agent.investment_analyzer", "InvestmentAnalyzer"),
    "InvestingHistory": ("stock_disagreement.agent.investing_history", "InvestingHistory"),
    "BaseOptimizer": ("stock_disagreement.agent.agent_distribution", "BaseOptimizer"),
    "SimulatedAnnealingOptimizer": ("stock_disagreement.agent.agent_distribution", "SimulatedAnnealingOptimizer"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value
