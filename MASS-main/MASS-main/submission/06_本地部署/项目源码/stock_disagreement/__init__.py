"""Public package API with optional LLM dependencies loaded on demand."""

from importlib import import_module
from typing import Any


_EXPORTS = {
    "OpenAIModel": ("stock_disagreement.utils.llm", "OpenAIModel"),
    "StockDisagreementAgent": ("stock_disagreement.agent", "StockDisagreementAgent"),
    "Modality": ("stock_disagreement.agent", "Modality"),
    "BasicStockSelector": ("stock_disagreement.agent", "BasicStockSelector"),
    "RandomStockSelector": ("stock_disagreement.agent", "RandomStockSelector"),
    "IndustryEqualStockSelector": ("stock_disagreement.agent", "IndustryEqualStockSelector"),
    "IndustryBasisStockSelector": ("stock_disagreement.agent", "IndustryBasisStockSelector"),
    "MVEqualStockSelector": ("stock_disagreement.agent", "MVEqualStockSelector"),
    "InvestmentAnalyzer": ("stock_disagreement.agent", "InvestmentAnalyzer"),
    "InvestingHistory": ("stock_disagreement.agent", "InvestingHistory"),
    "BaseOptimizer": ("stock_disagreement.agent", "BaseOptimizer"),
    "SimulatedAnnealingOptimizer": ("stock_disagreement.agent", "SimulatedAnnealingOptimizer"),
    "StockDisagreementTrainer": ("stock_disagreement.exp", "StockDisagreementTrainer"),
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
