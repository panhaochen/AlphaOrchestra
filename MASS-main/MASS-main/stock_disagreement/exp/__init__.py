"""Experiment entry points, imported lazily to keep pure components lightweight."""

from typing import Any

__all__ = ["StockDisagreementTrainer"]


def __getattr__(name: str) -> Any:
    if name != "StockDisagreementTrainer":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from .trainer import StockDisagreementTrainer

    globals()[name] = StockDisagreementTrainer
    return StockDisagreementTrainer
