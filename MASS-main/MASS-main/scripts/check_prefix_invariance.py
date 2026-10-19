from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from stock_disagreement.config import RESULT_DIR


def _frame(run_dir: Path, name: str) -> pd.DataFrame:
    path = run_dir / name
    if not path.exists():
        raise FileNotFoundError(f"Missing run artifact: {path}")
    return pd.read_parquet(path)


def _compare_exact(left: pd.DataFrame, right: pd.DataFrame, keys: list[str], columns: list[str]) -> list[str]:
    merged = left[keys + columns].merge(
        right[keys + columns], on=keys, suffixes=("_short", "_long"), how="outer", indicator=True,
    )
    errors: list[str] = []
    unmatched = int((merged["_merge"] != "both").sum())
    if unmatched:
        errors.append(f"{unmatched} rows do not share the same keys")
    matched = merged[merged["_merge"] == "both"]
    for column in columns:
        a = matched[f"{column}_short"]
        b = matched[f"{column}_long"]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            equal = np.allclose(a.to_numpy(), b.to_numpy(), rtol=0.0, atol=1e-12, equal_nan=True)
            max_difference = float(np.nanmax(np.abs(a.to_numpy(dtype=float) - b.to_numpy(dtype=float)))) if len(a) else 0.0
            if not equal:
                errors.append(f"{column} differs (max absolute difference {max_difference:.12g})")
        elif not a.fillna("<NA>").equals(b.fillna("<NA>")):
            errors.append(f"{column} differs")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify that extending a run leaves its historical prefix unchanged.")
    parser.add_argument("--short_run", required=True)
    parser.add_argument("--long_run", required=True)
    args = parser.parse_args()

    short_dir = RESULT_DIR / "runs" / args.short_run
    long_dir = RESULT_DIR / "runs" / args.long_run

    short_signals = _frame(short_dir, "signals.parquet")
    long_signals = _frame(long_dir, "signals.parquet")
    prefix_end = int(short_signals["Date"].max())
    long_signals = long_signals[long_signals["Date"] <= prefix_end]
    signal_columns = [
        "Signal", "Signal_mean", "Signal_std", "ExposureControl",
        "DisagreementSeverity", "DisagreementType", "Signal_1d", "Signal_5d", "Signal_10d",
    ]
    errors = [
        f"signals: {message}"
        for message in _compare_exact(short_signals, long_signals, ["Date", "Stock"], signal_columns)
    ]

    short_routing = _frame(short_dir, "routing.parquet")
    long_routing = _frame(long_dir, "routing.parquet")
    long_routing = long_routing[long_routing["date"] <= prefix_end]
    routing_columns = [
        "selected", "candidate_count", "planned_active_count", "actual_active_count",
        "market_complexity", "estimated_cost", "standalone_score", "historical_rank_ic",
        "historical_observations", "exploration_bonus", "normalized_realized_cost",
    ]
    errors.extend(
        f"routing: {message}"
        for message in _compare_exact(short_routing, long_routing, ["date", "agent_index"], routing_columns)
    )

    short_states = _frame(short_dir, "market_states.parquet")
    long_states = _frame(long_dir, "market_states.parquet")
    long_states = long_states[long_states["date"] <= prefix_end]
    state_columns = [column for column in short_states.columns if column != "date"]
    errors.extend(
        f"market state: {message}"
        for message in _compare_exact(short_states, long_states, ["date"], state_columns)
    )

    print(f"Compared prefix through {prefix_end}.")
    if errors:
        print("PREFIX INVARIANCE: FAILED")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)
    print("PREFIX INVARIANCE: PASSED")


if __name__ == "__main__":
    main()
