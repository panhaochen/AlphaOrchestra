from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from stock_disagreement.backtest import BacktestConfig, run_long_only_backtest
from stock_disagreement.config import first_existing_data_path, result_path, set_dataset_variant


def combine_benchmark_and_spread(
    top_returns: pd.Series,
    bottom_returns: pd.Series,
    benchmark_returns: pd.Series,
    active_gross_exposure: float,
) -> pd.Series:
    """Return a benchmark plus a dollar-neutral MASS ranking overlay.

    ``active_gross_exposure`` is the sum of the long and short notionals.  A
    value of 0.20 therefore means +10% in the high-score basket and -10% in
    the low-score basket.  Each leg is produced by the same next-open,
    transaction-cost-aware engine used by the long-only backtest.
    """
    if not 0.0 <= active_gross_exposure <= 1.0:
        raise ValueError("active_gross_exposure must be in [0, 1]")
    leg_weight = active_gross_exposure / 2.0
    return benchmark_returns + leg_weight * (top_returns - bottom_returns)


def combine_benchmark_and_top_excess(
    top_returns: pd.Series,
    benchmark_returns: pd.Series,
    active_weight: float,
    active_gate: pd.Series,
) -> pd.Series:
    """Blend from the benchmark into the high-score basket when a causal gate is active."""
    if not 0.0 <= active_weight <= 1.0:
        raise ValueError("active_weight must be in [0, 1]")
    gate = pd.to_numeric(active_gate, errors="coerce").fillna(0.0).clip(0.0, 1.0)
    weight = active_weight * gate
    return benchmark_returns + weight * (top_returns - benchmark_returns)


def summarize_returns(returns: pd.Series, benchmark_returns: pd.Series, annualization: int) -> dict[str, float]:
    returns = pd.to_numeric(returns, errors="coerce").fillna(0.0)
    benchmark_returns = pd.to_numeric(benchmark_returns, errors="coerce").fillna(0.0)
    total_return = float((1.0 + returns).prod() - 1.0)
    benchmark_total_return = float((1.0 + benchmark_returns).prod() - 1.0)
    years = max(len(returns) / annualization, 1.0 / annualization)
    annualized_return = float((1.0 + total_return) ** (1.0 / years) - 1.0) if total_return > -1 else -1.0
    volatility = float(returns.std(ddof=1) * np.sqrt(annualization)) if len(returns) > 1 else 0.0
    excess = returns - benchmark_returns
    tracking_error = float(excess.std(ddof=1) * np.sqrt(annualization)) if len(excess) > 1 else 0.0
    curve = (1.0 + returns).cumprod()
    drawdown = curve / curve.cummax() - 1.0
    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_volatility": volatility,
        "sharpe": annualized_return / volatility if volatility > 0 else 0.0,
        "max_drawdown": float(drawdown.min()),
        "positive_day_rate": float((returns > 0).mean()),
        "benchmark_total_return": benchmark_total_return,
        "active_total_return": float((1.0 + total_return) / (1.0 + benchmark_total_return) - 1.0)
        if benchmark_total_return > -1 else 0.0,
        "annualized_tracking_error": tracking_error,
        "information_ratio": float(excess.mean() / excess.std(ddof=1) * np.sqrt(annualization))
        if len(excess) > 1 and excess.std(ddof=1) > 0 else 0.0,
        "trading_days": float(len(returns)),
    }


def main() -> None:
    parser = argparse.ArgumentParser("Backtest a benchmark plus a cost-aware MASS ranking overlay.")
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--output_tag", default="benchmark_enhanced_candidate")
    parser.add_argument("--signal_column", default="Signal_10d", choices=["Signal_1d", "Signal_5d", "Signal_10d", "Signal"])
    parser.add_argument("--evaluation_start_date", type=int, required=True)
    parser.add_argument("--evaluation_end_date", type=int, required=True)
    parser.add_argument("--top_n", type=int, default=15)
    parser.add_argument("--rebalance_every", type=int, default=10)
    parser.add_argument("--active_gross_exposure", type=float, default=0.20)
    parser.add_argument("--overlay_mode", choices=["long_short", "risk_off_top"], default="long_short")
    parser.add_argument("--regime_lookback", type=int, default=60)
    parser.add_argument("--initial_capital", type=float, default=10_000_000.0)
    parser.add_argument("--commission_bps", type=float, default=3.0)
    parser.add_argument("--slippage_bps", type=float, default=5.0)
    parser.add_argument("--sell_stamp_duty_bps", type=float, default=10.0)
    parser.add_argument("--impact_bps", type=float, default=10.0)
    parser.add_argument("--max_participation_rate", type=float, default=0.05)
    args = parser.parse_args()

    run_dir = result_path("runs", args.run_id)
    signal_path = run_dir / "signals.parquet"
    if not signal_path.exists():
        raise FileNotFoundError(f"Missing run signals: {signal_path}")
    if args.top_n <= 0:
        raise ValueError("top_n must be positive")

    set_dataset_variant("csi300")
    signals = pd.read_parquet(signal_path)
    prices = pd.read_parquet(first_existing_data_path("base_data.parq"))
    benchmark_path = Path("stock_disagreement/dataset/csi300/benchmark_000300.parquet")
    if not benchmark_path.exists():
        raise FileNotFoundError(f"Missing benchmark: {benchmark_path}")
    benchmark = pd.read_parquet(benchmark_path).sort_values("Date")
    benchmark["Date"] = benchmark["Date"].astype(int)
    benchmark["benchmark_return"] = pd.to_numeric(benchmark["Close"], errors="coerce").pct_change(fill_method=None)

    config = BacktestConfig(
        initial_capital=args.initial_capital,
        top_n=args.top_n,
        rebalance_every=args.rebalance_every,
        commission_bps=args.commission_bps,
        slippage_bps=args.slippage_bps,
        sell_stamp_duty_bps=args.sell_stamp_duty_bps,
        impact_bps=args.impact_bps,
        max_participation_rate=args.max_participation_rate,
        max_position_weight=1.0 / args.top_n,
    )
    top = run_long_only_backtest(
        signals,
        prices,
        signal_column=args.signal_column,
        evaluation_start_date=args.evaluation_start_date,
        evaluation_end_date=args.evaluation_end_date,
        config=config,
    )
    bottom_signals = signals.copy()
    bottom_signals[args.signal_column] = -pd.to_numeric(bottom_signals[args.signal_column], errors="coerce")
    bottom = run_long_only_backtest(
        bottom_signals,
        prices,
        signal_column=args.signal_column,
        evaluation_start_date=args.evaluation_start_date,
        evaluation_end_date=args.evaluation_end_date,
        config=config,
    )

    daily = top.daily[["Date", "return", "turnover", "transaction_cost"]].rename(columns={
        "return": "top_return", "turnover": "top_turnover", "transaction_cost": "top_transaction_cost"
    })
    daily = daily.merge(
        bottom.daily[["Date", "return", "turnover", "transaction_cost"]].rename(columns={
            "return": "bottom_return", "turnover": "bottom_turnover", "transaction_cost": "bottom_transaction_cost"
        }),
        on="Date",
        how="inner",
    )
    daily = daily.merge(benchmark[["Date", "benchmark_return"]], on="Date", how="left")
    daily["benchmark_return"] = daily["benchmark_return"].fillna(0.0)
    if args.overlay_mode == "risk_off_top":
        if args.regime_lookback <= 1:
            raise ValueError("regime_lookback must be greater than one")
        regime = benchmark[["Date", "Close"]].copy()
        regime["moving_average"] = pd.to_numeric(regime["Close"], errors="coerce").rolling(
            args.regime_lookback, min_periods=args.regime_lookback
        ).mean()
        # Today's portfolio is determined using only information available at
        # the preceding close, matching the next-open signal execution rule.
        regime["active_gate"] = (
            pd.to_numeric(regime["Close"], errors="coerce") < regime["moving_average"]
        ).astype(float).shift(1).fillna(0.0)
        daily = daily.merge(regime[["Date", "active_gate"]], on="Date", how="left")
        daily["active_gate"] = daily["active_gate"].fillna(0.0)
        daily["return"] = combine_benchmark_and_top_excess(
            daily["top_return"], daily["benchmark_return"], args.active_gross_exposure, daily["active_gate"]
        )
    else:
        daily["active_gate"] = 1.0
        daily["return"] = combine_benchmark_and_spread(
            daily["top_return"], daily["bottom_return"], daily["benchmark_return"], args.active_gross_exposure
        )
    daily["excess_return"] = daily["return"] - daily["benchmark_return"]
    daily["spread_return"] = daily["top_return"] - daily["bottom_return"]
    daily["equity"] = args.initial_capital * (1.0 + daily["return"]).cumprod()
    daily["benchmark_equity"] = args.initial_capital * (1.0 + daily["benchmark_return"]).cumprod()

    metrics = summarize_returns(daily["return"], daily["benchmark_return"], config.annualization)
    leg_weight = args.active_gross_exposure / 2.0
    if args.overlay_mode == "risk_off_top":
        overlay_turnover = float(
            (args.active_gross_exposure * daily["active_gate"] * daily["top_turnover"]).mean()
        )
        overlay_cost = float(
            (args.active_gross_exposure * daily["active_gate"] * daily["top_transaction_cost"]).sum()
        )
    else:
        overlay_turnover = float(leg_weight * (daily["top_turnover"] + daily["bottom_turnover"]).mean())
        overlay_cost = float(leg_weight * (daily["top_transaction_cost"] + daily["bottom_transaction_cost"]).sum())
    metrics.update({
        "active_gross_exposure": float(args.active_gross_exposure),
        "active_gate_rate": float(daily["active_gate"].mean()),
        "spread_total_return": float((1.0 + daily["spread_return"]).prod() - 1.0),
        "average_overlay_turnover": overlay_turnover,
        "estimated_overlay_transaction_cost": overlay_cost,
    })

    output_dir = run_dir / "backtests" / args.output_tag
    output_dir.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(output_dir / "daily.parquet", index=False)
    top.trades.assign(leg="long_high_score").to_parquet(output_dir / "top_trades.parquet", index=False)
    bottom.trades.assign(leg="short_low_score_proxy").to_parquet(output_dir / "bottom_trades.parquet", index=False)
    payload = {
        "status": "development_candidate_not_independent_blind_test",
        "method": (
            "CSI300 benchmark + causal risk-off high-score MASS overlay"
            if args.overlay_mode == "risk_off_top"
            else "CSI300 benchmark + dollar-neutral high-minus-low MASS overlay"
        ),
        "metrics": metrics,
        "config": {
            **asdict(config),
            "active_gross_exposure": args.active_gross_exposure,
            "overlay_mode": args.overlay_mode,
            "regime_lookback": args.regime_lookback,
        },
        "limitations": [
            "Developed after inspecting the 2026 evaluation period; requires a new forward test.",
            "The short leg is a research proxy and requires borrow, derivatives, or long-only underweight implementation.",
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Backtest artifacts saved to: {output_dir}")


if __name__ == "__main__":
    main()
