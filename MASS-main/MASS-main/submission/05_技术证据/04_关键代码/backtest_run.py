import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from stock_disagreement.backtest import BacktestConfig, run_long_only_backtest
from stock_disagreement.config import first_existing_data_path, result_path, set_dataset_variant


def main() -> None:
    parser = argparse.ArgumentParser("Backtest a saved MASS run with next-open execution.")
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--output_tag", default=None, help="Optional subdirectory name for preserving variants.")
    parser.add_argument("--signal_column", default="Signal_5d", choices=["Signal_1d", "Signal_5d", "Signal_10d", "Signal"])
    parser.add_argument("--evaluation_start_date", type=int, default=None)
    parser.add_argument("--evaluation_end_date", type=int, default=None)
    parser.add_argument("--initial_capital", type=float, default=10_000_000.0)
    parser.add_argument("--top_n", type=int, default=20)
    parser.add_argument("--rebalance_every", type=int, default=1)
    parser.add_argument("--commission_bps", type=float, default=3.0)
    parser.add_argument("--slippage_bps", type=float, default=5.0)
    parser.add_argument("--sell_stamp_duty_bps", type=float, default=10.0)
    parser.add_argument("--impact_bps", type=float, default=10.0)
    parser.add_argument("--max_participation_rate", type=float, default=0.05)
    parser.add_argument("--max_position_weight", type=float, default=0.10)
    parser.add_argument("--min_signal", type=float, default=None)
    parser.add_argument("--industry_balanced", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--min_value_quantile", type=float, default=0.0)
    parser.add_argument("--market_regime_control", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--regime_lookback", type=int, default=60)
    parser.add_argument("--risk_off_exposure", type=float, default=0.5)
    parser.add_argument("--residualize_signal", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    run_dir = result_path("runs", args.run_id)
    signal_path = run_dir / "signals.parquet"
    if not signal_path.exists():
        raise FileNotFoundError(f"Missing run signals: {signal_path}")
    set_dataset_variant("csi300")
    signals = pd.read_parquet(signal_path)
    prices = pd.read_parquet(first_existing_data_path("base_data.parq"))
    benchmark_path = Path("stock_disagreement/dataset/csi300/benchmark_000300.parquet")
    benchmark = pd.read_parquet(benchmark_path) if benchmark_path.exists() else None
    market_exposure = None
    regime = None
    if args.market_regime_control:
        if benchmark is None:
            raise FileNotFoundError(f"Market regime control requires {benchmark_path}")
        regime = benchmark[["Date", "Close"]].copy().sort_values("Date")
        regime["Date"] = regime["Date"].astype(int)
        regime["Close"] = pd.to_numeric(regime["Close"], errors="coerce")
        regime["MovingAverage"] = regime["Close"].rolling(args.regime_lookback, min_periods=args.regime_lookback).mean()
        regime["TargetExposure"] = np.where(
            regime["MovingAverage"].notna() & (regime["Close"] < regime["MovingAverage"]),
            args.risk_off_exposure,
            1.0,
        )
        market_exposure = regime[["Date", "TargetExposure"]]
    config = BacktestConfig(
        initial_capital=args.initial_capital,
        top_n=args.top_n,
        rebalance_every=args.rebalance_every,
        commission_bps=args.commission_bps,
        slippage_bps=args.slippage_bps,
        sell_stamp_duty_bps=args.sell_stamp_duty_bps,
        impact_bps=args.impact_bps,
        max_participation_rate=args.max_participation_rate,
        max_position_weight=args.max_position_weight,
        min_signal=args.min_signal,
        industry_balanced=args.industry_balanced,
        min_value_quantile=args.min_value_quantile,
        residualize_signal=args.residualize_signal,
    )
    result = run_long_only_backtest(
        signals,
        prices,
        signal_column=args.signal_column,
        evaluation_start_date=args.evaluation_start_date,
        evaluation_end_date=args.evaluation_end_date,
        config=config,
        market_exposure=market_exposure,
    )
    if benchmark is not None and not result.daily.empty:
        benchmark["Date"] = benchmark["Date"].astype(int)
        if "Return" not in benchmark:
            benchmark = benchmark.sort_values("Date")
            benchmark["Return"] = pd.to_numeric(benchmark["Close"], errors="coerce").pct_change(fill_method=None)
        benchmark = benchmark[["Date", "Return"]].rename(columns={"Return": "benchmark_return"})
        result.daily = result.daily.merge(benchmark, on="Date", how="left")
        result.daily["benchmark_return"] = result.daily["benchmark_return"].fillna(0.0)
        result.daily["excess_return"] = result.daily["return"] - result.daily["benchmark_return"]
        strategy_growth = float((1.0 + result.daily["return"]).prod())
        benchmark_growth = float((1.0 + result.daily["benchmark_return"]).prod())
        excess = result.daily["excess_return"]
        tracking_error = float(excess.std(ddof=1) * np.sqrt(config.annualization)) if len(excess) > 1 else 0.0
        result.metrics.update({
            "benchmark_total_return": benchmark_growth - 1.0,
            "active_total_return": strategy_growth / benchmark_growth - 1.0 if benchmark_growth > 0 else 0.0,
            "annualized_tracking_error": tracking_error,
            "information_ratio": float(excess.mean() / excess.std(ddof=1) * np.sqrt(config.annualization))
            if len(excess) > 1 and excess.std(ddof=1) > 0 else 0.0,
        })
        if regime is not None:
            executed = regime[["Date", "TargetExposure"]].copy().sort_values("Date")
            executed["executed_exposure"] = executed["TargetExposure"].shift(1).fillna(1.0)
            result.daily = result.daily.merge(executed[["Date", "executed_exposure"]], on="Date", how="left")
            result.daily["executed_exposure"] = result.daily["executed_exposure"].fillna(1.0)
            result.daily["matched_benchmark_return"] = (
                result.daily["executed_exposure"] * result.daily["benchmark_return"]
            )
            matched_growth = float((1.0 + result.daily["matched_benchmark_return"]).prod())
            matched_excess = result.daily["return"] - result.daily["matched_benchmark_return"]
            result.metrics.update({
                "matched_benchmark_total_return": matched_growth - 1.0,
                "active_return_vs_matched_benchmark": strategy_growth / matched_growth - 1.0
                if matched_growth > 0 else 0.0,
                "information_ratio_vs_matched_benchmark":
                    float(matched_excess.mean() / matched_excess.std(ddof=1) * np.sqrt(config.annualization))
                    if len(matched_excess) > 1 and matched_excess.std(ddof=1) > 0 else 0.0,
            })
    output_dir = run_dir / "backtests" / (args.output_tag or args.signal_column)
    output_dir.mkdir(parents=True, exist_ok=True)
    result.daily.to_parquet(output_dir / "daily.parquet", index=False)
    result.trades.to_parquet(output_dir / "trades.parquet", index=False)
    (output_dir / "summary.json").write_text(
        json.dumps({"metrics": result.metrics, "config": result.config}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result.metrics, indent=2))
    print(f"Backtest artifacts saved to: {output_dir}")


if __name__ == "__main__":
    main()
