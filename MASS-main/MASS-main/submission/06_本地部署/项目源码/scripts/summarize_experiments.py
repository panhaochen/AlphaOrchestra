from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


LABEL_TO_SIGNAL = {
    "1d": ("1_15_labelB", "Signal_1d"),
    "5d": ("5_15_labelB", "Signal_5d"),
    "10d": ("10_15_labelB", "Signal_10d"),
}


def daily_corr(frame: pd.DataFrame, label: str, signal: str, rank: bool) -> pd.Series:
    values: dict[int, float] = {}
    for date, group in frame.groupby("Date"):
        pair = group[[label, signal]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(pair) < 3 or pair[label].nunique() < 2 or pair[signal].nunique() < 2:
            continue
        left = pair[label].rank() if rank else pair[label]
        right = pair[signal].rank() if rank else pair[signal]
        value = left.corr(right)
        if np.isfinite(value):
            values[int(date)] = float(value)
    return pd.Series(values, dtype=float)


def summarize(values: pd.Series) -> dict[str, float | int]:
    n = int(len(values))
    mean = float(values.mean()) if n else math.nan
    std = float(values.std()) if n > 1 else math.nan
    ratio = mean / std if np.isfinite(std) and std > 0 else math.nan
    sem = std / math.sqrt(n) if n > 1 else math.nan
    critical = float(stats.t.ppf(0.975, n - 1)) if n > 1 else math.nan
    t_stat = mean / sem if np.isfinite(sem) and sem > 0 else math.nan
    p_value = float(2 * stats.t.sf(abs(t_stat), n - 1)) if n > 1 else math.nan
    return {
        "value": mean,
        "ir": ratio,
        "std": std,
        "valid_days": n,
        "ci95_low": mean - critical * sem if n > 1 else math.nan,
        "ci95_high": mean + critical * sem if n > 1 else math.nan,
        "p_value": p_value,
    }


def evaluate_run(run_dir: Path, labels: pd.DataFrame) -> tuple[list[dict], dict]:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    arguments = manifest["arguments"]
    signals = pd.read_parquet(run_dir / "signals.parquet")
    evaluation_start = int(arguments.get("evaluation_start_date") or signals["Date"].min())
    signals = signals[signals["Date"] >= evaluation_start]
    merged = signals.merge(labels, on=["Stock", "Date"], how="inner")

    rows: list[dict] = []
    for horizon, (label, preferred_signal) in LABEL_TO_SIGNAL.items():
        signal = preferred_signal if preferred_signal in merged.columns else "Signal"
        for metric, rank in (("IC", False), ("RankIC", True)):
            result = summarize(daily_corr(merged, label, signal, rank))
            rows.append(
                {
                    "run_id": run_dir.name,
                    "evaluation_start_date": evaluation_start,
                    "evaluation_end_date": int(arguments["end_date"]),
                    "horizon": horizon,
                    "signal": signal,
                    "metric": metric,
                    **result,
                }
            )

    calls_path = run_dir / "llm_calls.parquet"
    resources = {
        "run_id": run_dir.name,
        "cache_mode": manifest.get("cache", {}).get("mode"),
        "cache_hits": manifest.get("cache", {}).get("hits", 0),
        "cache_misses": manifest.get("cache", {}).get("misses", 0),
        "cache_writes": manifest.get("cache", {}).get("writes", 0),
    }
    if calls_path.exists():
        calls = pd.read_parquet(calls_path)
        for column in (
            "llm_request_count",
            "cached_request_count",
            "total_tokens",
            "billed_total_tokens",
            "latency_seconds",
            "billed_latency_seconds",
            "monetary_cost",
            "billed_monetary_cost",
        ):
            resources[column] = float(calls[column].sum()) if column in calls else math.nan
    return rows, resources


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute experiment metrics from saved run artifacts.")
    parser.add_argument("--runs-dir", type=Path, default=Path("stock_disagreement/res/runs"))
    parser.add_argument("--labels", type=Path, default=Path("stock_disagreement/dataset/ih_label.parq"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    parser.add_argument("run_ids", nargs="*")
    args = parser.parse_args()

    run_ids = args.run_ids or [
        "q1_original_seed42",
        "q1_router_seed42",
        "q1_router_diagnosis_seed42",
        "q1_full_multihorizon_seed42",
        "q2_full_multihorizon_seed42",
        "q2_adaptive_alpha_seed42",
        "q3_original_seed42",
        "q3_router_seed42",
        "q3_router_diagnosis_seed42",
        "q3_full_multihorizon_seed42",
        "q3_adaptive_alpha_seed42",
    ]
    labels = pd.read_parquet(args.labels)[
        ["Stock", "Date", *(value[0] for value in LABEL_TO_SIGNAL.values())]
    ]
    metric_rows: list[dict] = []
    resource_rows: list[dict] = []
    for run_id in run_ids:
        run_dir = args.runs_dir / run_id
        if not (run_dir / "manifest.json").exists():
            print(f"Skipping missing run: {run_id}")
            continue
        metrics, resources = evaluate_run(run_dir, labels)
        metric_rows.extend(metrics)
        resource_rows.append(resources)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_frame = pd.DataFrame(metric_rows)
    resources_frame = pd.DataFrame(resource_rows)
    metrics_frame.to_csv(args.output_dir / "experiment_metrics.csv", index=False)
    resources_frame.to_csv(args.output_dir / "experiment_resources.csv", index=False)
    print(metrics_frame.to_string(index=False))
    print(f"Wrote {args.output_dir / 'experiment_metrics.csv'}")
    print(f"Wrote {args.output_dir / 'experiment_resources.csv'}")


if __name__ == "__main__":
    main()
