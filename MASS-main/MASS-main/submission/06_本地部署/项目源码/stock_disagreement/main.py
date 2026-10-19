import argparse
import sys
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import platform
import random
import re

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np
import pickle as pkl
from stock_disagreement import StockDisagreementTrainer
from stock_disagreement.config import (
    CHAT_MODEL_NAME,
    LLM_BASE_URL,
    REASONING_MODEL_NAME,
    VISION_MODEL_NAME,
    data_path,
    first_existing_data_path,
    result_path,
    set_dataset_variant,
)
from stock_disagreement.utils.llm import GLOBAL_RESPONSE_CACHE

def calculate_ic_rankic(
    res: pd.DataFrame,
    stock_labels: pd.DataFrame,
    evaluation_start_date: int | None = None,
):
    def daily_corr(frame: pd.DataFrame, label: str, signal: str, rank: bool = False) -> pd.Series:
        values = {}
        for date, group in frame.groupby("Date"):
            pair = group[[label, signal]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(pair) < 3 or pair[label].nunique() < 2 or pair[signal].nunique() < 2:
                continue
            left = pair[label].rank() if rank else pair[label]
            right = pair[signal].rank() if rank else pair[signal]
            correlation = left.corr(right)
            if np.isfinite(correlation):
                values[date] = float(correlation)
        return pd.Series(values, dtype=float)

    def summarize(values: pd.Series) -> tuple[float, float, int]:
        if values.empty:
            return float("nan"), float("nan"), 0
        mean = float(values.mean())
        std = float(values.std())
        ratio = mean / std if np.isfinite(std) and std > 0 else float("nan")
        return mean, ratio, len(values)
  
    signal_by_label = {
        "1_15_labelB": "Signal_1d",
        "5_15_labelB": "Signal_5d",
        "10_15_labelB": "Signal_10d",
    }
    label_cols = list(signal_by_label)
    if evaluation_start_date is None:
        evaluation_start_date = int(res["Date"].min())
    sub_res = res[res["Date"] >= int(evaluation_start_date)]
    signal_cols = ["Signal", "Signal_mean", "Signal_std"] + [
        column for column in signal_by_label.values() if column in sub_res.columns
    ]
    df = sub_res[["Stock", "Date"] + signal_cols].merge(stock_labels[["Stock", "Date"] + label_cols], on=["Stock", "Date"])
    for label_col in label_cols:
        horizon_signal = signal_by_label[label_col]
        if horizon_signal not in df.columns:
            horizon_signal = "Signal"
        temp_df = df[[label_col, horizon_signal, "Signal_mean", "Signal_std"] + ["Stock", "Date"]].copy()
      
        ic_values = daily_corr(temp_df, label_col, horizon_signal)
        rankic_values = daily_corr(temp_df, label_col, horizon_signal, rank=True)
        ic_values_mean = daily_corr(temp_df, label_col, "Signal_mean")
        rankic_values_mean = daily_corr(temp_df, label_col, "Signal_mean", rank=True)
        ic_values_std = daily_corr(temp_df, label_col, "Signal_std")
        rankic_values_std = daily_corr(temp_df, label_col, "Signal_std", rank=True)

        ic, icir, ic_days = summarize(ic_values)
        rankic, rankic_ir, rankic_days = summarize(rankic_values)
        ic_mean, icir_mean, mean_days = summarize(ic_values_mean)
        rankic_mean, rankicir_mean, _ = summarize(rankic_values_mean)
        ic_std, icir_std, std_days = summarize(ic_values_std)
        rankic_std, rankicir_std, _ = summarize(rankic_values_std)
        print(f"for {label_col} using {horizon_signal}, ic: {ic}, icir: {icir}, rankic: {rankic}, rankicir: {rankic_ir}, valid_days: {min(ic_days, rankic_days)}")
        print(f"for {label_col}, ic_mean: {ic_mean}, icir_mean: {icir_mean}, rankic_mean: {rankic_mean}, rankicir_mean: {rankicir_mean}, valid_days: {mean_days}")
        print(f"for {label_col}, ic_std: {ic_std}, icir_std: {icir_std}, rankic_std: {rankic_std}, rankicir_std: {rankicir_std}, valid_days: {std_days}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Stock disagreement argument parser.")
    parser.add_argument("--num_investor_type", type=int, default= 16)
    parser.add_argument("--num_agents_per_investor", type=int, default= 32)
    parser.add_argument("--stock_pool", type=str, default="ih", choices=["ih", "csi_300", "csi_500", "csi_1000", "start_up_100"])
    parser.add_argument("--stock_num", type=int, default=20)
    parser.add_argument("--selected_stock_num", type=int, default=5)
    parser.add_argument("--start_date", type=int, default= 20221202)
    parser.add_argument("--end_date", type=int, default= 20240102)
    parser.add_argument(
        "--evaluation_start_date",
        type=int,
        default=None,
        help="Start date used only for reported metrics; earlier dates remain available for online profile warm-up.",
    )
    parser.add_argument("--use_prev_stock", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_self_reflection", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use_macro_data", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_agent_distribution_modification", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--optimizer_look_back_window", type=int, default=5)
    parser.add_argument("--allow_possible_data_leakage", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use_adaptive_router", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use_heterogeneous_agent_pool", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--router_top_k", type=int, default=0)
    parser.add_argument("--router_min_k", type=int, default=1)
    parser.add_argument("--router_max_k", type=int, default=0)
    parser.add_argument("--router_diversity_lambda", type=float, default=0.35)
    parser.add_argument("--router_cost_lambda", type=float, default=0.05)
    parser.add_argument("--router_exploration_lambda", type=float, default=0.0)
    parser.add_argument("--router_family_diversity_lambda", type=float, default=0.0)
    parser.add_argument("--use_disagreement_diagnosis", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--abstention_threshold", type=float, default=0.45)
    parser.add_argument("--profile_maturity_days", type=int, default=15)
    parser.add_argument("--profile_label_column", type=str, default="1_15_labelB")
    parser.add_argument("--use_contextual_aggregator", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--aggregator_performance_lambda", type=float, default=1.0)
    parser.add_argument("--aggregator_confidence_lambda", type=float, default=0.6)
    parser.add_argument("--aggregator_evidence_lambda", type=float, default=0.15)
    parser.add_argument("--aggregator_agreement_lambda", type=float, default=0.4)
    parser.add_argument("--aggregator_cost_lambda", type=float, default=0.05)
    parser.add_argument("--use_conditional_debate", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max_debate_agents_per_stock", type=int, default=2)
    parser.add_argument("--max_debate_stocks_per_date", type=int, default=5)
    parser.add_argument("--max_agent_calls_per_date", type=int, default=0)
    parser.add_argument("--max_debate_calls_per_date", type=int, default=0)
    parser.add_argument("--max_tokens_per_date", type=int, default=0)
    parser.add_argument("--max_latency_seconds_per_date", type=float, default=0.0)
    parser.add_argument("--max_monetary_cost_per_date", type=float, default=0.0)
    parser.add_argument("--signal_alpha", type=float, default=0.5)
    parser.add_argument("--signal_alpha_1d", type=float, default=None)
    parser.add_argument("--signal_alpha_5d", type=float, default=None)
    parser.add_argument("--signal_alpha_10d", type=float, default=None)
    parser.add_argument("--use_adaptive_signal_alpha", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--signal_alpha_grid_step", type=float, default=0.1)
    parser.add_argument("--signal_alpha_min_observations", type=int, default=20)
    parser.add_argument("--signal_alpha_lookback_observations", type=int, default=60)
    parser.add_argument("--signal_alpha_shrinkage", type=float, default=0.5)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--run_id", type=str, default="")

    args = parser.parse_args()
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)

    arguments = vars(args).copy()
    config_hash = hashlib.sha256(
        json.dumps(arguments, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:8]
    requested_run_id = args.run_id.strip()
    if requested_run_id:
        run_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", requested_run_id)
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_id = f"{timestamp}_{config_hash}"
    run_dir = result_path("runs", run_id)
    if run_dir.exists():
        raise FileExistsError(
            f"Run directory already exists: {run_dir}. Choose a new --run_id."
        )

    stock_pool_name = args.stock_pool
    assert isinstance(stock_pool_name, str)
    set_dataset_variant("csi300" if stock_pool_name == "csi_300" else None)
    stock_pool_file = data_path(f"{stock_pool_name}.parq")
    if stock_pool_file.exists():
        stock_pool = pd.read_parquet(stock_pool_file)
    elif stock_pool_name == "ih":
        stock_pool = pd.read_parquet(first_existing_data_path("ih_label.parq"))[["Stock", "Date"]].drop_duplicates()
    else:
        raise FileNotFoundError(f"Stock pool file is missing: {stock_pool_file}")
    industry = pd.read_parquet(first_existing_data_path("stock_basic_data.parq"))
    stock_pool = stock_pool.merge(industry[["Stock", "Name", "Industry"]], on=["Stock"], how="left")
    label_file = {
        "ih": "ih_label.parq",
        "csi_300": "csi_300_label.parq",
    }.get(stock_pool_name, "all_ashare_label.parq")
    stock_labels = pd.read_parquet(first_existing_data_path(label_file, "all_ashare_label.parq"))

    trainer = StockDisagreementTrainer(
        num_investor_type=args.num_investor_type,
        num_agents_per_investor=args.num_agents_per_investor,
        stock_selector_for_per_investor=args.selected_stock_num,
        stock_pool=stock_pool,
        stock_labels=stock_labels,
        stock_num = args.stock_num,
        start_date = args.start_date,
        end_date = args.end_date,
        use_prev_stock = args.use_prev_stock,
        use_self_reflection = args.use_self_reflection,
        use_macro_data = args.use_macro_data,
        use_agent_distribution_modification = args.use_agent_distribution_modification,
        look_back_window = 10,
        optimizer_look_back_window = args.optimizer_look_back_window,
        data_leakage = args.allow_possible_data_leakage,
        use_adaptive_router=args.use_adaptive_router,
        use_heterogeneous_agent_pool=args.use_heterogeneous_agent_pool,
        router_top_k=args.router_top_k,
        router_min_k=args.router_min_k,
        router_max_k=args.router_max_k,
        router_diversity_lambda=args.router_diversity_lambda,
        router_cost_lambda=args.router_cost_lambda,
        router_exploration_lambda=args.router_exploration_lambda,
        router_family_diversity_lambda=args.router_family_diversity_lambda,
        use_disagreement_diagnosis=args.use_disagreement_diagnosis,
        abstention_threshold=args.abstention_threshold,
        profile_maturity_days=args.profile_maturity_days,
        profile_label_column=args.profile_label_column,
        use_contextual_aggregator=args.use_contextual_aggregator,
        aggregator_performance_lambda=args.aggregator_performance_lambda,
        aggregator_confidence_lambda=args.aggregator_confidence_lambda,
        aggregator_evidence_lambda=args.aggregator_evidence_lambda,
        aggregator_agreement_lambda=args.aggregator_agreement_lambda,
        aggregator_cost_lambda=args.aggregator_cost_lambda,
        use_conditional_debate=args.use_conditional_debate,
        max_debate_agents_per_stock=args.max_debate_agents_per_stock,
        max_debate_stocks_per_date=args.max_debate_stocks_per_date,
        max_agent_calls_per_date=args.max_agent_calls_per_date,
        max_debate_calls_per_date=args.max_debate_calls_per_date,
        max_tokens_per_date=args.max_tokens_per_date,
        max_latency_seconds_per_date=args.max_latency_seconds_per_date,
        max_monetary_cost_per_date=args.max_monetary_cost_per_date,
        signal_alpha=args.signal_alpha,
        signal_alpha_1d=args.signal_alpha_1d,
        signal_alpha_5d=args.signal_alpha_5d,
        signal_alpha_10d=args.signal_alpha_10d,
        use_adaptive_signal_alpha=args.use_adaptive_signal_alpha,
        signal_alpha_grid_step=args.signal_alpha_grid_step,
        signal_alpha_min_observations=args.signal_alpha_min_observations,
        signal_alpha_lookback_observations=args.signal_alpha_lookback_observations,
        signal_alpha_shrinkage=args.signal_alpha_shrinkage,
        random_seed=args.random_seed,
    )
    res = trainer.run()
    run_dir.mkdir(parents=True, exist_ok=False)

    res.to_parquet(run_dir / "signals.parquet")
    with open(run_dir / "agent_distributions.pkl", "wb") as f:
        pkl.dump(trainer.agent_distributions, f)
    audit_files = {}
    for name, frame in trainer.audit_frames().items():
        if frame.empty and len(frame.columns) == 0:
            path = run_dir / f"{name}.json"
            path.write_text("[]", encoding="utf-8")
        else:
            path = run_dir / f"{name}.parquet"
            frame.to_parquet(path, index=False)
        audit_files[name] = path.name
    manifest = {
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "arguments": arguments,
        "configuration_hash": config_hash,
        "models": {
            "base_url": LLM_BASE_URL,
            "chat": CHAT_MODEL_NAME,
            "reasoning": REASONING_MODEL_NAME,
            "vision": VISION_MODEL_NAME,
        },
        "random_seed": args.random_seed,
        "python": platform.python_version(),
        "agent_modalities": [int(agent.modality) for agent in trainer.agents],
        "agent_types": [int(agent.agent_type) for agent in trainer.agents],
        "agent_investor_types": [int(agent.investor_type) for agent in trainer.agents],
        "agent_personas": [agent.persona for agent in trainer.agents],
        "cache": GLOBAL_RESPONSE_CACHE.stats(),
        "artifacts": {
            "signals": "signals.parquet",
            "agent_distributions": "agent_distributions.pkl",
            **audit_files,
        },
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"Run artifacts saved to: {run_dir}")
    calculate_ic_rankic(res, stock_labels, args.evaluation_start_date)
