from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def safe_number(value):
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    return value


def describe(series: pd.Series) -> dict[str, float | int | None]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return {"count": 0, "mean": None, "min": None, "median": None, "p95": None, "max": None}
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "min": float(values.min()),
        "median": float(values.median()),
        "p95": float(values.quantile(0.95)),
        "max": float(values.max()),
    }


def summarize_run(run_dir: Path, evaluation_start: int | None = None) -> dict:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    routing = pd.read_parquet(run_dir / "routing.parquet") if (run_dir / "routing.parquet").exists() else pd.DataFrame()
    calls = pd.read_parquet(run_dir / "llm_calls.parquet") if (run_dir / "llm_calls.parquet").exists() else pd.DataFrame()
    debates = pd.read_parquet(run_dir / "debates.parquet") if (run_dir / "debates.parquet").exists() else pd.DataFrame()
    diagnostics = pd.read_parquet(run_dir / "diagnostics.parquet") if (run_dir / "diagnostics.parquet").exists() else pd.DataFrame()

    if evaluation_start is not None:
        for frame in (routing, calls, debates, diagnostics):
            if not frame.empty and "date" in frame:
                frame.drop(frame[frame["date"].astype(int) < int(evaluation_start)].index, inplace=True)

    output = {
        "run_id": manifest["run_id"],
        "configuration_hash": manifest.get("configuration_hash"),
        "arguments": manifest.get("arguments", {}),
        "models": manifest.get("models", {}),
        "cache": manifest.get("cache", {}),
    }
    if not routing.empty:
        by_date = routing.sort_values(["date", "agent_index"]).groupby("date", as_index=False).first()
        selected_by_date = routing.groupby("date")["selected"].sum()
        output["routing"] = {
            "trading_days": int(by_date["date"].nunique()),
            "candidate_agents": int(routing["candidate_count"].max()),
            "actual_active_agents": describe(by_date["actual_active_count"]),
            "selected_rows_per_date": describe(selected_by_date),
            "market_complexity": describe(by_date["market_complexity"]),
            "activation_fraction_mean": float(
                by_date["actual_active_count"].sum() / by_date["candidate_count"].sum()
            ),
        }
    if not calls.empty:
        success = calls["success"].fillna(False).astype(bool) if "success" in calls else pd.Series(dtype=bool)
        output["llm_calls"] = {
            "records": int(len(calls)),
            "successful_records": int(success.sum()),
            "success_rate": float(success.mean()) if len(success) else None,
            "request_count": float(pd.to_numeric(calls.get("llm_request_count"), errors="coerce").fillna(0).sum()),
            "cached_request_count": float(pd.to_numeric(calls.get("cached_request_count"), errors="coerce").fillna(0).sum()),
            "total_tokens_including_cache": float(pd.to_numeric(calls.get("total_tokens"), errors="coerce").fillna(0).sum()),
            "billed_total_tokens": float(pd.to_numeric(calls.get("billed_total_tokens"), errors="coerce").fillna(0).sum()),
            "reported_monetary_cost": float(pd.to_numeric(calls.get("monetary_cost"), errors="coerce").fillna(0).sum()),
            "note": "Reported monetary cost is not used as evidence when the gateway returns zero.",
        }
    if not debates.empty:
        pre = pd.to_numeric(debates.get("pre_severity"), errors="coerce")
        post = pd.to_numeric(debates.get("post_severity"), errors="coerce")
        output["debates"] = {
            "records": int(len(debates)),
            "dates_triggered": int(debates["date"].nunique()),
            "stocks_debated": int(debates[["date", "stock"]].drop_duplicates().shape[0]),
            "successful_records": int(debates["success"].fillna(False).astype(bool).sum()),
            "mean_pre_severity": safe_number(pre.mean()),
            "mean_post_severity": safe_number(post.mean()),
            "mean_severity_reduction": safe_number((pre - post).mean()),
        }
    if not diagnostics.empty:
        output["diagnostics"] = {
            "records": int(len(diagnostics)),
            "action_counts": {
                str(key): int(value)
                for key, value in diagnostics["recommended_action"].value_counts(dropna=False).items()
            },
            "severity": describe(diagnostics["severity"]),
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser("Build a reproducible submission evidence snapshot from saved runs.")
    parser.add_argument("--runs_root", default="stock_disagreement/res/runs")
    parser.add_argument("--output", default="reports/submission_evidence_snapshot.json")
    args = parser.parse_args()

    root = Path(args.runs_root)
    specs = [
        ("q2_enhanced", "csi300_64candidate_dynamic_q2_blind_seed42", 20240401),
        ("q2_control", "csi300_4agent_original_q2_control_seed42", 20240401),
        ("2026_causal_extended", "csi300_64candidate_frozen_2026_extended_causal_seed42", 20260105),
    ]
    payload = {
        "status": "frozen_submission_evidence_snapshot",
        "runs": {
            label: summarize_run(root / run_id, evaluation_start)
            for label, run_id, evaluation_start in specs
        },
        "claims": {
            "q2_signal_comparison_source": "reports/q2_blind_64_vs_4_acceptance.md",
            "prefix_invariance": "PASSED for prefix_invariance_short_v3_seed42 vs prefix_invariance_long_v3_seed42",
            "important_boundary": "The Q2 control is the repository's four-agent original-style implementation, not a reproduced 512-agent paper run.",
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"Saved evidence snapshot to: {output.resolve()}")


if __name__ == "__main__":
    main()
