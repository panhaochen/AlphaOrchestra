from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BacktestConfig:
    initial_capital: float = 10_000_000.0
    top_n: int = 20
    rebalance_every: int = 1
    commission_bps: float = 3.0
    slippage_bps: float = 5.0
    sell_stamp_duty_bps: float = 10.0
    impact_bps: float = 10.0
    max_participation_rate: float = 0.05
    max_position_weight: float = 0.10
    min_signal: float | None = None
    industry_balanced: bool = False
    min_value_quantile: float = 0.0
    residualize_signal: bool = False
    annualization: int = 252


@dataclass
class BacktestResult:
    daily: pd.DataFrame
    trades: pd.DataFrame
    metrics: dict[str, float]
    config: dict[str, Any]


def run_long_only_backtest(
    signals: pd.DataFrame,
    prices: pd.DataFrame,
    signal_column: str = "Signal_5d",
    evaluation_start_date: int | None = None,
    evaluation_end_date: int | None = None,
    config: BacktestConfig | None = None,
    market_exposure: pd.DataFrame | None = None,
) -> BacktestResult:
    """Backtest point-in-time signals with execution at the next trading-day open."""
    config = config or BacktestConfig()
    required_signal = {"Stock", "Date", signal_column}
    required_price = {"Stock", "Date", "Open", "Close"}
    if not required_signal.issubset(signals.columns):
        raise ValueError(f"Signals are missing columns: {sorted(required_signal - set(signals.columns))}")
    if not required_price.issubset(prices.columns):
        raise ValueError(f"Prices are missing columns: {sorted(required_price - set(prices.columns))}")
    if config.initial_capital <= 0 or config.top_n <= 0 or config.rebalance_every <= 0:
        raise ValueError("initial_capital, top_n, and rebalance_every must be positive")

    signal_columns = ["Stock", "Date", signal_column]
    if "Industry" in signals:
        signal_columns.append("Industry")
    sig = signals[signal_columns].copy()
    px_columns = [column for column in ["Stock", "Date", "Open", "Close", "Value"] if column in prices]
    px = prices[px_columns].copy()
    sig["Date"] = sig["Date"].astype(int)
    px["Date"] = px["Date"].astype(int)
    sig["Stock"] = sig["Stock"].astype(str).str.zfill(6)
    px["Stock"] = px["Stock"].astype(str).str.zfill(6)
    sig[signal_column] = pd.to_numeric(sig[signal_column], errors="coerce")
    for column in ["Open", "Close", "Value"]:
        if column in px:
            px[column] = pd.to_numeric(px[column], errors="coerce")
    sig = sig.dropna(subset=[signal_column]).sort_values(["Date", "Stock"])
    px = px.dropna(subset=["Open", "Close"]).sort_values(["Date", "Stock"])
    # Never carry the final portfolio beyond the period for which signals
    # actually exist.  This matters when the local price dataset is updated
    # after an older saved run was produced.
    signal_end_date = int(sig["Date"].max())
    later_price_dates = px.loc[px["Date"] > signal_end_date, "Date"]
    natural_end_date = (
        int(later_price_dates.min()) if not later_price_dates.empty else signal_end_date
    )
    backtest_end_date = min(
        natural_end_date,
        int(evaluation_end_date) if evaluation_end_date is not None else natural_end_date,
    )
    px = px[px["Date"] <= backtest_end_date]
    if not 0.0 <= config.min_value_quantile < 1.0:
        raise ValueError("min_value_quantile must be in [0, 1)")
    if config.min_value_quantile > 0 and "Value" in px:
        sig = sig.merge(px[["Stock", "Date", "Value"]], on=["Stock", "Date"], how="left")
    dates = sorted(px["Date"].unique().tolist())
    if len(dates) < 2:
        raise ValueError("At least two price dates are required")

    next_date = {dates[index]: dates[index + 1] for index in range(len(dates) - 1)}
    exposure_map: dict[int, float] = {}
    if market_exposure is not None:
        required_exposure = {"Date", "TargetExposure"}
        if not required_exposure.issubset(market_exposure.columns):
            raise ValueError("market_exposure requires Date and TargetExposure columns")
        exposure_frame = market_exposure[["Date", "TargetExposure"]].copy()
        exposure_frame["Date"] = exposure_frame["Date"].astype(int)
        exposure_frame["TargetExposure"] = pd.to_numeric(exposure_frame["TargetExposure"], errors="coerce")
        exposure_map = dict(zip(exposure_frame["Date"], exposure_frame["TargetExposure"].clip(0.0, 1.0)))
    targets: dict[int, dict[str, float]] = {}
    signal_dates = sorted(sig["Date"].unique().tolist())
    rebalance_dates = signal_dates[:: config.rebalance_every]
    for date in rebalance_dates:
        if date not in next_date:
            continue
        frame = sig[sig["Date"] == date].dropna(subset=[signal_column])
        if config.min_value_quantile > 0 and "Value" in frame:
            cutoff = frame["Value"].quantile(config.min_value_quantile)
            frame = frame[frame["Value"] >= cutoff]
        if config.min_signal is not None:
            frame = frame[frame[signal_column] >= config.min_signal]
        selection_signal = signal_column
        if config.residualize_signal:
            frame = _residualize_cross_section(frame, signal_column)
            selection_signal = "ResidualSignal"
        if config.industry_balanced and "Industry" in frame:
            frame = _industry_balanced_selection(frame, selection_signal, config.top_n)
        else:
            frame = frame.sort_values([selection_signal, "Stock"], ascending=[False, True]).head(config.top_n)
        count = len(frame)
        if count == 0:
            targets[next_date[date]] = {}
            continue
        raw_weights = (
            frame["IndustryWeight"].to_numpy(dtype=float)
            if config.industry_balanced and "IndustryWeight" in frame
            else np.full(count, 1.0 / count)
        )
        target_exposure = float(exposure_map.get(int(date), 1.0))
        raw_weights = np.minimum(raw_weights * target_exposure, config.max_position_weight)
        targets[next_date[date]] = dict(zip(frame["Stock"], raw_weights))

    price_by_date = {int(date): frame.set_index("Stock") for date, frame in px.groupby("Date", sort=True)}
    cash = float(config.initial_capital)
    shares: dict[str, float] = {}
    last_close: dict[str, float] = {}
    daily_records: list[dict[str, Any]] = []
    trade_records: list[dict[str, Any]] = []
    previous_equity = float(config.initial_capital)

    for date in dates:
        day = price_by_date[int(date)]
        open_map = day["Open"].to_dict()
        close_map = day["Close"].to_dict()
        value_map = day["Value"].to_dict() if "Value" in day else {}
        mark_open = lambda stock: float(open_map.get(stock, last_close.get(stock, 0.0)))
        pre_trade_equity = cash + sum(quantity * mark_open(stock) for stock, quantity in shares.items())
        day_cost = 0.0
        day_notional = 0.0

        if int(date) in targets and pre_trade_equity > 0:
            target_weights = targets[int(date)]
            all_stocks = sorted(set(shares) | set(target_weights))
            desired = {
                stock: target_weights.get(stock, 0.0) * pre_trade_equity
                for stock in all_stocks
            }
            deltas = {
                stock: desired[stock] - shares.get(stock, 0.0) * mark_open(stock)
                for stock in all_stocks
                if mark_open(stock) > 0
            }
            ordered = sorted(deltas, key=lambda stock: (deltas[stock] > 0, stock))
            for stock in ordered:
                price = mark_open(stock)
                requested = float(deltas[stock])
                traded_value = float(value_map.get(stock, np.inf))
                capacity = (
                    max(traded_value, 0.0) * config.max_participation_rate
                    if np.isfinite(traded_value)
                    else abs(requested)
                )
                notional = float(np.sign(requested) * min(abs(requested), capacity))
                if notional == 0:
                    continue
                participation = abs(notional) / traded_value if np.isfinite(traded_value) and traded_value > 0 else 0.0
                variable_bps = config.commission_bps + config.slippage_bps
                variable_bps += config.impact_bps * np.sqrt(max(participation, 0.0))
                if notional < 0:
                    variable_bps += config.sell_stamp_duty_bps
                cost = abs(notional) * variable_bps / 10_000.0
                if notional > 0 and notional + cost > cash:
                    rate = variable_bps / 10_000.0
                    notional = max(cash / (1.0 + rate), 0.0)
                    cost = notional * rate
                if notional == 0:
                    continue
                shares[stock] = shares.get(stock, 0.0) + notional / price
                if abs(shares[stock]) < 1e-12:
                    shares.pop(stock, None)
                cash -= notional + cost
                day_cost += cost
                day_notional += abs(notional)
                trade_records.append({
                    "Date": int(date),
                    "Stock": stock,
                    "side": "buy" if notional > 0 else "sell",
                    "price": price,
                    "notional": abs(notional),
                    "cost": cost,
                    "participation": participation,
                })

        last_close.update({stock: float(value) for stock, value in close_map.items() if np.isfinite(value)})
        equity = cash + sum(quantity * float(last_close.get(stock, 0.0)) for stock, quantity in shares.items())
        daily_return = equity / previous_equity - 1.0 if previous_equity > 0 else 0.0
        gross_exposure = sum(
            abs(quantity * float(last_close.get(stock, 0.0))) for stock, quantity in shares.items()
        ) / equity if equity > 0 else 0.0
        daily_records.append({
            "Date": int(date),
            "equity": equity,
            "cash": cash,
            "return": daily_return,
            "turnover": day_notional / (2.0 * pre_trade_equity) if pre_trade_equity > 0 else 0.0,
            "transaction_cost": day_cost,
            "gross_exposure": gross_exposure,
            "positions": len(shares),
            "rebalanced": int(date) in targets,
        })
        previous_equity = equity

    daily = pd.DataFrame(daily_records)
    if evaluation_start_date is not None:
        daily = daily[daily["Date"] >= int(evaluation_start_date)].reset_index(drop=True)
    metrics = summarize_backtest(daily, config)
    return BacktestResult(
        daily=daily,
        trades=pd.DataFrame(trade_records),
        metrics=metrics,
        config=asdict(config),
    )


def _industry_balanced_selection(frame: pd.DataFrame, signal_column: str, top_n: int) -> pd.DataFrame:
    """Select stocks using point-in-time equal-universe industry quotas."""
    work = frame.copy()
    work["Industry"] = work["Industry"].fillna("Unknown").astype(str)
    counts = work.groupby("Industry")["Stock"].count().sort_index()
    selection_count = min(top_n, len(work))
    exact = counts / counts.sum() * selection_count
    quotas = np.floor(exact).astype(int)
    remaining = selection_count - int(quotas.sum())
    for industry in (exact - quotas).sort_values(ascending=False, kind="mergesort").index[:remaining]:
        quotas.loc[industry] += 1
    selected: list[pd.DataFrame] = []
    shares = counts / counts.sum()
    for industry, quota in quotas.items():
        if quota <= 0:
            continue
        part = work[work["Industry"] == industry].sort_values(
            [signal_column, "Stock"], ascending=[False, True]
        ).head(int(quota)).copy()
        if not part.empty:
            part["IndustryWeight"] = float(shares.loc[industry]) / len(part)
            selected.append(part)
    if not selected:
        return work.head(0)
    result = pd.concat(selected, ignore_index=True)
    total = result["IndustryWeight"].sum()
    if total > 0:
        result["IndustryWeight"] /= total
    return result


def _residualize_cross_section(frame: pd.DataFrame, signal_column: str) -> pd.DataFrame:
    """Remove same-day industry and log-liquidity exposures without future information."""
    work = frame.copy()
    y = pd.to_numeric(work[signal_column], errors="coerce")
    design_parts = [pd.Series(1.0, index=work.index, name="Intercept")]
    if "Industry" in work:
        industry = work["Industry"].fillna("Unknown").astype(str)
        design_parts.append(pd.get_dummies(industry, prefix="Industry", drop_first=True, dtype=float))
    if "Value" in work:
        value = pd.to_numeric(work["Value"], errors="coerce").clip(lower=1.0)
        log_value = np.log(value)
        log_value = log_value.fillna(log_value.median())
        scale = log_value.std(ddof=0)
        if np.isfinite(scale) and scale > 0:
            design_parts.append(((log_value - log_value.mean()) / scale).rename("LogValue"))
    design = pd.concat(design_parts, axis=1).astype(float)
    valid = y.notna() & np.isfinite(design).all(axis=1)
    work["ResidualSignal"] = y
    if valid.sum() > design.shape[1]:
        beta, *_ = np.linalg.lstsq(design.loc[valid].to_numpy(), y.loc[valid].to_numpy(), rcond=None)
        work.loc[valid, "ResidualSignal"] = y.loc[valid] - design.loc[valid].to_numpy() @ beta
    return work


def summarize_backtest(daily: pd.DataFrame, config: BacktestConfig) -> dict[str, float]:
    if daily.empty:
        return {}
    returns = daily["return"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    total_return = float((1.0 + returns).prod() - 1.0)
    years = max(len(returns) / config.annualization, 1.0 / config.annualization)
    annualized_return = float((1.0 + total_return) ** (1.0 / years) - 1.0) if total_return > -1 else -1.0
    annualized_volatility = float(returns.std(ddof=1) * np.sqrt(config.annualization)) if len(returns) > 1 else 0.0
    sharpe = annualized_return / annualized_volatility if annualized_volatility > 0 else 0.0
    curve = (1.0 + returns).cumprod()
    drawdown = curve / curve.cummax() - 1.0
    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe": float(sharpe),
        "max_drawdown": float(drawdown.min()),
        "positive_day_rate": float((returns > 0).mean()),
        "average_daily_turnover": float(daily["turnover"].mean()),
        "total_transaction_cost": float(daily["transaction_cost"].sum()),
        "average_gross_exposure": float(daily["gross_exposure"].mean()),
        "trading_days": float(len(daily)),
    }
