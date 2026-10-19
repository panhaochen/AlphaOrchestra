from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


K_FIELDS = (
    "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,"
    "tradestatus,pctChg,peTTM,pbMRQ,psTTM,pcfNcfTTM,isST"
)


def normalize_stock_code(code: str) -> str:
    value = str(code).strip().lower()
    if "." in value:
        parts = value.split(".")
        return next((part for part in parts if part.isdigit() and len(part) == 6), parts[-1])
    return value.zfill(6)


def expand_constituent_snapshots(snapshots: pd.DataFrame, trading_dates: pd.Series) -> pd.DataFrame:
    required = {"SnapshotDate", "Stock"}
    if not required.issubset(snapshots.columns):
        raise ValueError(f"Snapshots require columns {sorted(required)}")
    snaps = snapshots.copy()
    snaps["SnapshotDate"] = snaps["SnapshotDate"].astype(int)
    snaps["Stock"] = snaps["Stock"].map(normalize_stock_code)
    dates = pd.DataFrame({"Date": sorted(pd.Series(trading_dates).astype(int).unique())})
    snapshot_dates = pd.DataFrame({"SnapshotDate": sorted(snaps["SnapshotDate"].unique())})
    mapping = pd.merge_asof(
        dates.sort_values("Date"), snapshot_dates, left_on="Date", right_on="SnapshotDate", direction="backward"
    ).dropna(subset=["SnapshotDate"])
    mapping["SnapshotDate"] = mapping["SnapshotDate"].astype(int)
    return (
        mapping.merge(snaps, on="SnapshotDate", how="inner")
        .sort_values(["Date", "Stock"])
        .drop_duplicates(["Date", "Stock"])
        .reset_index(drop=True)
    )


def cross_sectional_label(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    valid = numeric.dropna()
    if len(valid) < 3:
        return pd.Series(np.nan, index=values.index)
    lower, upper = valid.quantile([0.01, 0.99])
    clipped = numeric.clip(lower, upper)
    std = clipped.std()
    if not np.isfinite(std) or std <= 0:
        return pd.Series(0.0, index=values.index)
    return (clipped - clipped.mean()) / std


def build_derived_tables(raw: pd.DataFrame, constituents: pd.DataFrame, industries: pd.DataFrame) -> dict[str, pd.DataFrame]:
    frame = raw.copy()
    frame["Stock"] = frame["Stock"].map(normalize_stock_code)
    frame["Date"] = frame["Date"].astype(int)
    frame = frame.sort_values(["Stock", "Date"]).drop_duplicates(["Stock", "Date"])
    numeric = ["Open", "High", "Low", "Close", "AdjClose", "Volume", "Value", "Turn", "PE_TTM", "PB"]
    for column in numeric:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    # Normalize legacy/two-request cache files to the same adjusted OHLC
    # convention used by the current single-request downloader.
    adjustment = frame["AdjClose"] / frame["Close"].replace(0, np.nan)
    for column in ("Open", "High", "Low", "Close"):
        frame[column] = frame[column] * adjustment
    group = frame.groupby("Stock", group_keys=False)
    frame["Return_1d"] = group["AdjClose"].pct_change(fill_method=None)
    frame["Ret_5"] = group["AdjClose"].pct_change(5, fill_method=None)
    frame["Ret_20"] = group["AdjClose"].pct_change(20, fill_method=None)
    frame["Vol_20"] = group["Return_1d"].rolling(20, min_periods=10).std().reset_index(level=0, drop=True)
    volume_mean = group["Volume"].rolling(20, min_periods=10).mean().reset_index(level=0, drop=True)
    frame["Volume_Ratio_20"] = frame["Volume"] / volume_mean.replace(0, np.nan)
    frame["EP"] = 1.0 / frame["PE_TTM"].replace(0, np.nan)
    frame["BP"] = 1.0 / frame["PB"].replace(0, np.nan)

    for horizon in (1, 5, 10):
        frame[f"Forward_{horizon}d"] = group["AdjClose"].shift(-horizon) / frame["AdjClose"] - 1.0
        frame[f"{horizon}_15_labelB"] = frame.groupby("Date")[f"Forward_{horizon}d"].transform(cross_sectional_label)

    pool = constituents[["Stock", "Date"]].copy()
    pool["Stock"] = pool["Stock"].map(normalize_stock_code)
    selected = pool.merge(frame, on=["Stock", "Date"], how="left", validate="one_to_one")
    selected = selected.merge(industries[["Stock", "Name", "Industry"]], on="Stock", how="left")
    selected["Industry"] = selected["Industry"].fillna("Unknown")
    selected["Name"] = selected["Name"].fillna(selected["Stock"])

    factor_sources = ["Ret_5", "Ret_20", "Vol_20", "Volume_Ratio_20", "EP", "BP"]
    for index, source in enumerate(factor_sources):
        selected[f"price_value_feature_{index}"] = selected.groupby("Date")[source].transform(cross_sectional_label)

    base = selected[["Stock", "Date", "Open", "High", "Low", "Close", "Volume", "Value", "PE_TTM", "PB"]].copy()
    base["ADJ"] = selected["AdjClose"]
    base["Mean"] = selected["AdjClose"]
    base["TMean"] = selected["AdjClose"]
    base["Close_Avg"] = selected["AdjClose"]
    base["MV"] = np.nan
    base["FREE_MV"] = np.nan
    base["SHARE"] = np.nan
    base["alpha"] = selected["Return_1d"]
    labels = selected[["Stock", "Date", "1_15_labelB", "5_15_labelB", "10_15_labelB"]].copy()
    labels["Time"] = labels["Date"]
    features = selected[["Stock", "Date", *[f"price_value_feature_{i}" for i in range(6)]]].copy()
    industry_ret = (
        selected.groupby(["Date", "Industry"], as_index=False)["Return_1d"]
        .mean()
        .rename(columns={"Return_1d": "Daily_Return"})
    )
    return {
        "pool": selected[["Stock", "Date"]],
        "base": base,
        "labels": labels,
        "features": features,
        "industries": industries[["Stock", "Name", "Industry"]].drop_duplicates("Stock"),
        "industry_ret": industry_ret,
    }


def _result_frame(result) -> pd.DataFrame:
    rows = []
    while result.error_code == "0" and result.next():
        rows.append(result.get_row_data())
    if result.error_code != "0":
        raise RuntimeError(f"BaoStock error {result.error_code}: {result.error_msg}")
    return pd.DataFrame(rows, columns=result.fields)


def download_snapshot(bs, date: str) -> pd.DataFrame:
    try:
        result = bs.query_hs300_stocks(date=date)
    except TypeError:
        result = bs.query_hs300_stocks(date)
    frame = _result_frame(result)
    if frame.empty:
        return frame
    code_col = "code" if "code" in frame else "code_name"
    name_col = "code_name" if "code_name" in frame else code_col
    return pd.DataFrame({"SnapshotDate": int(date.replace("-", "")), "Stock": frame[code_col].map(normalize_stock_code), "Name": frame[name_col]})


def download_history(bs, code: str, start: str, end: str) -> pd.DataFrame:
    prefix = "sh" if code.startswith(("6", "9")) else "sz"
    vendor_code = f"{prefix}.{code}"
    # Scheme B consistently uses forward-adjusted prices for signals and labels.
    # BaoStock returns volume, amount, and valuation fields in the same request,
    # so a second unadjusted-price download is unnecessary.
    raw = _result_frame(bs.query_history_k_data_plus(vendor_code, K_FIELDS, start, end, "d", "2"))
    if raw.empty:
        return raw
    raw["AdjClose"] = raw["close"]
    return raw.rename(columns={
        "date": "Date", "code": "VendorCode", "open": "Open", "high": "High", "low": "Low",
        "close": "Close", "volume": "Volume", "amount": "Value", "turn": "Turn", "peTTM": "PE_TTM", "pbMRQ": "PB",
    }).assign(Stock=code)


def download_industry(bs, code: str) -> pd.DataFrame:
    prefix = "sh" if code.startswith(("6", "9")) else "sz"
    frame = _result_frame(bs.query_stock_industry(code=f"{prefix}.{code}"))
    if frame.empty:
        return pd.DataFrame([{"Stock": code, "Name": code, "Industry": "Unknown", "IndustryDate": None}])
    row = frame.iloc[-1]
    return pd.DataFrame(
        [{
            "Stock": code,
            "Name": row.get("code_name", code),
            "Industry": row.get("industry") or "Unknown",
            "IndustryDate": row.get("updateDate"),
            "IndustryClassification": row.get("industryClassification"),
        }]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a free, point-in-time CSI 300 dataset with BaoStock.")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2023-12-29")
    parser.add_argument("--dataset-root", type=Path, default=Path("stock_disagreement/dataset"))
    parser.add_argument("--sleep", type=float, default=0.05)
    parser.add_argument("--build-only", action="store_true", help="Build from previously downloaded raw parquet files.")
    args = parser.parse_args()
    variant = args.dataset_root / "csi300"
    raw_dir = variant / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    snapshots_path = raw_dir / "constituent_snapshots.parquet"
    history_path = raw_dir / "daily_history.parquet"
    if not args.build_only:
        import baostock as bs
        login = bs.login()
        if login.error_code != "0":
            raise RuntimeError(f"BaoStock login failed: {login.error_msg}")
        try:
            start_ts = pd.Timestamp(args.start)
            previous_month_end = start_ts - pd.offsets.BMonthEnd(1)
            if previous_month_end >= start_ts:
                previous_month_end -= pd.offsets.BMonthEnd(1)
            month_ends = pd.DatetimeIndex([previous_month_end]).append(
                pd.date_range(args.start, args.end, freq="BME")
            ).unique().sort_values()
            snapshots = []
            for snapshot_number, date in enumerate(month_ends, start=1):
                for offset in range(0, 8):
                    candidate = date - pd.offsets.BDay(offset)
                    part = download_snapshot(bs, candidate.strftime("%Y-%m-%d"))
                    if not part.empty:
                        snapshots.append(part)
                        break
                time.sleep(args.sleep)
                print(f"Downloaded constituent snapshot {snapshot_number}/{len(month_ends)}", flush=True)
            new_snapshots = pd.concat(snapshots, ignore_index=True).drop_duplicates(["SnapshotDate", "Stock"])
            if snapshots_path.exists() and snapshots_path.stat().st_size > 8:
                old_snapshots = pd.read_parquet(snapshots_path)
                snapshot_frame = pd.concat([old_snapshots, new_snapshots], ignore_index=True)
            else:
                snapshot_frame = new_snapshots
            snapshot_frame = snapshot_frame.drop_duplicates(["SnapshotDate", "Stock"], keep="last")
            snapshot_frame.sort_values(["SnapshotDate", "Stock"]).to_parquet(snapshots_path, index=False)
            stock_dir = raw_dir / "stocks"
            stock_dir.mkdir(parents=True, exist_ok=True)
            codes = sorted(new_snapshots["Stock"].unique())
            for stock_number, code in enumerate(codes, start=1):
                stock_path = stock_dir / f"{code}.parquet"
                part = download_history(bs, code, args.start, args.end)
                if not part.empty:
                    if stock_path.exists() and stock_path.stat().st_size > 8:
                        part = pd.concat([pd.read_parquet(stock_path), part], ignore_index=True)
                    part = part.drop_duplicates("Date", keep="last").sort_values("Date")
                    part.to_parquet(stock_path, index=False)
                time.sleep(args.sleep)
                if stock_number == 1 or stock_number % 25 == 0 or stock_number == len(codes):
                    print(f"Downloaded/cached stock {stock_number}/{len(codes)}", flush=True)
            stock_files = sorted(stock_dir.glob("*.parquet"))
            if not stock_files:
                raise RuntimeError("No stock histories were downloaded.")
            pd.concat((pd.read_parquet(path) for path in stock_files), ignore_index=True).to_parquet(history_path, index=False)
            industry_path = raw_dir / "stock_industries.parquet"
            existing_industries = (
                pd.read_parquet(industry_path)
                if industry_path.exists() and industry_path.stat().st_size > 8
                else pd.DataFrame()
            )
            known_codes = set(existing_industries["Stock"].astype(str)) if not existing_industries.empty else set()
            missing_codes = [code for code in codes if code not in known_codes]
            if missing_codes:
                industry_rows = []
                for stock_number, code in enumerate(missing_codes, start=1):
                    industry_rows.append(download_industry(bs, code))
                    time.sleep(args.sleep)
                    if stock_number == 1 or stock_number % 50 == 0 or stock_number == len(missing_codes):
                        print(f"Downloaded industry {stock_number}/{len(missing_codes)}", flush=True)
                industries = pd.concat([existing_industries, *industry_rows], ignore_index=True)
                industries.drop_duplicates("Stock", keep="last").to_parquet(industry_path, index=False)
        finally:
            bs.logout()

    snapshots = pd.read_parquet(snapshots_path)
    history = pd.read_parquet(history_path)
    trading_dates = history["Date"].astype(str).str.replace("-", "", regex=False).astype(int)
    history["Date"] = trading_dates
    constituents = expand_constituent_snapshots(snapshots, trading_dates)
    counts = constituents.groupby("Date")["Stock"].nunique()
    if counts.empty or counts.min() < 250 or counts.max() > 350:
        raise ValueError(f"Implausible CSI 300 constituent counts: {counts.describe().to_dict()}")
    industry_path = raw_dir / "stock_industries.parquet"
    if industry_path.exists() and industry_path.stat().st_size > 8:
        names = pd.read_parquet(industry_path)[["Stock", "Name", "Industry"]]
    else:
        names = snapshots.sort_values("SnapshotDate").groupby("Stock", as_index=False).tail(1)[["Stock", "Name"]]
        names["Industry"] = "Unknown"
    tables = build_derived_tables(history, constituents, names)
    tables["pool"].to_parquet(args.dataset_root / "csi_300.parq", index=False)
    tables["labels"].to_parquet(args.dataset_root / "csi_300_label.parq", index=False)
    tables["base"].to_parquet(variant / "base_data.parq", index=False)
    tables["features"].to_parquet(variant / "price_feature.parq", index=False)
    tables["industries"].to_parquet(variant / "stock_basic_data.parq", index=False)
    tables["industry_ret"].to_parquet(variant / "industry_ret.parq", index=False)
    metadata = {
        "source": "BaoStock",
        "start": str(pd.to_datetime(history["Date"].astype(str)).min().date()),
        "end": str(pd.to_datetime(history["Date"].astype(str)).max().date()),
        "labels": "forward adjusted-close returns, 1%/99% winsorized and cross-sectionally standardized",
        "constituents": "latest successfully downloaded monthly snapshot not later than each trading date",
        "industry": "BaoStock static CSRC classification as downloaded; not point-in-time historical industry",
        "news_enabled": False,
    }
    (variant / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Built CSI 300 dataset under {variant}")


if __name__ == "__main__":
    main()
