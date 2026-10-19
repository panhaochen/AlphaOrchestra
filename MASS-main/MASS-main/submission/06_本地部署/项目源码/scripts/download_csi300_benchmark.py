from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


FIELDS = "date,code,open,high,low,close,preclose,volume,amount,pctChg"


def _rows(result) -> pd.DataFrame:
    rows: list[list[str]] = []
    while result.error_code == "0" and result.next():
        rows.append(result.get_row_data())
    if result.error_code != "0":
        raise RuntimeError(f"BaoStock error {result.error_code}: {result.error_msg}")
    return pd.DataFrame(rows, columns=result.fields)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the free CSI 300 benchmark from BaoStock.")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2023-12-29")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("stock_disagreement/dataset/csi300/benchmark_000300.parquet"),
    )
    args = parser.parse_args()

    import baostock as bs

    login = bs.login()
    if login.error_code != "0":
        raise RuntimeError(f"BaoStock login failed: {login.error_msg}")
    try:
        result = bs.query_history_k_data_plus(
            "sh.000300", FIELDS, start_date=args.start, end_date=args.end,
            frequency="d", adjustflag="3"
        )
        frame = _rows(result)
    finally:
        bs.logout()

    if frame.empty:
        raise RuntimeError("BaoStock returned no CSI 300 benchmark observations.")
    frame = frame.rename(
        columns={
            "date": "Date", "code": "VendorCode", "open": "Open", "high": "High",
            "low": "Low", "close": "Close", "preclose": "PreClose", "volume": "Volume",
            "amount": "Value", "pctChg": "PctChg",
        }
    )
    frame["Date"] = frame["Date"].str.replace("-", "", regex=False).astype(int)
    for column in ("Open", "High", "Low", "Close", "PreClose", "Volume", "Value", "PctChg"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["Return"] = frame["Close"].pct_change(fill_method=None)
    frame = frame.sort_values("Date").drop_duplicates("Date")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and args.output.stat().st_size > 8:
        frame = pd.concat([pd.read_parquet(args.output), frame], ignore_index=True)
        frame = frame.sort_values("Date").drop_duplicates("Date", keep="last")
        frame["Return"] = frame["Close"].pct_change(fill_method=None)
    frame.to_parquet(args.output, index=False)
    metadata = {
        "source": "BaoStock",
        "symbol": "sh.000300",
        "requested_start": args.start,
        "requested_end": args.end,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "adjustment": "index level, no adjustment (adjustflag=3)",
        "rows": int(len(frame)),
        "first_date": int(frame["Date"].min()),
        "last_date": int(frame["Date"].max()),
    }
    args.output.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(f"Saved benchmark to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
