# Free CSI 300 dataset (Scheme B)

This pipeline builds a point-in-time CSI 300 dataset from BaoStock without claiming compatibility with the unpublished MASS labels.

## Build

Run from the repository root in the `mass` environment:

```powershell
python -m stock_disagreement.data.csi300_free `
  --start 2018-01-01 `
  --end 2023-12-29
```

Downloads are cached per stock under `stock_disagreement/dataset/csi300/raw/stocks`. Re-running resumes missing stocks. The pipeline uses one forward-adjusted daily request per stock and applies that price convention consistently to model inputs, features, and labels. To rebuild derived files without network access:

```powershell
python -m stock_disagreement.data.csi300_free --build-only
```

## Outputs

- `stock_disagreement/dataset/csi_300.parq`: daily point-in-time membership.
- `stock_disagreement/dataset/csi_300_label.parq`: custom 1/5/10-day labels.
- `stock_disagreement/dataset/csi300/base_data.parq`: OHLCV and valuation data.
- `stock_disagreement/dataset/csi300/price_feature.parq`: six reproducible price/value features.
- `stock_disagreement/dataset/csi300/stock_basic_data.parq`: names and declared industry classification.
- `stock_disagreement/dataset/csi300/industry_ret.parq`: daily industry returns.
- `stock_disagreement/dataset/csi300/metadata.json`: provenance and definitions.

The loader automatically prefers the `csi300` subdirectory when `--stock_pool csi_300` is selected. Existing IH experiments continue to use the root-level example files.

## Label definition

For each stock, forward adjusted-close returns are computed over 1, 5, and 10 trading days. On each prediction date, returns are winsorized at the 1st and 99th percentiles and standardized across the constituent cross-section. The prediction-date membership determines eligibility; future removal from the index does not invalidate a label.

## Known limitation

Version 1 uses BaoStock's static CSRC industry classification downloaded at build time. It is useful for cross-sectional grouping but is not point-in-time historical industry membership; do not claim a survivorship-free industry-neutral test. News is intentionally excluded. `MV`, `FREE_MV`, and `SHARE` remain missing rather than being fabricated; the currently exercised agent modalities do not consume them.
