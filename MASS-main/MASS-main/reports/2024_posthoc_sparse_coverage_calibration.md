# 2024 post-hoc sparse-coverage calibration

This document records a post-hoc diagnosis of the failed 2024 frozen blind run. It is not a blind result.

## Root cause

The routed agents produced non-zero signals for approximately 9-15 stocks per day (mean 14.27), while the original portfolio requested 20 holdings and imposed industry quotas. Consequently, stocks that had not been evaluated by an agent and had a zero signal entered the portfolio through quota filling and deterministic tie breaking.

## Frozen correction selected for the next test

- Signal: `Signal_10d`
- Eligibility: strictly positive signal (`min_signal = 1e-12`)
- Holdings: at most 15
- Rebalance interval: 10 trading days
- Industry balancing: disabled
- Signal residualization: disabled
- Market-regime exposure control: disabled
- Trading costs and liquidity constraints: unchanged

## 2024 calibration result

- Strategy total return: 25.21%
- CSI 300 total return: 14.68%
- Active total return: 9.18%
- Information ratio: 1.01
- Sharpe ratio: 1.06
- Maximum drawdown: -18.76%
- Total simulated transaction cost: CNY 574,971

Artifacts: `stock_disagreement/res/runs/csi300_2024_frozen_blind_seed42/backtests/calibrated_sparse_coverage_2024`

The correction is frozen before inspecting any 2025 outcome. The 2025 test must be reported regardless of result.
