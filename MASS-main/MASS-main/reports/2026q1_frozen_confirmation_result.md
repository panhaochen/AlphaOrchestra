# 2026 Q1 frozen confirmation result

## Status

**Strong confirmation**, under the acceptance rule recorded before execution in `reports/2026q1_frozen_confirmation_protocol.md`.

The run used the frozen 64-candidate architecture and parameters without post-result selection. The confirmation period was 2026-01-05 through 2026-03-31 (56 valid trading days), following a 2025 Q4 profile warm-up.

## Signal results

| Horizon | IC | ICIR | RankIC | RankICIR |
|---|---:|---:|---:|---:|
| 1 day | -0.006634 | -0.056886 | -0.011906 | -0.115013 |
| 5 day | -0.000797 | -0.007387 | 0.008562 | 0.091316 |
| 10 day (primary) | **0.015088** | **0.169251** | **0.028888** | **0.338750** |

The confirmation is horizon-specific. The system showed strong medium-horizon ranking quality at 10 days, weak positive rank performance at five days, and failed at one day.

## Predeclared portfolio result

Portfolio rule: `Signal_10d`, top 15 positive signals, rebalance every 10 trading days.

| Metric | Result |
|---|---:|
| Total return | **5.446%** |
| Annualized return | 26.953% |
| Annualized volatility | 18.060% |
| Sharpe ratio | **1.492** |
| Maximum drawdown | **-5.222%** |
| Positive-day rate | 53.571% |
| Average daily turnover | 9.079% |
| Transaction cost | 134,123 |
| CSI 300 total return | **-3.885%** |
| Active return | **9.709%** |
| Information ratio | **2.777** |

## Commercial interpretation

This independent frozen confirmation supports the commercial thesis that a large heterogeneous candidate pool with sparse activation, disagreement-directed debate, and contextual aggregation can generate useful medium-horizon cross-sectional rankings while avoiding full-pool inference cost. It also demonstrates positive absolute and benchmark-relative performance in a falling market during the specified confirmation quarter.

The result does not establish a guaranteed or production-grade annual return. The evaluation contains only 56 trading days, the one-day signal failed, and annualized statistics extrapolate a short observation window. Claims must therefore be limited to frozen quarterly confirmation and medium-horizon signal effectiveness.

## Artifacts

- Run: `stock_disagreement/res/runs/csi300_64candidate_frozen_2026q1_confirmation_seed42`
- Backtest: `backtests/frozen_2026q1_confirmation_portfolio`
