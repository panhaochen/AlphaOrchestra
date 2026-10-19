# 2026 January--September frozen extension result

## Frozen signal evaluation

The system was run from 2025-10-09 through 2026-09-30, using 2025 Q4 for profile warm-up and 2026-01-05 onward for evaluation. Architecture, routing bounds, signal alpha, horizon, and portfolio rules were unchanged after the Q1 confirmation.

| Horizon | IC | ICIR | RankIC | RankICIR | Valid days |
|---|---:|---:|---:|---:|---:|
| 1 day | 0.005460 | 0.057325 | 0.001365 | 0.014783 | 180 |
| 5 day | 0.006100 | 0.068250 | 0.008275 | 0.091652 | 176 |
| 10 day (primary) | **0.006567** | **0.082269** | **0.010463** | **0.122469** | 171 |

All three horizons remained positive over the extended window. The strongest result remained the predeclared 10-day horizon, although its strength was lower than in the standalone Q1 run.

## Frozen 10-day portfolio

| Period | Strategy return | Benchmark return | Active return | Sharpe | Information ratio | Max drawdown |
|---|---:|---:|---:|---:|---:|---:|
| 2026 Jan--Sep | -4.680% | -5.882% | **+1.277%** | -0.367 | 0.096 | -15.029% |
| Q1 segment | +0.595% | -3.885% | **+4.662%** | 0.148 | 1.381 | -4.582% |
| Q2 segment | -4.102% | +11.896% | **-14.298%** | -0.927 | -4.388 | -13.182% |
| Q3 segment | -1.190% | -12.488% | **+12.910%** | -0.261 | 2.785 | -10.201% |

The extended system preserved positive cross-sectional signal quality and modest full-period benchmark-relative value, but did not produce positive absolute return. Regime dependence is material: it strongly outperformed in falling-market Q1/Q3 and substantially underperformed in rising-market Q2.

## Reproducibility caveat

The Q1 overlap in this longer run was not identical to the earlier standalone Q1 run. Overlapping `Signal_10d` values had correlation of approximately 0.678, and router selections began diverging on the fourth warm-up trading day. Consequently, the earlier standalone Q1 return and the extended-run Q1 segment must not be presented as interchangeable replications. The longer, single continuous run is the authoritative result for January--September reporting.

This run-length sensitivity must be resolved before production deployment. Commercial claims should use the full continuous-run results and disclose the Q2 failure rather than relying only on the stronger standalone Q1 result.

## Artifact locations

- Run: `stock_disagreement/res/runs/csi300_64candidate_frozen_2026_extended_seed42`
- Full-period backtest: `backtests/frozen_2026_extended_portfolio`
- Segment backtests: `backtests/frozen_2026_q1_segment`, `frozen_2026_q2_segment`, and `frozen_2026_q3_segment`
