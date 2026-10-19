# AlphaOrchestra Q2 blind acceptance: 64-candidate architecture vs 4-agent control

## Protocol

- Calibration window: 2024-01-02 to 2024-03-29.
- Blind evaluation window: 2024-04-01 to 2024-06-28 (59 valid days).
- Universe and sampling: CSI 300 data, 50-stock daily working universe, three stocks selected per active agent.
- Frozen signal rule: `alpha=1.0`, selected using the calibration window before observing Q2 results.
- Frozen portfolio rule: `Signal_10d`, top 15, rebalance every 10 trading days, positive signals only.
- Enhanced system: 64 heterogeneous candidates, dynamically activating 8--24 agents, disagreement diagnosis, conditional debate, and contextual aggregation.
- Control: four agents with heterogeneous pool, router, diagnosis, debate, and contextual aggregation disabled.

## Blind signal results

| Horizon | Metric | 64 candidates | 4-agent control | Difference |
|---|---:|---:|---:|---:|
| 1 day | IC | 0.008090 | 0.009355 | -0.001265 |
| 1 day | RankIC | 0.008869 | 0.008363 | +0.000506 |
| 5 day | IC | 0.014423 | 0.008339 | +0.006085 |
| 5 day | RankIC | 0.020645 | 0.009768 | +0.010877 |
| 5 day | RankICIR | 0.227861 | 0.141083 | +0.086779 |
| 10 day | IC | 0.011056 | 0.006926 | +0.004129 |
| 10 day | RankIC | 0.018666 | 0.009591 | +0.009075 |
| 10 day | RankICIR | 0.188458 | 0.122253 | +0.066205 |

The enhanced architecture improved 5-day RankIC by 111% and 10-day RankIC by 95% relative to the control. It did not improve 1-day linear IC, although 1-day RankIC was slightly higher.

## Frozen portfolio results

| Metric | 64 candidates | 4-agent control | Difference |
|---|---:|---:|---:|
| Total return | -1.487% | -1.854% | +0.367 pp |
| CSI 300 return | -2.144% | -2.144% | -- |
| Active return | +0.671% | +0.295% | +0.375 pp |
| Information ratio | 0.297 | 0.169 | +0.128 |
| Sharpe ratio | -0.392 | -0.509 | +0.117 |
| Maximum drawdown | -11.635% | -8.995% | -2.640 pp |
| Average daily turnover | 8.270% | 8.727% | -0.457 pp |
| Transaction cost | 134,475 | 136,899 | -2,424 |

## Acceptance conclusion

The Q2 blind test supports the claimed architectural increment at medium horizons: the 64-candidate sparse system produced materially higher 5-day and 10-day cross-sectional ranking quality than the four-agent control, and the predeclared 10-day portfolio achieved higher active return and information ratio with slightly lower turnover and transaction cost.

This is an initial commercial validation, not evidence of a production-ready absolute-return product. Absolute return and Sharpe remained negative during the blind quarter, and maximum drawdown was worse than the control. The defensible commercial claim is improved medium-horizon ranking and benchmark-relative performance under a sparse, auditable multi-agent inference budget.

## Artifact locations

- Enhanced run: `stock_disagreement/res/runs/csi300_64candidate_dynamic_q2_blind_seed42`
- Enhanced backtest: `backtests/frozen_64candidate_q2_portfolio`
- Control run: `stock_disagreement/res/runs/csi300_4agent_original_q2_control_seed42`
- Control backtest: `backtests/frozen_4agent_q2_control_portfolio`
