# 2024 Frozen Blind-Test Protocol

This protocol was frozen before any 2024 model output was generated.

- Universe: point-in-time CSI 300 constituents.
- Warm-up: 2023-10-09 through 2023-12-29.
- Blind evaluation: 2024-01-02 through 2024-12-31.
- Random seed: 42.
- Candidate agents: four original investor types.
- Active agents per date: three, selected by the adaptive sparse router.
- Agent calls per date: maximum three.
- Disagreement diagnosis and contextual aggregation: enabled.
- Conditional debate and adaptive signal alpha: disabled.
- Primary signal: 10-day signal with fixed alpha 0.7.
- Portfolio: top 20, rebalanced every 10 trading days.
- Controls: bottom 20% liquidity exclusion, point-in-time industry balancing, industry/log-liquidity residualization.
- Market regime: 100% exposure above the causal 60-day CSI 300 moving average; 50% below it.
- Execution: next trading-day open.
- Costs: commission 3 bps, slippage 5 bps, sell stamp duty 10 bps, square-root impact 10 bps, 5% participation cap.
- Primary commercial metrics: total return, CSI 300 active return, exposure-matched active return, information ratio, drawdown, turnover and transaction cost.

The 2024 period must not be used for parameter changes after the run. Any subsequent configuration is a new development experiment and requires a later untouched holdout.
