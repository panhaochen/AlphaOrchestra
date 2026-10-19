# 2025 frozen sparse-coverage protocol

Frozen before inspecting 2025 model outcomes.

## Signal-generation period

- Warm-up/calibration history: 2024-01-02 through 2024-12-31
- Blind evaluation: 2025-01-02 through 2025-12-31
- Universe: point-in-time CSI 300
- Random seed: 42
- Candidate agents: 4
- Routed agents per date: 3
- Adaptive router, disagreement diagnosis, and contextual aggregator enabled
- Conditional debate and adaptive signal alpha disabled

## Portfolio rule

- Signal: `Signal_10d`
- Only stocks with strictly positive non-zero agent coverage are eligible
- Maximum holdings: 15
- Rebalance every 10 trading days
- No industry quota, residualization, or market-timing overlay
- Existing next-open execution, liquidity limits, commissions, slippage, impact, and stamp duty remain enabled

## Governance

- The configuration must not be changed after the first 2025 result is observed.
- The result must be retained and reported whether positive or negative.
- 2026 remains untouched as the final recent-period confirmation set.
