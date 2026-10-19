# 2025 frozen sparse-coverage result

The frozen 2025 result is retained regardless of outcome.

## Result

- Strategy total return: 8.25%
- CSI 300 total return: 17.66%
- Active total return: -8.00%
- Information ratio: -0.77
- Sharpe ratio: 0.46
- Maximum drawdown: -13.33%
- Simulated transaction cost: CNY 644,339
- Trading days: 243

All evaluated signal horizons had negative IC and RankIC in 2025. The frozen sparse-coverage correction therefore did not generalize from its 2024 calibration result.

## Diagnosis

- Agent 0 and agent 1 had materially negative 10-day rank IC in 2025.
- Agent 2 and agent 3 were approximately neutral rather than reliably positive.
- The adaptive router selected comparatively better agents, but the candidate pool did not contain sufficiently strong experts.
- Static and trailing-IC-weighted quantitative factor composites were also tested post hoc and did not beat the 2025 CSI 300 benchmark.

## Governance status

This is a failed frozen validation, not a commercial alpha proof. It supports the engineering value of routing, caching, audit artifacts, and deterministic portfolio construction, but it does not support a claim of persistent excess return.

During post-hoc factor analysis, aggregate 2026 factor IC statistics were inadvertently computed. Consequently, 2026 can no longer be described as a fully untouched blind factor-validation set. Any later 2026 result must be labelled partially contaminated or confirmatory rather than strictly blind.
