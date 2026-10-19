# 2026 Q1 frozen confirmation protocol

Protocol recorded before execution on 2026-10-09.

- Warm-up/profile period: 2025-10-09 through 2025-12-31.
- Confirmation period: 2026-01-05 through 2026-03-31.
- Candidate architecture: 64 heterogeneous agents (16 families, four replicas).
- Dynamic activation bounds: 8--24 agents per day.
- Enabled modules: adaptive sparse router, disagreement diagnosis, contextual aggregation, and conditional debate.
- Frozen signal parameters: `alpha_1d=alpha_5d=alpha_10d=1.0`; adaptive alpha disabled.
- Primary signal endpoint: 10-day RankIC and RankICIR.
- Frozen portfolio endpoint: `Signal_10d`, top 15 positive signals, rebalance every 10 trading days, with no post-result parameter selection.
- Comparator evidence remains the previously completed four-agent Q2 control. No 2026 parameter changes are permitted after viewing confirmation results.

Acceptance interpretation:

- Strong confirmation: positive 10-day RankIC and positive active portfolio return.
- Partial confirmation: positive 10-day RankIC but non-positive active return, or the reverse.
- Failed confirmation: non-positive 10-day RankIC and non-positive active return.

The result must be reported regardless of direction.
