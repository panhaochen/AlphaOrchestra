# Scalable expert-pool acceptance specification

AlphaOrchestra supports a large heterogeneous candidate pool while activating only a bounded subset each day.

## Reference commercial configuration

- Candidate experts: 64 (`16 investor families x 4 independent replicas`)
- Available scale-up: 128, 256, or 512 by increasing replicas
- Heterogeneity: persona, data modality combination, investor family, and independent agent identity
- Daily active range: 8-24 experts
- Active count: `ceil(8 + market_complexity * (24 - 8))`
- Selection objective: historical utility + exploration - modality overlap - family duplication - realized cost
- Hard daily call ceiling: 24 agent calls, plus separately budgeted debate calls

## Audit evidence

Every run records the following in `routing.parquet`:

- all candidate agents, not only selected agents;
- selected flag and agent identity;
- candidate, planned-active, and actual-active counts;
- market complexity;
- historical RankIC, exploration bonus, estimated cost, and realized normalized cost.

The run manifest records all candidate modalities, investor families, and personas. This makes sparse activation and cost savings independently auditable.

## Smoke-test configuration

Use a short date range before any long experiment. The intended arguments are:

`--num_investor_type 16 --num_agents_per_investor 4 --use_heterogeneous_agent_pool --use_adaptive_router --router_top_k 0 --router_min_k 8 --router_max_k 24 --max_agent_calls_per_date 24`

Conditional debate remains separately bounded through `max_debate_calls_per_date`, `max_debate_agents_per_stock`, and `max_debate_stocks_per_date`.
