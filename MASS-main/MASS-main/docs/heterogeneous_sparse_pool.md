# Heterogeneous sparse agent pool

`--use_heterogeneous_agent_pool` enables the scalable sparse-routing variant of MASS.

When enabled:

- every agent receives a stable, unique `agent_id` used by routing, online performance profiles, cost profiles, and aggregation;
- `investor_type` remains available as the broader behavioral family;
- agents cycle through distinct available modality combinations before combinations repeat;
- each investor family receives a behavioral mandate, and replicas receive an explicit independent-reasoning instruction;
- manifests record agent IDs, investor families, modalities, and personas;
- routing traces record both the unique agent identity and its investor family.

The flag defaults to disabled so historical runs and the original type-level aggregation remain reproducible. Scaling experiments with more than one agent per investor type should enable it; otherwise replicas of the same type continue to be pooled as in the legacy implementation.

Before a long experiment, run a short smoke test and verify:

1. `manifest.json` contains unique `agent_types` and non-empty `agent_personas`.
2. `routing.parquet` contains the requested number of selected unique agents per date.
3. `aggregation.parquet` contains separate rows for the selected agent IDs.
4. `budgets.parquet` contains no overruns.
