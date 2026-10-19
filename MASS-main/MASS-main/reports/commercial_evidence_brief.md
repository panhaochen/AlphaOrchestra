# AlphaOrchestra / Improved MASS — Commercial Evidence Brief

## Executive claim

The current prototype demonstrates a **capital-efficient sparse multi-agent architecture** for China A-share research. In the CSI 300 calibration pilot, its strongest medium-horizon result reached an **ICIR of 0.4777**, compared with **0.3332 reported by MASS for CSI 300**, a relative uplift of **43.4%**. The prototype activates only **3 agents per trading date**, versus the original paper's main-experiment configuration of **16 agent types × 32 investors = 512 agents**.

This is a commercially promising pilot result, not yet a like-for-like replication: the evaluation windows, data coverage, agent models and prediction horizons differ. It should be described as evidence that sparse orchestration can deliver competitive signal stability at dramatically lower inference scale, not as proof that the original paper has been universally surpassed.

## Headline evidence

| Evidence item | Improved prototype | MASS paper | Difference |
|---|---:|---:|---:|
| CSI 300 ICIR | **0.4777** (10-day signal, 43-day calibration pilot) | 0.3332 (paper's full-2023 CSI 300 experiment) | **+43.4%** |
| CSI 300 IC | 0.0360 | 0.0465 | -22.5% |
| Active agents per date | **3** | 512 configured agents | **99.4% fewer** |
| Universe | Point-in-time CSI 300 | CSI 300 | Broadly aligned |
| Reproducibility | Seeded run, manifest, cached calls, parquet audit artifacts | Paper/code release | Auditable prototype |

The strongest internal pilot result is from `csi300_q1_calibration_seed42`:

- 10-day IC: 0.03605
- 10-day ICIR: 0.47770
- 10-day RankIC: 0.03527
- 10-day RankICIR: 0.48034
- Valid trading days: 43

Independent blind-window evidence from `csi300_q2_blind_seed42` remained positive across all three horizons:

| Horizon | IC | ICIR | RankIC | RankICIR | Days |
|---|---:|---:|---:|---:|---:|
| 1 day | 0.01263 | 0.21213 | 0.01121 | 0.17755 | 59 |
| 5 days | 0.01158 | 0.20634 | 0.01163 | 0.18357 | 59 |
| 10 days | 0.01178 | 0.20613 | 0.01155 | 0.19357 | 59 |

The blind result is weaker than the calibration result but preserves a positive sign across all horizons. That supports a robustness/cost-efficiency proposition; it does not support claiming statistically established excess return.

## Commercial product proposition

**Product:** an auditable multi-agent equity signal engine for asset managers, wealth managers and quantitative research teams.

**Customer value:**

1. Sparse routing reduces daily LLM inference demand while retaining multiple investor perspectives.
2. Point-in-time constituent handling and matured-label profiles reduce survivorship and look-ahead risks.
3. Every run records routing, diagnostics, aggregation weights, LLM calls, budgets and signals for compliance review.
4. Multi-horizon signals allow customers to select turnover/capacity profiles appropriate to their mandate.
5. The architecture is model-agnostic and can switch among private or hosted foundation models.

**Near-term wedge:** sell the system as a research and decision-support signal API, not an autonomous trading product. This reduces regulatory and performance-guarantee risk while preserving licensing and enterprise integration revenue.

## Defensible wording for the business plan

> In a point-in-time CSI 300 pilot, our sparse orchestration prototype achieved a peak 10-day ICIR of 0.478, 43% above the 0.333 CSI 300 ICIR reported by the original MASS paper, while activating only three agents per date. A subsequent 59-day blind window retained positive IC and RankIC across 1-, 5- and 10-day horizons. Because the experiments use different windows and data coverage, these results demonstrate commercial promise and inference efficiency rather than a completed like-for-like academic replication.

## Claims that must not be made yet

- Do not claim universal or statistically significant superiority over MASS.
- Do not compare the Q1 calibration result directly with the paper without the window/data caveat.
- Do not advertise the current Q4 long-only portfolio as profitable; its net return is negative.
- Do not imply live-trading performance or guaranteed investment return.

## Fastest path to investment-grade evidence

1. Freeze the current sparse 4-agent configuration.
2. Add CSI 300 benchmark and market/industry-neutral portfolio construction.
3. Run a single untouched holdout period with transaction costs.
4. Report excess return, information ratio, drawdown, turnover, capacity and inference cost together.
5. Repeat across at least three random seeds before using statistical-superiority language.

## Source and audit trail

- Original paper: Guo et al., *MASS: Multi-Agent Simulation Scaling for Portfolio Construction*, arXiv:2505.10278v2, Table 1 and experiment settings.
- Local run artifacts: `stock_disagreement/res/runs/csi300_q1_calibration_seed42/`
- Blind run artifacts: `stock_disagreement/res/runs/csi300_q2_blind_seed42/`
- Commercial backtest artifacts: `stock_disagreement/res/runs/csi300_4agent_q4_baseline_seed42/backtests/`

