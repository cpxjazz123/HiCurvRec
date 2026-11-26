# Issue #261 run log — eight-bundle Stage2–Stage3 correlation audit

## Identity

- issue: `#261`; title: `Analysis only: full eight-bundle GNN/curvature Stage2–Stage3 correlation audit (R2 100–107)`
- body sha256: `022ad858552a4d067b9a0e15abdff5d8f99e58404ad3d5574efe31cd59c38e9d`
- issue state at freeze: **OPEN**
- `INFRA_MODE`: **FROZEN** — the issue explicitly says analysis only, no new training, no new mechanism, reuses public Infra; it authorises only issue-specific read-only analysis scripts, tests and reports.
- training required: **NO**. The issue's close condition is analysis evidence, and explicitly forbids Stage2/Stage3/TIGER retraining. Training count must remain **0**.

## Frozen acceptance checklist

| id | issue requirement | measurable criterion | data / output |
|---|---|---|---|
| C1 | 8/8 four-piece bundles actually checked | for each 100–107: source tree, checkpoint, SID, recall/test record exist and hashes match recorded binding/ledger | `INVENTORY.json`; no ledger-only pass |
| C2 | freeze metric math and data scope before ranking | `METRIC_CONTRACT.md` exists with formulas, 24,587 SID rows, 57,439 evaluation rows, NA rules | file sha256 |
| C3 | unified per-arm metrics A–F | every listed metric computed or NA with a concrete evidence-backed reason; all 8 arms retained | `METRICS.csv` / `.json` |
| C4 | correlations and sensitivity | Pearson/Spearman, n, direction, absolute ranking, leave-one-out, exclude-107, exclude-100 five-expert; no point deletion | `CORRELATIONS.csv` / `.json`, `SENSITIVITY.csv` |
| C5 | mechanism-chain and R06b counterexample | adjacent-stage associations and competing explanations in report; no causal overclaim | `REPORT.md`, `VERDICT.md` |
| C6 | no drift / no training | all eight source/checkpoint/SID/Stage3 records unchanged; no new Stage2/Stage3 run | hashes in `INVENTORY.json`, `PROGRESS.json` |
| C7 | reproducible delivery | issue-specific scripts + deterministic tests; persisted commands/logs; committed and pushed | `checks/`, `REPORT.md`, commit SHA |

## Scope

Only read-only analysis of the eight completed R2 bundles and frozen assets. No mechanism edit, no training, no new public Infra. Missing values are `NA` with reason, never zero or imputed.

## Initial data checks

- Local immutable Stage2 workspaces and result directories exist for all eight arms.
- Local Stage3 `test_final.json` and prediction shards exist for all eight arms.
- R2 ledger has entries 100–107 with hashes; local files will be compared against them, not treated as proof by path alone.
- Stage2 shared layer records exist for all eight SIDs under `results/stage2_RQ-VAE/<arm>/stage2_metrics/`.

## Timeline

- 2026-11-26: issue frozen, INFRA_MODE FROZEN, analysis scope established.
