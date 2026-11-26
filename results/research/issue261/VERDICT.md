# Issue #261 verdict

## Acceptance table

| criterion | requirement | observed | evidence | verdict |
|---|---|---|---|---|
| C1 | 8/8 four-piece bundles actually checked | 8/8 checkpoint/SID/archive/export/R2-recall hashes PASS; candidate source/config PASS; 3000/3000 epochs + train_end PASS; prediction rows 57,439 PASS | `INVENTORY.json` | **PASS** (100–102 historical infra receipt drift separately documented; candidate files unchanged) |
| C2 | frozen metric contract before ranking | `METRIC_CONTRACT.md` fixed before correlation ranking; formulas, populations, NA rules and tolerances specified | `METRIC_CONTRACT.md` | **PASS** |
| C3 | unified A–F metrics | SID, behavior, graph relations, roles, quant/reconstruction, full 28,391,948-triple hard/continuous curvature, heldout, loss/gate/token, downstream metrics computed per arm; unavailable submetrics listed with evidence | `METRICS.csv/.json`, `FULL_METRICS.csv/.json`, `TOKEN_GATE.json`, `MISSING.csv` | **PASS with documented NA** |
| C4 | correlations and sensitivity | Pearson/Spearman, n, direction, CI/p, leave-one-out, exclude-107 and exclude-100 | `CORRELATIONS.csv/.json`, `SENSITIVITY.csv` | **PASS** |
| C5 | mechanism chain + R06b counterexample | adjacent-stage associations reported; R06b code movement vs no recall gain analyzed with competing explanations; no causal overclaim | `REPORT.md` | **PASS** |
| C6 | no drift/no training | `INFRA_SNAPSHOT_START/END` identical; no Stage2/Stage3/TIGER run; training count 0; candidate/source and R2 recall hashes checked | `INFRA_SNAPSHOT_*.json`, `PROGRESS.json`, `INVENTORY.json` | **PASS** (historical pre-existing public-infra drift for 100–102 is recorded, not introduced by this issue) |
| C7 | reproducible delivery | issue scripts/tests, persisted execution logs, commit/push and remote SHA | `REPORT.md`, `tests/`, `checks/`; commit `8af30c14`, remote `main` equal | **PASS** |

## Headline result

The eight completed arms are all below the 0.065 adoption line:
`test R@10` 0.046432–0.052734. In exploratory n=8 associations with test R@10,
the strongest SID/behaviour signals are L1 positive prefix hit (+0.856) and L1
hard ER (+0.839); L2 distinct prefixes is −0.571 and L3 collision is −0.193.
The full-table hard-q curvature micro-MAE has Pearson −0.724 and reconstruction
L2 error −0.885, but these are architecture/capacity-confounded. The strongest
heldout structural association is +0.958, but it is not an independent causal
test. L1 positive remains +0.923 excluding 107 and +0.851 excluding 100; the L2
distinct relation becomes −0.893 when 107 is excluded, so it is not robust on its
own.

R06b/107 is the specific counterexample: the same-checkpoint token switch moved
52.19% of L2 codes and per-role fractions 0.413–0.575, yet test R@10 was 0.049600
vs R05 0.050349. Large code movement is not sufficient for recommendation gain.

## Knowledge increment

- New measurements: the complete eight-arm matrix, full-table 3-expert
  curvature/reconstruction, and direct GNN-token/gate statistics.
- Replication: the earlier preliminary L1 value was n=7; the complete eight-arm
  value is +0.856. The preliminary L2-prefix and collision signs replicate, but
  their evidence is weaker than the L1 signal.
- Negative knowledge: no evidence that L2/L3 structural movement alone improves
  downstream recommendation; R06b is an explicit counterexample.
- Not established: causal direction, the missing validation-stage metrics, and
  predicted-vs-target curvature correlation (the frozen evaluator does not emit
  those vectors; see `MISSING.csv`).

## Close check

1. Production code implemented: **YES** (analysis scripts only; no mechanism or framework changes).
2. Necessary tests executed: **YES** (`pytest results/research/issue261/tests -q`: 6 passed).
3. Required training completed: **YES—none required**, and training count is 0.
4. Infra boundary respected: **YES—FROZEN**, start/end snapshot identical; historical infra receipt drift in 100–102 is documented in `INFRA_FAULTS.md`.
5. Executable work remains: **NO** within this issue's analysis scope; the unresolved items are explicitly unavailable measurements, not skipped executable checks.
