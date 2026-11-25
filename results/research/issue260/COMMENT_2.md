### #260 round-1 diagnosis: cost-only coupling is assignment-inert

Read-only probe: `results/research/issue260/diagnosis/r05_probe.json` (R05 checkpoint
`f93da1a7…`, SID `dc206dba…`, export matched exactly; no training). `DIAGNOSIS.md` and
`HYPOTHESES.json` updated.

**Measured switch effects on hard codes (same encoder/codebooks):**

| switch | L1 | L2 | L3 |
|---|---:|---:|---:|
| behavior off (role mixture → global) | 0.0000 | 0.00110 | 0.00110 |
| curvature off (all → Euclidean) | 0.01566 | 0.01688 | 0.01672 |
| gate uniform (topology gate removed) | 0.00500 | 0.00504 | 0.00496 |

**Root cause**: the quantiser reproduces the encoder almost exactly (per-level MSE
`1.91e-5 / 5.39e-6 / 2.32e-6`), so role/curvature weights on the distance cost almost never change the
nearest codeword. R05's role mixtures stayed near-uniform (entropy 1.508–1.584 vs max 1.585; weights
0.187–0.424), the level-3 gate collapsed (entropy 0.176, 99.59% one expert), and topology tokens add
only 0.026 accuracy over degree-only role classification (0.566 vs 0.540; majority 0.385). The three
level placements in R03–R05 were therefore testing an inert cost channel, not three different levers.

**Hypothesis statuses**: H1/H2/H3/H5/H6 `SUPPORTED` by measurement; H4 (single-run Stage3 variance)
`INCONCLUSIVE`; new H7 (role must partition the L2 candidate set inside the cost matrix) `UNTESTED`.

**Next**: preregistered local check of a role-partitioned L2 cost (role chooses a code subset; Sinkhorn/
floor unchanged). It must change L2 assignments and raise L2 prefix sharing without changing L1 before
any formal Stage2 launch. If the local prediction fails, no training is burned.
