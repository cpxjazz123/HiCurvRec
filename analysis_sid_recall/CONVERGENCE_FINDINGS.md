# The curvature estimate converges; the recall association does not

Two questions, two answers. The estimator is measurable at a small, fixed
budget. The association between the standardised curvature and the recall is
**reproducible at one budget and not stable across budgets**, so it cannot be
claimed.

## Sets and budgets

* **Admitted pairs**: 110 (the directory-name rule, `sid_recall.csv`). The live
  features and `curvature/report.md` now use this set.
* **Convergence ladder**: 73 arms, started before the admission rule changed to
  the name rule. It measures the *estimator*, which does not depend on which
  pairs are admitted. Nested sample: one seeded order of each quotient's edges
  per (arm, level), reduced to its first 8/16/32/64/128 entries; null = 1–8
  size-preserving label shuffles, 32 edges each; 1,152 edge solves per arm.
* **Replication check**: the same 110 pairs measured twice at 32 edges and 3
  null replicates, with independent per-row seeds.

## 1. The estimator converges by 64 edges and 4 replicates

| level | mean @64 | mean @128 | \|Δmean\|/SE | estimator SE | sd across arms | signal/noise |
|---|---|---|---|---|---|---|
| L1 | +0.4317 | +0.4317 | 0.02 | 0.0042 | 0.0359 | 8.59 |
| L2 | −0.1564 | −0.1566 | 0.01 | 0.0079 | 0.1258 | 15.89 |
| L3 | −0.2434 | −0.2431 | 0.02 | 0.0077 | 0.0823 | 10.67 |

The shift between the 64-edge and the 128-edge estimate is within 0.02 of the
estimator's own standard error at every level, and the across-arm spread is
8.6–15.9 times that error: differences **between arms** are resolvable. The null
mean and standard deviation move by at most 1.1% between 4 and 8 replicates
(L1 0.4959 ± 0.0260, L2 −0.1480 ± 0.1430, L3 −0.2442 ± 0.0866). A measurement
with 64 edges and 4 replicates is therefore sound.

## 2. The association reproduces at a fixed budget…

The same 110 pairs, two independent measurements at 32 edges / 3 replicates:

| feature | measurement 1 | measurement 2 |
|---|---|---|
| `z_curv_L1` | −0.120 (p 0.238) | −0.074 (p 0.462) |
| `z_curv_L2` | **+0.204 (p 0.039)** | **+0.201 (p 0.038)** |
| `z_curv_L3` | +0.103 (p 0.288) | +0.179 (p 0.063) |
| profile (`L1−L3`) | −0.121 (p 0.225) | −0.091 (p 0.364) |

Only the second level reaches p ≈ 0.04, and it does so twice with the same
coefficient.

## 3. …and moves with the budget and the null count

The same arms, the same observed sample (the 128-edge mean), only the number of
null replicates changed — the replicates are nested, so the 3-replicate value
uses the first three of the eight:

| replicates | L1 | L2 | L3 |
|---|---|---|---|
| 2 | +0.061 (p 0.612) | −0.136 (p 0.252) | −0.094 (p 0.415) |
| 3 | +0.103 (p 0.392) | −0.206 (p 0.086) | −0.031 (p 0.804) |
| 4 | +0.132 (p 0.283) | **−0.263 (p 0.028)** | −0.074 (p 0.533) |
| 6 | +0.155 (p 0.206) | **−0.250 (p 0.026)** | −0.082 (p 0.497) |
| 8 | +0.154 (p 0.203) | −0.185 (p 0.112) | −0.104 (p 0.376) |

For the second level the coefficient runs from −0.14 to −0.26 and p from 0.25 to
0.026 **on the same arms and the same observed sample**, and its sign is the
opposite of the +0.20 measured at 32 edges on the admitted set. The level-1
coefficient drifts the other way (+0.06 → +0.15). The significance is a property
of the replicate budget, not of the data.

## 4. What was not run

The 110-arm ladder was stopped at 16:03: the process had 8:49 of wall time and
8 s of CPU, `/proc/<pid>/wchan` was `cl_sync_io_wait`, and its blocking syscall
was `read(3)` on the interpreter's TensorFlow `.pyc` on `/fs04` (Lustre). Every
run of this environment imports TensorFlow at start-up, and that import stalls
on the network filesystem; this is the same 3–9 minute start-up cost seen in the
other runs. The ladder's conclusion (the estimator converges) is a property of
the estimator, so the 73-arm run stands; no 110-arm ladder number is reported.

## Verdict

1. **The measurement is trustworthy** at 64 edges and 4 null replicates.
2. **The association is not**: it reproduces at 32 edges / 3 replicates, but
   changes sign and crosses p = 0.05 as the replicate count or the edge budget
   changes, so no direction can be claimed.
3. Making the level-2 coefficient a real finding would require a pre-registered
   fixed budget (64–128 edges, ≥8 replicates), a pre-declared one-sided
   hypothesis, and an independent set to confirm on — not another adjustment of
   the budget on this table.

Artefacts: `curvature/convergence/convergence.md` (ladder, 73 arms),
`per_arm_convergence.json` (every arm's ladder and null replicates),
`curvature/curvature_summary.json` (the 110-pair measurement),
`curvature/report.md`, `PAIR_TIERS.md`.
