# Issue #284 — VERDICT

**Decision: the chain-ALL predictive increment is not established; #277 is retained. The descriptive
rule is conditional (only in the low-L1-collision regime), chain is not structurally special, the
over-merging hypothesis is not supported, and 129 SIDs are underpowered by ≈3× for the observed
effect. A clearly labelled research candidate B and its exploratory predictions are frozen; no
Stage3 was started and no public infra was modified.**

## C1–C5

| C | evidence | verdict |
|---|---|---|
| C1 | 9/9 #283 artifacts hash-verified; 253/253 per-SID recompute vs JOIN_TABLE, 0 mismatches; baseline and both #283 candidates reproduced exactly (0.0029719/0.6666; 0.0026990/0.7665; 0.0028218/0.7156 with identical picks); 129 labels 120/9, 13/6; 2 new time-postordered labels with exact SID+run binding; R2/R3/#272 export versions separated | PASS |
| C2 | ALL decomposition and bounds; collinearity (ALL vs l1_pos_hit 0.984; ALL_norm 0.742); structure contrast (chain ≈ fork/merge/revchain); primary/secondary matched sets with CIs, LOO, strong-only, counterexamples; conditional bands with CIs; residual association 0.082 [−0.110, 0.266] | PASS |
| C3 | `cond_refine`/`p3_split` exact conditional metrics; no recall association (0.001/−0.028); over-merging not supported; three-item vs catalogue collision 0.883; candidates A/B/C/D frozen and executed; CIs, BH family tests, strict 120, 0.065 NOT_EVALUABLE | PASS |
| C4 | research candidate B frozen (alpha 0.01, coefficients incl. negative all1 −0.1125, domain, input hashes); 122 pending + 2 sensitivity + 9 recovered unlabelled SIDs; #277 and the 124-row file unchanged; power analysis (374 SIDs needed) | PASS |
| C5 | CONTRACT/HYPOTHESES/DIAGNOSIS; audits; per-SID chain table; matched pairs/counterexamples; OOF/CI; conditional data; frozen research formula; tests (5 pass); persistent logs; infra drift 0; commit+push | PASS |

## Hypotheses

| id | status | reason |
|---|---|---|
| H32, H34–H37 (inherited) | unchanged | #283 statuses stand |
| H33 (matched marginals: ALL associates with recall) | **REFUTED** (new evidence) | #283's marginal-matched 0.191 [0.045, 0.325] does not survive the stricter primary controls: 0.122 [−0.239, 0.451]; the earlier result was driven by the coarser controls |
| H38 (ALL survives marginals + six metrics + L2/collision/L3/occupancy) | **REFUTED** | primary matched CI covers 0 for all1 and ALL_norm |
| H39 (association conditional on L2/L3 refinement) | **REFUTED** | cond_refine vs recall 0.001 [−0.202, 0.205]; matched sign negative (n.s.); no over-merging evidence |
| H40 (few high-recall/weak-identity SIDs dominate) | **REFUTED** | leave-one-out ranges stay narrow (0.032–0.246 primary; 0.123–0.295 secondary); strong-only 0.067/0.169 |
| H41 (chain is structurally special) | **REFUTED** | chain 0.395 ≈ fork 0.388 / merge 0.401 / revchain 0.399 |
| H42 (any pre-registered candidate passes the gate) | **REFUTED** | B/D/selected CIs cover 0; C is significantly worse; all gate=false |
| H43 (129 SIDs underpowered) | **SUPPORTED** | A−B mean 2.73e−4 vs MDE 4.64e−4 at 129; 374 (cluster 353) needed for 80 % |
| H44 (new: raw ALL is a collinearity proxy of l1_pos_hit; full-cohort coefficient flips sign) | **SUPPORTED** | ρ=0.984; B coefficients +0.066 (l1_pos) and −0.1125 (all1) |
| H45 (new: the ALL–recall ranking is a low-collision-regime rule) | **SUPPORTED** | low `l1_neg` band 0.907 [0.779, 0.963]; high band 0.039 [−0.296, 0.400] |

## Knowledge increment

1. The #283 fixed-candidate gain is a collinearity artifact of a variable 0.984-correlated with the
   existing L1 positive rate; the full-cohort coefficient is negative, so the "new" axis is not
   identified beyond `l1_pos_hit`.
2. The only robust descriptive rule is conditional: ALL ranks recall within the low L1-negative /
   low-collision regime (ρ up to 0.91) and is uninformative in the high bands — a boundary for any
   future mechanism, not a universal claim.
3. Chain/revchain/fork/merge carry the same signal; no structure-specific geometric claim survives.
4. The over-merging hypothesis is not supported at the SID level: nearly all L1-ALL triples refine at
   L2 and the amount does not predict recall.
5. With the observed effect the cohort needs ≈374 independent SIDs for 80 % power; the correct
   statement is "underpowered at 129", not "impossible".

## Closure checklist (skill §6.4)

1. Production code implemented: yes — `common284.py`, `audit284.py`, `chain284.py`, `match284.py`,
   `model284.py`, `freeze284.py`, tests; all executed with logs/exit codes.
2. Required tests executed: yes — 5 deterministic tests pass (frozen bounds, conditional hand case,
   raw-unit conversion, matching predicate, feature identities) plus every assertion in the runs
   (253/253 recompute, #283 reproduction, gate computations).
3. Required training completed: **N/A** — analysis-only issue; no Stage2/Stage3.
4. Infra boundary respected: yes — FROZEN; start/close snapshots added/removed/changed = 0; assets
   referenced by sha256, nothing rebuilt; only research-local files added.
5. Executable unfinished work: none in scope.
