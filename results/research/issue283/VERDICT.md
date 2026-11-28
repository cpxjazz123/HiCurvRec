# Issue #283 — VERDICT

**Decision: negative on the pre-registered predictive gate; descriptive positives recorded; #277 kept
unchanged; no new formula, no new predictions, no Stage3.** The joint triplet assignment is the unique
non-marginal degree of freedom (ALL), it is measurably aligned at k=1 (1.4–2.0× the permutation null)
and weakly associated with recall under matched marginals (+0.191 [0.045, 0.325]), but no joint
feature passes the paired SID-cluster bootstrap criterion at 129 SIDs.

## C1–C5

| C | evidence | verdict |
|---|---|---|
| C1 | 7 deterministic tests; partitions sum to support; only 12 allowed transitions; rename invariance; 22 feasible LCP triples; equal-marginal hand example D1/D2 | PASS |
| C2 | 28,391,948 rows / all valid; canonical class recomputation 0 mismatches; 36 role classes + Orbit; coverage per centre/class; #282 negative audit (level 2 = 1.35 %); analytic permutation + degree-stratified nulls verified 316/320 within 3σ (`PERM_CHECK.json`) | PASS |
| C3 | full per-SID/class/level table (`JOIN_TABLE` 14.9 MB → two gzip parts, sha256 in `STATS_MANIFEST.json`), LCP + transitions + macro/micro; only ALL is non-marginal; ALL above null at k=1, below at k=2/3; matched-marginal chain 509 pairs +0.191 [0.045, 0.325]; 146 matched-SID contrasts with cluster bootstrap; FDR residual associations; #274 R2 `NOT_AVAILABLE` with reason | PASS |
| C4 | baseline #277 reproduced exactly 0.0029719/0.6666; 129 rows/5 folds; 120/9, 13/6, 0 rows >0.065; inner-fold-only selection (all1_chain/am_mb_chain); paired CI [−1.57e−4, +4.57e−4] → **fails**; fixed all1_chain [−1.06e−4, +6.35e−4] → fails; #277 retained | PASS (executed; criterion negative) |
| C5 | CONTRACT/HYPOTHESES/DIAGNOSIS; tests; stats + manifests; coverage/negative/null audits; matched comparison/counterexamples; OOF/CI/ablation/residual; persistent commands/logs/exit codes; REPORT/VERDICT; infra drift 0; commit+push | PASS |

## Hypotheses

| id | status | reason |
|---|---|---|
| H28 (report type-conditioned) | INCONCLUSIVE (inherited) | unchanged from #282 |
| H29 (no type info beyond overall rate) | SUPPORTED (inherited) | unchanged from #282 |
| H30 (type contrast adds increment) | REFUTED (inherited) | unchanged from #282 |
| H31 (support/collinearity limit) | SUPPORTED (inherited) | unchanged from #282 |
| H32 (ALL is the unique non-marginal dof, measurable) | **SUPPORTED** | algebra + exact counts; chain k=1 ALL 0.063 vs null 0.045; transitions/LCP invariants |
| H33 (matched marginals: ALL associates with recall) | **SUPPORTED (weak)** | 509 marginal-matched chain pairs, ρ=+0.191 [0.045, 0.325], sign agreement 0.56 |
| H34 (pre-frozen joint feature passes the OOF criterion) | **REFUTED** | selected CI [−1.57e−4, +4.57e−4]; fixed all1_chain [−1.06e−4, +6.35e−4] |
| H35 (ALL lift concentrated in high-recall SIDs) | **SUPPORTED (descriptive)** | all1_chain_lift 0.0221 high vs 0.0135 low (ρ=0.465); all3 0.510 |
| H36 (redundant with marginals or support-limited → null) | **INCONCLUSIVE** | redundancy refuted (ALL above null, not marginal-determined) and support ample (6.3M rows); the predictive null comes from resolution at 129 SIDs, not from redundancy/support |
| H37 (new: population-2 ALL re-measures l1_pos_hit) | **SUPPORTED** | validation ALL1 vs `l1_pos_hit` ρ=0.980/0.981 across 253 SIDs |

## Knowledge increment

1. The joint 5-partition is exactly one-parameter beyond the pairwise marginals (ALL); AM/MB/AB cannot
   be claimed as independent information.
2. The empirical SID preserves the triple intersection at the coarse prefix (k=1: 1.4–2.0× null,
   triangles ≈1.9×) and depletes it at k=2/3.
3. The best joint candidate is the **raw** ALL at k=1 (fixed arm: MAE −9.1 %, ρ +0.10); subtracting the
   SID-specific null (the "lift") significantly hurts — the null is not the right normaliser.
4. The population-2 joint ALL is 0.98-correlated with the existing `l1_pos_hit`: no new independent
   signal from the validation-context construction.
5. No joint feature survives the pre-registered CI at 129 SIDs; the direction is not closed as
   "impossible" — it is unresolved at this label budget (the tested set X_tested is not the whole
   allowed space; nothing here proves non-existence).

## Closure checklist (skill §6.4)

1. Production code implemented: yes — `joint283.py`, `rows283.py`, `stats283.py`, `diag283.py`,
   `diag283b.py`, `match283.py`, `model283.py`, `perm_check283.py`, `finalize283.py` all executed.
2. Required tests executed: yes — 7 deterministic tests pass; every SID/class/level assertion ran
   (partitions, transitions, LCP); null verification ran (320 tests).
3. Required training completed: **N/A** — analysis-only issue (§范围: no Stage2/Stage3).
4. Infra boundary respected: **yes** — FROZEN; `INFRA_SNAPSHOT_START` vs `INFRA_SNAPSHOT_CLOSE`
   added/removed/changed = 0; assets referenced by sha256, no asset rebuilt; only research-local
   scripts and derived files added (raw tables excluded by `.gitignore`, gzip evidence committed).
5. Executable unfinished work: none in scope.
