# Issue #275 — VERDICT

**Success criterion (C7 of the issue): not met.** The search produced no formula that beats the
strongest control on out-of-fold MAE (it is significantly worse), and its ranking quality degrades.
The pre-registered rule therefore applies: **a stable, simple, generalising symbolic formula from
SID metrics / reconstruction error / effective rank / curvature features to Stage3 recall could not
be confirmed**; the contract forbids post-hoc fixes, so the analysis closes as an explicit negative.

## Evidence

* **Q_SID outer LOFO (n=40, 8 families)**: PySR MAE 0.003491 / Spearman +0.110 vs ridge 0.002497 /
  +0.600; paired family-bootstrap **Δ = +0.000994, CI [+0.000614, +0.001925]** → PySR significantly
  worse. Best single feature 0.004150 / +0.198; mean 0.003997 / −0.397; prior `l1_pos_hit` 0.003716 / +0.452.
* **Q_VEC ablation (n=16, 3 folds)**: PySR − median Δ = +0.000562, CI [−0.00086, +0.00103] → no vector
  increment from quantisation error + effective rank. No queue supports the "vector quality adds
  beyond SID" hypothesis.
* **Q_GEO**: 3 sha-matched SIDs → unidentifiable; no curvature/geometry claim is made.
* **Threshold criterion**: 0 labels ≥ 0.06 (bound range 0.0382–0.0573) → `NOT_EVALUABLE`.
* **Time split**: usable labels are single-month (2026-11) → temporal validation unavailable
  (report-only; no generalisation claim).
* **Expression stability**: the same input pair (`l1_hard_er`, `l1_pos_hit`) recurs in all 8 fold
  expressions, but the functional form flips between affine, reciprocal and multiplicative forms while
  the coefficients change by 1–2 orders of magnitude — no stable simple form. Independent SymPy
  recomputation reproduces PySR predictions exactly (max abs diff 0.0 on all 11 folds); Pareto fronts
  (5–8 points per fold) are stored in `ANALYSIS_RESULTS.json`.

## C1–C10

| id | requirement | evidence | verdict |
|---|---|---|---|
| C1 | auditable run inventory + identity gate | 239 records → 234 units; 62 bound (54 by record identity, 8 by R2 ledger); 172 legacy units counted and excluded from fitting; protocol mix recorded | PASS (with documented identity gap) |
| C2 | matched metrics only; re-check the 12 high-recall arms | 54 layer features recomputed by frozen evaluator; `l3_collision` equals the gate definition; the 12 high-recall arms are legacy/unbound (SID absent), not used | PASS |
| C3 | three queues, matched-only, no unbound record | Q_SID 40/54 · Q_VEC 16/26 · Q_GEO 3 (unidentifiable) | PASS |
| C4 | pre-registered contract, no post-hoc edits | `ANALYSIS_CONTRACT.md` frozen before the search; LOFO/inner CV/controls/CI/threshold all frozen; labels never used for screening | PASS |
| C5 | fixed PySR budget, documented version, determinism | PySR 0.19.4 / Julia 1.10.14, operators and limits frozen in `PYSR_ENV.md`, seed 42, rerun verified identical; expressions independently recomputed with SymPy | PASS |
| C6 | judged by the pre-registered criterion | PySR worse than strongest control: mean (+0.00099, CI excludes 0), ranking (−0.49 Spearman) | PASS (negative result) |
| C7 | compare with controls; decide increment | ridge control beats PySR and the prior; vector/geometry increments absent (CI covers 0 / n=3) | PASS |
| C8 | prospective predictions for queue arms | not applicable: no usable formula was confirmed; predicting with the rejected model would break the pre-registration | N/A (documented) |
| C9 | deliverables with reproduction | REPORT/VERDICT/PYSR_ENV/HYPOTHESES + scripts + JSONs + `tests/test_analysis.py` (4 checks, all pass) | PASS |
| C10 | no training, no Stage3, INFRA_MODE=FROZEN, no infra drift | no GPU job started by this issue; protected paths show **zero** files modified since the round start (12:30); three `tools/` files carry pre-existing modifications with mtimes 2026-11-25/27, i.e. from earlier sessions, not this round | PASS |

## Secondary criterion detail

| check | value |
|---|---|
| ΔMAE PySR vs controls | vs mean −0.00051 (better), vs median −0.00018 (better), vs ridge **+0.00099 (worse, CI excludes 0)** |
| Spearman degradation | PySR +0.110 vs ridge +0.600 (−0.49) |
| single-control vs PySR | prior `l1_pos_hit` 0.003716 / +0.452 — PySR is better than the prior alone but worse than the 6-feature linear combination |
| time split | NOT AVAILABLE (single month) |
| simple expression | yes (each fold 2–3 nodes) but unstable across folds → no freeze |

## Counterexamples

1. The `1/l1_hard_er` forms chosen in 5 of 8 folds diverge as `l1_hard_er → 0` (pole), a fragility the
   training domain masks but a generalisation risk.
2. Fold `graphlet` selects a multiplicative interaction while `af3`/`g4_closure` select reciprocals, and
   the coefficient signs/scale disagree → the "formula" is not one function across families.
3. Q_VEC: PySR Spearman is below the fold mean/median while the 1-feature model reaches +0.497 on 16
   rows — an n=16 artefact; no conclusion is drawn from it.

## Knowledge increment (next round)

* **H1 SUPPORTED**: L1 fidelity (`l1_pos_hit` +0.501, `l1_hard_er` +0.380) is the strongest single
  indicator among SID features — the 0.06-failure arms are largely prefix-fidelity failures.
* **H2 INCONCLUSIVE**: capacity/refinement features appear in the ridge but no fold-level isolation.
* **H3 REFUTED**: quantisation error + effective rank add nothing out-of-fold (CI covers 0).
* **H4 UNTESTED**: curvature/geometry features remain unidentifiable at n=3.
* **Null REFUTED**: a linear SID combination explains ρ≈0.60 of recall ordering, above the fold mean/median.
* **New lead (descriptive only, not used to select)**: final-epoch `recon` MSE has Spearman **−0.858**
  (n=23) — the strongest association in the whole table. It was outside the pre-registered core set, so
  it becomes a next-round hypothesis to be pre-registered and tested prospectively, never a post-hoc fit.

## Closing checklist (skill §6.2)
1. Production code: `build_inventory.py`, `build_table.py`, `run_analysis.py` (+ frozen inputs only).
2. Independent checks: `tests/test_analysis.py` (metrics arithmetic, reversal count, label aggregation vs
   inventory, expression identity) — 4/4 pass.
3. Mechanism experiments / negative result: reported honestly; no metric overstatement; 0 positives →
   `NOT_EVALUABLE` stated as such.
4. §6.3 infrastructure feedback: no public-component change was needed; zero drift this round (verified by
   modification times against the round start).
5. Evidence: `RUN_INVENTORY.json`, `JOINT_TABLE.json`, `ANALYSIS_RESULTS.json`, `analysis.log`,
   `HYPOTHESES.json`, this file.
