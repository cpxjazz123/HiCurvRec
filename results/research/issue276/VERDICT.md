# Issue #276 — VERDICT

**Question:** can a mechanism-agnostic relation from SID features to Stage3 Recall@10 be discovered,
using SID identity (not mechanism family) as the unit, with the historical high-recall assets recovered?

**Answer:** no symbolic formula passes; a **simple linear predictor on six SID-only features** does,
OOF, including the recovered >0.06 band (13/13, 0 false positives). PySR is significantly worse than
that control. Vector features do not improve calibration. Curvature geometry is unidentifiable (n=3).

## C1–C10

| C | requirement | evidence | verdict |
|---|---|---|---|
| C1 | recover high-recall data and identity | 12/12 named arms content-bound (6 L1 ledger, 6 L2 collision) — `recovery/AUDIT.md`; 91 records >0.06, 26 with a bound arm; unbound arms recorded with reasons | PASS |
| C2 | same-SID aggregation, no cross-fold leakage | unit = discrete SID; per-SID mean + spread in `label_sources`; folds by SID group; no duplicate content crosses folds | PASS |
| C3 | SID-only feature table | 253 rows, frozen evaluator, definitions in `TABLE_MANIFEST.json` | PASS |
| C4 | vector queue with corrected forward | `vector_forward.json`: 74 arms, bias-included `PR(z)`, `PR(q)`, `quant_err`, `recon_mse`, `recon_cos`; stored bias-less series flagged (71/74 differ); #275 batch-loss `recon` not mixed in | PASS |
| C5 | geometry queue | 3 sha-matched SIDs from the #271 assets, descriptive only, coverage stated | PASS (unidentifiable) |
| C6 | per-SID table/manifest/audit | `SID_TABLE.json` + `TABLE_MANIFEST.json`; 73 collision checks equal, 1 flagged; #274 R1/R2/R3 SIDs audited (`AUDIT_274.json`) — not on disk, not in table, no blind predictions | PASS |
| C7 | pre-registered SID-level validation | 5 folds by SID group, band-stratified 17/99/13, seed 42, `FOLDS.json` bound to table sha `4c927914…`; inner selection per training folds; every SID exactly one OOF | PASS |
| C8 | frozen features + controls + criteria | 8 frozen SID features; ridge6 = #275 six-feature control; mean/median/single/prior/ridge13; SID-group bootstrap CI; success criterion fixed before fit | PASS (negative for PySR) |
| C9 | PySR fixed budget, Pareto, SymPy recompute | PySR 0.19.4, `+,-,*,/,square`, maxsize 10, depth ≤4, 100 iterations, 8 populations, seed 42; per-fold expressions + Pareto in `ANALYSIS_276.json`; SymPy recompute `max|Δ|` recorded per fold | PASS |
| C10 | deliverables, no Stage3, commit/push | all paths above; no training/Stage3 started by this issue; infra snapshot start/close | PASS |

## Pre-registered success criterion

"Usable formula" = OOF MAE strictly below the strongest control **and** paired CI excludes 0 **and**
OOF ρ ≥ best single OOF ρ. PySR: MAE 0.004067 vs ridge6 0.003059, Δ +0.001008 CI [+0.000490, +0.001538]
→ **criterion not met; no symbolic formula frozen**. The usable predictor is the *linear* ridge on the
six SID-only features (`ridge6`), reported as the simple validated expression of this round.

## Root cause resolution (from `DIAGNOSIS.md`)

| id | explanation | outcome |
|---|---|---|
| R1 | #275's negative was an artifact of family-LOFO/zero positives | **partially refuted**: under SID grouping with 13 positives, PySR is still significantly worse than the linear control; the PySR-vs-ridge ordering is robust. The ridge-vs-controls conclusion also survives |
| R2 | label-range truncation hid a window/threshold relation | **resolved**: 12 arms recovered, band covered in every fold; the >0.06 band *is* identifiable (13/13, 0 FP by ridge6) |
| R3 | the recon ρ −0.858 was protocol mixing | **resolved**: stored vector series uses a bias-less forward; the unified decoder MSE gives ρ −0.087 (n=73); #275 used final-epoch batch loss on 23 rows — the −0.858 does not transfer |
| N | no feature set beats the fold mean/median | **refuted**: ridge6 MAE 0.003059 vs median 0.004424 (31 % lower) |

## Knowledge increment / hypotheses

* **H1 REFUTED** on the new design: `l1_pos_hit` alone has OOF ρ −0.283 over 129 SIDs (train-selected
  single features also fail: ρ −0.149). The #275 descriptive single-feature ranking does not generalise
  across SID groups; the six-feature combination does.
* **H2 SUPPORTED**: fidelity-only ridge (3 features) 0.003507/ρ 0.473 vs six-feature 0.003221/ρ 0.512
  under identical folds and α (exploratory, `EXTRAS.json`).
* **H3 INCONCLUSIVE**: vector features raise the 13-feature ridge ρ (.552→.699 on the same 73 rows) but
  worsen MAE (0.00342→0.00570); no calibration increment, ranking increment present.
* **H4 UNTESTED**: geometry n=3.
* **H5 REFUTED**: the grouping change alone does not flip PySR vs the linear control.
* **H6 REFUTED**: the −0.858 recon relation is a protocol artifact (batch loss, bias-less forward, n=23).
* **H7 SUPPORTED**: high-recall assets recovered (12/12 named).
* **H8 INCONCLUSIVE**: `PR(z)` shows a window shape (6–8 band mean 0.0595 vs 0.0463 below 4 and 0.0529
  at ≥12) but ρ −0.031; not identifiable as a formula.
* **N REFUTED** (see above).

## Counterexamples / limits

1. Ridge6 over-predicts low-recall arms with good fidelity (`TIGER_cogeo_region_argmax` 0.0398→0.0574,
   `TIGER_dul_rq` 0.0260→0.0424), so the threshold margin is thin outside the observed band.
2. One SID (`bootstrap_code_stability__resample`, recall 0.0634) has a stored collision inconsistent
   with its content (0.03709 vs 0.04132) and is kept secondary; the 13/13 threshold result includes it
   via its arm label but the identity is weaker.
3. 124 of 253 SID rows have no recall label; the label range is 0.0260–0.0648, no ≥0.065.
4. 13 positives is a small band; band-stratified folds place 2–3 positives per fold and the CI accounts
   for SID groups, but the threshold table is not a validated classifier.

## Closure checklist (§6.4)

1. Production code implemented: analysis/recovery scripts above are the actual executed code. **YES**
2. All necessary tests executed with logs: `execution.run` receipts under `logs/` (recover, recover_git,
   features, table, folds, analysis, audit, extras); analysis PASS rc=0. **YES**
3. Required training done: none required (issue forbids Stage2/Stage3). **YES (N/A)**
4. Infra boundary: FROZEN; `INFRA_SNAPSHOT_START.json` → close snapshot compared; expected zero drift
   (all writes under `results/research/issue276/`). **YES**
5. Executable work left inside the issue scope: none — the issue's deliverables are the audit, table,
   contract, folds, code/tests, OOF metrics/CI, Pareto, verdict. **NO**
