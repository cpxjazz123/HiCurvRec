# Issue #273 run log — same-centre triplet contrast for sectional curvature

Issue: #273 "Stage2-only: preserve local curvature differences with same-centre triplet contrast"
(state OPEN at the start of this round). Body sha256
`fc85224d759f58eb7dab113bfa50b4018d0c8721b3d45bde37d05c543bc4ec44` (also stored verbatim in
`results/research/issue273/ISSUE_BODY.md`).

## INFRA_MODE

**FROZEN.** The issue body says verbatim: "单Agent；INFRA_MODE=FROZEN，复用全量图、三元组、参考点、
#269冻结κ_t与S*、R2及现有训练/验证资产。不修改公共infra，不新建公共资产". No authorisation sentence for
new public infrastructure exists, so nothing under `tools/`, `assets/`, the Stage3 entry, the Skill or
`CLAUDE.md` may change. Baseline snapshot (start of round 1) in `INFRA_SNAPSHOT_START.json`.

## Training decision

**Training is required.** The acceptance metrics (local geometry on the hard q, SID L2/effective rank,
reconstruction) are properties of a trained Stage2 candidate; they cannot be recomputed from existing
artefacts. Per §4 each formal candidate is 3000 epochs, seed 42, global batch 4096, the same legal
hardware protocol, final checkpoint. Stage3/T5/final-test/Recall are forbidden by the issue in every
outcome.

## Frozen acceptance list (C1..C15)

Baseline identities (frozen, full precision — no rounded thresholds):

| id | criterion | threshold | data | who computes / where |
|---|---|---|---|---|
| C1 | independent common-ruler local MAE | `<= 1.8935063111536612` (the #271 same-criterion value) | training graph triplets, frozen ruler scale 0.08433422446250916 | `results/research/issue271/local_geometry.py` (extended only by the issue-273 analysis script, same assets/scale) |
| C2 | local Spearman | `> 0.015850898303185326` (the frozen TIGER value) | same | same |
| C3 | local sign accuracy | `> 0.4324533490974272` (the frozen TIGER value) | same | same |
| C4 | predicted/target variance ratio | `>= 0.5` | same | issue-273 analysis script |
| C5 | regression slope of prediction on target | `>= 0.1` | same | issue-273 analysis script |
| C6 | centre-balanced within-centre difference RMSE | `< 2.0024533063041208` (the #269 value, measured before training by `results/research/issue273/within_centre_metrics.py`; log `results/research/issue273/executions/checks/issue273_within_centre_TIGER_calibrated_kappa_rq/`) | same | issue-273 analysis script (self-check vs the frozen evaluator: Spearman Δ0, sign Δ0, MAE Δ4.4e-7 from the frozen float16 storage) |
| C7 | within-centre ordering (centre-balanced pairwise order agreement) | `> 0.5513554042990889` (the #269 value, same measurement); the issue-272 style within-centre Spearman is reported as a secondary statistic (the #269: 0.10645665601043144, reproduced from the frozen #272 evidence) | same | same |
| C8 | stratified report (Graphlet/Orbit, sign, degree decile) | reported, no threshold | same | same (per_class/degree_decile of the frozen evaluator) |
| C9 | full fixed-ruler global J | `<= 0.338335882630658` (the #269 value) | full graph, common ruler | `results/research/issue272/ruler_inventory.json` + the frozen evaluator |
| C10 | 1-hop J | `<= 0.798295` (1.05 x #269 0.760281) | same | same |
| C11 | 2-hop J | `<= 0.278106` (1.05 x #269 0.264863) | same | same |
| C12 | distinct L2 prefixes of the exported hard SID | `>= 10918` (#265 historical) | exported `item_sids.json` | `results/research/issue271/collect_metrics.py` on the run's SID |
| C13 | effective rank of the SID code usage | `>= 8` | same | same |
| C14 | positive/negative prefix ER, behaviour neighbourhood, Graphlet/Orbit fidelity | not below the frozen TIGER plus the pre-frozen tolerance | same | same |
| C15 | reconstruction error | `<= 1.05 x ` the #265 same-criterion error (the exact value is recomputed at full precision; the contract value is 8.589e-5) | training reconstruction | the run's metrics + the #265 recompute |
| C16 | SID collision rate | **recorded only** — never a screening or closing criterion | same | recorded |

The issue additionally requires: the actual SID identity consistency must be checked; the comparisons
must freeze the asset identity and scale; MAE success obtained by compressing the curvature
prediction is not acceptable (hence C4/C5).

## Mechanism scope of round 1 (single primary change)

Parent = the **#269 fixed local supervision** (`TIGER_calibrated_kappa_rq`: frozen kappa_t A.1
pointwise Huber on the final hard q, no global distance training term — explicitly **not** the #272
global term). Round 1 adds exactly one thing: the **same-centre difference protection**
`L_delta` with a low incremental gradient budget and progressive enablement. Everything else
(architecture, 768->32 bottleneck, 3x256 codes, Euclidean selection, residual/floor/last-level
Sinkhorn, reconstruction/commitment/codebook, SID protocol, seed, batch, epochs) is unchanged.

## Reused assets (hashes in INFRA.md)

- frozen kappa_t cache `results/stage2_RQ-VAE/TIGER_calibrated_kappa_rq/kappa_cache_s0p068284/`
  (manifest sha256 `4b1d777bee24c0743239076a937d6a38bfc220c542908ccac53ce5b501b2fcbe`),
- the frozen local evaluator `results/research/issue271/local_geometry.py`
  (sha256 `6260b8bb89bcd9b1f359a0bdc36652c2ce82a8d1f5e7c79efbec885b61b3727a`),
- `results/research/issue271/collect_metrics.py` (sha256 `390f655cb97dd38daaa49675d189898dd191cc7715203b48e8d51a5ad3389d9f`),
- `assets/manifest.json` (sha256 `cfbd9d9c0d3c5282f9fdd0c0b28798cbf7a86479c2626b702e499b589acc5b6b`),
- the #269 baseline SID/checkpoint (`a89110b86168eb09621e5f9cfea6f6668253d6105e9f2a6647234b195efc725f` /
  `896e69b7bfc701a4fbd7413190cf8aadf0334b43d935ef3cbf66a73aae3c7410`).

## Round log

### Round 1 (started 2026-11-27)

- Read the issue; froze the C-list above; INFRA_MODE=FROZEN; the training decision (yes).
- Infra snapshot written to `INFRA_SNAPSHOT_START.json`.
- Diagnosis: `DIAGNOSIS.md`; hypotheses: `HYPOTHESES.json`; contract: `CONTRACT.md`.
- The #272 formal run (`stage2_0a2be6e306e145fe91b3d7be08a37c69`) is still in flight; per §1.0a the
  top-level tree is archived only after that run releases it, so the #273 candidate tree is created
  after the #272 run closes (its immutable snapshot keeps the run valid meanwhile).

### Round 1 implementation (2026-11-27)

- The #272 run finished and its tree was archived to
  `stage2_RQ-VAE/_archive/TIGER_codebook_budget_j_rq`; the owner closed #272 as a **negative
  result** (comment 8) and the round-3 table is recorded in `results/research/issue272/VERDICT.md`
  and issue comment 9. The #273 tree was then created at
  `stage2_RQ-VAE/TIGER_samecentre_delta_rq` (archived after the run to
  `stage2_RQ-VAE/_archive/TIGER_samecentre_delta_rq`) from the #269 immutable source snapshot
  (`results/stage2_RQ-VAE/_runs/stage2_bba78cf7c160467da1c8ae7de9b6e26f/TIGER_calibrated_kappa_rq/src`).
- Implementation (one primary mechanism, the same-centre difference protection):
  - `model/delta.py` — `same_centre_difference_loss(predicted, target, centre_population)`:
    the centre-balanced mean over the `m(m-1)/2` same-centre pairs of
    `((p_t - p_s) - (y_t - y_s))^2`, divided by the centre's valid triplet count; exactly zero for
    `m < 2`; refuses a population smaller than the sample.
  - `model/mechanism.py` — the `delta_weight` frozen buffer, `last_delta_value`,
    `calibrate_delta_weight(...)` (the per-module gradient-norm budget with `rho = 0.02` against the
    **weighted** pointwise term, the realised share and the cosine recorded), the `combined_loss`
    term `lambda_delta * min(1, global_step/200) * L_delta`, the provenance contract path
    (`results/research/issue273/CONTRACT.md`) and the checkpoint validation of `delta_weight`.
  - `train_rqvae.py` — the delta calibration and its broadcast, the 200-step ramp, and the
    `delta_loss` / `delta_weight` / `delta_scale` metrics.
  - The frozen `kappa_t` cache stays the #269 candidate's read-only table
    (`results/stage2_RQ-VAE/TIGER_calibrated_kappa_rq/kappa_cache_s0p068284`); the #272 global
    distance term is **not** inherited.
- Contract correction before the run (`CONTRACT.md` §3/§4/§5): the `lambda_delta` formula is the
  **per-module minimum** of the gradient-norm ratios (a single global ratio could push one codebook
  past the cap while the aggregate satisfies it), and the constant-prediction row states that the
  term is positive (the error spread equals the target's own spread) and offset-invariant.
- Tests: `tests/test_samecentre_delta.py` (14 dedicated tests: the hand-computed value, the
  centre-balanced normalisation, the same-centre pairing, the signed difference, the translation
  invariance, the constant-prediction value and offset invariance, the variance equivalence, the
  mispairing refusal, the coincident points, the masked rows, the frozen kappa not being an input,
  the hard-q STE gradient, the numerical stability, the single triplet) with the independent
  brute-force reference `tests/reference_delta.py`. The inherited suite was updated where the new
  mechanism changed its contract (the provenance fixture's contract path, the sample's optional
  `triple_population`). Full candidate suite: **137 passed**
  (`results/research/issue273/checks/candidate_tests.log`).
- Formal run dispatched: `stage2_98834b02a3e542a7a48492034ccf76e4`, the real entry
  `stage2_RQ-VAE/TIGER_samecentre_delta_rq/train_rqvae.py` (now
  `stage2_RQ-VAE/_archive/TIGER_samecentre_delta_rq/train_rqvae.py`), the single visible GPU (the frozen
  recipe's single-card branch), the worker PID 1524537, the logs under
  `results/stage2_RQ-VAE/_runs/stage2_3d8bc9a4ef5340e690b25efa20ad2c8d/TIGER_samecentre_delta_rq/result/logs/`.

### Round 1 result and closure (2026-11-28)

- The first dispatch was refused by the smoke: the delta calibration ran under `torch.no_grad()`, so
  the loss had no graph. Fixed, covered by a regression test, and re-dispatched as
  `stage2_3d8bc9a4ef5340e690b25efa20ad2c8d` (the smoke passed: one epoch, the identifier exported,
  collision 0.048278).
- The run completed the **full 3000-epoch budget** (1303.7 s, 0.45 s per epoch). The exported
  artefacts: `item_sids.json` (sha256 `f34bde311958f92b6ab1736bbfdfde8aea212e409dc95c8a8f45942d850b405b`)
  and `rqvae_best.pth` (sha256 `a7f17036fdee521a680b927b59db36e1a39cc38fe65f5844cc430562347b9e29`).
- The owner stopped the frozen evaluation before it produced the table. The two criteria that were
  measured both fail (C12 L2 5,966 < 10,918; C15 recon 9.792938e-05 > 8.589e-05) and eleven are
  NOT_RUN; nothing is estimated. Verdict: `results/research/issue273/VERDICT.md`; the
  machine-readable record: `results/research/issue273/round1_sid_structure.json`; the conclusion
  comment: `results/research/issue273/COMMENT_FINAL.md`.
- The tree is archived to `stage2_RQ-VAE/_archive/TIGER_samecentre_delta_rq`; the top level of
  `stage2_RQ-VAE/` holds no active arm.
- The CLAUDE.md §63 layered metric record is marked **missing** (the dispatched run wrote no
  `archive_binding.json` next to the SID); `results/research/issue273/write_layer_metrics.py` is the
  wrapper for the next run that carries one.
- The infra end snapshot equals the start snapshot (added 0, removed 0, modified 0).
- The issue is closed by the owner's instruction as an **incomplete negative result**; Stage3 was
  forbidden in every outcome and was not run.
- The next round can settle the one open question with the frozen artefacts that are still in
  place: the #273 run's own `within_centre_metrics.py` output against the #269 baseline
  (`difference_rmse` 2.002453, `within_centre_spearman` 0.106457).
