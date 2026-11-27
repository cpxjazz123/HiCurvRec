# Issue #271 run log

- issue: 271 — "Globally constrained local sectional curvature supervision for RQ-VAE"
- body sha256: `198ac63423e330b159ab6b0cee3ab57e283ea5714ab48286a8b8b2b5bbe09823`
- frozen at: 2026-11-27 (before any mechanism code)

## Modes and decisions

**INFRA_MODE = FROZEN.** Issue quote: "INFRA_MODE=FROZEN，复用已有全量训练图距离、三元组/参考点、
#269冻结κ缓存、训练/评估及R2流程。只允许候选机制和专属测试、分析，不修改公共Infra或重建公共资产。"
No `BUILD` authorization exists in the issue; public trees (`tools/iteration/`,
`tools/research/stage2/`, `tools/research/verification/`, `assets/manifest.json`, Stage3 entrance,
`CLAUDE.md`, the skill) are read-only.

**Training is required.** Acceptance depends on quantities that only new weights can produce:
full-graph J of the candidate hard q, collision/quantisation diagnostics, and Stage3 R@10. Issue §3 and §4
require one Stage2 3000-epoch run and one Stage3 150-epoch run. This is recorded before any code.

**Mechanism scope (what may change).** One new candidate tree
`stage2_RQ-VAE/TIGER_global_j_rq/` (copy of the archived #269 tree, because issue §1 explicitly requires
inheriting the frozen κ_t / A.1 local objective) plus its own tests, and issue-local analysis under
`results/research/issue271/`. No public tool, no `assets/` generator, no Stage3 entrance, no skill.

## Acceptance checklist (frozen from the issue body)

| id | requirement | criterion / output |
|---|---|---|
| C1 | local term | `L_curv = mean(origin-value error)` on the real hard `q = Σ_l c_l`; #269 `κ_t` and `S*=0.06828367045803868` frozen; no geometry-ruler tuning; weight calibration follows #269's gradient-norm rule |
| C2 | global ruler | `J = mean_pairs[(d_ij/(g_ij)-1)²]`, common Euclidean ruler; S* unit and #269 map verified (factor 2 from `w=q/(2S*)`), no double scaling; exclude self/unreachable; symmetric/undirected distinction recorded; full-graph evaluation from existing graph asset in chunks; training uses a fixed stratified pair sample with recorded coverage and weights; no new public asset |
| C3 | reference threshold | recompute `J_TIGER` on the frozen TIGER checkpoint with the same ruler; `J_max = J_TIGER + ε`; ε calibrated before training from numerical/sampling error; freeze per-stratum and tail diagnostics |
| C4 | constrained optimisation | primal-dual / augmented Lagrangian with non-negative `μ`; form, step, update frequency and gradient budget frozen before training; log violation and μ; a decreasing penalty is not accepted as feasibility |
| C5 | gradients and tests | local and global terms act on final hard q; STE routes to encoder and codebooks; graph/κ/S* carry no gradient; real-batch probe of gradient norms/angles/updates/feasibility; tests for κ=0 limit, masks, dual sign, hard-q/STE, zero/coincident/NaN; mutation catches a non-effective global constraint |
| C6 | Stage2 | one candidate, 3000 epochs, seed 42, original global batch, final checkpoint, existing admission; per-epoch logs of total/unweighted/weighted gradients, local-global conflict, `J`, μ, parameter updates, performance, coverage; single-card run is legal only with `GENREC_SINGLE_GPU=1` and is disclosed as not fully comparable |
| C7 | SID diagnostics | hard SID export and re-encode consistency; three-code collision `< 0.40`; effective rank, codebook usage, reconstruction/quantisation error, L2 diversity, L3 refinement, positive/negative prefix ER, graphlet/orbit, true training-neighbourhood fidelity; check #269-style over-merging |
| C8 | geometry success | full-graph `J ≤ J_max`; independent frozen reference/common-ruler local A.1 error `< TIGER`; report Spearman, sign accuracy, strata and tails; report vs #265/#269; failure is a result, not a reason to relax |
| C9 | Stage3 gate | if legal admission passes, run Stage3; if a hard gate fails, repair within the legal mechanism, never relax post hoc; never report an unrun Stage3 as PASS |
| C10 | Stage3 | one official 150-epoch run, seed 42, beam 20, frozen data/model protocol; freeze SID and prerequisites before the single final test; recompute Recall/NDCG@5/10; compare TIGER canonical R@10 `0.05520639286895663` and target `0.065`; no TIGER retrain; state that the joint candidate cannot be attributed to local curvature alone |
| C11 | four answers | global constraint satisfied? local fidelity improved? SID avoided compression and retained structure? downstream Recall improved? record counterexamples and limits |
| C12 | deliverables | DIAGNOSIS/HYPOTHESES/CONTRACT; candidate code and tests; persistent receipts; full/covered geometry assets + manifest; training/gradient/performance reports; SID/structure diagnostics; independent Stage3 recomputation; R2 four-piece set; VERDICT; zero-drift infra snapshot; commit and push |

## Infra checkpoints

| checkpoint | file | result |
|---|---|---|
| ① before work | `INFRA_SNAPSHOT_START.json` | recorded below |
| ② before official training | pending | |
| ③ before archive/close | pending | |

`assets/` reuse: `assets/train/visible_undirected.npz` (graph), `assets/v8_runtime_layout/targets.npy` and
`target_offsets.npy` (κ-triplet table), `assets/distance_histogram.npz`/`node_moments.npz` (distance summary),
`stage1_GeneEmbedding/output/sentence_t5.npy`, `results/stage2_RQ-VAE/TIGER_calibrated_kappa_rq/kappa_cache_s0p068284/`
(frozen κ_t cache), the frozen TIGER checkpoint. Each is cited by `path:sha256` in the contract/verdict.

## Old-mechanism archive (§1.0a)

`stage2_RQ-VAE/TIGER_calibrated_kappa_rq` is the only active mechanism level entry; it is archived to
`stage2_RQ-VAE/_archive/TIGER_calibrated_kappa_rq` before the new candidate is copied. The baseline and
`_archive` stay in place.

## Threshold calibration (before training)

- fixed pair table: 2,048 centres, 16,469 pairs, 0 duplicate pairs; coverage recorded in
  `geometry/global_pairs.json`.
- full J on the frozen TIGER checkpoint: **0.739634552674212** (1-hop 15.158, 2-hop 2.675,
  3-hop 0.792, 4-5 0.291, >=6 0.175).
- sample estimate 0.764825, bootstrap SE 0.021153, |sample-full| 0.025190
  -> epsilon = 2*SE = **0.04230598283915425**, J_max = **0.7819405355133663**.
- the same ruler on #269: full J **0.338336** (1-hop 0.760, 2-hop 0.265, 3-hop 0.311,
  4-5 0.394, >=6 0.561). The issue's "J 0.183 -> 0.326" is the #270 three-expert ruler, not this
  Euclidean common ruler: under this ruler #269 is far below TIGER. Recorded as a ruler-dependence
  finding, not used to change J_max.
- real-batch module check (`MODULE_CHECKS.json`): lambda 5.88e-6; local gradient 6.88e-4,
  global gradient 0.0902, cosine 0.121; encoder/codebook gradients nonzero from both terms,
  decoder zero, hard-q gradient present, mu=0 makes the constraint term exactly 0.

## Stage2 result (formal)

- run `stage2_cc9a31a9325b439f98c8b1779aaa32aa`, world_size 1, 3000 epochs, seed 42, batch 4096,
  wall 1909 s (final J included), single-card disclosed.
- full-graph J **0.351359** vs J_max 0.781941 -> **P1 PASS**; strata: 1-hop 0.642, 2-hop 0.260,
  3-hop 0.323, 4-5 0.412, >=6 0.580. (TIGER 0.739635; #269 0.338336 on the same ruler.)
- mu stayed **0** for all epochs (the constraint was never violated; the safety bound is non-binding);
  the final step-estimate J was 0.3354, the exact full J 0.3514.
- collision 0.152113 < 0.40; SID unique 3-code 20,847; L2 groups 6,097 (vs #269 6,246; no #269-style
  compression); L2/L1 23.82; L3/L2 3.42; L3 refinement gain 2.883 bits; all three codebooks use 256/256;
  quant error 0.00939 (below the 0.035 diagnostic band, warning only); effective rank 4.50.
- independent common-ruler local A.1 (frozen #270 evaluator, SCALE 0.084334): MAE **1.8935** vs TIGER
  **7.3092** (P2 error PASS); but Spearman **-0.00548** vs TIGER 0.01585 and sign accuracy 0.3926 vs
  0.4325 are worse. The error reduction does not carry the rank/sign quality; reported as a mixed result.
- Stage3 started 18:05 with the SID; export gate allow (collision 0.1518, checkpoint reproduces SID,
  174 items rewritten), layer record missing is a warning (manual §63 record to be written).
