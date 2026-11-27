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
