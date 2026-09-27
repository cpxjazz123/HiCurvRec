# Iter31 S00 Source Packet — Source Truth

```text
STAGE_ID=S00_SOURCE_TRUTH
ROUND=1
ITERATION=31
ITERATION_PURPOSE=PERFORMANCE_SEEKING_MECHANISM
SWEEP_OR_REPLICATION_ITERATION=NO
ROOT_CAUSE_ITERATION=NO
```

## Task
Independently extract controlling rules, verified repository facts, conflicts, and unresolved prerequisites for fresh Iter31 registration. Do not implement, launch, tune, or infer scientific outcomes. Agent A and Agent B must use this same packet and primary sources; do not read the other candidate.

## User direction and iteration boundary
The user explicitly cancelled the active Iter30 fixed-curvature-mapping comparison and requested a fresh direction: Curvature-Consistent Cross-Layer Hyperbolic Residual Aggregation (HRA), citing arXiv:2609.26342. Their proposal is to map each layer's codeword from curvature `c_l` to reference curvature `c0` by `exp0(c0, log0(c_l, q_l))`, aggregate in reverse-nested Möbius order at `c0`, then apply `log0(c0, ·)` before the existing Euclidean decoder. They requested only the aggregation change, not d-HSTE or other new components. This is a user-specified proposal, not yet an adjudicated Iter31 hypothesis or implementation authorization. Iter30 is explicitly `ITERATION_CANCELLED_BY_USER_DIRECTION_CHANGE`, not scientifically infeasible; no Stage2/Stage3/GPU run occurred. Its cancellation archive was pushed in commit `21e0488dcad39b3fd277b071a12daec59b56ba2f` to the sole allowed GitHub `origin/main`, with local/remote hashes equal. Use Iter31 rather than reusing Iter30.

## Controlling repository and skill rules
- Root `CLAUDE.md` is highest repository authority. Work only on `main`; only remote `origin=https://github.com/cpxjazz123/HiCurvRec.git`; never use worktrees or force-push. Verify pushed `main` against GitHub `refs/heads/main`.
- Project scripts hard-code paths/parameters; no CLI args or environment overrides. Stage2 artifacts belong only under `results/stage2_RQ-VAE/curvature_RQ-VAE_iter31/`; Stage3 artifacts only under `results/stage3_T5Train/curvature_RQ-VAE_iter31/`. Iteration source tree must not contain `.pth`, `.npy`, or `item_sids.json` products.
- Stage2 descriptive SID metrics are not gates. Stage3 decides adoption; root goal is strict `test_R@10 > 0.065`.
- Before any actual Stage2 run, perform the required one-checkpoint/one-batch gradient-path checks, including each active mechanism loss; fail blocks training.
- `curvature-rqvae-iter` requires independent A/B + Judge C for every new pipeline stage and canonical-only propagation. One structural mechanism per iteration. No seed/matched-seed replication, parameter sweep, sensitivity, ablation-only, reverse-control-only, or root-cause iteration. Use one protocol-locked seed. If a registered mechanism is mathematically invalid or infeasible, terminate before expensive runs; do not retune in place.
- FCCR-1 fixed closed-form curvature is the current contract unless a higher-priority instruction or canonically adjudicated between-iteration transition changes it. The requested HRA mechanism is outside FCCR-1's mapping-only experiment scope; S02/S04 must explicitly determine and record whether/how a new contract is validly registered. Do not silently treat FCCR-1 as already amended.

## Primary HRA paper evidence
Primary source: [arXiv:2609.26342, Geometry-Aware Hyperbolic Residual Quantization](https://arxiv.org/pdf/2609.26342), version 1 dated 2026-09-22.
- §4.1 defines HRA as pairing **left** Möbius residual subtraction `r_i = (-q_i) ⊕_c r_{i-1}` with reverse-nested aggregation `q_1 ⊕_c (q_2 ⊕_c (... ⊕_c q_N))`; its exact telescoping argument uses one shared curvature `c` throughout.
- §4.2 separately introduces block-level discounted Hyperbolic STE (d-HSTE). The paper's full GHRQ method combines HRA and d-HSTE; these are independent changes.
- Appendix A.6 fixes paper-model curvature at `c=1` for all hyperbolic models. Recommendation uses Beauty, four layers, so it is not a directly comparable HiCurvRec Instruments run.
- §6.6/Table 8 reports HRA-only ablation and says it is the least faithful recommendation configuration; the full HRA+d-HSTE configuration is much more faithful. This is risk evidence, not a direct Stage3 prediction for the proposed cross-curvature variant.
- Table 3 reports full GHRQ `R@10=0.0604` and naive hyperbolic `R@10=0.0606` on the paper's Beauty setup; do not treat these as HiCurvRec baseline results.

## Verified Iter29 implementation facts
Primary files: `stage2_RQ-VAE/curvature_RQ-VAE_iter29/curvature_RQ-VAE.py`, `curvature_config.py`, `modules/rqvae.py`, `modules/quantize.py`, `modules/hyperbolic.py`, `modules/loss.py`, and `scripts/computed_behavior_branching.json`.
- Iter29 has `SEED=42`, `MAX_GLOBAL_STEPS=100000`, three residual layers, codebook size 256, `MIDPOINT_LAYER_MASK=[False, False, False]`, and fixed per-layer curvatures `[1.3660953164241916, 0.7347829661951981, 0.6439958072706683]`.
- `Quantize.forward` returns tangent codebook embeddings `e_l` with the training straight-through expression `x + (embedding - x).detach()`. The manifold codeword is formed in residual update as `q_l=exp0(c_l,e_l)`.
- All three active residual layers use left Möbius subtraction: `_mobius_add_t(-embedding_h, residual_h, curvature)`. After log map, `_step5_transport` applies `_transport_between_t` from `c_l` to `c_{l+1}` for the next layer.
- Current `_step6_sum_embeddings` instead sums the tangent embeddings in Euclidean coordinates. `forward` passes that sum to the unchanged decoder, then computes reconstruction loss using layer-0 curvature `c0`; `ReconstructionLoss` maps both decoded output and target with `exp0(c0, ·)` and measures Poincaré distance.
- `_expmap0_t`, `_logmap0_t`, `_mobius_add_t`, and `_transport_between_t` are implemented in `modules/hyperbolic.py`. The proposed codeword map is equivalent to carrying a codeword's origin tangent coordinate across curvature.
- A direct 2D arithmetic witness using Iter29 `c1=0.7347829661951981`, `c0=1.3660953164241916`, `x=[0.30,0.10]`, and `y=[-0.08,0.25]` gives `T(x ⊕_{c1} y)=[0.2318117865,0.3233175819]` versus `T(x) ⊕_{c0} T(y)=[0.2492224376,0.3166295024]`, with L2 difference `0.01865103694`. This artificial point-pair is not a model-data or performance result; it disproves a universal gyrogroup-homomorphism assumption for this radial transfer.
- The paper's exact HRA proof assumes a common curvature; Iter29 uses per-layer curvatures and cross-layer residual transport. A Step6-only common-reference aggregation is therefore not proven to invert the actual residual cascade. Do not claim exact telescoping unless S02/S04 proves it for this implementation. A valid performance hypothesis may make a narrower claim about common-reference decoder aggregation, but that distinction must be adjudicated.
- The paper's d-HSTE is not part of the requested change. Adding it would violate the user's one-place scope unless a Judge finds that the requested HRA-only mechanism cannot be validly registered; no unreviewed extra mechanism may be added.

## Existing Iter29 run evidence
- `stage2_RQ-VAE/curvature_RQ-VAE_iter29/logs/protocol_manifest_iter29.md` records seed 42, 100,000 Stage2 steps, 3×256, Stage3 seed 42, 150 epochs, beam 20, and `n_eval=57439`. It identifies Iter26 as Iter29's direct mapping-control baseline; a new HRA protocol must independently lock its own parent and comparison baseline in S01.
- Iter29's recorded Stage3 result is `results/stage3_T5Train/curvature_RQ-VAE_iter29/logs/Amazon_2023_Instruments/Sep-27-2026_06-11-34/test_final.json`: `test_recall@10=0.05921064085377531`, `n_eval=57439`. Iter29 Stage2 best checkpoint and raw SIDs exist under its result root.
- `computed_behavior_branching.json` records `branching=[19.324911558712664,1.4605688962651735,1.0148104414712726]`, `raw_residual_medians=[1.0,0.10941,0.09331]`, and the curvature vector above. Its provenance explicitly lacks historical checkpoint/SID replay and byte-identity proof; do not overstate that evidence.

## Decisions still open for later stages
1. S00: authoritative source snapshot and unresolved-contract conflicts.
2. S01: select parent and a protocol-compatible single-run baseline from exact primary records; do not assume the previous iteration is automatically the baseline.
3. S02: state one falsifiable HRA cross-curvature hypothesis without unsupported exact-telescoping claims.
4. S03: trace each equation input and codeword/tangent semantics.
5. S04: define a machine-readable contract only if the proposed operation is mathematically and semantically coherent under the existing per-layer residual transport.
6. S05/S06: keep one factor and produce an exact minimal patch plan; no source code changes before Judge approval.
7. No Stage2 or Stage3 authorization exists at S00.
