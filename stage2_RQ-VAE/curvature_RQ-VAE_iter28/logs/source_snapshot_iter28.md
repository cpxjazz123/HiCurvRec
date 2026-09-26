# source_snapshot_iter28 (CAO-1)

## 1. Repository rules (priority-ordered)

1. `CLAUDE.md` (root): hard-coded paths, no-CLI scripts, four-card DDP launchers, Stage2/Stage3 unified artifact directories, GitHub-only remote, single `main` branch, mandatory commit+push per iter (§8/§10/§11/§12/§13).
2. `.claude/skills/curvature-rqvae-iter/SKILL.md`: 2+1 deliberation, CAO-1 contract (active since iter27, after the user retired FCCR-1).  §2.9 Feasibility Abort, §2.10 Autonomous Continuation.
3. `mechanism_contract_iter28.json`: `contract_version="CAO-1"`, `new_mechanism="sinkhorn_riemannian_ema_codebook"`, `srema_beta_default=0.99`.
4. `iteration_bridge.md`: iter27→iter28 carryover; iter27 was `ITERATION_ABORTED_INFEASIBLE` because user-fixed `trust_radius_fraction=0.5` did not engage the trust-region projection.  iter28 keeps the same mechanism family (Sinkhorn + hyperbolic geometry) but **replaces** AdamW's Euclidean codebook update with a Sinkhorn-weighted Riemannian EMA.
5. Historical evidence: iter18 `R@10=0.05988962203380978` (canonical baseline); iter27 ABORTED; iter26 FCCR-1 NO-GO; iter11/iter25 mid-table.

## 2. Stage1/Stage2/Stage3 paths (already wired in `curvature_config.py`)

- `STAGE0_DIR`: `/home/wlia0047/ar57/wenyu/GeneRec/results/stage0_build_parquet`
- `ITEM_EMB_NPY`: `…/stage1_GeneEmbedding/output/sentence_t5.npy`
- `RQVAE_OUT_DIR`: `/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE_iter28/out/rqvae/instruments`
- `ITEM_SIDS_JSON`: `/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE_iter28/item_sids.json`
- `SIDS_NPY`: `/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE_iter28/dataset/Instruments/sids_for_hgrec.npy`
- Stage3 `CODE_PATH`: `…/results/stage2_RQ-VAE/curvature_RQ-VAE_iter28/item_sids.json`
- Stage3 `LOG_PATH`: `…/results/stage3_T5Train/curvature_RQ-VAE_iter28/logs/`
- Stage3 `SAVE_PATH`: `…/results/stage3_T5Train/curvature_RQ-VAE_iter28/ckpt/`

## 3. iter27→iter28 carryover (verbatim from `iteration_bridge.md`)

- iter27 ABORTED because user-fixed `trust_radius_fraction = 0.5` produced `clip_frac = 0` across 200 MVG steps + 1000-step diagnostic probe; activation required tightening the registered constant, violating SKILL.md §2.9 "registered-spec immutability after hypothesis lock".
- iter28 picks up the canonical Sinkhorn + hyperbolic geometry machinery but **changes the codebook update mechanism** (AdamW → SREMA), not just the AdamW hyperparameter.

## 4. New contract (CAO-1) hard invariants

- Curvature is **still cyclic learnable** (inherited verbatim from iter18: `C_CYCLIC_MIN=0.05`, `C_CYCLIC_MAX=1.5`, `C_CYCLIC_PERIOD=100_000`, `c_layer_scale` per-layer bounded).
- The new mechanism is a **post-`optimizer.step()` Sinkhorn-weighted Riemannian EMA codebook update**.  Codebook `layers.X.embedding.weight` is **excluded** from the AdamW optimizer (custom iter28 check; `check_step5_optimizer` is intentionally skipped because it requires the codebook to be inside the optimizer).
- No new `nn.Parameter`; no new loss term; no new Sinkhorn assignment rule (the existing Sinkhorn is the input).
- SREMA hyperparameters: `beta = 0.99` (EMA momentum; per-step contribution `1 - β = 0.01`); summary print every 1000 global steps.

## 5. Unresolved conflicts

None.  CAO-1 is the active contract since iter27; iter28 keeps CAO-1 with a different `new_mechanism`.

## 6. Verified repo facts (independent readback)

| Fact | Path | Result |
|---|---|---|
| `MECHANISM_NAME` | `stage2_RQ-VAE/curvature_RQ-VAE_iter28/curvature_config.py:13` | `iter28_sinkhorn_riemannian_ema_codebook` |
| `srema_update` defined | `modules/quantize.py` | yes (post-step Riemannian EMA using cached Sinkhorn) |
| `cache_srema_inputs` defined | `modules/quantize.py` | yes (cache during forward) |
| `RqVae.srema_update` defined | `modules/rqvae.py` | yes (per-layer wrapper) |
| `ITER28_SREMA_*` constants | `curvature_RQ-VAE.py` | `0.99`, `1000` |
| Codebook excluded from AdamW | `build_curvature_conditioned_adamw` | yes (custom check; `check_step5_optimizer` skipped) |
| git HEAD | `bdcbbf9` | clean |
| remote HEAD | `bdcbbf9` | matches |
| parent iter18 `R@10` | `results/stage3_T5Train/curvature_RQ-VAE_iter18/logs/…/test_final.json` | `0.05988962203380978` |