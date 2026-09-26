# stage2_execution_plan_iter28

## 1. Launch command (zero CLI args; no env overrides)

```bash
cd /home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/curvature_RQ-VAE_iter28
nohup /home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 \
  curvature_RQ-VAE.py \
  > logs/train_run.log 2>&1 &
```

The script `curvature_RQ-VAE.py` internally forks
`torchrun --standalone --nproc_per_node=4 --master_port=50200 -u
curvature_RQ-VAE.py`; the actual training logs land in
`logs/train_migrated.log`.  No environment overrides are allowed
(CLAUDE.md §1).

## 2. Inputs

| Input | Path |
|---|---|
| Stage1 embeddings | `/home/wlia0047/ar57/wenyu/GeneRec/stage1_GeneEmbedding/output/sentence_t5.npy` |
| Stage1 sidecar | `/home/wlia0047/ar57/wenyu/GeneRec/stage1_GeneEmbedding/output/item_ids.json` |
| Stage0 train.parquet | `/home/wlia0047/ar57/wenyu/GeneRec/results/stage0_build_parquet/train.parquet` |
| Warm-start checkpoint | `/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE_iter8/out/rqvae/instruments/rqvae_best.pth` |
| Residual layer norms | `out/decoder/instruments_hgrec_configs/hgrec_iter28_sinkhorn_riemannian_ema_codebook/layer_norms.json` (fallback: iter1 `[0.001, 0.932889, 1.0]`) |

## 3. Outputs (must be written)

| Output | Path |
|---|---|
| Best checkpoint | `results/stage2_RQ-VAE/curvature_RQ-VAE_iter28/out/rqvae/instruments/rqvae_best.pth` |
| Raw 3-token SIDs | `results/stage2_RQ-VAE/curvature_RQ-VAE_iter28/out/rqvae/instruments/sids_raw.npy` |
| 4-token SIDs (TIGER-compatible) | `results/stage2_RQ-VAE/curvature_RQ-VAE_iter28/dataset/Instruments/sids_for_hgrec.npy` |
| Item SIDs JSON | `results/stage2_RQ-VAE/curvature_RQ-VAE_iter28/item_sids.json` |
| Stage2 training log | `stage2_RQ-VAE/curvature_RQ-VAE_iter28/logs/train_migrated.log` |
| SREMA summary line | `[iter28_srema_summary step=N] ...` every `ITER28_SREMA_LOG_EVERY=1000` steps |
| Stage2 base log | `stage2_RQ-VAE/curvature_RQ-VAE_iter28/logs/train_run.log` |

## 4. Wallclock budget

iter18 (same protocol) finished in **~7 minutes** on 4×A40 GPUs.
iter28 adds:
- per-step `srema_update` (~3 layers × pairwise `[K=256, B=640]` `log_{p_j}(z_i)` + Sinkhorn-weighted sum + exp0/log0 projection);
- cached Sinkhorn assignment + residual (no extra forward cost).

Estimated overhead: ~15% on the per-step time (MVG-measured).  Wallclock
budget: **10 minutes**, with a 15-minute hard cap.

## 5. Stop conditions (re-checked every 1000 global steps)

- The SREMA summary line
  `[iter28_srema_summary step=N] median_update=[...] max_update=[...]
  frac_updated=[...] saturation_max=[...]` must appear at least once per
  1000-step window.
- Every printed `median_update`, `max_update`, `saturation_max` must be
  finite; every `saturation_max < 1` (no escape).
- `frac_updated ≥ 0.5` for at least one layer per window (otherwise
  classify F-1 inactive, autonomous abort per SKILL.md §2.9/§2.10).
- No `NaN/Inf` may appear in any of the printed values.
- Step10 must report `first optimizer.step ok`.

## 6. Autonomy boundary (per SKILL.md §2.9/§2.10)

- If F-1 (inactive) is observed at any 1000-step window: stop and
  autonomously abort as `ITERATION_ABORTED_INFEASIBLE`.  Do not ask the
  user.  Do not retune inside iter28.
- If F-2 (escape) is observed: stop and abort.
- If Stage2 completes normally: launch Stage3 with `scripts/run_stage3_iter27.py`.

## 7. Per-1000-step invariants to verify post-run

- `c_l(t)` trajectory matches iter18's trajectory at step 0/25k/50k/100k
  to within `1e-4` (the SREMA does not write `c_l(t)`).
- `beta2_l` is unchanged at all four checkpoints (frozen at init from iter18).
- `clip_frac`-style saturation `< 1.0` (i.e. codebook stays inside the ball).

## 8. Stage3 wiring (planned for after Stage2)

```bash
cd /home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/curvature_RQ-VAE_iter28
nohup /home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10 \
  scripts/run_stage3_iter27.py \
  > logs/_stage3_run.log 2>&1 &
```

Reads `item_sids.json` from `results/stage2_RQ-VAE/curvature_RQ-VAE_iter28/`,
writes to `results/stage3_T5Train/curvature_RQ-VAE_iter28/{logs,ckpt}/`,
runs 150 epochs with beam=20 / n_eval=57439 (identical to iter18).