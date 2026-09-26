# stage2_execution_plan_iter27

## 1. Launch command (zero CLI args; no env overrides)

```bash
cd /home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/curvature_RQ-VAE_iter27
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
| Residual layer norms | `out/decoder/instruments_hgrec_configs/hgrec_iter27_hyperbolic_codebook_trust_region/layer_norms.json` (fallback: iter1 `[0.001, 0.932889, 1.0]`) |

## 3. Outputs (must be written)

| Output | Path |
|---|---|
| Best checkpoint | `results/stage2_RQ-VAE/curvature_RQ-VAE_iter27/out/rqvae/instruments/rqvae_best.pth` |
| Raw 3-token SIDs | `results/stage2_RQ-VAE/curvature_RQ-VAE_iter27/out/rqvae/instruments/sids_raw.npy` |
| 4-token SIDs (TIGER-compatible) | `results/stage2_RQ-VAE/curvature_RQ-VAE_iter27/dataset/Instruments/sids_for_hgrec.npy` |
| Item SIDs JSON | `results/stage2_RQ-VAE/curvature_RQ-VAE_iter27/item_sids.json` |
| Stage2 training log | `stage2_RQ-VAE/curvature_RQ-VAE_iter27/logs/train_migrated.log` |
| Trust-region summary JSONL | `stage2_RQ-VAE/curvature_RQ-VAE_iter27/logs/iter27_trust_region.jsonl` |
| Stage2 base log | `stage2_RQ-VAE/curvature_RQ-VAE_iter27/logs/train_run.log` |

The orchestrator must verify that every output path exists after the
run completes (a non-empty `rqvae_best.pth`, a 3-token SID file of shape
`(24587, 3)`, a 4-token SID file of shape `(24587, 4)`, a JSON with
`24587` keys each holding a 4-element list).

## 4. Wallclock budget

iter18 (same protocol) finished in **~7 minutes** on 4×A40 GPUs.
iter27 adds:

- one extra `_poincare_distance_t` call per layer per step
  (`pairwise [K=256, K=256]` ≈ 65,536 entries);
- one `_expmap0_t` and one `_logmap0_t` per layer per step.

Estimated overhead: < 5% on the per-step time.  Wallclock budget: **10
minutes**, with a 15-minute hard cap.

## 5. Stop conditions (re-checked every 1000 global steps)

- The trust-region summary line
  `[iter27_trust_region_summary step=N] clip_frac=[...] delta=[...]
  tau=[...] delta_over_tau=[...]` must appear at least once per
  1000-step window.
- Every printed `delta_over_tau` value must be finite and `> 0`.
- No `NaN/Inf` may appear in any of the printed values.
- The first 10 lines of `train_migrated.log` must contain `[Step1]` /
  `[Step2]` / `[Step3]` / `[Step4]` markers and the
  `[Step5] iter18 curvature-conditioned AdamW beta2: ...` line (the
  Step5 print was inherited from iter18 and should appear unchanged).
- Step10 must report `first optimizer.step ok`.

## 6. Known caveat (carried forward)

The MVG Layer-D sweep showed that the canonical
`trust_radius_fraction = 0.5` does **not** cause `clip_frac > 0` under
the iter18 ckpt's per-step AdamW displacement scale (`~0.01` vs trust
radius `~0.06–0.28`).  This means the mechanism, as currently
parameterized, is a no-op safety bound under iter18's regime; iter27
reduces to iter18 by default.

If the user approves tightening `trust_radius_fraction` to `0.05`
(one-factor deviation), the mechanism will fire in the `[0.05, 0.20]`
clip-frac range and the run will test the **active** hypothesis.  This
decision is pending user input; until then the canonical `0.5` is used.

## 7. Per-1000-step invariants to verify post-run

- `c_l(t)` trajectory matches iter18's trajectory at step 0/25k/50k/100k
  to within `1e-4` (the trust-region does not write `c_l(t)`).
- `beta2_l` is unchanged at all four checkpoints (frozen at init).
- `clip_frac ∈ [0.00, 0.95]` for every printed summary line; if
  `clip_frac > 0.99` for any layer for the entire run, classify as
  `MECHANISM_DESTRUCTIVE` (F-2).

## 8. Stage3 wiring (planned for after Stage2)

```bash
cd /home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/curvature_RQ-VAE_iter27
nohup /home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10 \
  scripts/run_stage3_iter27.py \
  > logs/_stage3_run.log 2>&1 &
```

Reads `item_sids.json` from `results/stage2_RQ-VAE/curvature_RQ-VAE_iter27/`,
writes to `results/stage3_T5Train/curvature_RQ-VAE_iter27/{logs,ckpt}/`,
runs 150 epochs with beam=20 / n_eval=57439 (identical to iter18).