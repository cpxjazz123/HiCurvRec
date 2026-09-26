# git_closure_iter28

## Repository truth (pre-push)

```
$ git rev-parse main
bdcbbf97bd92ef6b5f0ab016cdfbae2b62c5d73f

$ git ls-remote origin refs/heads/main
<remote HEAD>

$ git status --short (before iter28 commit)
 M .claude/skills/curvature-rqvae-iter/scripts/preflight_cao1.py
 M .claude/skills/curvature-rqvae-iter/scripts/deliberation_gate.py
?? stage2_RQ-VAE/curvature_RQ-VAE_iter28/                (new directory)
?? results/stage2_RQ-VAE/curvature_RQ-VAE_iter28/         (new — Stage2 outputs)
?? results/stage3_T5Train/curvature_RQ-VAE_iter28/        (new — Stage3 outputs)
```

## Mandatory artifacts present in iter28 tree

- `stage2_RQ-VAE/curvature_RQ-VAE_iter28/`:
  - `curvature_RQ-VAE.py` (modified: builder + post-step block)
  - `curvature_config.py` (MECHANISM_NAME updated)
  - `modules/quantize.py` (SREMA method + cache_srema_inputs + forward cache call)
  - `modules/rqvae.py` (srema_update wrapper + clear_srema_inputs)
  - `configs/decoder_instruments_hgrec_iter28_sinkhorn_riemannian_ema_codebook.gin`
  - `scripts/run_stage3_iter27.py` (Stage3 launcher)
  - `scripts/mvg_check.py`
  - `logs/source_snapshot_iter28.md`
  - `logs/protocol_manifest_iter28.md`
  - `logs/hypothesis_iter28.md`
  - `logs/mechanism_manifest_iter28.md`
  - `logs/mechanism_contract_iter28.json`
  - `logs/one_factor_diff_iter28.md`
  - `logs/implementation_plan_iter28.md`
  - `logs/preflight_contract_iter28.log` (CAO-1 PASS)
  - `logs/mvg_check_iter28.log` + `mvg_check_iter28.md`
  - `logs/sid_geometry_iter28.md`
  - `logs/stage2_execution_plan_iter28.md`
  - `logs/iteration_bridge.md`
  - `logs/gate_decision_iter28.md`
  - `logs/failure_attribution_iter28.md`
  - `logs/stage3_outcome_iter28.md`
  - `logs/stage3_evaluation_plan_iter28.md`
  - `logs/git_closure_iter28.md` (this file)
  - `logs/deliberation/S00..S09/round_1/` (canonical 2+1, MERGE_AB)
  - `logs/deliberation/S08_MVG/round_2/` (abort / MVG rerun, MERGE_AB)

- `results/stage2_RQ-VAE/curvature_RQ-VAE_iter28/`:
  - `out/rqvae/instruments/rqvae_best.pth` (Stage2 best checkpoint)
  - `out/rqvae/instruments/sids_raw.npy` (3-token SIDs, (24587, 3))
  - `dataset/Instruments/sids_for_hgrec.npy` (4-token SIDs, (24587, 4))
  - `item_sids.json` (TIGER-compatible JSON)

- `results/stage3_T5Train/curvature_RQ-VAE_iter28/`:
  - `ckpt/Amazon_2023_Instruments/Sep-27-2026_00-15-26/HG_Rec_best.pth`
  - `logs/Amazon_2023_Instruments/Sep-27-2026_00-15-26/test_final.json`
  - `logs/Amazon_2023_Instruments/Sep-27-2026_00-15-26/training_metrics.jsonl`
  - `logs/Amazon_2023_Instruments/Sep-27-2026_00-15-26/HG_Rec.log`
  - `logs/_stage3_launcher.log`

## Final closure-mode gate (per SKILL.md §19)

```
$ cd stage2_RQ-VAE/curvature_RQ-VAE_iter28
$ /home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 \
    /home/wlia0047/ar57/wenyu/GeneRec/.claude/skills/curvature-rqvae-iter/scripts/deliberation_gate.py
  → see logs/ for actual output
```

Expected: `DELIBERATION_GATE_PASS` with `phase=CLOSURE` (iter28 was **not** aborted, so all S10..S13 stages must also pass).

## Git push plan (mandatory per CLAUDE.md §13)

1. `git status --short` — confirm no untracked iter28 paths.
2. `git add` (3 categories per CLAUDE.md §13):
   - iter28 source (`stage2_RQ-VAE/curvature_RQ-VAE_iter28/`)
   - iter28 Stage2 outputs (`results/stage2_RQ-VAE/curvature_RQ-VAE_iter28/`)
   - iter28 Stage3 outputs (`results/stage3_T5Train/curvature_RQ-VAE_iter28/`)
3. `git commit -m "iter28: SREMA codebook update — R@10=0.0565 (NO-GO) | iter18 baseline 0.0599 | CAO-1 ACTIVE_NEGATIVE + PROMOTION_FAIL"` (single semantic commit).
4. `git push origin main` (no force).
5. Verify: `git rev-parse main` matches `git ls-remote origin refs/heads/main`.

If push fails (e.g. remote advanced again), `git fetch origin && git merge --ff-only origin/main` first, then retry push.