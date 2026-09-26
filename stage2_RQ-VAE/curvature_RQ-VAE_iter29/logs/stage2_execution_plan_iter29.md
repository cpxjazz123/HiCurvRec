# Iter29 Stage2 execution plan — S09 canonical

## Authorization boundary

This plan authorizes only a *method*, not a present launch. No preflight check, hash, global deliberation gate, process launch, or Stage2/Stage3 execution has been run or is implied complete by this plan. Stage2 remains blocked until this plan and the S09 Judge are materialized, the no-argument global deliberation gate actually reports `DELIBERATION_GATE_PASS` for the pre-Stage2 phase, and every exact prelaunch check below succeeds. Any missing, stale, mismatched, or ambiguous evidence blocks launch; do not substitute another input or protocol.

S08 is `MVG_PASS` for bounded implementation/activation evidence only: fixed-curvature implementation and invariance, backward/model-gradient health, and matched-control counterfactual. It is not Stage2 or Stage3 performance evidence. Its record explicitly did not rehash the Stage1/Stage0 input files for S08 and emitted no batch fingerprint. S03 establishes historical method/value provenance for the locked raw residual medians, not checkpoint-level reproduction or historical byte identity. Preserve those evidence limits.

The registered protocol is `FCCR-1_iter29_bounded_alternative_closed_form_vs_iter26`, single-factor, iter26 parent/direct control, sole scientific change the registered fixed closed-form mapping. Keep the equation, physical JSON keys, values, parent, warm-start, inputs, seed, model, optimizer/loss/Sinkhorn, train duration, and all other iter26 conditions locked. Candidate fixed curvature `[1.3660953164241916, 0.7347829661951981, 0.6439958072706683]`; ordered inputs are `branching=[19.324911558712664, 1.4605688962651735, 1.0148104414712726]` and physical key `raw_residual_medians=[1.0, 0.10941, 0.09331]`. Do not regenerate the historical input JSON with the iter26 writer or fall back to normalized residual scales.

## Required prelaunch authorization and exact checks — all still pending

Perform only after this canonical plan and its Judge exist:

1. **Global deliberation gate.** From `/home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/curvature_RQ-VAE_iter29`, run the skill's exact no-argument `deliberation_gate.py` invocation. Require observed output `DELIBERATION_GATE_PASS` for the pre-Stage2 phase and the completed latest-round S00–S09 Judge/canonical chain. Record actual output. Missing/wrong phase/non-PASS blocks launch; candidate agreement alone is insufficient.
2. **Config and destination.** Run the root-required `grep RQVAE_OUT_DIR /home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/curvature_RQ-VAE_iter29/curvature_config.py`, record the actual matching line, and confirm it is exactly `/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE_iter29/out/rqvae/instruments` (inside the repository-level `results/stage2_RQ-VAE/curvature_RQ-VAE_iter29/` tree, iter number 29). Confirm `RAW_SIDS_NPY`, `SIDS_NPY`, `ITEM_SIDS_JSON` resolve respectively to that out directory's `sids_raw.npy`, `.../results/stage2_RQ-VAE/curvature_RQ-VAE_iter29/dataset/Instruments/sids_for_hgrec.npy`, and `.../results/stage2_RQ-VAE/curvature_RQ-VAE_iter29/item_sids.json`. No model/data artifact may be written into the source iter29 tree.
3. **Final input identity.** Immediately before starting the one process, calculate and record SHA-256 of each exact file below, verify the configured consumer paths match these paths, and require exact equality to S01. Any mismatch, absent file, or path mismatch blocks execution; never substitute an input:
   - `/home/wlia0047/ar57/wenyu/GeneRec/stage1_GeneEmbedding/output/sentence_t5.npy` — `6490c71753952b2d8788bb7e4e52cff6a5b0b59819a2cfbd737bee7bc07a77fb`.
   - `/home/wlia0047/ar57/wenyu/GeneRec/stage1_GeneEmbedding/output/item_ids.json` — `3df1f3ce6a468ae541c929a149f4dbd5e806f545b89d786b9c2d6801563b7c30`.
   - `/home/wlia0047/ar57/wenyu/GeneRec/results/stage0_build_parquet/train.parquet` — `80597e38d081029e57e7434c92ff7be2c872f82aebc2e5f072ef973d73e88815`.
   - Warm-start `/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE_iter8/out/rqvae/instruments/rqvae_best.pth` — `189e0affb156b9cdbd11f38a6fdb1127804e3f3095edd23c55f84825beb56c3b`.
   These are final prelaunch rechecks; the S08 record is not a substitute.
4. **Warm-start use.** Confirm the locked checkpoint exists and its final SHA-256 matches. During the only run, `logs/train_migrated.log` must contain the actual line beginning `[Step4] iter29 warm-start loaded`, naming that exact iter8 checkpoint and reporting a positive number of transferred tensors. Trainer source skips only the legacy `layers.{0,1,2}.c_layer_scale` and `layers.{0,1,2}._fixed_c` keys. The missing-checkpoint branch instead warns that training will use random initialization. That warning, missing loaded line, wrong path, zero transfer, or unexpected warm-start error invalidates this protocol run; no retry/relaunch.
5. **Fresh-output/stale-output protection.** Before launch, inventory and record absence or exact pre-run identity (path, size, mtime, SHA-256) for every relevant iter29 target: the entire `RQVAE_OUT_DIR` Stage2 products including `rqvae_best.pth`, `rqvae_step_*.pt`, `sids_raw.npy`/other SID outputs; external `results/stage2_RQ-VAE/curvature_RQ-VAE_iter29/item_sids.json`; and external `.../dataset/Instruments/sids_for_hgrec.npy`. Resolve any pre-existing external JSON/NPY by establishing absence or safely quarantining/preserving it so it cannot be mistaken for this run's result; if stale identity/freshness cannot be unambiguously separated, block launch. Record Step0 cleanup's actual count. Step0 clears allowlisted files only *inside* `RQVAE_OUT_DIR`; it does not clear the external Stage3-facing JSON or NPY. Never rely on the cleaner or eventual file existence alone to prove external export freshness.
6. **Locked settings and one-factor check.** Confirm source/config still encode seed 42, three layers, codebook size 256 each, `MAX_GLOBAL_STEPS=100000`, same registered model/protocol (including batch and optimizer), fixed candidate curvature, and four ranks on port 50200. Python receives no CLI flags/overrides. The final S08 evidence supports the required gradient-path check; it establishes total-loss/backward and nonzero trainable model gradients, not gradient on fixed curvature (which must remain non-trainable). Any changed mechanism/protocol, extra gate/loss, or contract drift blocks launch.

## Exactly one supervised foreground Stage2 invocation

Do not run `nohup ... &` from an unsupervised shell, and do not use shell backgrounding, retries, automatic restarts, or a second invocation. The root-required no-argument Python entry is retained, but run it as a foreground process owned/monitored by the project process supervisor (`hub start`) so process identity, logs, and terminal exit can be observed. Suggested supervisor launch specification (from the source iter29 cwd; execute only after all gates above pass):

```text
hub start
  name=iter29-stage2
  cwd=/home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/curvature_RQ-VAE_iter29
  application=/bin/bash
  args=["-c", "exec /home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 curvature_RQ-VAE.py > logs/train_run.log 2>&1"]
  pty=false
```

The shell uses `exec` and no `&`: the supervised foreground process becomes the exact Python entry; Python's only argument is the script path (no training CLI arguments). The redirect preserves root-required outer `logs/train_run.log`; the trainer's hard-coded torchrun child captures rank/worker output in `logs/train_migrated.log`. Its internal launcher uses `/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/torchrun --standalone --nproc_per_node=4 --master_port=50200 <absolute script> -u`. Do not invoke torchrun directly or alter its arguments. Record supervisor process name/identity, start time and eventual exit status; follow both logs and the supervised process to terminal exit. No restart policy; on failure preserve evidence and stop this run only.

## In-run validity and completion criteria

Require four-rank DDP startup and expected input loading. Require the actual warm-start loaded-line evidence above. Observe no unexpected crash, nonzero exit, NaN/Inf, backward/optimizer/check failure, invalid/corrupt checkpoint, fixed-curvature contract failure, input/protocol deviation, or failed final 3-token SID generation. Require logged fixed-curvature invariance at global steps 0, 25,000, 50,000, and 100,000, each matching the registered vector within `1e-6`, and completion at `global_step=100000` with final checkpoint save and final raw 3-token SID generation.

There is **no Stage2 proxy/performance gate** and no early stopping because of Gini, collision rate, unique SID counts, layer utilization, entropy, coarse/fine ratios, HitRate@50, or any other descriptive statistic. Record those only descriptively. Continue to the locked 100,000 steps unless actual execution invalidity or a skill §2.9 feasibility-abort condition occurs. Never adjust the registered mechanism or protocol during a run. A failed/invalid attempt is recorded as such and is not repeated.

## Exact post-run output and export verification

Successful process exit and checkpoint presence alone do not establish valid export: the trainer catches/logs exceptions from final 4-token export and does not re-raise them. Before declaring Stage2 completion or permitting any later analysis, inspect both logs and bind every output to this sole invocation using pre-run state plus actual post-run timestamps, sizes and SHA-256 values:

- Fresh `/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE_iter29/out/rqvae/instruments/rqvae_best.pth`, saved at final global step 100000, and fresh raw `/.../out/rqvae/instruments/sids_raw.npy` with shape `N×3` and valid token range; include any retained `rqvae_step_*.pt` in the output inventory.
- Fresh `/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE_iter29/dataset/Instruments/sids_for_hgrec.npy`, shape `N×4`, and fresh `/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE_iter29/item_sids.json`; verify JSON item coverage and every row corresponds to the 4-token NPY (not just the sampled rows checked internally by export). Check row alignment with the 3-token raw SID source and ensure output mtime is after the supervised run start; record actual output digests (there are no preregistered expected output hashes).
- Exact worker-log marker `[export] SID_WIRING_PASS: 4-token extension npy + JSON 校验通过`, plus absence of `[Stage3 export FAIL ...]`; marker and fresh file evidence are both mandatory because external outputs can survive Step0 cleanup.
- Terminal `[train] done at global_step=100000` evidence, successful supervised-process exit, warm-start loaded line, expected fixed-curvature invariant lines, and no contract/error markers.

Missing, stale, partial, mismatched, corrupt, unmarked, wrong-path, or ambiguous output invalidates Stage2 completion and blocks downstream work. Do not retry to obtain missing outputs; preserve logs and outputs as failure evidence. No S09 action may launch or evaluate Stage3.

## Downstream boundary

Even if Stage2 and export verification pass, do **not** launch `scripts/run_stage3_iter29.py`, `stage3_T5Train/train_HG-Rec.py`, or any Stage3/evaluation process now. First complete and adjudicate S10 Stage2/SID analysis using actual verified outputs; only then can S11 independently audit and Judge-authorize Stage3 wiring/evaluation. Stage3 is a separate later one-run stage. S08 MVG and this S09 decision establish neither performance nor Stage3 authorization.

## Primary evidence used

Root `CLAUDE.md` §§0–3, 6, 10–11; `skill://curvature-rqvae-iter` §§2.5, 2.7, 2.9, 10, 17–18; iter29 S01 protocol manifest and S00–S08 canonical Judges; S07 preflight log; S08 MVG Judge/log; iter29 `curvature_config.py`; `curvature_RQ-VAE.py` (warm-start, fixed-curvature checks, Step0, export and launcher); `modules/step_checks.py` (Step0 allowlist); and `scripts/run_stage3_iter29.py` (routing only). This is an audit plan, not evidence that pending gates/checks have been run.