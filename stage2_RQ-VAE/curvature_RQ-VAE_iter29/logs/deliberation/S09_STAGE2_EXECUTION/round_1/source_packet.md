# S09_STAGE2_EXECUTION round 1 source packet

ROLE=AGENT_CANDIDATE
ROUND=1
STAGE_ID=S09_STAGE2_EXECUTION

## Canonical authority

Use only Judge-approved canonical evidence through S08, root `CLAUDE.md`, `skill://curvature-rqvae-iter`, and current primary source. S07 official preflight passed and the round-2 S07 Judge accepted the exact physical input lock. S08's final round-2 Judge classified the sole one-run MVG as `MVG_PASS`; it proves implementation/activation only, not Stage2 or Stage3 performance. The canonical one-factor/control is iter26; FCCR-1 candidate values are `[1.3660953164241916, 0.7347829661951981, 0.6439958072706683]`.

## Locked Stage2 execution contract

- Source cwd: `/home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/curvature_RQ-VAE_iter29`.
- No-argument training entry: `/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 curvature_RQ-VAE.py`.
- Entry auto-forks four-rank torchrun at master port 50200; hard-coded launcher captures worker output at `logs/train_migrated.log`; outer launcher output may be retained as `logs/train_run.log`.
- Locked Stage2 settings: seed 42; 3 layers/codebook size 256; `MAX_GLOBAL_STEPS=100000`; checkpoint cadence and optimizer settings unchanged from the canonical iter29 protocol; no Stage2 SID metric gate/early-stop.
- S01 warm-start: iter8 checkpoint `/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE_iter8/out/rqvae/instruments/rqvae_best.pth`, SHA-256 `189e0affb156b9cdbd11f38a6fdb1127804e3f3095edd23c55f84825beb56c3b`. It was independently hash-checked before S08; recheck immediately before Stage2 and require exact match. After launch, require `train_migrated.log` to show the actual `iter29 warm-start loaded ... from <locked path>` line; the trainer otherwise has a random-init fallback and must not be accepted on that branch.
- Input identities from canonical S01: Stage1 embedding `stage1_GeneEmbedding/output/sentence_t5.npy` SHA-256 `6490c71753952b2d8788bb7e4e52cff6a5b0b59819a2cfbd737bee7bc07a77fb`; item-ID sidecar `.../output/item_ids.json` SHA-256 `3df1f3ce6a468ae541c929a149f4dbd5e806f545b89d786b9c2d6801563b7c30`; Stage0 train parquet `results/stage0_build_parquet/train.parquet` SHA-256 `80597e38d081029e57e7434c92ff7be2c872f82aebc2e5f072ef973d73e88815`. Recheck locked inputs as the final prelaunch audit and block on mismatch.
- `curvature_config.py` must resolve `RQVAE_OUT_DIR` to `/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE_iter29/out/rqvae/instruments`; run the required `grep RQVAE_OUT_DIR <iter29>/curvature_config.py` immediately before launch and record its actual output.
- All Stage2 outputs must be under the repository result subtree `results/stage2_RQ-VAE/curvature_RQ-VAE_iter29/`; source iter29 tree may contain code/configs/scripts/logs/inputs/pycache only. The final Stage3 SID files are separately routed to that same results iter29 root.

## Fresh-output and failure controls

The training step-zero cleaner removes only `RQVAE_OUT_DIR`. The final `item_sids.json` path is outside that cleaned directory; final 4-token export exceptions are logged but not propagated as a process failure. Therefore, before the sole Stage2 run, independently establish the iter29 result destination has no prior/stale `item_sids.json` / `sids_for_hgrec.npy` or other stale Stage2 products; do not silently consume or overwrite an unrelated artifact. Record this pre-run absence/state. After the single run, require a fresh final raw SID export, 4-token NPY/JSON and `SID_WIRING_PASS` log marker, with file timestamps/hashes or other direct evidence tying them to this run. If the export is missing, partial, stale, or the success marker is absent, block Stage3; no Stage2 retry or Stage3 consumption of stale data.

## Deliberation-gate sequencing

This S09 plan and Judge must be canonical before running the global no-argument `deliberation_gate.py` from this iteration directory. Require `DELIBERATION_GATE_PASS` for the pre-Stage2 phase and all exact S00–S09 latest-round Judge/canonical artifacts. Only after the gate passes, the final RQVAE path/hash/output checks pass, and the warm-start identity/use is assured may one Stage2 invocation be started under process supervision. Stage2 runs once; never stop based on descriptive Gini/collision/entropy/unique-SID proxies. Do not launch Stage3 from this S09 stage.

## Candidate objective and limits

Agent A and Agent B independently audit the entire one-run launch/control/observability plan, including startup paths/ports, exact output routes, protocol hashes, fresh artifact protections, warm-start proof, Step2 descriptive-only policy, and post-run stop/continue criteria. They do not run hashes, gates, tests/build/formatter, start processes, launch GPU work, or edit source. They write only their own candidate artifacts under this round. Judge C selects the canonical launch plan and writes `logs/stage2_execution_plan_iter29.md`; only after that decision may the orchestrator run the required no-argument deliberation gate and, if every preflight passes, launch Stage2 once.
