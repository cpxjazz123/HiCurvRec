### Formal R05 Stage2 started — the one preregistered run

- **run id** `issue256_r05_formal`; **arm** `TIGER_role_conditioned_l2_rq`
- workspace `results/stage2_RQ-VAE/_runs/issue256_r05_formal/TIGER_role_conditioned_l2_rq/`
  (`src/` immutable snapshot, `result/` independent result directory); `FORMAL_RUN.json` records all 79 admitted
  source hashes and the admission fingerprint `a1b734e3…`
- launcher pid `3791797` (`torchrun --nproc_per_node=4 --master_port=50202`), four ranks `3791803–3791806`,
  GPUs 0–3
- recipe: 3000 epochs, per-rank batch 1024, global batch 4096, seed 42, no early stop; the arm name is the only
  identity change relative to the R04 line
- expected wall: 2067 s of epochs (0.6935 s/epoch measured over the inclusive 31–60 window) plus one-time
  initialisation and the legal final re-export
- one run only: no TIGER retraining, no seed sweep, no second R05 dispatch; this workspace is never overwritten

The run is executing now; the Stage2 metrics, checkpoint, exported SID and the collision/quality gate follow on
completion, then the single frozen Stage3 150-epoch protocol.
