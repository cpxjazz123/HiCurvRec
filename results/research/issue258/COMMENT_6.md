### Stage3 started — one frozen 150-epoch test

- variant `TIGER_role_conditioned_l2_rq`; pid `4141232`; mode `ddp`, four visible cards (0–3)
- SID: `results/stage2_RQ-VAE/_runs/issue256_r05_formal/TIGER_role_conditioned_l2_rq/result/item_sids.json`
  (sha256 `dc206dba…`), Stage2 run id `issue256_r05_formal`
- archive binding PASS (`28055c28…`); layered metrics reused (`14ade7f4…`)
- collision gate: **0.055273 < 0.40**, no waiver; quantisation error 0.007134, effective rank 4.333
- frozen protocol verified on the actual trainer: `NUM_EPOCHS=150`, `NO_EVAL=True`, `SKIP_TEST=False`,
  `SEED=42`, `BEAM_SIZE=20`; no early stop
- log: `results/stage3_T5Train/TIGER_role_conditioned_l2_rq/logs/_stage3_run.log`

The test result is read once after the frozen protocol; it is exploratory (the test split has been used by all
earlier arms) and does not change the already-falsified Stage2 signature.
