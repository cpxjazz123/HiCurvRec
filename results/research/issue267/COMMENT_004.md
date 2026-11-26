## Stage2 admission PASS — training launched

- Real-entry smoke: PASS, 1 epoch + SID export, collision `0.051409` (<0.40); no behaviour graph required.
- Launched four-rank DDP: `run_id=stage2_caa891559565421d9b74703b6af643c5`, launcher PID `984851`, torchrun PID `986235`, ranks `986368`–`986371`; port `50202`; seed 42; 3000 epochs, no early stop.
- Frozen source: `results/stage2_RQ-VAE/_runs/stage2_caa891559565421d9b74703b6af643c5/TIGER_relative_curvature_ranking_rq/src`.
- Result logs: same run `result/logs/`.
- Relative calibration is executed before the parent structural budget; pre-Stage3 `SCORE_PREDICTION.json` will be computed before Stage3 or any new Recall read.
