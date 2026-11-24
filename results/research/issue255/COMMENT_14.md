## Corrected formal Stage2 started: `issue255_formal_r1`

The dispatcher accepted all exact current-source credentials, froze the immutable generation and launched the sanctioned four-rank trainer. Workers3140077–3140080 occupy the four L40S cards. Observed epoch29/3000, loss0.0006837871624156833 (finite); no completed-run claim.

Run root: `results/stage2_RQ-VAE/_runs/issue255_formal_r1/TIGER_prefix_curvature_gate_rq/{src,result}`. Frozen recipe:3000epochs, four ranks, per-rank1024/global4096, seed42, port50202. `FORMAL_RUN.json` binds `DISPATCH_CREDENTIALS.json` SHA-256 `1c7cf29c6a154d20a0718c258d41c8bd62d2b333af0c2fe07cb428c6a1556ad1`.

Actual initialization used the corrected full-original gradient: norm0.0011229729279875755 versus structural12.809191703796387; λ=0.00008766930113779381. Reconstruction-only norm0.00019794188847299665 is recorded separately and is not the numerator. Per-level frozen scales:0.350431889295578,0.3611699044704437,0.4350320100784302. Executed `model/mechanism.py` SHA-256:`7cea8be51adcaee041da2035d2b8285d46488f8197f4460726a13d0c92da1c28`.

Evidence: `FORMAL_RUN.json`, `DISPATCH_CREDENTIALS.json`, `FORMAL_HEALTH.jsonl`, and the generation's `result/logs/training_metrics.jsonl`. Stage3 and full final evaluation await completion of this one corrected run. The invalid first generation is preserved and excluded.
