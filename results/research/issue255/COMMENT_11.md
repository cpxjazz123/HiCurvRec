## C9: corrected production-path four-rank diagnostic PASS

Command (required Stage2 Python): `-B -m tools.issue255.launch_validation`, exit0. Run `issue255_validation_3054403_583806839`, actual worker PIDs3055842–3055845; all four rank reports say `pass`.

Actual full per-rank training batch1024, original+structural training losses, backward/clip/optimizer update, DDP synchronized parameter hashes, checkpoint reload/export and switches were exercised. Every rank's three hard cumulative prefixes exactly decoded its actual selected codes. Attention retained four real tokens plus null and normalized over that axis.

Each individual real layer loss reached encoder, its current codebook and GNN/Q/K/V/score; future-book direct gradients were zero for earlier prefixes. Rank0 encoder norms before λ/3:12.0897124,42.5807752,34.6405166; after λ/3:0.000445514,0.001569129,0.001276526. Cosines against the actual reconstruction component:0.2768325,0.1877833,0.0822781. These are rank0 diagnostic-batch observations, not catalogue-level or final-training claims.

Evidence: `results/research/issue255/VALIDATION_MILESTONE.json` and its four exact report hashes; raw reports under `validation/issue255_validation_3054403_583806839/`. Current-source complete test receipt and dedicated profile/60-epoch inclusive performance checks are still pending; corrected formal training has not started.
