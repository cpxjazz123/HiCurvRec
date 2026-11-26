## Stage2 complete

- Run: `stage2_88816eeca29442c7b3284f4d4a68f610`; 3000 epochs, four ranks, seed42.
- Final collision: `0.04534916825964941` (<0.40); SID shape `[24587, 4]`, `unique_after_extend=24587`, final-epoch checkpoint.
- Wall: `811.659 s` inclusive; mean epoch from metrics ~`0.245` s; relative calibration at init alpha `0.3075788718799832`, 378 valid same-centre pairs.
- SID/checkpoint: `results/stage2_RQ-VAE/_runs/stage2_88816eeca29442c7b3284f4d4a68f610/TIGER_relative_curvature_ranking_rq/result/`.
- Next: pre-Stage3 score prediction before any new Recall read, then Stage3.
