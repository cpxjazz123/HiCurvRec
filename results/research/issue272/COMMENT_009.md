### Round-3 final table (late evidence, measured after the closure comment)

The round-3 run (`stage2_0a2be6e306e145fe91b3d7be08a37c69`, `TIGER_codebook_budget_j_rq`, 3000 epochs,
seed 42) finished after the closure comment was posted. Its final PASS table is
`results/research/issue272/round3_pass_table.json` (the common-ruler J comes from the candidate's own
`model/global_constraint.py`; the local and within-centre statistics come from
`results/research/issue273/within_centre_metrics.py`, whose self-check reproduces the frozen #271
evaluator: Spearman delta 0, sign delta 0, MAE delta 4.4e-7 from the frozen float16 storage).

PASS: C2 Spearman 0.079759 (> 0.015851), C4 variance ratio 1.320687 (>= 0.5), C5 slope 0.110958
(>= 0.1), C7 within-centre ordering 0.142088 (> the #269's 0.106457), C9 full common-ruler J
0.220079 (<= 0.338336), C15 reconstruction 7.7089e-5 (<= 8.589e-5).
FAIL: C1 local MAE 2.459864 (<= 1.893506), C3 sign 0.404876 (> 0.432453), C6 difference RMSE
2.286232 (vs the #269's 2.002453), C10 one-hop J 1.919914 (<= 0.798295), C11 two-hop J 0.331553
(<= 0.278106), C12 L2 distinct prefixes 7,463 (>= 10,918), C13 effective rank 2.213754 (>= 8),
mu active 26/3000 = 0.87%. Collision 0.133689 (recorded only, not a criterion).
`all_pass` = false: a negative result, consistent with the closure comment.

Transfer to Issue #273: the codebook-side gradient budget restores the ordering and the dispersion
(Spearman 0.079759 vs the TIGER's 0.015851, variance ratio 1.320687 vs the #269's 0.42, within-centre
ordering 0.142088 vs the #269's 0.106457) and lowers the global J (0.220079), but it does not restore
the signed differences (C6 2.286232 vs 2.002453), the local MAE or the SID capacity. The missing
statistic is therefore the signed within-centre difference, not the ordering objective — which is
exactly the term Issue #273 adds, leaving the codebook budget untouched.
