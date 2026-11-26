## Close: negative result, all acceptance items complete

The single mechanism (same-centre relative sectional-curvature ranking on the
hard q) was implemented, trained once under the frozen protocol and evaluated
once; no retry, seed search or weight selection.

- Stage2: `stage2_88816eeca29442c7b3284f4d4a68f610`, 3000 epoch, 4 ranks,
  seed42, collision `0.04534916825964941`, wall `811.659 s`.
- Stage3: `test_R@10 0.05078431031180905` vs parent #265 `0.052594926791900976`
  (`-3.4426%`), `n_eval 57439`.
- Full-table diagnostics improved (q Pearson `+0.02130` vs `-0.03310`,
  within-centre median `0.09119` vs `0.08695`) but did not convert to recall.
- Frozen pre-Stage3 score S `1.2921` again out of range (predicted highest,
  actual 6/11), 11-arm Pearson `0.5135`, Spearman `0.8636`, Kendall tau `0.8182`.
- Evidence committed at `98e261a4`; R2 `r2:genrec/110/`; infra end diff PASS.
