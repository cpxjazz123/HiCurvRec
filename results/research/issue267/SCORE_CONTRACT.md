# Issue #267 pre-Stage3 score contract

Frozen before the new arm's Stage2 output. This is a diagnostic formula, not an
optimization target or an admission gate.

## Frozen formula

```
S = 0.52137*(1 - N_R(R))
  + 0.19389*N_gate(log(G_gate))
  + 0.28474*N_L3(log(G_L3))
```

`R` = raw reconstruction component of the endpoint forward (not total RQ loss).
`G_gate` = raw unweighted L2 norm of the **structural loss** gradient on the
`structure` gate parameters. `G_L3` = raw unweighted L2 norm of the **RQ-VAE
loss** gradient on all L3 codebook parameters. `N_x` is the frozen min-max
normalization; no clipping.

Original ranges:

- `R`: `[0.00008205277845263481, 0.00010617653606459498]`
- `G_gate` raw: `[0.0003398736207435491, 0.05941492674715333]`
  = log range `[-7.986936713148453, -2.8232097921424755]`
- `G_L3` raw: `[0.00004965698388494281, 0.00015923990773577049]`
  = log range `[-9.910371515029908, -8.745098639235927]`

Zero/None gradients are not replaced by an epsilon; they are marked
`NOT_COMPUTABLE` and make the formula inapplicable.

## Historical scores (10 arms 100–109)

`101 .2281179533`, `104 .4475563464`, `100 .5689364228`, `107 .5782102943`,
`106 .5789591370`, `105 .5797449191`, `108 .6454995374`, `109 .6535990918`,
`103 .6698993639`, `102 .7411182932`, in Recall-ascending order. Historical
Pearson ≈ `0.928526`, Spearman=Kendall=1; minimum gap ≈ `0.000749`. These are
post-hoc diagnostic observations from the issue and are not causal evidence.

## New arm procedure

After formal Stage2, before Stage3 or reading new Recall:

1. Read the final checkpoint, fixed 4096 training batch, seed 42 and
   `draw(0,0,1,track=False)`;
2. compute R, G_gate and G_L3 with real forward/autograd, no optimizer and no
   parameter update;
3. compute S with the frozen constants, persist `SCORE_PREDICTION.json` and
   comment the issue;
4. only after that start Stage3. S is never used to accept/reject a candidate or
   to choose a mechanism weight.

The 11th arm's prediction is its S value and its position relative to the ten
historical Recall scores; out-of-range predictions are retained, not clipped.
After Stage3, report the actual rank, pairwise inversions, Pearson/Spearman/Kendall
and whether the pre-Stage3 interval was correct; no refit or deletion.
