### R06b formal result: negative; graphlet-specific residual does not improve recall

- Stage2 `stage2_127a3cca78dc49bf899ded006b2a11ca`: 3000 epochs, final collision **0.1005**,
  checkpoint/SID exported; layer record `c5f0e739...` (L2 prefixes 7714, L3 collision 0.1005).
- First R06 dispatch `stage2_f704...` aborted at epoch 1334 because the raw token mean
  (norm 2.42) collapsed the L2 codebook (spacing x0.076). The scale fix uses the mean L2
  codeword norm (0.046); focused tests 19 passed.
- Token-switch ablation on the trained checkpoint: L1 changed 0/24,587; L2 changed **52.19%**;
  per-role L2 change **0.413–0.575** (P4:0 through K1_3:1). The graphlet-conditioned change
  is real at the code level.
- Stage3 final: `test_R@10` **0.049600** (R05 comparator **0.050349**, delta **-0.000749**);
  below the 0.065 adoption line. R2 archive `r2:genrec/107/`.

This is another negative operationalisation of "different graphlets need different geometric
biases": the bias changes the L2 assignment, but does not improve downstream ranking. The
matched mixed-curvature evidence remains positive (F0 vs pure E: +0.00303 recall@10,
95% CI +0.00152..+0.00456, p=1.06e-4), so the defensible conclusion is mixed curvature,
not per-graphlet geometric bias.
