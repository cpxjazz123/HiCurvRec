# BEHDIR-L4 -- round three, implementation state

Selected by `round_20261031_065821_615` (one first place, mean rank 2.0,
IN_SCOPE). The mechanism adds a FOURTH SID level (64 codewords) whose direction is
the item's train-only successor profile: a rank-32 PPMI embedding of its row of
the symmetrised transition matrix, gated into the tangent direction
(`u_i = normalise(g_i * vhat_i + h_i)` with `g = 1 + tanh(W b)`, `h = P b`,
`P` and `W` starting at zero) and followed by a curvature-step arrival point
`q = exp_{p_i}(s u_i)` whose nearest codeword, measured in the Poincare metric,
becomes the fourth token. The behaviour profile is built from the training
transitions only (PPMI + truncated SVD rank 32, row-normalised).

## What is done

* The round-two mechanism (curved chain, model wiring, tests, trainer,
  precompute) was merged into this tree as the baseline, per §19b.
* `CODEBOOK_NUM = 4`, `CODEBOOK_SIZE = (256, 256, 256, 64)`; the chain's codebooks
  became a per-level `ParameterList` (the levels do not share a size).
* The successor profile was added to the precompute (`behaviour` in
  `behaviour_constants.npz`) and bound on the model
  (`set_behaviour_profile`).
* The fourth level is implemented and the tree parses; the constructor and the
  chain were smoke-tested on tensors of two different codebook sizes.

## What does not work yet

The four-level entry point writes `train_start` (84 s) and then produces **no
`train` event**: the first epoch does not complete, the process has to be killed.
The two-level constructor is fine, so the failure is inside the four-level step --
the most likely candidates are the `(batch, 64, 32)` log-map expansion or the
per-step codeword broadcast interacting with DDP. This is a debugging state, not a
result: **round three has no Stage 2 SID, no Stage 3 run and no recall value.**

## Where the two completed rounds stand (for context)

| round | mechanism | L1/L2/L3 prefixes | test_recall@10 |
| --- | --- | --- | --- |
| one | curved chain on the frozen text geometry | 251/1769/2201 | 0.05439 |
| two | concentration-scaled geodesic step to the successor barycentre | 230/1251/1298 | 0.05165 |
| gate | -- | -- | **> 0.065** |
