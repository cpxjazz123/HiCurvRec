# BEHDIR-L4 -- round three

Selected by `round_20261031_065821_615` (one first place, mean rank 2.0,
IN_SCOPE). The mechanism adds a FOURTH SID level (64 codewords) whose direction is
the item's train-only successor profile: a rank-32 PPMI embedding of its row of
the symmetrised transition matrix, gated into the tangent direction
(`u_i = normalise(g_i * vhat_i + h_i)`, `g = 1 + tanh(W b)`, `h = P b`, `P` and
`W` zero-initialised) and followed by a curvature step whose arrival point
`q = exp_{p_i}(s u_i)` is quantised in the Poincare metric. The level has its own
quantisation term, because an argmin carries no gradient.

## Defects found and fixed while wiring it

1. the codebooks could not be one stacked tensor (levels 256/256/256/64) -- they
   are a per-level `ParameterList` now;
2. `_restart_dead_codes` still wrote through `codebooks.data`, which does not
   exist on a `ParameterList`;
3. the fourth codeword received no gradient at all (argmin), so the level was
   pinned at its random initialisation; it now carries the standard vector
   quantisation pair. Verified: `P` gradient 1868.9, `W` 172.9, level-4 codebook
   0.0399 on a batch of 32.

## Stage 2 (four GPUs, 3000 epochs, 584.4 s)

| level | distinct prefixes | largest group |
| --- | --- | --- |
| L1 | 208 | 727 |
| L2 | 2,766 | 268 |
| L3 | 6,235 | 229 |
| L4 | 6,302 | 229 |

Width 5 (four tokens plus the disambiguation token), `final_collision = 0.744`,
every item covered. The partition is **much deeper** than either earlier round
(round one 2,201 L3 prefixes, round two 1,298), which is exactly the capacity
direction the evidence pointed at. Technical validity passed with the best L1
Hard ER of the three rounds: 4.92 (round two 5.20, round one 6.09).

## Stage 3

Launched under the frozen protocol (150 epochs, seed 42, beam 20, no mid-run
eval, test on), four GPUs, pid 954286. The recall reading is pending.

## The result

Stage 3 finished all 150 epochs (2518 s) and wrote the test record at epoch 150:

| metric | value |
| --- | --- |
| test_recall@5 | 0.03236 |
| **test_recall@10** | **0.04823** |
| test_ndcg@5 | 0.02155 |
| test_ndcg@10 | 0.02668 |

**The target 0.065 was NOT reached, and this is the worst of the three rounds.**

## The three rounds together

| round | mechanism | L1 / L2 / L3 / L4 prefixes | test_recall@10 |
| --- | --- | --- | --- |
| one | curved chain on the frozen text geometry | 251 / 1,769 / 2,201 / -- | 0.05439 |
| two | concentration-scaled geodesic step to the successor barycentre | 230 / 1,251 / 1,298 / -- | 0.05165 |
| three | fourth level driven by the PPMI successor profile | 208 / 2,766 / 6,235 / 6,302 | 0.04823 |
| gate | -- | -- | **> 0.065** |
| frozen baseline | plain three-level RQ-VAE on the text geometry | -- | about 0.0679 |

**The capacity hypothesis is refuted.** Round three produced the deepest partition
of the three (6,302 distinct fourth-level prefixes against round two's 1,298
third-level ones) and scored the LOWEST. Deeper prefixes did not buy recall; the
three monotonically decreasing scores track how much each mechanism moved the
quantizer away from the plain text geometry, not how many prefixes it created.

Consistent with issue 126, where every behaviour-aware SID-quality metric that
worked in the offline study failed to transfer into a training gain: what predicts
SID quality from the outside is behaviour contrast, but rewriting the SID
construction with that signal has now cost recall three times in a row.
