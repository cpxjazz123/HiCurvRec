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
