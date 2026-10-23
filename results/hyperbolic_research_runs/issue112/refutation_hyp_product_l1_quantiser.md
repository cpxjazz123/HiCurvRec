# The approved candidate's cold run: REFUTED by its own run

`hyp_product_l1_quantiser` (the role-supervised product-geodesic level-0 quantiser) was launched
on 2026-10-23 at 23:19 through the sanctioned path (the MCP's `stage2_start`, which ran the six CGV
gates itself and passed them), with the operator's authorisation, seed 42, 3000 epochs, cold
initialisation, 4-GPU DDP. The reservation was taken first: `rounds = 1`.

**The run finished. The level-0 partition collapsed. The frozen ER gate fails. There is no Stage-3.**

## 1. The measurements

| quantity | the arm | the archived control |
|:---|---:|---:|
| the level-0 codes: distinct | **1** | 245 |
| the level-0 largest cell | **24587 (every item)** | 237 |
| the level-1 codes: distinct / largest cell | 180 / 549 | 256 / 261 |
| the level-2 codes: distinct / largest cell | 256 / 220 | 256 / 293 |
| the Hard-L1 ER | **1.0** | 16.334505666275888 |
| the positive share / the negative share | 1.0 / 1.0 | 0.07908428720083246 / 0.004841547630309337 |
| the frozen gate (ER > 16.38953942232631) | **FAILED** | — |

The exported `sids_for_hgrec.npy` and `item_sids.json` agree column for column, and the level-0
column is the constant 148 for all 24587 items. The trainer's own last line agrees: `raw_unique =
13635/24587` — the three-token SIDs have only 13635 distinct values, because the level carries no
information at all.

The D2 is diagnostic only: the conditional level-2 discrimination is 1.0 on the train split and
1.0 held out (the control's ratio 0.645), which is what a collapsed level-0 makes inevitable.

## 2. Why: the level-0 assignment is a norm comparison, not a geometry comparison

The mechanism's own distance, on the trained checkpoint, with the mechanism's own function:

| | value |
|:---|---:|
| the level-0 codebook's row norms | min **0.29712**, mean 2.90704, max 3.10475 |
| the winning code | 148 |
| its row norm | **0.29712 — the smallest of all 256** (rank 0) |
| the distinct argmins over 1024 items | **1** |
| the relative distance spread (max−min)/mean | 1.0667 (the distances are *not* flat) |

So the distances do vary, yet one codeword is the argmin for every item: the codeword nearest the
origin of the Poincare ball is the argmin for points that are themselves near the origin. The
assignment compares radii; the geometry never gets to decide.

Two controls settle the cause:

* **the scale is not the culprit**: applying the tangent scale to the codewords as well as to the
  items (the one-sided scale looked like the suspect) gives the *same* single argmin, 148. The
  scale symmetry is not what breaks it.
* **equalising the codebook rows' norms** lifts the distinct argmins from 1 to **9** out of 256,
  with a largest cell of 1010 of 1024 — still degenerate. The degeneracy is the mechanism's, not one
  unlucky codeword.

The codebooks themselves are healthy: 256 distinct rows in every level, well separated (the level-0
rows' mean pairwise distance 3.84, the level-1's 0.370, the level-2's 0.123), and the level norms
shrink down the chain as they should. **The codebook did not collapse; the assignment did.**

## 3. What this refutes, and what it does not

It refutes the candidate as implemented and declared: the level-0 assignment minimises
`d_{c_s}^2(x_s, A_j) + d_{c_b}^2(x_b, B_j)` with the codebook's raw tangent rows mapped into the
ball, and that objective is degenerate on this data. The proposal's first hypothesis — that the
product-geodesic assignment changes the level-0 partition without coarsening it — is false as
stated: the assignment does not produce a partition at all.

It does **not** refute the family "a hyperbolic level-0 assignment" in general, and it does not
speak to the role supervision, whose own measurement is unaffected (the content half's gradient is
exactly 0.0 and the behaviour half's 1.10e-01).

## 4. The verification suite's blind spot, and the gate that closes it

This arm passed: the CGV's G1–G6 (G5's pooled codebook gradient within 1.490e-08 of the
independent pooled reference, the ranks bitwise identical), the H1–H7 gates, the join_R and join_E,
and the implementation-parity check. **None of them asked whether the level's assignment
distinguishes the catalogue.** The gates test finiteness, gradient reach, the update rule, the DDP
aggregation and that the real entry point moves parameters. A level that assigns every item to the
same codeword passes all of them.

`tools/harness2_genrec/partition_degeneracy.py` is that missing gate: a level must use more than
16 of its 256 codes and no cell may hold more than a quarter of the catalogue. Its tests
(`tools/harness2_genrec/tests/test_partition_degeneracy.py`, 5 passed) use the collapsed arm as the
fixture: the gate refuses it, and it accepts the archived control.

**Every future candidate must pass it before it is allowed a GPU run.**

## 5. The honest accounting

One cold run, 3000 epochs, 4 GPUs, about 19 minutes of wall time. The result is a negative one and
it is a clean one: the mechanism's assignment is degenerate, the ER collapses to 1.0, and the frozen
gate fails, so no Stage-3 is run. The campaign's guardrail forbids a second cold run of this
mechanism under any name, and the portfolio moves on.

The campaign's own prediction — "if the trained largest cell grows the way the cohort's high-ER
arms grew theirs, the mechanism will have reproduced the campaign's central failure" — was right in
kind and wrong in degree: the cell did not grow from 237 to 374, it swallowed the whole catalogue.
