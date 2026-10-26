# round_20261027_103131_805 — symmetric coherence as the objective (iteration 6)

## What the round was for

Iteration 5 established that the **directed** behaviour structure and the **symmetric**
coherence metric are different signals: five designs that injected predecessors, successors,
transition masses or hard negatives into the assignment all pushed `cooccur_overlap_L3` down
(0.0584, 0.0571, 0.0596 against the plain baseline's 0.0756), while the only symmetric design
in this repository raised it (0.0756 -> 0.0786-0.0820).

This round therefore inverted the composition, and the objective specified it exactly:

    k* = argmax_k [ -d_c(z_i, v_k)^2 / T + lambda * <A_i, g_k> + eps * directed_(i,k) ]

with `A_ij` the frozen train-only **symmetric** co-consumption cosine, `g_k` the mean
co-consumption row of the items currently held by code `k` (refreshed per epoch), `d_c` the
Poincare geodesic distance, `lambda` about 1 and `eps` at most `0.01 * lambda` so that the
directed signal can only break near-ties and cannot move the partition. Three controls were
required: behaviour-off, geometry-off and **directed-off** (`eps = 0`).

Five ideation proposals came back, all in that family: `profile-curved-assignment`,
`profile-curved-residual`, `cogeo-region-argmax`, `symgeo-profile-memory` and
`symgeo-region`.

## Why the round stopped at scoring

The scoring phase raised `AGENT_INFRA_ERROR`: reviewer `scoring_2` broke the rubric schema
twice — its `SCORE:` line for `profile-curved-assignment` carries **no rationale for the
`novelty` dimension** (`novelty=UNKNOWN` in the first attempt, a bare `novelty=2` in the
retry), and the rubric validator requires a rationale per dimension. The retry reused the same
reviewer with the same prompt and produced the same malformed line, so the round was stopped
for the operator instead of spending more reviewer runs on it.

Under issue #121 rule 19c this is an agent infrastructure error: it is retried and must not be
read as a scientific verdict on the five proposals. The fix belongs in the rubric prompt: ask
for `dimension=score | rationale` for **every** dimension including `novelty`, and let the
validator reject a bare `novelty=2` exactly as it rejected `novelty=UNKNOWN`.

## What is left to do

1. fix the rubric prompt and re-run the scoring phase of this round (the five proposals and
   their evidence bundles are on disk and need no regeneration);
2. implement, verify and train the selected proposal with the frozen budget;
3. apply the gate before spending Stage 3: `cooccur_overlap_L3` must clear the plain
   baseline's 0.0756, the target band being 0.15 and the best observed value 0.168.
