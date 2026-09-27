# direction_decision_iter9 (Agent B — direction judge)

Inputs:
- Bottleneck (Agent G iter8): remaining 0.0055 gap to 0.065; iter8 broke ceiling by +0.0030 via c-linear Sinkhorn ε; refining the schedule is the natural next step.
- Failed-mechanism ledger: iter8 curvature_dependent_sinkhorn (类 3, count 1) — but with positive delta, mechanism is in the right direction.
- Mechanism pool: P9A (power-law ε), P9B (clamp lower), P9C (per-codebook adaptive).

Scoring (a–f):

| Candidate | (a) fixes bottleneck | (b) paper support | (c) curriculum compat | (d) Stage-1 ceiling risk | (e) Gap-closing | (f) Curvature-relevant | Total |
|---|---|---|---|---|---|---|---|
| P9A (α=0.5 power-law) | yes — direct amplification of iter8's mechanism | yes (Geneva 2022, SGDR 2017) | high — same c schedule | medium (high) | yes — "amplify iter8 mechanism" | yes — schedule/assignment | 6/6 |
| P9B (clamp ε lower) | partial — only changes minimum | weak | high | low | partial | yes — schedule | 3/6 |
| P9C (per-codebook adaptive) | yes — but structural change at Quantize | weak | medium | high (high) | partial | yes — codebook | 3/6 |

Recommendation: **P9A** (power-law ε with α=0.5).

Reasoning: iter8 broke the ceiling via `effective_eps = sk_eps * (c / c_max)`; the linear factor only mildly softens at low c (ratio c_min/c_max ≈ 0.033). Replacing linear with a power-law `** 0.5` doubles the contrast at high c and triples the contrast at low c, amplifying the cycle without numerical instability (α=0.5 keeps the curve bounded between 0 and 1).

Backup: P9B (clamp lower) if P9A doesn't break the ceiling further.

Decision: iter9 is `power_law_sinkhorn`, single mechanism P9A with α=0.5.