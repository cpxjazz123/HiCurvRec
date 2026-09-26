# direction_decision_iter11 (Agent B — direction judge)

Inputs:
- Bottleneck (Agent G iter10): iter8's per_layer [0.20, 0.23, 0.22] is the local optimum. iter9 (power-law ε) and iter10 (5 iters) shifted L2 to dominate and regressed. iter11 must add orthogonal gain without disturbing per_layer balance.
- Failed-mechanism ledger: iter9 power_law_sinkhorn (类 3, -0.0027); iter10 sinkhorn_iters_5 (类 3, -0.0001). Same family refinements both regressed.
- iter8 (类 3) is still the best Stage-2 mechanism.

Scoring:

| Candidate | (a) fixes bottleneck | (b) paper | (c) curriculum compat | (d) Stage-1 ceiling risk | (e) Gap-closing | (f) Curvature-relevant | Total |
|---|---|---|---|---|---|---|---|
| P11A (c-mod commitment α=0.25) | yes — orthogonal, preserves iter8 balance | yes (VQ-VAE 2017, anon 2024) | high | low | yes — "orthogonal mechanism preserving balance" | yes — curvature-modulated loss | 6/6 |
| P11B (per-layer commitment) | yes — but structural change | medium | medium | medium | partial | yes | 3/6 |
| P11C (EMA codebook) | yes — but no curvature relevance | medium | medium | medium | partial | no — anti-pattern | 3/6 (anti-pattern) |

Recommendation: **P11A** (c-modulated commitment weight, α=0.25).

Reasoning: iter8's local optimum is per_layer balance. iter9 and iter10 confirmed that any change to the assignment step shifts this balance. P11A is the only candidate that touches curvature (via commitment loss weighting) without disturbing the assignment step. Mild α=0.25 scaling keeps the high-c commitment pull slightly stronger without breaking iter8's per_layer balance.

Backup: none — P11A is the only safe option.

Decision: iter11 is `curvature_scaled_commitment`, single mechanism P11A.