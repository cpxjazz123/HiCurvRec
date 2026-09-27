# direction_decision_iter13 (Agent B — direction judge)

Inputs:
- Bottleneck (Agent G iter12): 12-iter cap reached. User elected to continue Stage-2 micro-tuning. iter8 still best (test_R@10=0.0595).
- Failed-mechanism ledger: iter9 (power-law ε, regressed), iter10 (5 iters, regressed), iter11 (c-mod commitment, +0.0003), iter12 (period 50k, regressed). All within same family refinements.

Scoring:

| Candidate | (a) fixes bottleneck | (b) paper | (c) curriculum compat | (d) Stage-1 ceiling risk | (e) Gap-closing | (f) Curvature-relevant | Total |
|---|---|---|---|---|---|---|---|
| P13A (per-layer log_tau_l) | yes — orthogonal learnable parameter | yes (Hinton 2015, anon 2024) | high | low | yes — "orthogonal learnable param" | yes — quantization temperature | 6/6 |
| P13B (per-layer fixed τ) | partial — no learnability | medium | high | medium | partial | yes | 4/6 |
| P13C (curriculum τ) | yes — but complex | weak | medium | high | partial | yes | 3/6 |

Recommendation: **P13A** (per-layer learnable log_tau_l).

Reasoning: iter11's orthogonal mechanism (c-modulated commitment) only added +0.0003. P13A is a stronger orthogonal mechanism: a learnable per-layer temperature that the optimizer can tune during the 100k step budget. Initialized to identity (log_tau_l=0), so iter13 starts exactly at iter8's mechanism and diverges only if the optimizer finds a better local optimum.

Backup: none.

Decision: iter13 is `per_layer_temperature`, single mechanism P13A.