# direction_decision_iter10 (Agent B — direction judge)

Inputs:
- Bottleneck (Agent G iter9): iter8's linear ε is local optimum; iter9 power-law ε regressed. Need to refine a *different* quantization aspect.
- Failed-mechanism ledger: iter9 power_law_sinkhorn (类 3, count 1) — different mechanism refinement within the same family, regressed.
- iter8 (类 3) is still the best Stage-2 mechanism.

Scoring:

| Candidate | (a) fixes bottleneck | (b) paper | (c) curriculum compat | (d) Stage-1 ceiling risk | (e) Gap-closing | (f) Curvature-relevant | Total |
|---|---|---|---|---|---|---|---|
| P10A (5 iters + iter8 ε) | yes — orthogonal refinement | yes (Cuturi 2013, Geneva 2022) | high | low | yes — "refine different quantization aspect" | yes — assignment | 6/6 |
| P10B (variable iters per layer) | yes — but structural change | weak | medium | high | partial | yes | 3/6 |
| P10C (8 iters) | yes — diminishing returns | medium | high | medium | partial | yes | 4/6 |

Recommendation: **P10A** (sk_iters 3 -> 5, otherwise iter8's mechanism unchanged).

Reasoning: iter8 broke the ceiling by +0.0030 with linear ε; power-law refinement hurt. A different quantization aspect — convergence count of the Sinkhorn iteration — is the natural next step. 5 iters is the canonical value recommended by Cuturi 2013 / Geneva 2022 for adaptive-ε setups. Iteration time increases by ~30%, but the gain should be in tighter per-layer assignments.

Backup: P10C (8 iters) if P10A doesn't move the needle.

Decision: iter10 is `sinkhorn_iters_5`, single mechanism P10A.