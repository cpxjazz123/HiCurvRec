# direction_decision_iter12 (Agent B — direction judge)

Inputs:
- Bottleneck (Agent G iter11): iter11's c-modulated commitment preserved iter8's balance, only +0.0003 gain. Final attempt must double the curvature cycle frequency to get more high-c phases within the 100k step budget.
- Failed-mechanism ledger: iter9 (power-law ε), iter10 (5 iters), iter11 (c-mod commitment). All 类 3.
- iter8 (类 3) is still the best Stage-2 mechanism.

Scoring:

| Candidate | (a) fixes bottleneck | (b) paper | (c) curriculum compat | (d) Stage-1 ceiling risk | (e) Gap-closing | (f) Curvature-relevant | Total |
|---|---|---|---|---|---|---|---|
| P12A (period 50k) | yes — doubles cycle frequency | yes (SGDR 2017) | high | low | yes — "more high-c phases per training" | yes — schedule | 6/6 |
| P12B (period 25k) | yes — but too fast | medium | medium | high | partial | yes | 3/6 |
| P12C (period 75k) | partial — minor | weak | high | low | partial | yes | 3/6 |

Recommendation: **P12A** (period 50k).

Reasoning: iter11's c-modulated commitment was orthogonal to the curvature cycle. Doubling cycle frequency within the 100k step budget gives more high-c phases per item, more c-modulated commitment-pull signal, while keeping iter8's per_layer balance.

Backup: none — final attempt.

Decision: iter12 is `short_period_faster_cycle`, single mechanism P12A (period 50k).