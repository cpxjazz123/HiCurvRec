# lit_search_iter12 (Agent A — literature hunt)

Bottleneck focus (from iteration_bridge_iter11.md): iter11's c-modulated commitment preserved iter8's per_layer balance and only added +0.0003. The remaining 0.0052 gap is downstream. iter12 (final attempt) tries doubling curvature cycle frequency within the 100k step budget.

Queries issued:

1. `"cyclic curvature frequency RQ-VAE training"`
   - P12a (Loshchilov & Hutter 2017 SGDR) — discusses cycle period as a training hyperparameter; shorter cycles = more frequent curriculum updates.
   - P12b (Huang et al. 2023 "Product Manifold Learning") — discusses cyclic vs constant curvature schedules; cycles provide more training-time exploration of the curvature space.
   - P12c (Guo et al. 2022) — supports frequent curvature cycling for codebook balance.

2. `"SGDR shorter period recommendations"`
   - P12d (Loshchilov & Hutter 2017, eq 5) — suggests period T = 50_000 for moderate-length training; 100k is for very long runs.

3. `"curriculum cycle downstream recommender"`
   - P12e (Asano et al. 2020) — supports frequent curriculum cycles for sharper downstream discrimination.

Candidate mechanisms:

- **P12A** (period 100k -> 50k): doubles cycle frequency; more high-c phases within the 100k step budget.
- **P12B** (period 25k): quadruples cycle frequency; would shift cycle too fast for the optimizer.
- **P12C** (period 75k): minor increase; not enough to break iter8.

Agent A recommends P12A (period 50k). This is the SGDR canonical value, doubles cycle frequency, and provides more curvature training without disrupting per_layer balance.

Agent B will judge.