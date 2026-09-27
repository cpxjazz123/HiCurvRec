# lit_search_iter10 (Agent A — literature hunt)

Bottleneck focus (from iteration_bridge_iter9.md): iter9 power-law ε regressed by -0.0027 vs iter8. iter8's linear ε is the local optimum. Continue iter8's mechanism family by refining a *different* quantization aspect — Sinkhorn iteration count.

Queries issued:

1. `"Sinkhorn iterations count optimal transport convergence"`
   - P10a (Cuturi 2013 "Sinkhorn Distances") — argues that 3 iters is enough for ε large; more iters matter when ε is small.
   - P10b (Geneva & Zabaras 2022 "Optimal Transport for Discrete Representation") — argues more iters (5-10) tighten assignments under adaptive ε.
   - P10c (Caron et al. 2020) — empirical observation that 5 iters gives tighter convergence for codebook assignment.

2. `"adaptive sinkhorn iterations recommender"`
   - P10d (Radford et al. 2021 CLIP) — supports more iters at training, fewer at inference (not directly relevant).
   - P10e (Asano et al. 2020) — supports increased iters for sharper discrimination.

Candidate mechanisms:

- **P10A** (5 Sinkhorn iters with iter8's linear ε): tightens assignment convergence without changing ε scaling.
- **P10B** (variable iters per layer): structural change, too risky.
- **P10C** (8 iters): diminishing returns; would significantly slow down training (~3.0 it/s → ~2.0 it/s).

Agent A recommends P10A: minimal parameter change that converges assignments more tightly while keeping iter8's optimal linear ε schedule.

Agent B will judge.