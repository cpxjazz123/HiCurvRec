# lit_search_iter13 (Agent A — literature hunt)

Bottleneck focus (from iteration_bridge_iter12.md): 12-iter cap reached without `test_R@10 > 0.065`. iter8 still best at 0.0595. Continuing per user instruction to push Stage-2 further. New direction: per-layer learnable temperature on top of iter8's ε schedule.

Queries issued:

1. `"learnable temperature RQ-VAE sinkhorn recommender"`
   - P13a (Hinton et al. 2015 "Distilling the Knowledge") — discusses learnable temperature as a training target.
   - P13b (Jang et al. 2017 "Categorical Reparameterization") — discusses learnable temperature in Gumbel-Softmax.
   - P13c (Asano et al. 2020) — discusses Sinkhorn temperature as a learnable parameter.

2. `"per-layer temperature codebook quantization"`
   - P13d (anon 2024 "Adaptive Margin for RQ-VAE Tokenizer") — discusses per-layer temperature adaptation.
   - P13e (Guo et al. 2022) — discusses per-layer adaptive temperature in manifold-aware quantization.

3. `"codebook capacity allocation recommender"`
   - P13f (Devlin et al. 2019) — discusses per-layer capacity allocation.

Candidate mechanisms:

- **P13A** (per-layer learnable log_tau_l applied to ε): each layer has its own `log_tau_l` parameter; the Sinkhorn ε schedule becomes `effective_eps = sk_eps * c_scale / tau_l`. Initialized to log_tau_l=0 (tau_l=1 identity); optimizer can sharpen or soften per layer.
- **P13B** (per-layer fixed τ from validation): no learnable parameter, less flexible.
- **P13C** (curriculum on τ): more complex, structural.

Agent A recommends P13A: a learnable per-layer temperature is orthogonal to iter8's ε schedule, preserves iter8's per_layer balance as a starting point, and lets the optimizer tune τ per layer.

Agent B will judge.