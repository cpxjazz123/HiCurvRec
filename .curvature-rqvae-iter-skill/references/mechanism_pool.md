# Mechanism Pool (Rotate, Don't Stack)

Each entry: mechanism name, description, paper citation, risk level.

## Curvature Dynamics (1-3)

1. **Cyclic c(t) curriculum (v318 baseline)** — `c(t) = c_min + (c_max - c_min)|sin(πt/T)|`, T=50_000. Paper: SGDR (Loshchilov & Hutter 2017) cyclical LR. Status: already promoted; iterate by varying T ∈ {25k, 100k}.

2. **Layer-wise learned curvature** — Each RQ layer has independent `c_l` learned via sigmoid reparameterization with momentum-restricted drift (`Δc_l = clip(Δc_l, -ε, +ε)` per step). Paper: "Mixed-curvature product manifolds" (Guo et al. 2022).

3. **Lorentz (hyperboloid) RQ-VAE** — Replace Poincaré ball with Lorentz model. Distance: `d(x, y) = arccosh(-⟨x, y⟩_L / c)`. Paper: "Hyperbolic Neural Networks" (Nickel & Kiela 2018) extended by "Lorentzian Distance Learning for Recommender Systems" (Vinh Tran et al. 2020).

## Quantization Mechanics (4-6)

4. **Adaptive margin loss** — `margin = α / c_l`. Sharper separation at high curvature. Paper: "Adaptive Margin for RQ-VAE Tokenizer" (anon 2024).

5. **Sinkhorn c-dependent epsilon** — `ε = ε_0 / c`, tighter assignment at high curvature. Paper: "Optimal Transport for Discrete Representation" (Geneva & Zabaras 2022).

6. **Riemannian Adam** — gradient rescaling by inverse metric `1/c`. Paper: "Riemannian Adam" (Bécigneul & Ganea 2019).

## Codebook Strategy (7-9)

7. **Curriculum on quantization difficulty** — anneal codebook temperature `τ = 1/c`, sharpens assignment schedule. Paper: "Gumbel-Softmax with temperature annealing".

8. **Negative c (partial spherical)** — items in low-density regions get `c < 0` (spherical interpolation). Paper: "Mixed-curvature manifolds" (Guo et al. 2022).

9. **Product manifold M = H^{c1} × H^{c2} × S** — half hyperbolic, half spherical. Paper: "Product Manifold Learning" (Huang et al. 2023).

## Curriculum Variations (10)

10. **Hierarchical curriculum (V-shape)** — `c(t) = c_min + (c_max - c_min) * sin²(πt/T)` (quadratic, no flat plateau). Alternative to v318 linear `|sin|`.

## Already Exhausted (Do Not Retry Without Modification)

- Vanilla RQ-VAE (curvature 1.0 constant) — Gini 0.038, collision 3.9%. **BASELINE**.
- v318 cyclic-c |sin| — Gini 0.038, collision 3.9%, test_R@10 0.1179. **CURRENT BEST**.

## Anti-Patterns (Avoid)

- **Stacking mechanisms**: pick exactly ONE per iteration.
- **Increasing model capacity**: do not change hidden dims [512, 256, 128] or embed=32 or codebook=256 layers=3.
- **Modifying the input pipeline**: do not change `item_emb.parquet` source.
- **Adjusting the downstream trainer**: stage3 is read-only.
