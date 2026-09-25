# iteration_bridge_post_iter15 (Agent G — bottleneck for next iter)

## Dominant bottleneck

- **Stage-1 ceiling** (~0.0595–0.060) persists; iter15 frozen residual `u_l` **regressed** to test_R@10=**0.0582**.
- Hypothesis tested: learnable `c_layer_scale` drift confounds layer-wise curvature story → **rejected** at downstream level (iter11 with learnable `u_l` still beats iter15).
- **iter11** remains best Stage-2 point on test_R@10 (**0.0598**); **iter8** still strong reference (**0.0595**).

## Forbidden next steps

- Re-run iter15 or minor tweaks to freeze-only `u_l` without a new orthogonal mechanism.
- Re-introduce iter14-style behavior `b_l` without a new hypothesis (already flat vs iter11).

## Suggested search space (not decided here)

- Orthogonal to layer-scale freeze: e.g. Riemannian optimizer (P15B backup), product-manifold codebook (P15C), or a mechanism that changes **quantization dynamics** without removing learnable layer scale entirely.
- Any next iter must record Agent D snapshots + Stage3 gate the same way.
