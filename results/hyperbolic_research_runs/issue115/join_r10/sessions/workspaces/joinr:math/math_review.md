# Mathematical review — candidate v7

## Verdict: REVISE

The proposed unsigned Gram-area operator is mathematically meaningful on the stated finite-dimensional tangent spaces, but the degeneracy behavior and the training surrogate as written are not correct. These are repairable with a small formula/wording change; no second mechanism is implicated.

### 1. Scalar and degeneracies

For real tangent vectors, Cauchy–Schwarz gives

`G = ||u||² ||v_k||² - <u,v_k>² >= 0`.

Thus `A_k = 0.5 sqrt(G)` is a well-defined nonnegative scalar, including when `u=0`, `v_k=0`, or the vectors are collinear. The listed clamp also makes the logarithms computationally bounded, though the domain statement should make clear that clamping is an implementation of the logarithm, not a change to the geometric domain.

The stated tie-breaking explanation is wrong. Since `S_k=A_k/(1+d_c(p,z_k))`, whenever `A_k=0`, `S_k=0` regardless of distance. In particular, if `q=p`, every code has score zero and the distance factor cannot break the tie; only the lowest-index rule can. For `z_k=p`, that code has score zero and ties with every collinear code, again independently of distance. Non-collinear codes have positive score and do not tie with them.

### 2. Smoothing and gradients

The proposed `Ã_k = sqrt(G + ε²)` is not a smoothing of the declared `A_k`: it omits the factor `1/2`, so as `ε→0` it converges to `sqrt(G)=2A_k`, not to `A_k`. Even with the factor restored, the difference is not uniformly `O(ε²)` off the degenerate set: for fixed `G>0`,

`0.5 sqrt(G+ε²) - 0.5 sqrt(G) = ε²/(4 sqrt(G)) + O(ε⁴)`.

This is `O(ε²)` for each fixed positive `G`, but the coefficient diverges as `G→0`.

The minimal corrected surrogate is `Ã_k = 0.5 sqrt(G+ε²)` (with ε expressed in the same area units as `sqrt(G)`). It is smooth in `u,v_k` for every ε>0 and converges to `A_k` everywhere, including the degenerate set. The unsmoothed norm-of-wedge area is locally Lipschitz and generally nondifferentiable where the wedge vanishes; it is not true that it must be nondifferentiable at every such input (for example, some special parameterizations can have a zero derivative). The claim should say “may be nondifferentiable” or “is generally nonsmooth” there.

### 3. Flat limit and control

The flat-limit algebra is correct if `×` is defined as the **norm of the exterior/wedge product**, not the ordinary three-dimensional cross product: in 32 dimensions there is no ordinary vector-valued cross product. Explicitly the limit is `0.5 sqrt(||q-p||²||z_k-p||² - ((q-p)·(z_k-p))²)`.

The limit presumes the points and tangent scales have a well-defined Euclidean limit as `c→0`; state how codebook points are represented/scaled in that limit. The proposed control does discriminate operationally only in the narrow registered sense: identical assignments in that control falsify the candidate on this dataset. It does not mathematically prove a general re-parameterization, and differing assignments alone do not prove useful representation quality. Retain it as a falsifier, but call it an empirical flat-control test rather than a proof of identity.

### 4. Capacity and identity

There are no added learned parameters; the scoring rule changes the assignment and hence can change the learned code usage. The score is not determined by `d_c(p,z_k)` alone: it also depends on the target direction through `u` and the angle/Gram term. Therefore it is not merely a swap of the distance metric. The “no parameters” and “not a distance-only function” statements are sound, with the qualification that this is a different assignment objective, not additional parameter capacity.

## Minimal required revision

1. Replace the distance-based tie claims for zero-area cases with the actual rule: zero-area codes all have score zero; ties among them go to the lowest index. In the `q=p` case all codes tie and lowest index wins.
2. Define `G` and use `Ã_k=0.5 sqrt(G+ε²)`. Qualify the `O(ε²)` statement as pointwise for fixed `G>0`, and describe the unsmoothed area as generally nonsmooth at zero wedge.
3. Replace the 32-D `×` notation with the explicit Gram/exterior-product norm, and specify the scaling/representation used to take `c→0`.
4. Describe assignment equality in the flat control as an empirical falsifier on the frozen run, not proof of a general re-parameterization.
