# Mathematical review — candidate v7

## Verdict: REVISE

The basic unsigned Gram-area is mathematically sound in exact arithmetic, but the candidate does not yet define a numerically well-defined operator over its stated domain, and the stated flat limit/control has a material inconsistency. Minimal corrections are below. This is not a review of any signed area or second score term; neither appears in `candidate.md`.

## 1. Definition and degeneracies

For finite tangent vectors `u,v_k` in a positive-definite tangent metric, the Gram matrix is positive semidefinite and
`D = ||u||²||v_k||² - <u,v_k>² >= 0`. Thus `A_k = ½ sqrt(D)` is a well-defined nonnegative scalar; it is zero exactly for linearly dependent vectors (including either zero vector), and positive otherwise. The four listed cases are covered algebraically: `u=0`, `v=0`, collinearity, and finite vectors near the boundary. The stated tie ordering resolves zero-score ties, assuming finite distances and a deterministic code index.

However, “computed as the 2×2 Gram determinant (never as a difference of large products)” does not specify a numerically stable determinant algorithm. A Gram determinant still equals the subtraction of two nearly equal products if evaluated conventionally; clamping a negative result to zero avoids NaNs but can silently turn a positive, nearly-collinear area into zero. Further, the domain restriction on points does not bound `log_p(z)` as `p` approaches the boundary, and the text's clamp is ambiguous: it says `||·||` for a logarithm without defining whether the clipped quantity is a point, tangent vector, or argument to an inverse hyperbolic function. The reported clamp rate does not itself define the resulting score.

**Minimal fix:** specify stable evaluation (e.g. compute in an orthonormal tangent basis using a numerically stable 2D wedge/QR or SVD, with precision and tolerance), and precisely define each Poincaré log-map clamp. State whether clamping changes the operands used in the score. The tie rule is sufficient for the mathematical zero cases, but implementation must use it consistently.

## 2. Epsilon surrogate and gradients

The surrogate `Ã=½ sqrt(D+ε²)` is correct for `ε>0`, with `ε` having units of squared tangent length because `D` has units of tangent length to the fourth power. It converges pointwise to `A` as `ε→0`, including at `D=0`; off the degenerate set its error is
`Ã-A = ε²/[4(Ã+A)]`, hence is `O(ε²)` for fixed `D>0`. Also `0 <= Ã-A <= ε/2` everywhere.

The gradient statement is incomplete/misleading. `A` is nondifferentiable on the dependent-vector set, not merely “at `A=0`” as a scalar shorthand; off that set it is differentiable. For `ε>0`, the surrogate is smooth as a function of the Gram entries (and smooth wherever the underlying log maps are smooth), but that does not guarantee smoothness through any clipping/projection boundary. At `u=0`, its derivative with respect to `u` is zero, while at `ε=0` the claimed “smoothing” indeed disappears. The text says the surrogate is used for training but defines assignment through `A`; it should explicitly say whether gradients optimize `Ã` while discrete assignment/audit use `A`.

**Minimal fix:** state the gradient variables and the above smoothness qualification; define training-vs-inference use explicitly.

## 3. Flat limit and control

The algebraic limit in §3 is correct if `p,q,z_k` are held fixed in common Euclidean coordinates and `u→q-p`, `v→z_k-p`; the Euclidean 2-vector wedge norm in the formula is the Gram area. The denominator tends to `1+||z_k-p||`.

But the control description says it retains the original `p,q` and codebook rows while setting `c→0`, “no re-embedding.” Those points were produced/declared in the curvature-1 Poincaré ball; with its stated domain, they satisfy `||p||,||z||<1`, while the flat-limit mapping and tangent vectors require a specific identification of those coordinates across curvature. In particular, `p=exp_o^c(sx)` depends on `c`; holding the curvature-1 point fixed is not the same intervention as recomputing `exp_o^{c→0}(sx)`. The text does not say which coordinates are the frozen ones, nor how the codebook rows—also subject to curvature-dependent projection—are mapped. Therefore the displayed limit is a valid formal fixed-coordinate limit, but it is not established as the limit of the proposed arm as operationally specified.

The control can discriminate only if it compares assignments under a fully specified matched-coordinate intervention. As written, “flat arm does not reproduce those same decisions” is a proposed criterion, not a mathematical guarantee. Agreement does not show curvature is necessary; indeed the text's own criterion requires non-reproduction, which is the relevant discriminator. A useful control should state the exact conversion from frozen latent/code coordinates to each curvature and avoid changing scale or projection policy at the same time.

**Minimal fix:** define whether the flat arm freezes tangent coordinates (`sx` and corresponding codebook tangent coordinates) and recomputes points, or freezes host coordinates and interprets them in Euclidean space. Reconcile that choice with the displayed limit and projection rule. Predefine the decision-comparison statistic/tolerance.

## 4. Capacity and metric identity

No learned parameters are added. The score is not reducible to a candidate-only distance `d(p,z_k)`: it depends on `q` through `u`, and thus can alter rankings in a target-dependent way. So “not a distance-only replacement” is justified under the narrow definition stated.

It is nevertheless a metric/geometry-derived scoring rule and could be described as a target-conditioned metric-like dissimilarity; no intrinsic learned capacity is added. The score's area factor is scale-dependent and the denominator does not generally make it a metric (there is no asserted symmetry or triangle inequality). The candidate appropriately avoids claiming non-equivalence to every metric reformulation. The area-zero ablation is a meaningful test of whether this factor changes assignments, but a ≥5% flip rate alone does not establish useful changes; the frozen outcome thresholds and controls must do that work.

## Required minimal revisions before approval

1. Specify stable Gram-area computation and an unambiguous log-map/clamp definition.
2. Clarify that `Ã` is the training objective and say which quantities receive gradients; qualify smoothness at clipping/projection boundaries.
3. Define the flat-arm coordinate matching and projection/scale policy consistently with the formal `c→0` limit.

Until these are fixed, the mechanism is plausible but its numerical operator and principal control are not sufficiently pinned down for approval.
