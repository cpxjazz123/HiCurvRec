# Revision 5 — Role M final

## Frozen protocol and disposition

The only surviving candidate is **Mechanism 1: unsigned geodesic 2-simplex area in the 32-D ambient Poincaré ball**. Mechanism 2 is **WITHDRAWN** and is not a candidate for implementation, evaluation, or review in this revision. The latent dimension (32), three 256-entry codebooks, three integer SID tokens, data/splits, Stage-2/Stage-3 recipes, and strict task gates remain frozen. The task gates are Hard-L1 ER `>16.38953942232631`, Recall@10 `>0.06406796775709883`, and NDCG@10 `>0.03530651701940677`, all required.

## Mechanism 1 — unsigned 32-D geodesic simplex area assignment

Fix curvature `-c`, `c>0`, and the open Poincaré ball `B_c ⊂ R^32`; `p` is the host, `q=x` the target, and `z_k` a code point. All are projected to the existing declared strict-interior numerical domain before scoring. Define the geodesic 2-simplex `Σ(p,q,z)` as the unique totally geodesic triangular surface with those vertices (including its interior in the unique totally geodesic 2-plane they span). Its area is the **unsigned Riemannian 2-volume** induced from the ambient hyperbolic metric. It has no orientation datum, sign, or reflection-sensitive component.

Exact area formula: choose any smooth regular parametrization `F: Δ→Σ`, where `Δ={(s,t):s≥0,t≥0,s+t≤1}`. At each `(s,t)`, let `E_1=dF(∂_s)` and `E_2=dF(∂_t)`, the two geodesic-simplex edge/tangent vectors in the host tangent geometry, with metric `g_F=λ_F²I`, `λ_y=2/(1-c||y||²)`. Then

`A_c(p,q,z) = ∫_Δ sqrt(det([[g_F(E_1,E_1), g_F(E_1,E_2)], [g_F(E_2,E_1), g_F(E_2,E_2)]])) ds dt`.

Equivalently, the integrand is the square root of the Gram determinant of the two simplex tangent vectors in the ambient metric. The vectors can be evaluated in host tangent coordinates by expressing `F` in the normal chart at `p` (`Log_p(F(s,t))`) and applying the differential of `Exp_p`; the integration domain is bounded by the three geodesic edges, not assumed to be an affine triangle in that chart. This definition is independent of parametrization and uses no learned or fixed frame. In the flat limit, for fixed bounded configurations, the simplex tends to the affine triangle spanned by `u=Log_p(q)` and `v=Log_p(z)`, and

`lim_{c→0} A_c(p,q,z) = 1/2 sqrt(det([[u·u,u·v],[v·u,v·v]])) = 1/2 ||u∧v||`.

In particular the area is nonnegative, invariant under every common hyperbolic isometry including reflections, and tends to the ordinary unsigned Euclidean triangle area (not zero). No signed-area, orientation, orientation-preserving-isometry-only, or sign-reversal claims apply.

Use score `S_k = -α A_c(p,q,z_k) + β d_c(q,z_k)^2`, with fixed preregistered finite `α>0, β>0`; choose the lowest score, resolving exact ties by smallest code index. The area is zero when vertices coincide or are collinear (including a zero geodesic edge); the area score remains defined. The simplex volume is continuous but need not have a smooth gradient at rank-deficient/degenerate triangles. For gradient-based training, use the explicit smooth surrogate `A_{c,ε}=sqrt(A_c²+ε_A²)-ε_A` with fixed preregistered `ε_A>0` in score and ablation alike; it is exactly zero at zero area, has finite derivative there, and converges to `A_c` as `ε_A→0`. Report `ε_A`, and do not describe the unsmoothed rank-deficient locus as differentiable. Exact boundary points are excluded; apply the frozen projection/clamp policy before evaluating the operator and report clamp frequency. Nonfinite or post-projection out-of-domain values fail the run.

The `c→0` control uses the same bounded chart, parameters and tie rule; its area term is the unsigned Euclidean Gram area above, not zero. Therefore the relevant Euclidean control is the explicitly computed score `-α(1/2||u∧v_k||)+β||q-z_k||²`, with assignments required to agree wherever its minimum has a positive margin. Also report the distance-only control, but do not conflate it with the flat-limit control.

### Capacity and distinction

The area is a deterministic scalar function of the existing host, target and code points. It introduces no frame, subspace parameter, projection, learned parameter, code, token, or additional trainable capacity. The geodesic 2-plane is the unique geometric span of the three points, not a globally selected role subspace or role-layout encoding; all input points remain in the original 32-D ball and no point is mapped into a learned/fixed 2-D subspace. The operation is a symmetric unsigned simplex-volume contribution, not a signed-direction mechanism.

## Evaluation-path audit and pairing rule

At every evaluation pass, in the **stage loop `for stage in stages`**, construct score matrices on the identical held-out examples and same frozen hosts/codebooks: `full` uses `-α A_{c,ε}+βd²`; `ablated` sets only the area contribution to zero and retains the identical distance term, points, projection, and tie rule. Call `assignment_audit.emit(path, full, ablated, stage=stage, components=...)` once per stage. The audit implementation is `assignment_audit.audit(full, ablated)`, invoked by `emit`.

Write one artifact per stage at `evaluation/assignment_audit/stage_{stage}.json` (or the run-scoped equivalent `runs/{run_id}/evaluation/assignment_audit/stage_{stage}.json`); preserve the emitted JSON with the evaluation results. Include the full area and distance score components and clamp/error counts. The frozen audit gates from `assignment_audit.GATE` are **flip rate ≥ 5% in every stage** and **tie share ≤ 10% in every stage**. Tie share is the audit's full-score winner/runner-up gap test (`≤1e-12`). A failed stage withdraws the candidate; do not substitute pooled success for a per-stage failure. Report the audit contribution statistic and per-example winner/margin trace emitted by the module.

**Pairing rule:** Mechanism 1 may not be reviewed again, advanced, or represented as decision-effective without the emitted per-stage assignment-audit artifact. A function's presence or a prose claim is not evidence that the audit ran. Missing artifact, failed threshold, or mismatched full/ablated examples invalidates the evaluation. Passing this audit establishes decision-level contribution only; it does not establish recommendation utility or novelty. The frozen task gates and implementation/boundary checks remain separately mandatory.

## Staged review disposition

The staged v4 mathematics review, invariants review, prior-art review, and first-party `assignment_audit.py` are considered together. The signed formulation and all sign/orientation/reflection claims are superseded and deleted. The old requirement to use an explicitly oriented 2-D subspace is superseded by the precisely defined 32-D unsigned simplex volume above. No mechanism-2 formula or rationale is carried forward.

**Mechanism 1 STATUS: READY_FOR_JOIN_R** — unsigned 32-D operator, flat limit, degeneracy/smoothing, boundary/tie policy, capacity statement, and mandatory emitted audit are specified. This status permits review/testing only; it is not experimental approval. Review pairing requires the emitted per-stage audit artifact.

**Mechanism 2 STATUS: WITHDRAWN** — withdrawn by decision; the type-inconsistent gyro-composition candidate is not revised or eligible for review. It must not reappear as a surviving or alternate mechanism in this revision.
