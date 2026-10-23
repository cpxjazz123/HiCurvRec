# Revised hyperbolic SID proposals (Role M)

## TYPE CONVENTIONS

Frozen protocol for both candidates: 32-dimensional latent space; three codebooks of 256 entries; three integer SID tokens; original data, splits, Stage-2/Stage-3 recipes, and strict gates unchanged. Success requires **all** Hard-L1 ER `>16.38953942232631`, Recall@10 `>0.06406796775709883`, and NDCG@10 `>0.03530651701940677`.

Fix one curvature `-c`, `c>0`, and open Poincaré ball `B_c={y∈R^32:c||y||²<1}`. `o=0`. Every `p,x,z_k` below is a **BALL POINT**. Every `Log_a(b)`, `Exp_a(v)` input vector, parallel-transport output, and gyration argument/output explicitly called a vector is a **TANGENT VECTOR at the stated base point**. `g_a=λ_a²I`, `λ_a=2/(1-c||a||²)`. The current residual decoder must map its Euclidean latent to/from `B_c` through the same declared chart for all arms; no extra code capacity or tokens.

Numerical policy: all model points are projected before scoring to radius at most `(1-ε)/√c`, with fixed `ε=10^-6`; report the fraction projected and the pre-projection maximum radius. Non-finite values or points outside the ball after projection fail the run. Score equations refer to the projected points. Exact mathematical domain remains the open ball.

## MECHANISM 1 (revised) — Ordered-simplex signed area assignment

At stage `s`, host `p` and target `x` are BALL POINTS and `q=Exp_p(Log_p(x))=x` (written this way to make the residual basepoint explicit). Each candidate `z_{s,k}` is a BALL POINT. Put `u=Log_p(q)` and `v_k=Log_p(z_{s,k})`, both TANGENT VECTORS AT `p`.

Define `A_c(p,q,z)` as the oriented Riemannian area of the geodesic 2-simplex with ordered vertices `(p,q,z)`. Its orientation convention is intrinsic and computable: when `u,v` are linearly independent, orient their span by the ordered basis `(u,v)` and integrate the positive Riemannian area form in that orientation over the simplex. Thus its signed value is positive for this ordering, and exchanging `q,z` reverses the orientation and sign. No ambient frame or global orientation is used; the ordered edges themselves orient this individual plane. Compute magnitude using the hyperbolic triangle angle defect `(π-θ_p-θ_q-θ_z)/c`, assigning the sign from the ordered simplex. If `u` or `v` is zero, or they are linearly dependent (including repeated/collinear vertices), define `A_c=0`; use zero area gradient on this degenerate set (a stated subgradient convention, not a claim of differentiability there). The score is undefined only for invalid/out-of-domain input before the declared projection or non-finite computation; after projection it is defined. A degenerate triangle has area zero, not an undefined score.

Score and hard choice:
`S_k = -A_c(p,q,z_{s,k}) + β d_c(q,z_{s,k})²`, `k*=argmin_k S_k`, with fixed finite `β>0`. Ties resolve by smallest code index. Use the existing Stage-2 hard assignment and straight-through path; any soft-assignment training relaxation must use this same score and the stated zero subgradient at degeneracy.

The ordered-simplex convention is invariant under orientation-preserving hyperbolic isometries applied to all vertices, and its orientation reverses under swapping the last two vertices. Do not claim invariance under reflections with a fixed vertex ordering: reflections reverse the oriented area and can change the score. Gauss–Bonnet is an implementation check on nondegenerate triangles only.

**Flat control:** hold `β`, points in the fixed bounded chart, and all other parameters fixed as `c→0`. `A_c=O(c)` and the score tends to `β||q-z_k||²`; predict convergence to the Euclidean nearest-code assignment wherever that Euclidean minimum has a positive margin. This control does *not* claim that finite-curvature argmins must change. At the frozen declared `c`, the paired area-zero ablation must establish material decision-level dependence (thresholds below); otherwise withdraw the mechanism.

## MECHANISM 2 (revised) — Origin-chart ordered composition with tangent gyration

Use one chart and one tangent space throughout the score. The host `p`, target `x`, and code `z_{s,k}` are BALL POINTS. Define TANGENT VECTORS AT `o`: `h=Log_o(p)`, `r=Log_o(x)`, and `v_k=Log_o(z_{s,k})`. The chart basepoint `o` is fixed; no vector at `p` is inserted into this formula. Möbius addition `⊕_c` is used only on BALL POINTS. Its formula is the standard Poincaré-ball Möbius addition; the domain is `B_c×B_c→B_c`.

Define a path-specific transport `PT^{rad}_{o→p}` as Levi-Civita parallel transport along the unique radial geodesic from `o` to `p`; it maps TANGENT VECTORS AT `o` to TANGENT VECTORS AT `p` and preserves their Riemannian norms. This transport is reported as a diagnostic only, not silently substituted into the score. Define the gyration point map `Gyr[a,b](y)=-(a⊕b)⊕(a⊕(b⊕y))` for BALL POINTS `a,b,y`; its tangent action at `o` is `D_oGyr[a,b]`, mapping TANGENT VECTORS AT `o` to TANGENT VECTORS AT `o` (gyrations fix `o`).

For each candidate form the BALL POINT `m_k=(p⊕_c z_{s,k})⊕_c x`, and the TANGENT VECTOR AT `o`, `δ_k=D_oGyr[p,z_{s,k}](r)-r`. Score:
`S_k = ||Log_o(m_k)||²_{g_o} + γ ||δ_k||²_{g_o}`, where fixed `γ>0`; choose minimum with lowest-index tie break. Both terms are norms of TANGENT VECTORS AT `o`; no point-coordinate subtraction or mixed-basepoint tangent comparison occurs. The first term measures ordered host-code-target composition; the second isolates the gyration action on the target tangent vector. The tangent derivative is computed from the declared point map, not an assumed point/vector identification.

Domain: projected `p,x,z∈B_c`; exact Möbius addition maps interior points to interior points, so `m_k∈B_c` and `Log_o(m_k)` exists. The gyration derivative is defined for these interior arguments. Use the numerical projection policy in TYPE CONVENTIONS at each point operation. Any projection during forward scoring counts toward clamp rate. A run with clamp rate `>0.1%` in any stage is killed; the permitted rate is therefore at most `0.1%`, not an unreported numerical repair. `PT^{rad}` is a path-defined norm-preserving isometry and may be independently checked, but does not affect `S_k`.

**Flat control:** keep `γ`, chart inputs, and parameters fixed while `c→0`. Möbius addition tends to vector addition and `D_oGyr→I`; the score tends to `||h+v_k+r||²+0`. It must agree with that explicitly computed Euclidean ordered-composition score for every positive-margin example, not necessarily with the original quantizer. At fixed `c`, the gyration ablation must materially affect actual assignments to support the proposed mechanism.

## NON-VACUOUS INVARIANTS

Correctness identities (area degeneracy/Gauss–Bonnet, transport norm preservation, flat limits, ball closure) are implementation checks only and are not evidence of mechanism utility. The following are the non-vacuous, identity/no-op-failing tests. Run paired arms with same initialization, data order, optimizer, recipes and seed; change only the named operator term. Measure per-stage and pooled statistics on validation examples, with bootstrap 95% confidence intervals. For both mechanisms, a no-op operator that leaves assignments unchanged necessarily fails the required changed-code fraction.

* **Candidate 1:** compare trained full score with paired `A=0` score/arm, retaining the same distance term and projection. Require changed-code fraction `≥10%` in each stage and at least `5%` of examples with full-vs-ablation top-1 score margin change `≥10^-4` times the per-example median absolute full-score scale. Also report area values, distance values, score covariance, assignment agreement and partition statistics. Identity/no-op area gives 0% changed codes and fails.
* **Candidate 2:** compare trained full score with paired `γ=0` score/arm, retaining ordered composition and all else. Require changed-code fraction `≥10%` in each stage and at least `5%` of examples with gyration contribution changing the winning-vs-runner-up score margin by `≥10^-4` times the per-example median absolute full-score scale. Also report distribution of `||δ_k||_{g_o}`, component scores, assignment agreement and partition statistics. Identity/no-op gyration gives 0% changed codes and fails.

## PRE-REGISTERED FALSIFIERS

1. **Materiality / top-1 margin (both):** thresholds in NON-VACUOUS INVARIANTS are fixed before training and evaluated per stage; failure in any stage withdraws that candidate. Report changed-code fraction and margin criterion, not merely variance or synthetic witnesses.
2. **Boundary/clamp:** per-stage projection rate `>0.1%`, any non-finite score, or any post-projection out-of-ball point kills the run. Report maximum radius and raw clamp counts.
3. **Frozen task gate (both):** strict conjunction: Hard-L1 ER `>16.38953942232631`; Recall@10 `>0.06406796775709883`; NDCG@10 `>0.03530651701940677`. No threshold, data, split, latent dimension, code count, or recipe changes. Report all metrics even if an earlier gate fails.
4. **Candidate 1:** area-zero paired arm must meet the changed-code and margin thresholds; the `c→0` positive-margin predictions must match the explicit Euclidean control. Failure of either or of the frozen task gate kills it.
5. **Candidate 2:** `γ=0` paired arm must meet the changed-code and margin thresholds; positive-margin flat-control choices must match `argmin ||h+v_k+r||²`. Failure of either, clamp threshold, or frozen task gate kills it.

## WHY NOT HiHPQ

The reviewed HiHPQ evidence is abstract-level overlap in hyperbolic product quantization/codebook attention for image retrieval (DOI `10.1609/aaai.v38i5.28261`; details not verified beyond the abstract). Neither candidate claims the broad idea of hyperbolic quantization as novel. Candidate 1 is specifically an ordered host-target-code geodesic-simplex area term; Candidate 2 is specifically an ordered host-code-target Möbius composition with the differential gyration action on a named tangent vector. The cited abstract does not establish either assignment operator. This is a bounded distinction, not a claim of priority or exhaustive novelty; prior-art status remains ADJACENT pending full-text and broader search.

Neither is a metric swap: candidate 1 adds a signed three-vertex curvature-area term that vanishes in the fixed-parameter flat limit; candidate 2 adds an order-dependent gyration action that vanishes in the flat limit. In each case the paired zero-term ablation is mandatory: if assignments do not meet the pre-registered materiality thresholds, the design is operationally just its control (and is withdrawn), regardless of terminology or metric results.

**Candidate 1 STATUS: READY_FOR_JOIN_R** — operator, orientation convention, degeneracy policy and falsifiers are specified; experimental and novelty claims remain conditional on the registered tests.

**Candidate 2 STATUS: READY_FOR_JOIN_R** — score is type-correct on `T_o`, transport path and gyration derivative are defined, and domain/clamp kill criterion is explicit; it remains conditional on review and registered tests.
