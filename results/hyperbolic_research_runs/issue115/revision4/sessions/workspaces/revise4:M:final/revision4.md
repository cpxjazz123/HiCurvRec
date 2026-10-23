# Revision 4 — Role M response to the third JOIN R

This revision fixes the operators and their measurable tests, rather than treating the staged
proposal's former READY status as approval. The latent remains 32-D, with three 256-entry
codebooks and three integer SID tokens; data, splits, recipes, and the strict task gates remain
frozen. The first-party `assignment_audit.py` is the authoritative decision-level audit.

## Mechanism 1 — signed area in a fixed, oriented, totally geodesic plane

Fix once, before training, a two-dimensional linear subspace `V ⊂ T_o B_c = R^32`, together with
an ordered orthonormal basis `(e1,e2)` that gives `V` its orientation. `V` determines the complete
totally geodesic Poincaré disk `D = Exp_o(V)`. The basis and subspace are constants shared across
all stages, examples, hosts, targets, and codewords; they are not optimized, selected from data,
or stored per role.

Every ball point `y` is mapped explicitly by
`π_V(y) = Exp_o(P_V Log_o(y)) ∈ D`, where `P_V` is Euclidean orthogonal projection in `T_o`.
This is the map used for the area term only. For host `p`, target `q=x`, and candidate `z_k`,
compute the signed Riemannian area `A_V(p,q,z_k)` of the geodesic triangle with ordered vertices
`(π_V(p), π_V(q), π_V(z_k))` inside `D`; its sign is the orientation induced by `(e1,e2)` and
that vertex order. Thus the sign is defined even when the three projected points are collinear
(where area is zero), and swapping `q,z_k` reverses the sign. The area magnitude can be computed
by the hyperbolic angle defect divided by `c`, with sign determined by the oriented disk. The
score remains `S_k = -A_V(p,q,z_k) + β d_c(q,z_k)^2`, minimized with smallest-code-index tie-break.

If any mapped vertices coincide or are collinear, set area to exactly zero. The signed area is
not differentiable across the collinear set; for gradient-based training use the declared zero
subgradient there (no claim of a smooth gradient). Strictly interior points are the mathematical
domain. Apply the frozen radial projection policy before mapping/scoring; report projection
counts and fail on non-finite values or points still outside the ball. Exact boundary points are
invalid. Ties are resolved by smallest code index in both full and counterfactual assignment.

The signed area is invariant only under orientation-preserving hyperbolic isometries that preserve
the fixed datum `(V,(e1,e2))`; do not claim invariance under arbitrary isometries or reflections.
The flat-limit and Gauss–Bonnet claims apply to this projected 2-D triangle, not to the original
32-D simplex. The fixed projection adds zero learned parameters and no per-role or per-code
degrees of freedom. It is not a role-layout equivalent: the same data-independent projection is
applied to every point, and neither roles nor code positions determine the plane or its orientation.

**Counterfactual.** On exactly the same frozen checkpoints, examples, hosts, targets, and code
vectors, form the full score matrix and an ablated matrix with only `-A_V` set to zero; do not
retrain for this primary audit. This isolates whether area itself changes the argmin. A separately
trained area-zero arm may be reported as an additional optimization control, not as a substitute.

## Mechanism 2 — origin-tangent ordered composition and gyration penalty

Use the Poincaré ball `B_c={y∈R^32:c||y||²<1}` with curvature `-c`, origin `o=0`, standard
Möbius addition
`a⊕_c b = [ (1+2c<a,b>+c||b||²)a + (1-c||a||²)b ] /
             [1+2c<a,b>+c²||a||²||b||² ]`,
and origin metric `g_o=4I` (since `λ_y=2/(1-c||y||²)`). All `p,x,z_k` and all operands of `⊕`
are ball points. `Log_o` outputs vectors in the single tangent space `T_o`; all norms below use
`g_o`. Define the gyro-action point map
`Gyr[a,b](y)=-(a⊕b)⊕(a⊕(b⊕y))`; it fixes `o`. Its differential `D_oGyr[a,b]` is a linear map
`T_o→T_o`, so `δ_k = D_oGyr[p,z_k](Log_o(x)) - Log_o(x)` is a well-typed tangent vector.
There is no point subtraction, vector-at-`p` comparison, or use of parallel transport in the score.

For capacity control set `T_s := I_32` for every stage, fixed and non-learned: zero trainable
parameters, operator norm 1, inverse norm 1, and condition number 1. (Thus the code tangent is
`v_k=Log_o(z_k)`; no `T_s`-transformed code or unused transport symbols occur.) Score
`S_k = ||Log_o((p⊕_c z_k)⊕_c x)||²_{g_o} + γ||δ_k||²_{g_o}`, with fixed `γ>0`; choose the
minimum and break ties by smallest code index. Exact Möbius operations keep interior points in
the open ball. Use the stated projection policy at input points; any additional projection during
scoring is counted, and a stage projection rate above 0.1%, any non-finite score, or any remaining
out-of-ball point kills the run.

This construction has no learned transform capacity and is not the refuted transported-frame
form: code vectors remain origin-chart points in the same fixed `T_o`, no code frame is transported
to the host, and the penalty is the differential gyration action on the target's origin tangent
vector. The radial transport, if logged for diagnostics, is not part of the operator.

**Same-parameter, same-point ablation.** For each identical score matrix, checkpoint, and example,
set only `γ||δ_k||²` to zero. Do not alter `T_s`, composition, parameters, codebooks, optimization,
or inputs. A separately retrained `γ=0` arm is an additional control only, never a replacement
for this counterfactual. Also run the matched `c=0` control with the same `T_s=I`, same `γ`, same
score implementation and optimization recipe, and the same inputs/initialization; use Euclidean
addition and `D_oGyr=I` there. This checks the flat ordered-composition score and is distinct from
the same-point term ablation.

## Audit placement and pre-registered decision rules (both mechanisms)

In evaluation, immediately after the model emits each stage's full and ablated `(n_examples,256)`
score matrices on the identical validation examples and before calculating partition or task
metrics, call `assignment_audit.audit(full_scores, ablated_scores)` once per stage. Save its
complete return object in the run artifacts, with stage identifier, checkpoint, seed, and example
count. Do not substitute pooled-only reporting or a retrained-arm comparison. For mechanism 1 the
ablated matrix has area removed; for mechanism 2 it has only the gamma penalty removed.

The staged audit's fixed `GATE` is binding: **flip rate ≥ 0.05 in every stage** and **tie share ≤
0.10 in every stage**. Here tie share is the audit's share of full-score winner/runner-up gaps
`≤1e-12`. Report per-stage flip rate, tie share, scaled contribution, and audit verdict; a failed
stage withdraws that candidate. Keep the audit's deterministic lowest-index `argmin` tie handling.
These audit gates supersede the earlier inconsistent 10%/margin wording in `revised_proposals.md`.
The frozen task conjunction also remains mandatory for a performance claim: Hard-L1 ER
`>16.38953942232631`, Recall@10 `>0.06406796775709883`, and NDCG@10 `>0.03530651701940677`.
Audit success establishes decision participation only, not task utility or novelty.

## Status

**Mechanism 1 STATUS: READY_FOR_JOIN_R** — the fixed oriented 2-D totally geodesic domain,
point mapping, degeneracy/subgradient, boundary and tie rules, capacity argument, matched
counterfactual, audit call, and binding thresholds are specified. This is readiness for the next
review, not empirical approval; failure of any stage audit or frozen task gate withdraws it.

**Mechanism 2 STATUS: READY_FOR_JOIN_R** — both terms are typed in `T_o` under the explicit
Poincaré convention, unused symbols are removed, `T_s` is fixed identity with zero learned
capacity and condition number 1, and the same-point ablation and matched flat control are
specified. This is readiness for the next review, not empirical approval; failure of any stage
audit, clamp rule, or frozen task gate withdraws it.
