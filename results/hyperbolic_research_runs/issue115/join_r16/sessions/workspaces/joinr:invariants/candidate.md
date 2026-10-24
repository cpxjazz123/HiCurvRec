# CANDIDATE v7 — one mechanism, one page (issue #115, JOIN R round 7)

There is no other candidate. The gyro-composition proposal is withdrawn and does not appear here.

## 1. The operator

**Domain.** Every point named below is a point of the Poincaré ball of curvature `-c`, i.e.
`‖p‖ < 1/sqrt(c)` for all `p`; a code `z_k` that would leave the ball is projected back before
the score is evaluated and the projection is counted in the clamp rate. The frozen encoder
produces a 32-dimensional latent `x`. Level 0 of the RQ chain quantises it with a
codebook of 256 rows `z_k ∈ R^32`. The host geometric space is the Poincaré ball of curvature `-c`
(`c = 1`), and for a batch we map points with the exponential map at the origin,
`p = exp_o^c(s·x)` for the arm's measured tangent scale `s`. The base point of every quantity below is
the **current residual host** `p` (the point being quantised); the candidates are `z_k`.

Let `u = log_p^c(q)` and `v_k = log_p^c(z_k)` be the tangent vectors at `p` pointing to the target `q`
and to code `k`. The mechanism scores code `k` by the **unsigned tangent-space wedge area** of the two tangent vectors `u` and `v_k` (NOT the curved
geodesic-simplex area -- the name matters, and the earlier drafts used the wrong one):

    A_k = ½ · sqrt( Γ ),   Γ = ‖u‖²‖v_k‖² − ⟨u,v_k⟩²  computed as the 2×2 Gram determinant
                          [[‖u‖², ⟨u,v_k⟩], [⟨u,v_k⟩, ‖v_k‖²]] (never as a difference of large
                          products), then Γ is clamped at 0 from below against round-off.

and the assignment score is

    S_k = A_k / (1 + d_c(p, z_k))      (area, tempered by the geodesic distance d_c)

with `argmin_k S_k` choosing the code. `A_k` is a real non-negative number with the units of the
tangent metric squared; it is zero exactly when `u` and `v_k` are collinear and positive otherwise.
There is **no sign, no orientation, no plane, no frame and no reflection claim**: an isometry that
fixes `p` maps `u` and `v_k` by the same orthogonal map, so `A_k` is invariant under it.

## 2. Degeneracies and their smoothing

* `q = p` (`u = 0`): `A_k = 0` for every `k`, so all codes tie on the area; `S_k = 0` for EVERY code (the distance factor divides 0), so the tie is broken by an explicit
  lexicographic rule: (A_k, d_c(p,z_k), k) compared in that order, ascending. This is part of the
  operator's definition, not an implementation detail, and the audit records the chosen key.
* `z_k = p` (`v_k = 0`): `A_k = 0` and `S_k = 0`; a code at the host point ties with every collinear code on the area; the lexicographic rule above
  decides, and it is the same rule in every degenerate case.
* `u ∥ v_k`: `A_k = 0` exactly, no NaN and no division.
* Near-boundary points: every `log_p^c` is evaluated with the standard clamp
  `‖·‖ ≤ (1 − 1e-6)/sqrt(c)`; the **clamp rate is a reported number and a kill criterion** (>0.1% of
  evaluations in any stage fails the run as numerically uninterpretable).
* Gradient: `A_k` is Lipschitz but not differentiable at `A_k = 0`. Training requires `ε > 0` (at `ε = 0` the surrogate does not smooth). `ε` carries the units of a
squared tangent length, because it is added to `Γ`. Training uses
  `Ã_k = ½ · sqrt(‖u‖²‖v_k‖² − ⟨u,v_k⟩² + ε²)`   (identical to §4b)
  `A_k` to `O(ε²)` off the degenerate set. The emitted audit records both `A_k` and `Ã_k`.

## 3. Flat limit

As `c → 0`, `log_p^c → · − p` and `d_c → ‖·‖`, so `A_k → ½‖(q − p) × (z_k − p)‖` and
`S_k → ½‖(q−p)×(z_k−p)‖ / (1 + ‖z_k − p‖)`: a wedge-area penalty. The flat control is a
**pre-registered arm** (`c → 0` with everything else identical). The arm keeps `p`, `q` and the codebook ROWS in their ORIGINAL coordinates and only sets `c → 0`
(i.e. the same tangent vectors, the same scale `s`, the flat metric); no re-embedding is performed.
Its evidentiary force is limited and stated as such: agreement between the flat arm and the curved arm shows that the curvature is not what
moves the assignment, which is a NECESSARY condition for the mechanism to be non-trivial; it does not
show that the curved arm's effect is useful. The decisive test is the joint outcome of that control and
the area-zero ablation: the mechanism counts only if the area term changes decisions (flip rate ≥ 5%)
AND the flat arm does not reproduce those same decisions.

## 4. Capacity and identity

The mechanism adds **no parameters**: it changes only how `argmin` scores a code. It is not a token
layout (the three tokens stay the host's), not a transported frame (nothing is transported), and not a
metric swap in the narrow, defensible sense: the score is not a distance-only replacement, because it
depends JOINTLY on the target direction and the candidate direction. No claim is made that it is
distinguishable from every conceivable metric-based reformulation -- no experiment could establish that.

## 4b. The surrogate converges

`Ã_k(ε) = ½·sqrt(D_k + ε²)` with `D_k = ‖u‖²‖v_k‖² − ⟨u,v_k⟩² ≥ 0` by Cauchy–Schwarz.

The error is **not** claimed to be uniformly small. What is claimed, and checked numerically rather
than asserted: `Ã_k − A_k = ε² / (4·(Ã_k + A_k))`, which is bounded by `ε/2` everywhere, tends to zero
pointwise as `ε → 0`, and is at most `ε²/(4·√D_k)` for `D_k > 0`. Near the degenerate set the
absolute error therefore vanishes but its RELATIVE size can be large, so no uniform relative bound is
claimed and the audit reports `D_k`, `A_k` and `Ã_k` side by side for every example.

## 5. The evidence the review is paired with (mandatory)

`assignment_audit.emit(path, full, ablated, stage=…, components={"area":…, "distance":…},
clamp_count=…, error_count=…)` is called **per level, inside the evaluation loop, before any metric is
read**, and writes one JSON per stage with: the winning index under the full and the area-zero score,
the winner/runner-up margins, the per-example components, the flip rate, the tie share and the
clamp/error counts.

Pre-registered thresholds, frozen before any run:
* **flip rate ≥ 5%** per level between the full score and the area-zero score;
* **tie share ≤ 10%** per level;
* **clamp rate > 0.1%**, any non-finite score, or any out-of-ball point ⇒ the run fails.

No review of this candidate counts without that emitted artifact.

## 6. Falsifiers, in order

1. The area-zero ablation leaves the assignments unchanged (flip rate below 5%) ⇒ the mechanism is a
   no-op ⇒ **rejected**.
2. The `c → 0` control reproduces the assignments ⇒ **rejected** as a re-parameterisation.
3. The partition degenerates (a level with ≤16 distinct codes, or a cell >25% of the catalogue) ⇒
   **rejected** by the partition gate.
4. The frozen ER does not strictly exceed 16.38953942232631 ⇒ **rejected**; no Stage3.
5. ER passes but Stage3 Recall@10 ≤ 0.06406796775709883 or NDCG@10 ≤ 0.03530651701940677 ⇒
   **rejected**; the triple is conjunctive.

## 7. Status

`READY_FOR_JOIN_R` (round 7). Nothing here is experimental approval; the frozen protocol
(32-D latent, three 256-code codebooks, three SID tokens, the original data, splits and recipes) is
untouched.


## 8. The emitted artifact (schema 3 IS the binding contract; schemas 1 and 2 are withdrawn): it carries the operands and REFUSES a score it cannot recompute

`assignment_audit.emit_v3(path, full, ablated, gram=…, geodesic=…, …)` — `gram` is `(n, codes, 3)`
holding `‖u‖²`, `‖v_k‖²`, `⟨u,v_k⟩`; `geodesic` is `d_c(p,z_k)`.

The function **recomputes** `A_k = ½·sqrt(max(Γ,0))` and `S_k = A_k/(1+d_c)` from those operands and
**raises** when the stored scores disagree by more than `1e-9`. A file that cannot be recomputed is not
written. It also emits: the counts by operation type with the exact rate formula (and the STRICT
`> 0.1%` clamp boundary), the one tie rule in its binding block, the units of `ε` (a squared tangent
length), `area` and `distance`, and an explicit list of what this file does NOT decide — **the flat
`c→0` control comparison and the partition gate are separate artifacts**.


## 9. An ACTUAL emitted example (not a prose schema)

Produced by `assignment_audit.emit_v3` on a seeded toy batch and committed at
`results/hyperbolic_research_runs/issue115/audit_example_schema3.json`; it carries the operands,
the recomputation errors, the tie rule, the per-operation counts and the two separate-artifact
labels, exactly as a run's file will.
