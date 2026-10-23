# CANDIDATE v7 — one mechanism, one page (issue #115, JOIN R round 7)

There is no other candidate. The gyro-composition proposal is withdrawn and does not appear here.

## 1. The operator

The frozen encoder produces a 32-dimensional latent `x`. Level 0 of the RQ chain quantises it with a
codebook of 256 rows `z_k ∈ R^32`. The host geometric space is the Poincaré ball of curvature `-c`
(`c = 1`), and for a batch we map points with the exponential map at the origin,
`p = exp_o^c(s·x)` for the arm's measured tangent scale `s`. The base point of every quantity below is
the **current residual host** `p` (the point being quantised); the candidates are `z_k`.

Let `u = log_p^c(q)` and `v_k = log_p^c(z_k)` be the tangent vectors at `p` pointing to the target `q`
and to code `k`. The mechanism scores code `k` by the **unsigned 2-simplex area** the two geodesics
`p→q` and `p→z_k` span:

    A_k = ½ · sqrt( ‖u‖²‖v_k‖² − ⟨u, v_k⟩² )        (the Gram determinant, no orientation)

and the assignment score is

    S_k = A_k / (1 + d_c(p, z_k))      (area, tempered by the geodesic distance d_c)

with `argmin_k S_k` choosing the code. `A_k` is a real non-negative number with the units of the
tangent metric squared; it is zero exactly when `u` and `v_k` are collinear and positive otherwise.
There is **no sign, no orientation, no plane, no frame and no reflection claim**: an isometry that
fixes `p` maps `u` and `v_k` by the same orthogonal map, so `A_k` is invariant under it.

## 2. Degeneracies and their smoothing

* `q = p` (`u = 0`): `A_k = 0` for every `k`; the tie is broken by `d_c(p, z_k)` alone.
* `z_k = p` (`v_k = 0`): `A_k = 0`, and `S_k ≤ 0` after tempering, so that code is never chosen by the
  area term; the distance term decides.
* `u ∥ v_k`: `A_k = 0` exactly, no NaN and no division.
* Near-boundary points: every `log_p^c` is evaluated with the standard clamp
  `‖·‖ ≤ (1 − 1e-6)/sqrt(c)`; the **clamp rate is a reported number and a kill criterion** (>0.1% of
  evaluations in any stage fails the run as numerically uninterpretable).
* Gradient: `A_k` is Lipschitz but not differentiable at `A_k = 0`. Training uses
  `Ã_k = sqrt(‖u‖²‖v_k‖² − ⟨u,v_k⟩² + ε²)` with `ε = 1e-6`, which is smooth everywhere and agrees with
  `A_k` to `O(ε²)` off the degenerate set. The emitted audit records both `A_k` and `Ã_k`.

## 3. Flat limit

As `c → 0`, `log_p^c → · − p` and `d_c → ‖·‖`, so `A_k → ½‖(q − p) × (z_k − p)‖` and
`S_k → ½‖(q−p)×(z_k−p)‖ / (1 + ‖z_k − p‖)`: a wedge-area penalty. The flat control is therefore a
**pre-registered arm** (`c → 0` with everything else identical); if its assignments match the curved
arm's, the mechanism is a re-parameterisation and the candidate dies.

## 4. Capacity and identity

The mechanism adds **no parameters**: it changes only how `argmin` scores a code. It is not a token
layout (the three tokens stay the host's), not a transported frame (nothing is transported), and not a
metric swap (the distance is a tempering factor, and the area is not a function of `d_c` alone).

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
