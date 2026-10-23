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

* `q = p` (`u = 0`): `A_k = 0` for every `k`, so all codes tie on the area; `S_k = 0` for EVERY code (the distance factor divides 0), so the tie is broken by an explicit
  lexicographic rule: (A_k, d_c(p,z_k), k) compared in that order, ascending. This is part of the
  operator's definition, not an implementation detail, and the audit records the chosen key.
* `z_k = p` (`v_k = 0`): `A_k = 0` and `S_k = 0`; a code at the host point ties with every collinear code on the area; the lexicographic rule above
  decides, and it is the same rule in every degenerate case.
* `u ∥ v_k`: `A_k = 0` exactly, no NaN and no division.
* Near-boundary points: every `log_p^c` is evaluated with the standard clamp
  `‖·‖ ≤ (1 − 1e-6)/sqrt(c)`; the **clamp rate is a reported number and a kill criterion** (>0.1% of
  evaluations in any stage fails the run as numerically uninterpretable).
* Gradient: `A_k` is Lipschitz but not differentiable at `A_k = 0`. Training uses
  `Ã_k = ½ · sqrt(‖u‖²‖v_k‖² − ⟨u,v_k⟩² + ε²)`   (identical to §4b)
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

## 4b. The surrogate converges

`Ã_k(ε) = ½·sqrt(D_k + ε²)` with `D_k = ‖u‖²‖v_k‖² − ⟨u,v_k⟩² ≥ 0` by Cauchy–Schwarz. For `D_k > 0`,
`|Ã_k − A_k| ≤ ε²/(8·sqrt(D_k)) → 0`; on the degenerate set `D_k = 0`, `Ã_k = ε/2 → 0`. The surrogate is
smooth everywhere, its gradient is finite, and it agrees with the declared area off the degenerate set.

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


## 8. The emitted artifact's schema (so a reviewer can recompute without re-running)

`assignment_audit.emit` writes exactly this, one file per level:

```
{ "schema_version": 1, "kind": "assignment_audit", "stage": "<level index>",
  "candidate_sha256": "<the frozen proposal hash>",
  "examples": { "count": N, "index": [the item indices, ascending, one per row],
                "split": "train" | "dev", "echo": "<the row's SID prefix>" },
  "units": { "area": "tangent metric squared (s = <scale>), c = <curvature>",
             "distance": "geodesic length", "score": "area / (1 + distance)" },
  "per_example": { "winner_full": [..N..], "winner_ablated": [..N..],
                   "margin_full": [..N..], "flipped": [..N bools..],
                   "area": [..N..], "distance": [..N..] },
  "summary": { "flip_rate": .., "tie_share": .., "changed_codes": ..,
               "min_flip_rate": 0.05, "max_tie_share": 0.10, "ok": .. },
  "clamp_count": .., "error_count": .., "thresholds_frozen_before_metrics": true }
```

A reviewer recomputes from this file alone: the flip rate (from `winner_full` and `winner_ablated`),
the changed-code count, the per-example area/distance values that produced each score, and the
clamp rate (from `clamp_count` and `examples.count`). The audit does NOT prove that the winners came
from the declared rule; that is the CGV's job, and the claim above is limited accordingly.
