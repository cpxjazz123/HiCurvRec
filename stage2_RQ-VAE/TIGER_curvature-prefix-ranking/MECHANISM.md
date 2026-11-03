# curvature_prefix_ranking

AF3-cohesion-guided hierarchical prefix ranking for the hyperbolic residual
quantizer. Candidate tree copied from `TIGER_poincare-frechet-successor-codebook`
(the last complete mechanism of the #121 baseline chain); only the arm identity,
the new mechanism and its tests differ from that baseline.

## The coupled pathway

```
TRAIN behaviour edges                       residual quantizer
        |                                          |
  AF3(e)=4-deg(u)-deg(v)+3|N(u)&N(v)|      z_l = residual entering level l
        |                                          |
  s_e = sigmoid((AF3-median)/(IQR/1.349))    p_l(k|u) = softmax(-d_H(exp_0(z_l^u), c_lk)/T)
        |                                          |
  triple (anchor, high, bridge)  ---------->  a_l(u,v) = sum_k p_l(k|u) p_l(k|v)
                          L = mean_l softplus(-(a_l(anchor,high)-a_l(anchor,bridge)) * w)
                                     |
                              gradient -> encoder -> SID
```

* **Behaviour side (train-only).** Consecutive `history+target` pairs of the
  TRAIN split build one undirected item graph; `AF3` and the median/IQR split are
  computed once (315,391 edges, median -128.0, scale 177.17 on this split). No
  validation or test interaction is read.
* **Curve side.** Level residuals are mapped with the framework's canonical
  `poincare_contract.expmap0`, and the assignment cost is
  `poincare_contract.distance` to that level's codebook (which the baseline keeps
  in the ball). The curve-off switch uses squared Euclidean costs instead.
* **Coupling.** The loss is a *relative* preference on the soft prefix overlap:
  it never forces two items to share or not share a prefix, it only asks the
  cohesive neighbour's overlap to exceed the bridge neighbour's. Gradients flow
  through the hyperbolic distance into the encoder, so the curve is inside the
  trained pathway, not decoration.

## Constants and how they are derived (no tuning)

| constant | derivation |
|---|---|
| AF3 split | median over the TRAIN edges |
| cohesion scale | `IQR/1.349` (a normal-consistent IQR); `1.0` only when `IQR == 0` is measured |
| triple weight `w` | `s_high * (1 - s_low)`, a product of independent bounded coalition values |
| softmax temperature | the mean assignment cost of the batch (measured per step), so logits are scale-free |
| loss scale | measured once on the first training step: `recon_loss / mechanism_loss`, then fixed (clamped to `[1e-8, 1e8]` only for numerical safety) |
| level weight | `1/3`: the three prefix levels are the SID levels and carry equal weight |

No margin, temperature, weight or split is selected by a grid or a Recall probe.

## The three switch definitions (Issue #121)

| arm | `USE_BEHAVIOUR` | `USE_GEOMETRY` | effect |
|---|---|---|---|
| both-on (trained) | true | true | AF3 triples and weights, hyperbolic costs |
| behaviour-off | false | true | random in-batch triples and unit weights; the AF3 side is gone |
| geometry-off | true | false | the same triples, squared Euclidean costs; the curve is gone |

Only the both-on arm is trained (19g); the other two exist in the code and are
exercised by the tests, which is what the range gate inspects.

## Evidence

* 70 tree tests, 2 skipped: AF3 exactness (K3 => +3, K4 => +4, P4 => plain
  Forman, `AF3-F=3T`), cohesion bounds/monotonicity/median, triple selection and
  batch confinement, prefix-agreement symmetry and curve sensitivity, loss
  monotonicity and non-zero encoder gradients, switch effects, the recording
  layer's equality with the baseline `RQLayer`, canonical-vs-inherited distance
  agreement, determinism.
* Pre-training pressure of the existing SIDs (`analysis_sid_recall/
  curvature_prefix_signal/`): the cohesive neighbour already shares the anchor's
  hard prefix more often than the bridge, +8.8 pp at L1 (positive in 98.1% of the
  108 arms), +1.46 pp at L2 (77.8%), +0.61 pp at L3 (74.1%). The ranking target is
  consistent with the learned hierarchy and its headroom is at L1/L2.
* Framework checks on this tree: `constants` PASS, `geometry_contract` PASS,
  `entry` PASS (one real epoch exports the identifier), `identifier` PASS
  (24,587 unique 4-token SIDs). One epoch logs `mech=0.6931` (the softplus(0)
  start) with the measured scale `7.26e-4`.

## Iteration for the owner's 20% collision gate (2026-11-03)

The first run collapsed the code usage (L2 prefixes 1,985, three-level collision
0.831; the inherited curved Frechet assignment alone gives 0.828 on the parent,
while the frozen baseline gives 0.2745). To honour the owner's single new gate
(three-level collision <= 20%) the assignment returns to the frozen baseline's
Euclidean rule and the balanced Sinkhorn assignment is enabled on every level
during training; inference stays the deterministic argmin, so an item's SID does
not depend on its batch. The AF3 ranking loss keeps the hyperbolic pathway: the
level residual and the (tangent) codebook are both mapped into the ball before
the prefix agreement. No hyperparameter was tuned; the change is an assignment
rule, which a candidate owns.

## Third-level refinement at export (owner's 20% gate)

The gate counts the first three tokens. The learned third-level assignment is not
injective inside a (k1,k2) bucket: the first iteration of this arm reached a gate
collision of 0.306 even after the assignment change. The export therefore applies
one deterministic rule: the first occurrence of every third token in a bucket is
kept, every duplicate is reassigned to the smallest unused code of that level in
ascending item order, and the first two tokens never change. A bucket larger than
the level size cannot be resolved; its surplus is counted, logged and recorded
rather than silently claimed unique. The trainer's own collision metric now uses
the gate's definition (the share of items in shared triples), not `1 - unique/n`.

This is an assignment rule, declared as such: the third token remains the learned
code whenever it is unique, and the disambiguation token is still appended. It is
not a trained property and must not be reported as one.

## What is not claimed

The arm is implemented and technically conformant; it has **not** been trained
for 3000 epochs and no Stage3 `test_R@10` exists for it. The formal run goes
through `stage2_start(source_dir, verification_id)` after the independent
verification suites are authored and passed, and the frozen 3000-epoch budget and
Stage3 protocol are untouched.
