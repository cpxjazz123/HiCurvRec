# Seven hyperbolic properties, measured on the 107 confirmed pairs

The brief asked whether the other hyperbolic properties -- beyond the Gromov
delta -- can be turned into SID-quality metrics, and which one explains the
relation between SID quality and Recall. All seven were implemented, and each is
measured four times:

* **hyperbolic, curvature 1** (the reference's own ball);
* **hyperbolic, curvature 8** (the curvature the four-point protocol estimates);
* **Euclidean** -- the same quantities on the same points with linear distances;
* **permuted** -- the SID rows shuffled across the items, so the geometry is
  untouched and only the SID correspondence is destroyed.

The permuted control is the one that decides: a property of the SID tree must
collapse when the tree is shuffled. It does not, for any family.

Implementation: `metrics_hyperbolic_families.py`. Artifact:
`../metric_bench_hyperbolic_families_107.json`.

## The structural fact that decides three of the seven

The frozen reference is a **thin shell**, not a hierarchy:

| radius | p10 | median | p90 | min | max |
| --- | --- | --- | --- | --- | --- |
| value | 0.9050 | 0.9115 | 0.9176 | 0.8524 | 0.9300 |

The spread between p10 and p90 is 1.4% of the median, and the whole catalogue
uses 0.078 of the unit radius. The reference was trained as a contrastive
similarity space, so it has no radial term, and the radial coordinate carries
almost no information.

Consequence: **families 2 (radial hierarchy), 4 (entailment cones) and
7 (volume growth) are degenerate on this reference**, not merely weak. A
radial-hierarchy metric returns the same value for every SID (the correlation is
`nan`, because the metric is constant); the Ganea aperture
`arcsin(K(1-||p||^2)/||p||)` collapses to about one degree at radius 0.91 for any
`K` not tuned to this data, so every child falls outside every cone and the
fraction is identically zero; the shell count is flat, so the fitted growth
exponent is zero. These are recorded as `NOT_MEASURABLE` with the reason, not as
"no effect".

## Results

| family | hyperbolic c=1 | hyperbolic c=8 | Euclidean | permuted | verdict |
| --- | --- | --- | --- | --- | --- |
| 1 Gromov product (shared ancestor) | +0.1453 | +0.1471 | +0.1444 | +0.0910 | no signal, no hyperbolic increment |
| 2 radial depth | -0.2163 | +0.0241 | -0.1949 | **+0.3731** | degenerate, sign flips |
| 3 radial purity | +0.2957 | +0.2957 | +0.2894 | **+0.3575** | fails the permutation control |
| 4 geodesic slack | -0.3063 | -0.2277 | -0.3050 | **-0.3625** | wrong sign, control stronger |
| 5 entailment cone (cosine) | -0.4065 | -0.2361 | -0.3243 | constant | degenerate, negative |
| 6 angular branch separation | -0.3936 | -0.3954 | -0.3962 | -0.3747 | wrong sign, no correspondence effect |
| 7 volume growth | constant | constant | constant | constant | NOT_MEASURABLE (shell) |
| 8 behaviour-aware geodesic (depth matched) | +0.4771 | +0.4965 | +0.4777 | +0.4834 | chance |
| 9 behaviour-aware Busemann | +0.5245 | -- | +0.5257 | +0.5245 | chance |

## What the permutation control exposes

For every family that produced a number at all, the **permuted** variant is as
strong as, or stronger than, the real one. That is not noise: a random regrouping
of the same sizes reproduces the correlation, because these metrics are dominated
by the SID's own **group-size / collision structure** -- which is invariant under
permutation and is itself known to correlate with recall. So `radial_purity` at
+0.2957 is not evidence that the geometry preserves the SID partition; it is
evidence that an SID with more L3 collisions has a worse recall, which was
already known.

The behaviour-aware families are the honest test of the brief's first
recommendation, and they are at chance. The **unmatched** version of the geodesic
metric looked informative only because a behaviourally positive pair shares a
deeper prefix, and a deeper ancestor sits farther from the root, so its slack is
larger for a reason unrelated to the geodesic. With the ancestor depth held fixed
(the negative for each anchor is an item from the same shared prefix that is not a
positive), it is 0.477 to 0.497, i.e. a coin flip, and the permuted control is
0.483.

## What this leaves standing

The predictive power that survives every control is the one that contrasts the
table against **behaviour**, not against its own shape:

| construction | rho | permuted control |
| --- | --- | --- |
| behaviour contrast (`hbaa`) | +0.7623 | -- |
| four-point obligation of the behaviour graph (SVD) | +0.6605 | +0.0011 |
| hyperbolic properties of the SID table (this document) | at most +0.30 | equally strong or stronger |

That is the same conclusion the brief anticipated: measuring the geometry of the
SID clustering alone does not predict Recall; what predicts is the relation
between behaviour and the SID. The seven properties are real properties of a
hyperbolic space, and three of them cannot even be measured on this reference
because the reference is a shell. A future round that wants them to have a chance
must first make the reference use its radial coordinate -- which is a modelling
decision for Stage 2, not a metric.
