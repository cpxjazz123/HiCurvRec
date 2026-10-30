# The delta-guided four-point SID consistency: the protocol, fixed before the numbers

Everything below is decided BEFORE any correlation with Recall is computed. The
metric may fail; the protocol is what makes its failure interpretable.

## The three inputs

* the **frozen reference** of Issue #126 (per-item source and target
  representations, curvature 1, temperature 1.195), read only;
* the **exported SID table** of each of the 107 confirmed pairs (and the 77 of them
  that never took part in choosing a metric);
* the **recall** of each pair, used only at the very end.

No validation or test interaction is read. Nothing is trained. The item↔SID
correspondence is never permuted in the metric itself; the permuted table is a
CONTROL, measured separately.

## Version 2 -- what the first computation got wrong

The first run of this protocol is superseded. Three corrections, each found by
checking the code against the definition:

1. **The delta was not the Gromov delta.** The first version took the fixed
   pairing difference `|S3 - S2|`. The delta of a quadruple is the gap between
   the LARGEST and the SECOND LARGEST of the three pairing sums, halved:
   `delta_4 = (S_max - S_second) / 2`. The fixed difference changes when the
   four points are relabelled and is not zero for a tree metric, so it measured
   something else -- and it was also what the curvature was estimated from.
2. **The two deltas were on different scales.** The data's delta was computed
   with plain Euclidean distances on the tangents while the model's used
   Poincare distances; as the curvature goes to zero the hyperbolic distance
   tends to `sqrt(2)` times the Euclidean one. The data's delta is therefore
   measured in the REFERENCE'S OWN geometry -- its curvature-1 ball, with the
   same distance convention -- and the model at curvature `c` uses the same
   convention on the re-mapped tangents.
3. **The analysis set is the whole 107 pairs**, with no split and no second set.

With the corrections the criterion has an INTERIOR optimum: `c_delta = 1.0` with a
gap of 0.0000, which is the reference's own curvature -- the estimator recovers
it, so the scale is consistent now. The earlier `c_delta = 0.05` was an artefact
of the two mistakes above.

## Step 1 -- the curvature from the Gromov delta

The reference's representations are points of the curvature-1 ball. Their **tangent**
coordinates are the data space:

    t_i = log_1(x_i) = artanh(|x_i|) * x_i / |x_i|

The curvature is chosen so that the four-point behaviour of the hyperbolic model
matches the behaviour of the DATA, which is the standard estimation of a curvature
from the Gromov delta (Chami et al., 2019): the curvature whose quadruples are as
close to a tree as the data's quadruples are.

    grid          : c in {0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8}
    data delta    : the quadruples' delta in the REFERENCE'S OWN geometry (its
                    curvature-1 ball, the same distance convention the model uses)
    model delta   : the same quadruples in the ball of curvature c, reached by
                    re-mapping the SAME tangent vectors: x = exp_c(t)
    delta         : (S_max - S_second) / 2 over the three pairing sums
    criterion     : minimise |median |delta|_model(c) - median |delta|_data|
    tie-break     : the smaller curvature

`c_delta` is computed ONCE, from the reference alone. No recall enters it.

## Step 2 -- the metric

For every SID table:

1. **quadruples**: 20000 quadruples (A, B, C, D) drawn from the catalogue with
   the seed 42 -- the SAME quadruples for every table, so the tables are compared
   on the same items;
2. **the three pairings** of the quadruple:
   S1 = d(A,B) + d(C,D), S2 = d(A,C) + d(B,D), S3 = d(A,D) + d(B,C);
3. **the preferred pairing** of a metric is the one with the SMALLEST sum: in a
   tree that is the pairing that does NOT cross the two branches;
4. **the tree-likeness weight** of the quadruple is
   w = 1 / (1 + |delta| / m), with delta = the model delta (S3 - S2, the deviation
   of the four-point condition) and m = the MEDIAN |delta| of the sampled
   quadruples, computed ONCE at c_delta and shared by every table and every
   variant, so the weights are not a per-table quantity;
5. **the score** is the weighted agreement:
   Q = sum_q w_q * 1[the geometry and the SID tree prefer the same pairing]
       / sum_q w_q
   with the SID tree's distances being the path lengths of the prefix tree (the
   codes from the first mismatch onwards).

Q is in [0, 1] and is ONE number per table. A random item↔SID correspondence is
worth about 1/3.

## Step 3 -- the controls

| variant | curvature | purpose |
|---|---|---|
| A | c_delta | the hypothesis |
| B | 0.5 c_delta | a weaker curvature |
| C | 2 c_delta | a stronger curvature |
| D | 0.001 | nearly flat |
| E | Euclidean on the tangent vectors | does the hyperbolic geometry add anything? |
| F | the item↔SID correspondence permuted (seed 42) | is the metric only about the SID's shape and codebook usage? |

Every variant re-maps the SAME tangent representations into its own ball; no
variant reuses coordinates that may lie outside its ball.

## Step 4 -- what decides the outcome

* Spearman of each variant with Recall@10, on the 107 pairs and on the 77 pairs
  that never took part in choosing a metric;
* the comparison that matters: variant A against variant E (Euclidean) and
  against D (nearly flat). If the curvature chosen FROM the delta does not beat
  them, the delta-guided curvature carries no information;
* the permuted control must land near 1/3, and its correlation with Recall must be
  near zero.

The Gromov delta is NOT computed on the SID prefix tree itself: a tree's delta is
zero by definition, so it cannot separate the tables.

The claim that would be supported is NOT "a smaller delta means a better recall".
The claim is: **the curvature estimated from the Gromov delta makes the SID's
four-point structure predict the recall better than other curvatures and better
than the Euclidean control.**
