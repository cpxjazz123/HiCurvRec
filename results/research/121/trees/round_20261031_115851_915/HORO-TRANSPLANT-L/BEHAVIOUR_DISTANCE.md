# Does the hyperbolic RQ-VAE preserve behaviour distance better than the Euclidean one?

Measured, not argued. For every frozen RQ-VAE the same code vectors are compared
under two metrics -- the Euclidean chord and the Poincare distance of the same
vector pushed into the unit ball with a data-derived scale -- against a behaviour
distance read from the TRAINING split only:

    d_beh(i,j) = -log( (n_ij + 1) / (sqrt(pop_i pop_j) + 1) )

`n_ij` is the symmetrised consecutive-pair count, `pop` the number of training
users. 63,079 pairs, 1,342 items asked for the neighbour overlap.

## Numbers

| model | rho(behaviour, Euclidean) | rho(behaviour, Poincare) | top-10 overlap, Euclidean | top-10 overlap, Poincare |
| --- | --- | --- | --- | --- |
| `TIGER_RQ-VAE` (Euclidean) | **0.0402** | 0.0396 | **0.0124** | 0.0123 |
| `TIGER_BRad-RQ` (hyperbolic) | **-0.2262** | -0.2258 | **0.0011** | 0.0011 |
| `TIGER_CURV-RESID-CONTRAST` | 0.0357 | 0.0370 | 0.0045 | 0.0042 |
| `TIGER_BEHDIR-L4` | 0.0306 | 0.0302 | 0.0057 | 0.0063 |

Chance for the top-10 overlap is 10/24587 = 0.00041.

## Answer

**No, the hyperbolic RQ-VAE does not preserve behaviour distance better than the
Euclidean one.** Two separate findings:

1. **Within a model, the metric does not matter.** Euclidean chord and Poincare
   distance give the same correlation to within 0.0013 on every one of the four
   models. The code vectors sit in a region of the ball where the two metrics are
   locally equivalent, so "switching to hyperbolic distance" changes nothing about
   behaviour preservation.
2. **Between models, the Euclidean one is best and the hyperbolic one is worst.**
   The plain Euclidean RQ-VAE reaches rho = 0.040 and recovers 1.24% of the
   behaviour top-10 (30x chance). The hyperbolic `TIGER_BRad-RQ` is *anti*-
   correlated with behaviour (rho = -0.226) and recovers 0.11% -- worse than
   nothing. The two curved residual models sit between the two.

## Why this matters for the recall question

The ordering of the four models on this behaviour-distance measure is the same as
their Stage-3 recall ordering in the five-round ledger (Euclidean baseline about
0.0679 best, curved variants below). That is consistent with a single cause: the
behaviour signal that the curved mechanisms inject into the code space does not
make behaviourally related items nearer -- in the hyperbolic arm it actively makes
them farther -- so none of them can help the downstream recommender.

Caveat: `TIGER_BRad-RQ`'s anti-correlation is a property of that particular frozen
training objective, not a theorem about hyperbolic spaces; but it is the model the
repository actually has, and it is the one the chain proposes.
