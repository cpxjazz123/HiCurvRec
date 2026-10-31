# Three single-term auxiliary losses on the frozen RQ-VAE

Baseline: `stage2_RQ-VAE/TIGER_RQ-VAE`, copied three times. Each arm adds **exactly
one** term to `recon + quant`; the other two stay implemented and unit-tested but
switched off (`AUX_MODE` in `model/aux.py`). Everything else — data, budget,
launcher, export — is the baseline's.

## The shared machinery

Soft assignment at level `l`, with a temperature taken from the batch itself:

    p_l(i) = softmax_k( -d_l(i,k) / median(d_l) )        (median detached)

Ball map with a batch scale (detached): `phi(x) = tanh(||x||/s) x/||x||`,
`s = max ||x||`; Poincare distance `d_H` in the unit ball; `D = diameter` of the
unit ball, so a distance divided by `D` lives in `[0,1]`.

Every term is standardized (zero mean, unit variance, detached statistics) over
its pair set, so its shape is scale-free. Its MAGNITUDE still has to compete with
the reconstruction term, which decays to ~2e-4: an equal nominal weight lets the
auxiliary objective own the codebooks and collapses the SID (measured on the first
attempt: one distinct level-2 prefix, total loss 4.9e4 against recon 2e-4). The
term is therefore rescaled at every step by

    scale = recon_loss.detach() / |aux.detach()|

so it enters the objective at the magnitude of the reconstruction loss itself.
The ratio is measured, not tuned, and `AUX_WEIGHT` stays 1.0.

## 1. Gromov product / SID alignment (`AUX_MODE="gp"`)

    A_ij = sum_l  prod_{l' <= l} <p_l'(i), p_l'(j)>                  shared prefix depth
    c_l(i) = sum_k p_l(i,k) phi(e_lk)                                 expected ancestor
    H_ij = sum_l  prod_{l' <= l} (1 - d_H(c_l'(i), c_l'(j)) / D)      shared curved depth
    L_GP = 1 - corr( A_ij , H_ij )        over i != j

The left side is what the SID tree says (how many levels two items share); the
right side is what the hyperbolic codebook says (how deep their expected code
points agree). The term pulls them together.

## 2. Tree distortion (`AUX_MODE="tree"`)

    T_ij = sum_l (1 - <p_l(i), p_l(j)>)                expected SID tree distance
    R_ij = d_H(phi(z_i), phi(z_j)) / D                reference hyperbolic distance
    L_tree = 1 - corr( T_ij , R_ij )      over i != j

A stress term: the tree distance between two items has to reproduce the distance
between their pre-quantization representations in the ball.

## 3. Hyperbolic Frechet clustering (`AUX_MODE="cluster"`)

Groups are the prefixes of the first level that separates the batch (a hierarchical
fallback: at initialization one level can put the whole batch on one code, and a
single group gives no contrast). A centre is the soft-mass-weighted mean of the
members' ball points; the term is

    L_cluster = 1 - ( inter - intra ) / ( inter + intra )

compact inside a prefix, separated across prefixes.

## The requirement that the gradient reaches the codebooks

`tests/test_aux_loss.py` builds the model the way the trainer does (KMeans codebook
initialization), runs a real batch, backpropagates the total objective and asserts
that every level's codebook has a **non-zero** gradient, and that the encoder does
too. `AUX_MODE="none"` is asserted to reproduce `recon + quant` exactly.

## Why the terms are not constants in disguise

* the softmax temperature is the median distance of the batch, detached;
* the ball scale is the batch maximum norm, detached;
* the standardization uses the batch's own mean and standard deviation, detached;
* the only free number is the weight, and the standardization makes 1.0 the
  unit-scale choice.

## Why the terms are correlations and not squared differences

The first implementation used a squared difference of standardized pair statistics.
That objective has a trivial minimum: make both sides constant (all items on one
code, or all latents on one point) and the loss is exactly zero. It collapsed the
SID to one or two level-2 prefixes in the first run, twice, with and without the
reconstruction-relative rescaling.

The correlation form keeps the same alignment meaning and removes the degenerate
optimum: a constant side has zero correlation, which is the WORST value of the
loss, not the best. The cluster term uses the normalized ratio for the same
reason: it is 1 at collapse and 0 when prefixes are compact and mutually
separated.
