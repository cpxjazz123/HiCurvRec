# Three single-term auxiliary losses on the frozen RQ-VAE

Baseline: `stage2_RQ-VAE/TIGER_RQ-VAE`, copied three times. Each arm adds **exactly
one** term to `recon + quant`; the other two stay implemented and unit-tested but
switched off (`AUX_MODE` in `model/aux.py`).

The earlier version of these three terms measured the wrong objects and collapsed
the SID (one or two distinct level-2 prefixes). The definitions below are the
corrected ones, and the tests in `tests/test_aux_loss.py` check them in three
layers: the definitions, the structure the terms prefer, and the gradients.

## The frozen reference

Every term compares the SID against a **frozen** reference geometry: the latents
of the frozen baseline encoder, mapped into the unit ball once
(`train_rqvae._frozen_reference`). It does not move while a tree trains, so a
term that compares against it **cannot be minimised by collapsing the tree** --
the other side of the comparison does not move.

## 1. Gromov product / SID alignment (`AUX_MODE="gp"`)

For a metric space `X` and a base point `o`:

    G_X(i,j) = ( d_X(o,i) + d_X(o,j) - d_X(i,j) ) / 2

The term asks the Gromov product of the **SID tree** and the Gromov product of the
**reference geometry** to agree:

    T(i,j) = L - LCP(i,j)                       (the SID tree distance)
    R(i,j) = d_H(phi(z_ref(i)), phi(z_ref(j)))  (the reference distance)
    L_GP = mean_{i != j} ( z(G_tree(i,j)) - z(G_ref(i,j)) )^2

`LCP` is the expected longest common prefix of the soft assignment,
`sum_l prod_{l' <= l} <p_l'(i), p_l'(j)>` -- a **product** over levels, because a
level counts only if every level above it agrees.

## 2. Tree distortion (`AUX_MODE="tree"`)

    L_tree = mean_{i != j} ( z(T(i,j)) - z(R(i,j)) )^2

with the same LCP tree distance `T` and the same frozen reference distance `R`.
This is the "the SID tree distance reproduces the reference hyperbolic distances"
term; it is the one that the earlier per-level sum got wrong (a shared level-2
code must not shorten the distance when level 1 already differs).

## 3. Hyperbolic Frechet clustering (`AUX_MODE="cluster"`)

Every item has two positions in the ball: its **pre-quantization** position
`phi(z_i)` and the **expected code point** of its prefix. A group's centre is the
**hyperbolic Frechet mean** of its members -- the minimiser of
`sum_i w_i d_H(x_i, m)^2`, computed by Riemannian iterations. The weights enter
the gradient, never the points.

    intra = mean_c [ mean_{i in c} d_H(item_i, m_c)^2 + mean_{i in c} d_H(code_i, c_c)^2 ]
    inter = mean_{c != c'} d_H(c_c, c_c')^2
    L_cluster = 1 - (inter - intra) / (inter + intra)

The ratio is 1 at collapse (intra = inter = 0) and 0 when the prefixes are compact
and mutually separated; the second half of `intra` is what gives the codebooks a
gradient.

## The gradient scale, measured

Matching the **magnitude** of the losses is not enough and was measurably wrong:
the auxiliary term is O(1) while the reconstruction term decays to ~1e-4, and the
measured auxiliary gradient on the codebooks was **4.6e6 times** the baseline's.
The term is therefore rescaled at every step by the **measured ratio of the two
codebook gradients**, read from the same forward pass:

    scale = ||grad_codebook(recon + quant)|| / ||grad_codebook(aux)||

so the auxiliary pull on the codebooks is at most as strong as the baseline's own
pull. No constant is involved: both norms are measured.

Note that in this model the decoder path is detached from the codebook
(straight-through), so the baseline's codebook gradient comes from the quantizer
term; that is the gradient the auxiliary term competes with.

## The three test layers

1. **Definitions** -- the tree distance follows the longest common prefix
   (`[1,2]` vs `[3,2]` is a distance of 2, `[1,2]` vs `[1,9]` is 1); the Gromov
   product matches its formula; the Frechet mean differs from the arithmetic mean
   and has the smaller mean squared hyperbolic distance.
2. **Structure** -- a healthy tree scores better than a collapsed one under the
   alignment terms; the collapsed batch is the worst case for the cluster term;
   and the alignment terms cannot be minimised by collapsing, because the
   reference side is frozen.
3. **Gradients** -- the auxiliary term reaches the codebooks at every level; its
   directional derivative along the negative gradient is negative (a descent
   direction); and the auxiliary contribution to the codebook gradient that
   actually enters the objective is bounded by the baseline's own.

## Status

The Stage 2 / Stage 3 numbers these arms produced before the correction
(0.0476, 0.0391, 0.0484) came from the **old** definitions and are void: the
terms were not the objects they were named after.

## The Stage 2 results (four cards, frozen recipe)

| arm | L1 codes | L2 prefixes | L3 triples | duplicate triples | final loss | recon |
| --- | --- | --- | --- | --- | --- | --- |
| `TIGER_RQ-VAE` (baseline) | 247 | 7,414 | 20,223 | 0.177 | -- | -- |
| `TIGER_AUX-GP-SID` | **157** | **5,517** | **17,719** | **0.279** | 0.00264 | 8.4e-5 |
| `TIGER_AUX-TREE-DISTORT` | **208** | 2,578 | 6,877 | 0.720 | 0.0347 | 1.1e-4 |
| `TIGER_AUX-FRECHET-CLUSTER` | 21 | 192 | 2,103 | 0.914 | 0.00825 | 1.4e-4 |

**GP is healthy**: it keeps 96% of the baseline's level-1 cells, 74% of the
level-2 prefixes and 88% of the triples, with the duplicate rate close to the
baseline's. It trains stably (the loss stays at 2.6e-3) and its reconstruction
term is the same order as the baseline's.

**TREE keeps its coarse level** (208 of 247 cells) but compresses the deep ones:
the tree distance term is satisfied by a tree with few deep cells.

**CLUSTER trains now but is still compressed.** Three implementations were tried:
a hard grouping with detached centres (the arm collapsed to 1/1/3 and the term
had no gradient at all), a hard grouping with differentiable centres (the item
positions collapsed and took the partition with them), and the current one: a SOFT,
fully vectorised clustering whose centres are the mass-weighted hyperbolic Frechet
means with one Riemannian step. It trains stably and keeps 21/192/2103 against
the baseline's 247/7414/20223 -- no longer a collapse, but the vocabulary is
still traded away for the objective's own geometry.

The general lesson, which the three arms share: a term that is about PAIRWISE
structure leaves the vocabulary unconstrained, and the quantizer will trade
vocabulary for whatever the term asks. The partition guard (soft mutual
information against the frozen reference) is what keeps the coarse level alive
(L1 208 of 247 for TREE, 157 for GP); the deeper levels need the same kind of
protection, which none of the three terms provides on its own.

The three mechanisms therefore stand at different states, and the Stage 3 runs
are deliberately not started yet.
