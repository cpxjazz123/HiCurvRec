# The community-guided graph reconstruction (CGQ) mechanism

## What the arm changes and what it does not

The frozen baseline `TIGER_RQ-VAE` encodes an item's semantic embedding into
three residual-quantised levels and reconstructs it. The community arms before
this one changed the level-1 *assignment cost*: the community decided which
items should share a level-1 code, and the identifiers came out coarser. The
measurements in this repository say that route does not carry: the coarse level
improved and the recall fell.

This arm keeps the quantiser, the three levels, the codebook sizes and the
balanced assignment exactly as they are, and adds **one** loss term:

> the quantised representation of an item must predict the distribution of its
> behaviour neighbours, where the neighbours are weighted by the community
> model's own geometry.

Nothing about the assignment changes. The mechanism enters through the loss,
not through the cost matrix, and the loss is what the quantiser is trained to
minimise.

## The frozen inputs

Three frozen artefacts, all derived from the **training split only**:

1. **The behaviour graph** `A_ij`: two items are adjacent when they follow each
   other in some user's training sequence (the frozen undirected co-occurrence
   convention of the community package). 24556 nodes, 315391 edges.
2. **The community model's two representations**: the free-branch point
   `u_i^E` (64-d) and the Poincaré-ball point `u_i^H` (64-d) of every item,
   exported by `tools/research/science/community_curvature/export_representation.py`
   from the frozen seed-42 community model (the curvature-aware GCN arm
   `E_free_64_hyper_64_curvgcn`).
3. **The community assignment** `pi_i`: the frozen model's soft distribution
   over the 64 communities.

## The edge weight

The community geometry re-weights the behaviour edges, and only the behaviour
edges. With `A_ij` the adjacency and `s_ij` the geometric affinity:

```
s_ij = exp[ -1/2 ( d_E(u_i^E, u_j^E)^2 / (m_E + eps)
                 + d_H(u_i^H, u_j^H)^2 / (m_H + eps) ) ]

w_ij = A_ij * (1 + s_ij)
```

`d_E` is the Euclidean distance in the free branch, `d_H` the contract's
Poincaré distance in the ball branch at the teacher's own curvature (the
`distance` of `tools/research/geometry/poincare_contract.py`, the same
operator the community model itself uses). `m_E` and `m_H` are the **median**
squared distance over the training edges of each branch, so the two branches
enter on a comparable scale; `eps = 1e-12`.

Three properties this construction has, and they are the reason for it:

* `s_ij` is in `(0, 1]`, so `w_ij` is in `[1, 2]` **on an existing edge** and
  `0` everywhere else. A real edge is never deleted and a non-edge is never
  created; the community can only decide how much an observed behaviour
  relation counts.
* A cross-community edge keeps a real weight, so the supervision says nothing
  about which community a behaviour relation belongs to.
* Two items that merely share a community and never co-occur in a sequence get
  `w = 0`: they are not positives. They may appear in a candidate set as
  contrast terms, which the loss treats as *candidates*, not as known
  irrelevances.

## The loss

The quantised representation of item `i` is the **sum of its three codeword
vectors**:

```
q_i = sum_{l=1..3} e^{(l)}_{c_i^{(l)}}
```

not the level-1 code and not the residual. The whole point is that the
identifier is the three codes together, so the loss must see all three.

For an anchor `i` with candidate set `C_i` (its real neighbours, some
same-community candidates and some other-community candidates), the model
predicts a distribution over the candidates from the cosine similarity of the
quantised representations:

```
p_theta(j | i, C_i) = softmax_{j in C_i} ( cos(q_i, q_j) / tau )
```

and the target is the behaviour neighbourhood, renormalised inside the
candidate set:

```
t_ij = w_ij / sum_{k in C_i} w_ik
```

The cross-entropy against that target is the term:

```
L_CGQ = -1/|B_+| sum_{i in B_+} sum_{j in C_i} t_ij log p_theta(j | i, C_i)
```

`B_+` is the set of anchors whose candidate set contains at least one real
neighbour; an anchor with no observed behaviour relation contributes nothing,
because there is no target for it. `tau = 0.2` is fixed.

The total objective is the baseline's own plus the new term:

```
L_total = L_reconstruction + L_quantisation + lambda(t) * L_CGQ
```

## The two gradient paths, and why both are needed

The baseline's straight-through estimator is `x_q = x + (q_e - x).detach()`:
the gradient reaches the **encoder** and not the codebook. A loss built on that
tensor alone therefore never moves a codeword directly, and a claim that the
codebook is supervised by the new term would be false.

So the term is computed on **two** representations of the same item:

* `q_i^ste`: the straight-through sum, whose gradient reaches the encoder;
* `q_i^cb`: the plain sum of the three codeword vectors **without** any
  detach, whose gradient reaches the codebook.

and the two losses are averaged:

```
L_CGQ = 1/2 * L(q^ste) + 1/2 * L(q^cb)
```

The candidate representations `q_j` use the same two-path construction, so both
halves see the same geometry. The test in `tests/` asserts what the code
cannot assert by construction: after one backward of the term alone, the
encoder's parameters and the codebook's parameters have **both** received a
non-zero gradient. A mechanism that only wires one of the two paths fails it.

## The schedule, the temperature and the gradient budget

`lambda(t)` is zero for the first 1500 epochs and rises linearly to one over
the second 1500. The first half of the run is the baseline's own objective, so
the quantiser first learns a representation that reconstructs the semantics and
only then learns to predict the behaviour neighbourhood. The loss cannot
dominate the reconstruction from the first step.

The new term's **gradient contribution is capped at 10% of the original
objective's**, checked separately for the encoder and for the codebook, and the
stricter of the two scalings is the one applied. Concretely, per step:

```
g_base_enc  = || d L_original / d encoder_parameters ||
g_cgq_enc   = || d L_CGQ     / d encoder_parameters ||
g_base_cb   = || d L_original / d codebook_parameters ||
g_cgq_cb    = || d L_CGQ     / d codebook_parameters ||

scale_enc = min(1, 0.10 * g_base_enc / g_cgq_enc)
scale_cb  = min(1, 0.10 * g_base_cb  / g_cgq_cb)
scale     = min(scale_enc, scale_cb)
```

The cap is computed on the *applied* gradient norms, so it holds whatever the
scale is: the term never moves the encoder or the codebook by more than a
tenth of what the baseline's own objective moves them by. The scale is a
read-only statistic of the step (it is computed with `torch.autograd.grad` on
a retained graph, and it never enters the backward pass itself), so it does not
change what the optimiser sees.

## The candidate set

Per anchor, the candidate set is built from the frozen structures only:

* the anchor's real behaviour neighbours (mean degree 12.8, capped at the
  32 heaviest by `w_ij`);
* up to 8 other items of the anchor's own community, which carry `w = 0`;
* up to 8 items of other communities, which also carry `w = 0`.

The non-neighbours are there to give the softmax a denominator. They are
*candidates*, and the loss says nothing stronger about them than that.

## What the run has to show

The arm's own training record answers three questions, and the answer has to be
in the record rather than in a narrative:

1. **Does the community weight change the target?** The distribution over the
   real neighbours must differ from the uniform one over the same neighbours.
   If `w_ij` were constant the term would be an unweighted contrastive loss and
   the community would have contributed nothing. The record reports the
   weight's spread and the target's entropy against the uniform target's.
2. **Does the term change the encoder, the codebook and the exported
   identifiers?** Gradient norms for both parameter groups, and the change in
   the exported identifiers between the epoch before the term starts and the
   final epoch.
3. **Are the neighbours better preserved without wrecking the hierarchy?** The
   neighbour overlap of the exported identifiers (do two real behaviour
   neighbours share a level-1 code?) against the frozen baseline's, together
   with the number of distinct level-2 prefixes, the level-3 collision rate
   and the semantic reconstruction error. These are computed on the **exported**
   identifiers, not on the training-time soft assignment.

## What this arm does not claim

A better neighbour overlap in the identifiers is not a better recall. The
identifiers are the input of Stage 3, and Stage 3 decides the recall. The arm
is judged by `test_R@10` against the frozen baseline's `0.055537`, measured
under the same Stage 3 protocol, and the AF3 gap is not a gate.
