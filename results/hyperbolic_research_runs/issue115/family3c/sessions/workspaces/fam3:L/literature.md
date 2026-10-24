# Role L — literature screen and unoccupied mechanism ideas

## Scope and evidence standard

The target is a non-trivial hyperbolic operation in the rule that selects one of the frozen discrete SID codes. The 32-dimensional latent, three 256-entry codebooks, three integer tokens, dataset/splits and Stage 2/3 recipes remain fixed. A literature overlap is not a claim that a paper beats the frozen thresholds. “FULL_TEXT” below means the cited arXiv manuscript is publicly available and was inspected for the stated scope; it does not mean every implementation detail was independently reproduced. No directly matching hyperbolic *discrete recommender SID assignment* method was confirmed in this screen. That is a bounded search conclusion, not a novelty proof.

## Verified literature and precise overlap

| Identifier | Title | Year | Access | Precise overlap with the proposed problem |
|---|---|---:|---|---|
| [arXiv:1705.08039](https://arxiv.org/abs/1705.08039) | *Poincaré Embeddings for Learning Hierarchical Representations* — Maximilian Nickel and Douwe Kiela | 2017 | **FULL_TEXT** | Establishes learning continuous representations in the Poincaré ball using hyperbolic distance and optimization. This is relevant geometric representation/metric precedent; it does **not** define residual vector quantization, a three-token SID, or a hyperbolic operation for discrete code selection. |
| [arXiv:1805.09112](https://arxiv.org/abs/1805.09112) | *Hyperbolic Neural Networks* — Octavian Ganea, Gary Bécigneul and Thomas Hofmann | 2018 | **FULL_TEXT** | Defines neural operations using the Poincaré ball, including Möbius operations and tangent-space maps. It is operator-level precedent for finite-dimensional hyperbolic computation, not a discrete SID/codebook selection method. In particular, its existence does not validate inserting a Möbius associator into a score (the frozen protocol already rejects that family). |
| [arXiv:1805.09786](https://arxiv.org/abs/1805.09786) | *Hyperbolic Attention Networks* — Octavian Ganea, Gary Bécigneul and Thomas Hofmann | 2018 | **FULL_TEXT** | Applies hyperbolic representations and attention to graph-structured data. Overlap is hyperbolic geometry participating in a learned weighting/aggregation operation; it does not select discrete VQ/SID code indices. |
| [arXiv:1904.02239](https://arxiv.org/abs/1904.02239) | *Hyperbolic Image Embeddings* — Yash Patel, Giorgos Tolias and Jorma H. Laaksonen | 2019 | **FULL_TEXT** | Uses hyperbolic embeddings for visual retrieval. Overlap is retrieval geometry and representation learning; no three-level discrete SID assignment operator is specified. |

**Boundary:** these are genuine, citable hyperbolic-geometry precedents, but none is presented as a direct code-selection match. The initial local literature cache/search log had noisy broad-query returns and was not treated as evidence for overlap. No DOI/arXiv citation for a directly matching hyperbolic residual-codebook selector is asserted. A focused follow-up should inspect papers returned by exact queries such as “hyperbolic vector quantization”, “hyperbolic VQ-VAE”, and “hyperbolic semantic IDs”; until individually checked, those are leads, not citations.

## Candidate mechanism ideas not occupied by the inspected papers

These are proposals, not claims of novelty or performance. Each acts at code-index selection. The occurrence-rate statistic must be computed on the frozen assignment stream **before** training/proposal acceptance; report numerator, denominator, and rate per level. Evaluate every valid item/level assignment (and, where relevant, each candidate code comparison), with ties and invalid rows included in the denominator rather than silently dropped.

### 1. Hyperbolic midpoint-neighborhood assignment (candidate)

At level `l`, after the frozen residual quantizer provides its ordinary Euclidean shortlist, compare the input residual `x` to shortlisted prototypes `c` by the Poincaré-ball distance from `x` to the geodesic midpoint between `x` and `c`, with a fixed nonzero prototype-independent scale term omitted. This tests a genuine geodesic construction at the discrete choice, rather than using a codeword radius or replacing the first-level objective with product-geodesic radius scoring. **Pre-rate:** fraction of frozen choices whose winning code changes when the midpoint score replaces the baseline score; compute exactly by replaying frozen embeddings/codebooks, no training.

**Finite-precision declaration required before implementation:** fix curvature `kappa>0`, input/prototype ball radius `r_max < 1/sqrt(kappa)`, float64 score arithmetic, and projection `x <- x * min(1, r_max/||x||)` (zero vector unchanged). Define the geodesic midpoint as `exp_x(0.5*log_x(c))`; implement the standard Poincaré exp/log formulas with each square-root/atanh argument clamped to the open domain by `eps=1e-12`, and norms below `eps` treated as zero. Compute the candidate score as squared Poincaré distance from the frozen residual to that midpoint. Compare scores with absolute tolerance `tau=1e-12`: lower score wins outside tolerance; within tolerance use the frozen baseline index, then lower integer code as deterministic fallback. Any NaN/Inf score falls back to baseline. This definition is a specification, not permission to implement until formula-level boundary tests are written.

### 2. Signed geodesic-side preference for a frozen transition pair (candidate)

For a current code candidate and the next-item representation available in the frozen training pair, use the signed side of the candidate-to-anchor geodesic in a two-dimensional tangent subspace to bias the discrete choice. It is not the parked unsigned wedge-area mechanism: sign is essential, and the operation is applied to code selection rather than as an unsigned area regularizer. **Pre-rate:** fraction of frozen transitions with both (i) non-degenerate projected tangent vectors and (ii) a signed score difference exceeding the comparison tolerance; calculate from the assignment/transition stream without fitting.

**Finite-precision declaration required:** choose the two tangent axes deterministically from fixed latent coordinate indices (no fitted basis); compute `log_p(q)` in float64 after projection to `r_max`; clamp norm-squared denominators below `eps=1e-12`; set the signed 2D determinant to zero if either projected tangent norm is below `eps`. Score only within the already-frozen candidate set. Use explicit coefficient `lambda` fixed before runs, `tau=1e-12` absolute tie tolerance, baseline-index fallback for ties and all non-finite values. Boundary cases (zero anchor separation, boundary projection, non-finite inputs) therefore produce zero signed preference and baseline selection.

### 3. Hyperbolic two-point Fréchet interpolation as a candidate-code score (candidate)

Construct an anchor from two frozen, observed context vectors by geodesic interpolation, then choose the code whose geodesic distance to that anchor is smallest among a fixed shortlist. The mechanism differs from assigning a lone item by its codeword radius: it uses a context-dependent point formed by a nontrivial manifold operation. **Pre-rate:** on the frozen sequence stream, fraction of contexts where the interpolated anchor changes the baseline winner; also report the fraction where the two context points are separated by more than `eps` (actual operation-active rate).

**Finite-precision declaration required:** specify the two context positions in advance; project points into radius `r_max`; compute `a ⊕_t b` via `exp_a(t log_a(b))` with fixed `t=1/2`, float64, `eps=1e-12` clamps for norms and inverse-function domains. If point distance is at most `eps`, anchor is exactly `a`; if any value is non-finite, revert to baseline assignment. Compare squared geodesic distances with `tau=1e-12`; ties preserve baseline then lowest code index. No trainable anchor or extra token is introduced.

### 4. Hyperbolic boundary-crossing robustness choice (candidate)

Select the code with the best worst-case score under a fixed, tiny geodesic perturbation in a predeclared direction derived from the frozen adjacent-item difference. This puts a hyperbolic operation in discrete code selection and tests whether assignments are locally stable, rather than adding a post-hoc tie-break. **Pre-rate:** fraction of frozen inputs for which the perturbation changes the winning code; additionally report fraction whose unperturbed top-two margin is at most the maximum perturbation-induced score change. Both are computable from the frozen artifacts.

**Finite-precision declaration required:** perturbation magnitude `delta` and direction rule fixed before evaluation; use `exp_x(delta*v)` in float64 after radius projection. Normalize `v` only when its norm exceeds `eps=1e-12`; otherwise perturbation is zero. Evaluate baseline and perturbed candidates on the same frozen shortlist; clamp all inputs to `r_max`; use absolute `tau=1e-12`, preserve baseline winner on ties/non-finite perturbation. The report must give the active perturbation rate before this idea can be promoted.

## Gate before any experiment

1. Freeze the exact operator/domain/epsilon/clamp/tolerance/boundary specification (the sketches above still need formula-level review and unit tests).
2. Replay only frozen artifacts and publish occurrence/change rates first. Reject a mechanism whose active condition is too rare or whose assignment stream is unchanged; do not add a rare-case tie-break.
3. Keep all frozen protocol elements unchanged. A candidate is successful only if one mechanism strictly exceeds all three thresholds: Hard-L1 ER `16.38953942232631`, Stage 3 Recall@10 `0.06406796775709883`, and NDCG@10 `0.03530651701940677`.

## Search and verification note

The citation identifiers above resolve to arXiv records with public manuscripts. This screening did not establish a DOI-backed primary paper whose method exactly performs hyperbolic geometry inside discrete three-level SID code selection. Therefore “not occupied” means only “not occupied by the four inspected works”; it is not an exhaustive prior-art or patentability opinion. Do not upgrade any candidate to an implementation plan without the prescribed occurrence-rate audit and finite-precision review.
