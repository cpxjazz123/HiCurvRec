# Iter26 Literature Search Report

This iteration is a one-factor replication of Iter16's closed-form fixed curvature
mechanism with the residual input corrected from the Iter1 calibration
`[0.001, 0.932889, 1.0]` to the raw residual medians `[1.0, 0.10941, 0.09331]`. It is
not a candidate-driven iteration: the dominant bottleneck from the Iter25 bridge is
that the previous attempt encoded the wrong residual vector. No new search is needed;
the literature evidence gathered for Iter16 and Iter25 already covers the
surrounding claims.

## Carried-over evidence (cross-checked after Iter25)

The following primary sources were re-read and confirmed relevant to the bounded
prior-mapping family of mechanisms (P1 in the Iter25 direction decision):

- Chen, King, Yang, Zhou, and Ying, *Hyperbolic Representation Learning: Revisiting
  and Advancing*, ICML/PMLR 2023,
  https://proceedings.mlr.press/v202/yang23u/yang23u.pdf — per-layer hyperbolic
  layer equations with input/output curvature (κ_ℓ-1, κ_ℓ); provides the
  layer-wise parameterization that Iter16's closed-form `c_l` approximates.
- van Spengler, Berkhout, and Mettes, *Poincaré ResNet*, ICCV 2023,
  https://openaccess.thecvf.com/content/ICCV2023/html/van_Spengler_Poincare_ResNet_ICCV_2023_paper.html
  — identity-based initialization that preserves per-layer residual norms in
  hyperbolic residual networks; the closest literature support for fixing `c_l`
  from measured residual medians.
- Kratsios, Hong, and Sáez de Ocáriz Borde, *Capacity Bounds for Hyperbolic Neural
  Network Representations of Latent Tree Structures*, arXiv 2023 / *Neural
  Networks* 2024, https://arxiv.org/abs/2308.09250 — capacity bound for embedding
  weighted trees in hyperbolic space; the theoretical underpinning for the
  branching-capacity prior used as one of the two `c_l` inputs.
- Welz, Flek, and Karimi, *Multi-Hop Reasoning for Question Answering with
  Hyperbolic Representations*, arXiv 2025, https://arxiv.org/abs/2507.03612 —
  curvature ablation showing that data-derived δ-hyperbolicity initialization
  outperforms random initialization; closest prior for the "fixed c initialized
  from data" design choice Iter16 + Iter26 share.

## Why no new search

Iter26 changes exactly one input to Iter16's existing closed-form mapping; the
mechanism, the formula, the constants (`c_base=0.5`, `alpha=0.2`), the cyclic
disable, the learnable-scale disable, and the curvature-regularization disable
are all preserved from Iter16. The Agent A/B/C pipeline is not triggered because
there is no candidate selection: the experiment is "did Iter16 use the wrong
residual?". Literature is reused verbatim from the Iter25 record; no new
candidate needs to be ranked.

## Adjacent evidence (carried over from Iter25 search)

- Hyperbolic Residual Quantization (HRQ), arXiv 2025, https://arxiv.org/abs/2505.12404
  — residual VQ performed natively in hyperbolic space; supports the
  residual-based closed-form idea but is a separate contribution.
- ManifoldMind, arXiv 2025, https://arxiv.org/abs/2507.02014 — probabilistic
  hyperbolic recommender with per-entity learnable curvature; supports the
  layer-curvature-prior line but uses learnable κ, not fixed closed-form.

## Unverifiable (carried over)

- Two URL-only hits from the Iter25 search (`openreview.net/pdf?id=5pd4eD0rzi`,
  `arxiv.org/abs/2609.26342`) remain unverifiable and are not cited as evidence.

## Conclusion

Iter26 is an experiment, not a candidate selection. The Iter25 direction
decision's "no valid recommendation → keep the pre-registered Iter16 closed-form
with raw residual" line still applies. No new ranking is required. The unique
contribution of this iteration is the corrected input vector and the four-point
`c_l` invariance contract.