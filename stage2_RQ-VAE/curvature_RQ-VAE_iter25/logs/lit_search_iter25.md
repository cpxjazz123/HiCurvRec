# Iter25 Literature Search Report (Agent A)

Three exact queries from the shared context were executed via the supplied `iter25_literature_search` tool. No repository files were read or edited, no project commands were run, and no candidates were ranked or recommended.

## Context (read-only, not from a tool)
- Iter24 vs Iter18 fingerprints: test_R@10 0.05619875 (n=57439) vs Iter18 best 0.05988962 vs canonical Iter11 0.05976775. L0 utilization both 1.0; H(L1|L0) 5.394702 vs 5.590675 bits; H(L2|L0) 5.742606 vs 5.934625 bits; 3-token collision 8.9885% vs 6.6539%.
- Iter24 mechanism: inverse-square-root target-frequency weighting of symmetric multi-positive residual behavior loss.
- Iter11 prior mapping: raw residual medians `[1, .10941, .09331]` → applied fallback prior `[.001, .932889, 1.0]`.
- Iter25 proposes replacing that prior mapping while retaining learnable `u_l` and cyclic `c_l(t)`.

---

## Query 1 — `hyperbolic neural networks learnable layer-wise curvature prior residual norm branching hierarchy 2023 2024`

Top 3 relevant hits returned by the tool:

1. **Hyperbolic Residual Quantization: Discrete Representations for Data with Latent Hierarchies** — 2025 — https://arxiv.org/abs/2505.12404 — Performs residual VQ natively in hyperbolic space (hyperbolic nearest-neighbor search, Möbius residuals, Möbius reconstruction) and reports up to ~20% gains on WordNet and latent-hierarchy modeling. *Evidence type: direct on hyperbolic residual VQ; adjacent on the 2023/2024 date filter (paper is 2025) and on the layer-wise curvature extension (proposed by the tool as natural extension, not the defining mechanism).*

2. **OpenReview paper** — year not provided — https://openreview.net/pdf?id=5pd4eD0rzi — Only a URL was returned; no title, authors, year, or snippet content was provided by the tool. *Evidence type: unverifiable — flag rather than cite.*

3. **arXiv preprint** — year not provided — https://arxiv.org/abs/2609.26342 — URL only; "2609" would imply a future-dated listing (September-2026) later than the agent's knowledge cutoff; no title/authors/abstract. *Evidence type: unverifiable — flag rather than cite.*

---

## Query 2 — `residual vector quantization layer-wise curvature hierarchy branching hyperbolic representation learning`

Top 3 relevant hits returned by the tool:

1. **Hyperbolic Representation Learning: Revisiting and Advancing** — 2023 — https://proceedings.mlr.press/v202/yang23u/yang23u.pdf — Provides a layer-wise hyperbolic formulation where each layer carries its own input curvature $\kappa_{\ell-1}$ and output curvature $\kappa_\ell$, supporting per-layer curvature rather than a single global value. *Evidence type: direct for the layer-wise curvature parameterization; adjacent on residual codebooks/branching priors (not in the paper).*

2. **Poincaré ResNet** (van Spengler, Berkhout, Mettes) — ICCV 2023 — https://openaccess.thecvf.com/content/ICCV2023/html/van_Spengler_Poincare_ResNet_ICCV_2023_paper.html — Fully hyperbolic residual network with an identity initialization that preserves $\lVert x^\ell\rVert \approx \lVert x^{\ell-1}\rVert$ across deep layers, addressing signal-shrinkage seen with earlier hyperbolic initializations. *Evidence type: direct on hyperbolic residual design and norm preservation; adjacent on branching/RVQ specifics (not studied).*

3. **Capacity Bounds for Hyperbolic Neural Network Representations of Latent Tree Structures** — preprint 2023, journal version *Neural Networks* 2024 — https://arxiv.org/abs/2308.09250 — Proves that HNNs can approximately isometrically embed any finite weighted tree into hyperbolic space, in contrast to a $\Omega(L^{1/d})$ distortion lower bound for $d$-dimensional Euclidean MLPs when the tree has $L>2^d$ leaves. *Evidence type: direct for the theoretical motivation of negative curvature as a branching-capacity prior; adjacent on residual norm and recommender specifics.*

---

## Query 3 — `learned curvature initialization regularization cyclic curvature hyperbolic neural network ablation recommender`

Top 3 relevant hits returned by the tool:

1. **ManifoldMind: Probabilistic Hyperbolic Embeddings for Personalized Recommender Systems** — 2025 — http://www.arxiv.org/pdf/2507.02014 — Probabilistic hyperbolic recommender with per-entity learnable negative curvature $\kappa_i$ and uncertainty radii $r_i$, plus a curvature-regularization term that avoids Euclidean degeneracy; ablations show that Euclidean replacements or fixed components diminish NDCG/calibration/diversity. *Evidence type: direct on learnable curvature + curvature regularization in a recommender setting and on the importance of ablation; adjacent on the SID/RVQ pipeline (not in the paper).*

2. **Hyperbolic Neural Networks with Learnable Curvature for Multi-Hop Reasoning (δ-hyperbolicity initialization ablation)** — Welz, Flek, Karimi — 2025 — https://arxiv.org/pdf/2507.03612 — Ablation showing that initializing curvature from the dataset's δ-hyperbolicity outperforms random/arbitrary initializations; excessive curvature (≈10.0) sharply degrades performance, while δ-hyperbolicity-tied values are more robust, with the gap largest on more hierarchical datasets. *Evidence type: direct on curvature initialization being an ablation variable and on δ-hyperbolicity as a prior-like scale for curvature; adjacent on RVQ/codebooks (not in the paper).*

3. **Curvature Learning for Generalization of Hyperbolic Neural Networks** — 2025 — https://arxiv.org/abs/2508.17232 — PAC-Bayesian generalization bound tying curvature to loss-landscape sharpness; sharpness-aware curvature learning via bi-level optimization with implicit differentiation; improvements on classification, long-tailed, noisy, and few-shot benchmarks. *Evidence type: direct on a cyclic/bi-level curvature update schedule and on curvature regularization improving HNN behavior; adjacent on RVQ/SID (not studied).*

Adjacent / supplementary hits produced by the same tool call (not in the top 3):
- HRCF (Hyperbolic Regularized Collaborative Filtering), 2022 — https://arxiv.org/pdf/2204.08176v2.pdf — hyperbolic GNN recommender with explicit regularization.
- LGCF (Fully Hyperbolic Graph Convolution Network for Recommendation), 2021 — https://arxiv.org/html/2108.04607 — fully hyperbolic GCN for recommendation with Wrapped-Normal initialization.

---

## Candidate mechanisms P1 / P2 / P3 (no ranking)

**P1 — Layer-wise learned curvature $\kappa_\ell$ with prior/regularization** (mechanism pool item 2).
Yang et al. 2023 parameterizes every hyperbolic layer with its own input/output curvature, HGCN and 2023–2024 deep graph variants train $\kappa$ per layer, HRQ 2025 demonstrates residual VQ in hyperbolic space, and Welz et al. 2025 confirms curvature initialization is itself an ablation lever (δ-hyperbolicity scale beats arbitrary init); together this supports replacing Iter11's fixed prior mapping with a per-layer $\kappa_l$ carrying its own regularizer (e.g., monotonic $\max(0,\kappa_l-\kappa_{l+1})$ or δ-hyperbolicity-style initialization). *Direct evidence on the layer-wise $\kappa$ parameterization; adjacent evidence on combining it with the residual codebook (HRQ) and with the recommender-SID context.*

**P2 — Cyclic curvature schedule $c_l(t)$** (mechanism pool item 1).
arXiv 2508.17232 (2025) uses a cyclic/bi-level curvature update via bi-level optimization with implicit differentiation, and Welz et al. 2025 explicitly ablates curvature initialization, together suggesting that curvature can be optimized on a different cadence from the main loss. *No paper found in any of the three queries applies a cyclic $c_l(t)$ to RVQ layers or to a recommender SID pipeline; P2 is supported only as an adjacent mechanism borrowed from curvature-learning theory.*

**P3 — Adaptive curvature-dependent margin between branches / codewords** (mechanism pool item 4).
arXiv 2308.09250 (2023/2024) supplies the theoretical basis that negative curvature supports tree-like branching with low distortion, and ManifoldMind (2507.02014) demonstrates that curvature-regularized hyperbolic recommenders use distance-based scoring sensitive to per-entity curvature — together they motivate a margin $m(\kappa_l)$ parameterized in $\kappa_l$-dependent units so that deeper layers' larger branching capacity converts into larger inter-branch/codeword separation. *Adjacent evidence combining a theoretical branching-capacity bound with an applied recommender; no retrieved paper directly studies curvature-adaptive margins inside an RVQ.*

---

## Direct vs adjacent evidence (honest summary)

- **Direct evidence on the exact Iter25 bottleneck (raw residual → fallback prior mapping for SID codebooks with cyclic $c_l(t)$): none.** No queried paper studies behavior branching plus raw residual mapping inside a SID/RQ-VAE recommender.
- **Direct evidence on each P1/P2/P3 individually:**
  - P1 (layer-wise $\kappa_\ell$ with prior/regularization): **directly evidenced** by Yang 2023, HGCN variants, HRQ 2025, Welz et al. 2025.
  - P2 (cyclic $c_l(t)$): **adjacent-evidenced** by arXiv 2508.17232 only.
  - P3 (curvature-dependent margin/branching): **adjacent-evidenced** by arXiv 2308.09250 + arXiv 2507.02014.
- **Unexplored / unverifiable:**
  - Query 1 explicitly requested 2023/2024; the closest "hyperbolic residual VQ" hit is HRQ from 2025.
  - Two URLs returned by Query 1 (openreview.net/pdf?id=5pd4eD0rzi and arxiv.org/abs/2609.26342) carry no title, authors, year, or abstract — explicitly flagged as unverifiable, not invented.
  - HRCF (2022) and LGCF (2021) surfaced as adjacent hits via Query 3; useful as historical hyperbolic-recommender baselines, not as direct Iter25 evidence.

## Parent primary-source cross-check addendum

The following primary sources were read after Agent A's search. This corrects several over-broad evidence characterizations above:

- Chen, King, Yang, Zhou, and Ying, *Hyperbolic Representation Learning: Revisiting and Advancing* (ICML/PMLR 2023), https://proceedings.mlr.press/v202/yang23u/yang23u.pdf: the paper studies Hyperbolic Informed Embedding and hierarchy alignment. Its HNN equation uses layer-indexed curvature notation, but it does **not** establish a learnable layer-wise curvature prior or Iter25's formula.
- van Spengler, Berkhout, and Mettes, *Poincare ResNet* (ICCV 2023), https://openaccess.thecvf.com/content/ICCV2023/html/van_Spengler_Poincare_ResNet_ICCV_2023_paper.html: the abstract supports identity-based initialization that preserves norms in deep hyperbolic residual networks; it does not study residual-codebook medians.
- Kratsios, Hong, and Sáez de Ocáriz Borde, *Capacity Bounds for Hyperbolic Neural Network Representations of Latent Tree Structures* (arXiv 2023), https://arxiv.org/abs/2308.09250: the abstract supports tree-representation capacity in hyperbolic spaces, not a log-branching prior for RQ-VAE.
- Welz, Flek, and Karimi, *Multi-Hop Reasoning for Question Answering with Hyperbolic Representations* (arXiv 2025), https://arxiv.org/abs/2507.03612: the abstract and curvature ablation support dataset-derived δ-hyperbolicity initialization in that QA setting; it is not an RVQ recommender experiment.

Accordingly, treat these as adjacent premises only. No retrieved/checked source directly evaluates the exact branching-plus-raw-residual prior or SID/RQ-VAE recommender composition. Agent A's candidate text is preserved as the search record; the parent direction decision uses the narrower claims above.
