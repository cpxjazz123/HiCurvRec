# Prior-art map: verified paper ↔ old repository mechanism ↔ proposed operator (issue #111)

The three-way map issue #111 requires: for every operator a proposal would introduce, which
**verified paper** already covers it, which **old repository mechanism** is nearest to it, and what
the **exact mathematical difference** is. The paper side comes from
`subspace_literature_matrix.json`; the mechanism side from
`results/hyperbolic_research_runs/prior_art_registry.json` (the #110 registry);
the operator side from `subspace_mechanism_proposals.json`.

A row whose paper or mechanism column says "none verified" is a *negative* result of this session's
searches, not a claim that no such work exists. The arXiv API was unreachable (HTTP 429) for all ten
queries, so an arXiv-only paper could have been missed; `subspace_literature_matrix.json` records
that limit explicitly.

## 1. The operators of the three proposals

### P1 `hyp_product_l1_quantiser` — product-geodesic level-0 assignment with factor-masked supervision

| | |
|:---|:---|
| **operator** | `j* = argmin_j [ d_c²(exp₀(s_s·z[:16]), exp₀(a_j)) + d_c²(exp₀(s_b·z[16:]), exp₀(b_j)) ]`, with `d_c` the Poincaré geodesic of curvature −1, and each factor's codewords receiving only its own role's gradient |
| **verified paper overlap** | **Gu et al. 2019** (`https://openalex.org/W2923452197`, ICLR 2019): a product of constant-curvature factors as a representation. This is the *substrate*, and #111 states it is not novelty. **HVQ-VAE 2025** (`10.1016/j.cviu.2025.104392`): intrinsic hyperbolic vector quantisation. **SAHL 2026** (`10.48550/arxiv.2610.25093`, code read): subspace-structured SIDs — but with PCA `whiten=True` blocks, K-means and a balanced residual chain over *sentence* embeddings. **Weighted mixed-curvature product manifold 2023** (`10.48550/arxiv.2307.04514`): per-component weighting of a product manifold. |
| **old mechanism nearest** | the #109 c-family codebook-geometry arms (`directed_codeword_margin_geometry`, `directed_separated_codeword_geometry`, `directed_pair_separation_codebook_geometry`) and `directed_assignment_prefix_preservation`; all are Euclidean margin/repulsion/assignment objectives. The R8 hyperbolic arms are `r8_07` (gyro-translation), `r8_08` (Fréchet barycentres), `r8_09` (radial hierarchy), `r8_10` (graph GW). |
| **exact difference** | none of the above makes the **assignment** the argmin of a *sum of two factor geodesics*: Gu et al. has no codebook; HVQ-VAE has one hyperbolic codebook and a Euclidean-in-spirit assignment; SAHL's subspaces are orthogonal PCA blocks of a *whitened Euclidean* space assigned by K-means; the repo arms all assign by a single Euclidean metric. The factor-masked supervision (content gradient on one half, transition gradient on the other) has no counterpart in any of them. |
| **verdict** | `novel` at the mechanism level; the *subspace* and *product-manifold* ideas are prior art and must be cited, not claimed |

### P2 `hyp_transported_residual` — the residual lives in the level-0 codeword's tangent frame

| | |
|:---|:---|
| **operator** | `r = PT_{x→a}(s·z − log₀(a))` and `ẑ = log₀(a) + PT⁻¹_{a→0}(r̂)`: parallel transport along the geodesic from the item's ball point to the assigned codeword, before levels 2 and 3 quantise the residual |
| **verified paper overlap** | **none verified.** The store holds no paper on Riemannian/geodesic vector quantisation (the query `Riemannian vector quantization manifold codebook geodesic` returned MIMO codebook design and Riemannian optimisation over covariance matrices, nothing on discrete tokens) and none on transporting a quantisation residual. Parallel transport itself is textbook hyperbolic geometry; the *application* to a residual chain is what is at stake. |
| **old mechanism nearest** | the r8 residual/transport family (`TIGER_r8_*`, and #107's `directed_residual_tangent_transport`, `directed_behavior_residual_tangent_transport`) plus `behavior_residual_geometry_transport` in the #109 cohort. All define the residual in the **ambient** space: `r = z − code`. |
| **exact difference** | in a Euclidean geometry parallel transport *is the identity*, so every old residual-transport arm is exactly the `c → 0` limit of this operator. The difference is not a coefficient: `PT_{x→a}` scales and rotates the residual by a factor that depends on the ball metric at both endpoints (`λ_x/λ_y` times a gyration), so the level-2 codebook learns a frame that is *relative to the level-0 codeword*. The declared test `p2_level0_bit_identity` fixes the boundary: the L1 tokens must not change. |
| **verdict** | `novel` as a quantiser mechanism; the underlying operator is textbook, so the claim must be about the *residual frame*, never about parallel transport itself |

### P3 `hyp_factor_conflict_projection` — the conflict measured in the codeword's metric

| | |
|:---|:---|
| **operator** | project `g_beh` against `g_rec` when `⟨g_beh, g_rec⟩_g < 0`, with `g_x = (2/(1−c‖x‖²))² I` evaluated at each codeword |
| **verified paper overlap** | none verified for this construction. |
| **old mechanism nearest** | **`r4_08`** — the repository's Euclidean PCGrad (`r4_project_conflict`, rule `project_l1_against_base`), which every #109 "tangent rotation" arm inherits. This is a *direct structural overlap*: both project one gradient against another. |
| **exact difference** | only the metric. `r4_08` uses the ambient Euclidean inner product; this operator uses a position-dependent one whose scale diverges at the ball boundary. That difference is real but narrow: it changes *which* coordinates are projected, not the family of operation. |
| **verdict** | **`overlap`** — recorded as a diagnostic of whether the ball metric changes the conflict structure, **not** promoted as a candidate mechanism. The decision memo ranks it last for exactly this reason. |

## 2. The operators the proposals deliberately do *not* use, and who owns them

| operator | owner | why it is excluded here |
|:---|:---|:---|
| Möbius/gyro translation of points | `r8_07` (ER ≈ 15.0360) | failed at the mathematical level; #110 forbids re-running it |
| Fréchet barycentre of successors | `r8_08` (ER ≈ 6.7968) | same |
| radial / per-layer hierarchy reweighting | `r8_09` (ER ≈ 15.8633) | same |
| graph-code Gromov–Wasserstein | `r8_10` (ER ≈ 1.0007) | same |
| Busemann / horosphere level set | **#110's current R10 arm** (`results/hyperbolic_research_runs/proposals/r10_horosphere_busemann.json`) | a second arm on the same operator would duplicate #110's mechanism registry |
| sparse learned gate over product spaces | **Switch Spaces 2021** (`10.48550/arxiv.2102.08688`) | verified prior art; a gate would be a re-implementation |
| PCA block + balanced residual quantisation + occupancy cap | **SAHL 2026** (`10.48550/arxiv.2610.25093`) | verified prior art, and Euclidean; #111 explicitly forbids it |
| hyperbolic codebook with standard commitment losses | **HVQ-VAE 2025** (`10.1016/j.cviu.2025.104392`) | verified prior art for "put the codebook in hyperbolic space" |
| contrastive/transition term on the codebook | **CoST 2024** (`10.1145/3640457.3688178`) and the repo's `r8_01` | verified prior art for the idea |
| adaptive SID branching | **Not All Branches Are Equal 2026** (`10.1145/3773078.3831756`) | the branch structure is already a first-class optimised object |

## 3. What the mapping implies for #111's mandatory questions

* **Q1 (is factorisation a better inductive bias, or capacity loss?)** The verified prior art
  answers the *capacity* half: SAHL obtains subspace-structured SIDs with K = 16–64 digits at 256
  codes per digit, so subspace structure is compatible with large capacity — but that says nothing
  about our frozen 3 tokens. In this repository the capacity question is sharper than in the
  literature: 256 joint level-0 codewords, 3 tokens, 32 dimensions. P1 keeps 256 joint codewords
  (it is *not* a 256×256 product quantiser), and P2 keeps every level at 256; neither inflates
  capacity. Whether the *effective* partition improves is exactly what the falsifiers measure.
* **Q2 (are the roles identifiable or vacuous?)** The literature's factor roles are either
  **learned jointly** (Gu et al.), **weighted per component** (2023), **gated per input**
  (Switch Spaces) or **PCA blocks** (SAHL — a variance ordering, not a role). None of them
  *declares* a role and enforces it. P1's factor-masked supervision is the only construction here
  that makes the role structural, and it carries an explicit inertness test
  (`p1_joint_assignment_not_inert`, and the `joint_vs_semantic_only_agreement` measurement).
* **Q3 (do both objectives reach the real codebook and the export?)** Yes by construction in P1
  (the joint assignment is the export's level 0) and in P2 (levels 2–3 index the transported
  residual). P3 reaches the codebook but not the export directly — another reason it is a
  diagnostic, not a candidate.
* **Q4 (can subspace-specific preservation prevent ER-up/downstream-down?)** This is the open
  question and no verified paper answers it for a 3-token SID. It is a prediction, and the
  failure analysis (`subspace_failure_analysis.md`) supplies the guardrails that would make a
  "yes" believable: T1–T4 (non-control L0 fingerprint, G_H1 at matched occupancy, recorded ER
  shares, no concentration).
* **Q5 (does the factorisation change the inference-relevant SID, or only train an auxiliary
  predictor?)** P1 and P2 change the hard SID; P3 does not. This is the discriminator the memo
  uses to rank P3 out.
* **Q6 (novelty vs the literature)** See the table above: the subspace *concept*, the product
  manifold, gating, hyperbolic codebooks and contrastive SID training are all prior art. What is
  not: a **product-geodesic assignment with factor-masked roles** (P1) and a **codeword-frame
  residual chain** (P2).
* **Q7 (is a coherent quantiser design possible under 3×256×32?)** P1 and P2 are each internally
  coherent and capacity-preserving. The honest caveat: whether a *subspace* design can be
  coherent under 3 tokens specifically is unproven, because every verified subspace-SID paper uses
  16–64 tokens. P1 sidesteps this by putting the factorisation inside one 256-way level-0 codebook
  rather than across tokens — and that is a design decision #111 should record, because it is the
  only way this family stays comparable with the frozen protocol.
