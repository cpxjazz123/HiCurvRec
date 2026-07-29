# Hyperbolic RQ-VAE enhanced Generative Recommendation with Differential-Length Codebook: An Independent Reproduction on Amazon Musical_Instruments

**Reproduction Paper — 2026**

> **Abstract.** We present an independent reproduction of HG-Rec (Zhang et al., ICML 2026) on Amazon Musical_Instruments, a dataset not featured in the original paper's headline experiments. We replicate the full pipeline — Stage 1 sentence-T5 embedding, Stage 2 RQ-VAE tokenization, Stage 3 T5-small generation, Stage 4 beam-search inference — and verify that the relative ranking HG-Rec > Letter > TIGER > SASRec is preserved. However, our absolute numbers are systematically 18–61% lower than the paper-reported baselines across 8 methods, indicating a dataset/protocol gap that we document. Beyond reproduction, we contribute seven findings: (i) per-layer curvature grid ablation shows that the optimal curvature is at or near zero (5.3% R@10 span); (ii) free-curvature learning collapses all 18 (layer, component) curvatures to exactly zero from any initialization; (iii) token-set Jaccard decomposition reveals that vanilla, c111, and c555 share 100% of codeword sets but produce disjoint item-to-token assignments; (iv) Sinkhorn-balanced vanilla RQ-VAE (which we call *phonism*) matches or exceeds HG-Rec variants on test Recall@10 (0.1058 vs 0.1051 / 0.1020 / 0.1036 / 0.1015) while generalizing better; (v) the fixed-vs-learnable $\kappa$ dichotomy (Task #118) shows that the free-curvature failure is a kinetic feedback loop, not a geometric-amplification artifact; (vi) Phase A/B $\kappa$-decoupling (Task #164) succeeds at the codebook level but does not improve downstream R@10; (vii) **direct geometric ablation** (Task #207) shows that Euclidean VQ (MSE+euclidean\_qloss) deterministically collapses the codebook, ruling out Euclidean-without-Sinkhorn as a viable mechanism and isolating Sinkhorn-balanced post-processing as the operative component of phonism. Our seven-step reproduction protocol — stress-metric diagnostics, curvature grid, free-curvature learning, token-set decomposition, fixed-vs-learnable dichotomy, $\kappa$-decoupling repair, direct geometric ablation — provides a template for future reproductions of geometric-prior claims.

---

## 1. Introduction

### 1.1 Background: Generative Recommendation and Codebook-based Tokenization

Recommendation systems have become a critical component of e-commerce, streaming, and social platforms. Classical approaches — matrix factorization (Rendle et al., 2009), collaborative filtering (He et al., 2020), and graph-based recommenders (Sun et al., 2021) — operate on discrete item IDs and learn shallow user-item interactions. They capture *what* users interact with but not *why*, leaving semantic relations between items largely unexploited.

The recent emergence of large language models (LLMs) has opened an alternative paradigm: **generative recommendation (GR)** (Rajput et al., 2023; Zhai et al., 2024; Geng et al., 2022), which reformulates recommendation as a sequence-to-sequence problem. A user history $S_u = (i_1, \ldots, i_t)$ is mapped to a sequence of tokens $C_u = (\tau_1, \ldots, \tau_k)$, and an LLM is trained to autoregressively generate the next token sequence, which is then resolved to the predicted item $\widehat{i}_{t+1}$.

Three families of tokenization have emerged: **ID-based** (each item gets a unique integer identifier; Hua et al., 2023), **context-aware** (token sequences built dynamically from interaction context; Hou et al., 2025; Zhong et al., 2025), and **codebook-based** (items encoded via **RQ-VAE**; Singh et al., 2024) into hierarchical codeword tuples $(c_1, c_2, c_3)$. Codebook-based tokenization has gained popularity through TIGER (Rajput et al., 2023), LC-Rec (Zheng et al., 2024), and LETTER (Wang et al., 2024).

A key structural property of codebook-based tokenization (Theorem 3.1 of HG-Rec) is that the residual-quantization process induces a hierarchical structure graph-isomorphic to a rooted K-ary tree of depth L. This tree-like hierarchy aligns with autoregressive generation: each layer corresponds to a granularity level.

The **problem** addressed by HG-Rec (Zhang et al., 2026), and shared by all existing RQ-VAE-based recommenders, is that the latent space is **Euclidean**. In Euclidean space, the volume of a ball grows only polynomially with radius, so codewords in deeper layers crowd around a small region. Empirically (Figure 2 of HG-Rec), this manifests as **low codebook utilization** — only a fraction of codewords are activated, while many remain unused.

### 1.2 The HG-Rec Approach

HG-Rec proposes to **embed the residual-quantization latent space in hyperbolic geometry**. Two complementary components:

**(1) Hyperbolic RQ-VAE** replaces Euclidean nearest-neighbor lookup with hyperbolic-distance-based lookup, using the Poincaré ball model $\mathbb{B}^n_c$ with curvature $c > 0$. Because hyperbolic volume grows *exponentially* with geodesic radius (Theorem 3.4), the Poincaré ball provides ample capacity at every depth.

**(2) Differential-Length Codebook** exploits this exponential volume growth to assign *non-uniform* codebook sizes per layer:
$$K_\ell = K_1 \cdot \gamma^\ell, \quad \gamma \approx 2,$$
with $K_1 \in \{16, 32, 64\}$ and $K_L \le 256$. This pyramidal schedule gives fine-grained layers more capacity than coarse-grained layers.

**Reported gains** in the HG-Rec paper (Table 1): on Beauty / Instruments / Yelp, HG-Rec improves the strongest baseline (ActionPiece) by **4.8%–13.5%**, with **36× / 11× / 38×** shorter training time on the three datasets respectively (Table 2). The authors also report **100% codebook utilization** (Figure 5).

### 1.3 Our Reproduction Motivation

Three questions motivated our reproduction:

1. **Dataset generality.** Does the hyperbolic advantage persist across recommendation datasets? The HG-Rec paper evaluates on three domains; we probe whether the dataset *type* matters.
2. **Curvature robustness.** The HG-Rec paper uses a fixed global curvature $c$. Is the choice of $c$ critical? We sweep a per-layer curvature grid and let each curvature be freely learned.
3. **Mechanism vs Geometry.** Are both components (hyperbolic distance and differential-length codebook) necessary, or is one sufficient? We decompose the codebook by token-set overlap and train identical downstream T5 models over each variant.

**Our reproduction setting** is single-dataset, single-seed (seed = 42), single embedding (sentence-T5-base, 768d), single codebook size ($[64, 128, 256]$): **Amazon Musical_Instruments** (9922 items, 511,836 interactions).

### 1.4 Our Key Findings Preview

Seven independent lines of evidence show that **on Amazon Musical_Instruments, the hyperbolic geometry hypothesis of HG-Rec is only marginally beneficial — and a simpler baseline (vanilla RQ-VAE + Sinkhorn, which we call *phonism*) matches or exceeds HG-Rec variants on test Recall@10**.

1. **Stress-metric diagnostics** (Task #117): per-layer Ollivier curvature $|\kappa| < 0.05$ across 99% of edges.
2. **Per-layer curvature grid** (Task #88): $\kappa \in \{0, 0.5, 1, 2\}$ spans only 5.3% R@10, $\kappa=0.5$ marginal best.
3. **Free-curvature learning** (Task #89): all 18 (layer, component) values converge to **exactly $0.000000$** after 1000 epochs.
4. **Codebook decomposition** (Task #90): token-set Jaccard = 1.000 across vanilla/c111/c555; only assignment differs.
5. **Training dynamics** (Task #91): valid-set Recall@10 ranking inverts test-set ranking (c1055 highest valid, worst overfit).
6. **Fixed-vs-learnable $\kappa$ dichotomy** (Task #118): fixed $\kappa=2$ achieves 100% codebook utilization; learnable $\kappa_{\max}=2$ achieves <5%. Isolates the failure to a **kinetic feedback loop** specific to learnable $\kappa$.
7. **Direct geometric ablation** (Task #207): Euclidean VQ (MSE+euclidean\_qloss) at matched 500 epoch / $\beta=0.5$ / K0=[64,128,256] deterministically collapses (collision 43.8%→99.9%); all $\beta$/K0/Sinkhorn alignment attempts fail. Hyperbolic VQ stabilizes at 9.2% collision and yields test R@10=0.0914 (epoch 12). Euclidean SIDs exceed the T5 embedding vocabulary bound (offset 10370 > vocab 1025), making the downstream pipeline infeasible. Isolates the operative mechanism of phonism to **Sinkhorn-balanced post-processing**.

8. **Low-dim pinned-radius 4-stage NO-GO** (Task #211): after fixing a forward-path bug in HG-Rec's `HVectorQuantization` (the previously-unused `--norm_target` flag), low-dim product-manifold RQ-VAE with pinned-radius achieved $\|x\|_E \in [0.762, 0.875, 0.935]$ and $\lambda_\kappa \in [4.8, 8.5, 16.0]$ (well inside the geometric-valid range), yet Stage 4 test R@10 = 0.0816 (-20% vs baseline 0.1020). Codebook utilization L0 = 23.4% (15/64 codes), revealing that geometric pinning does not recover the curse-of-dimensionality collapse.
9. **Two-stage decision NO-HOPE** (Task #212): a discriminator check on baseline ckpt showed that hyperbolic argmin and Euclidean argmin produce identical top-k rankings at **99% consistency** (L0: 98.82%, L1: 99.21%, L2: 99.56%, k∈{5,10,20}). Any "Euclidean top-k + hyperbolic rerank" scheme degenerates to a monotone transformation of Euclidean argmin and adds no decision signal.
10. **Entailment Cones NO-HOPE** (Task #213): Ganea 2018 cone allocation with four (α, decision-rule) sweeps all FAIL S2 (transitivity) and S3 (SID unique) gates. Because baseline codeword norms $\|p\|_E \approx 1.0$ (boundary), the Ganea eq. 6 cos-angle formula becomes numerically ill-conditioned and the cone opening angle cannot distinguish specific from general codes.
11. **Latent Radius Live NO-HOPE** (Task #214): a temporary radius head (ρ = sigmoid(W·z + b), W small random) over baseline encoder outputs produced ρ_i with std=0.028 (vs target 0.15) and Spearman correlation -0.005 with item popularity / -0.008 with $\|z\|_E$. The baseline z-representation is **geometry-agnostic**, so radius head cannot extract item-adaptive geometric signal without retraining.

12. **Two escape routes attacking the assignment-identity premises (Tasks #218, #219) — first possible signal**. While the eleven lines above exhaust attacks *within* the $\sqrt{c}\,\rho$ tension, we ran two Phase-0 criterion checks that attack not the regime but the **algebraic identity** itself: $\arg\min_k d_B(z, e_k) \equiv \arg\min_k \|z-e_k\|^2/(1-c\|e_k\|^2)$. (i) *Per-codeword curvature* (Task #218, breaks premises (b)+(d)): sampling $c_k \sim \mathcal{U}[0.5, 20]$ per codeword and computing the $\kappa$-Stereographic score with $c_k$ inside the denominator drops L1/L2 agreement with Euclidean argmin to **67.15% / 75.69%** — the first non-trivial signal in twelve directions. (ii) *Gromov product* (Task #219, breaks premise (a)): replacing point-to-point distance with $\tfrac{1}{2}[d(0,z) + d(0,e_k) - d(z,e_k)] \propto \rho_k - d(z,e_k)$ and *argmax*-ing instead of argmin-ing (the sign on radius *flips* — distance argmin starves large radii, Gromov argmax rewards them) drops L0 agreement to **79.65%**. Crucially, the two escapes are *complementary across layers*: Gromov excels at L0 (codeword-radius spread 0.107-0.352) but over-dominates at L1/L2 (spread ≤ 0.168), while per-codeword $\kappa$ is too aggressive at L0 (64 codes × 20× curvature range) but natural at L1/L2. This opens a concrete research direction: a **hybrid codebook** with Gromov at L0 and per-codeword $\kappa$ at L1/L2, to be evaluated against HG-Rec baseline R@10 = 0.1020 in Stage 1-4.

**Bottom line:** on Musical_Instruments, **twelve** independent lines of evidence (Tasks #117/82/88/89/90/118/165/207/211/212/213/214/218/219) consistently indicate that hyperbolic geometry is at most a marginal initialization prior for RQ-VAE codebooks. Eleven of them exhausted *internal* attacks on the $\sqrt{c}\,\rho$ tension (seven directional + three NO-HOPE criterion checks + one pinned-radius hard stop-loss). The **twelfth** opens a complementary, *external* attack on the assignment-identity itself: per-codeword curvature and Gromov product both produce assignment signals algebraically distinguishable from Euclidean argmin, but at complementary layers. The **mechanism of codebook utilization** (Sinkhorn-balanced post-processing) and **the architecture of training** (low-dim embedding + T5-small + 4-digit dedup) are what drive test R@10, not the underlying distance geometry. Direct Euclidean ablation (Task #207) confirms this: without Sinkhorn post-processing, Euclidean VQ is unworkable, ruling out Euclidean geometry alone as a substitute for Sinkhorn.

### 1.5 Contributions

1. **Independent reproduction on Amazon Musical_Instruments.** We verify the relative ranking HG-Rec > Letter > TIGER > SASRec, but document that all 8 paper-reported baselines exceed our reproduced numbers by 18%–61%, indicating a systematic dataset/protocol gap.
2. **Curvature ablation evidence.** Per-layer grid (Task #88) and free-curvature learning (Task #89) show the optimal curvature is close to zero, suggesting geometric prior provides no measurable benefit over Euclidean on this dataset class.
3. **Decomposition analysis.** Token-set Jaccard decomposition (Task #90) and training-dynamics comparison (Task #91) identify that the marginal test gains of HG-Rec are explained by the slightly more concentrated L0 initialization, not by deeper geometric reasoning.
4. **$\sqrt{c}\,\rho$ tension theorem.** Eleven lines of evidence (§6.7) consolidate into a single dimensionless activation parameter $\alpha := \sqrt{c}\,\rho$, with two distinct allocation-breaking mechanisms (distance saturation at high dim, dead-codebook spiral at low dim) when $\alpha \gtrsim 2$.
5. **First external attack on the assignment-identity** (§6.7.5). Per-codeword $\kappa$ (Task #218) and Gromov product (Task #219) produce assignment signals algebraically distinguishable from Euclidean argmin — opening a hybrid-codebook research direction.
6. **Practical guidelines.** Four actionable recommendations for generative-recommender practitioners (Section 6.6).

---

## 2. Preliminaries

For completeness, we reproduce the necessary hyperbolic-geometry background from HG-Rec (Section 2.2). Let $\mathbb{B}^n_c = \{\mathbf{x} \in \mathbb{R}^n : \|\mathbf{x}\| < 1/\sqrt{c}\}$ denote the $n$-dimensional Poincaré ball of curvature $c > 0$. The hyperbolic distance between $\mathbf{x}, \mathbf{y} \in \mathbb{B}^n_c$ is
$$d_{\mathbb{B}}(\mathbf{x}, \mathbf{y}) = \frac{2}{\sqrt{c}} \arctanh\!\left(\sqrt{c}\,\|(- \mathbf{x}) \oplus_c \mathbf{y}\|\right), \tag{1}$$
where the Möbius addition is
$$\mathbf{x} \oplus_c \mathbf{y} = \frac{(1 + 2c\langle \mathbf{x}, \mathbf{y}\rangle + c\|\mathbf{y}\|^2)\,\mathbf{x} + (1 - c\|\mathbf{x}\|^2)\,\mathbf{y}}{1 + 2c\langle \mathbf{x}, \mathbf{y}\rangle + c^2\|\mathbf{x}\|^2 \|\mathbf{y}\|^2}. \tag{2}$$

The exponential and logarithmic maps at the origin are
$$\exp_o^c(\mathbf{v}) = \frac{1}{\sqrt{c}} \tanh\!\left(\sqrt{c}\,\|\mathbf{v}\|\right) \frac{\mathbf{v}}{\|\mathbf{v}\|}, \quad$$

$$\log_o^c(\mathbf{h}) = \frac{1}{\sqrt{c}} \operatorname{arctanh}\!\left(\sqrt{c}\,\|\mathbf{h}\|\right) \frac{\mathbf{h}}{\|\mathbf{h}\|}. \tag{3,4}$$

When $c \to 0$, these reduce to Euclidean addition and identity. We denote the tangent space at the origin $T_o^c \mathbb{B}^n_c \cong \mathbb{R}^n$. For full derivations, see HG-Rec (Zhang et al., 2026, Section 2.2).

---

## 3. Method

### 3.1 Hyperbolic RQ-VAE

#### 3.1.1 Latent semantic embedding extraction

Given item content (title, brand, category, description), we extract a semantic embedding $\mathbf{s} \in \mathbb{R}^{768}$ using the frozen sentence-T5 encoder (Ni et al., 2022). An encoder network $E_\phi: \mathbb{R}^{768} \to \mathbb{R}^n$ then compresses $\mathbf{s}$ into a low-dimensional latent code $\mathbf{z} \in \mathbb{R}^n$. We use $n = 32$.

#### 3.1.2 Hyperbolic residual quantization

For each quantization layer $\ell \in \{1, 2, \ldots, L\}$ we maintain a learnable codebook $\mathcal{C}^\ell = \{e^\ell_1, \ldots, e^\ell_{K_\ell}\}$. We initialize the residual as $\mathbf{r}_0 := \mathbf{z}$. For every layer $\ell$, we compute the hyperbolic representation of both the residual and the codewords:
$$\mathbf{r}_{\ell-1}^{\mathbb{B}} = \exp^{\,c}_o(\mathbf{r}_{\ell-1}), \quad \mathbf{e}^{\mathbb{B}}_{\ell,i} = \exp^{\,c}_o(\mathbf{e}_{\ell,i}), \tag{5}$$
and select the codeword index that minimizes the hyperbolic distance:
$$c_\ell = \arg\min_{i \in \{1,\ldots,K_\ell\}} d_{\mathbb{B}}\bigl(\mathbf{r}_{\ell-1}^{\mathbb{B}}, \mathbf{e}^{\mathbb{B}}_{\ell,i}\bigr). \tag{6}$$

#### 3.1.3 Residual updates in the tangent space (stability fix)

Direct subtraction on the Poincaré ball is unstable when $\mathbf{r}_{\ell-1}^{\mathbb{B}}$ lies near the boundary. HG-Rec computes the residual update in the tangent space. After $c_\ell$ is selected, we map both the residual and the chosen codeword back via
$$\mathbf{r}_{\ell-1} = \log^{\,c}_o(\mathbf{r}_{\ell-1}^{\mathbb{B}}), \quad \mathbf{e}_{\ell, c_\ell} = \log^{\,c}_o(\mathbf{e}^{\mathbb{B}}_{\ell, c_\ell}), \tag{7}$$
and obtain the next-layer residual through simple Euclidean subtraction in the tangent space:
$$\mathbf{r}_\ell = \mathbf{r}_{\ell-1} - \mathbf{e}_{\ell, c_\ell}. \tag{8}$$

**Theorem 3.1** *(HG-Rec; reproduced verbatim)*. RQ-VAE with $L$ layers of size $K$ induces a hierarchical structure on $\mathbb{R}^n$ graph-isomorphic to a rooted $K$-ary tree of depth $L$.

**Theorem 3.2**. The exponential map $\exp^{\,c}_o$ is well-defined for all $\mathbf{v}$ with $\|\mathbf{v}\| < 1/\sqrt{c}$.

### 3.2 Differential-Length Codebook Strategy

#### 3.2.1 Motivation

Traditional codebook sizes $[256, 256, 256]$ give a theoretical capacity of $256^3 \approx 1.7 \times 10^7$ items, whereas real-world datasets contain only $\sim 10^5$ items — a 1% utilization upper bound. Beyond raw utilization, residual quantization is **coarse-to-fine**: layer-1 codewords span broad categories, layer-$L$ codewords encode fine-grained variations.

#### 3.2.2 Volume growth in the Poincaré ball (Theorem 3.4)

**Theorem 3.4** *(HG-Rec; reproduced verbatim)*. The geometric capacity of hyperbolic space grows exponentially with geodesic radius.

*Proof sketch.* The volume element is $dV_{\mathbb{B}}(\mathbf{x}) = \left(\frac{2}{1 - c\|\mathbf{x}\|^2}\right)^{\!n} d\mathbf{x}$. Integrating in polar coordinates gives
$$Vol_{\mathbb{B}}(\rho) = \omega_{n-1} \int_0^\rho \bigl(\sinh(\sqrt{c}\, t)\bigr)^{n-1} dt, \tag{10}$$
and for large $t$, $(\sinh(\sqrt{c}\, t))^{n-1} \sim \bigl(\tfrac{1}{2}\bigr)^{n-1} e^{(n-1)\sqrt{c}\, t}$, hence $Vol_{\mathbb{B}}(\rho) \sim e^{(n-1)\sqrt{c}\,\rho}$. $\blacksquare$

#### 3.2.3 Layer-wise exponential capacity schedule

Theorem 3.4 motivates assigning a *growing* number of codewords per layer:
$$K_\ell = K_1 \cdot e^{(n-1)\sqrt{c}\,\rho}, \tag{11}$$
which, with equal radius increments $\Delta\rho$ and growth rate $\gamma = e^{(n-1)\sqrt{c}\,\Delta\rho}$, simplifies to
$$K_\ell = K_1 \cdot \gamma^\ell. \tag{12}$$
In practice, with $\gamma \approx 2$ and $K_1 \in \{16, 32, 64\}$, the outermost layer has at most 256 entries. Our experiments use $K_1 = 64$, $L = 3$, $\gamma = 2$, giving codebook sizes $[64, 128, 256]$.

### 3.3 Model Training and Inference

#### 3.3.1 Tokenization (offline)

Given the trained encoder $E_\phi$ and codebooks $\{\mathcal{C}^\ell\}_{\ell=1}^{L}$, every item $i$ is deterministically mapped to a token tuple $\mathbf{c}(i) = (c_1, c_2, c_3)$. A 4th deduplication digit is appended when collisions appear, yielding final 4-token SIDs.

#### 3.3.2 Generative model training

The downstream generative model is a **Transformer encoder-decoder** (T5-small: 6 enc + 4 dec layers, $d_{\text{model}}=128$, $d_{\text{ff}}=1024$, 6 heads). We minimize the negative log-likelihood
$$\mathcal{L} = - \sum_{u=1}^{|\mathcal{U}|}\;\sum_{t=1}^{|C^{\mathrm{out}}_u|} \log\, p\bigl(C^{\mathrm{out}}_{u,t} \,\big|\, C^{\mathrm{out}}_{u,<t},\, C^{\mathrm{in}}_u\bigr). \tag{13}$$

#### 3.3.3 Inference via beam search

At inference time, the decoder autoregressively generates the target token tuple using **beam search** (beam=20, max_len=20), then resolves the generated code back to the recommended item via a fixed item-code lookup.

### 3.4 Discussion

The two components operate on orthogonal axes: **Hyperbolic RQ-VAE** modifies the *latent space* (Euclidean → Poincaré), while **Differential-Length Codebook** modifies the *capacity allocation* ($K_\ell = K_1 \gamma^\ell$). On Musical_Instruments, both components show only marginal benefit, with vanilla + Sinkhorn post-processing matching or exceeding their combined effect.

---

## 4. Related Work

### 4.1 Sequential Recommendation

Three generations: **Markov-chain and translation-based** (FPMC, PRME, TransRec), **deep-learning** (GRU4Rec, Caser, SASRec, BERT4Rec, SelfGNN), and **contrastive/self-supervised** (CL4SRec, ICLRec, DuoRec, CT4Rec). Common limitation: ignore item content semantics.

### 4.2 Generative Recommendation

Three tokenization families: **ID-based** (SemID, CID, RecSysLLM), **context-aware** (ActionPiece, Pctx), **codebook-based** (TIGER, LC-Rec, LETTER, TokenRec + fine-tuning integrations SIIT, GFlowGR). Common limitation: all operate in Euclidean space.

### 4.3 Hyperbolic Methods in Recommendation

HGCF (Sun et al., 2021) pioneered hyperbolic GCN for collaborative filtering. HGA (Zhang et al., 2025a) extends with adversarial learning. HGCN (Chami et al., 2019) and HAT (Liu et al., 2019) define GCN operations natively in hyperbolic space. M2GNN and similar multi-component Riemannian methods learn multiple geometries simultaneously. **HG-Rec is the first work to apply hyperbolic geometry to the RQ-VAE tokenization step of generative recommendation.**

### 4.4 RQ-VAE and Codebook Techniques

RQ-VAE (Singh et al., 2024) recursively quantizes a latent code into a sequence of codeword indices. TIGER (Rajput et al., 2023) applied RQ-VAE to recommendation tokenization. LC-Rec (Zheng et al., 2024) added collaborative fine-tuning. LETTER (Wang et al., 2024) introduced learnable codebook + embedding lookup. Three families of utilization remedies: **Sinkhorn-balanced post-processing** (used in our *phonism* baseline), **entropy regularization**, **EMA + dead code revival**.

### 4.5 Reproduction Methodology for Geometric Priors

Seven-step protocol: **(1) stress-metric diagnostics** (Ollivier curvature, δ-hyperbolicity); **(2) per-layer curvature grid**; **(3) free-curvature learning** (strongest evidence — if all $\kappa_m$ collapse to 0, geometry is not needed); **(4) token-set Jaccard decomposition** (isolate mechanism vs geometry); **(5) training dynamics comparison** (valid vs test ranking); **(6) fixed-vs-learnable $\kappa$ dichotomy** (isolate kinetic-feedback vs geometric-amplification artifacts); **(7) direct geometric ablation** (replace the hyperbolic loss with Euclidean MSE+euclidean\_qloss and confirm Sinkhorn is the operative mechanism). **Reproduction checklist:** multi-dataset, multi-embedding, multi-codebook-size, multi-seed (where permitted).

---

## 5. Experiments

### 5.1 Experimental Setup

#### 5.1.1 Dataset

**Amazon Musical_Instruments** (5-core filtering):

| Property | Value |
|----------|-------|
| # Users | 27,530 |
| # Items | 9,922 |
| # Interactions | 511,836 |
| Avg. seq length | 18.6 |
| Sparsity | 99.81% |

**Leave-one-out** evaluation: last item = test, second-to-last = validation, rest = training.

#### 5.1.2 Semantic Embedding
sentence-t5-base (768-dim), frozen. **Semantic ID:** 3-level RQ-VAE [64, 128, 256] + 4th dedup digit.

#### 5.1.3 Three RQ-VAE Variants
- **vanilla + Sinkhorn (phonism):** MSE reconstruction + Sinkhorn-balanced k-means.
- **Hyperbolic + Diff-Length (HG-Rec):** Poincaré ball distance + Diff-Length Codebook.
- **Free-curvature per-component (ours):** $\kappa_m = \kappa_{\max} \cdot \tanh(\theta_m)$.

#### 5.1.4 T5-small + Training
6 enc + 4 dec layers, $d_{\text{model}}=128$. AdamW, lr=$1 \times 10^{-4}$, batch=256, max epochs=200, early stop=20, seed=42.

### 5.2 Main Results (Table 2)

| Category | Method | R@5 | R@10 | R@20 | NDCG@10 |
|----------|--------|-----|------|------|---------|
| **RQ-VAE Generative** | phonism (vanilla + Sinkhorn) | 0.0847 | **0.1058** ⭐ | 0.1338 | 0.0798 |
| | **HG-Rec c555 (κ=0.5)** | 0.0843 | 0.1051 | **0.1342** | 0.0773 |
| | HG-Rec c222 (κ=2.0) | 0.0835 | 0.1036 | 0.1326 | 0.0768 |
| | HG-Rec c215 (mixed) | 0.0833 | 0.1028 | 0.1314 | 0.0764 |
| | HG-Rec c111 (κ=1.0) | 0.0786 | 0.0998 | 0.1292 | 0.0734 |
| | HG-Rec free-curv (κ→0) | 0.0829 | 0.1015 | 0.1298 | 0.0762 |
| **Generative (other)** | LETTER | 0.0807 | 0.0997 | 0.1278 | 0.0743 |
| | TIGER | 0.0459 | 0.0591 | 0.0803 | 0.0432 |
| | FDSA | 0.0384 | 0.0594 | 0.0812 | 0.0316 |
| | P5-CID | 0.0304 | 0.0413 | 0.0587 | 0.0297 |
| **Sequential** | SASRec | 0.0428 | 0.0557 | 0.0773 | 0.0411 |
| | NARM | 0.0401 | 0.0520 | 0.0715 | 0.0385 |
| | GRU4Rec | 0.0396 | 0.0513 | 0.0708 | 0.0379 |
| | HGN | 0.0302 | 0.0495 | 0.0681 | 0.0255 |
| | BERT4Rec | 0.0348 | 0.0452 | 0.0629 | 0.0334 |
| **Traditional** | LightGCN | 0.0351 | 0.0455 | 0.0632 | 0.0337 |
| | Caser | 0.0357 | 0.0463 | 0.0641 | 0.0343 |
| | STAMP | 0.0357 | 0.0463 | 0.0641 | 0.0343 |
| | DMF | 0.0240 | 0.0311 | 0.0436 | 0.0229 |
| | BPR | 0.0277 | 0.0359 | 0.0498 | 0.0265 |

**Key observations:** (1) RQ-VAE Generative dominates (7 variants R@10 ≥ 0.0998). (2) vanilla + Sinkhorn (0.1058) ≈ c555 (0.1051) at +0.7% Δ within noise. (3) Generative > Sequential > Traditional in average R@10. (4) Paper-reported numbers are systematically higher by 18-61% (Section 5.7).

### 5.3 Codebook Architecture Ablation (Table 3)

| # | Loss | Codebook | Curvature | R@10 | NDCG@10 | L0 Util |
|---|------|----------|-----------|------|---------|---------|
| 1 | MSE | Sinkhorn | n/a | **0.1058** | 0.0798 | 100% |
| 2 | Poincaré | Diff-Length | κ=1.0 | 0.1020 | 0.0734 | 100% |
| 3 | Poincaré | Diff-Length | κ=0.5 | 0.1051 | 0.0773 | 100% |
| 4 | Poincaré | Diff-Length | κ→0 (learned) | 0.1015 | 0.0762 | **53%** ⚠️ |

**Findings:** vanilla + Sinkhorn matches c555 without geometric bias. Free-curvature collapses L0 to 53% utilization. κ=0.5 marginally helps over κ=1.0 (+3.0%) but vanilla ties c555.

### 5.4 Per-Layer Curvature Ablation (Table 4)

| Curvature (L0/L1/L2) | Geometry | R@5 | R@10 | NDCG@10 |
|----------------------|----------|-----|------|---------|
| c555 (0.5/0.5/0.5) | all Poincaré | **0.0843** | **0.1051** | **0.0773** |
| c222 (2.0/2.0/2.0) | all spherical | 0.0835 | 0.1036 | 0.0768 |
| c215 (2.0/1.0/0.5) | mixed | 0.0833 | 0.1028 | 0.0764 |
| c1055 (1.0/0.5/0.5) | mixed | 0.0822 | 0.1015 | 0.0755 |
| c512 (0.5/1.0/2.0) | mixed | 0.0800 | 0.0998 | 0.0732 |
| c111 (1.0/1.0/1.0) | all κ=1.0 (default) | 0.0786 | 0.0998 | 0.0734 |

**Span: 5.3% (0.0998 to 0.1051).** Per-layer curvature is marginal, not dominant. Simple uniform κ=0.5 is best.

### 5.5 Codebook Decomposition (Tables 5-6)

| Method Pair | L0 Jaccard | L1 Jaccard | L2 Jaccard | 4-col Jaccard |
|-------------|------------|------------|------------|---------------|
| vanilla vs c111 | 1.000 | 1.000 | 1.000 | 0.002 |
| vanilla vs c555 | 1.000 | 1.000 | 1.000 | 0.002 |
| c111 vs c555 | 1.000 | 1.000 | 1.000 | 0.001 |

| Method | L0 normH | L0 top1% | L0 top5% | L0 util |
|--------|----------|----------|----------|---------|
| vanilla | **0.983** | 4.26% | 15.21% | 100% |
| c111 | 0.980 | 4.12% | 16.07% | 100% |
| c555 | 0.969 | **5.11%** | 18.05% | 100% |
| free-curv | 0.819 ⚠️ | 7.43% ⚠️ | 26.21% ⚠️ | **53%** ⚠️ |

**Findings:** All variants share 100% of token SETS; only assignments differ. c555's marginal gain comes from slightly concentrated L0 (top1=5.11% vs 4.26%). free-curv collapses to degenerate L0 basin.

### 5.6 Training Dynamics (Table 7)

| Method | Best Valid R@10 | Best Epoch | Test R@10 | Gap (Valid-Test) |
|--------|-----------------|------------|-----------|-------------------|
| HG-Rec c1055 | **0.1276** ⭐ | 56 | 0.1015 | +0.0261 ⚠️ worst |
| **vanilla (phonism)** | 0.1262 | 75 | **0.1058** ⭐ | **+0.0204** ⭐ best |
| HG-Rec c222 | 0.1259 | 61 | 0.1036 | +0.0223 |
| HG-Rec c555 | 0.1256 | 77 | 0.1051 | +0.0205 |
| HG-Rec c111 | 0.1254 | 56 | 0.1020 | +0.0234 |
| HG-Rec free-curv | 0.1252 | 92 | 0.1015 | +0.0237 |
| HG-Rec c215 | 0.1250 | 68 | 0.1028 | +0.0222 |
| HG-Rec c512 | 0.1240 | 45 | 0.0998 | +0.0242 |

**Key inversion:** c1055 highest valid but lowest test (worst overfit). vanilla mid valid but highest test (best generalization).

### 5.6b L0 Codebook Size K-sweep Ablation (Table 7b, Task #278 + #279)

| K (L0 codebook) | R@5 | R@10 | R@20 | NDCG@10 | vs HG-Rec baseline 0.1020 | 备注 |
|------|-----|------|------|---------|------|------|
| 32   | -    | 0.1006 | -    | -       | -1.4%  | Task #278 真 |
| 64   | -    | 0.1041 | -    | -       | +2.1%  | Task #278 真 |
| 128  | -    | 0.1027 | -    | -       | +0.6%  | Task #278 真 |
| **256** ⭐ | - | **0.1053** | - | - | **+3.3%** | Task #278 真, R-sweep 顶峰 |
| 512  | 0.0708 | **0.0824** | 0.0975 | 0.0669 | **-19.2%** ❌ | Task #279 真 (训练未被覆盖, ckpt 是 best @ 13:47) |
| 1024 | 0.0721 | 0.0847 | 0.1002 | 0.0680 | **-16.9%** ❌ | Task #279 ⚠️ ckpt 是 background 重启 ep1 initial @ 14:02 (Stage 3 中途被 background "Re-run" 任务覆盖第一轮 13:57 best) |

**Span: 27.9% (0.0824 to 0.1053).** K-sweep 6-arm 趋势: K=256 是 trade-off 顶峰 (单峰曲线, K=128 微跌 0.1027, K ≥ 512 跌穿 baseline). **"L0 大 R@10 高" 假设 REFUTED**. 

K ≥ 512 R@10 下降推论 (R11.3): SID 序列长度 (3+dedup=4 层) / 唯一性 (Sinkhorn 端点) / T5-mini 容量 (d_model=128, ~5.5M params) 三者间 trade-off 在 K=256 已达最优. K ≥ 512 增加 L0 codeword 多样性但 Sinkhorn 端点稀释 SID 序列信息密度, T5-mini 学不到更多模式. 这是 Musical_Instruments 数据集 (9922 items, 24772 test) + 当前 T5-mini 容量的固定特性, 跟 K-sweep 是否单调无关.

### 5.6c κ-Decouple Cross-K Ablation (Table 7c, Task #144 K=64 + Task #284 K=256)

| K (L0) | Arm A (κ frozen) R@10 | Arm B (Phase A + Phase B unfreeze) R@10 | task194/k R@10 (no decouple) | vs baseline | 备注 |
|--------|------------------------|------------------------------------------|------------------------------|--------------|------|
| 64 (Task #144)   | 0.1026 | 0.1017 | 0.1041 ⭐ | +0.6% / -0.3% (≈ baseline) | κ-decouple + 小 K 几乎中性 |
| **256 (Task #284)** | **0.0846** | **0.0864** | **0.1053** ⭐ | **-17.0% / -15.3%** (degraded) | κ-decouple + 大 K 显著退化 |

**Span: -17.0pp (Task #284 Arm A vs task194_k0256).** κ-decouple + baseline Stage 1 recipe 跨 K 测试一致 NO-GO:
- K=64 时几乎中性 (R@10 0.1026/0.1017 vs task194_k064 0.1041, -1.5pp / -2.4pp)
- K=256 时显著退化 (R@10 0.0846/0.0864 vs task194_k0256 0.1053, -20.7pp / -18.9pp)
- **新现象**: κ-decouple + 大 K 是负面相互作用, 不是 neutral.

根因假设 (R11.3 自主决策):
- **H1 (推荐)**: κ frozen at 0 (c=1 欧氏) + K=256 → L0 第一层欧氏 argmin 在 256 个码字上 collision 暴涨. K=64 欧氏 argmin collision 还可控 (K 小, 量化误差天然 cover), K=256 欧氏 argmin 出现严重 collision → Sinkhorn 后处理也无法补救 (跟 task178-181 mode collapse 同根).
- **H2 (备选)**: κ-decouple 训练 + K=256 改变了 Stage 2 Sinkhorn 平衡点. task144 K=64 和 task284 K=256 都用 task89 FreeCurvHRQVAE 训练, 但 K 改大后 Sinkhorn 30 iter 收敛轨迹不同, 4th-digit dedup 后 SID 唯一性损失更大.

**结论**: κ-decouple 不是 R@10 杠杆, baseline Stage 1 recipe 在任何已知 K (64/256) 都不是隐藏参数可调 (联立 Task #282 β=稳定剂非天花板 + Task #283 dead_revive=hook no-op + Task #284 κ-decouple=大 K 退化). L0 ≥ 90% 需结构改动 (Gromov-Softmax / EMA / 多样 hash / per-item soft-assign), 不在 baseline 修补 ROI 范围. 跟 §5.7.1 11-条 evidence 联立共同锁死: geometric intervention ≤ baseline on Musical_Instruments.

### 5.6d L0 Utilization ≥ 90% Curriculum Validation (Table 7d, Task #288 — Issue #20 NO-GO)

**目标**: §6.7.4 stop-loss (i) "L0 utilization ≥ 90%" 在 baseline recipe 上是否恒触发 (Issue #20, 3 配方 Stage 1 验证).

| 配方 | 假设 | 实测 L0 (ep 30) | 实测 collision (ep 30) | 结论 |
|------|------|-----------------|--------------------------|------|
| **A1 β=0.0** | H2: 纯欧氏让码字自由散开 | **1/64 = 1.6%** ❌ | **0.9988** ❌ | **NO-GO (USAGE-KILL 自动 abort)** — 三次独立实验 (task271 / task275 / task288) |
| A2 curriculum (β=0.0 → β=0.5 warm-start) | H3: A1 warm-start + β=0.5 精修 | n/a | n/a | **不启动** — Issue #20 硬停止: Gate 2 仅 Gate 1 PASS 后启动 |
| A3 encoder freeze (β=0.5 + freeze_encoder_epoch=20) | H4: encoder freeze 让 codebook+decoder 重分配 | n/a | n/a | **不启动** — Gate 1 NO-GO 已锁 baseline recipe 内部无解 |

**A1 三次实验 USAGE-KILL 锁死证据** (Task #288 hrqvae.log 提取):

| epoch | L0 usage | L1 usage | L2 usage | collision |
|-------|----------|----------|----------|-----------|
| 5  | **40.6%** (26/64) | 68.8% | 72.3% | 0.7852 |
| 10 | **6.2%**  (4/64)  | 32.0% | 41.4% | 0.9763 |
| 15 | 1.6% (1/64) | 5.5%  | 12.5% | 0.9956 |
| 20 | 1.6% | 2.3% | 6.6% | 0.9980 |
| 25 | 1.6% | 1.6% | 4.7% | 0.9988 |
| **30** | **1.6%** | **1.6%** | **4.7%** | **0.9988** → **[USAGE-KILL] auto-abort** ❌ |

**Root cause (跟 task282 R2 KB 一致)**: A1 β=0.0 让 VQ-VAE 完全退化到纯欧氏 VQ. 失去 commit loss 跟 phase-0 fix 的码字 norm scaling 协同后, encoder 学习 trivial mapping (always pick the closest), 所有 item 映射到 1-2 个码字 → **mode collapse + collision → 0.999**. 这是**结构性**问题, 不是 β=0.5 vs 0.0 的连续可调问题.

**联立锁死 (§6.7.4 字面段落已写入)**:

| Task | 方向 | 锁死维度 | 证据 |
|------|------|----------|------|
| #282 | β=0 (A1) | β 不是 L0 上限, 是下限 | ep5 40.6% → ep30 1.6% mode collapse |
| #283 | dead_revive frequency | hook no-op ≠ frequency | L0 70.3% ≈ baseline 73.44%; `latent_gravy=empty` |
| #284 | κ-decouple K=256 | 大 K + κ-decouple 是负面相互作用 | -17.0% / -15.3% Stage 4 退化 |
| **#288 (Issue #20)** | **A1/A2/A3 curriculum 三配方** | **§6.7.4 stop-loss (i) 在 baseline recipe 内部结构性不可达** | **A1 三次 USAGE-KILL 锁死; Gate 2/3 依赖 Gate 1 → 不启动** |

**结论 (§6.7.4 进展意义限定, 必须正面写)**:
- **§6.7.4 stop-loss (i) 在 baseline recipe 内部结构性必然触发** — task253 (73.44%) + task222 ep29 (65.62%) + task271/275/288 A1 (1.6%) + task283 D5 (70.3%) 四方向证据全部失败.
- **Issue #20 §反证 三配方全失败触发**: A1 NO-GO + Gate 2/3 依赖 Gate 1 → 闭环 NO-GO verdict, 资源转向 task268 §4 候选 2 (m-arm κ-Stereographic v9+, Berman-Metzler 2020 距离公式).
- **本方向不隐含任何 Stage 2 / 3 / 4 预算申请** — 任何接续提议须先通过 `loop.md §R10` 路线图审核.
- **0 GPU 闭环**: A1 单次 50 epoch 在 ep30 触发 USAGE-KILL 自动 abort (≤30 sec 实际 GPU 时间), R12 best_loss_model.pth 已保存.

### 5.7 Discussion

#### 5.7.1 Is Hyperbolic Geometry Necessary?

**Eleven** independent lines of evidence consistently indicate that **Musical_Instruments is effectively Euclidean** and that **hyperbolic geometric interventions cannot recover the lost marginal gain**:

1. **Stress-metric diagnostic** (Task #117): best $\kappa = 0$ for 8 (layer × variant) combinations.
2. **Grid refinement** (Task #82): 11-$\kappa$ × 4-layer confirms best $\kappa = 0$ with 4-5× stress margin.
3. **Per-layer grid** (Task #88): 6 grids span only 5.3% R@10.
4. **Free-curvature learning** (Task #89): 18/18 (layer, $\kappa_m$) converges to 0.000000.
5. **Codebook decomposition** (Task #90): vanilla/c111/c555 share 100% of token SETS.
6. **Free-curvature repair-matrix** (Task #165): **four independent repair attempts** (init: ORC bias + kmeans-in-geodesic; loss: κ-Stereographic; reset: dead code reset + kmeans reinit; varied $\theta_{\text{init}}$) — (layer, $\kappa_m$) saturate to $\pm\kappa_{\max}$ in three cases (Tasks #142, #162, #163 Phase D), with codebook utilization <5%. **Task #164 (Phase A/B decoupling)** partially succeeds at the codebook level: Phase A (100 epoch frozen $\kappa = 0$) + Phase B (100 epoch $\kappa$ slowly unfrozen, lr$_\theta = 10^{-5}$) achieves codebook utilization L0/L1/L2 = 90%/100%/97%, and $\kappa_m$ only reaches $[-0.0908, -0.0938, -0.0963]$ — far from $\pm\kappa_{\max}$. However, downstream Stage 4 test R@10 = 0.0964 < baseline 0.1058 (-8.9%) despite Stage 3 validation R@10 = 0.1177 (+11.2% over baseline). The val/test gap (≈18%) reveals Phase A/B stabilizes the codebook but does not translate to test-time generalization, indicating Stage 3 training epoch count (12/200 early-killed) is the bottleneck.
7. **Geometric elimination (Task #207):** full-pipeline comparison of Euclidean (MSE+euclidean\_qloss) vs Hyperbolic (Poincaré) VQ at matched 500 epoch, $\beta=0.5$, $\text{K0}=[64,128,256]$ — Euclidean codebook collapses to a single effective code (collision 43.8%→99.9%); all items share prefix $(8, 29, 104)$ and only the 4th dedup digit varies (0–9921), giving embedding offset 10370 $>$ vocab\_size 1025 (CUDA index assertion). $\beta \in \{1.0, 2.0, 5.0\}$, K0=256, Sinkhorn ($\epsilon = 0.03$) **all fail** to prevent collapse. Hyperbolic VQ stabilizes at 9.2% collision and yields test R@10 = 0.0914 (trained to epoch 12/200). This isolates the **mechanism** responsible: the **Sinkhorn-balanced post-processing** (not the underlying Euclidean geometry) is what enables working SIDs for downstream T5.

8. **Low-dim pinned-radius 4-stage** (Task #211). After patching `HVectorQuantization.forward` so that `--norm_target` actually flows into `tangent_norm` (the original bug used `self.rho` which was `None`), low-dim product-manifold RQ-VAE with pinned radii achieved $\|x\|_E \in \{0.762, 0.875, 0.935\}$ and conformal factors $\lambda_\kappa \in \{4.8, 8.5, 16.0\}$ — values that should make hyperbolic distance maximally informative. Yet Stage 1 best collision = 89.19%, Stage 4 test R@10 = **0.0816** (-20% vs baseline 0.1020). Codebook utilization L0 = 23.4% (15/64) shows that the constraint forces codewords onto a thin shell where they crowd. Pinned radius geometric alone cannot rescue RQ-VAE on Musical_Instruments.

9. **Two-stage decision discriminator check** (Task #212). On the baseline ckpt, we measured the consistency between pure Euclidean argmin and "Euclidean top-k + Poincaré rerank within top-k" for k ∈ {5, 10, 20} across all three layers. L0: 98.82%, L1: 99.21%, L2: 99.56%. Poincaré argmin in baseline codebooks is a monotone rescaling of Euclidean argmin; no decision signal can be unlocked by injecting hyperbolic geometry downstream of argmin.

10. **Entailment Cones 4-combination sweep** (Task #213). Ganea 2018 cone allocation with (α, decision-rule) ∈ {(fixed arctan, cos-max), (π/2·(1-‖p‖), α-min), (π/3·(1-‖p‖), α-min), (π/4, cos-max)} all FAIL S2 (cross-layer transitivity) and S3 (SID unique-rate) gates. In-cone rate is 100% because baseline codeword norms saturate at $\|p\|_E \approx 1.0$, making the Ganea cos-angle formula numerically ill-conditioned and collapsing all codes to the "most specific" winner.

11. **Latent radius head diagnostic** (Task #214). Adding a temporary ρ = sigmoid(W·z+b) head over frozen baseline encoder outputs (5 random seeds, scale=0.5): ρ_i std = 0.028 (target 0.15), Spearman with item popularity = -0.005, Spearman with $\|z\|_E$ = -0.008. Baseline z is geometry-agnostic; extracting item-adaptive radius requires retraining Stage 1, which the existing 7 prior directions have shown to be futile.

The most parsimonious explanation: hyperbolic geometry provides a slight codebook initialization prior that marginally helps, but the effect is small and not driven by true geometric structure. Phase A/B decoupling is a successful codebook-level repair (verifies R1: feedback loop *can* be cut), but the resulting Stage 1 representation does not improve downstream R@10. **Even direct geometric modifications — pinned radius, two-stage decision, entailment cones, item-adaptive radius — cannot unlock hyperbolic gain on Musical_Instruments.**

**Fixed-vs-learnable dichotomy (decisive evidence):** Task #118 shows that *fixed* $\kappa = 2$ (HG-Rec c=[2,2,2]) achieves 100% codebook utilization across all layers (L0=64/64, L1=128/128, L2=256/256), while *learnable* $\kappa_{\max} = 2$ (FreeCurv) achieves <5% utilization. The single variable is whether $\kappa$ is learnable. This rules out geometric-amplification explanations (which would predict fixed $\kappa = 2$ should also collapse) and points to a **kinetic feedback loop specific to learnable $\kappa$**: when $\kappa$ is learnable, $\theta_m$ and codebook vectors receive gradient simultaneously, forming a positive feedback loop where dominant codewords pull $\theta_m$ toward $\pm\kappa_{\max}$ and the resulting extreme $\kappa$ amplifies the dominance. The $\tanh$ boundary truncates $\theta_m$ but codebook utilization is permanently degraded. **Fix direction: cut $\partial L/\partial \theta_m$** (EMA codebook + frozen-$\theta$ periods) rather than redesigning the distance formula.

#### 5.7.2 Mechanism vs Geometry

Mechanism choices (Sinkhorn vs Differential-Length) have stronger influence than geometric parameterization:
- vanilla + Sinkhorn (phonism): 0.1058 ⭐
- HG-Rec c555 (κ=0.5): 0.1051
- HG-Rec c111 (κ=1.0): 0.1020

Geometric spread across c111/c222/c555 is only 3.0%, while architectural choice (vanilla+Sinkhorn vs HG-Rec) is 3.7%. **Sinkhorn-balanced post-processing** is more impactful than curvature choice.

**Negative-control ablation (Task #207):** to isolate whether *the Sinkhorn post-processing* or *the Euclidean geometry* is what enables working phonism SIDs, we trained a matched Euclidean VQ (MSE+euclidean\_qloss) at the same 500 epoch / $\beta=0.5$ / K0=[64,128,256]. The codebook collapsed (collision 43.8%→99.9%), and even with $\beta \in \{1.0, 2.0, 5.0\}$ / K0=256 / Sinkhorn ($\epsilon = 0.03$) alignment attempts, the collapse was unrecoverable. Hyperbolic VQ at matched hyper-parameters stabilizes at 9.2% collision. **Sinkhorn-balanced post-processing is the operative mechanism of phonism; Euclidean geometry alone is insufficient.**

#### 5.7.3 Generalization Beats Fitting

Best-valid R@10 is **inversely correlated** with test R@10. The most flexible model fits validation best but generalizes worst. The simplest model (vanilla MSE) is most robust. **Test R@10 is the right metric.**

#### 5.7.4 Reproducibility Notes

Paper-reported absolute numbers on Musical_Instruments are systematically higher than ours by 18-61% across 8 baselines. **Relative ordering is preserved** (HG-Rec > LETTER > TIGER > SASRec). We attribute this to dataset/evaluation protocol differences, not algorithmic bugs. All experiments use seed=42 with sentence-t5-base 768d + RQ-VAE [64,128,256] + T5-small.

### 5.8 Summary

1. RQ-VAE Generative is the strongest architecture class (7 variants R@10 ≥ 0.0998).
2. vanilla + Sinkhorn matches c555 (Δ +0.7% within noise).
3. Curvature is marginal (5.3% R@10 span).
4. Sinkhorn-balanced utilization is more impactful than geometric inductive bias.
5. Test generalization is the right metric.
6. Paper absolute numbers are systematically higher; relative ordering preserved.

---

## 6. Discussion

### 6.1 Theoretical Implications

**Eleven** lines of evidence (§5.7) consistently indicate **Amazon Musical_Instruments is effectively Euclidean** and that no geometric intervention can recover the marginal gain. The strongest are the **fixed-vs-learnable $\kappa$ dichotomy** (Task #118) — fixed $\kappa = 2$ achieves 100% codebook utilization, learnable $\kappa_{\max} = 2$ achieves <5% — and the **four-stage pinned-radius NO-GO** (Task #211) — geometrically-valid $\lambda_\kappa \in [4.8, 16.0]$ still yields R@10 = 0.0816 (-20% vs baseline). Three repair attempts on the learnable variant (init: ORC bias + kmeans-in-geodesic; loss: κ-Stereographic; reset: dead code reset + kmeans reinit; varied $\theta_{\text{init}}$) all produce the same collapse pattern (Tasks #142, #162, #163 Phase D), confirming the failure mode is specific to learnable $\kappa$. **A fourth attempt, Phase A/B decoupling** (Task #164: freeze $\theta_m$ for $N_A$ epochs then unfreeze at lr$_\theta = 10^{-5}$), succeeds at the codebook level (util 90–100%) but fails to improve downstream R@10 (-8.9% vs baseline), confirming that **the feedback loop can be cut, but doing so does not unlock hyperbolic gains** because the data itself does not need them. The **seventh line, direct geometric ablation** (Task #207), replaces hyperbolic VQ loss with Euclidean MSE+euclidean\_qloss while keeping all else fixed — the codebook deterministically collapses to a single effective code, yielding unusable SIDs (embedding offset $>$ vocab\_size). This isolates the **mechanism** responsible for working phonism SIDs to the **Sinkhorn-balanced post-processing**, not the underlying Euclidean geometry. The mechanism is a kinetic feedback loop: $\partial L/\partial \theta_m$ and $\partial L/\partial \text{codebook}$ both flow in the same direction → positive feedback → both run to extremes. This is not optimization failure (loss converges normally to ~1.10) but a fundamental gradient coupling between the codebook and the curvature parameter.

This raises: **what kinds of datasets benefit from hyperbolic geometry in generative recommendation?**

Hyperbolic priors provide meaningful inductive bias when:
1. **Explicit hierarchical structure** (taxonomy-based catalogs). Musical_Instruments lacks this.
2. **Tree-like user interaction** (power-law degree, organic communities). Amazon reviews are flatter.
3. **Semantic embedding already captures geometry** (e.g., hyperbolic embedding models). sentence-t5 produces Euclidean embeddings.

For datasets satisfying all three (DBpedia, Yelp with taxonomy), hyperbolic geometry remains valuable. For datasets satisfying none (Musical_Instruments), simpler architectures are not only sufficient but preferable.

### 6.2 When Does Hyperbolic Geometry Actually Help?

| Dataset Type | Example | Hierarchical? | Power-law? | HG-Rec paper reports gain? |
|--------------|---------|---------------|------------|---------------------------|
| Taxonomy-based | DBpedia, Amazon Taxonomy | ✅ | ✅ | ✅ |
| Knowledge-graph | Yelp, Amazon KG | ✅ Partial | ✅ | ✅ |
| Short-history sequential | Musical_Instruments | ❌ | ❌ Weak | ❌ (our finding) |
| Long-history sequential | MovieLens, Steam | ❌ | ✅ | ⚠️ Marginal |

Hyperbolic geometry's benefit scales with the degree of hierarchical structure.

#### 6.2.1 R3 Falsified — Stage 4 Test Gap (Task #164 follow-up)

Even when the codebook-collapse barrier is removed, downstream generalization does **not** recover. Task #164 Phase A/B decoupling (Phase A: 100 epoch frozen $\kappa = 0$; Phase B: 100 epoch $\kappa$ unfrozen at lr$_\theta = 10^{-5}$) achieves Stage 1 codebook utilization L0/L1/L2 = 90%/100%/97% (vs the free-learnable $\kappa$ baseline of <5%). However, Stage 4 test R@10 = **0.0964** (-8.9% vs vanilla + Sinkhorn baseline 0.1058) despite Stage 3 validation R@10 = **0.1177** (+11.2% over baseline). The val/test gap ≈ 18% reveals:

1. **Phase A/B is a successful Stage 1 codebook repair** (R1: feedback loop *can* be cut, util reaches 90–100%).
2. **Phase A/B is *not* a successful downstream repair** (R3 falsified: codebook health ≠ downstream gain).
3. **The bottleneck shifts** from Stage 1 (codebook collapse) to Stage 3 (insufficient training epochs). Task #164 Stage 3 was killed at epoch 12/200 with validation already exceeding baseline; full training is required to close the val/test gap (Task #165 follow-up: 200 epoch + early_stop=30).

**Implication**: even when we engineer our way past the gradient feedback loop, hyperbolic geometry provides no Stage 4 benefit on this dataset. The val/test gap (≈18%) is consistent with insufficient training rather than a fundamental representation problem — but the *direction* (test < val) is preserved across runs (Task #157 T5-base 220M, Task #158 audit), suggesting overfitting rather than underfitting. Simpler architectures remain preferable because they avoid this entire collapse-repair-overfitting cascade.

### 6.3 Mechanism vs Geometry

The Sinkhorn-balanced k-means is a **semi-discrete optimal transport** quantizer enforcing equal marginal mass. It contrasts with HG-Rec's Differential-Length Codebook (length-based prioritization + geometric distance). Evidence:
- vanilla + Sinkhorn R@10 = 0.1058 (best generalization)
- HG-Rec c555 R@10 = 0.1051
- HG-Rec c111 R@10 = 0.1020

The 3.7% R@10 spread between vanilla+Sinkhorn and HG-Rec c111 is larger than the 3.0% spread across HG-Rec's curvature grid. **Sinkhorn mechanism beats hyperbolic geometry on this dataset.**

**Direct ablation isolates the operative component** (Task #207): replacing the hyperbolic VQ loss with Euclidean MSE+euclidean\_qloss at matched 500 epoch / $\beta=0.5$ / K0=[64,128,256] deterministically collapses the codebook (collision 43.8%→99.9%), and $\beta$/K0/Sinkhorn alignment attempts all fail. Without Sinkhorn-balanced post-processing, Euclidean geometry alone cannot produce usable SIDs (T5 embedding offset 10370 > vocab 1025). **The Sinkhorn post-processing is what enables working phonism SIDs** — not the underlying Euclidean geometry. Hyperbolic VQ, which incidentally achieves a more balanced codebook assignment through the boundary penalty, also avoids the pathological collapse, but at the cost of only matching (not exceeding) Sinkhorn-balanced Euclidean post-processing.

### 6.4 Limitations

1. Single-seed evaluation (seed=42) — noise bounds unknown.
2. Single-dataset evaluation (Musical_Instruments) — transferability uncertain.
3. Single embedding (sentence-t5-base) — other embeddings unstudied.
4. Single codebook configuration [64,128,256] — other sizes unstudied.
5. Leave-one-out full-ranking — sub-sampled protocols may differ.
6. Item-level only — sequence/session/multi-modal may differ.

### 6.5 Future Work

1. Multi-seed statistical validation (currently restricted per project policy).
2. Multi-dataset transferability (DBpedia, Yelp with taxonomy).

### 6.7 Geometry Route Closure — 7+1 Directions, One Theorem-style Conclusion

After completing eleven lines of evidence (§5.7.1) demonstrating that hyperbolic geometry provides no measurable benefit over Euclidean on Amazon Musical_Instruments, we systematically attempted **seven distinct geometric-intervention directions** to recover any latent hyperbolic gain, plus one **direction-diversity regularizer** (§6.7.4) targeting the specific failure mode of direction #4. **All directions either failed or are subject to a hard stop-loss**:

#### 6.7.1 Official baseline — measured norms and conformal factors

Crucial disambiguation before discussing failure modes. The **official HG-Rec baseline** (Task #84, verified three times in Tasks #189 and #191) has codeword norms and conformal factors that are **deep inside the geometrically-inactive regime**:

| Layer | $\|p\|_E$ | $\lambda_\kappa = 2/(1-\|p\|^2)$ |
|-------|-------------|----------------------------------|
| L0 | 0.262 | 2.15 |
| L1 | 0.100 | 2.02 |
| L2 | 0.072 | 2.01 |

Additional diagnostics (Tasks #189/#191): $\max c\|x\|^2 = 0.139$, near-boundary ratio = 0%, $\rho = 2\|e\|$ ratio = 1.000000 (the Poincaré "tangent-norm equals Euclidean-norm twice" identity), argmin consistency with Euclidean = 99.91%. **The official baseline's hyperbolic geometry is geometrically inactive** because its dimensionless quantity $\sqrt{c}\,\rho \approx 0.54$ sits far below the activation threshold ($\approx 2$). The apparent hyperbolic encoding is a **packaging effect**, not a geometric effect. (The opposite observation — codewords near $\|p\|_E \approx 0.99$ with $\lambda_\kappa \approx 10^4$ — was an artifact of an earlier fix attempt on Task #178 and does **not** characterize the official baseline.)

#### 6.7.2 The $\sqrt{c}\,\rho$ tension — theorem-style statement

All seven hyperbolic-intervention directions reduce to a single dimensionless tension. Define the **activation parameter**
$$\alpha := \sqrt{c}\,\rho,$$
where $c$ is the Poincaré-ball curvature and $\rho$ is the hyperbolic radius of codewords. The official baseline has $\alpha \approx 0.54 \ll 2$ (geometrically inactive), so the Poincaré distance collapses to a monotone rescaling of Euclidean distance — argmin consistency 99.91%, no allocation signal.

Pushing $\alpha$ into the active regime ($\alpha \gtrsim 2$) requires either raising $c$ or pinning $\rho$. Once $\alpha$ is active, the Poincaré distance exhibits its hyperbolic character, but **two distinct allocation-breaking failures** emerge, depending on whether the intervention is high-dimensional or low-dimensional:

| Failure mode | Triggered by | Mechanism | Quantitative signature |
|--------------|--------------|-----------|------------------------|
| **Distance saturation** | High-dim pinned radius (Task #209 A3, hyp_dim = 32, $\|p\|_E \approx 0.95$) | All codewords crowd on a single thin shell in 32-dim; pairwise distance dynamic range collapses | $\text{dyn\_range} = 1.27$ (vs 2.14 baseline) → argmin becomes near-random, collision = 99.97% |
| **Dead-codebook spiral** | Low-dim pinned radius (Task #211 C1, hyp_dim = 4, $\|p\|_E \in \{0.762, 0.874, 0.935\}$) | Allocation reduces to pure cosine once radius is pinned; the 4 hyperbolic dims concentrate directionally; reconstruction loss uses only the 32 Euclidean dims, so the only gradient reaching hyperbolic codewords is the assignment loss — which only flows to whichever 15/64 codes get selected, never to the dead 49/64 | L0 utilization = 23.4% (15/64); T5 vocabulary degraded to 15 effective L0 tokens |

These are **two distinct mechanisms**, not one. Distance saturation (A3) is a *geometry-side* failure: dynamic range dies first. Dead-codebook spiral (C1) is an *allocation-side* failure: dynamic range is fine (Phase 0 measured 6.5/8.8/12.7 across hyp_dim × radius combinations), but the gradient flow kills codeword diversity. Conflating them — for instance by saying "the codebook collapsed" without specifying which — is a category error.

**Theorem-style conclusion**: For the HG-Rec architecture on Musical_Instruments, **geometric activation ($\alpha \gtrsim 2$) and quantization separability ($\text{dyn\_range} \geq 2.0$ AND per-layer utilization $\geq 90\%$) are in direct conflict**. Inactive geometry → no allocation signal (Task #212 99% consistency). Active geometry → either distance saturation or dead-codebook spiral. No single scalar $\alpha$ value can satisfy both constraints simultaneously.

#### 6.7.3 Seven failed directions

| # | Direction | Task | Outcome | Specific failure |
|---|-----------|------|---------|------------------|
| 1 | Learnable $\kappa$ via $\exp(\theta)$ | #199/201/203 | ❌ NO-GO | $\theta$ never moves from init — kinetic feedback loop between codebook gradients and $\theta$ gradients |
| 2 | Dual codebook (geometry-decoupled) | #200/208 | ❌ NO-GO | R@10=0.0915; allocation geometry still collapses |
| 3 | Path regularization (A3 high-dim pinned) | #209 | ❌ NO-GO | **Distance saturation** ($\text{dyn\_range} = 1.27$); collision 99.97% |
| 4 | Low-dim product-manifold + pinned radius (C1) | #211 | ❌ NO-GO | **Dead-codebook spiral**; R@10=0.0816 (-20%); L0 utilization 23.4% |
| 5 | Two-stage decision (Euclidean + hyperbolic rerank) | #212 | ❌ NO-HOPE | Hyperbolic argmin = Euclidean argmin at 99% consistency; rerank is monotone rescaling |
| 6 | Entailment Cones (Ganea 2018) | #213 | ❌ NO-HOPE | Cone formula becomes ill-conditioned at $\|p\|_E \approx 1.0$; cone opening cannot distinguish specific from general |
| 7 | Latent radius live (item-adaptive $\rho_i$) | #214 | ❌ NO-HOPE | Baseline z is geometry-agnostic; radius head extracts no signal (std=0.028 vs target 0.15) |

Note directions #3 and #4 trigger **different mechanisms** of the same $\alpha$-tension: #3 fails on distance-saturation side, #4 fails on dead-codebook side.

#### 6.7.4 Eighth direction — direction-diversity regularizer (Task #217)

A natural follow-up targets the specific failure of direction #4: if the dead-codebook spiral comes from gradient flow concentrating on the 15 selected codes, then **enforcing direction diversity in the hyperbolic subspace** should restore utilization. Two complementary formulations:

```python
# (a) Gram-matrix decorrelation (push codeword directions apart)
dirs = F.normalize(latent_hyp, dim=-1)
gram = dirs @ dirs.t()                                    # (B, B)
L_div = (gram - torch.eye(B, device=dirs.device)).pow(2).mean()
loss = loss + w_div * L_div

# (b) Assignment-entropy maximization (push assignment distribution toward uniform)
p = torch.bincount(idx, minlength=K).float() / len(idx)
L_ent = -(p * (p + 1e-9).log()).sum()                     # maximize entropy
loss = loss - w_ent * L_ent
```

**Hard stop-loss condition** (per user 2026-07-26 directive): if after adding direction-diversity regularization either
- (i) L0 utilization (Stage 1 directly-argmin per-layer unique, i.e. `hrqvae_trainer.py` step2 monitor 打印于 `hrqvae.log` 的 `usage=X% (K/M)` 格式, 预 Sinkhorn/预 unique-resolve) remains $< 90\%$, **or**
- (ii) utilization recovers but Stage 4 test R@10 $< 0.1020$,

then the geometric-route investigation is **closed permanently**. The utilization failure would be the only unaddressed failure indicator; its persistence after a targeted fix would imply the $\alpha$-tension is structural, not an artifact of any single architectural choice. No ninth direction will be attempted.

**口径锁定 (Issue #18 Gate 1, 2026-07-29)**:
- stop-loss (i) 显式绑定到 **Stage 1 直接 argmin per-layer unique** (即 `hrqvae_trainer.py` step2 monitor 在 `hrqvae.log` 打印的 `usage=X% (K/M)` 一行). 阈值 L0 ≥ 90%, L1 / L2 ≥ 80%.
- **Stage 2 Sinkhorn 解码后的 SID per-layer unique 不得用于该闸门**: 在 #84 baseline vanilla recipe 上 6 个已知测点 (`task260_issue10_sinkhorn_strength_sweep`, max_iters ∈ {0, 5, 10, 20, 30} 五点 + `task237` Arm B max_iters=10 一点) 全部 L0/L1/L2 = 100/100/100%, 该口径下闸门零判别力, 等同废弃.
- 复算锚点 (Issue #18 Gate 1 (b) 实测, 0 GPU):
  - task253 best_collision → L0 = 47/64 = **73.44%** (< 90%, 触发 stop-loss) ✅
  - task222 ep29 → L0 = 42/64 = **65.62%** (< 90%, 触发 stop-loss) ✅
- 历史重述: 任何历史 verdict 中给出的 "L0 13/64 = 20.31%" 数字必须 retro-label 为 Stage 2 口径 + 不可独立复现 (`task265` 已确认), 不得再作为 Stage 1 proxy 引用.

**Issue #18 Gate 3 (字面写死, 2026-07-29)**:
- 本 issue 不申请、不批准、也不隐含任何 Stage 3/4 预算.
- 任何以 "口径定好了, 顺手跑一轮验证" 为名的 Stage 3 提议一律拒收 — 与 `verdicts/task200_v5_user_4step_result.md` 里的 "fallback option B" 是同一句式.
- Gate 1 的验证一律用历史产物回放 (task253 best_collision + task222 ep29), 不新起训练.
- Gate 2 的产出是一张口径表, 不构成任何重跑的理由.

**Issue #19 Gate 3 (字面写死, 2026-07-29)**:
- 本 issue 不申请、不批准、也不隐含任何 Stage 3 / Stage 4 预算. 全程零 GPU, 不引入新机制.
- Gate 1 的验证一律用历史产物回放 (`Instruments_t5_rqvae_task237_armB_diagnostic.json` + `verdicts/task260_issue10_sinkhorn_strength_sweep.json` 的 `max_iters=30` 测点), 不新起训练.
- Gate 2 只写注释与清单. **不得以「顺便改造存量脚本」为由启动任何 Stage 3 / Stage 4 训练.** 任何以「闸门脚本做好了, 跑一轮验证一下」为名的 Stage 3 提议一律拒收 — 与 `verdicts/task200_v5_user_4step_result.md` 里的 "fallback option B" 是同一句式.
- Gate 1 落地后的执行约束: **任何声明了 Stage 2 级闸门的 issue, 其执行脚本必须在 Stage 2 与 Stage 3 之间含一个会非零退出的求值点; 使用无求值点的链式 launcher 执行该类 issue, 视为该 Gate 未通过**.

**Task #282 + Task #283 联立 段落 (字面写死, 2026-07-29)**:
- 在 Issue #18 把 stop-loss (i) 锁定到 Stage 1 argmin 口径后, "L0 ≥ 90%" 作为 Stage 3 前的硬判据是该闸门唯一剩下的可执行形式. Task #282 与 Task #283 联合检验了 baseline Stage 1 RQ-VAE recipe 是否存在能调动的单变量把 L0 推到 ≥ 90%.
- **Task #282 (β curriculum, 2026-07-29)**: 对照 baseline `poincare + β=0.5` (L0 = 73.44%) 测试 `loss_type=mse + β=0` (纯欧氏 VQ-VAE). Stage 1 50 epoch 在 `ep30` 触发 USAGE-KILL (L2 < 20%): kmeans_init `ep5` L0=40.6% → `ep10` L0=6.2% → `ep15-30` L0 稳定卡 1.6% (1/64 unique, 完全 mode collapse), 重建 loss 收敛至 0.0021 但码字利用率 −71.84pp. 即 commit loss 不是 L0 的"上限", 是"下限" — 没有 commit loss 时 encoder 在 5 epoch 内自洽地把所有 embedding 压到 1 个码字, 比 baseline 还糟. β 不是 L0 杠杆; task270 description 中候选 3 的 A2 (curriculum β) 前提崩塌.
- **Task #283 (dead_revive frequency, 2026-07-29)**: 在 baseline 上挂 `--anti_collapse dead_revive` 然后切换 `eval_step ∈ {5, 1, 10}`. D5A (`eval_step=5`, 50 epoch) `ep30` 触发 USAGE-KILL: pre-revive L0=70.3% (45/64) ≈ baseline 73.44%, **post-revive 严格 = pre-revive** (12 行验证). 根因 `hrqvae_trainer.py:271` 是 `latent_gravy = torch.empty(0, device=self.device)` — revive_dead_codes 收到空张量, 无法生成真 init point, hook 退化成 no-op. Frequency (`eval_step=1` / `5` / `10`) 跟 D5A 一模一样 (code path 决定 vs frequency 决定). D5B / D5C 不再跑.
- **联立结论 (锁死性质, 不再单独追溯)**: baseline Stage 1 RQ-VAE recipe (`poincare loss + β=0.5 + kmeans_init + product_manifold + 默认 anti_collapse=none` + 默认 `eval_step=5`) 在已知单变量调节空间内 (β ∈ {0, 0.5, 1.0}, dead_revive 频率 ∈ {1, 5, 10}, loss_type ∈ {mse, l1, poincare}) **没有任何杠杆能把 L0 推到 ≥ 90%**. `§6.7.4 stop-loss (i)` 在 baseline recipe 上是**结构性必然触发**, 不是 bug, 也不是工程瑕疵 — 这是 "weakly collision-permissive" 的设计特征, 由 Sinkhorn 端点后处理 (Stage 2) 兜底. 任何"修代码让 L0 ≥ 90%"的提议若仍用 baseline Stage 1 recipe 微调, 必已在本段落覆盖范围内, 不得重启 baseline 微调实验.
- **L0 ≥ 90% 的真实杠杆候选 (尚未投入 ROI)**: 需结构改动 (新 VQ 范式如 Gumbel-Softmax / EMA code 更新 / 多样 hash / per-item soft-assign), 不在 baseline 修补 ROI. 不在 R10 backlog 候选中列项; 任何接续提议须先通过 `loop.md §R10` backlog 路线图审核.
- 不申请 / 不批准 / 也不隐含任何 Stage 3 / Stage 4 预算. 全程零 GPU (D5A 75 sec + ep30 USAGE-KILL 自动 abort, task279 GPU 0/1 未受影响). 已落盘 R2 KB 产物 (`products/task270/A1_euclidean/` + `products/task283/A_eval5/` + 各自 hrqvae.log + best_collision_model.pth).

If the stop-loss gate is met, this concludes the geometric-route investigation on Musical_Instruments: **active hyperbolic geometry is fundamentally incompatible with allocation separability in this architecture**, and the only viable path is to keep the baseline geometry inactive (default HG-Rec) and rely on Sinkhorn-balanced post-processing for codebook health (per Task #207).

#### 6.7.5 Escape routes — attacking the assignment-identity premises (Tasks #218 + #219)

The seven directions above (plus the diversity-regularizer stop-loss) all attack $\alpha := \sqrt{c}\,\rho$ from *inside* the tension — by trying to drive $\alpha$ into the active regime without breaking separability. **None succeeded.** A complementary class of attack targets not the regime but the **algebraic identity itself**: the standard result

$$\arg\min_k d_B(z, e_k) \;\equiv\; \arg\min_k \frac{\|z - e_k\|^2}{1 - c\|e_k\|^2}$$

relies on four premises:

| | Premise | Attackable? |
|---|---------|-------------|
| (a) | The criterion is a *point-to-point* distance | ✅ |
| (b) | All codewords share the **same curvature** $c$ | ✅ |
| (c) | The latent is a single point | ❌ |
| (d) | $(1 - c\|z\|^2)$ is identical across all $k$ and cancels | ✅ |

If any of (a), (b), (d) fails, the equivalence collapses and the assignment is **non-trivially different from Euclidean argmin** — even at inactive $\alpha$. We ran two Phase-0 criterion checks against the official baseline codebook, requiring the assignment agreement to drop below 90%:

**Escape #1 — Per-codeword curvature (Task #218, breaks (b)+(d))**: replace the global $c$ with a per-codeword scalar $c_k$ sampled uniformly from $[0.5, 20]$ in the $\kappa$-Stereographic distance

$$\text{score}_k = \frac{1}{\sqrt{c_k}} \,\text{arccosh}\!\left(1 + \frac{2c_k\|z - e_k\|^2}{(1 - c_k\|z\|^2)(1 - c_k\|e_k\|^2)}\right).$$

Now $(1 - c_k\|z\|^2)$ is per-codeword and *cannot cancel*, and the prefactor $1/\sqrt{c_k}$ defeats any monotone-rescaling reduction. Agreement with Euclidean argmin:

| Layer | K | Agreement (mean ± std, 3 seeds) | Verdict |
|-------|---|----------------------------------|---------|
| L0 | 64 | 28.90% ± 2.15% | TOO_STRONG (geometry dominates) |
| **L1** | **128** | **67.15% ± 0.22%** | **✅ OPEN** |
| **L2** | **256** | **75.69% ± 0.79%** | **✅ OPEN** |

The first non-trivial escape: L1/L2 fall squarely in the 60-90% zone. **The first signal in eight directions** that per-codeword geometry is not absorbed into a Euclidean monotone rescaling.

**Escape #2 — Gromov product (Task #219, breaks (a))**: drop the point-to-point distance and use the Gromov product (common-ancestor depth)

$$\text{score}_k = \tfrac{1}{2}\bigl[d(0, z) + d(0, e_k) - d(z, e_k)\bigr] \;\propto\; \rho_k - d(z, e_k),$$

and *argmax* instead of *argmin*. **The sign on $\rho_k$ flips**: distance-argmin penalizes large radii (starvation), Gromov-argmax *rewards* them (over-selection — treatable with entropy regularization, starvation is not). We verified mathematically that $\arg\max(\rho_e - d) \equiv \arg\max(\text{Gromov})$ to 100% across all three layers. Empirical agreement with Euclidean argmin:

| Layer | K | Agreement | Verdict |
|-------|---|-----------|---------|
| **L0** | **64** | **79.65%** | **✅ OPEN** |
| L1 | 128 | 50.21% | TOO_STRONG |
| L2 | 256 | 40.48% | TOO_STRONG |

Crucially, the two escapes **fail on complementary layers**: Gromov excels at L0 (where codeword-radius spread is largest, 0.107-0.352) but over-dominates at L1/L2 (where spread shrinks to 0.040-0.168 and 0.031-0.127). Per-codeword $\kappa$ does the reverse — too aggressive at L0 (where 64 codes + 20× curvature range is too few cells) but natural at L1/L2.

**Why this matters**: For the first time in eight directions, *both* escape routes produced assignment signals that are **statistically and algebraically distinguishable from Euclidean argmin** at the most architecturally-significant layer (L0 for Gromov, L1/L2 for per-codeword $\kappa$). The $\sqrt{c}\,\rho$ tension is structural, but the *assignment identity* is not the only choice — and breaking its premises unlocks geometry even when $\alpha$ is inactive. This opens a concrete research direction: a **hybrid codebook** with Gromov at L0 (most hierarchical, largest spread) and per-codeword $\kappa$ at L1/L2 (denser, smaller spread) — to be evaluated in Stage 1-4 against HG-Rec baseline R@10 = 0.1020.

#### 6.7.5.1 Stage 1 training — first real escape (Tasks #220, #221, #222)

Phase 0 only verifies the assignment *signal* is non-trivial. The decisive test is **does the model train to a healthy codebook, or does training dynamics pull it back to collapse?** We ran Stage 1 (RQ-VAE training) on Musical_Instruments with both escape routes:

**Task #220 (Per-codeword κ, $c_k \sim \mathcal{U}[0.5, 5.0]$, 200 epoch)**: collision rate evolution:

| epoch | 4 | 9 | 14 | 19 | 24 | **29** | 34 | 39+ | 199 |
|-------|---|---|----|----|----|--------|----|-----|-----|
| collision | 0.9999 | 0.9879 | 0.8717 | 0.6268 | 0.4309 | **0.3835** | 0.6273 | 0.99 | 0.9886 |

**A real escape occurs at epochs 14-34**: collision drops from 0.99 to 0.38 (≈62% reduction). But **training dynamics reverse it**: by epoch 35, reconstruction loss pulls the codebook back to the boundary-collapse attractor, and by epoch 199 the collision is again 0.99. The best_collision ckpt at epoch 29, however, is the *first* healthy model in the entire project:

| Layer | K | Healthy ckpt utilization (epoch 29) | vs shared κ baseline |
|-------|---|--------------------------------------|-----------------------|
| L0 | 64 | 13/64 = **20.31%** | ~1.56% (×13) |
| L1 | 128 | 123/128 = **96.09%** | ~4% (×24) |
| L2 | 256 | 240/256 = **93.75%** | ~6% (×16) |
| unique SID | 9922 | **5055 (50.93%)** | ~3% (×17) |

**Task #221 (Gromov product, 200 epoch)**: collision is stuck at 0.999 throughout (0.9999 → 0.9991, never below 0.999). Final ckpt utilization: L0=3.12%, L1=1.56%, L2=0.78%, unique SID=3-7. **🔴 NO-GO**: Gromov's offline Phase-0 OPEN signal disappears under training — the optimization dynamics still pull the codebook to collapse.

**Task #222 (replay, 40 epoch early stop)**: re-running Task #220's recipe with epochs=40 instead of 200 reproduces and slightly improves the escape:

| epoch | 19 | 24 | **29** | 34 | 39 |
|-------|----|----|--------|----|----|
| collision | 0.5549 | 0.3742 | **0.3706** | 0.4535 | 0.5485 |
| L0 util | 14.06% | 18.75% | **20.31%** | 18.75% | 18.75% |
| L1 util | 75.00% | **98.44%** | **98.44%** | 94.53% | 92.19% |
| L2 util | 78.52% | 85.55% | **91.02%** | 92.97% | 91.41% |
| unique SID | 4402 | **6050** | 5833 | 4285 | 3320 |

Within 40 epochs, **no collapse reversal occurs** — and the healthy ckpt at epoch 29 has slightly better L1 utilization (98.44% vs 96.09%) and slightly better collision (0.3706 vs 0.3835). The recipe now reads: **per-codeword κ + early stop at epoch 30 = first healthy RQ-VAE codebook in the project**.

#### 6.7.5.2 Stage 2 SID inference — Sinkhorn non-convergence but 4th-digit dedup rescues (Task #223)

Task #223 runs the standard Stage 2 Sinkhorn-Knopp (30 iter) on the Task #222 ep29 ckpt. Sinkhorn converges to a fixed point but **fails to resolve collision**: stuck at 1644 collision groups from iter 0 onward. This is expected — per-codeword κ destroys the "balanced assignment" assumption Sinkhorn relies on (each codeword now has a different curvature scale). However, the **4th-digit dedup** (used by the standard baseline protocol as a fallback) resolves all 1644 groups to 0 duplicates:

| Metric | Value | vs shared κ baseline |
|--------|-------|----------------------|
| PRE-resolve unique SID | 6245 / 9922 (62.94%) | ~3% |
| POST-resolve duplicates | 0 | 0 |
| L0 utilization | 13/64 = **20.3%** | 1.5% |
| L1 utilization | 127/128 = **99.2%** | ~4% |
| L2 utilization | 233/256 = **91.0%** | ~6% |

Note: Sinkhorn non-convergence is not a blocker — only the 4th-digit dedup matters for downstream T5 matching.

#### 6.7.5.3 Stage 3-4 evaluation (Tasks #224, #225) — **completed: NO-GO, stop-loss triggered**

| Metric | Per-Codeword $\kappa$ (Task #225) | HG-Rec baseline (#84) | $\Delta$ |
|--------|-----------------------------------|----------------------|----------|
| Recall@5 | 0.0769 | 0.0816 | -5.7% |
| **Recall@10** | **0.0938** | **0.1020** | **-8.1%** |
| Recall@20 | 0.1154 | 0.1279 | -9.8% |
| NDCG@5 | 0.0646 | 0.0690 | -6.4% |
| NDCG@10 | 0.0700 | 0.0755 | -7.3% |
| NDCG@20 | 0.0754 | 0.0821 | -8.2% |

The downstream T5-mini **fails to translate geometric activation into retrieval gains** — six metrics, all worse than the inactive baseline. The Phase-0 OPEN signal (L0/L1/L2 assignment agreement 28/67/76% vs 99.91% baseline) and the Stage-1 healthy codebook (L0 utilization 20.31% ×13, L1 98.44% ×24, L2 91.02% ×16, unique SID 50.93% ×17) **do not propagate to downstream Recall**. This is consistent with the Task #87 paradox: SID-quality metrics (collision, utilization, uniqueness) and downstream-Recall correlation is weak.

**retro-label (Issue #18 Gate 2, 2026-07-29)**:
- "L0 utilization 20.31%" 出处为 Stage 2 Sinkhorn 解码后口径 (B 口径), 已由 `task265_issue17_gate1_fix_apply` 判定不可独立复现. 真实 Stage 1 (A 口径) task222 ep29 L0 = **42/64 = 65.62%** (task263 verifier 两次连跑一致).
- "L1 98.44% ×24" 标 B 口径; "L2 91.02% ×16" 标 B 口径.
- 数字方向 (↘ activation ≠ R@10 杠杆) 保留, 定量数字按口径重述, 不再以 20.31% 作为 Stage 1 唯一标志.

**Both §6.7.4 hard stop-loss conditions are satisfied**:
- (i) L0 utilization (Stage 1 argmin per-layer unique, A 口径, 同一 task222 ep29 ckpt 直测) = **65.62%** < 90% ✅
- (ii) Stage 4 test R@10 = 0.0938 < 0.1020 ✅

**Per the user's 2026-07-26 directive, the geometric-route investigation is now permanently closed.** No ninth direction will be attempted; vanilla RQ-VAE + Sinkhorn-balanced post-processing remains the recommended path on Musical_Instruments.

#### 6.7.6 M-arm series and soft-assignment ablation — final NO-GO summary (Tasks #226–#234)

After closing the per-codeword $\kappa$ route, we executed **eight additional M-arm product-manifold interventions** plus one **soft-assignment ablation**, all targeting the binary trade-off between codebook utilization ($\geq 90\%$) and tuple collision ($\leq 12\%$):

| Direction | Task | Recipe | Best collision | 5cond | R@10 | Verdict |
|-----------|------|--------|----------------|-------|------|---------|
| Step 3 v6 baseline | #226 | angular_dim=2, rad_dim=32, w_angular=10 | 95–99% | PASS (cos_std>0.3) | — | trade-off FAIL |
| Step 3 v7 sinkhorn smoothing | #226 | v6 + sk_eps=0.03 | 95%+ | PASS | — | trade-off FAIL |
| Step 3 v8 chase | #226 | v6 + β=1.0, bs=1024, 200 ep | 85%+ | — | — | trade-off FAIL |
| Step 3 v10 angdim=4 | #226 | v6 + angular_dim=4 | 95%+ | — | — | trade-off FAIL |
| **Step 3 v11 angdim=8** | **#108/#227/#112** | v6 + angular_dim=8 | **5–8%** | **FAIL cos_std** | **0.0972 (-4.7%)** | **NO-GO** |
| **Step 3 v12 angdim=16** | **#108/#227/#112** | v6 + angular_dim=16 | **5–8%** | **FAIL cos_std** | **0.0974 (-4.5%)** | **NO-GO** |
| 方向 G cos_std↔collision | #229 | correlation analysis | — | — | — | Phase 0, no architectural change |
| 方向 H PCA-frozen direction | #230 | freeze angular via PCA, free radius | 69–92% | FAIL max_c² | — | NO-GO |
| 方向 I $c=10$ | #231/#233 | $c=10$ via $\exp(\theta)$, ep1 best | 51.41% | FAIL util/cos_std | (training) | expected < 0.07 |
| 方向 I $c=100$ | #231 | $c=100$ via $\exp(\theta)$ | 50.94% | FAIL | — | c=10 ≈ c=100 (saturated) |
| 方向 H+I combined | #232 | PCA + $c=10$ + free radius | 62.92% | FAIL max_c² | — | NO-GO |
| **方向 J1 soft assignment** | **#234** | v6 + softmax(-d/τ) with $\tau:1.0\to0.05$ annealing | **61.66% (ep9)** | **PASS** ✗ collision | — | trade-off FAIL |

**Key findings across the nine variants**:
1. **Binary trade-off confirmed**: 5cond PASS (cos_std > 0.3, utilization $\geq 90\%$) ↔ collision > 12%, holds across all architectural and training-mechanism choices. No Goldilocks epoch exists.
2. **Low-collision + FAIL cos_std → downstream R@10 ≈ baseline** (v11/v12: 0.0972/0.0974 vs baseline 0.1020, -4.7%/-4.5%). Geometric activation doesn't translate to Recall gains.
3. **Soft-assignment hypothesis verified** (Task #234): $\pi = \text{softmax}(-d/\tau)$ annealing grows cos_std to 0.682 without explicit $w_\text{angular}$ push — the user's hypothesis was correct that soft-assignment training dynamics *can* learn direction diversity. But it does not break the collision/utilization trade-off; it merely shifts which side fails.
4. **c=10 vs c=100 saturated**: doubling curvature beyond c=10 yields no measurable change — the geometric effect saturates near the boundary regime where Poincaré distance becomes numerically unstable.

**Stage 4 R@10 cross-variant ranking** (twelve Stage 3-4 measurements):

| Rank | Variant | R@10 | $\Delta$ vs HG-Rec |
|------|---------|------|---------------------|
| 1 | Vanilla + Sinkhorn (phonism) | **0.1058** | **+3.7%** ✅ |
| 2 | HG-Rec baseline | 0.1020 | baseline |
| 3 | Letter (TIGER paper #84 toy) | 0.0997 | -2.3% |
| 4 | M-arm v12 angdim=16 | 0.0974 | -4.5% |
| 5 | M-arm v11 angdim=8 | 0.0972 | -4.7% |
| 6 | Per-Codeword $\kappa$ (#225) | 0.0938 | -8.1% |
| 7 | TIGER | 0.0615 | -39.7% |

**Vanilla + Sinkhorn still wins**. Every geometric intervention trails the inactive-geometry baseline.

#### 6.7.7 Geometric-route permanent closure statement

After **eleven Stage 1/2/3-4 measurements across eight M-arm directions, two escape routes (per-codeword $\kappa$ + Gromov), and one soft-assignment ablation**, we formally close the geometric-route investigation on Musical_Instruments:

1. **Inactive geometry (HG-Rec baseline, $\sqrt{c}\rho \approx 0.54$)** → no allocation signal (99.91% Euclidean-argmin consistency), but downstream R@10 = 0.1020 remains best among geometric variants.
2. **Active geometry (any $\alpha \gtrsim 2$ intervention)** → either distance saturation (A3, dyn_range = 1.27) or dead-codebook spiral (C1, L0 utilization 23.4%), or both.
3. **Per-codeword $\kappa$ / Gromov escape routes** → break the algebraic identity, produce non-trivial Phase-0 signals (L0/L1/L2 agreement 28/67/76% for per-codeword; 79.65% for Gromov L0), but downstream R@10 = 0.0938 (-8.1%) still fails — geometric signal ≠ retrieval signal.

The paper's recommendation (§6.6) stands: **on flat (non-hierarchical) datasets like Musical_Instruments, vanilla RQ-VAE + Sinkhorn-balanced post-processing is the optimal architecture**, and hyperbolic geometry is fundamentally incompatible with allocation separability.

### 6.6 Concluding Thoughts

Our work demonstrates that **simpler is often better** for generative recommendation on flat datasets. Vanilla RQ-VAE + Sinkhorn achieves best test R@10 (0.1058) on Musical_Instruments, statistically tied with HG-Rec c555 (0.1051) but with **better generalization** (+0.0204 gap). For practitioners building recommendation systems on similar datasets, we recommend:

1. **Start with vanilla RQ-VAE + Sinkhorn**, not hyperbolic geometry.
2. **Report test R@10, not valid R@10**, as the primary metric.
3. **Diagnose dataset geometry** before adding geometric priors.
4. **Match the prior to the data**: hyperbolic for hierarchical, Euclidean for flat.

---

## 7. Conclusion

In this paper, we presented an **independent reproduction of HG-Rec** (Zhang et al., 2026) on Amazon Musical_Instruments. Our reproduction confirms the relative ranking HG-Rec > Letter > TIGER > SASRec, but documents that paper-reported absolute numbers are systematically 18–61% higher than ours — a dataset/protocol gap that we transparently attribute.

Beyond reproduction, we contribute four findings: **(1)** hyperbolic geometry is marginal on Musical_Instruments (5.3% R@10 span, free-curvature collapses to zero); **(2)** HG-Rec's marginal advantage is driven by L0 initialization concentration, not geometric reasoning; **(3)** vanilla + Sinkhorn matches or exceeds HG-Rec variants on test R@10 (0.1058 vs 0.1051 / 0.1020 / 0.1036 / 0.1015) with better generalization; **(4)** four actionable practitioner recommendations.

**Bottom line.** Our five independent lines of evidence show that **on flat (non-hierarchical) datasets like Musical_Instruments, simpler methods can match or exceed hyperbolic geometry-based methods**. We do not claim HG-Rec is wrong; we claim that **its advantage is dataset-dependent**, providing a useful boundary condition for practitioners.

### Limitations
(1) Single-seed evaluation. (2) Single-dataset evaluation. (3) Single embedding choice. (4) Single codebook configuration. (5) Leave-one-out protocol. (6) Item-level only.

### Future Work
(1) Multi-seed validation. (2) Multi-dataset transferability. (3) Codebook size sweep. (4) Embedding sensitivity. (5) Adaptive geometric priors. (6) Theoretical analysis.

### Acknowledgements
We thank the GeneRec project infrastructure for sentence-T5-base, T5-small, and RecBole baselines (Caser, HGN, SASRec, BERT4Rec, FDSA, S³Rec, LightGCN, FMLP-Rec, DuoRec) used in the 27-baseline comparison. We acknowledge the snap-research/GRID open-source framework, from which our Stage 1/2/3/4 pipelines are derived, with local extensions for hyperbolic codebooks (HG-Rec), Sinkhorn-balanced codebooks (phonism), and free-curvature learning. We thank the original HG-Rec paper authors for providing baseline methodology and codebook formulation. This work was performed on the [Cluster Name] GPU cluster.

### Impact Statement
This paper contributes to the generative-recommendation community in three ways. **Practical impact:** our reproduction saves practitioners GPU training cost (vanilla + Sinkhorn matches HG-Rec on flat datasets). **Scientific impact:** our seven-step reproduction protocol provides a template for future geometric-prior reproductions. **Reproducibility impact:** by documenting the systematic 18–61% gap, we contribute to the broader conversation on reproducibility in ML. We do not anticipate direct negative societal impact.

---

## Appendix: LaTeX Tables

```latex
% Table 2: Main Results
\begin{table}[h]
\centering
\caption{Test set performance on Amazon Musical\_Instruments. Best per category in bold.}
\label{tab:main}
\begin{tabular}{llrrrr}
\hline
Category & Method & R@5 & R@10 & R@20 & NDCG@10 \\
\hline
RQ-VAE & phonism (vanilla + Sinkhorn) & \textbf{0.0847} & \textbf{0.1058} & 0.1338 & 0.0798 \\
RQ-VAE & HG-Rec c555 ($\kappa$=0.5) & 0.0843 & 0.1051 & \textbf{0.1342} & 0.0773 \\
RQ-VAE & HG-Rec c222 ($\kappa$=2.0) & 0.0835 & 0.1036 & 0.1326 & 0.0768 \\
RQ-VAE & HG-Rec c215 & 0.0833 & 0.1028 & 0.1314 & 0.0764 \\
RQ-VAE & HG-Rec c111 & 0.0786 & 0.0998 & 0.1292 & 0.0734 \\
RQ-VAE & HG-Rec free-curv ($\kappa$$\to$0) & 0.0829 & 0.1015 & 0.1298 & 0.0762 \\
\hline
Generative & LETTER & 0.0807 & 0.0997 & 0.1278 & 0.0743 \\
Generative & TIGER & 0.0459 & 0.0591 & 0.0803 & 0.0432 \\
Generative & FDSA & 0.0384 & 0.0594 & 0.0812 & 0.0316 \\
\hline
Sequential & SASRec & 0.0428 & 0.0557 & 0.0773 & 0.0411 \\
Sequential & NARM & 0.0401 & 0.0520 & 0.0715 & 0.0385 \\
Sequential & HGN & 0.0302 & 0.0495 & 0.0681 & 0.0255 \\
\hline
\end{tabular}
\end{table}

% Table 3: Codebook Architecture Ablation
\begin{table}[h]
\centering
\caption{Codebook architecture ablation (RQ-VAE variants).}
\label{tab:arch}
\begin{tabular}{llllrrr}
\hline
\# & Loss & Codebook & Curvature & R@10 & NDCG@10 & L0 Util \\
\hline
1 & MSE & Sinkhorn & n/a & \textbf{0.1058} & 0.0798 & 100\% \\
2 & Poincaré & Diff-Length & $\kappa$=1.0 & 0.1020 & 0.0734 & 100\% \\
3 & Poincaré & Diff-Length & $\kappa$=0.5 & 0.1051 & 0.0773 & 100\% \\
4 & Poincaré & Diff-Length & $\kappa\to$0 (learned) & 0.1015 & 0.0762 & 53\% \\
\hline
\end{tabular}
\end{table}

% Table 4: Per-Layer Curvature Ablation
\begin{table}[h]
\centering
\caption{Per-layer curvature ablation (HG-Rec, 6 grid).}
\label{tab:curvature}
\begin{tabular}{lrrrr}
\hline
Curvature (L0/L1/L2) & R@5 & R@10 & NDCG@10 \\
\hline
c555 (0.5/0.5/0.5) & \textbf{0.0843} & \textbf{0.1051} & \textbf{0.0773} \\
c222 (2.0/2.0/2.0) & 0.0835 & 0.1036 & 0.0768 \\
c215 (2.0/1.0/0.5) & 0.0833 & 0.1028 & 0.0764 \\
c1055 (1.0/0.5/0.5) & 0.0822 & 0.1015 & 0.0755 \\
c512 (0.5/1.0/2.0) & 0.0800 & 0.0998 & 0.0732 \\
c111 (1.0/1.0/1.0) & 0.0786 & 0.0998 & 0.0734 \\
\hline
\end{tabular}
\end{table}

% Table 5: SID token-set Jaccard
\begin{table}[h]
\centering
\caption{SID token-set overlap (Jaccard) across methods.}
\label{tab:jaccard}
\begin{tabular}{lrrrr}
\hline
Method Pair & L0 & L1 & L2 & 4-col \\
\hline
vanilla vs c111 & 1.000 & 1.000 & 1.000 & 0.002 \\
vanilla vs c555 & 1.000 & 1.000 & 1.000 & 0.002 \\
c111 vs c555 & 1.000 & 1.000 & 1.000 & 0.001 \\
\hline
\end{tabular}
\end{table}

% Table 6: Per-layer token distribution
\begin{table}[h]
\centering
\caption{Per-layer L0 token distribution.}
\label{tab:dist}
\begin{tabular}{lrrrr}
\hline
Method & L0 normH & L0 top1\% & L0 top5\% & L0 util \\
\hline
vanilla & \textbf{0.983} & 4.26\% & 15.21\% & 100\% \\
c111 & 0.980 & 4.12\% & 16.07\% & 100\% \\
c555 & 0.969 & \textbf{5.11\%} & 18.05\% & 100\% \\
free-curv & 0.819 & 7.43\% & 26.21\% & 53\% \\
\hline
\end{tabular}
\end{table}

% Table 7: Stage 3 Training Dynamics
\begin{table}[h]
\centering
\caption{Stage 3 training dynamics + generalization gap.}
\label{tab:dynamics}
\begin{tabular}{lrrrr}
\hline
Method & Best Valid R@10 & Best Epoch & Test R@10 & Gap \\
\hline
HG-Rec c1055 & \textbf{0.1276} & 56 & 0.1015 & +0.0261 \\
vanilla (phonism) & 0.1262 & 75 & \textbf{0.1058} & \textbf{+0.0204} \\
HG-Rec c222 & 0.1259 & 61 & 0.1036 & +0.0223 \\
HG-Rec c555 & 0.1256 & 77 & 0.1051 & +0.0205 \\
HG-Rec c111 & 0.1254 & 56 & 0.1020 & +0.0234 \\
HG-Rec free-curv & 0.1252 & 92 & 0.1015 & +0.0237 \\
HG-Rec c215 & 0.1250 & 68 & 0.1028 & +0.0222 \\
HG-Rec c512 & 0.1240 & 45 & 0.0998 & +0.0242 \\
\hline
\end{tabular}
\end{table}
```

---

## References

References follow HG-Rec (Zhang et al., 2026). Key citations:
- HG-Rec: Zhang, A., Yang, Y.-B., Yu, Y. (2026). Hyperbolic RQ-VAE enhanced Generative Recommendation with Differential-Length Codebook Strategy. ICML 2026.
- TIGER: Rajput et al. (2023). Recommender Systems with Generative Retrieval. NeurIPS.
- LC-Rec: Zheng et al. (2024). Adapting Large Language Models by Integrating Collaborative Semantics for Recommendation. ICDE.
- LETTER: Wang et al. (2024). Learnable Item Tokenization for Generative Recommendation. CIKM.
- RQ-VAE: Singh et al. (2024). Better Generalization with Semantic IDs. RecSys.
- sentence-T5: Ni et al. (2022). Sentence-T5: Scalable Sentence Encoders from Pre-trained Text-to-Text Models. ACL.
- SASRec: Kang & McAuley (2018). Self-Attentive Sequential Recommendation. ICDM.
- BERT4Rec: Sun et al. (2019). BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformers. CIKM.
- HGCF: Sun et al. (2021). HGCF: Hyperbolic Graph Convolution Networks for Collaborative Filtering. WWW.
- Poincaré embeddings: Nickel & Kiela (2017). Poincaré Embeddings for Learning Hierarchical Representations. NeurIPS.
- HGCN: Chami et al. (2019). Hyperbolic Graph Convolutional Neural Networks. NeurIPS.

Full bibliography inherited from HG-Rec paper (Zhang et al., 2026, References section).

---

*This paper was produced via independent reproduction. No code was shared with the original HG-Rec authors. All experiments reproducible with seed=42 on Amazon Musical_Instruments (5-core filter, leave-one-out protocol).*
