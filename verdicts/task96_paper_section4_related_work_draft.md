# Task #96 — Paper Section 4 Related Work Draft (5-Subsection, Reproduction-Focused)

> **完成日期**: 2026-07-24
> **状态**: ✅ 闭环 (writeup only, 0 GPU)
> **核心交付**: 论文 Section 4 (Related Work) 完整 markdown 草稿, 5 子节覆盖 Sequential Recommendation / Generative Recommendation / Hyperbolic Methods in Rec / RQ-VAE and Codebook Techniques / Reproduction Methodology (复现独有). 与 HG-Rec 论文 §5 (2 段简短版) 互补 (复现版扩展为 5 子节 + 1 复现方法论小节).

---

## 1. Section 4 (Related Work) — 完整 Markdown 草稿

### 4.1 Sequential Recommendation

Sequential recommendation aims to predict the next item a user will interact with given their chronologically ordered history $S_u = (i_1, i_2, \ldots, i_t)$. Three generations of methods have emerged.

**Markov-chain and translation-based methods.** FPMC (Rendle et al., 2010) combines matrix factorization with first-order Markov chains to model personal transitions. PRME (Feng et al., 2015) embeds users and items in a metric space and uses personalized ranking for POI recommendation. TransRec (He et al., 2017) treats users as translation operators between items. These methods scale linearly with sequence length and struggle to capture long-range dependencies.

**Deep-learning methods.** GRU4Rec (Hidasi et al., 2016) pioneered RNN-based session recommendation. Caser (Tang & Wang, 2018) embeds the recent item history as a 2-D "image" and applies convolutional filters. SASRec (Kang & McAuley, 2018) introduced a single-head self-attentive model that captures long-range item-item correlations with $\mathcal{O}(L^2)$ attention. BERT4Rec (Sun et al., 2019) extended this to bidirectional masked-item prediction with the cloze objective. SelfGNN (Liu et al., 2024) further augments sequence models with self-supervised graph signals. These models substantially improved accuracy but operate on **discrete item IDs**, ignoring the rich semantic content of items.

**Contrastive and self-supervised methods.** CL4SRec (Xie et al., 2021), ICLRec (Chen et al., 2022), DuoRec (Qiu et al., 2022), and CT4Rec (Zhang et al., 2025b) augment sequence data with item cropping, masking, reordering, or contrastive views to alleviate data sparsity. They improve generalization but, like their supervised counterparts, do not leverage item semantics beyond IDs.

**Common limitation.** Across all three generations, sequential recommenders treat items as opaque IDs and do not exploit textual content. This is the gap that generative-recommendation methods (Section 4.2) aim to fill.

---

### 4.2 Generative Recommendation

Generative recommendation (GR) reformulates next-item prediction as sequence-to-sequence token generation, leveraging the pretrained semantic knowledge of LLMs. The crucial design choice is **how to tokenize items**.

**ID-based tokenization.** SemID and CID (Hua et al., 2023) encode items via collaborative signals or semantic similarity into discrete IDs. RecSysLLM (Chu et al., 2023) introduces a masking mechanism for token sequences to inject entity knowledge. These methods produce compact IDs but provide no semantic structure between tokens.

**Context-aware tokenization.** ActionPiece (Hou et al., 2025) and Pctx (Zhong et al., 2025) tokenize actions or user contexts dynamically based on sequence history. They produce highly expressive token sequences but introduce context-dependent inference cost.

**Codebook-based tokenization.** TIGER (Rajput et al., 2023), LC-Rec (Zheng et al., 2024), LETTER (Wang et al., 2024), and TokenRec (Qu et al., 2024) use **RQ-VAE** (Section 4.4) to encode each item into a hierarchical codeword tuple $(c_1, c_2, c_3)$ from three codebook layers. The hierarchical structure naturally aligns with autoregressive generation. LC-Rec, LETTER, SIIT (Chen et al., 2024), and GFlowGR (Wang et al., 2025) further integrate tokenization with end-to-end fine-tuning, allowing the LLM to adapt to recommendation-specific signals.

**Common limitation of existing GR methods.** All codebook-based tokenizers (TIGER, LC-Rec, LETTER, TokenRec, SIIT, GFlowGR) operate in **Euclidean space**. They induce a tree-like quantization structure (Theorem 3.1) but lose the natural fit of hyperbolic geometry for hierarchies. This limitation motivates HG-Rec (Section 4.3).

---

### 4.3 Hyperbolic Methods in Recommendation

Hyperbolic space $\mathbb{B}^n_c$ has been proposed as an alternative to Euclidean space for tasks involving hierarchical or tree-like structures, due to its **exponential volume growth** with geodesic radius (Theorem 3.4).

**Hyperbolic collaborative filtering.** HGCF (Sun et al., 2021) pioneered hyperbolic graph convolution networks for collaborative filtering, demonstrating that user-item graphs benefit from hyperbolic embeddings. HGA (Zhang et al., 2025a) extends this with adversarial learning in hyperbolic space. These works focus on the **interaction graph** rather than the **tokenization latent space**.

**Hyperbolic metric learning.** HyperML and similar approaches embed users and items in the Poincaré ball for top-N recommendation. They typically use a fixed global curvature.

**Hyperbolic GNNs.** HGCN (Chami et al., 2019) and HAT (Liu et al., 2019) define graph-convolution operations natively in hyperbolic space via tangent-space message passing.

**Multi-manifold recommendation.** M2GNN and similar multi-component Riemannian methods (cf. Task #70 baseline) learn multiple geometries simultaneously. Our free-curvature ablation (Task #89) belongs to this family but is applied to RQ-VAE tokenization.

**Hyperbolic multimodal representation.** Hyperbolic image-text models (Desai et al., 2023; Zhang et al., 2025d) embed paired modalities in Poincaré space, exploiting tree-like taxonomies (e.g., WordNet).

**HG-Rec (Zhang et al., 2026) is the first work to apply hyperbolic geometry to the RQ-VAE tokenization step of generative recommendation.** It introduces both hyperbolic residual quantization (Section 3.1) and a differential-length codebook (Section 3.2) that exploits Theorem 3.4. The paper reports 4.8–13.5% improvement over ActionPiece on Beauty / Instruments / Yelp.

---

### 4.4 RQ-VAE and Codebook Techniques

RQ-VAE (Residual Quantization Variational Autoencoder; Singh et al., 2024) recursively quantizes a latent code $\mathbf{z}$ into a sequence of codeword indices $(c_1, c_2, \ldots, c_L)$ by subtracting the nearest codeword at each layer. The structure is provably graph-isomorphic to a K-ary tree of depth L (Theorem 3.1 of HG-Rec). The original RQ-VAE formulation uses **Euclidean nearest-neighbor lookup** at each layer.

**Tokenization applications.** TIGER (Rajput et al., 2023) applied RQ-VAE to recommendation tokenization, enabling T5-small generation of item token tuples. LC-Rec (Zheng et al., 2024) added collaborative fine-tuning. LETTER (Wang et al., 2024) introduced a learnable codebook + embedding lookup for end-to-end fine-tuning. TokenRec (Qu et al., 2024) added task-specific tokenization losses.

**Codebook utilization techniques.** Standard RQ-VAE often exhibits low codebook utilization (5–20% in TIGER/LC-Rec/LETTER), where some codewords are never activated. Three families of remedies exist:

- **Sinkhorn-balanced post-processing** (used in our *phonism* baseline): iteratively rebalances the assignment matrix to enforce uniform marginal mass on every codeword. Achieves 100% utilization by construction.
- **Entropy regularization** (multiple RQ-VAE variants): adds an entropy term to the VQ loss encouraging uniform codeword usage.
- **EMA + dead code revival** (e.g., Task #67 / Task #68 in our project): periodically reinitializes unactivated codewords via exponential moving average of incoming vectors.

**Hyperbolic RQ-VAE** (HG-Rec, Section 3.1) replaces Euclidean lookup with hyperbolic-distance lookup, claiming 100% utilization by geometric construction. Our codebook decomposition analysis (Task #90, Section 5.5) shows that on Musical_Instruments, vanilla + Sinkhorn matches hyperbolic RQ-VAE in utilization, suggesting **the mechanism (Sinkhorn) is more impactful than the geometry (hyperbolic)** for codebook balance.

---

### 4.5 Reproduction Methodology for Geometric Priors

Beyond algorithmic novelty, **reproducing geometric-prior claims** (e.g., "hyperbolic geometry helps recommendation") requires its own methodology. Based on our 90+ reproduction tasks across this project, we distill the following five-step protocol that we believe should be standard practice.

**Step 1: Stress-metric diagnostics.** Before adopting any geometric prior, compute Ollivier curvature $\kappa_e$ on the empirical item-item similarity graph and δ-hyperbolicity on the latent embeddings. If $|\kappa_e| < 0.1$ across $>99\%$ of edges (as on Musical_Instruments, Task #70 / #117), the data is effectively Euclidean and hyperbolic priors are unlikely to help.

**Step 2: Per-layer curvature grid.** Sweep $\kappa$ over a meaningful grid (e.g., $\{0, 0.5, 1, 2\}$) for each codebook layer independently. If the optimal $\kappa$ is at or near the boundary of the grid, extend the grid. If the span of test R@10 across the grid is within single-seed noise ($<5\%$), the curvature choice is not critical.

**Step 3: Free-curvature learning.** Initialize $\kappa_m = \kappa_{\max} \cdot \tanh(\theta_m)$ with $\theta_m = 0$ (giving $\kappa_m = 0$) and let $\theta_m$ be learned. If all $\kappa_m$ converge to $0$ from any initialization (as in Task #89), the data does not benefit from any non-zero curvature. This is the **strongest** evidence because it removes the hyperparameter bias.

**Step 4: Token-set Jaccard decomposition.** Compare the sets of activated codewords (per layer) across variants (vanilla / low-κ / high-κ). If Jaccard = 1.000 (as in Task #90), the variants share the same codebook vocabulary and any test difference must come from the assignment of items to codewords, not from the vocabulary itself. This isolates the **mechanism vs geometry** question.

**Step 5: Training dynamics comparison.** Compare valid-set vs test-set R@10 across configurations. If the valid ranking inverts the test ranking (as in Task #91), report test R@10 as the primary metric and explicitly flag the valid-test gap as a measure of overfitting.

**Reproduction checklist** (recommendations for future reproductions of geometric-prior papers):
1. **Multiple datasets**: validate on at least 3 datasets with differing hierarchical structure (e.g., DBpedia, Amazon-Taxonomy, Amazon-Musical_Instruments).
2. **Multiple embeddings**: try sentence-T5, flan-T5, BGE, and ideally a domain-specific embedding.
3. **Multiple codebook sizes**: sweep $[32, 64, 128]$, $[64, 128, 256]$, $[128, 256, 512]$.
4. **Multiple seeds**: although restricted in our project per [[user-no-multiseed-override]], multi-seed evaluation remains the gold standard for noise quantification.

We apply this protocol to HG-Rec's hyperbolic claim in Sections 5.3–5.6 and reach the conclusion that, **on Musical_Instruments, the geometric prior is marginal** — see Section 6.1 for the synthesis.

---

## 2. 与论文其他章节的衔接

- **上游 (Section 1 Introduction)**: Task #95 §1.1-1.2 引用了本节的 Sequential Rec / GR / Hyperbolic Methods 三大类.
- **上游 (Section 3 Method)**: Task #94 §3.1-3.3 是本节 HG-Rec 描述的工程实现.
- **下游 (Section 5 Experiments)**: Task #92 §5.3-5.7 实验是本节 §4.4 RQ-VAE 与 §4.5 复现方法论的应用.
- **下游 (Section 6 Discussion)**: Task #93 §6.1 Theoretical Implications 引用本节 §4.5 五步协议作为标准.

## 3. 与 HG-Rec 论文 §5 Related Works 的关键差异

| 维度 | HG-Rec 论文 | 我们的复现版本 |
|------|------------|---------------|
| Sequential Rec | 1 段 (传统 + 深度 + 对比) | 4 段 (Markov / Trans / 深度 / 对比) 更细 |
| Generative Rec | 1 段 (ID / context / codebook) | 同结构, 加 4 个 tokenization + fine-tuning 集成 |
| Hyperbolic Methods | 无独立子节 (合并到 §1 引言) | **4.3 独立子节** (HGCF / HyperML / HGCN / M2GNN / 多模态) |
| RQ-VAE / Codebook | 无独立子节 (合并到 §2 Preliminaries) | **4.4 独立子节** (RQ-VAE 应用 + 3 类利用率技术) |
| Reproduction Methodology | 无 | **4.5 五步协议 + 复现 checklist 4 条** |

## 4. 产物清单

- `verdicts/task96_paper_section4_related_work_draft.md` — 本草稿 (Section 4)
- `descriptions/task96_paper_section4_related_work_draft.md` — 任务描述

## 5. 复现完整性核验

| 维度 | HG-Rec 论文 §5 | 本草稿 | 状态 |
|------|---------------|--------|------|
| Sequential Recommendation | 1 段 (合并) | §4.1 4 段细分 (Markov / Trans / 深度 / 对比) | ✅ 扩展 |
| Generative Recommendation | 1 段 | §4.2 3 类 + 集成 tokenization 4 篇 | ✅ 扩展 |
| Hyperbolic Methods | 无独立 | §4.3 5 个子家族 (HGCF/HyperML/HGCN/M2GNN/多模态) | ✅ 独立 |
| RQ-VAE / Codebook | 无独立 | §4.4 RQ-VAE + 3 类利用率技术 + Sinkhorn vs 双曲对比 | ✅ 独立 |
| Reproduction Methodology | 无 | §4.5 五步协议 + 4 条 checklist | ✅ 复现独有 |

---

**核心交付**: 论文 Section 4 (Related Work) 完整 markdown 草稿, 5 子节覆盖 Sequential Rec / Generative Rec / Hyperbolic Methods in Rec / RQ-VAE and Codebook / Reproduction Methodology (复现独有). 与 HG-Rec 论文 §5 互补 (复现版扩展为 5 子节, 含 4.3/4.4 独立子节 + 4.5 复现方法论).

result: Task #96 — Paper Section 4 Related Work Draft 完成. 完整 markdown 草稿覆盖 5 子节 (Sequential Rec / GR / Hyperbolic Rec / RQ-VAE / Reproduction Methodology). 复现视角独有 4.5 节 (五步协议 + 4 条 checklist). 接续 Task #95 Section 1 + Task #94 Section 3 + Task #92 Section 5 + Task #93 Section 6 形成完整 paper deliverable. 论文 7-section 结构还差 Section 7 Conclusion.
