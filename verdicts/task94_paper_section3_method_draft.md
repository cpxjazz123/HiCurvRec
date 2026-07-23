# Task #94 — Paper Section 3 Method Draft (Hyperbolic RQ-VAE + Differential-Length Codebook + T5)

> **完成日期**: 2026-07-24
> **状态**: ✅ 闭环 (writeup only, 0 GPU)
> **核心交付**: 论文 Section 3 完整 markdown 草稿, 复现 HG-Rec Method 三要素 + Discussion, 含 6 个核心公式的 LaTeX 形式 + Theorem 3.1/3.2/3.4 statement.

---

## 1. Section 3 (Method) — 完整 Markdown 草稿

### 3.1 Hyperbolic RQ-VAE

The overall framework of HG-Rec, depicted in Figure 3, rests on three sequential stages:
(1) encoding items into semantic embeddings via a frozen sentence-T5 encoder;
(2) quantizing these embeddings into hierarchical codeword tuples via **hyperbolic residual vector quantization**;
(3) training a Transformer encoder-decoder to autoregressively generate the next item's token tuple from the user's interaction history.

We focus on (2) — the hyperbolic RQ-VAE — in this section, before deferring the capacity-allocation question to Section 3.2.

#### 3.1.1 Latent semantic embedding extraction

Given an item and its textual content (title, brand, category, description), we extract a semantic embedding $\mathbf{s} \in \mathbb{R}^{768}$ using the pre-trained sentence-T5 encoder (Ni et al., 2022), kept frozen throughout training. An encoder network $E_\phi: \mathbb{R}^{768} \to \mathbb{R}^n$ then compresses $\mathbf{s}$ into a low-dimensional latent code $\mathbf{z} \in \mathbb{R}^n$. This code lives in the tangent space $T_o^{\,c}\mathbb{B}^n_c$ at the origin of the Poincaré ball, which is isomorphic to Euclidean $\mathbb{R}^n$. The latent dimension $n = 32$ is used throughout our experiments.

#### 3.1.2 Hyperbolic residual quantization

For each quantization layer $\ell \in \{1, 2, \ldots, L\}$ we maintain a learnable codebook $\mathcal{C}^\ell = \{e^\ell_1, \ldots, e^\ell_{K_\ell}\} \subset \mathbb{R}^n$, where $K_\ell$ is the layer-dependent codebook size (Section 3.2) and each $e^\ell_i$ is a learnable codeword embedding.

We initialize the residual as $r_0 := \mathbf{z}$. For every layer $\ell$, we compute the hyperbolic representation of both the residual and the codewords:

$$
\mathbf{r}_{\ell-1}^{\mathbb{B}} = \exp^{\,c}_o(\mathbf{r}_{\ell-1}), \quad
\mathbf{e}^{\mathbb{B}}_{\ell,i} = \exp^{\,c}_o(\mathbf{e}_{\ell,i}), \tag{5}
$$

where $\exp^{\,c}_o(\cdot)$ is the exponential map at the origin of the Poincaré ball (Eq. 3). The selected codeword index is the one that minimizes the hyperbolic distance (Eq. 1):

$$
c_\ell = \arg\min_{i \in \{1,\ldots,K_\ell\}} d_{\mathbb{B}}\bigl(\mathbf{r}_{\ell-1}^{\mathbb{B}}, \mathbf{e}^{\mathbb{B}}_{\ell,i}\bigr). \tag{6}
$$

#### 3.1.3 Residual updates in the tangent space (stability fix)

Direct subtraction on the Poincaré ball is unstable when $\mathbf{r}_{\ell-1}^{\mathbb{B}}$ lies near the boundary (where $\|x\|^2 \to 1/c$). HG-Rec therefore computes the residual update **in the tangent space** rather than the manifold. Concretely, after $c_\ell$ is selected we map both the residual and the chosen codeword back via

$$
\mathbf{r}_{\ell-1} = \log^{\,c}_o(\mathbf{r}_{\ell-1}^{\mathbb{B}}), \quad
\mathbf{e}_{\ell, c_\ell} = \log^{\,c}_o(\mathbf{e}^{\mathbb{B}}_{\ell, c_\ell}), \tag{7}
$$

and obtain the next-layer residual through simple Euclidean subtraction in the tangent space:

$$
\mathbf{r}_\ell = \mathbf{r}_{\ell-1} - \mathbf{e}_{\ell, c_\ell}. \tag{8}
$$

Theorem 3.1 *(HG-Rec paper; reproduced verbatim)*. Assuming RQ-VAE employs $L$ layers of residual quantization with each layer having $K$ codewords, the residual-quantization process induces a hierarchical structure on $\mathbb{R}^n$ that is graph-isomorphic to a rooted $K$-ary tree of depth $L$. *(Proof: Appendix B.1 of the original paper.)*

Theorem 3.2. The exponential map $\exp^{\,c}_o: T^{\,c}_o\mathbb{B}^n_c \to \mathbb{B}^n_c$ is well-defined for all $v$ with $\|v\| < 1/\sqrt{c}$. *(Proof: Appendix B.2.)*

> **Theoretical justification**: this design decouples similarity evaluation (performed in hyperbolic space, to capture the intrinsic tree-like hierarchy of Theorem 3.1) from residual update (performed in the tangent space, where standard backpropagation remains numerically stable). The empirical advantage is that HG-Rec reaches its minimum collision rate in only $\sim$165 epochs on Beauty, versus 9740 for vanilla RQ-VAE (Section 4.3.3 of the original paper).

---

### 3.2 Differential-Length Codebook Strategy

#### 3.2.1 Motivation: the underutilization problem

Traditional codebook sizes $[256, 256, 256]$ give a theoretical representational capacity of $256^3 \approx 1.7 \times 10^7$ items, whereas real-world recommendation datasets typically contain only $\sim 10^5$ items. This implies an *upper-bound codebook utilization of roughly 1\%* — even with zero collisions, only a tiny fraction of codewords will ever be activated. Beyond raw utilization, residual quantization is **coarse-to-fine**: layer-1 codewords span broad semantic categories, while layer-$L$ codewords encode fine-grained variations. Fine-grained representations therefore demand substantially larger representational capacity than coarse-grained ones — but traditional $[256, 256, 256]$ gives them equal capacity, structurally mismatching the data.

#### 3.2.2 Volume growth in the Poincaré ball (Theorem 3.4)

Theorem 3.4 *(HG-Rec paper; reproduced verbatim)*. The geometric capacity of hyperbolic space grows exponentially with geodesic radius.

*Proof sketch.* The volume element on the Poincaré ball is
$$
dV_{\mathbb{B}}(x) = \left(\frac{2}{1 - c\|x\|^2}\right)^{\!n} dx.
$$
Integrating in polar coordinates gives
$$
Vol_{\mathbb{B}}(\rho) = \omega_{n-1} \int_0^\rho \bigl(\sinh(\sqrt{c}\, t)\bigr)^{n-1} dt, \tag{10}
$$
where $\omega_{n-1}$ is the surface-area constant of the unit $(n{-}1)$-sphere. For large $t$, $\sinh(\sqrt{c}\, t) \sim \tfrac{1}{2}e^{\sqrt{c}\, t}$, hence $(\sinh(\sqrt{c}\, t))^{n-1} \sim \bigl(\tfrac{1}{2}\bigr)^{n-1} e^{(n-1)\sqrt{c}\, t}$. Therefore $Vol_{\mathbb{B}}(\rho) \sim e^{(n-1)\sqrt{c}\,\rho}$. $\blacksquare$

#### 3.2.3 Layer-wise exponential capacity schedule

Theorem 3.4 motivates assigning a *growing* number of codewords per layer:

$$
K_\ell = K_1 \cdot e^{(n-1)\sqrt{c}\,\rho}, \tag{11}
$$

where $K_1$ is the first-layer codebook size and $\rho$ is the geodesic radius. Setting equal radius increments $\Delta\rho$ between adjacent layers, $\rho_\ell = \rho_0 + \ell\,\Delta\rho$, and absorbing the constants into a *growth rate* $\gamma \triangleq e^{(n-1)\sqrt{c}\,\Delta\rho}$, Eq. (11) simplifies to

$$
K_\ell = K_1 \cdot \gamma^\ell. \tag{12}
$$

In practice, to keep the outermost codebook at a tractable size, HG-Rec enforces $K_L \le 256$ and chooses $\gamma \approx 2$, $K_1 \in \{16, 32, 64\}$. For our experiments we use $K_1 = 64$, $L = 3$, $\gamma = 2$, giving the codebook sizes $[64, 128, 256]$ used in ablation Table 3 and reproduced in our Stage 2 RQ-VAE training.

> **Connection to Theorem 3.1**. Because the codebook follows an exponential schedule of Theorem 3.4, the corresponding K-ary tree (Theorem 3.1) becomes a *geometrically widening* tree rather than a balanced one — early branches are few (coarse categories) and deep branches fan out exponentially (fine distinctions). This is the precise sense in which the codebook "aligns with the tree-like structure."

---

### 3.3 Model Training and Inference

#### 3.3.1 Stage 1–2: Tokenization (offline)

Given the trained hyperbolic RQ-VAE encoder $E_\phi$ and codebooks $\{\mathcal{C}^\ell\}_{\ell=1}^{L}$, every item $i \in \mathcal{I}$ is deterministically mapped to a token tuple $\mathbf{c}(i) = (c_1, c_2, \ldots, c_L) \in \prod_\ell \{1,\ldots,K_\ell\}$. The resulting item sequences $\mathcal{A} = \{S_1, \ldots, S_{|\mathcal{U}|}\}$ are converted once offline into token sequences $\mathcal{C} = \{C_1, \ldots, C_{|\mathcal{U}|}\}$ via $\mathbf{c}(\cdot)$, and these token sequences are used both for training and inference of the downstream generative model. Note: in our pipeline the codebook sizes are $[64, 128, 256]$ giving 3-level tokens, but at inference we append a 4th *deduplication* digit (Section 5.5) when collisions appear, yielding final 4-token SIDs.

#### 3.3.2 Stage 3: Generative model training

The downstream generative model is a **Transformer encoder-decoder** (Raffel et al., 2020). For a user $u$ with token history $C^{\mathrm{in}}_u$, we train it to autoregressively generate the next-item token tuple $C^{\mathrm{out}}_u$ by minimizing the negative log-likelihood loss

$$
\mathcal{L} = - \sum_{u=1}^{|\mathcal{U}|}\;\sum_{t=1}^{|C^{\mathrm{out}}_u|} \log\, p\bigl(C^{\mathrm{out}}_{u,t} \,\big|\, C^{\mathrm{out}}_{u,<t},\, C^{\mathrm{in}}_u\bigr). \tag{13}
$$

Concretely, $C^{\mathrm{in}}_u$ is fed into the T5 encoder; the decoder then produces the next item's tokens left-to-right. The vocabulary consists of all $K_1 + K_2 + K_3$ codeword indices plus the necessary special tokens (PAD, EOS, SEP). We use T5-small (6 encoder layers, 4 decoder layers, $d_{\text{model}}=128$, $d_{\text{ff}}=512$, 4 attention heads) following TIGER (Rajput et al., 2023).

#### 3.3.3 Stage 4: Inference via beam search

At inference time, the decoder autoregressively generates the target token tuple using **beam search** (Rajput et al., 2023), then resolves the generated code $(c_1, c_2, c_3, c_4)$ back to the recommended item via a fixed item-code lookup table. The top-$N$ candidates (here $N \in \{5, 10, 20\}$) are output as the recommendation list. When the same item has multiple code tuples (a possibility under dedup), the lookup is performed by canonical minimum-code matching.

---

### 3.4 Discussion

The two components of HG-Rec — *hyperbolic RQ-VAE* (Section 3.1) and *differential-length codebook* (Section 3.2) — operate on orthogonal axes:

- **Hyperbolic RQ-VAE** modifies the *latent space* in which quantization takes place, replacing the Euclidean distance metric with the Poincaré geodesic distance. The benefit is the alignment between the tree-like hierarchy of Theorem 3.1 and the exponential volume growth of Theorem 3.4.
- **Differential-length codebook** modifies the *capacity allocation* across layers, replacing the uniform $[K, K, K]$ schedule with the exponential $K_1 \cdot \gamma^\ell$ schedule. The benefit is that fine-grained layers receive proportionally more capacity than coarse-grained layers.

Crucially, the two components do not conflict: the exponential schedule of Eq. (12) is itself derived from Theorem 3.4, so the capacity allocation implicitly assumes a hyperbolic embedding. In the ablation Table 3 of the original paper, removing the hyperbolic RQ-VAE (w/o H) but keeping the differential-length codebook still yields $\text{NDCG@10} = 0.0905$ on Instruments (vs $0.0879$ for traditional $[128, 128, 128]$ codebook); removing the differential-length codebook (w/o D) but keeping hyperbolic RQ-VAE yields $0.0931$; the full HG-Rec achieves $0.0945$. The two effects compound.

**Why we did not observe the full HG-Rec advantage on our Musical_Instruments reproduction.** In our reproduction on Amazon Musical_Instruments (9922 items, 511836 interactions), the vanilla RQ-VAE + Sinkhorn-balanced codebook achieves test R@10 = **0.1058**, matching or exceeding the reproduced HG-Rec variants (c555: 0.1051, c111: 0.1020). We attribute this to two factors inherent to the dataset:

1. **Musical_Instruments is not strongly hierarchical** (Section 6.2 of our paper): items are semantically related but not taxonomically nested. The benefit of hyperbolic geometry in Theorem 3.4 then vanishes.
2. **Codebook utilization can be enforced by other means**: Sinkhorn-balanced k-means (our vanilla baseline) achieves 100\% utilization by construction, replicating the role played by differential-length codebook but without geometric structure.

These observations motivate the per-layer curvature ablation of Section 5.4, the codebook decomposition analysis of Section 5.5, and the training-dynamics study of Section 5.6.

---

## 2. 表格与公式索引

### 公式编号索引 (与 HG-Rec 论文对齐)

| Section | HG-Rec eq. | 内容 |
|---------|-----------|------|
| 3.1.2 | (5) | exponential map for residual + codewords |
| 3.1.2 | (6) | argmin hyperbolic distance |
| 3.1.3 | (7) | logarithmic map back |
| 3.1.3 | (8) | tangent-space residual update |
| 3.2.2 | (9) | Poincaré ball volume element |
| 3.2.2 | (10) | integrated volume |
| 3.2.3 | (11) | layer-wise exponential capacity |
| 3.2.3 | (12) | growth-rate simplification $K_\ell = K_1 \gamma^\ell$ |
| 3.3.2 | (13) | NLL training objective |

### Theorem 索引

| Theorem | 内容 | 草稿位置 |
|---------|------|----------|
| 3.1 | RQ-VAE induces K-ary tree (graph isomorphic) | 3.1.3 |
| 3.2 | exp map well-defined | 3.1.3 |
| 3.4 | volume grows exponentially | 3.2.2 |

(Theorem 3.3 在 HG-Rec paper 也是 tangent-space map stability, 这里引用 3.2 已涵盖同样内容.)

---

## 3. 与论文其他章节的衔接

- **上游 (Section 2 Preliminaries)**: Section 3.1 引用了 Section 2.2 的 Poincaré 距离公式 (1), Möbius 加法 (2), exponential map (3), logarithmic map (4). 这些公式在 HG-Rec 论文中位于 Section 2.2.
- **下游 (Section 4 Experiments)**: Task #92 Section 5 即对应 HG-Rec 论文 Section 4 的复现 (含 Table 2 ranking, 6 curvature grid, codebook decomp, training dynamics).
- **下游 (Section 5 Discussion)**: Task #93 Section 6 包含 4 项 Practical Recommendations, 与本节末段"为何 Musical_Instruments 没观察到 HG-Rec 优势"自然呼应.

---

## 4. 产物清单

- `verdicts/task94_paper_section3_method_draft.md` — 本草稿 (Section 3)
- `descriptions/task94_paper_section3_method_draft.md` — 任务描述

---

## 5. 复现完整性核验

| 维度 | 论文原 Section | 本草稿 | 状态 |
|------|---------------|--------|------|
| 3.1 Hyperbolic RQ-VAE | 论文 77-111 行 | 3.1.1-3.1.3 + Theorem 3.1, 3.2 | ✅ 公式 (5)(6)(7)(8) + 优化稳定性论证 |
| 3.2 Differential-Length Codebook | 论文 113-137 行 | 3.2.1-3.2.3 + Theorem 3.4 | ✅ 公式 (10)(11)(12) + γ≈2 实践选参 |
| 3.3 Training & Inference | 论文 139-149 行 | 3.3.1-3.3.3 | ✅ 公式 (13) + T5-small 配置 + beam search |
| 3.4 Discussion | 论文 151-153 行 | 3.4 | ✅ 两组件正交轴 + Musical_Instruments 实证缺位解释 |
| Theorem 引用 | 论文 67/110/117 | Thm 3.1 / 3.2 / 3.4 | ✅ (Thm 3.3 合并到 3.2 同义) |

---

**核心交付**: 论文 Section 3 (Method) 完整 markdown 草稿, 4 子节覆盖 HG-Rec 方法三要素 + Discussion, 含 9 个核心公式的 LaTeX 形式 + Theorem 3.1/3.2/3.4 statement. 接续 Task #92 Section 5 + Task #93 Section 6 形成完整 paper deliverable.

result: Task #94 — Paper Section 3 Method Draft 完成. 完整 markdown 草稿覆盖 4 子节 (Hyperbolic RQ-VAE 公式 5-8 + Theorem 3.1/3.2 / Differential-Length Codebook 公式 10-12 + Theorem 3.4 / Training+Inference 公式 13 / Discussion 两组件正交性). 与 Task #92 Section 5 + Task #93 Section 6 形成完整 paper 闭环. 草稿可直接 copy 到论文写作.
