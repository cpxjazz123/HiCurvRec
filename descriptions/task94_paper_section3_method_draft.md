# Task #94 — Paper Section 3 Method Draft (analytical writeup, no GPU)

> **任务目的**: 起草论文 Section 3 (Method) 完整 markdown 草稿, 复现 HG-Rec 论文方法三要素: Hyperbolic RQ-VAE / Differential-Length Codebook / T5 Generative Backbone. 接续 Task #92 Section 5 + Task #93 Section 6 形成完整 paper deliverable. 用于支撑 paper 复现版本的"方法"章节.

> **完成日期**: (in progress)
> **状态**: 🟡 待启动

---

## 1. 背景

论文复现版本已有:
- Task #92: Section 5 Experiments 完整 (含 27 baseline, 6 curvature grid, codebook decomp, training dynamics)
- Task #93: Section 6 Discussion + Limitations + Future Work

缺失: Section 3 Method 是 paper 的核心方法章节, 需复现 HG-Rec 论文的: (a) Hyperbolic RQ-VAE 量化器, (b) Differential-Length Codebook 分配, (c) T5 生成式训练/推断. 这是 paper 的"方法论"对应章节.

`papers/HG-Rec.md` 行 69-153 是论文 Section 2 (Preliminaries) + Section 3 (Method) 原文, 行 77/113/139/151 是四个 subsections 3.1/3.2/3.3/3.4 的分界.

## 2. Section 3 结构

### 3.1 Hyperbolic RQ-VAE (行 77-111)
- 上下文: 0-th layer 残差 $r_0 = z$ 在切空间
- 公式 (5): exponential map 把残差与码字都映射到 Poincaré ball
- 公式 (6): argmin hyperbolic distance
- 公式 (8): logarithmic map 把 hyperbolic 量投影回切空间, 在切空间里算 residual update
- Theorem 3.1: RQ-VAE 诱导 K-ary tree graph isomorphism (行 67-69)
- Theorem 3.2: exponential map well-defined (行 110)

### 3.2 Differential-Length Codebook (行 113-137)
- 问题: [256,256,256] 总容量 ~10^8 ≫ 真实数据集 ~10^5 → 利用率 1‰
- Theorem 3.4: 庞加莱球体积 $V_{\mathbb{B}}(\rho) \sim e^{(n-1)\sqrt{c}\rho}$ (行 117-127)
- 公式 (11): $K_\ell = K_1 \cdot e^{(n-1)\sqrt{c}\rho}$
- 公式 (12): 化简为 $K_\ell = K_1 \cdot \gamma^\ell$, $\gamma = e^{(n-1)\sqrt{c}\Delta\rho}$
- 实践中取 $\gamma \approx 2$, $K_1 \in \{16, 32, 64\}$, 总层 L=3 → [K1, 2K1, 4K1]

### 3.3 Model Training and Inference (行 139-149)
- 训练: Transformer encoder-decoder (Raffel et al. 2020) autoregressive over tokens
- 公式 (13): 负对数似然 $\mathcal{L} = -\sum_{u,t} \log p(C_{u,t}^{out} | C_{u,<t}^{out}, C_u^{in})$
- 推断: decoder + beam search over discrete indices

### 3.4 Discussion (行 151-153)
- 解析"双曲 vs 欧氏"几何差异: 欧氏多项式增长 → 拥挤效应; 双曲指数增长 → 天然适配 hierarchy
- 两个组件无冲突: hyperbolic RQ-VAE 改的是"latent 空间", differential-length codebook 改的是"capacity 分配"

## 3. 预算

| 阶段 | 估算时间 | GPU |
|------|---------|-----|
| 阅读 HG-Rec.md + .json | ~10 min | 0 |
| Section 3 markdown 起草 | ~30 min | 0 |
| LaTeX 公式 + Theorem 引用 | ~10 min | 0 |
| **总计** | **~50 min** | **0 GPU** |

## 4. 产物清单

- `verdicts/task94_paper_section3_method_draft.md` — Section 3 Method 完整 markdown 草稿
- 含 4 子节 (3.1 Hyperbolic RQ-VAE / 3.2 Differential-Length Codebook / 3.3 Training+Inference / 3.4 Discussion)
- 含 4 个公式 (5/6/8/11/12/13) 完整 LaTeX
- 含 Theorem 3.1/3.2/3.4 的 statement (proof 在附录, 不需展开)

## 5. 关联

- 前置: Task #84/87/88/89/90/91/92/93/95 (全部闭环)
- 后续: Section 4 Related Work + Section 1 Introduction 起草, 论文最终写作, Submission 准备

---

**核心交付**: 论文 Section 3 Method 完整 markdown 草稿, 复现 HG-Rec 论文方法三要素 (含 LaTeX 公式 + Theorem 引用).
