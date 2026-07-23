# Task #96 — Paper Section 4 Related Work Draft (analytical writeup, no GPU)

> **任务目的**: 起草论文 Section 4 (Related Work) 完整 markdown 草稿, 复现视角扩展 HG-Rec 论文 §5 (Related Works, 行 268-272). 覆盖 5 个文献家族 + 1 个复现方法论小节. 接续 Task #95 Section 1 + Task #94 Section 3 + Task #92 Section 5 + Task #93 Section 6 形成 paper deliverable 完整 7-section 结构.

> **完成日期**: (in progress)
> **状态**: 🟡 待启动

---

## 1. 背景

论文 Section 1/3/5/6 已闭环. 缺失 Section 4 (Related Work). 它是 paper 的"文献定位"章节, 承担 4 个功能:
1. **梳理相关工作**: Sequential Rec / Generative Rec / Hyperbolic Rec / RQ-VAE 四大文献家族
2. **识别 gap**: HG-Rec 之前的方法都在 Euclidean 空间, 没有 hyperbolic RQ-VAE
3. **定位贡献**: 我们的复现与原始 HG-Rec 在文献中的位置
4. **复现方法论** (我们独有): 复现 hyperbolic geometry 假设的实验设计经验

`papers/HG-Rec.md` 行 268-272 是论文原始 §5 Related Works, 仅 2 段 (Sequential + Generative), 不够全面. 复现版本应扩展为 5 个子节.

## 2. Section 4 结构

### 4.1 Sequential Recommendation (传统 + 深度 + 自监督)
- Markov chain: FPMC (Rendle et al. 2010), PRME (Feng et al. 2015)
- Translation-based: TransRec (He et al. 2017)
- Deep learning: GRU4Rec (Hidasi et al. 2016), Caser (Tang & Wang 2018), SASRec (Kang & McAuley 2018), BERT4Rec (Sun et al. 2019)
- 自监督 / 对比学习: CL4SRec (Xie et al. 2021), ICLRec (Chen et al. 2022), DuoRec (Qiu et al. 2022), CT4Rec (Zhang et al. 2025b)
- 共同局限: 忽略 item 内容语义

### 4.2 Generative Recommendation
- ID-based tokenization: SemID, CID (Hua et al. 2023); RecSysLLM (Chu et al. 2023)
- Context-aware: ActionPiece (Hou et al. 2025), Pctx (Zhong et al. 2025)
- Codebook-based: TIGER (Rajput et al. 2023), LC-Rec (Zheng et al. 2024), LETTER (Wang et al. 2024), TokenRec (Qu et al. 2024)
- 集成 tokenization + fine-tuning: LC-Rec, LETTER, SIIT (Chen et al. 2024), GFlowGR (Wang et al. 2025)
- 共同 gap: 都在 Euclidean 空间做 tokenization, 没有 hyperbolic RQ-VAE

### 4.3 Hyperbolic Methods in Recommendation
- HGCF (Sun et al. 2021): hyperbolic GCN for collaborative filtering
- HyperML (Zhang et al. 2025a): hyperbolic metric learning
- M2GNN (multi-manifold GNN, Task #70 baseline): Riemannian 多流形
- MERLOT / hyperbolic image-text (Desai et al. 2023): hyperbolic multimodal
- HG-Rec (Zhang et al. 2026): 第一篇把 hyperbolic 引入 RQ-VAE 的工作
- **共同的 inductive bias**: 适合 hierarchy / 幂律分布

### 4.4 RQ-VAE and Codebook Techniques
- RQ-VAE 原始 (Singh et al. 2024): 用于 recommendation tokenization
- Sinkhorn-balanced codebook (Task #79 phonism 部分): post-processing 平衡利用率
- Entropy regularization (多篇 RQ-VAE 衍生)
- EMA-restart / dead code revival (Task #67/68)
- 我们的 5 重证据 (§5.5) 显示机制 (Sinkhorn) > 几何 (hyperbolic)

### 4.5 Reproduction Methodology for Geometric Priors (NEW)
- 复现 hyperbolic geometry 假设的标准流程 (基于本项目 90+ 任务经验):
  1. Stress-metric diagnostics (Ollivier 曲率 / δ-hyperbolicity)
  2. Per-layer curvature grid ablation
  3. Free-curvature 学习 (验证是否收敛到 0)
  4. Token-set Jaccard 分解 (隔离几何 vs 机制)
  5. Training dynamics 对比 (valid vs test ranking)
- 复现 checklist 4 条: 多数据集 / 多 embedding / 多 codebook size / multi-seed

## 3. 预算

| 阶段 | 估算时间 | GPU |
|------|---------|-----|
| 读 HG-Rec.md §5 + 相关文献索引 | ~10 min | 0 |
| Section 4 markdown 起草 | ~40 min | 0 |
| 4.5 复现方法论 (我们独有) | ~15 min | 0 |
| **总计** | **~65 min** | **0 GPU** |

## 4. 产物清单

- `verdicts/task96_paper_section4_related_work_draft.md` — Section 4 完整 markdown 草稿
- 含 5 子节 (4.1 Sequential Rec / 4.2 GR / 4.3 Hyperbolic Rec / 4.4 RQ-VAE / 4.5 Reproduction Methodology)

## 5. 关联

- 前置: Task #94 (Section 3) + Task #92 (Section 5) + Task #93 (Section 6) + Task #95 (Section 1)
- 后续: Section 7 Conclusion, 论文最终写作, Submission 准备

---

**核心交付**: 论文 Section 4 (Related Work) 完整 markdown 草稿, 5 子节覆盖 Sequential Rec / Generative Rec / Hyperbolic Rec / RQ-VAE / Reproduction Methodology (复现独有).
