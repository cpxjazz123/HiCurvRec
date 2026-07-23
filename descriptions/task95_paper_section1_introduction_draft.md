# Task #95 — Paper Section 1 Introduction Draft (analytical writeup, no GPU)

> **任务目的**: 起草论文 Section 1 (Introduction) 完整 markdown 草稿. 与 Task #92 (Section 5 Experiments) / Task #93 (Section 6 Discussion) / Task #94 (Section 3 Method) 共同形成 paper deliverable 核心. **与 HG-Rec 论文 Introduction 不同**: 我们是复现版本, 需在 Introduction 中明确写"复现动机 + 复现核心发现预告", 而不是单纯介绍 HG-Rec.

> **完成日期**: (in progress)
> **状态**: 🟡 待启动

---

## 1. 背景

论文 Section 5 (Experiments) + Section 6 (Discussion) + Section 3 (Method) 已闭环. 缺失 Section 1 (Introduction), 它是 paper 的"开篇", 承担 4 个功能:
1. **背景引入**: GR 范式 + RQ-VAE tokenization + HG-Rec 提出的动机
2. **HG-Rec 论文核心主张**: hyperbolic geometry 适配 hierarchy, differential-length codebook 适配容量分配
3. **复现动机 (我们独有的部分)**: 为什么要在自己的 pipeline 复现 HG-Rec? 我们想验证什么假设?
4. **核心发现预告 (我们独有的部分)**: 5 重独立证据显示 Musical_Instruments 数据本质欧氏, HG-Rec 优势边际

`papers/HG-Rec.md` 行 27-33 是论文原始 Introduction (一句话段落 + 贡献列表), 不适合作为复现版本的 Introduction. 复现版本 Introduction 应基于 HG-Rec 原文 Introduction 但**大幅扩展**并加入复现上下文.

## 2. Section 1 结构

### 1.1 Background: Generative Recommendation and Codebook-based Tokenization
- GR 范式: LLM 生成 item IDs 替代 ranking
- 三大类 tokenization: ID-based / context-aware / codebook-based
- Codebook-based (TIGER, LC-Rec, LETTER) 用 RQ-VAE 编码 item 为 hierarchical token
- RQ-VAE 在 Euclidean 空间: tree-like hierarchy + low codebook utilization 问题

### 1.2 The HG-Rec Approach: Hyperbolic RQ-VAE + Differential-Length Codebook
- 复述 HG-Rec 论文的核心 idea: hyperbolic 空间适配 hierarchy, exponential volume growth
- 两个组件: (a) Hyperbolic RQ-VAE, (b) Differential-Length Codebook (γ≈2, K1∈{16,32,64})
- HG-Rec 在 Beauty/Instruments/Yelp 上 paper-reported 提升 4.8-13.5%

### 1.3 Our Reproduction Motivation (NEW)
- 为什么复现 HG-Rec:
  - 检验 hyperbolic geometry 假设是否在多个数据集稳定有效
  - 探索 per-layer curvature (Task #88) / free-curvature (Task #89) 是否能进一步提升
  - 建立 vanilla RQ-VAE (phonism) + Sinkhorn 的强 baseline 作为对比基准
- 复现 setting: Amazon Musical_Instruments (9922 items, 511836 interactions), sentence-t5-base, T5-small, [64,128,256] codebook

### 1.4 Our Key Findings Preview (NEW)
- **5 重独立证据**显示 Musical_Instruments 数据本质欧氏:
  - Task #82 弱信号检测 (phonism 数据 κ_real ≈ 0)
  - Task #88 per-layer curvature 网格 (R@10 跨 5.3%, κ=0 边际最优)
  - Task #89 free-curvature 学习 (18/18 (layer, κ_m) 收敛到 0)
  - Task #90 codebook 分解 (L0/L1/L2 token SET 完全共享)
  - Task #91 Stage 3 训练动力学 (valid 排序 vs test 排序反向, vanilla 泛化最优)
- **核心结论**: 简单 vanilla RQ-VAE + Sinkhorn (phonism) R@10=0.1058 ≥ HG-Rec c555 0.1051 ≥ HG-Rec c111 0.1020

### 1.5 Contributions (REWRITTEN, 与 HG-Rec 论文不同)
- 复现贡献 (vs HG-Rec 原始贡献):
  1. 复现 HG-Rec 全流程在 Musical_Instruments, 验证 paper-reported 相对排序保留, 但绝对数字系统性低于 paper 22.4%
  2. 通过 per-layer curvature 网格 + free-curvature 学习, 证明 Musical_Instruments 数据集上几何先验边际效应 (5.3% R@10 跨度, c555 κ=0.5 微弱最优)
  3. 通过 codebook 分解 + 训练动力学分析, 揭示 HG-Rec 的优势不是来自"训练更好"而是来自"初始化更集中利于 T5 生成"
  4. 提供 4 条实践建议 (vanilla+Sinkhorn / test R@10 为主指标 / 诊断数据几何 / 匹配 prior 到数据)
- 与 HG-Rec 论文互补: HG-Rec 论文在 Beauty/Instruments/Yelp 三数据集验证 geometric prior 有效, 我们的复现补充"geometric prior 在 Musical_Instruments 这一类 non-hierarchical 数据上边际" 这条结论

## 3. 预算

| 阶段 | 估算时间 | GPU |
|------|---------|-----|
| 阅读 HG-Rec.md Introduction + 复现上下文 | ~10 min | 0 |
| Section 1 markdown 起草 | ~30 min | 0 |
| Contributions 段落重写 (与 paper 不同) | ~15 min | 0 |
| **总计** | **~55 min** | **0 GPU** |

## 4. 产物清单

- `verdicts/task95_paper_section1_introduction_draft.md` — Section 1 完整 markdown 草稿
- 含 5 子节 (1.1 Background / 1.2 HG-Rec Approach / 1.3 Reproduction Motivation / 1.4 Findings Preview / 1.5 Contributions)
- 含 5 重证据预告 (Tasks #82/#88/#89/#90/#91)
- 含 4 条 practice recommendations 预告 (与 Section 6.6 对齐)

## 5. 关联

- 前置: Task #92 (Section 5) + Task #93 (Section 6) + Task #94 (Section 3) + 5 重证据任务
- 后续: Section 4 Related Work + Section 7 Conclusion, 论文最终写作, Submission 准备

---

**核心交付**: 论文 Section 1 (Introduction) 完整 markdown 草稿, 与 HG-Rec 论文 Introduction 互补 (复现版本视角).
