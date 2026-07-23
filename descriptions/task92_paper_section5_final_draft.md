# Task #92 — Paper Section 5 Final Draft (analytical writeup, no GPU)

> **任务目的**: 整合 Task #84/87/88/89/90/91/94/95 所有闭环结论, 起草论文 Section 5 (Experiments) 完整版本, 含 LaTeX 表格 + 章节结构 + Discussion, 可直接用作 paper 写作基础.

> **完成日期**: (in progress)
> **状态**: 🟡 待启动

---

## 1. 背景

所有实验任务闭环 (Task #84/88/89/90/91/95), 论文 Section 5 需要最终整合:
- **Main Results**: 27 baseline 完整 ranking (Task #87 v2)
- **Codebook Architecture**: vanilla / hyperbolic / free-curv 对比 (Task #84/88/89)
- **Per-Layer Curvature Ablation**: 6 网格 (Task #88)
- **Codebook Decomposition**: token SET 共享, item assignment 独立 (Task #90)
- **Training Dynamics**: 8 配置 valid 聚类, test 排序反转 (Task #91)
- **Paper Reproducibility**: 8/8 baseline 复现低于 paper 18-61% (Task #94 #95)

## 2. Section 5 结构

### 5.1 Experimental Setup
- Dataset: Musical_Instruments (Amazon)
- Embedding: sentence-t5-base 768d
- Codebook: [64, 128, 256] + dedup
- T5-small (6 enc + 4 dec, d_model=128)
- Train: 200 epochs, batch=256, lr=1e-4, early_stop=20

### 5.2 Main Results (Table 2 reproduction)
- R@5, R@10, NDCG@5, NDCG@10
- 8 RQ-VAE 系 + 9 generative + 5 sequential + 5 traditional = 27 baselines

### 5.3 Ablation: Codebook Architecture
- vanilla RQ-VAE + Sinkhorn (phonism R@10=0.1058)
- Hyperbolic RQ-VAE + Differential-Length Codebook (HG-Rec R@10=0.1020)
- Free-curvature per-component (Task #89 R@10=0.1015)

### 5.4 Ablation: Per-Layer Curvature (Task #88)
- 6 grid c555/c222/c215/c1055/c512/c111
- R@10 span 5.3%, c555 marginal best

### 5.5 Codebook Decomposition Analysis (Task #90)
- L0/L1/L2 token SET 全共享 (vanilla/c111/c555 Jaccard=1.000)
- 4-col SID Jaccard=0.001 (实质独立)
- L0 entropy trade-off

### 5.6 Training Dynamics (Task #91)
- 8 配置 valid R@10 聚类 [0.1240, 0.1276] (3.6%)
- 收敛 epoch 一致 (3-4)
- Generalization gap 排序

### 5.7 Discussion: Geometry vs Mechanism
- 数据本质欧氏 (5 重证据)
- 中度曲率保留 K-Means spread (H1/H2/H3)
- Generalization > Fitting (H4)
- Paper absolute numbers 系统性偏低 18-61%

## 3. 预算

| 阶段 | 估算时间 | GPU |
|------|---------|-----|
| 整合 verdict 资料 | ~30 min | 无 |
| LaTeX 表格生成 | ~15 min | 无 |
| Section 5 markdown 起草 | ~45 min | 无 |
| **总计** | **~90 min** | **0 GPU** |

## 4. 产物清单

- `verdicts/task92_paper_section5_draft.md` — Section 5 完整草稿
- `verdicts/task92_table2_ranking.csv` — 27 baseline CSV (paper Table 2)
- `verdicts/task92_curvature_ablation_table.tex` — LaTeX Table for per-layer curvature
- `verdicts/task92_stage3_dynamics_table.tex` — LaTeX Table for training dynamics

## 5. 关联

- 前置: Task #84, #87, #88, #89, #90, #91, #94, #95 (全部闭环)
- 后续: 论文最终写作, Submission 准备

---

**核心交付**: 论文 Section 5 完整 markdown 草稿, 可直接 copy 到 paper.