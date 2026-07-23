# Task #98 — Paper Stitching (single-file papers/paper.md, writeup only)

> **任务目的**: 将 Task #92/93/94/95/96/97 的 6 个 section 草稿 + HG-Rec 论文 §2 Preliminaries 整合为单一可提交的 paper.md 文件. 形成 paper deliverable 完整单文件版本, 供 submission 准备阶段直接使用.

> **完成日期**: (in progress)
> **状态**: 🟡 待启动

---

## 1. 背景

论文 7-section 结构已闭环 (Task #92/93/94/95/96/97 + HG-Rec §2 引用). 但目前各 section 散落在 6 个 verdict 文件中, 不便于直接提交. 需要一个 `papers/paper.md` 单文件版本, 按学术 paper 标准格式整合.

## 2. 整合目标

### 2.1 输出文件
- `papers/paper.md` — 单一 paper 文件, 含 7 sections + LaTeX 公式 + 表格 + 参考文献
- 含 Title / Authors / Abstract / 7 sections / Appendix

### 2.2 整合步骤
1. 创建 paper 头部 (Title + Authors + Abstract + Introduction)
2. 嵌入 Task #95 Section 1 完整 markdown
3. 嵌入 HG-Rec §2 Preliminaries 关键公式 (eq. 1-4)
4. 嵌入 Task #94 Section 3 完整 markdown
5. 嵌入 Task #96 Section 4 完整 markdown
6. 嵌入 Task #92 Section 5 完整 markdown
7. 嵌入 Task #93 Section 6 完整 markdown
8. 嵌入 Task #97 Section 7 完整 markdown
9. 嵌入 Appendix LaTeX (Table 2/4/7)
10. 嵌入 References (合并 HG-Rec 论文 refs + 复现额外 refs)

### 2.3 整合规则
- LaTeX 公式编号保持原 HG-Rec 论文编号 (eq. 1-13)
- Section 编号统一为 1-7
- 表格编号: Table 1 (dataset) / Table 2 (main results) / Table 3 (codebook ablation) / Table 4 (curvature) / Table 5-6 (decomposition) / Table 7 (dynamics)
- 内部交叉引用统一: §X.Y / Table X / Eq. (X) / Task #N
- Reference 列表合并去重

## 3. 预算

| 阶段 | 估算时间 | GPU |
|------|---------|-----|
| 读 6 个 section verdict | ~5 min | 0 |
| paper.md 整合写入 | ~30 min | 0 |
| 公式 + 引用核验 | ~10 min | 0 |
| **总计** | **~45 min** | **0 GPU** |

## 4. 产物清单

- `papers/paper.md` — 完整 paper 单文件, 7 sections + appendix + references
- `descriptions/task98_paper_stitching.md` — 本任务描述
- `verdicts/task98_paper_stitching_result.md` — 整合报告

## 5. 关联

- 前置: Task #92/93/94/95/96/97 (全部闭环, 6 个 section 草稿)
- 后续: 论文 Submission 准备 (PDF 转换 / 参考文献 BibTeX / 格式微调)

---

**核心交付**: `papers/paper.md` 单一可提交 paper 文件.
