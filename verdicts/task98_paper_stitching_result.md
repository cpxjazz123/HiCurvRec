# Task #98 — Paper Stitching Result (single-file papers/paper.md)

> **完成日期**: 2026-07-24
> **状态**: ✅ 闭环 (writeup only, 0 GPU)
> **核心交付**: `papers/paper.md` 单一可提交 paper 文件, 7 sections + Appendix LaTeX + References, 整合 Task #92/93/94/95/96/97 + HG-Rec §2 Preliminaries. 论文 submission-ready 单文件版本.

---

## 1. 整合产物

| 文件 | 行数 | 大小 | 内容 |
|------|------|------|------|
| `papers/paper.md` | ~440 | ~30 KB | Title + Abstract + 7 sections + Appendix LaTeX + References |

## 2. 整合源映射

| Section | 来源 | Task |
|---------|------|------|
| Abstract | 新写 | n/a |
| §1 Introduction | verdicts/task95_paper_section1_introduction_draft.md | #95 |
| §2 Preliminaries | papers/HG-Rec.md §2.2 (Poincaré 距离/Möbius 加法/exp-log map) | HG-Rec 引用 |
| §3 Method | verdicts/task94_paper_section3_method_draft.md | #94 |
| §4 Related Work | verdicts/task96_paper_section4_related_work_draft.md | #96 |
| §5 Experiments | verdicts/task92_paper_section5_draft.md | #92 |
| §6 Discussion | verdicts/task93_paper_section6_draft.md | #93 |
| §7 Conclusion | verdicts/task97_paper_section7_conclusion_draft.md | #97 |
| Appendix LaTeX | Task #92 §5 Appendix | #92 |
| References | HG-Rec 论文 + 复现额外引用 | 合并 |

## 3. 整合规则验证

| 规则 | 实现 | 验证 |
|------|------|------|
| LaTeX 公式编号对齐 HG-Rec (1-13) | ✅ | eq. 1-4 在 §2, eq. 5-8 在 §3.1, eq. 9-12 在 §3.2, eq. 13 在 §3.3 |
| Section 编号 1-7 | ✅ | 7 个 ## 级标题 |
| 表格编号 Table 1-7 | ✅ | Table 2 (main) / 3 (arch) / 4 (curvature) / 5 (Jaccard) / 6 (dist) / 7 (dynamics) + Table 1 (dataset) |
| 内部交叉引用 §X.Y / Table X / Eq. | ✅ | "§5.7" / "Table 2" / "Eq. (1)" |
| References 合并 | ✅ | HG-Rec 关键引用 + 复现额外 |

## 4. 论文统计

| 指标 | 数值 |
|------|------|
| 总 sections | 7 + Preliminaries + Appendix |
| 总 tables | 7 (Table 1-7) |
| 总 LaTeX 公式 | 13 (eq. 1-13) |
| 总 Theoreorem 引用 | 3 (Thm 3.1 / 3.2 / 3.4) |
| 总 references | 10+ 关键引用 (HG-Rec + 复现额外) |
| 估计 word count | ~8000 words |
| 估计 pages (LaTeX 双栏) | ~10-12 pages |

## 5. Paper Deliverable 完整结构

| Section | Task | 状态 |
|---------|------|------|
| 1 Introduction | #95 | ✅ |
| 2 Preliminaries | HG-Rec 引用 | ✅ |
| 3 Method | #94 | ✅ |
| 4 Related Work | #96 | ✅ |
| 5 Experiments | #92 | ✅ |
| 6 Discussion | #93 | ✅ |
| 7 Conclusion + Ack + Impact | #97 | ✅ |
| Appendix LaTeX (Tables) | #92 | ✅ |
| References | 合并 | ✅ |
| **paper.md 单文件** | **#98** | **✅** |

## 6. 后续可选

- PDF 转换 (pandoc / LaTeX)
- 参考文献 BibTeX 整理
- 格式微调 (会议模板 / 单栏 / 双栏)
- Cover letter / supplementary material
- Code release (Stage 1-4 pipeline + 复现脚本)

---

**核心交付**: `papers/paper.md` 单一可提交 paper 文件, 7 sections + Appendix LaTeX + References, 整合 Task #92/93/94/95/96/97 + HG-Rec §2 Preliminaries. 论文 submission-ready 单文件版本, 可直接用于 PDF 转换 / 格式微调 / Submission 准备.

result: Task #98 — Paper Stitching 完成. papers/paper.md 单文件版本产出, 7 sections + Appendix LaTeX + References, 整合 6 个 section 草稿 + HG-Rec §2 Preliminaries. 论文 submission-ready 单文件, 含 7 个 Table + 13 个 LaTeX 公式 + 3 个 Theorem 引用 + 10+ 关键参考文献.
