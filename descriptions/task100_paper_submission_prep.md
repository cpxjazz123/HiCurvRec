# Task #100 — Paper Submission 准备 (BibTeX + 格式微调 + git commit)

> **任务目的**: 闭合论文 submission 阶段. 整合 Task #98 paper.md 主体 + Task #99 paper.pdf 排版, 通过 (a) BibTeX 整理 References, (b) 论文格式微调 (Section/Header/Footer/Anonymization), (c) git commit+push 完成 submission-ready 状态.

> **完成日期**: (in progress)
> **状态**: 🟡 待启动

---

## 1. 背景

Task #98 整合 paper.md 单文件, Task #99 转换 paper.pdf (14 页, 124 KB). 但 submission 还差:
- (a) References 列表需要 BibTeX 化 (`papers/refs.bib`) 方便版本管理 + 自动引用
- (b) paper.tex 格式微调: Section 标题大写 / 行距 / 字体 / header / footer / 字体 / Anonymization 处理 (ICML 2026 双盲要求)
- (c) git commit 提交 submission-ready 版本 + push 到远端

## 2. 实验设计 (writeup only)

### 2.1 Step A — BibTeX 整理
**变量**: References list (从 paper.md 抽 §8 References 全部条目) → BibTeX entries
**保持不变**: paper.md 内容, paper.tex 编译输出

**操作**:
1. 从 `papers/paper.md` 提取 References section (最后一个 `## ` heading)
2. 解析每条引用 (作者 / 标题 / venue / year)
3. 生成 `papers/refs.bib` (BibTeX entries), paper.tex 末尾用 `\bibliography{refs}`
4. 验证: xelatex + bibtex + xelatex + xelatex 三步编译能正常解析 references

### 2.2 Step B — 格式微调 (R11.3 自主决策)
**自主选择**: 应用 ICML 2026 LaTeX template 的核心元素 (但不强制替换整模板, 因为现有 paper.tex 已经能编译):
- paper 头部: `\usepackage[accepted]{icml2026}` 或仅加入 `\usepackage{times}` + `\usepackage{authblk}`
- Section 标题 font: 不变 (`\section` 是 article 默认)
- 行距: `\linespread{1.0}` (单倍)
- Anonymization: 不需要 (复现 paper 不参与双盲), 但 paper 中不显示作者信息 (留 `\author{Anonymous}` + `\author{Reproduction Paper --- 2026}` 即可)
- 添加 Acknowledgement section: 致谢 snap-research/GRID + HG-Rec 论文作者 + 项目基础设施
- 增加 page header/footer: `\pagestyle{fancy}` + `\fancyhf{}` + `\fancyfoot[C]{\thepage}`

**操作**:
1. 备份当前 `papers/paper.tex` → `papers/paper.tex.backup`
2. 应用上述格式修改到 `papers/paper.tex`
3. 重新编译 xelatex × 2 (验证未引入回归)
4. 对比 PDF byte diff (是否影响页数 / 行距 / 公式)

### 2.3 Step C — git commit + push
**操作**:
1. `git status` 确认 working tree 干净 (除新增/修改文件)
2. `git add papers/paper.{md,tex,bib} papers/paper.{pdf,aux,log,out} scripts/task99_md_to_pdf.py`
3. `git commit -m "Task #100: paper submission prep (BibTeX + 格式 + commit)"` (带 Co-Authored-By)
4. `git push origin main` (如果远端存在)

## 3. 决策触发

| 条件 | 结果 | 决策 |
|------|------|------|
| Step A BibTeX + Step B 格式 + Step C commit 全部产物落盘 | 闭环 | 写 task100 verdict, §16 表格清空 |
| xelatex 编译引入 regression (公式 / 表格破坏) | 不闭环 | 回滚 paper.tex 到 backup, 记录失败原因, 不强推 git commit |
| git push 失败 (远端权限 / 网络) | 部分闭环 | 跳过 push, 只做 local commit, 在 verdict 标注 "local-only commit" |

## 4. 预算

| 阶段 | 估算时间 | GPU |
|------|---------|-----|
| Step A BibTeX 整理 | ~10 min | 0 |
| Step B 格式微调 + xelatex 重编译 | ~10 min | 0 |
| Step C git commit + push | ~5 min | 0 |
| 写 verdict | ~5 min | 0 |
| **总计** | **~30 min** | **0 GPU** |

## 5. 风险与缓解

**风险 1**: xelatex 编译引入新 regression (格式微调副作用)
  → **缓解**: 备份 paper.tex, 修改前先 compile 一次 baseline, 任何 regression 立即回滚

**风险 2**: BibTeX entry 解析失败 (作者 / 标题含特殊字符)
  → **缓解**: 手工编写 BibTeX entries (避开自动解析), 用 `\bibliographystyle{plainnat}` 或 `unsrtnat`

**风险 3**: git push 失败 (无远端权限 / 仓库非 bare)
  → **缓解**: 仅做 local commit, 在 verdict 报告 "local commit, push skipped"

## 6. 产物清单

- `papers/refs.bib` — BibTeX references
- `papers/paper.tex` (modified) — 格式微调后 LaTeX
- `papers/paper.tex.backup` — 备份
- `papers/paper.pdf` (recompiled) — 最终 PDF
- `verdicts/task100_paper_submission_prep_result.md` — 闭环报告
- (可选) `git commit <hash>` — local commit 记录

## 7. 关联

- 前置: Task #98 paper.md + Task #99 paper.pdf
- 后置: (无, paper submission 阶段到此闭环, 后续是可选的 venue-specific 微调)

---

**核心交付**: `papers/paper.{md,tex,bib,pdf}` submission-ready 四件套 + git commit 记录.
