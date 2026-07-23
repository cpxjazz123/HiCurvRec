# Task #100 — Paper Submission 准备结果 (BibTeX + 格式 + commit)

> **完成日期**: 2026-07-24
> **状态**: ✅ 已闭环
> **核心交付**: papers/paper.{md,tex,bib,pdf} 四件套 + 一次 git commit (3b982f6) 提交 submission 版本. 论文 submission-ready 状态完成, 后续可选 venue-specific 微调.

---

## 1. 闭环判据 (Step A + B + C 全部完成)

| Step | 任务 | 产物 | 验证 |
|------|------|------|------|
| A | BibTeX 整理 | `papers/refs.bib` | 12 entries + 1 misc, plainnat 风格 |
| B | 格式微调 | `papers/paper.tex` (modified) + `papers/paper.tex.backup` | xelatex 重编译成功, 14 pages PDF |
| C | git commit | commit `3b982f6` | local commit 落盘, 远端 push 跳过 (无 origin remote) |

## 2. Step A — BibTeX (papers/refs.bib)

12 个 BibTeX entry, 风格 `plainnat`, 用 `&apos;` 处理 apostrophe, 用 `Poincar{\'e}` 处理 accent:

| Key | 引用 |
|-----|------|
| `zhang2026hgrec` | HG-Rec (ICML 2026) |
| `rajput2023tiger` | TIGER (NeurIPS 2023) |
| `zheng2024lcrec` | LC-Rec (ICDE 2024) |
| `wang2024letter` | LETTER (CIKM 2024) |
| `singh2024rqvae` | RQ-VAE (RecSys 2024) |
| `ni2022sentencet5` | sentence-T5 (ACL 2022) |
| `kang2018sasrec` | SASRec (ICDM 2018) |
| `sun2019bert4rec` | BERT4Rec (CIKM 2019) |
| `sun2021hgcf` | HGCF (WWW 2021) |
| `nickel2017poincare` | Poincaré embeddings (NeurIPS 2017) |
| `chami2019hgcn` | HGCN (NeurIPS 2019) |
| `hgrec2026bib` | Inheritance to HG-Rec full bibliography |

**R11.3 决策**: 保持 inline reference list 不替换为 `\bibliography{refs}` (避免 bibtex 编译依赖). BibTeX 文件作为 **forward-compatible**, paper 内 References 仍以 inline list 显示. 注释中说明切换路径.

## 3. Step B — 格式微调 (papers/paper.tex)

### 3.1 改动列表

| # | 改动 | 原因 |
|---|------|------|
| 1 | `\usepackage{times}` | Submission 时更接近 ICML template 字体 |
| 2 | `\usepackage{fancyhdr}` + `\setlength{\headheight}{14pt}` | 每页 header/footer, 修复 "headheight too small" warning |
| 3 | `\fancyhead[L]{HG-Rec Reproduction --- Musical\_Instruments}` | Page header 标识 |
| 4 | `\fancyhead[R]{\thepage}` | Page number 右上 |
| 5 | `\section*{Acknowledgements}` (新增, References 之前) | 致谢 snap-research/GRID + HG-Rec 作者 + GPU 资源 |

### 3.2 R11.3 决策 (拒绝项)
- ❌ `\usepackage{authblk}` — 当前 TeX Live 未安装, 移除后仅保留简单 `\author{}` 行
- ❌ `\bibliography{refs}` — 避免引入 bibtex 编译依赖, inline list 保持

### 3.3 编译验证

| 检查 | 结果 |
|------|------|
| xelatex × 2 | ✅ 0 errors |
| Page count | ✅ 14 pages (与 Task #99 一致) |
| PDF size | ✅ 106842 bytes (vs Task #99 124959, Δ -14.4%, 因 PDF 工具版本/字体压缩差异) |
| Acknowledgements 渲染 | ✅ 致谢段落显示于 References 之前 |
| fancyhdr header 显示 | ✅ 每页 header 显示 "HG-Rec Reproduction --- Musical\_Instruments" |

## 4. Step C — Git commit (3b982f6)

### 4.1 Commit 内容

| 文件 | 操作 |
|------|------|
| `papers/paper.md` | A (新增) |
| `papers/paper.tex` | A (Step B 修改) |
| `papers/paper.tex.backup` | A (baseline 备份) |
| `papers/paper.pdf` | A (重编译) |
| `papers/paper.aux` / `paper.out` | A (build artifacts) |
| `papers/refs.bib` | A (Step A) |
| `descriptions/task99_paper_md_to_pdf.md` | A (Task #99 文档化) |
| `descriptions/task100_paper_submission_prep.md` | A (Task #100 task definition) |
| `scripts/task99_md_to_pdf.py` | A (Task #99 脚本) |

### 4.2 R11.3 决策
- ❌ `git push origin main` — 无 origin remote 配置 (R11.3 fallback: local commit only, 在本节报告)
- ❌ `git add` 所有 untracked 文件 — 仅 stage submission 相关文件 (data/, src/, products/, configs/ 等 framework 不在 commit 中, 避免污染)

## 5. 验证 (R11.3 自主验证)

| 检查 | 结果 |
|------|------|
| papers/refs.bib 存在 | ✅ 12 entries |
| papers/paper.tex 包含 `\\usepackage{times}` + `\\usepackage{fancyhdr}` + `\\setlength{\\headheight}{14pt}` | ✅ |
| papers/paper.tex 包含 Acknowledgements section | ✅ |
| papers/paper.pdf 重编译后 14 pages | ✅ |
| git log 显示 commit 3b982f6 标题 "Task #100: paper submission prep..." | ✅ |
| descriptions/ R9 连续性 | ✅ max=100, 无空洞 |
| Task #99 description 已补 (R9 修复) | ✅ `descriptions/task99_paper_md_to_pdf.md` |

## 6. 已知限制

| 限制 | 影响 | 后续可选 |
|------|------|---------|
| `\\usepackage{authblk}` 缺失 | 无法使用完整 author affiliation block | 安装 `texlive-publishers` 包或保留 inline `\author{}` |
| inline references 而非 `\bibliography` | 切换 venue template 时需重写 | 在 self-contained 版本下稳定 |
| 无 git remote | local commit only, 无法 push | 用户/管理员配置 origin 后 push |
| `\times Roman` 字体在某些元素上 fall-back 到默认 | Section title/footnote 正常 | `\usepackage{mathptmx}` 加数学字体 |

## 7. 后续可选 (非闭环必要)

- 安装 authblk 包 / 切换完整 ICML template (如果会议要求)
- 配置 git remote + push
- 切换 `\bibliography{refs}` 完整 BibTeX 流程 (需 bibtex + xelatex 三步编译)
- Supplementary material 整理 (code release / Dockerfile)

## 8. 关联

- 前置: Task #98 paper.md + Task #99 paper.pdf
- 后置: (无, paper submission 闭环, 后续是 venue-specific 微调)

---

**核心交付**: papers/paper.{md,tex,bib,pdf} submission 四件套 + commit `3b982f6`. Paper submission 阶段全闭环. 14 页 PDF (106 KB), 12 条 BibTeX references, Times 字体 + fancyhdr header/footer + Acknowledgements section 完整. Local commit only (无 origin remote, 用户可后续配置 push).

result: Task #100 — Paper Submission 准备全闭环. papers/refs.bib (12 BibTeX entries) + papers/paper.tex (Times 字体 + fancyhdr + Acknowledgements section) + papers/paper.pdf (14 pages, 106 KB) 四件套落地. Local git commit 3b982f6 提交. 论文 submission-ready 状态完成.
