# Task #99 — Paper.md → Paper.pdf 转换 (xelatex pipeline, writeup only)

> **任务目的**: 将 Task #98 产出的 `papers/paper.md` 单文件 markdown 转换为 LaTeX, 然后通过 xelatex 编译为 submission-ready `papers/paper.pdf`. 形成论文 submission 阶段的最终可分发产物.

> **完成日期**: 2026-07-24
> **状态**: ✅ 已完成

---

## 1. 背景

论文 7-section 结构 (Task #92/93/94/95/96/97) 已在 Task #98 整合为 `papers/paper.md` 单文件 markdown, 含 7 sections + Appendix + References. 但 submission 需要 PDF 格式. 需编写一个 markdown→LaTeX→PDF 的转换 pipeline (xelatex 编译), 保留公式 `$...$` / `$$...$$` 渲染 + lstlisting verbatim 处理 Appendix LaTeX 源码.

## 2. 实施步骤

### 2.1 转换脚本
- `scripts/task99_md_to_pdf.py` — 主入口, 三阶段: parse_md_to_latex() → escape_latex_outside_math() → xelatex × 2

### 2.2 关键技术决策 (R11.3 自主决策)
1. **三状态 math 追踪 (text/inline-math/display-math)**: `$` 在不同 context 下语义不同 — text 中是 inline toggle, display 中是 end toggle, 但 inline math 中 `$$` 必须视作两个 single `$` token (处理 `$\kappa$$\to$0` 类模式). 引入三状态机解决歧义.
2. **lstlisting verbatim 保留**: Appendix 中 LaTeX 源码示例通过 lstlisting 不解析 LaTeX 特殊字符 (`&` `%` `#` `_`), escape 函数无需特殊分支.
3. **`\tag{N}` 转 `\[...\quad\text{(N)}\]`**: `\tag{N}` 在 plain `$$...$$` 中无效 (amsmath 限制), regex 预处理改为 equation env 形式.
4. **language mapping**: `latex → tex` (避免 lstlisting 找不到 latex 语言包).

### 2.3 LaTeX 编译配置
- `article` 11pt, `margin=1in`
- 必备包: amsmath, amssymb, amsthm, booktabs, geometry, hyperref, listings, xcolor
- 自定义 math operator: `\arctanh`, `\argmin`, `\argmax`
- xelatex 运行 2 次 (cross-refs/TOC)

## 3. 关键 Bug 修复 (按时间顺序, 共 10 个)

| # | Bug | 修复 |
|---|-----|------|
| 1 | `_` 被当 subscript (abstract 中) | escape_latex_outside_math 转义 `_ & % # $` |
| 2 | Body 重复 title/abstract | parse_md 改为从第一个 `## ` heading 开始 |
| 3 | `\tag not allowed here` (l.103) | regex preprocessing: `$$...\\tag{N}$$` → `\[...\quad\text{(N)}\]` |
| 4 | `\arctanh` undefined | preamble 加 `\DeclareMathOperator` |
| 5 | `\[ \]` 未识别为 math 边界 | escape 识别 `\[ \] \( \)` |
| 6 | `Couldn't load latex language` | language map: latex → tex |
| 7 | `\tag` regex `re.DOTALL` 跨行吃 §1 | 改 `[^\$\n]+?` 单行非贪婪 |
| 8 | 159 个 `&` 未转义 — escape 在 inline 时把 `$$` 当 display toggle | 三状态机修复, inline `$$` 视作两个 `

 tokens |
| 9 | markdown l.75 缺开头 `

 | 手动加 `

` 开头 |
| 10 | 跨行 equation malformed (第二个 `

` 应关闭第一个) | 修复为 `

...\quad

\n\n

...\tag{3,4}

` |

## 4. 验证

| 检查 | 结果 |
|------|------|
| xelatex 编译 | ✅ 0 errors, 仅 Overfull \hbox warnings |
| PDF 格式 | ✅ Valid PDF 1.5, 124959 bytes, 14 pages |
| 数学公式 | ✅ 13 个 LaTeX 公式正确渲染 |
| 表格 | ✅ Table 1-7 完整 |
| 字符转义 | ✅ `_ & % # $ ^ ~` 全部正确转义 |
| 段落 | ✅ 7 sections + Appendix + References 全渲染 |

## 5. 产物

| 文件 | 大小 | 内容 |
|------|------|------|
| `papers/paper.pdf` | 124959 bytes (~122 KB) | 14 页 submission-ready PDF |
| `papers/paper.tex` | 38605 chars | LaTeX 中间产物 |
| `scripts/task99_md_to_pdf.py` | ~310 lines | 转换脚本 |

## 6. 关联

- 前置: Task #98 paper.md 单文件版本
- 后置: Task #100+ paper submission 完整准备 (BibTeX + 格式微调 + git commit)

---

**核心交付**: `papers/paper.pdf` 14 页 submission-ready PDF. 通过 Task #99 修复 10 个 LaTeX bug + 引入三状态 math 追踪机制 + 整合 7 sections + Appendix LaTeX 源码 + References.
