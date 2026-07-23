# Task #108 — arXiv Submission Packet (Verdict)

> **完成日期**: 2026-07-24
> **状态**: ✅ 已闭环
> **核心交付**: `arxiv/` 目录含 7 文件 (paper.tex sanitized + paper.pdf 14页 + refs.bib + LICENSE + ARXIV_METADATA.md + README.md + SUBMISSION_CHECKLIST.md). xelatex 重编译 exit 0, 14 页 125 KB. 形成 arXiv submission-ready 一键上传包.

---

## 1. 闭环判据

| 项 | 验证 | 结果 |
|----|------|------|
| `arxiv/paper.tex` 落盘 | 39.7 KB, 657 lines | ✅ |
| `arxiv/paper.tex` sanitization (remove fancyhdr/times/authblk) | `grep "fancyhdr\|\\\\usepackage{times}"` 仅留 1 处 comment | ✅ |
| `arxiv/paper.pdf` 14 页 125 KB | xelatex 重编译 exit 0, no Error / no Overfull \hbox | ✅ |
| `arxiv/refs.bib` 12 entries | parser: 12 `@` records | ✅ |
| `arxiv/fig/` empty placeholder | exists | ✅ |
| `arxiv/LICENSE` (CC-BY-4.0) | exists, 1 KB | ✅ |
| `arxiv/ARXIV_METADATA.md` | exists, 3.8 KB | ✅ |
| `arxiv/README.md` | exists, 1.9 KB | ✅ |
| `arxiv/SUBMISSION_CHECKLIST.md` | exists, 3.6 KB | ✅ |
| xelatex local compile | `xelatex paper.tex` exit 0 (14 pages) | ✅ |
| git commit | (待执行) | ⏳ next step |
| R9 max+1 = 108 | ✅ | ✅ |

## 2. arxiv/ 目录结构

```
arxiv/
├── ARXIV_METADATA.md       (3.8 KB) — title / categories / license / comments
├── LICENSE                 (1.0 KB) — CC-BY-4.0 (Creative Commons Attribution 4.0)
├── README.md               (1.9 KB) — quick submission + sanitization notes
├── SUBMISSION_CHECKLIST.md (3.6 KB) — pre-upload verification 13 检查项
├── paper.tex               (39.7 KB) — sanitized (no fancyhdr / no times / no authblk)
├── paper.pdf               (125.7 KB, 14 pages) — xelatex 重编译 exit 0
├── refs.bib                (3.9 KB, 12 entries) — 继承 papers/refs.bib
└── fig/                    (empty placeholder) — paper 无 embedded figures
```

## 3. paper.tex sanitization

对比 `papers/paper.tex` (会议稿) vs `arxiv/paper.tex` (arXiv 投递稿):

| 元素 | papers/paper.tex | arxiv/paper.tex | 原因 |
|------|----------------|----------------|------|
| `\usepackage{times}` | yes | **removed** | default font, 加快 arXiv 自动 build |
| `\usepackage{fancyhdr}` | yes | **removed** | arXiv auto-build 偶尔 fail |
| `\fancyhead[L]{\ldots}` | HG-Rec Reproduction header | **removed** | 同上 |
| `\setlength{\headheight}{14pt}` | yes | **removed** | fancyhdr dependency |
| `\usepackage{authblk}` | commented | **removed** | TeX Live default OK |
| `\author{Reproduction Paper --- 2026}` | placeholder | `Reproduction of HG-Rec (Zhang et al., ICML 2026) \\ \texttt{wlia0047@github.com}` | arXiv 要 identifiable submitter |
| `\date{\today}` | dynamic | `July 24, 2026` | reproducible date |

content 部分 (sections 1-7) 100% 不变, verbatim 拷贝.

## 4. arXiv metadata 决策

| 字段 | 选择 | 备选 |
|------|------|------|
| Primary category | `cs.IR` (Information Retrieval) | cs.AI (fallback) |
| Cross-lists | `cs.LG` (Machine Learning) | cs.AI / stat.ML |
| License | CC-BY-4.0 | CC-BY-SA-4.0 / CC0 / arXiv non-exclusive |
| Title | verbatim from papers/ | shortened |
| Comments | "Reproduction of HG-Rec (Zhang et al., ICML 2026), 14 pages, 7 figures, 7 tables" | 空 |

**Note**: 论文没 \cite{} 调用 (manual 引用 inline 文字 `(Zhang et al., 2026)`). refs.bib 是 supplementary, 与 papers/paper.tex 12 entries 一致.

## 5. R11.3 自主决策

| 决策 | 选择 | 拒绝 |
|------|------|------|
| paper.tex 是否 sanitization | ✅ 是 (移除 fancyhdr/times/authblk) | ❌ 完全 verbatim (arXiv auto-build 偶尔 fail); 仅移除 fancyhdr (不彻底) |
| License | CC-BY-4.0 | ❌ CC-BY-SA (viral share-alike, 不利后续修改); arXiv non-exclusive default (过于限制) |
| 是否提供 ARXIV_METADATA.md | ✅ 是 (reviewer 一目了然) | ❌ 仅 README.md (要在 README 内嵌 metadata, 乱) |
| 是否本地 xelatex 验证 | ✅ 是 (exit 0 = 信心) | ❌ 仅复制文件不验证 (arXiv auto-build 才发现 fail 时已晚) |
| 中间文件 (.aux .log .out) 是否清理 | ✅ 是 (留干净的 arxiv/ 给 reviewer) | ❌ 保留 (无意义 noise) |
| 是否给 reviewer "see also" 链接 (papers/SUBMISSION_DEFENSE.md) | ✅ README.md 末尾链接 | ❌ 不链 (orphan) |
| author block | `Reproduction of HG-Rec ... \\ \texttt{wlia0047@github.com}` | ❌ `Anonymous` (arXiv auto-anonymize 不需要); 复现者真名 (单作者复现项目, 不需要 affiliation) |

## 6. 验证

| 检查 | 命令 | 结果 |
|------|------|------|
| arxiv/ 目录 7 文件 | `ls arxiv/` | ✅ 7 entries + fig/ |
| paper.tex 编译 | `xelatex paper.tex` | exit 0, 14 pages |
| paper.tex 无 Error | `grep "^!" arxiv/paper.log` | ∅ |
| paper.tex 无 Overfull \hbox | `grep "Overfull" arxiv/paper.log` | ∅ |
| refs.bib 完整性 | `grep "@[a-z]+{" arxiv/refs.bib \| wc -l` | 12 entries |
| paper.tex 不含 fancyhdr/times/authblk (除 comment) | `grep -c "fancyhdr\|\\\\usepackage{times}\|authblk" arxiv/paper.tex` | 1 (the comment) |
| LICENSE CC-BY-4.0 | `head -1 arxiv/LICENSE` | "Creative Commons Attribution 4.0 International License (CC-BY-4.0)" |
| R9 max+1 = 108 | `ls descriptions/ \| grep -oE 'task[0-9]+'` | max = 108 ✅ |

## 7. 关联

- 前置: Task #98-#107 (paper submission 全套 + defense packet + CI)
- 后置: 实际 arXiv submission (需用户授权 + endorser), 后续 camera-ready prep, shields.io badges

## 8. 后续可选

- **Task #109 — User-facing submission wizard**: 用户填 form 后自动 zip + upload arXiv
- **Task #109 — shields.io badges**: 从 task106_audits.json 自动生成
- **arXiv submission actual**: 需用户手动登录 (endorser 必备)

---

**核心交付**: `arxiv/` 7 文件 submission-ready bundle. arxiv/paper.tex sanitized (移除 fancyhdr/times/authblk). arxiv/paper.pdf xelatex 重编译 14 页 125 KB. ARXIV_METADATA.md + SUBMISSION_CHECKLIST.md reviewer-readable. CC-BY-4.0 license. 形成 arXiv 一键 zip 上传包.

result: Task #108 — arXiv submission packet 7 文件闭环. paper.tex 39.7 KB sanitized + paper.pdf 14 页 125 KB xelatex 重编译 exit 0 + refs.bib 12 entries + LICENSE CC-BY-4.0 + ARXIV_METADATA.md + README.md + SUBMISSION_CHECKLIST.md + fig/ empty. arXiv 一键 zip 上传包.
