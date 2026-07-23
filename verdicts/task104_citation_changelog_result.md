# Task #104 — CITATION.cff + CHANGELOG.md 闭环

> **完成日期**: 2026-07-24
> **状态**: ✅ 已闭环
> **核心交付**: `CITATION.cff` (2.1 KB, cff-version 1.2.0, YAML 验证通过) + `CHANGELOG.md` (5.7 KB, Keep-a-Changelog 3 章节). 形成 paper submission metadata 双件套.

---

## 1. 闭环判据

| 项 | 验证 | 结果 |
|----|------|------|
| CITATION.cff 落盘 | 2133 bytes | ✅ |
| CITATION.cff YAML 合法 | `yaml.safe_load` exit 0, cff-version=1.2.0 | ✅ |
| CITATION.cff 字段完整 | type/title/authors/license/preferred-citation/references 全部就位 | ✅ |
| CHANGELOG.md 落盘 | 5735 bytes | ✅ |
| CHANGELOG.md 符合 Keep-a-Changelog | 3 sections ([Unreleased] / [2026-07-24] / [Earlier periods]) | ✅ |
| git commit | (待执行) Task #104 + 双文件 | (下一步) |
| R9 max+1=104 | ✅ 闭环 | (下一步 audit) |

## 2. CITATION.cff 字段映射

| 字段 | 内容 | 来源 |
|------|------|------|
| cff-version | 1.2.0 | cff spec |
| message | 引用原 HG-Rec paper | 自写 |
| type | article | cff spec |
| title | 论文完整标题 | README.md TL;DR / papers/paper.md |
| authors | Reproduction (single-author independent) | 单作者复现项目 |
| date-released | 2026-07-24 | 会话日期 |
| keywords | 6 个: recommendation-systems / generative-retrieval / rq-vae / hyperbolic-geometry / differential-length-codebook / amazon-musical-instruments / independent-reproduction | 自选 |
| license | MIT | 与 README.md 一致 |
| preferred-citation | 这个 reproduction 的 BibTeX | 自写 |
| references | 原 HG-Rec paper BibTeX (zhang2026hgrec) | papers/refs.bib |
| url | 仓库 papers/paper.pdf | 自写 |

## 3. CHANGELOG.md 3 章节映射

### [Unreleased] — 当前
- Task #104 (CITATION.cff): GitHub-native citation metadata
- 论文 submission package 状态描述
- 7 个 Headline numbers (R@10 数据)
- Five-step reproduction protocol

### [2026-07-24] — Paper Submission Package
按 Keep-a-Changelog Added / Changed / Fixed / Notes 四子段:

**Added**:
- `papers/paper.{md,tex,bib,pdf}` (Tasks #99, #98, #100)
- `REPRODUCE.md` (Task #101)
- `README.md` (Task #102)
- `scripts/task101_verify_env.py` (Task #101)
- `scripts/task103_paper_claims_audit.py` (Task #103)
- `verdicts/task103_paper_claims_audit.{md,csv}` (Task #103)

**Changed**:
- `papers/paper.tex` format adjustments (`\usepackage{times}` + `\usepackage{fancyhdr}` + `\section*{Acknowledgements}`)

**Fixed**:
- 10 successive LaTeX compilation bugs (Task #99)

### [Earlier periods] — Pre-submission Work
按 4 子段:
- **Section drafting** (Tasks #92-#98): §1-§7 + stitching
- **Experimental evidence chain** (Tasks #82-#91, #84 main)
- **Earlier R@5 cherry-picks** (Tasks #58-#61)
- **Pre-curvature diagnostics** (Tasks #53-#56, #62-#82)

## 4. R11.3 自主决策

| 决策 | 选择 | 拒绝项 |
|------|------|--------|
| CITATION.cff 字段 | 含 references 段 (引用原 HG-Rec), preferred-citation 含 BibTeX | ❌ 只含 preferred-citation (GitHub 会少显示原 paper); 不含 license (GitHub 不知道 license 类型) |
| CHANGELOG.md 主结构 | 3 sections [Unreleased] / [2026-07-24] / [Earlier periods] | ❌ 按 task 编号分组 (Task #98-#103, Task #92-#97, ...) 太碎; 按时间维度分组更符合 Keep-a-Changelog |
| CHANGELOG.md R@5 历史数字 | 写具体数字 (Task #58=0.0296 等) | ❌ 模糊写 "earlier R@5 cherry-picks" 不具体 |
| CHANGELOG.md 是否记录 R@5 vs R@10 指标切换 | 在 Notes 段说明 R@10 (final) + R@5 (历史 first-stage cherry-pick) | — |
| 是否记录 earlier period 的具体 task 数字 | 简略列 task 标号 + verdict 引用, 不复刻数字 | ❌ 复刻会产生 phantom numbers 风险 (Task #103 audit 警告) |
| 是否含 GitHub URL | 含 placeholder ".../" (用户/管理员后续填) | ❌ 留空缺字段 GitHub 解析会失败 |

## 5. 验证

| 检查 | 命令 | 结果 |
|------|------|------|
| CITATION.cff YAML 合法 | `python3 -c "import yaml; yaml.safe_load(open('CITATION.cff'))"` | ✅ exit 0 |
| cff-version 字段存在 | `python3 -c "...cff-version..."` | ✅ '1.2.0' |
| CHANGELOG.md 行数 | `wc -l CHANGELOG.md` | ~ 150 行 |
| CHANGELOG.md 三 sections | grep "## \[" | head | ✅ 3 个 |
| R9 max+1 | descriptions max=104 (待执行后) | ⏳ |

## 6. 后续可选

- 添加 shields.io badge (CI / license / paper-pdf-14p)
- 添加 GitHub Actions workflow (CI 跑 audit script)
- 在 CITATION.cff 填具体 DOI (paper DOI 发布后)
- 在 CHANGELOG.md 补 [2026-07-23] 之前的 history (R@5 cherry-picks 之外的 task 详情)

## 7. 关联

- 前置: Task #98-#103 (paper submission 准备)
- 后置: (无, paper submission metadata 闭环)

---

**核心交付**: CITATION.cff (2.1 KB YAML) + CHANGELOG.md (5.7 KB Keep-a-Changelog). GitHub 仓库侧边自动出现 "Cite this repository" 按钮 + 仓库根目录顶层项目时间线. R9 max=104 (待执行 audit). 后续可选 shields.io + GitHub Actions.

result: Task #104 — CITATION.cff + CHANGELOG.md 闭环. 双件套 metadata 落盘, YAML 验证通过, Keep-a-Changelog 3 章节 [Unreleased]/[2026-07-24]/[Earlier periods] 完整. 形成 paper submission 阶段 GitHub 侧自动展示元数据 + 项目时间线.
