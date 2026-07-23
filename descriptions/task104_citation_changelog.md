# Task #104 — CITATION.cff + CHANGELOG.md 论文 metadata

> **任务目的**: 双交付物: (a) `CITATION.cff` — GitHub 自动 citation 元数据 (cff-version 1.2.0, type=title/author/doi/keywords/license/preferred-citation/references); (b) `CHANGELOG.md` — Keep-a-Changelog 格式, 记录 [Unreleased] + 这次会话闭环 (Task #98-#103) + Earlier periods (Section drafting + experimental evidence + R@5 cherry-picks + pre-curvature diagnostics) 三阶段. 闭环判据: 双文件落盘 + 符合规范.

> **完成日期**: (in progress)
> **状态**: 🟡 待启动

---

## 1. 背景

仓库已有 `papers/refs.bib` (BibTeX references) 和 `README.md` (人类入口). 但还需要:
- **CITATION.cff** — GitHub 自 2022 起支持, 用于仓库侧边 "Cite this repository" 按钮, 自动生成 APA / BibTeX / EndNote 等格式. 这是 paper submission 阶段缺失的最后一公里.
- **CHANGELOG.md** — Keep-a-Changelog 格式是开源项目标配. 仓库已有 100+ task verdicts, 但没有单一 changelog 给出高层视图. 适合 paper submission 的"项目时间线".

## 2. 实验设计 (writeup only)

### 2.1 CITATION.cff (cff-version 1.2.0)
字段:
- `cff-version`: 1.2.0
- `message`: 提示同时引用原 HG-Rec paper
- `type`: article
- `title`: 论文完整标题
- `authors`: Reproduction (single-author independent)
- `date-released`: 2026-07-24
- `keywords`: 6 个 (recommendation-systems, generative-retrieval, rq-vae, hyperbolic-geometry, differential-length-codebook, amazon-musical-instruments, independent-reproduction)
- `license`: MIT
- `preferred-citation`: 这个 reproduction 的 BibTeX 条目
- `references`: 原 HG-Rec paper 的 BibTeX 条目

### 2.2 CHANGELOG.md (3 sections)

| Section | 内容 |
|---------|------|
| [Unreleased] | Task #104 (本任务) — CITATION.cff; 当前 paper submission 已交付; Headline numbers; Reproduction protocol |
| [2026-07-24] — Paper Submission Package | Task #99-#103: paper.pdf, README.md, REPRODUCE.md, audit script. 10 LaTeX bugs fixed. |
| [Earlier periods] — Pre-submission work | Section drafting (#92-#97, #98); Experimental evidence (#82-#91, #84 main); Earlier R@5 cherry-picks (#58-#61); Pre-curvature diagnostics (#53-#56, #62-#82) |

每个 entry 含 **Added / Changed / Fixed / Notes** 子段 (Keep-a-Changelog 模板).

## 3. 决策触发

| 条件 | 决策 |
|------|------|
| 双文件落盘 + CITATION.cff YAML 有效 + CHANGELOG.md 符合 Keep-a-Changelog | ✅ 闭环 |
| YAML 解析失败 | ❌ 修复 cff-syntax, 重试 |
| CHANGELOG.md 与 verdicts/ 数字不一致 | ⚠️ 同步 gaps, 不静默更正 |

## 4. 预算

| 阶段 | 估算时间 | GPU |
|------|---------|-----|
| 写 CITATION.cff | ~10 min | 0 |
| 写 CHANGELOG.md | ~25 min | 0 |
| YAML 验证 | ~10 sec | 0 |
| 写 description + verdict + commit | ~10 min | 0 |
| **总计** | **~45 min** | **0 GPU** |

## 5. 风险与缓解

**风险 1**: CITATION.cff YAML 字段不符合 cff-version 1.2.0 spec
  → **缓解**: 用 Python yaml.safe_load 验证 + 参考 https://github.com/citation-file-format/citation-file-format 规范

**风险 2**: CHANGELOG.md 与 verdicts/ 数字不一致 (例如老 task R@5 数字我记错)
  → **缓解**: 对于我不确定的数字, 标 "见 verdicts/" 而不复刻, 或用反引号列出文件名让用户自核

## 6. 产物清单

- `CITATION.cff` — GitHub-native citation metadata (~ 90 行 YAML)
- `CHANGELOG.md` — Keep-a-Changelog 格式 (~ 150 行)
- `descriptions/task104_citation_changelog.md` — 本任务描述
- `verdicts/task104_citation_changelog_result.md` — 闭环报告
- git commit 含以上四件套

## 7. 关联

- 前置: Task #98-#103 (paper submission 准备)
- 后置: (无, paper submission metadata 闭环, 后续可选 shields.io badge / GitHub Actions / supplementary)

---

**核心交付**: CITATION.cff (GitHub auto-citation 按钮) + CHANGELOG.md (项目时间线).
