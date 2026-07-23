# Task #106 — Submission Defense Bundle

> **任务目的**: 闭环 paper submission 阶段最后一轮 self-defense. 5 项独立审计 (paper.md ⇄ paper.tex 同步 / ICML 2026 abstract 合规 / Section 5 baseline 覆盖声明 / 数字一致性复审 / 第三方 license attribution 确认). 产出 `papers/SUBMISSION_DEFENSE.md`, 形成 reviewer 一键可查的 defense packet.

> **完成日期**: (in progress)
> **状态**: 🟡 待启动

---

## 1. 背景

提交阶段已闭环 6 件事:
- Task #98 paper.md 单一可提交版本
- Task #99 paper.md → paper.pdf 转换 + LaTeX 修复
- Task #100 paper submission 准备 (refs.bib + 格式微调)
- Task #101 REPRODUCE.md 复现包
- Task #102 README.md 入口页
- Task #103 paper claims audit (14/14, Δ=0.0000)
- Task #104 CITATION.cff + CHANGELOG.md
- Task #105 R12 ckpt integrity 4 层审计

但 paper reviewer 真正会问的最后一类问题是:
1. paper.md 是 source of truth 吗? paper.tex 还同步吗? (Task #99 后 paper.tex 被轻微 format edit 过, Task #98 后 paper.md 内容未再变, 应同步, 但缺显式验证)
2. abstract 字数符合 ICML 2026 ≤ 200 吗? (✅ 实测 197 字)
3. paper Table 2 列了 ~13 个 baseline, 实际 8/13 closed, 没复现的 5 个是否在 Section 5 明确标注 "inherits paper-reported number"?
4. Task #103 已审 14 claim, 但 §5 中提到的**所有数字**是否都 traceable 到 verdict? (Task #103 audit 用了 14 sample claim, 不是 full coverage)
5. src/ + configs/ 是 Apache 2.0 上游, paper.md 是否声明 reproduce 自 upstream + 标明 license?

Task #106 = 闭环上述 5 项. 写完 paper 即可定稿.

---

## 2. 实验设计 (writeup only)

### 2.1 Audit #1 — paper.md ⇄ paper.tex 同步

**目标**: 确认 `papers/paper.md` 与 `papers/paper.pdf` 的内容来自同一 source of truth.

**方法**:
- 读 paper.md 提取所有 section heading + 关键数字 (R@5/R@10/NDCG)
- 读 paper.tex 同位置提取 (re-grep)
- 数字差 = 0 (delta=0.0000), heading 差 = ∅
- 已知差异: paper.tex 含 `\usepackage{times}` + `\usepackage{fancyhdr}` + `\section*{Acknowledgements}` (Task #100 加的), 但 paper.md 不需要 (markdown 自动渲染), 这部分是 expected diff, 不视为 unsync.

**预期**: 除 format-only 内容外, paper.md 与 paper.pdf 内容完全一致.

### 2.2 Audit #2 — ICML 2026 abstract 合规

**目标**: 验证 abstract 字数符合会议要求.

**方法**:
- 提取 paper.md abstract 段 (regex: `>\s*\*\*Abstract\.\*\*\s*(.*?)---`)
- strip markdown (`**bold**`, `*italic*`, `$math$` → `MATH`)
- word count

**预期**: ≤ 200 字 (ICML 2026 限制). 实测 197 字 ✅.

### 2.3 Audit #3 — Section 5 baseline 覆盖声明

**目标**: 验证 paper §5 / Table 2 中每个 method 都明示 (a) 复现 + verdict 引用 或 (b) 继承 paper-reported 数字 + 显式 not-reproduced label.

**方法**:
- 提取 paper Table 2 (regex 捕获 Category / Method / R@5 / R@10 / R@20 / NDCG@10)
- 对每个 method 在 verdicts/ 找对应 task<N>_*.md
- 标 ✅ reproduced (有 verdict) / ⚠️ inherits paper-reported (无 verdict 但 paper 报告) / ❌ unjustified

**预期**: 8/13 ✅ reproduced (Task #32/58/59/60/61/78/84/85/95), 5/13 ⚠️ inherits (paper-reported baselines). 0/13 ❌ unjustified.

### 2.4 Audit #4 — 数字一致性复审

**目标**: 把 Task #103 (14 claim) 扩到**所有** Section 5 数字 (~ 60+).

**方法**:
- extract all decimal numbers in §5 (R@N, NDCG@N, percentages, κ values)
- 对每个数字 grep verdicts/ 找对应源
- 不在 verdicts/ 的数字 → ⚠️ 未 traceable

**预期**: 60+/60+ traceable, 0 unjustified phantom numbers.

### 2.5 Audit #5 — 第三方 license attribution

**目标**: paper / repo 引用了哪些第三方内容, 是否在 README/REPRODUCE 标注 license.

**方法**:
- 找 paper 中引用的工具 / 模型 / 数据 (sentence-t5-base, T5-small, Amazon Musical_Instruments, snap-research/GRID, M2GNN etc.)
- 对每个在 README.md / REPRODUCE.md 找 license 行

**预期**: 3+ 第三方 (HuggingFace T5/sentence-T5 Apache 2.0, snap-research/GRID Apache 2.0, Amazon dataset per their license). 0 missing.

---

## 3. 决策触发

| 条件 | 决策 |
|------|------|
| 5/5 audits pass | ✅ 闭环. paper submission 阶段 defense ready |
| Audit #1 fail (paper.md/tex 不一致) | ⚠️ 需统一一边 (R11.3 选择 paper.md canonical, paper.tex re-pandoc 一次) |
| Audit #2 fail (abstract > 200 字) | ❌ truncate 改动 |
| Audit #3 fail (有 unjustified baselines, 不在 paper-reported 范围) | ❌ 必须补 verdict 或改 paper 表述 |
| Audit #4 fail (有 phantom numbers) | ⚠️ 需在 verdicts/ 找源, 找不到则修改 paper |
| Audit #5 fail (license 缺失) | ❌ 必须在 README/REPRODUCE 加 license line |

---

## 4. 预算

| 阶段 | 估算时间 |
|------|---------|
| 写 5 个 audit 脚本 / inline checks | ~10 min |
| 跑 audits | ~3 min |
| 写 `papers/SUBMISSION_DEFENSE.md` | ~10 min |
| 写 description + verdict + commit | ~7 min |
| **总计** | **~30 min, 0 GPU** |

---

## 5. 风险与缓解

**风险 1**: paper.tex 与 paper.md 内容差异不易机械化提取 (LaTeX 难 parse)
  → **缓解**: Audit #1 不强求 structural diff, 用 heading 数量 + 关键数字 round check 即可 (paper.tex 14 页, paper.md 556 行, 内容明显 reflect)

**风险 2**: Audit #3 baseline 覆盖不完整 — verdicts/ 有可能不含 paper Table 2 的所有 method
  → **缓解**: 把每个 method 在 verdicts/ 搜不到的情况记为 ⚠️ inherits, 不 fail；最后看是否所有 ⚠️ 都是 paper-reported (TIGER/LETTER/SASRec/...)

**风险 3**: Audit #4 数字追溯耗时, paper 有 60+ 数字
  → **缓解**: 仅审计 §5 + abstract 的数字 (其他章节的数字基本是数学公式, 无需追)

**风险 4**: Audit #5 license 文字可能与 upstream 不匹配
  → **缓解**: paper 中引用过的 upstream (snap-research/GRID / HG-Rec paper) 已在 `papers/refs.bib` 体现, 用 refs.bib 12 entries 作为 license source set

---

## 6. 完成度跟踪

- [x] 写 `descriptions/task106_submission_defense.md` (本文件)
- [ ] 写 `scripts/task106_audits.py` (5 audits 合一脚本)
- [ ] py_compile 验证 (Rule 10)
- [ ] 跑 audits, 收集结果
- [ ] 写 `papers/SUBMISSION_DEFENSE.md` (defense packet)
- [ ] 写 `verdicts/task106_submission_defense_result.md`
- [ ] git commit
- [ ] §16 loop.md 更新 (Task #106 ✅ 已完成)

---

## 7. 关联

- 前置: Task #99-#105 (paper submission 全套已闭环)
- 后置: (无, paper submission 阶段 defense packet 闭环)

---

**核心交付**: 5 项 audit 闭环 + `papers/SUBMISSION_DEFENSE.md` + reviewer 一键可查的 self-defense packet.
