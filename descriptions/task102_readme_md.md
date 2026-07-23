# Task #102 — README.md 顶层落地页

> **任务目的**: 创建根目录 `README.md`, 服务 paper reviewer 落地页. 与 CLAUDE.md (AI agent 内部指引) 和 REPRODUCE.md (复现执行步骤) 严格定位区分: README.md 是人类第一接触入口, 给出论文摘要 / 关键数字 / 三件套入口 (papers/REPRODUCE/src) / 引用 / License / 联系.

> **完成日期**: (in progress)
> **状态**: 🟡 待启动

---

## 1. 背景

Task #98 paper.md / Task #99 paper.pdf / Task #100 submission 准备 / Task #101 REPRODUCE.md 复现包 都已闭环. 但**根目录缺乏面向 paper reviewer / 外部协作者 / 自我归档场景的入口文件**. 评审人打开仓库, 首先看到的是 CLAUDE.md (AI agent 指南, 不适合人类读) 或散落目录, 没有 TL;DR + 关键数字 + 三件套 (paper+reproduce+code) 引导.

CLAUDE.md 已经存在 → 不能与 README.md 重复, 需明确分工:
- **README.md**: 对外人类入口, 介绍这是什么论文 / 关键数字 / 怎么跑 / 怎么引用 / License
- **CLAUDE.md**: 对内 AI agent 指引 (R1-R13 规则), 不变

## 2. 实验设计 (writeup only)

### 2.1 README.md 7 sections

1. **TL;DR** — 一句话 + 两个关键数字 (vanilla RQ-VAE + Sinkhorn 0.1058 vs HG-Rec κ=0.5 0.1051)
2. **What's in this repository** — 仓库结构 (papers/, src/, data/, descriptions/, verdicts/, products/, scripts/ + 上游框架 BLOGER/DECOR/HG-Rec/LETTER/RecBole/RippleNet 提示)
3. **Headline Findings** — 4 个核心结论 (hyperbolic marginal / Sinkhorn beats hyperbolic / token set Jaccard but item assignment distinct / systematic dataset-protocol gap)
4. **Reproducing the Results** — 一句话 + REPRODUCE.md 链接
5. **Citation** — BibTeX entry + 指向 refs.bib
6. **Acknowledgements** — snap-research/GRID + HG-Rec authors + GPU 集群 + RecBole team
7. **License** — MIT (paper + reproduction artifacts) + Apache 2.0 (upstream) + 各 framework 各自 license
8. **Contact** — issue tracker 引导
9. **Reading order** — newcomer 阅读路径

### 2.2 严格分工

| 文件 | 读者 | 内容 |
|------|------|------|
| README.md | 人类第一接触 | TL;DR + 数字 + 入口链接 |
| REPRODUCE.md | 想跑实验者 | Stage 1-4 命令 + 数据集 + 验证脚本 |
| CLAUDE.md | AI agent | R1-R13 规则 + 路径约束 |

不重复内容, README 给出**摘要+入口**, 不复制 REPRODUCE.md 的 Stage 命令.

## 3. 决策触发

| 条件 | 结果 |
|------|------|
| README.md 落盘 + 含 TL;DR 数字 + 论文/复现/code 三入口链接 + License | ✅ 闭环 |

## 4. 预算

| 阶段 | 估算时间 | GPU |
|------|---------|-----|
| 写 README.md | ~25 min | 0 |
| 写 description | ~5 min | 0 |
| 写 verdict + commit | ~10 min | 0 |
| **总计** | **~40 min** | **0 GPU** |

## 5. 风险与缓解

**风险 1**: 与 CLAUDE.md 内容重叠造成评审困惑
  → **缓解**: 明确 README.md §7 标注 "reading order", 并在文件顶部说 "human-facing landing, see CLAUDE.md for AI agent protocol"

**风险 2**: License 措辞错误 (未点名 upstream)
  → **缓解**: 写双 License (MIT + Apache 2.0 with explicit upstream attribution), 引用 LICENSE 目录的 norm

## 6. 产物清单

- `README.md` — 顶层落地页 (~ 200 行)
- `descriptions/task102_readme_md.md` — 本任务描述
- `verdicts/task102_readme_md_result.md` — 闭环报告
- git commit 含 README.md + description + verdict

## 7. 关联

- 前置: Task #98 paper.md / Task #99 paper.pdf / Task #100 submission / Task #101 REPRODUCE.md
- 后置: (无, 仓库人类入口闭环)

---

**核心交付**: README.md (人类第一接触入口, ~200 行, 含 TL;DR + Headline Findings + 三入口 + License).
