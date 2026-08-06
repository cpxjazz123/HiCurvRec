---
type: vault-guide
zh: "Vault 入口 — 方向A 4 个 + 方向B 4 个 + 3 个 reference"
created: 2026-08-05
tags:
  - guide
  - entry
---

# Vault 入口 (EXPERIMENT_GUIDE)

> **本 vault 主轴**: 按"方向 (A/B)"组织 — 方向A 4 个 + 方向B 4 个真实验 + 3 个 reference。
> **所有文件名 = 方法名** (中文直述，不出现编号)。

## 方向A 真实验 (4 个)

方向A = taskA 目录，主跑 hyp 路线 + 适配器路线：

- [[方向A 用 Hyp 端到端配方达成历史最佳|方向A 用 Hyp 端到端配方达成历史最佳 [BREAKTHROUGH]]]
- [[方向A 试图用 trust region + anchored 稳定 κ|方向A 试图用 trust region + anchored 稳定 κ [NO_GO]]]
- [[方向A 用距离分桶替换 KMeans 负采样|方向A 用距离分桶替换 KMeans 负采样 [NO_GO]]]
- [[方向A 用 T5 stage3 wrapper 注入 κ|方向A 用 T5 stage3 wrapper 注入 κ [MIXED]]]

## 方向B 真实验 (4 个)

方向B = taskB 目录，主跑 baseline 收尾 + T5 容量 + issue 收口：

- [[方向B 用 T5 容量 sweep (mini／small／base) 找甜点位|方向B 用 T5 容量 sweep (mini/small/base) 找甜点位 [MIXED]]]
- [[方向B 逐个裁定早期 issue 并定位塌缩根因|方向B 逐个裁定早期 issue 并定位塌缩根因 [MIXED]]]
- [[方向B 收口历史 open issue|方向B 收口历史 open issue [FAIL]]]
- [[方向B 建立 HG-Rec 端到端基线|方向B 建立 HG-Rec 端到端基线 [BASELINE]]]

## 3 个 reference (reference/)

方向A 和方向B 共用的辅助资料 — 诊断/扫描/housekeeping，不计入 status 分类：

- [[方向A 和方向B 共用的 α-sweep + beam20 reeval 数据集|方向A 和方向B 共用的 α-sweep + beam20 reeval 数据集]]
- [[方向A 和方向B 共用的 19 份诊断报告|方向A 和方向B 共用的 19 份诊断报告]]
- [[方向A 和方向B 共用的 Loop tick housekeeping 日志|方向A 和方向B 共用的 Loop tick housekeeping 日志]]

## 6 种"实验情况"节点 (status/)

- [[HG-Rec 端到端基线方法|HG-Rec 端到端基线方法 (BASELINE, A=0 B=1)]]
- [[Hyp 路线端到端配方方法 (历史最佳)|Hyp 路线端到端配方方法 (历史最佳) (BREAKTHROUGH, A=1 B=0)]]
- [[稳定化 κ 路径全部放弃方法|稳定化 κ 路径全部放弃方法 (NO_GO, A=2 B=0)]]
- [[Issue 收口裁定与协议诊断方法|Issue 收口裁定与协议诊断方法 (FAIL, A=0 B=1)]]
- [[系列内多 step 拆解方法|系列内多 step 拆解方法 (MIXED, A=1 B=2)]]
- [[扫描 sweep 数据集收集方法|扫描 sweep 数据集收集方法 (DATA, A=0 B=0)]]

## 9 种主题方法节点 (topic/)

- [[基线复现方法 (HG-Rec 原论文流水线)|基线复现方法 (HG-Rec 原论文流水线)]]
- [[Hyp 路线方法 (per-item radius + 自由 κ)|Hyp 路线方法 (per-item radius + 自由 κ)]]
- [[方向A 适配器方法 (T5 stage3 wrapper)|方向A 适配器方法 (T5 stage3 wrapper)]]
- [[方向B T5 容量 sweep 方法 (mini／small／base)|方向B T5 容量 sweep 方法 (mini/small/base)]]
- [[方向B 早期 issue 裁定方法 (Stage 2／3 防线建设)|方向B 早期 issue 裁定方法 (Stage 2/3 防线建设)]]
- [[α-sweep + beam20 reeval 数据集方法 (方向A／B 共用)|α-sweep + beam20 reeval 数据集方法 (方向A/B 共用)]]
- [[诊断报告方法 (19 份分主题审计, 方向A／B 共用)|诊断报告方法 (19 份分主题审计, 方向A/B 共用)]]
- [[Loop tick housekeeping 方法 (方向A／B 共用)|Loop tick housekeeping 方法 (方向A/B 共用)]]
- [[方向B Issue 收口裁定方法 (含 #76 SID-incompat)|方向B Issue 收口裁定方法 (含 #76 SID-incompat)]]

## 关于本 vault

- **本 vault 是 HG-Rec 实验的可视化整理**，与 verdicts/ 源数据完全分离
- **verdicts/ 内容不引用本 vault**：保持历史 verdict 文件纯净，避免破坏 pipeline
- **命名约定**: 文件名 = 方法名 (中文直述)，不出现任何编号
- **方向约定**: 方向A = taskA 目录，方向B = taskB 目录 (按 CLAUDE.md R25)
- **从 Windows 打开**: 下载本目录 (`~/Downloads/experiments_vault.tar.gz`) 解压后用 Obsidian File → Open vault as library 打开即可

## 数据文件

- `relations.json` — 节点关系图 (status / topic / experiment / reference)
- `experiments_index.json` — 索引 (按 category + direction 区分)
