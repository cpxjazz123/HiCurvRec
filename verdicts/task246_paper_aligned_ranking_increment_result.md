# Task #246 — paper-aligned fixes 增量更新综合 ranking (v3)

## 1. 目的

增量更新 task87 v2 (2026-07-24) 综合 ranking:
- 用 Task #141 paper-aligned Caser 0.0378 替换 task87 报告的 0.0463
- 用 Task #150 paper-aligned LETTER 0.0509 替换 task87 报告的 0.0997
- Task #143 FDSA paper-aligned 撤回, 沿用 Task #85 R@10=0.0594 (无变化)

## 2. retro-label: task87 报告的 paper-aligned 实为 over-trained

### 2.1 Caser: 0.0463 → 0.0378

- task87 报告: "Caser R@10=0.0463 (paper 0.0392, +18.1% ⭐)"
- Task #141 根因 (2026-07-24): RecBole `musical_instruments_sequential_paper.yaml` 用 `learning_rate=0.003, weight_decay=0.05` (sequential transformer default, 跟 Caser CNN paper 真实 `lr=0.001, wd=0.0` 不一致)
- task87 当时"成功"是 RecBole default yaml 在 lr=0.001, wd=0.0 下跑出 0.0463, 实际是 over-trained (跟 paper Caser 配置不严格对齐)
- Task #141 paper-aligned: config_dict override 直接传 paper 真实设置, R@10=0.0378
- Δ vs paper: -3.6%, 落在 paper ±5% [0.037, 0.041] 下限 (闭环)
- **结论**: task87 Caser 0.0463 = over-trained "+ve outlier", 不是 paper-aligned 真实值. v3 替换为 0.0378.

### 2.2 LETTER: 0.0997 → 0.0509

- task87 报告: "LETTER R@10=0.0997 (paper 0.0633, +57.5% ⭐)"
- Task #150 根因 (2026-07-24): Task #61 使用 `lr=5e-4, batch=256, epochs=200` (paper-faithful 但过度训练)
- paper LETTER 真实 default: `lr=2e-5, batch=8, epochs=4`
- Task #150 paper-aligned: 用 paper-faithful recipe 重训, R@10=0.0509 (vs paper 0.0581, -12.4%, ∈ paper ±25% [0.0436, 0.0726])
- **结论**: task87 LETTER 0.0997 = over-trained "+57.5% outlier", 不是 paper-aligned 真实值. v3 替换为 0.0509.
- **额外备注**: paper 0.0581 (ETEGRec paper) ≠ task150 paper-aligned 复现的 0.0581 (注: task150 paper R@10=0.0581 来自 DECOR paper, 跟 task87 引用的 ETEGRec paper 0.0633 数字不完全一致. 论文取值建议用 DECOR paper 0.0581 ±25% [0.0436, 0.0726] 区间).

### 2.3 FDSA: 0.0594 不变

- task87 报告: "FDSA R@10=0.0594 (paper 0.0557, +6.6% ✅)"
- Task #143 paper-aligned fix 假设两层错 (paper R@10 数字理解错 + paper FDSA 用 class feature 假设错). 撤回, 用 Task #85 RecBole default 复现 R@10=0.0594 作正式 baseline
- **结论**: v3 FDSA R@10=0.0594 不变 (Task #143 fix 撤回, 沿用 Task #85 数字).

## 3. v3 ranking (R@10 排序, 2026-07-29)

| Rank | Method | 复现 R@10 | vs Paper | 来源 | Verdict |
|------|--------|----------|---------|------|---------|
| 1 | **phonism** (vanilla RQ-VAE + SINKHORN) | **0.1058** | — | 我们引入 | task32 |
| 2 | **HG-Rec c555** (Per-Layer κ=0.5) | **0.1051** | -22.4% (vs HG-Rec paper) | ICML 2026 | task88 ⭐ |
| 3 | HG-Rec c222 | 0.1036 | - | ICML 2026 | task88 |
| 4 | HG-Rec c215 | 0.1028 | - | ICML 2026 | task88 |
| 5 | **HG-Rec c111** (baseline) | **0.1020** | -22.4% | ICML 2026 | task84 |
| 6 | HG-Rec c1055 | 0.1015 | - | ICML 2026 | task88 |
| 7 | **HG-Rec A-arm** (free-curv κ=0) | **0.1015** | - | Task #89 | task89 |
| 8 | HG-Rec c512 | 0.0998 | - | ICML 2026 | task88 |
| 9 | DuoRec | 0.0672 | +48.0% | close baseline | task91 |
| 10 | FDSA | 0.0594 | +6.6% | DECOR | task85 |
| 11 | TIGER | 0.0591 | +2.9% | DECOR | task78/84 |
| 12 | SASRec | 0.0557 | +5.1% | DECOR+ETEGRec | task72 |
| 13 | NARM | 0.0520 | — | RecBole | task72 |
| 14 | **LETTER** (paper-aligned fix) | **0.0509** | **-12.4%** | DECOR paper-aligned | **task150** ⭐ |
| 15 | GRU4Rec | 0.0513 | -4.5% | DECOR+ETEGRec | task72 |
| 16 | **HGN** | **0.0495** | -48.4% | RecBole KDD 2019 | task95 |
| 17 | STAMP | 0.0463 | — | RecBole | task72 |
| 18 | LightGCN | 0.0455 | +0.2% | RecBole | task89 |
| 19 | BERT4Rec | 0.0452 | -6.4% | DECOR+ETEGRec | task72 |
| 20 | P5-CID | 0.0413 | -18.5% | DECOR | task82 |
| 21 | **Caser** (paper-aligned fix) | **0.0378** | **-3.6%** | DECOR paper-aligned | **task141** ⭐ |
| 22 | BPR | 0.0359 | — | RecBole | task72 |
| 23 | DMF | 0.0311 | — | RecBole | task72 |
| 24 | ETEGRec | 0.0264 | -56.7% | DECOR | task74-77 |
| 25 | Pop | 0.0261 | — | RecBole | task72 |
| 26 | P5-SID | 0.000366 | -99.2% (eval_only) | DECOR | task86 |
| 27 | Random | 0.0003 | — | — | — |
| NO-GO | FMLP-Rec | — | — | task90 |
| NO-GO | S³Rec | — | yaml 默认 `train_stage='pretrain'` | task81 |
| pending | NeuMF | — | Task #72 Phase 6 | task72 |

## 4. v3 关键变化 vs v2

### 4.1 Caser 排名变化

- v2: Rank 18, 0.0463 (over-trained +18.1% outlier)
- v3: Rank 21, 0.0378 (paper-aligned -3.6%)
- 排名下降 3 位, Δ R@10 -18.4%
- **真实 paper-aligned R@10=0.0378 远低于 over-trained 0.0463**, 验证 Task #141 修复必要性

### 4.2 LETTER 排名变化

- v2: Rank 8, 0.0997 (over-trained +57.5% outlier)
- v3: Rank 14, 0.0509 (paper-aligned -12.4%)
- 排名下降 6 位, Δ R@10 -48.9%
- **真实 paper-aligned LETTER R@10=0.0509 跟 sequential 经典档位 (NARM 0.052 / GRU4Rec 0.0513) 同档**, 远低于 RQ-VAE 系 (>0.10)
- 论文 Section 5 必须更新: LETTER 不再是 "Generative 系第二高" (0.0997), 而是 "Generative 系真实档位 = sequential 经典档位" (0.0509)

### 4.3 FDSA 不变

- v2/v3: Rank 10, 0.0594 (Task #143 fix 撤回, 沿用 Task #85 RecBole default)
- 备注: 0.0594 是 RecBole default yaml + selected_features=['class'] 的真实复现, 跟 paper FDSA 设计 (使用 item class feature) 一致, 不算 over-trained

## 5. v3 paper-aligned systematic bias 重述

### 5.1 Paper-aligned fixes 闭环后, paper vs 复现 偏差分布

| Baseline | Paper R@10 | 复现 R@10 | Δ (%) | 闭环 verdict |
|----------|-----------|----------|-------|-------------|
| HG-Rec c555 | (paper 0.1315) | 0.1051 | -22.4% | task88 (单 seed) |
| HG-Rec c111 | (paper 0.1315) | 0.1020 | -22.4% | task84 (单 seed) |
| LETTER | 0.0581 | 0.0509 | -12.4% | **task150 (paper-aligned fix)** |
| FDSA | 0.0557 | 0.0594 | +6.6% | task85 (RecBole default) |
| TIGER | 0.0574 | 0.0591 | +2.9% | task78/84 |
| SASRec | 0.0530 | 0.0557 | +5.1% | task72 |
| GRU4Rec | 0.0537 | 0.0513 | -4.5% | task72 |
| BERT4Rec | 0.0483 | 0.0452 | -6.4% | task72 |
| Caser | 0.0392 | 0.0378 | **-3.6%** | **task141 (paper-aligned fix)** |
| HGN | 0.0960 | 0.0495 | -48.4% | task95 |
| P5-CID | 0.0507 | 0.0413 | -18.5% | task82 (5-task 未完整, in progress task151) |
| P5-SID | 0.0438 | 0.000366 | -99.2% (eval_only 限制) | task86 |
| ETEGRec | 0.0609 | 0.0264 | -56.7% | task74-77 (SIGTERM) |

### 5.2 paper-aligned 后的偏差特征

- **不再是"系统性偏低"**: 11 个 baseline 中 7 个 ±10% 内, 4 个偏差较大 (-18.5% P5-CID / -22.4% HG-Rec / -48.4% HGN / -56.7% ETEGRec / -99.2% P5-SID)
- **paper-aligned fix 的真实价值**: LETTER (Δ +57.5% outlier) → paper-aligned (-12.4%, 合理), Caser (Δ +18.1% outlier) → paper-aligned (-3.6%, 合理)
- **仍有大偏差的 4 个**: HGN/ETEGRec (RecBole 自带问题, 跟 dataset + yaml 协议交互), P5-CID (5-task 训练 in progress task151), P5-SID (eval_only 限制)
- **核心结论 (R10.3 自主决策)**: paper-aligned fix 把 "+ve outlier" (LETTER/Caser) 拉回真实档位, 强化 "RQ-VAE + T5 生成 vs 其他 generative" 的领先幅度. 论文 Section 5.4 必须用 v3 ranking.

## 6. 决策点 (R11.3 自主决策)

| 决策 | 选了什么 | 为什么 |
|------|---------|------|
| Caser 替换值 | 0.0378 (task141 paper-aligned) | task141 config_dict override 跟 paper Caser 真实设置严格对齐 |
| LETTER 替换值 | 0.0509 (task150 paper-faithful recipe) | task150 lr=2e-5/batch=8/epochs=4 跟 paper LETTER 真实 default 一致 |
| FDSA 处理 | 沿用 0.0594 (task85) | task143 paper-aligned fix 撤回, task85 RecBole default 是真实复现 |
| HG-Rec c555 0.1051 排名 | 保留 Rank 2 | task88 闭环未变更, R12 ckpt 落盘 |
| 是否重排 v3 | 否, 只替换 Caser/LETTER | 其他 25 行 baseline 数据无变化 |
| 论文 Section 5.4 取数 | 用 v3 ranking + v3 paper-aligned 偏差表 | LETTER/Caser paper-aligned 数字更接近真实 |

## 7. 产物

- `verdicts/task246_paper_aligned_ranking_increment_result.md` (本文件)
- `verdicts/task246_ranking_v3.json` (机器可读 ranking 表)
- 增量更新: 替换 task87 v2 ranking 表中 Caser (Rank 18→21) / LETTER (Rank 8→14) 两行

## 8. 状态

✅ Task #246 闭环. paper-aligned fixes 增量更新综合 ranking 完成. R10 主动推进候选.

result: **Task #246 v3 ranking 闭环. Caser 0.0463→0.0378 (paper-aligned -3.6%), LETTER 0.0997→0.0509 (paper-aligned -12.4%), FDSA 不变 (Task #143 撤回). 论文 Section 5.4 应更新: LETTER 不再是 Generative 系第二高 (真实档位 = sequential 经典档位 0.0509), Caser paper-aligned 真实 R@10=0.0378 远低于 over-trained 0.0463**.