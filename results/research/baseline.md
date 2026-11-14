# Stage2 参照基线：`TIGER_RQ-VAE`（owner 决定 2026-11-14；`TIGER_mixed_rq_F0` 转为历史对照）

> **owner 决定（2026-11-14）：只做 TIGER，永远只做 TIGER。**
> 从这条起，所有新机制都建在 **`TIGER_RQ-VAE`**（E32 全欧氏）上；`TIGER_mixed_rq_F0` 与本族
> 的角色路由臂（B/C/D）**留档为历史证据，不再投入**。下面的数字全部保留、全部可复核，只是主参照换人。

## 0. 参照基线的变更记录

| 日期 | 主参照 | 依据 |
|---|---|---|
| 2026-11-12 | `TIGER_mixed_rq_F0` = 0.057278156 | Issue #211 |
| **2026-11-14** | **`TIGER_RQ-VAE` = 0.056216** | owner 决定：只做 TIGER，永远只做 TIGER |

## 0b. 当前主参照：`TIGER_RQ-VAE`

| 项 | 值 / 路径 |
|---|---|
| 臂名 | **`TIGER_RQ-VAE`**（E32 全欧氏、三层三 token） |
| `test_recall@10` | **0.056216** |
| SID 表 | `analysis_sid_recall/r2/24/item_sids.json`（sha256 `53273cb605dc3210c20ad2ae5da778ce6fe7fd26bd8b3d6e356c2c7e0e7eed85`） |
| 同 SID 重跑（2026-11-14，冻结协议） | **0.054719** —— 同 SID、同协议两次运行差 **2.7%**，是本 issue 所有"臂间差"的噪声下限 |
| 源码树 | `analysis_sid_recall/r2/24/TIGER_RQ-VAE/`（归档副本，sha 逐文件核对） |

---

本文件是 Issue #211 的落地物，**不含任何训练**，也不改动已产出的任何结果或历史结论。

## 1. 参照基线

| 项 | 值 / 路径 |
|---|---|
| 臂名 | **`TIGER_mixed_rq_F0`** |
| Stage3 结果文件 | `results/stage3_T5Train/TIGER_mixed_rq_F0/logs/Amazon_2023_Instruments/Nov-09-2026_08-07-21/test_final.json` |
| `test_recall@10` | **0.057278155956754125** |
| `n_eval` | 57,439 |
| SID 表 | `results/stage2_RQ-VAE/TIGER_mixed_rq_F0/item_sids.json`（24,587 商品） |
| SID sha256 | **`afa89a10b1b197a7b56da4ad322c6ddd96a3c26c04238af326d88747378bb848`** |
| SID 结构（复算） | L1 使用码 79 · 二级前缀 4,151 · 不同三码 18,264 |
| 源码树 | `stage2_RQ-VAE/TIGER_mixed_rq_F0/` |

复算方式（可重跑）：读 `test_final.json` 取 `test_recall@10`；对 `item_sids.json` 求 sha256 与前三个码的去重计数。上表数字与文件内容一致。

## 2. 配方（**无结构项**）

E16×H16 原生联合码本、每层 256 联合码、三层三 token、α 混合（`ALPHA_INITIAL = 0.5`）、Sinkhorn 均衡分配（§19h）、`curvature = 1`、3000 epoch、seed 42、全局 batch 4096、四卡 50202、Stage3 冻结协议。
**没有**结构项、学习曲率、成对/对比项、蒸馏项、径向变换。

## 3. 历史对照（保留，不作主参照）

| 基线 | `test_R@10` | 说明 |
|---|---|---|
| `TIGER_RQ-VAE` | 0.056216 | E32 欧氏、三层三 token；2026-11-12 之前的参照 |
| **`TIGER_mixed_rq_F0`** | **0.057278156** | 本族允许类上限，**自此为主参照**；对 0.056216 为 **+1.9%** |

**注意（#198 的教训）**：不得用 `TIGER_RQ-VAE` 的 SID 侧数字（如 L1 247 码、二级前缀 7,414）与 E16×H16 族比较——那是不同架构，会把族差异误读成机制效应。

## 4. 声明

**自此以后，Stage2/Stage3 候选的参照基线为 `TIGER_mixed_rq_F0 = 0.057278156`**；`TIGER_mixed_rq = 0.056216` 仅作历史对照列保留。任何"机制有效"的主张都必须同时给出**双列差值**：

```
Δ(F0)      = recall − 0.057278156
Δ(历史基线) = recall − 0.056216
```

两者都必须出现，禁止只报其中一列。
