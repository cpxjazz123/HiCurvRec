---
type: audit
created: 2026-08-02
tags:
  - phase0
up: "[[index]]"
---
# Phase 0 (J-plan) 数据审计 — 2026-07-27

**目的**: 类别锚定半径方案 (用户新方向) 必须先过 — 路径深度必须有变化.

**数据**: Instruments.item.json = 9922 items; categories 字段为逗号分隔层级路径.

## 结果

### A. 路径深度分布

| depth | count | % |
|---|---|---|
| 0 (无字段) | 261 | 2.63% |
| 2 | 175 | 1.76% |
| 3 | 882 | 8.89% |
| **4** | **3787** | **38.17%** ←众数 |
| **5** | **3536** | **35.64%** ←次众数 |
| 6 | 1137 | 11.46% |
| 7 | 138 | 1.39% |
| 9+ 散点 | 6 | 0.06% |

- 有效 (depth≥2): n=9661 (**97.37%**)
- **mean=4.52, std=0.94**, min=2, max=16
- quartiles [25/50/75] = [4.0, 4.0, 5.0]
- ✅ **std=0.94 ≫ 0.5**, 远超深度变化闸门.

### B. 各层类别数

| 层 | distinct | top-3 |
|---|---|---|
| L1 | **1** | Musical Instruments (9661) ← 数据集根唯一 |
| L2 | 21 | Instrument Accessories / Live Sound / Amplifiers |
| L3 | **67** | Guitar & Bass Accessories / Guitar Effects / Cables |
| L4 | 196 | Electric Guitar Effects / Parts / Strings |
| L5 | 160 | Single Effects / Picks / Straps |
| L6 | 80 | Distortion / Drumsticks / Reverb |
| L7 | 17 | Flange / Cases / Mixers |
| L8+ | 极少 | (噪声, 多为 description 串误入) |

- ❌ L1=1 (Amazon 数据集特性: 所有 Musical_Instruments items 共享唯一根 "Musical Instruments")
- ✅ L2=21 (≥5 满足, 调整后)
- ✅ L3=67 (≥50 满足)
- ✅ L4=196 (实际可锚定层级)

### C. 覆盖率
- **97.37%** (≥80% ✅ 远超)

### D. baseline Spearman (item_emb ‖x‖_E vs 路径深度)
- n=9661, **ρ = -0.0045, p=0.66**
- ✅ 完全无相关 — 监督空间 100% 开放, J-plan 完全有改进空间.

## 闸门判定 (按用户原文)

| 闸门 | 标准 | 实测 | 通过 |
|---|---|---|---|
| A. 深度 std | ≥ 0.5 | 0.94 | ✅ |
| C. 覆盖率 | ≥ 80% | 97.37% | ✅ |
| B1. L1 类别数 | ≥ 5 | **1** | ❌ |
| B3. L3 类别数 | ≥ 50 | 67 | ✅ |
| D. Spearman | ≈ 0 | \|ρ\|=0.005 | ✅ |

**整体: ❌ 按字面不通过 (L1=1)**.

## 关键解读 (R11.3 自主决策思路)

### L1=1 是 Amazon 数据集**结构特性** 不是数据问题

Amazon 商业分类从根开始 ("Musical Instruments" → 子分类), 所有 items 都共享这一根.
这跟"数据集没有类别路径"是完全两回事 — 实际有 6 级层级 (从 L2 起算深度变化丰富).

### 用户的核心闸门是"深度有变化" (A)

std=0.94 远超 0.5 — 这是 J-plan **能 work** 的充分条件.
L1=1 跟"用类别监督锚定半径"是**正交**的两件事 — 后者利用的是 L2-L7 的丰富结构.

### 我建议的两种解读 (供用户决策)

**方案 1 — 严格按字面**: L1=1 不通过 → 停, 换 F/H 方案.
**方案 2 — 重新解读 L1 为 "L2 (第一层实质性分类) ≥ 5"**: 通过 (L2=21 ≥ 5) → 进 Phase 1.

### 我的推荐: 方案 2

理由:
- L1=1 是 Amazon 商业分类结构, 任何 Musical_Instruments 都长这样
- 用户原意"各层类别数"是 sanity check, 保证监督信号有信息量 — L2=21 完全达标
- 实际可用锚的层级 L2-L5 都丰富 (21, 67, 196, 160 distinct)
- 严格停 = 浪费 1 小时拿到一个**已知**的Amazon 数据集特征
- 走 F/H 等于回到 Phase 0B, 跟现在的 dead_revive/simvq 实验冲突

## 后续

**如果方案 1 (停)**:
- Phase 0B 持续跑完, 等待用户下个方向

**如果方案 2 (进)**:
- J1 (类别锚定 + 欧式) 必须搭 category-aware 数据接口 (`category_depth[item_i]`)
- 新代码改动: `data/category_anchor.py` + `model/hrqvae.py` 加 `w_anchor * ((rho_actual - rho_target)**2).mean()`
- `rho_target[i] = rho_min + (rho_max - rho_min) * (depth_i - 1) / (max_depth - 1)`, ρ∈[1.5, 2.9]
- **必须同时加 dead_revive** (已在 phase0b 实施里)
- J2/J3 走 d=16 + c_base=9 配置 (参 Phase 0A verdict)
