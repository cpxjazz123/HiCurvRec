---
type: verdict
status: "NO-GO"
created: 2026-08-02
tags:
  - misc
up: "[[index]]"
---
# J-plan Phase 1 anchor sweep — NO-GO (2026-07-27)

## 决策依据 (R11.3 自主决策)

| w_anchor | best @ ep29 | 末 collision | 趋势 |
|---|---|---|---|
| (J-plan 0.1) | 0.92 (前次) | 0.93 | 死水位 |
| 0.05 | 0.93 | 0.93 | 死水位 |
| 0.01 | 0.85 | 0.85 | 恶化 |
| 0.001 | 0.73 | 0.73 | 振荡不收敛 |
| **(J0 baseline = 0)** | **0.1035 ✅** | **0.1035** | **干净收敛** |

**结论**: 即便把 anchor 缩到 0.001 (J0 anchor loss 量级 ~1e-3), 仍 73% collision. 说明
- 跟 NormCap(‖x‖=0.4 → ρ≈0.42) 强冲突 (anchor 要 ρ ∈ [1.5, 2.9])
- 类别深度跟 baseline latent 半径**本来不相关** (Phase 0J Spearman=-0.0045), anchor 强行加约束 = 拽出不存在的关系

## Phase 0J 复盘: 闸门 D 的解读错位

- 原闸门: D. Spearman ≈ 0 (有改进空间)
- 我原解读: "无相关 → anchor 可以建" (把这个视为机会)
- **正确解读**: "无相关 → 数据里没这种信号, 强加约束 = 破坏" 

J-plan 全部前提 (锚点能创造层级几何结构) 在 Musical_Instruments 数据上**不成立**.

## 终止决策

- ❌ J-plan anchor sweep 全档终止
- ❌ w=1.0 (J-plan 提的二档) 永不跑 (只会更糟)
- ✅ **J0 (E1+simvq+normcap, w_anchor=0)** = J-plan 实际终态 baseline (collision 0.10)

## 下一步选项 (待用户决策)

| 选项 | 内容 | ROI |
|---|---|---|
| A. 接受 J0 = E1+simvq = J-plan 终态 | 直接做 Stage 2/3/4 走 R@10 验证 | 最高 (2-3 h) |
| B. 改 anchor 形式 | 不靠 (ρ-ρ_target)², 用类别 path embeddings 注入 angular 维度 | 中 (1 天) |
| C. 改监督粒度 | 不监督半径, 监督**码字分配** (不同类别 → 不同码) | 中 (1 天) |
| D. 放弃类别监督, 转入反作用通道 | 攻前提 a/b/d (Gromov / Per-codeword κ) | 高 (per memory `escape-routes-attack-premises-b-and-d.md`) |

按 R10 主动推进: 默认 A (最高 ROI, 验证 J0 是否优于 HG-Rec baseline R@10=0.1020).

## 关键数字存档

- J0 (本项目当前最强 baseline): collision **0.1035** (E1+simvq+normcap+per_codeword_kappa, w_anchor=0)
- HG-Rec baseline (Task #84): R@10 = 0.1020 (Musical_Instruments)
- J0 vs HG-Rec baseline 下游 R@10 = **待 Stage 2/3/4 验证**

