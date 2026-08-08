---
type: verdict
status: "NO-GO"
created: 2026-08-02
tags:
  - kappa
  - phase0
up: "[[index]]"
---
# Phase 0 (球面侧 κ 扫描) — NO-GO with caveat (2026-07-27)

## Step 0 通过 ✅
- **κ>0 = 球面**, κ<0 = 双曲, κ=0 = 欧式 (Berman-Metzler 标准约定, 跟代码一致)
- 球面侧 d_max ≈ π/√κ ✓ (κ=1.0 → d_max=3.1415, κ=3.0 → d_max=1.67 < π/√3=1.81)
- 无 NaN, 无 proj_needed
- dyn_ratio 球面 (1.21-2.10) > 双曲 (1.07-1.44) ✓ 确认动态范围优势

## Phase 0 失败 ❌

| layer | side | κ | agreement | usage | assign_H | dyn |
|---|---|---|---|---|---|---|
| L0 | sphere | +1 | **1.0000** | 0.250 | 1.67 | 1.196 |
| L0 | sphere | +3 | **1.0000** | 0.250 | 1.67 | 1.199 |
| L0 | sphere | +10 | **1.0000** | 0.250 | 1.67 | 1.209 |
| L1 | sphere | +3 | **1.0000** | 0.156 | 1.64 | 1.225 |
| L2 | sphere | +3 | **1.0000** | 0.039 | 0.03 | 1.278 |

**所有 27 个 (3 层 × 9 κ) 测试中, 球面侧都 100% 跟欧式一致**. 

**根因**: E1 baseline 用 NormCap 把 ‖z‖ 压到 0.04-0.19, 码字 ‖c‖ ≈ 0.04.
κ-Stereographic 距离公式 `d_κ = (2/√κ) · atan(√κ·r/(2·|1-κ·r²/4|))` 在 `√κ·r << 1` 极限下退化为 `r` (欧式). 
- √10 · 0.19 = 0.60 → atan(0.30) ≈ 0.29 (vs 欧式 0.19)
- √10 · 0.04 = 0.13 → atan(0.06) ≈ 0.06 (vs 欧式 0.04)
- 在 [0, K=64/128/256] 范围内, 这个微小单调偏差**不改变 argmin 排名** → agreement 100%

## ⚠️ 关键发现: NormCap 跟 κ-Stereographic 球面侧互斥

- NormCap: 把 ‖z‖ ≤ target (e.g. 0.4), 让范数小, 跟 Poincaré 边界饱和对抗
- κ-Stereographic 球面: 需要 ‖z‖ ≈ 1/√κ 量级才有几何区分力 (用户方案原文 ρ=0.847 即是)
- **两者范数尺度差 10-20x**, 球面几何被 NormCap 完全淹没

## 止损判定

按用户方案原文:
> "Phase 0 若球面侧 usage 也崩 → 几何线彻底关闭,转写作。"

但 usage 崩是 **NormCap 副产品**, 不是几何本身. 真实测试条件:
- E1 normcap ckpt → 球面隐形 ✓ (但这是测 NormCap latent, 不是测 κ 几何)
- **真实 κ 几何测试需要在自由范数 latent 上跑**

## 候选下一步 (R11.4 关键决策, 等用户决定)

| 选项 | 含义 | 行动 |
|---|---|---|
| A | 关闭球面线, 转写作 | 写 Phase 0 NO-GO + J-plan NO-GO 合并 verdict |
| B | 取消 NormCap 重做 E1 baseline, 跑 Phase 0 真实几何测试 | 6 h 训新 ckpt, 再 30 min Phase 0 |
| C | 不重训, 在 E1 latent 上**手动放大范数** (e.g. ‖z‖ *= 10) 模拟无 NormCap 状态 | 5 min CPU 重测, 看 agreement 是否下降 |
| D | 用中间 NormCap (e.g. 0.8) 重训 E1 | 1.5 h 训 + 30 min Phase 0 |

按 R11.3 推荐: **C** (5 min 快速验证) → 若 agreement 下降到 60-90% 区间, 进入 B/D 完整重训; 若仍 100%, 关闭球面线.

