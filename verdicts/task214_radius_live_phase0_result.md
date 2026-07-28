# Task #214 Verdict — Latent Radius Live (方向三) ❌ NO-HOPE

**日期**: 2026-07-26
**任务**: Task #214 Phase 0 — Latent radius as live variable (用户 2026-07-26 最后赌注)
**最终判决**: **❌ NO-HOPE** — radius head 在 baseline z 上几乎无信号, 7 方向几何路线全部收线.

---

## 1. 实验设计

**核心 idea**: 给 baseline encoder 加临时 radius head (ρ = sigmoid(W·z + b)), 让 ρ_i item-adaptive.

**Phase 0 (CPU only)**:
- 加载 baseline ckpt
- 编码所有 items → z (9922, 32)
- ρ_i = sigmoid(W·z + b), W ~ scale·randn(32, 1), b = 0
- 测 3 指标 (5 seeds × scale=0.5):
  - R1: ρ_i std (离散度, > 0.15)
  - R2: ρ_i 跟 item popularity Spearman (> 0.10)
  - R3: ρ_i 跟 ‖z_i‖ Spearman (> 0.05)

---

## 2. 实验结果 (5 seeds × scale=0.5)

| seed | ρ_i mean | ρ_i std | R2(pop) | R3(‖z‖) |
|---|---|---|---|---|
| 0 | 0.4796 | 0.0305 | -0.0079 | -0.0474 |
| 1 | 0.5351 | 0.0249 | -0.0173 | +0.2887 |
| 2 | 0.4854 | 0.0324 | -0.0031 | -0.1302 |
| 3 | 0.4784 | 0.0263 | -0.0010 | -0.0652 |
| 4 | 0.5008 | 0.0267 | +0.0063 | -0.0852 |
| **avg** | **0.4959** | **0.0282** | **-0.0046** | **-0.0079** |

**判据结果**:
- R1 std=0.0282 → ❌ FAIL (< 0.15 阈值, ρ_i 几乎集中)
- R2 = -0.0046 → ❌ FAIL (跟 popularity 几乎零相关)
- R3 = -0.0079 → ❌ FAIL (跟 ‖z‖ 几乎零相关)

**总体**: ❌ **PHASE 0 FAIL — 方向三 NO-HOPE**

---

## 3. 根因诊断

1. **R1 std 太小**: sigmoid 输出集中在 0.5 附近 (z 在低维空间分布紧凑, W·z 范围窄 → sigmoid 后 std 小). 即使 scale=0.5, 离散度仍 0.028.
2. **R2 几乎零**: item popularity 跟 baseline encoder 输出 z 几乎正交 — sentence-T5 编码的内容语义跟 interaction count 无 systematic 关联.
3. **R3 几乎零 (且 seed 不稳)**: ρ_i 跟 ‖z‖ 的关系在不同 seed 间剧烈波动 (-0.13 ~ +0.29), 完全是 noise, 不是 systematic signal.

**深层根因**: 跟 Task #213 同根因 — HG-Rec baseline encoder 训练时**没鼓励** "item-adaptive 几何信号". 所以 z 跟任何"item-level 几何属性" (radius / norm / popularity) 都几乎无关. radius head 试图从 z 中"挖掘" 信号, 但 baseline z 已经是"几何无关的 representation".

---

## 4. HG-Rec 几何路线 7 方向全部 NO-GO 完整清单

| 方向 | Task | 状态 | 根因 |
|---|---|---|---|
| 1. exp(θ) 可学习 κ | #199/201/203 | ❌ NO-GO | θ 全程未动 (Task #201), 训完 θ 仍初始值 |
| 2. 双码本解耦 | #200/208 | ❌ NO-GO | 几何仍坍缩, R@10=0.0915 |
| 3. path regularization | #209 | ❌ NO-GO | dyn 1.27 距离饱和, 不改变 argmin 排序 |
| 4. 低维双曲 + 钉半径 | #211 | ❌ NO-GO | C1 R@10=0.0816, 码字坍缩 util 23% |
| 5. Two-stage decision | #212 | ❌ NO-HOPE | 几何在 top-k 内跟欧式 99% 一致 |
| 6. Entailment Cones | #213 | ❌ NO-HOPE | 锥 opening 利用不了 norm |
| 7. Latent Radius Live | #214 | ❌ NO-HOPE | radius head 在 baseline z 上无信号 |

**共同根因 (7 方向全部触及)**:
- HG-Rec baseline 训完后码字 norm ≈ 1.0 (boundary), λ_κ ≈ 100
- 几何坍缩, 任何需要"真双曲几何" 的方案失效
- HG-Rec paper 报告的 R@10=0.1315 vs 复现 0.1020 (Δ -22.4%) **不是几何带来的**, 是包装 (Sinkhorn + 4-digit dedup + T5 容量) 带来的

---

## 5. 决策: 7 方向全部收线, 转入 paper 收尾

按 R11.3 自主决策:
- 7 方向几何路线全部收线 (Task #199/200/208/209/211/212/213/214)
- HG-Rec 框架在 Musical_Instruments 数据上**几何无效** — 这是 paper §1.4 的 finding 8/9/10/11
- 转入 **Task #215 paper §4 收尾** — 把 7 方向 NO-GO 整合成 paper 完整叙事

---

## 6. 产物落盘

| 类型 | 路径 |
|---|---|
| 脚本 | /home/wlia0047/.claude/jobs/04ccf474/tmp/task214_radius_live_phase0.py (CPU 1min 重跑) |
| 结果 JSON | /home/wlia0047/.claude/jobs/04ccf474/tmp/task214_phase0_results.json (5 seed × 3 指标) |
| Verdict | verdicts/task214_radius_live_phase0_result.md (本文档) |
| Description | descriptions/task214_latent_radius_live_phase0.md |

---

## 7. Paper §1.4 增量 finding 8/9/10/11 (整合)

新增到 papers/paper.md §1.4:

8. **钉半径低维双曲 4 stage NO-GO** (Task #211): 4 stage 闭环 C1/C2/C3 R@10=0.0816 (-20% vs baseline). 修复了 forward path `self.rho` vs `self.r_target_norm` bug 后, ‖x‖_E 完美钉住 (0.762/0.875/0.935), λ_κ 从 3571 降到 4.8/8.5/16.0 (数值有效), 但 R@10 仍 NO-GO. 根因: 低维 + 钉半径 = 码字挤薄壳, L0 utilization 23.4%.
9. **Two-stage decision NO-HOPE** (Task #212): 双曲几何在 baseline top-k 内跟欧式 argmin 排序 99% 一致 (L0 98.82%, L1 99.21%, L2 99.56%). 任何"欧式取候选 + 双曲重排" 都不改变最终选择.
10. **Entailment Cones NO-HOPE** (Task #213): 锥分配 4 组合 sweep (固定 arctan / inverse_radius / inverse_radius_third / fixed_quarter_pi) 全部 FAIL. 锥 opening 在 baseline 码字 norm 全 ≈ 1.0 时退化成"几乎全覆盖 + 决策坍缩".
11. **Latent Radius Live NO-HOPE** (Task #214): radius head 在 baseline encoder 输出 z 上几乎无信号 (R1 std 0.028 << 0.15, R2/R3 跟 popularity/‖z‖ 几乎零相关). baseline z 已经是"几何无关 representation".

**总结 (paper §1.4 结尾)**: 7 个方向 (Task #199/200/208/209/211/212/213/214) 全部 NO-GO, 共同根因是 **HG-Rec baseline 几何坍缩** (码字 norm ≈ 1.0, λ_κ ≈ 100). 这是 paper 的核心 mechanism finding — **HG-Rec paper 报告的提升不是几何带来的, 是包装 (Sinkhorn + 4-digit dedup + T5 容量) 带来的**.

---

(本文档覆盖 verdicts/task214_* 之前的临时记录; 完整结论已固化.)
