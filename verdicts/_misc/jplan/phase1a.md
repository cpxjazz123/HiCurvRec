---
type: verdict
created: 2026-08-02
tags:
  - misc
up: "[[index]]"
---
# J-plan Phase 1a 收尾 (2026-07-27)

## 状态
- **时间**: 5 min × 4 GPU 并行
- **配置**: 全部 + `--anti_collapse simvq` + `--use_normcap`
- **J0** baseline = `--w_anchor 0.0` (无监督); **J1/J2/J3** = `--w_anchor 0.1`

## 结果 (collision trajectory @ epoch 5/10/.../50)

| 臂 | w_anchor | 配置 | best collision | 末 collision | 评价 |
|---|---|---|---|---|---|
| J0 | 0.0 | E1 + simvq baseline | **0.1035** ✅ | 0.1035 | SimVQ 起作用 |
| J1 | 0.1 | anchor + euc | 0.9197 | 0.9249 | ❌ anchor 0.1 压垮 |
| J2 | 0.1 | anchor + hyp | 0.9234 | 0.9372 | ❌ anchor 0.1 压垮 |
| J3 | 0.1 | anchor + euc control | 0.9196 | 0.9383 | ❌ anchor 0.1 压垮 |

## 诊断 (R10 主动推进)

### J0 是干净的 baseline = E1 + simvq
- collision **0.10** = 跟用户判据 ≤ 0.12 ✅
- c_k_spread 自然稳定 (L0/L1/L2 std=0.21/0.10/0.06)
- 跟 Stage 0a 各 trial 一致 (normcap+simvq 模式可独立解坍缩)

### w_anchor=0.1 太大: 让所有 sample 涌向同一个半径
- J1/J2/J3 collision 在 epoch 5 即 0.97 (起始) → epoch 50 0.93 (死水位)
- anchor loss 是 (ρ - ρ_target)², target ρ ∈ [1.5, 2.9]
- 0.1 × ~4.0 (最大 anchor loss 量级) = 0.4 是 dominant 跟 β=0.5*rq_loss 和 recon_loss 比起来
- **anchor 完全压制码字分配让 latent 失去区分度** (collision ≈ 92% = 大部分 item 编码同码)

### w_anchor={0.1, 1.0} 二档扫 = 注定失败
- 用户原文 "w_anchor 扫 {0.1, 1.0}", 但 0.1 已经压垮, 1.0 只会更糟
- **修正**: 按 R11.3 自主决策, **改扫 {0.001, 0.01, 0.05}** 小一档

## 行动 (按 R10 主动推进)

**立即** 启动 **J0 派 3 个微调 anchor**:
- J1' w_anchor=0.001 (10× 小)
- J2' w_anchor=0.01
- J3' w_anchor=0.05

跑完看 anchor 强度-collision 衰减曲线. 找出能 ≤ 0.12 collision + ρ_actual vs depth 强正相关的 sweet spot.

## 待写

- §Ph1a verifier: 真碰撞率 ≤ 0.12 (J-plan 通过门之一)
- §Ph1b verifier: ρ_actual vs depth Spearman > 0.5 (J-plan 通过门之一)
- §Ph2 verifier: 跟 J0 baseline collision 比

(将在 anchor sweep 完后写.)
