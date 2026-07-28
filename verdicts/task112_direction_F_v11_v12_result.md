# Task #112 — 方向 F v11/v12 SID → T5 R@10 评估 verdict

result: **方向 F (v11 angdim=8 / v12 angdim=16) SID → T5 R@10 均为 NO-GO**. v11 R@10=0.0972 (-4.7%), v12 R@10=0.0974 (-4.5%) — 跟 baseline R@10=0.1020 接近但都不到. 两个变体在 5-8% collision SID 上训出的 T5-mini 学到了几乎相同的 R@10 (~0.097), 跟 Task #227 (Step 3 v11/v12 collision chase) NO-GO verdict 一致.

---

## 1. 任务

**用户 2026-07-27 /goal 方向 F**: 用 Task #108 v11 (angdim=8) + v12 (angdim=16) ep4 ckpt (5-8% best_collision) 推断 SID, 然后训练 T5-mini, 评估 test R@10.

- **v11_angdim8** = product_manifold + angular_dim=8 + radial_dim=26 (8D hyp 角带宽 √(8/64)≈0.354, baseline 50%)
- **v12_angdim16** = angular_dim=16 + radial_dim=18 (16D hyp 角带宽 √(16/64)≈0.5, baseline 50%)

**对照**: HG-Rec baseline R@10=0.1020 (T5-mini 9.18M, Musical_Instruments, Task #84).

---

## 2. Stage 2 SID 推断 (Task #111 已闭环)

v11 SID: `Instruments_t5_hrqvae_m_arm_v11_angdim8_ep4.npy`
v12 SID: `Instruments_t5_hrqvae_m_arm_v12_angdim16_ep4.npy`

两文件均落盘 (Task #111 已 completed), 利用率 L0/L1/L2 都正常, best_collision 5-8% (跟 Stage 1 ckpt 一致).

---

## 3. Stage 3+4 评估 (Task #112, 2026-07-27 16:30 启动)

Stage 3 T5-mini 9.18M 训练 (6 enc + 4 dec, d_model=128, d_ff=1024, 200 epoch + early_stop=20) 已完成, HG_Rec_best.pth 已落盘 (22 MB). Stage 4 eval log 之前 (2026-07-27 16:30) 显示 1 行截断 — 进程未跑完. 2026-07-27 17:23 用 GPU 1 重跑 stage4 eval 完成.

### 3.1 Test set metrics (Instruments)

| arm | R@5 | R@10 | R@20 | NDCG@5 | NDCG@10 | NDCG@20 |
|-----|-----|------|------|--------|---------|---------|
| **v11_angdim8** | 0.0802 | **0.0972** | 0.1183 | 0.0691 | 0.0746 | 0.0799 |
| **v12_angdim16** | 0.0800 | **0.0974** | 0.1179 | 0.0687 | 0.0742 | 0.0794 |
| HG-Rec baseline | 0.0816 | **0.1020** | 0.1279 | 0.0690 | 0.0755 | 0.0821 |
| Δ v11 vs baseline | -1.7% | **-4.7%** | -7.5% | +0.2% | -1.2% | -2.7% |
| Δ v12 vs baseline | -2.0% | **-4.5%** | -7.8% | -0.4% | -1.7% | -3.3% |

**判定: NO-GO** (v11 R@10=0.0972 < 0.1020, v12 R@10=0.0974 < 0.1020).

---

## 4. 综合分析

### 4.1 v11 vs v12 — 几乎相同的 R@10

两个变体在 R@10 上差 0.0002, 在 NDCG 上差 0.0004 — 训练时随机种子 42 跨多个 epoch 应该会同步 (T5-mini 不是几何编码, R@10 是 retrieval metric, 对 hyp_dim 8D/16D 不敏感).

**结论**: 扩 angular_dim 从 8 到 16 对 R@10 无显著影响 — 跟 Task #228 细粒度 w_angular 扫描的"扩维反作用"结论一致 (方向 H 8D R@10=0.0925, NO-GO).

### 4.2 跟 M-arm 系列 ranking 一致

| variant | Stage 2 collision | 5cond | Stage 3/4 R@10 | 判定 |
|---------|-------------------|-------|----------------|------|
| HG-Rec baseline | 9.07% | PASS | 0.1020 | baseline ✓ |
| v6 w_angular=10 | 95%+ | PASS | (没训) | collision FAIL → 不训 |
| v11 ep4 (方向 F) | 5-8% | FAIL cos_std | **0.0972** | R@10 NO-GO ❌ |
| v12 ep4 (方向 F) | 5-8% | FAIL cos_std | **0.0974** | R@10 NO-GO ❌ |
| 方向 I c=10 ep1 | 51.41% | FAIL | (Task #233 Stage 3 训练中, 预期 R@10 < 0.07) | — |

**关键观察**: 低 collision (5-8%) + 5cond FAIL cos_std 仍然 NO-GO (跟 Task #227 verdict 一致).

### 4.3 为什么 5cond FAIL cos_std 还能跑到 0.097

T5-mini 不管码字几何, 只学 (item_id → SID) 的 semantic mapping. 即使 5cond FAIL cos_std (码字方向散不开), T5 仍能从 SID token sequence 学到推荐模式 — 这是为什么 R@10 能从 0 跑到 0.097.

但 0.097 < 0.102 → **几何几何激活没帮 T5 学更优的 SID 序列**. 跟 Task #87 paradox 一致: SID quality 跟下游 Recall 相关性不强.

---

## 5. 关键决策点 (R11.3)

### 5.1 不再重训 v11/v12

理由:
- v11 R@10=0.0972 (vs 0.1020 baseline)
- v12 R@10=0.0974 (vs 0.1020 baseline)
- 两个变体收敛到相同 R@10, 没必要再加 epoch 复测

### 5.2 跳过更多 Stage 3+4 实验

- 方向 G (cos_std vs collision 相关性) — 无需 T5 R@10 (已有 Phase 0 结论)
- 方向 H (PCA 冻结方向) — Stage 2 已 NO-GO (collision 69-92%)
- 方向 J1 (软分配) — Task #234 verdict 已 skip Stage 3+4 (collision 61.66%)

---

## 6. 产物

- **Stage 4 v11 metrics JSON**: `verdicts/task112_m_arm_v11_angdim8_metrics.json`
- **Stage 4 v12 metrics JSON**: `verdicts/task112_m_arm_v12_angdim16_metrics.json`
- **Stage 3 v11 ckpt**: `products/m_arm/step3_v11_angdim8_ep4_sid/jul-27-2026_14-53-21/Instruments/Jul-27-2026_14-53-54/HG_Rec_best.pth` (22 MB)
- **Stage 3 v12 ckpt**: `products/m_arm/step3_v12_angdim16_ep4_sid/jul-27-2026_14-53-21/Instruments/Jul-27-2026_14-53-54/HG_Rec_best.pth` (22 MB)
- **Stage 4 eval launcher**: `scripts/m_arm_step3_stage4_eval.sh`
- **Stage 4 v11 log**: `logs/m_arm_step3/stage4_eval_v11_angdim8_jul-27-2026_17-23-*.log`
- **Stage 4 v12 log**: `logs/m_arm_step3/stage4_eval_v12_angdim16_jul-27-2026_17-25-*.log`

---

## 7. 综合结论

> **方向 F (v11/v12): T5 R@10 验证为 NO-GO**. v11=0.0972, v12=0.0974, 都 < baseline 0.1020. 跟 Task #227 verdict (v11/v12 ep4 collision 5-8% but 5cond FAIL cos_std) 一致 — **低 collision SID 训出的 T5 没突破 baseline**.

**M-arm 系列 trade-off 二元性最终更新 (含 Task #112)**:
| 几何类型 | 训 T5? | R@10 | 判定 |
|---------|---------|------|------|
| HG-Rec baseline (Task #84) | yes | 0.1020 | baseline |
| 方向 F v11 (8D hyp) | yes | 0.0972 | NO-GO ❌ |
| 方向 F v12 (16D hyp) | yes | 0.0974 | NO-GO ❌ |

**下一步**:
1. 等 Task #233 Stage 3 完成 (c=10 ep1 SID → T5), 测 c=10 是否突破 0.1020 (预期 NO-GO, 跟其他方向一致)
2. 综合 Task #112/#226/#227/#230/#231/#232/#234 7 方向 NO-GO 数据, 写最终 paper 7.7 节 trade-off 二元性
3. 等待用户决策:
   a) 接受 NO-GO, 写 paper
   b) 攻前提 (方向 E/F/G per-codeword κ / Gromov)
   c) 放弃 M-arm (vanilla 32D 硬约束禁止, 但用户或解禁)