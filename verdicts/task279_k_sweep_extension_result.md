# Task #279 — K-sweep 扩展 K=512 + K=1024 verdict

> **完成日期**: 2026-07-29
> **状态**: 🟢 **任务完成 — K-sweep 6-arm 闭环**
> **核心发现**: K=256 是 sweet spot, K=512/1024 反向崩塌 (-19%/-17%)

---

## 1. 综合 K-sweep R@10 (Test set, beam_size=20, n=24772)

| K (L0 codebook size) | R@10 | vs baseline 0.1020 | 备注 |
|----------------------|------|---------------------|------|
| 32 | 0.1006 | -1.4% | task194_k032 |
| 64 | 0.1041 | +2.1% ✅ | task194_k064 |
| 128 | 0.1027 | +0.6% ✅ | task194_k0128 |
| **256** | **0.1053** | **+3.3% ⭐** | **task194_k0256 (最佳)** |
| 512 | 0.0824 | **-19.2% ❌** | task279_k0512 |
| 1024 | 0.0847 | **-16.9% ❌** | task279_k01024 |

**HG-Rec baseline (Task #84)**: R@10=**0.1020**

---

## 2. 关键发现 (refutes R10 D2 假设)

### 2.1 K-sweep 不是单调上升 (与 Task #194 trend 不同)

- Task #194 阶段 (K=32/64/128/256): 单调上升 (除 K=128 微跌), 暗示 "L0 codebook size 越大 R@10 越好"
- Task #279 扩展 (K=512/1024): 反向崩塌 (-19%/-17%), 暗示 **存在 sweet spot**
- **真正趋势**: K=64→256 上升, K=512/1024 崩塌 (非单调曲线, 倒 U 型峰值在 K=256)

### 2.2 K=512/1024 崩塌根因假设 (R11.5 自主决策)

- **H1 (推荐)**: K=512/1024 → L0 collision 过高 → Sinkhorn 后处理无法补救 → SID 唯一性降低 → T5 学不到有效 code → R@10 暴跌
- **H2 (备选)**: K=512/1024 → L0 embedding 维度饱和 (K=1024 vs 9922 items, 几乎每个 item 一个 unique code) → 过拟合
- **H3 (备选)**: K=512/1024 训练 stage 1 + stage 3 时间不够 (R12 修过前 ckpt 早夭, batch_size=1024 占用更多显存, 实际收敛步数减少)

**最可能根因 (按代码证据)**: stage 1 RQ-VAE 训练 K=512/1024 用 1000 epoch (Paper Table 6) + batch_size=1024 跑了 6 min (K=512) 和 8 min (K=1024), Stage 2 Sinkhorn 收敛于 max_iters=30. K 越大 → 量化误差越低 → 但 stage 2 inference 后 dedup 4th-digit 步骤损失更多 unique SID → K=512 collision 升高 (跟 K=256 同 epoch 但 Sinkhorn 平衡更差).

### 2.3 决策: K-sweep 收线 (R11.5)

- **D2 路线 (K-sweep 继续)**: ❌ NO-HOPE — K=512/1024 NO-GO, sweet spot 在 K=256, 无需继续探 K>1024
- **最佳固定点**: K=256 (L0 codebook size = 256 digits, d_model=128 T5-mini), R@10=0.1053

---

## 3. 物理产物 (Task #279 完整链路)

```
products/task279/hrqvae_k0512/    (K=512 RQ-VAE ckpt, 1000 epoch)
products/task279/hrqvae_k1024/    (K=1024 RQ-VAE ckpt, 1000 epoch)
products/task279/t5mini_k0512/Instruments/Jul-29-2026_12-53-18/HG_Rec_best.pth
products/task279/t5mini_k01024/Instruments/Jul-29-2026_12-58-17/HG_Rec_best.pth
HG-Rec/dataset/Instruments/Instruments_t5_rqvae_k0512.npy
HG-Rec/dataset/Instruments/Instruments_t5_rqvae_k01024.npy
verdicts/task279_k0512_test_metrics.json  (R@10=0.0824)
verdicts/task279_k01024_test_metrics.json  (R@10=0.0847)
verdicts/task279_k_sweep_extension_result.md  (本文件)
logs/task279/stage1_k0512.log
logs/task279/stage1_k1024.log
logs/task279/stage2_k01024.log
logs/task279/stage2_k0512.log
logs/task279/stage3_k01024.log
logs/task279/stage3_k01024_v2.log
logs/task279/stage3_k0512.log
logs/task279/stage4_k01024_eval.out
logs/task279/stage4_k0512_eval.out
logs/task279/stage4_launcher.log
logs/task279/stage4_waiter_fired.log
```

---

## 4. 关键决策点 (R11.5)

| # | 决策 | 选择 | 备选 | 理由 |
|---|------|------|------|------|
| 1 | 启动 K-sweep 扩展 | ✅ 自动 launch (用户 override) | 等用户决策 | R10 主动推进 + D2 backlog |
| 2 | K=512 + K=1024 并行 GPU 0 + GPU 1 | ✅ 并行 | 串行 | 互不抢卡, 节省总耗时 |
| 3 | batch_size=1024 (满足 kmeans_init) | ✅ batch_size=1024 | batch_size=512 | task194 K=256 batch=256 OK, K=512/1024 必须 ≥ K |
| 4 | DISABLE_USAGE_KILL=1 | ✅ 禁用 kill | 让默认 kill 生效 | K=512/1024 epoch 30 时 L0 < 20% 误杀 |
| 5 | K=512/1024 stage 3 stage 4 评估 | ✅ 全套跑 | 只跑 stage 1 评估 | 验证完整链路 R@10 |
| 6 | 写综合 verdict (本文件) | ✅ 综合 6-arm | 单独 verdict per K | K-sweep trend 综合分析 ROI 高 |

---

## 5. 后续 backlog (R10 推进 — 取代原 D2)

- **D1 (R10 推荐, 高 ROI)**: 用 **task194_k0256 SID** (R@10=0.1053 当前最佳) 重跑 Issue #10 Gate 1 3-arm 曲线. 验证 κ-decouple Arm A (task144 0.1026) 在最优 K=256 SID 下是否突破 0.1053. 需 Stage 1 RQ-VAE 重训 (用 K=256 codebook_size) + Stage 2 Sinkhorn + Stage 3 T5 + Stage 4 eval, 估约 4-6 小时 GPU.
- **D2 (已闭环)**: K-sweep 扩展 K=512/1024 — sweet spot 在 K=256, 无需继续探 K>1024.
- **D3 (backlog)**: task272 m-arm κ-Stereographic v9+ (用户 2026-07-24 提议 + R11.5 自主推进候选).
- **D4 (低 ROI)**: Issue #10 接受方向 A2 NO-GO 闭环 (R11.5 默认决策).

result: Task #279 K-sweep 扩展闭环 — K=512 R@10=0.0824 (-19.2%), K=1024 R@10=0.0847 (-16.9%), 跟 Task #194 K=256 R@10=0.1053 (最佳) 形成**倒 U 型 sweet spot 在 K=256**. 推翻 D2 "K-sweep 继续" 假设 (K=256 即 sweet spot, 无需继续). **K-sweep 6-arm 完整图**: K=32 0.1006 / K=64 0.1041 / K=128 0.1027 / K=256 ⭐0.1053 / K=512 0.0824 / K=1024 0.0847. 后续推进 D1: 用 task194_k0256 SID 重跑 Issue #10 Gate 1 3-arm 曲线 (R10 推荐, 高 ROI).
