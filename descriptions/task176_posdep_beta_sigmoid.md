# Task #176 — κ-Stereographic + β(x) sigmoid MCKG-style 位置依赖

> **任务目的**: 验证 hypothesis"手工设计位置相关的 commitment weight β(x), 沿 MCKG margin 设计思路按 curvature-sign 分流, 给 encoder 局部自由度"是否能改善下游 Recall
> **承接**: Task #175 ORC LOCKED (κ 锁定), Task #174 D 臂 gating 退化为 A 臂, Task #164-#172 共 8 个 κ-Stereo 变体 NO-GO
> **完成日期**: (in progress)
> **状态**: 🟡 待启动

---

## 1. 背景

**用户 hypothesis (2026-07-25 §2 方法 A)**: "手工设计位置相关的 β(x), 比如用 sigmoid 压缩点到原点的距离作为输入. 双曲分支离原点近的地方 β 调小 (给更多自由度让 encoder 能跑出局部困住的区域), 离原点远的地方 β 调大; 球面分支反过来. 这跟 MCKG margin 设计同构."

**跟之前任务区别**:
- 之前所有任务都用 **固定 β** (e.g. β=0.25 default, β=1.0 in #174), 全空间均匀 commitment weight
- **Task #176**: β 变成 **位置依赖** β(x), 跟 ||x||² + κ 符号耦合, 给 encoder 不同区域不同 commitment penalty

**为什么可能有效**:
- κ=0 (Euclidean) 时 β(x) = β_base · sigmoid(0) = 0.5·β_base, 均匀退化为固定
- κ≠0 时不同位置不同 β, 允许局部自由/局部约束
- MCKG paper 用类似 design 让 margin 适应 curvature sign

---

## 2. 实验设计

**变量**: β(x) = β_base · sigmoid(sign(-κ) · ||x||² · scale)
**保持不变**:
- Stage 1 RQ-VAE 架构 (κ-Stereographic, num_hierarchies=3, learnable κ_m)
- M=3, e_dim=32, 11+11+10
- num_emb_list [32, 64, 256, 1]
- Stage 1 训练 (200 epoch, lr 1e-3)
- Stage 2-4 跟 baseline 一致
- Seed=42

**β(x) formula 详解**:
```
λ_κ(x) = 2 / (1 + κ · ||x||²)         # conformal factor
β(x) = β_base · sigmoid(sign(-κ) · ||x||² · scale)
```

**Curvature-sign logic**:
- κ < 0 (双曲): β(x) = β_base · sigmoid(+||x||²·scale)
  - 近原点 (||x||²≈0): sigmoid(0) = 0.5, β ≈ 0.5·β_base (较小, 给 encoder 自由度)
  - 远原点 (||x||² 大): sigmoid(large) ≈ 1, β ≈ β_base (大 commitment)
- κ > 0 (球面): β(x) = β_base · sigmoid(-||x||²·scale)
  - 近原点: sigmoid(0) = 0.5, β ≈ 0.5·β_base
  - 远原点: sigmoid(-large) ≈ 0, β ≈ 0 (无 commitment, 球面 "回到中心")

**Default hyperparams**: β_base=1.0, scale=10.0 (R11.3 决策, 待 Stage 1 跑后看)

**Stage 1 启动命令**:
```bash
python3 -u scripts/task176_posdep_beta_sigmoid_stage1_train.py \
    --beta_mode sigmoid_posdep \
    --beta_base 1.0 --beta_scale 10.0 \
    --M 3 --kappa_max 2.0 --epochs 200 --batch_size 256 --lr 1e-3 \
    --num_emb_list 32 64 256 1 --e_dim 32 --layers 512 256 128 \
    --loss_type poincare --beta 1.0 --quant_loss_weight 1.0 \
    --sk_epsilons 0.0 0.0 0.0 0.0 --sk_iters 50 \
    --kmeans_init --kmeans_iters 1000 \
    --dead_code_reset_every 20 --dead_code_reset_threshold 0.0 \
    --dead_code_replace_ratio 0.1 --seed 42 \
    --output_dir /home/wlia0047/ar57/wenyu/GeneRec/products/task176/posdep_beta_sigmoid/
```

**Stage 2/3/4 沿用 Task #175 ORC LOCKED 模式**, 只换 Stage 2 output file name (`_t5_rqvae_posdep_beta_sigmoid.npy`) + Stage 3 best ckpt path.

---

## 3. 决策触发 (vs baseline 0.1058)

| 指标条件 | 结果指标 | 决策 |
|----------|----------|------|
| test R@10 > 0.1058 | 🟢 GO | sigmoid β(x) 验证成功, 写 verdict 报告. |
| 0.1000 ≤ test R@10 ≤ 0.1058 | 🟡 MARGINAL | 跟 baseline 接近, NO-GO 但有 partial signal. |
| test R@10 < 0.1000 | ⛔ NO-GO | 假设证伪. |

---

## 4. 预算

| 阶段 | 估算时间 |
|------|---------|
| Stage 1 RQ-VAE 200 epoch | ~3 min (单 Phase, GPU 1) |
| Stage 2 codebook inference | ~3 min |
| Stage 3 T5-mini 9.18M | ~50 min |
| Stage 4 test eval | ~1 min |
| **总计** | **~57 min (~1 h)** |

---

## 5. 风险与缓解

**风险 1**: β(x) sigmoid 在 ||x||² → ∞ 时饱和到 β_base, 数值稳定 ✓
**风险 2**: κ=0 时 β=0.5·β_base ≠ β_base, 跟 #164/#169 baseline 偏移 → 比 #164 时需要 normalize. 实际 R11 决策: 不 normalize (偏离是有意 design).
**风险 3**: 之前 8 个 κ-Stereo 变体全 NO-GO, sigmoid β(x) 可能同样 NO-GO → 写 verdict 闭环.

---

## 6. 完成度跟踪

- [ ] Stage 1 launch
- [ ] Stage 1 finish (best_loss_model.pth 落盘)
- [ ] Stage 2 SID codebook inference
- [ ] Stage 3 T5-mini launch
- [ ] Stage 3 finish (HG_Rec_best.pth 落盘)
- [ ] Stage 4 test eval
- [ ] 写 verdict (`verdicts/task176_posdep_beta_sigmoid_result.md`)
- [ ] 更新 loop.md §16 + §15.4