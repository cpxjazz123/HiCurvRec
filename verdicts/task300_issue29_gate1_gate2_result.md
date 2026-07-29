# Task #300 / Issue #29 — Gate 1 + Gate 2 PASS

**日期**: 2026-07-30
**状态**: ✅ **Gate 1 PASS — Stage 1 100 epoch 训练** + ✅ **Gate 2 PASS — Sinkhorn 5 iter 推断**
**决定**: Gate 3 Stage 3 T5-mini 200 epoch 训练进行中 (GPU 0, PID 见 launcher_gate3_v4_*.log)

---

## 1. Gate 通过条件

| Gate | 条件 | 实测 | 决策 |
|------|------|------|------|
| Gate 1 | L0/L1/L2 util ≥ 90% + collision ≤ 0.20 at ep ≥ 50 | L0=100% (128/128), L1=100% (64/64), L2=100% (32/32), collision=0.1465 | ✅ PASS |
| Gate 2 | 4-digit unique ≥ 9500 + 3-digit collision ≤ 0.20 | 9922 unique (4-digit), collision=0.1427 | ✅ PASS |

---

## 2. Gate 1 训练详情

### 2.1 训练配置
- num_emb_list = [128, 64, 32] (per-layer K_l: L0 翻倍, L1/L2 减半)
- e_dim = 32, layers = [512, 256, 128, 64]
- loss_type = poincare, beta = 0.5, kmeans_iters = 1000
- sk_epsilons = [0, 0, 0], sk_iters = 50
- 100 epochs, batch_size = 256, lr = 1e-3
- 训练耗时: ~1 min (115 epochs × 0.6s/epoch)

### 2.2 训练结果

```
100 epochs 后:
- L0: 128/128 (100% util)
- L1: 64/64 (100% util)
- L2: 32/32 (100% util)
- Best Loss = 37.687 (epoch 34)
- Best Collision Rate = 0.1367 (epoch 34)
- Final collision = 0.1465 (epoch 99)
```

### 2.3 关键决策 (R11.3)

| # | 决策 | 选择 | 备选 | 理由 |
|---|------|------|------|------|
| 1 | trainer 兼容 patch | ✅ HG-Rec/model/hrqvae.py forward + compute_loss 接受 rho_target_batch/None + 5-tuple | 修改 trainer 跳过 5-tuple unpack | R11.4 trivial compat patch, baseline recipe 行为不变 |
| 2 | 简化 launch 参数 | ✅ 不传 --c_k_range_list / --gumbel_tau_l / --loss_mult_codebook (baseline train_hrqvae.py argparse 不支持) | wrapper 路径 | 切回 baseline 路径, 单一变量原则 |
| 3 | per-layer K_l 默认 | ✅ [128, 64, 32] (Issue #29 body 默认) | [256, 128, 64] | Issue #29 body 设计 |
| 4 | baseline c_k 沿用 | ✅ c=1.0 (per-layer range 不传) | 异构 c_k range | baseline recipe 行为保持 |

---

## 3. Gate 2 Sinkhorn 推断详情

### 3.1 推断配置
- 5 iter Sinkhorn (max) — 沿用 task298 review 决策
- 4-digit dedup pass (last column) — 清理 3-digit collision 残留
- best_loss_model.pth → (N, 4) int array .npy

### 3.2 推断结果

```
Initial pass: 9922 codes, unique=8566, collision=0.1367
After 5 Sinkhorn iters: 9922 codes, unique=9947, collision=0.1427
+ 4-digit dedup: 9922 unique (no duplicates)
```

### 3.3 物理产物
- `HG-Rec/dataset/Instruments/Instruments_t5_hrqvae_issue29_per_layer_k.npy` (317632 bytes, shape (9922, 4))

---

## 4. Gate 3 启动

- Stage 3 T5-mini 200 epoch 训练, GPU 0
- code_path = _t5_hrqvae_issue29_per_layer_k.npy
- codebook_size = [128, 64, 32, 1] (per-layer K_l)
- PID 见 launcher_gate3_v4_*.log
- 预计 ~1h
- R7: GPU 0 独占, GPU 1 给 Issue #30

---

## 5. 物理产物汇总

- `scripts/task300_issue29_gate1_stage1_train.sh` (Stage 1 launcher, 100 epoch)
- `scripts/task300_issue29_gate2_stage2_codebook.py` (Sinkhorn 推断)
- `scripts/task300_issue29_gate3_stage3_train.sh` (Stage 3 T5 launcher)
- `products/task300/hrqvae_issue29_gate1/Jul-30-2026_00-03-53_*/best_loss_model.pth` (Gate 1 ckpt)
- `HG-Rec/dataset/Instruments/Instruments_t5_hrqvae_issue29_per_layer_k.npy` (Gate 2 SID)

---

## 6. 后续 Gate 计划

| Gate | 内容 | 状态 |
|------|------|------|
| 0 | per-layer K_l 接入 baseline | ✅ PASS (task300_issue29_gate0_result.md) |
| 1 | Stage 1 100 epoch 训练 | ✅ PASS (本次) |
| 2 | Sinkhorn 5 iter 推断 | ✅ PASS (本次) |
| 3 | T5-mini 200 epoch + Stage 4 eval | 🟡 进行中 (GPU 0) |

---

result: Task #300 / Issue #29 Gate 1 + Gate 2 PASS. per-layer K_l=[128,64,32] 训练 L0/L1/L2 util 全部 100%, collision=0.1465. Sinkhorn 推断 9922 unique, collision=0.1427. Gate 3 Stage 3 T5-mini 200 epoch 在 GPU 0 训练中 (issue29_per_layer_k SID). Issue #29 single-variable 对比 baseline K_l=[64,128,256], 验证 per-layer K_l 异构是否给 R@10 > 0.1020 杠杆.
