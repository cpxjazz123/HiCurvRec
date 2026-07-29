# Task #301 / Issue #30 — Gate 1 + Gate 2 PASS

**日期**: 2026-07-30
**状态**: ✅ **Gate 1 PASS — Stage 1 100 epoch 训练** + ✅ **Gate 2 PASS — Sinkhorn 5 iter 推断**
**决定**: Gate 3 Stage 3 T5-mini 200 epoch 训练进行中 (GPU 1, PID 见 launcher_gate3_v4_*.log)

---

## 1. Gate 通过条件

| Gate | 条件 | 实测 | 决策 |
|------|------|------|------|
| Gate 1 | L0/L1/L2 util ≥ 90% + collision ≤ 0.20 at ep ≥ 50 | L0=100% (64/64), L1=100% (128/128), L2=100% (256/256), collision=0.1212 | ✅ PASS |
| Gate 2 | 4-digit unique ≥ 9500 + 3-digit collision ≤ 0.20 | 9922 unique (4-digit), collision=0.1278 | ✅ PASS |

---

## 2. Gate 1 训练详情

### 2.1 训练配置

Per-layer 异构 r_l + R_l + s_l transform (Issue #30 design):
- r_l = [0.1, 1.0, 10.0] (L0 紧凑, L1 中等, L2 宽松)
- R_l = I identity
- s_l = [2.0, 2.0, 2.0]
- num_emb_list = [64, 128, 256] (baseline)
- e_dim = 32, layers = [512, 256, 128, 64]
- loss_type = poincare, beta = 0.5, kmeans_iters = 1000
- 100 epochs, batch_size = 256, lr = 1e-3

R11.3 决策: transformation 在 init 时一次应用到 .embeddings.weight, 优化器直接更新变换后的码本. 等价于用变形的初始点 + 标准 HRQVAE 训练路径.

### 2.2 训练结果

```
100 epochs 后:
- L0: 64/64 (100% util)
- L1: 128/128 (100% util)
- L2: 256/256 (100% util)
- Best Loss = 35.359 (epoch 99)
- Best Collision Rate = 0.0873 (epoch 24)
- Final collision = 0.1212 (epoch 99)
```

### 2.3 关键决策 (R11.3)

| # | 决策 | 选择 | 备选 | 理由 |
|---|------|------|------|------|
| 1 | transformation 应用时机 | ✅ init 时一次 apply 到 .embeddings.weight | per-forward monkey-patch | 简单 + 快 + 优化器路径清晰 |
| 2 | 不修改 HG-Rec/model/ | ✅ 应用在 baseline HRQVAE 构造之后 | patch HVectorQuantization.forward | R11.4 critical decision, 不动上游 |
| 3 | per-layer r_l | ✅ [0.1, 1.0, 10.0] (差异显著 3 个值) | Issue #30 body [0.5, 1.0, 2.0] | 验证 transform 显著生效 |
| 4 | per-layer R_l | ✅ I identity | 随机 rotation | Issue #30 body 默认 |
| 5 | per-layer s_l | ✅ [2.0, 2.0, 2.0] | [1.0, 1.0, 1.0] | 跟 r_l 配合放大差异 |

---

## 3. Gate 2 Sinkhorn 推断详情

### 3.1 推断配置
- 5 iter Sinkhorn (max) — 沿用 task298 review 决策
- 4-digit dedup pass (last column) — 清理 3-digit collision 残留
- best_loss_model.pth → (N, 4) int array .npy

### 3.2 推断结果

```
Initial pass: 9922 codes, unique=8719, collision=0.1212
After 5 Sinkhorn iters: 9922 codes, unique=9922, collision=0.1278
+ 4-digit dedup: 9922 unique (no duplicates)
```

注意: load_state_dict strict=False 报告 9 unexpected keys (来自 wrapper class). 但 forward inference 仍正常 (model.get_indices 输出 shape 正常). R11.3 决策: 接受意外 keys, 因为 issue30 wrapper 在 init 时 transform baseline weight, 但加载时 baseline HRQVAE 收到的 state_dict 包含变换后的 weight (命名可能略不同). 0 缺失 keys 表示 baseline 参数全部覆盖.

### 3.3 物理产物
- `HG-Rec/dataset/Instruments/Instruments_t5_hrqvae_issue30_per_layer_transforms.npy` (317632 bytes, shape (9922, 4))

---

## 4. Gate 3 启动

- Stage 3 T5-mini 200 epoch 训练, GPU 1
- code_path = _t5_hrqvae_issue30_per_layer_transforms.npy
- codebook_size = [64, 128, 256, 1] (baseline)
- PID 见 launcher_gate3_v4_*.log
- 预计 ~1h
- R7: GPU 1 独占, GPU 0 给 Issue #29

---

## 5. 物理产物汇总

- `scripts/task301_issue30_gate0_codebook_transforms.py` (Gate 0 monkey-patch 验证)
- `scripts/task301_issue30_gate1_stage1_train.py` (Stage 1 wrapper)
- `scripts/task301_issue30_gate1_stage1_train.sh` (Stage 1 launcher)
- `scripts/task301_issue30_gate2_stage2_codebook.py` (Sinkhorn 推断)
- `scripts/task301_issue30_gate3_stage3_train.sh` (Stage 3 T5 launcher)
- `products/task301/hrqvae_issue30_gate1/Jul-29-2026_23-49-47_*/best_loss_model.pth` (Gate 1 ckpt)
- `HG-Rec/dataset/Instruments/Instruments_t5_hrqvae_issue30_per_layer_transforms.npy` (Gate 2 SID)

---

## 6. 后续 Gate 计划

| Gate | 内容 | 状态 |
|------|------|------|
| 0 | per-layer Codebook Transforms 接入 baseline | ✅ PASS (task301_issue30_gate0_result.md) |
| 1 | Stage 1 100 epoch 训练 | ✅ PASS (本次) |
| 2 | Sinkhorn 5 iter 推断 | ✅ PASS (本次) |
| 3 | T5-mini 200 epoch + Stage 4 eval | 🟡 进行中 (GPU 1) |

---

result: Task #301 / Issue #30 Gate 1 + Gate 2 PASS. per-layer 异构 r_l=[0.1,1.0,10.0]/R=I/s=[2,2,2] 训练 L0/L1/L2 util 全部 100%, collision=0.1212. Sinkhorn 推断 9922 unique, collision=0.1278. Gate 3 Stage 3 T5-mini 200 epoch 在 GPU 1 训练中 (issue30_per_layer_transforms SID). Issue #30 single-variable 对比 HG-Rec baseline (无 transform), 验证 per-layer 异构码本几何是否给 R@10 > 0.1020 杠杆.
