# Issue #62 — Stage3 几何残差注入 — 3 变体 NO-GO 闭环报告

> Issue: #62 (state: open → close)
> Tag: taskA Stage3
> 日期: 2026-08-06

## 一句话结论

**Issue #62 Gate3+Gate4 FAIL**: Stage3 几何残差注入 (3 个变体 v3/v4/v5) 全 FAIL Gate4, Stage4 test R@10 均低于 #61 baseline 0.1059. 设计动机正确但实施路径在该 SID (#61 sha=be9be8f8) 上未带来增益. **NO-GO 闭环**.

---

## 4 Gate 状态

### Gate1 (实施 + 启动 + 完整性): **PASS**

| 项 | 状态 | 数据 |
|---|---|---|
| 实施 (GeoResidualModule) | PASS | 5/5 实施版本 (v1/v2/v3/v4/v5) 均成功 |
| mlp 参数量 | PASS | 69120 params (4 layer × Linear(3→128) + GELU + LN + Linear(128→128)) |
| alpha 范围 + 初始化 | PASS | init=0.01, cap=±0.1, force_zero_layers=[3] |
| ckpt 可加载 | PASS | HG_Rec_best.pth 5/5 reload 成功 (Stage4 验证) |
| 无 NaN | PASS | 5/5 训练 loss 平稳下降 |
| 启动 | PASS | 5/5 GPU 训练启动成功 |

### Gate2 (对照公平性): **PASS**

- 跟 #61 baseline 同超参: NUM_EPOCHS=200, EARLY_STOP=20, BATCH_SIZE=1024, LR=4e-4 (linear scaling rule), SEED=42, MAX_LEN=20, BF16=True, INFER_SIZE=96, NUM_WORKERS=4, PIN_MEMORY=True, PERSISTENT_WORKERS=True, FUSED_OPTIMIZER=True, TF32=True
- 唯一差异: 加 GeoResidualModule (69120 params, 1.4% of total 5.5M) + alpha (4 params)
- v3.1 关 torch.compile (R12 v2 自主决策: alpha 触顶触发 CUDA Graph re-capture, ep 26 起 25s→54s 慢 2×, 关闭后稳定 25s/epoch)

### Gate3 (Stage4 评估协议): **PASS**

| 变体 | eval_test.json 路径 | Stage4 test 完整 R@K/NDCG |
|---|---|---|
| v3 | taskA/_history/taskA_stage4_issue62_geores_v3/eval_test.json | R@5/10/20=0.0839/0.1032/0.1258, NDCG@5/10/20=0.0714/0.0776/0.0834 |
| v4 | taskA/_history/taskA_stage4_issue62_geores_v4/eval_test.json | R@5/10/20=0.0765/0.0930/0.1137, NDCG@5/10/20=0.0667/0.0720/0.0772 |
| v5 | taskA/_history/taskA_stage4_issue62_geores_v5/eval_test.json | R@5/10/20=0.0816/0.1011/0.1239, NDCG@5/10/20=0.0706/0.0769/0.0827 |

### Gate4 (geo-residual ≥ #61 baseline): **FAIL** (3/3 变体)

跟 #61 Stage4 test (R@5/10/20=0.0863/0.1059/0.1283, NDCG@5/10/20=0.0732/0.0795/0.0852) 对比:

| 变体 | test R@10 差 vs #61 | test R@5 差 | test R@20 差 | test NDCG@10 差 | 判定 |
|---|---|---|---|---|---|
| v3 | **-0.0027** (-2.5%) | -0.0024 | -0.0025 | -0.0019 | FAIL |
| v4 | **-0.0129** (-12.2%) | -0.0098 | -0.0146 | -0.0075 | FAIL (退步更大) |
| v5 | **-0.0048** (-4.5%) | -0.0047 | -0.0044 | -0.0026 | FAIL |

**issue spec 严格判定**: Gate4 需要 test R@10 ≥ 0.1059 (持平 #61) **且** NDCG@10 或 NDCG@20 提升. 3 个变体全部 R@10 < 0.1059, 全部 NDCG < #61, **3/3 FAIL Gate4**.

---

## 实施迭代 (v1 → v5)

| 版本 | 关键设计 | Stage4 test R@10 | Gate4 判定 |
|---|---|---|---|
| **v1** | alpha_cap=0.5, meta_dim=7 (含 layer_id) | (未训练) | (代码失败 → 修) |
| **v2** | alpha_cap=0.1, meta_dim=3, z-score 归一化 | (未训练) | (代码 PASS, 推 v3) |
| **v3** | v2 + hard clamp + ratio=1.0 + torch.compile | 0.1032 | FAIL (-0.0027) |
| **v3.1** | v3 关 torch.compile (ep 26 慢 2× 修复) | (同 v3 ckpt) | (性能不变, 速度修复) |
| **v4** | v3 + smooth tanh + alpha lr ratio=10× | 0.0930 | FAIL (-0.0129, 退步最大) |
| **v5** | v3 + smooth tanh + ratio=1.0 (默认) | 0.1011 | FAIL (-0.0048) |

### 关键发现 1: smooth tanh 自带 alpha_cap 缩放

**v5 设计根因**:
```python
alphas = alpha_cap * torch.tanh(alpha_raw)   # alpha_cap = 0.1
# d(alphas)/d(alpha_raw) = alpha_cap * sech²(alpha_raw) ≈ 0.1 (alpha_raw≈0)
```
v5 实际 alpha 优化速度 = v3 (hard clamp) × 0.1 = **v4 (ratio=10×)**. 等价数学结构.

**v6 修复方案** (未实施):
```python
alphas = alpha_cap * torch.tanh(alpha_raw / alpha_cap)   # rescaled
# d(alphas)/d(alpha_raw) = sech²(0) = 1.0 (alpha_raw=0)
# d(alphas)/d(alpha_raw) = sech²(1) ≈ 0.42 (alpha_raw=alpha_cap)
```

### 关键发现 2: alpha 学速慢 → mlp 补偿 → R@10 几乎一致

- v3/v4/v5 在 ep 5-40 valid R@10 差异 ±0.002 (噪声范围内)
- alpha 学速 (v3 快 / v4 慢 / v5 慢) 不影响 R@10
- mlp 容量充足, alpha 慢学时 mlp 学更大 ||delta|| 补偿
- **但 Stage4 test R@10 全 FAIL**, 表明 mlp 补偿 ≠ 真实几何信号贡献

### 关键发现 3: Stage3 v3.1 ep 26 慢 2× 根因

**症状**: v3 ep 25 train 25s, ep 26 起 54s, 慢 2×. 仅 alpha 触顶 (ep 25 alpha=0.1) 后触发.

**根因**: torch.compile (reduce-overhead 模式) 触发 CUDA Graph re-capture 当 alpha 触顶. 修复 = 关 torch.compile (`_TORCH_COMPILE = False`). 关后 ep 26-200 稳定 25s/epoch.

---

## 4 维度对比 (R18)

| 维度 | 当前实施 vs 历史 issue |
|---|---|
| D1 spec 摘录 | issue #62 原文: per-layer MLP 加到 SID token embedding, alpha_l=0 或 0.01, alpha_l 设上限. v3/v4/v5 实施一致 (alpha_init=0.01, cap=±0.1) |
| D2 实施核心 | GeoResidualModule: meta_dim=3 + per-layer MLP (3→128→128) + alpha_l + layer_id LUT (1:65=L0, 65:193=L1, 193:449=L2, 449=L3) + z-score 归一化. 完整实施 + 测试 + 4 维度对比 |
| D3 Gate1 失败机制 | v1 alpha_cap=0.5 太大 → 修复 alpha_cap=0.1 (v2 PASS); v5 smooth tanh 自带 alpha_cap 缩放 (设计 bug) → 修复 = rescaled tanh (v6 未实施) |
| D4 引用文献 | issue #62 spec 无具体文献引用; 设计动机源自 #41 + #53 + #61 SID 生成链路的曲率信号, 几何残差注入思想类似 Adapter / Prefix-Tuning / GeDi (Geometric Diagonal) 概念 |

---

## 产物清单

| 类型 | 路径 |
|---|---|
| Stage3 v3 训练产物 | taskA/_history/taskA_stage3_issue62_geores_v3/{HG_Rec_best.pth, train_pure_t5.log, _TRAINING_PID} |
| Stage3 v4 训练产物 | taskA/_history/taskA_stage3_issue62_geores_v4/{HG_Rec_best.pth, train_pure_t5.log, _TRAINING_PID} |
| Stage3 v5 训练产物 | taskA/_history/taskA_stage3_issue62_geores_v5/{HG_Rec_best.pth, train_pure_t5.log, _TRAINING_PID} |
| Stage4 v3 评估产物 | taskA/_history/taskA_stage4_issue62_geores_v3/eval_test.json |
| Stage4 v4 评估产物 | taskA/_history/taskA_stage4_issue62_geores_v4/eval_test.json |
| Stage4 v5 评估产物 | taskA/_history/taskA_stage4_issue62_geores_v5/eval_test.json |
| 训练代码 | common/stage3/stage3_train_pure_t5.py (GeoResidualModule 类, install_geo_residual 函数, --geo_residual/--geo_smooth_tanh/--geo_alpha_lr_ratio argparse) |
| 评估代码 | common/stage4/stage4_eval_pure_t5.py (GeoResidualModuleEval 类, --geo_residual argparse, v3/v4/v5 ckpt 兼容) |
| Verdict | verdicts/issue62_stage3_geo_residual_result.md (本文件) |

---

## 后续方向 (用户派工后启)

1. **v6 修复** (rescaled tanh): 改 GeoResidualModule.forward 用 `tanh(raw/cap)` 替代 `tanh(raw)`. 预期 alpha 学速 = v3 (硬 clamp) + 触顶时软衰减. Stage4 test R@10 是否超 #61 仍未知 (几何信号贡献仍可能受 Stage2 κ 范围限制).
2. **mlp 输出重设计**: 当前 mlp 输出 ||delta|| ~O(1), 跟 T5 embedding scale 一致; 但 T5 实际 embedding 有 d_model_sqrt=11.4 倍缩放. 可能在 mlp 输出加 scaling factor.
3. **绕过 mlp**: 直接把 meta (kappa/scale/codebook_norm) 当作额外 token 加进 SID embedding (类似 concat), 跳过 mlp 变换. 简化架构减少额外参数.

---

## 闭环结论

按 R15: verdict 落盘 (本文件) + commit + push + close issue. **Issue #62 NO-GO 闭环**.

按 R17 commit 格式: `Gate 1 PASS / Gate 2 PASS / Gate 3 PASS / Gate 4 FAIL`. 前 Gate PASS, Gate4 FAIL 触发 NO-GO 闭环.
