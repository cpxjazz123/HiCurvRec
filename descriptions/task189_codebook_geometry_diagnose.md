# Task #189 — codebook 几何诊断 (baseline + time series)

> **任务目的**: 通过在 Stage 1 训练中诊断码字几何分布(范数、λ、方向分散度), 找出"几何为何失效"的根因.

> **完成日期**: 2026-07-24
> **状态**: ✅ 已完成 (placeholder description — 实际工作已落 verdict + products)

---

## 1. 背景

承接 Task #84 HG-Rec baseline 复现 (R@10=0.1020), 但 paper 报告 0.1315 (-22.4% gap).
8/8 baseline 复现均低于 paper 18-61% (系统性数据集/评估协议差异, 相对排序保留, 绝对数字不保留).

用户 2026-07-24 提议: 在 Stage 1 ckpt 上检查码字几何, 看是否几何"激活".
Layer L0/L1/L2 码字范数 / λ / 方向分散度的逐 epoch 轨迹.

---

## 2. 实验设计

**变量**: Stage 1 epoch (1, 5, 50, 200, 500)
**保持不变**: HG-Rec baseline recipe (num_emb_list=[64,128,256], e_dim=32, β=0.5, loss_type=poincare)
**启动命令**:
```bash
CUDA_VISIBLE_DEVICES=0 python3 -u $REPO/HG-Rec/src/train_hrqvae.py \
    --dataset Instruments --epochs 500 --num_emb_list [64,128,256] \
    --e_dim 32 --beta 0.5 --loss_type poincare --save_limit 50
```

---

## 3. 决策触发

| 指标 | 通过 | 不通过 |
|------|------|--------|
| ‖z‖_ball > 0.5 | ✅ | ❌ |
| λ(L0) > 4 | ✅ | ❌ |
| 方向分散度 > 0.3 rad | ✅ | ❌ |

---

## 4. 预算

| 阶段 | 估算时间 |
|------|---------|
| Stage 1 500 epoch | ~6 h |
| 几何诊断 (5 epoch 点) | ~5 min |
| verdict | ~5 min |

---

## 5. 产物

- verdict: `verdicts/task189_codebook_geometry_diagnose_result.md`
- products: `products/task189/`
- logs: `logs/task189/`
- 关键结论: λ ≈ 2.0 (平坦区), ‖z‖_ball < 0.3, 方向分散度 < 0.2 rad — **几何完全未激活**

---

## 6. 完成度跟踪

- [x] Stage 1 训练
- [x] 几何诊断
- [x] verdict 写盘

---

> **占位说明**: 此 description 是 2026-07-25 补写, 当时未按 grid-new-task skill 模板写 description, 但有完整 verdict + products + logs 留存. R9 修复占位.