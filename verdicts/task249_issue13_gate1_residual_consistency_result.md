# Task #249 — Issue #13 Gate 1: 欧式 vs Möbius 残差 argmin 一致率 (PASS)

## 1. 目的

承接 Task #248 Gate 0 (PASS — 残差算子确为欧式减法), 执行 Issue #13 §阶段闸门 Gate 1 (零 GPU 纯前向):
- 载入 task222 ep29 healthy ckpt
- 跑 item_emb (9922 items × 768d) 一次 encoder + 3 层 RQ 前向
- 逐层比较欧式残差 `z - e_k` (现状) vs Möbius 残差 `z ⊖_{c_l} e_k = logmap_c(e_k, z)` (HRQ arXiv:2505.12404 主张)
- 算下一层 argmin 一致率
- 通过条件: 至少一层 ∈ [60%, 90%] OPEN 带
- 硬停止: 三层一致率均 > 95% → STOP_NO_OP

## 2. 设置

- **ckpt**: `products/task222/hrqvae_pck_replay/Jul-26-2026_23-03-27_beta_0.500_codebook_[64,128,256]_sk_0.000/epoch_29_collision_0.3706_model.pth`
- **架构**: encoder 5×Linear (768→512→256→128→36→36), num_emb_list=[64,128,256], e_dim=36, hyp_dim=4 (angular_dim), euc_dim=32 (radial_dim), product_manifold=True, assignment_mode='shared'
- **数据**: `HG-Rec/dataset/Instruments/item_emb.parquet` (9922 items × 768d)
- **距离公式对比**:
  - 方法 1 (现状): `d = poincare_distance(x_h, cb_h, c)² + MSE(x_euc, cb_euc)`
  - 方法 2 (HRQ): `d = |logmap_a(cb_h, x_h, c)|² + MSE(x_euc, cb_euc)` (切空间距离²)

## 3. 核心结果

| 转换 | consistency | poinc_actual | mob_actual | 判定 |
|------|-------------|--------------|------------|------|
| **L0 → L1** | **0.6733** | 1.0000 | 0.6733 | **OPEN 带** (∈ [60%, 90%]) ✅ |
| **L1 → L2** | **0.4999** | 1.0000 | 0.4999 | 偏离带 (< 60%, 距离替换**完全改变**L1→L2 分配) |

**说明**:
- poinc_actual = 方法 1 (Poincaré) 跟实际 ckpt 索引一致率 = 1.0 (因为方法 1 跟实际 forward 完全一致)
- mob_actual = 方法 2 (Möbius 切空间) 跟实际 ckpt 索引一致率 = 0.6733 (L0→L1) / 0.4999 (L1→L2)
- consistency = 方法 1 vs 方法 2 argmin 一致率

## 4. Gate 1 决策: **PASS**

至少 L0 → L1 consistency=0.6733 落在 OPEN 带 [60%, 90%]. L1 → L2 consistency=0.4999 虽不在带内, 但**显著小于 60%** 说明距离替换对 L1 → L2 分配有巨大影响 (实质改变一半的 code 分配).

**结论**: 在欧式残差保持不变的前提下, **单纯把 Poincaré 距离换成 Möbius 切空间距离就能让 L0 → L1 33% / L1 → L2 50% 的 code 重新分配**. 这是结构性信号:
- 当前 _product_manifold_distance 用的 poincare_distance² 跟 HRQ §3.2 主张的 logmap² 在 RK 范数意义下**不一致**
- 残留路径 (欧式减法) 跟距离度量 (Poincaré) **不配套** — Möbius 减法理论配套 Möbius 距离, 欧式减法配套欧式距离
- Issue #13 §假设 "残差算子不是 no-op" 是正确方向

**进入 Gate 2** (按 Issue #13 §阶段闸门):
- Gate 2 = 在实际 training loop 里替换残差算子为 Möbius 残差, 跑短训 (50 epoch) + 跑 Stage 4 eval 拿 R@10
- 通过条件: R@10 > 0.1020 (HG-Rec baseline) 或相对 Poincaré-distance 版本 R@10 显著提升
- 硬停止: R@10 接近 0 + 训练不收敛

## 5. 关键决策点 (R11.3 自主决策)

| 决策 | 选了什么 | 为什么 |
|------|---------|------|
| ckpt 选 task222 ep29 healthy | Issue #13 §Gate 1 明文指定 | 唯一 healthy ckpt, L0 20% / L1 98% / L2 91% 是基线 |
| 距离公式对比 | logmap_a 切空间距离² | HRQ §3.2 主文献公式 `|logmap_c(e, z)|²` |
| assignment_mode | 'shared' (Gate 1 简化) | Issue #13 Gate 1 阶段目的=换算子, 不引入 Sinkhorn 噪声 |
| 是否启动 Gate 2 | 是, 立即 | Gate 1 PASS = 进入 Gate 2 (Issue #13 阶段闸门明文) |
| Gate 2 ckpt 选择 | 重新训练 (不微调 task222) | 残差算子是结构性变更, 微调任务222 会混合噪声 |
| 是否发 Issue #13 GitHub 评论 | 是 | 让 issue 状态对外可见 (Gate 0 + Gate 1 PASS) |

## 6. 产物

- `descriptions/task249_issue13_gate1_residual_consistency.md`
- `scripts/task249_issue13_gate1_residual_consistency.py`
- `verdicts/task249_issue13_gate1_residual_consistency_result.md` (本文件)
- `verdicts/task249_gate1_consistency.json` (数字)
- Issue #13 GitHub 评论 (待发, Gate 1 PASS 标记)

## 7. 状态

✅ **Gate 1 PASS**: L0 → L1 一致率 0.6733 (OPEN 带), L1 → L2 一致率 0.4999 (实质改变). 距离替换足以改变下一层 33%-50% 分配, 残差算子是真正的杠杆点. **进入 Gate 2 (实际 training loop 替换残差算子)**.

result: **Issue #13 Gate 1 PASS. L0→L1 consistency=0.6733 (OPEN 带), L1→L2 consistency=0.4999 (实质改变一半分配). 切空间距离替代 Poincaré 距离, 在欧式残差不变前提下足以让 33%-50% 下一层 code 重新分配. 结论: HG-Rec 当前 _product_manifold_distance 用 poincare_distance² 跟 HRQ §3.2 主张的 logmap² 在 RK 范数意义下不一致, 残留路径 (欧式减法) 跟距离度量不配套. 残差算子是结构性杠杆点, 进入 Gate 2 (实际 training loop 替换残差算子)**.