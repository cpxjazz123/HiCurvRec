# Task #183 — 半径设计 Phase 1: 半径正则化 RQ-VAE

> **任务目的**: 修复 HG-Rec 核心缺陷——半径是无人管的副产品，码字被残差被动推向球心，双曲效应为零。通过软正则化 `(‖codebook‖ - ρ_ℓ)²` 将码字推到双曲有效区，同时保留 argmin 的径向区分信号。

> **完成日期**: 2026-07-25 (in progress)
> **状态**: 🟢 Stage 1 RQ-VAE 训练中（epoch 39, coll 96.3%, fill=0.69-0.81）

---

## 1. 背景

**核心诊断** (用户 2026-07-25):
在 HG-Rec 中，没有任何 loss 关心码字离球心多远。半径完全由"残差有多大"被动决定——残差必然越来越小，码字必然挤在球心，双曲效应必然为零。

**Phase 1 v1 失败 (2026-07-25)**:
硬归一化 `z → ρ·z/‖z‖` 在 epoch 24 就发生 99.86% 模式坍缩。根因分析：硬归一化移除了 argmin 的径向区分信号，导致 k-means init 后所有码字角度相近 → argmin 退化 → 全部分配到同一码字 → 梯度平均 → 全部坍缩到同一点。

**Phase 1 v2 修复 (2026-07-25)**:
用软正则化 `(‖codebook‖ - ρ_ℓ)²` 代替硬归一化：
- **保留 argmin 的径向自由度**：码字范数可不同 → argmin 有区分信号
- **提供温和的范数约束**：正则化项把码字逐渐推到目标半径 ρ_ℓ
- **梯度正常流动**：不破坏 STE，不产生数值不稳定性

**epoch 39 验证结果**（Musical_Instruments, batch=1024, lr=1e-3）:
| 层 | target_ρ (tangent) | ‖codebook‖_actual | ball_fill | λ | 目标 λ |
|----|-------------------|-------------------|-----------|---|-------|
| L0 | 1.0 | 0.75 | 0.691 | 4.3 | 4.8 |
| L1 | 1.345 | 0.82 | 0.780 | 5.7 | 8.4 |
| L2 | 1.69 | 0.86 | 0.811 | 7.0 | 15.7 |

——码字范数正在稳步增长到目标值，collision 96.3%（正常，跟 vanilla 基线 epoch 9 的 95.9% 相当）。

**κ 不可辨识问题**:
- λ = 2/(1 - c·‖x‖²) → c 和 ‖x‖ 是同一个自由度 → encoder 几十万参数 vs κ 一个标量 → encoder 一定赢 → κ 梯度被抵消
- **Phase 1 v2 修复状态**: 固定半径后（码字在 fill=0.69-0.81 生长），κ 的良定义性仍然取决于 ‖x‖ 是否被约束到特定目标值。Phase 1 先只做半径控制，Phase 2 再做可学习 κ。

---

## 2. 实验设计

**变量**: RQ-VAE 码字范数正则化 `(‖codebook‖ - ρ_ℓ)²`, ρ_ℓ=[1.0, 1.345, 1.69], weight=0.1

**保持不变**:
- 数据集: Musical_Instruments (9922 items)
- RQ-VAE 架构: num_emb_list=[64,128,256], e_dim=32, M=3
- loss 公式: Phase 0.6 官方 loss（Poincaré distance² + β=1.0 on codebook_loss）
- seed=42, sk_eps=[0,0,0]（Sinkhorn OFF）
- 超参: beta=1.0, batch=1024, lr=1e-3, warmup=20, epoch=1000
- Stage 3: T5-small (6 enc + 4 dec, d_model=128), 200 epoch, early_stop=20
- Stage 4: 同 task84 eval protocol

**启动命令**:
```bash
python3 train_hrqvae.py \
    --data_path ./dataset/Instruments/item_emb.parquet \
    --num_emb_list 64 128 256 \
    --e_dim 32 \
    --layers 512 256 128 64 \
    --beta 1.0 \
    --loss_type poincare \
    --sk_epsilons 0.0 0.0 0.0 \
    --epochs 1000 \
    --batch_size 1024 \
    --lr 1e-3 \
    --learner AdamW \
    --weight_decay 0 \
    --warmup_epochs 20 \
    --lr_scheduler_type linear \
    --num_workers 4 \
    --kmeans_init True \
    --kmeans_iters 1000 \
    --radii 1.0 1.345 1.69 \
    --rho_reg_weight 0.1 \
    --ckpt_dir /fs04/ar57/wenyu/GeneRec/products/task183/hrqvae_radius_reg \
    --device cuda:0
```

### 决策触发 (vs HG-Rec baseline R@10=0.1020)

| Stage 1 指标 | Stage 3 R@10 | 决策 |
|-------------|--------------|------|
| fill ∈ [0.5, 0.9] + collision ≤ 15% (final) | ≥ 0.1020 (baseline) | ✅ 半径正则化有效，进入 Phase 2 κ_ℓ |
| fill ∈ [0.5, 0.9] | ≥ 0.1315 (paper) | ⭐ 超预期，半径设计全面成功 |
| fill < 0.2 或 collision > 50% (final) | 任意 | 🔴 正则化设计有问题 |

---

## 3. Phase 1 修改文件清单

| 文件 | 修改 |
|------|------|
| `HG-Rec/model/utils.py` | HVectorQuantization.__init__: 加 rho_reg_weight; forward: 移除硬归一化→加正则化 loss; get_codebook: 移除硬归一化 |
| `HG-Rec/model/hrqvae.py` | HRQVAE.__init__: 加 rho_reg_weight 参数 |
| `HG-Rec/train_hrqvae.py` | 加 --radii 和 --rho_reg_weight CLI 参数 |

---

## 4. 预算

| 阶段 | 估算时间 |
|------|---------|
| Phase 1 v2 实现 + 验证 | ~15 min（已完成） |
| Stage 1 RQ-VAE 1000 epoch | ~5 min（batch=1024, 进行中） |
| Stage 2 Codebook inference | ~5 min |
| Stage 3 T5-small 200 epoch | ~1.5 h |
| Stage 4 Eval | ~5 min |
| 总计 | ~2.5 h |

---

## 5. 风险与缓解

**风险 1**: rho_reg_weight=0.1 不足，1000 epoch 内码字到不了 ρ → 增大到 0.5 或 1.0
**风险 2**: 正则化导致 argmin 仍退化（码字范数相似→角向退化）→ 用 product manifold（Phase 1a）
**风险 3**: 码字到 ρ 后 collision 仍高 → 检查 training dynamics，可能需加大 reg weight

---

## 6. 完成度跟踪

- [x] Phase 1 v1 硬归一化 → v2 软正则化 pivot
- [x] py_compile 三文件验证
- [x] 50-epoch 验证：码字范数增长、无坍缩
- [x] Stage 1 1000 epoch 训练启动 (GPU 0, PID 3610003)
- [ ] Stage 1 完成 + 最终碰撞率 ≤ 15%
- [ ] Stage 2 codebook inference
- [ ] Stage 3 T5-small 训练
- [ ] Stage 4 Eval (R@10 vs 0.1020 baseline)
- [ ] ρ, λ 监控指标验证 (目标 fill=0.5-0.9)
- [ ] Verdict + §16 cleanup

---

## 7. 修订记录

| 日期 | 修订内容 |
|------|---------|
| 2026-07-25 | 首次创建（Phase 1 硬归一化） |
| 2026-07-25 | Phase 1 v2 pivot：硬归一化→软正则化 + epoch 39 验证结果 |
