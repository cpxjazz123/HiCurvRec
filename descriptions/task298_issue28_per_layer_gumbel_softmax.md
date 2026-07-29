# Task #298 — Issue #28 per-layer 异构 soft-assign Gumbel-Softmax τ_l

## 背景

承接 task298 verdict (Issue #26 conflict report) + owner feedback 2026-07-29 23:13 「不允许假设 owner 有 decision. AI 必须自行决策做出可以推进目的的决定」. AI 自主决策启动**唯一未尝试且 NORTH STAR §4 豁免**的方向: per-layer 异构 Gumbel-Softmax τ_l + per-layer c_k range (Issue #28).

**核心机制**:
- L0 (K=64): τ_0 = 1.0 (soft 均匀探索 64 个码字, 扩展 L0 capacity 突破 baseline 73.44% / task275 89.1% plateau)
- L1 (K=128): τ_1 = 0.5 (medium-hard 软分配)
- L2 (K=256): τ_2 = 0.1 (near-hard 逼近 baseline argmax)

每层独立 c_k range (沿用 task242 Arm A: U(1,5) / U(0.5,20) / U(0.5,20)) + 每层独立 Gumbel-Softmax 温度 τ_l + 每层独立 soft-assign 概率密度 = **per-layer 三维异构** (NORTH STAR §4 豁免).

**绕开 task297 K1 warm-start bug**: 不预训练 frozen codebook, 直接端到端 Stage 1 训练 100 epoch.

## 4-Gate 协议 (Issue #28 body)

| Gate | 内容 | 通过条件 | 失败动作 |
|------|------|---------|---------|
| **0** | 实现 `train_hrqvae_gumbel.py` (新增 per-layer τ_l + per-layer c_k range + Gumbel-Softmax soft-assign) | code commit + τ→0 收敛 baseline argmin | 不进 Gate 1 |
| **1** | Stage 1 端到端 100 epoch 训练 (GPU) | (a) L0≥90% (b) L1≥90% (c) L2≥90% (d) collision≤0.20 at ep≥50 | 不进 Gate 2, 关 issue |
| **2** | Sinkhorn 5 iter 推断 (cheap) | 4-digit unique ≥ 9500 + per-layer util 偏差 ≤ 5pp | 不进 Gate 3, 关 issue |
| **3** | T5-mini 200 epoch + Stage 4 eval | **R@10 > 0.1020** | 关 issue NO-GO |

## 数据

- Gate 0 脚本: `scripts/task298_issue28_gate0_gumbel_softmax.py` (新建)
- Gate 1 launcher: `scripts/task298_issue28_gate1_stage1_train.sh` (GPU)
- Gate 2 SID .npy: `scripts/task298_issue28_gate2_sid_codebook.py`
- Gate 3 T5 train + eval: 复用 task297 模板
- Issue: https://github.com/WENYULIANG123/GeneRec/issues/28

## 关键决策点 (R11.3 自主决策)

1. **next task id = 298** (max + 1)
2. **τ_l schedule**: L0=1.0 / L1=0.5 / L2=0.1 (Issue #28 body 明确, 跟 baseline argmax 兼容)
3. **c_k range**: 沿用 task242 Arm A (per-layer 异构 metric)
4. **绕开 task297 K1 warm-start bug**: 不预训练 frozen codebook, 直接端到端 Stage 1
5. **R12 ckpt 强制每个 epoch 保存**: 训练崩溃保住产物
6. **每 Gate 硬停止**: Issue #28 body 明确禁止跨 Gate 取数

## 不消耗 GPU (Gate 0 only)

Gate 0 = 零 GPU (代码实现 + 验证 τ→0 收敛 baseline argmin). Gate 1-3 需 GPU.