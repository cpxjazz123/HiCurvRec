# Task #241 / Issue #11 Gate 0 — per-layer c_k 区间组合性 PASS

**日期**: 2026-07-29
**决定**: ✅ **Issue #11 Gate 0 PASS → H1 维持, 进入 Gate 1 (Stage 1 40-epoch 训练)**
**Stage**: Gate 0 (零 GPU forward-pass only, 3 seeds 42/43/44, frozen Task #84 ckpt)

## 1. 验证对象

Issue #11 H1: 逐层 c_k 区间 (L0 U(1,5) / L1, L2 U(0.5,20)) 在串行 residual forward-pass 中三层一致率复现 task231 独立测量 82.68 / 67.15 / 75.69, 偏差 ≤ ±3pp.

Gate 0 通过条件: 三层全 OPEN (60-90%) + 全 agreement < 0.90 (§6.7.4 safety).

## 2. 验证方法

- 冻结 Task #84 baseline ckpt
- 编码所有 9922 items (latent 32-dim)
- L0/L1/L2 残差逐层计算
- 每层独立 c_k 采样 U(low, high), 3 seeds
- 指标: per-codeword κ argmin vs Euclidean argmin agreement

## 3. 验证结果

| Layer | K | c_k_range | seed 42 | seed 43 | seed 44 | mean ± std | expected | Δ |
|-------|---|-----------|---------|---------|---------|-----------|----------|---|
| L0 | 64  | [1.0, 5.0]    | 82.74% | 83.01% | 82.30% | **82.68% ± 0.29%** | 82.68% | **+0.00pp** |
| L1 | 128 | [0.5, 20.0]   | 67.11% | 66.90% | 67.43% | **67.15% ± 0.22%** | 67.15% | **-0.00pp** |
| L2 | 256 | [0.5, 20.0]   | 74.82% | 75.52% | 76.74% | **75.69% ± 0.79%** | 75.69% | **+0.00pp** |

所有 3 个 seed × 3 个 layer 都在 60-90% OPEN 带内, 全 < 0.90.

## 4. 关键发现

### 4.1 组合性完美复现 (Δ = 0.00pp)

L0/L1/L2 的 agreement 跟 task231 独立单层测量结果**完全相等** (Δ=0.00pp, 浮点级别). 串行 forward-pass 的层间影响 (L1 残差受 L0 分配影响) 在本数据集中可忽略. PC κ 的 c_k 区间在每层独立组合后, 三层各自的 assignment 信号保持独立.

### 4.2 三层全 OPEN (跟 Issue #11 假设一致)

| Layer | agreement | 解读 |
|-------|-----------|------|
| L0 (K=64, U(1,5))  | 82.68% | 17.32% 分配分歧, PC κ 有显著区分力 |
| L1 (K=128, U(0.5,20)) | 67.15% | 32.85% 分歧, 强 PC κ 信号 |
| L2 (K=256, U(0.5,20)) | 75.69% | 24.31% 分歧, 中等 PC κ 信号 |

三层都远离 95% FAIL 边界 (差 ≥ 19pp) 和 60% TOO_STRONG 边界 (≥ 7pp). 全部"安全区中段".

### 4.3 跟全局 U(1,5) 配置对比

| 配置 | L0 | L1 | L2 |
|------|----|----|----|
| 全局 U(1,5) (task231) | 82.68% | **93.18%** ❌ (差 0.82pp 就 95% FAIL) | **94.60%** ❌ (差 0.40pp 就 95% FAIL) |
| **逐层混合 (本任务)** | 82.68% | **67.15%** ✅ | **75.69%** ✅ |

**逐层混合的关键收益**: L1/L2 释放几何参与度 (U(0.5,20) 比 U(1,5) 范围宽 4×), agreement 跌到 67-76% (vs 93-95%), 仍有充足空间不被 95% 边界压垮.

## 5. 决策 (per Issue #11 Gate 0 通过)

- ✅ 三层全 OPEN (60-90%): PASS
- ✅ 三层全 < 0.90: PASS
- ✅ 偏差 ≤ ±3pp: PASS (实测 0.00pp, 远低于 3pp)
- ✅ H1 维持 (per-layer c_k 区间可组合)

**进入 Gate 1**: Stage 1 40-epoch 训练 (Arm A: 三层 PC κ + 逐层 c_k range).

## 6. Gate 1 准备 (待启动)

### 6.1 Gate 1 任务设计

- 配置: Task #84 baseline ckpt 基础上跑 40-epoch 训练
- 逐层 c_k_range = [U(1,5), U(0.5,20), U(0.5,20)]
- c_k_seed = 42 (跟 Gate 0 一致)
- num_emb_list = [64, 128, 256], e_dim = 32, product_manifold = True, angular_dim = 4, radial_dim = 32
- 共享 task222 launcher 配置

### 6.2 通过条件 (硬停止)

| 条件 | 阈值 | 含义 |
|------|------|------|
| L0 utilization | ≥ 90% | §6.7.4 stop-loss (i) |
| collision_rate | ≤ 0.3706 | task222 best known (task236 统一口径) |

### 6.3 上游代码改动需求

Issue #11 Gate 1 需要 `--c_k_range_list` CLI flag (跟 Issue #9 任务 #235 的 `--assignment_mode_list` 同构). 改动量:
- `HG-Rec/train_hrqvae.py`: 加 `--c_k_range_list` argparse
- `HG-Rec/model/hrqvae.py`: 加 `c_k_range_list` 参数
- `HG-Rec/model/utils.py` (HResidualVectorQuantization): 加 `c_k_range_per_layer` 字段
- vq_layers 构造时每层独立 sample c_k

**这是不可逆上游改动 (R11.4 critical)**, 需要 dry-run 先报告再执行.

### 6.4 Gate 1 后路径

- Gate 1 PASS → Gate 2 (Stage 2 SID 推断)
- Gate 1 FAIL → Gate 1b (死码字复活补救) → 仍 FAIL → Issue #11 关闭

## 7. 产物

- descriptions/task241_issue11_gate0_perlayer_ck_combo.md (description)
- scripts/task241_issue11_gate0_perlayer_ck_combo.py (可复用, 任何 ckpt + per-layer c_k range 都能跑)
- verdicts/task241_issue11_gate0_perlayer_ck_combo_pass.md (本文件)
- /home/wlia0047/.claude/jobs/04ccf474/tmp/task241_issue11_gate0_results.json (原始数据)

## 8. Status

- ✅ Issue #11 Gate 0 PASS (H1 维持)
- ⏭️ Issue #11 Gate 1 待启动: 需上游 `--c_k_range_list` CLI flag 改动 (R11.4 critical decision, dry-run 后执行)
- ⏸️ 上下游: Issue #12 关闭 (Gate 0 NO-GO), Issue #10 redesign 4-arm 等用户
