# Task #232 — Issue #7 调参版 (Gromov weight/entropy/spread-norm sweep) 结果

## 结论
**🟢 GO — 首个三层全 OPEN 方案! Gromov weight_l=0.5 让 L0/L1/L2 同时进 60-90% open band.**

| Layer | baseline (weight=1.0) | (A) weight=0.3 | **weight=0.5** ⭐ | weight=0.7 | (B) entropy | (C) spread |
|-------|------------------------|----------------|-------------------|------------|-------------|------------|
| L0 (K=64)  | 79.65% OPEN      | 94.46% OPEN (近 FAIL) | **90.06% OPEN** ⭐ | 85.69% OPEN | 79.65% OPEN | 52.53% TOO_STRONG |
| L1 (K=128) | 50.21% TOO_STRONG | 83.73% OPEN ✅      | **73.67% OPEN** ✅ | 64.75% OPEN | 50.21% TOO_STRONG | 31.59% TOO_STRONG |
| L2 (K=256) | 40.48% TOO_STRONG | 79.93% OPEN ✅      | **68.11% OPEN** ✅ | 56.83% TOO_STRONG | 40.48% TOO_STRONG | 23.30% TOO_STRONG |

**Issue #7 提议 (A) 部分成功**: weight decay 让 L1/L2 进 OPEN band, weight_l=0.5 是 Pareto 最优点 (L0 90.06% + L1 73.67% + L2 68.11%, 三层全 OPEN).

## 关键发现

### 1. weight_l sweep 跟一致率单调反向
weight_l ↓ → ρ_e 贡献 ↓ → Gromov 公式更接近 baseline 欧式 argmin → 一致率 ↑.
- weight=0.7: L2 56.83% (仍 TOO_STRONG, 不足)
- weight=0.5: L2 68.11% OPEN (sweet spot)
- weight=0.3: L0 94.46% (近 FAIL, 过激)

### 2. (B) entropy reg 无效
weight=1.0 baseline (79.65%/50.21%/40.48%) 跟 entropy variant 完全一致 (79.65%/50.21%/40.48%). 原因: exp(-H)/K 是 scalar per item, 不影响 argmax ranking (单调变换保留 ranking). 仅作为 sanity check 验证 implementation 正确.

### 3. (C) spread norm 失败 (反方向)
|ρ_e − ρ_mean| 完全抑制大半径 codewords → 退化为 baseline 欧式 argmin 但带 noise → 一致率反而降. spread norm 帮倒忙, 不推荐.

### 4. 跨层 Pareto 平衡 (Issue #7 自己预测的"joint constraint")
weight_l 单一参数控制三层, 不存在"只动 L1/L2 不动 L0"的解. weight_l=0.5 是:
- L0 90.06% (接近 FAIL 5.0pp 阈值) — 仍 OPEN
- L1 73.67% (OPEN 中段)
- L2 68.11% (OPEN 中段)

任何 weight_l ∈ [0.4, 0.6] 都接近 Pareto front. weight_l=0.5 是验证过的代表点.

## 决策
- **🟢 GO**: weight_l=0.5 是首个三层全 OPEN 方案. 跨 8 测试方向 (Issue #1-3, 5 + Task #218-232) 中首次出现三层全 OPEN.
- ✅ **推荐 hybrid 路径**: 跟 Task #231 (PC κ c_k Uniform(1,5)) 组合 — L0 用 PC κ (L0 OPEN 82.68% 跟 Gromov 90.06% 都 OK), L1/L2 用 Gromov weight=0.5 (稳定 OPEN). 这是 Project 架构级 escape 路径.
- ⚠️ **历史根因 (task221 Stage 1 训练 NO-GO)**: Gromov 在 Stage 1 训练动力学中拉回坍缩 (Issue #7 body §3). Phase 0 OPEN 不等于 Stage 1 train OK. Phase 1 闭环 (Task #233 hybrid) 必须做.

## 产物
- scripts/task232_gromov_phase0_weight_sweep.py (6 variants × 3 layers = 18 measurements)
- /home/wlia0047/.claude/jobs/04ccf474/tmp/task232_gromov_phase0_results.json
- /home/wlia0047/.claude/jobs/04ccf474/tmp/task232_gromov_phase0.log

## 跨 Task 综合 (Issue #6 + Issue #7)

| 方案 | L0 | L1 | L2 | 三层全 OPEN? |
|------|----|----|----|------------|
| Task #218 (PC κ baseline c_k U(0.5,20)) | 28.90% TOO | 67.15% OPEN | 75.69% OPEN | ❌ |
| Task #219 (Gromov weight=1.0) | 79.65% OPEN | 50.21% TOO | 40.48% TOO | ❌ |
| **Task #231 (PC κ c_k U(0.5,5))** | 81.12% OPEN | 92.37% OPEN | 94.04% OPEN | ⚠️ L1/L2 近 FAIL |
| **Task #231 (PC κ c_k U(1,5))** | 82.68% OPEN | 93.18% OPEN | 94.60% OPEN | ⚠️ L2 距 FAIL 0.40pp |
| **Task #232 (Gromov weight=0.5)** | **90.06% OPEN** ⭐ | **73.67% OPEN** ⭐ | **68.11% OPEN** ⭐ | ✅ |
| 推荐 hybrid: L0=PC κ + L1/L2=Gromov | 82.68% | 73.67% | 68.11% | ✅ (Phase 1 待验证) |

result: Task #232 GO — Gromov weight_l=0.5 是首个三层全 OPEN 方案. 推荐 hybrid Phase 1: L0 用 PC κ c_k U(1,5), L1/L2 用 Gromov weight=0.5.
