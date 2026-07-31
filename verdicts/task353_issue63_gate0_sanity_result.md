# Task #353 / Issue #63 Gate 0 — zero-GPU sanity 8/8 PASS

**日期**: 2026-07-31
**前置**: Gate -1 8/8 PASS (task352)
**任务**: Gate 0 zero-GPU 数值 sanity (8 项)
**结果**: ✅ Gate 0 PASS (8/8)

---

## 1. Gate 0 spec

| Test | 名称 | 关键数值 | 结果 |
|------|------|----------|------|
| T1 | κ=0 退化 ≡ Euclidean | max diff = 9.54e-07 | ✅ |
| T2 | κ=max 距离 bounded | d_max = 0.5638 < π/√κ ≈ 2.26 | ✅ |
| T3 | 距离对称性 | d(x,y)=d(y,x), d(x,x)=0 | ✅ |
| T4 | Finite + triangle inequality | violation = -0.207 ≤ 0 | ✅ |
| T5 | Assignment entropy | Layer 0/1/2 util = 0.500/0.547/0.559 | ✅ |
| T6 | Forward path 无 NaN/Inf | batch 1/8/64 全部 finite | ✅ |
| T7 | Recon loss 范围合理 | mean = 1.0008 < 1000 | ✅ |
| T8 | Codebook util ≥ 0.5 | L0=0.500, L1=0.547, L2=0.559 | ✅ |

## 2. 关键发现

### T1 — κ=0 退化
- `kappa_m = 0 → κ-Stereo distance = L2 norm (Euclidean)`
- max diff = 9.54e-07 (浮点精度内)
- 证明: κ=0 时 `(2/√0) atan(...)` 路径塌缩到 Euclidean

### T2 — κ 数值饱和
- `theta_m = 2.0 → kappa_m = 2·tanh(2) = 1.928` (tanh squashing)
- 距离 max = 0.5638 (远小于理论上限 π/√κ ≈ 2.26)
- 所有 distance 数值稳定，无 NaN/Inf

### T3-T4 — 度量公理
- 对称性: d(x,y) ≡ d(y,x) (max diff = 0)
- 自反性: d(x,x) = 0 (max = 1e-06, 浮点误差内)
- Triangle inequality: d(x,z) ≤ d(x,y) + d(y,z), violation = -0.207 ≤ 0 ✓

### T5-T8 — Codebook 健康
- Layer 0 (K=64): util = 0.500, entropy = 1.938/4.159 nats
- Layer 1 (K=128): util = 0.547, entropy = 2.779/4.852 nats
- Layer 2 (K=256): util = 0.559, entropy = 3.855/5.545 nats
- random init uniform → 健康 baseline (Stage 1 训练后应该 ≥ 90%)

## 3. Gate -1 + Gate 0 综合

✅ **两层 zero-GPU 预检全部 PASS**:
- Gate -1 (8/8): 实施基础 + 隔离 + 优化器 + forward + 接口
- Gate 0 (8/8): 数值稳定性 + 度量公理 + codebook baseline

进入 Gate 1: Stage 1 GPU 训练 (per Issue #63 spec).

## 4. Gate 1 spec

**目标**: Stage 1 训练 (Musical_Instruments + HG-Rec encoder + FreeCurvHRQVAE 顶层)
- L0/L1/L2 utilization ≥ 90%
- collision rate 跟 #55/#56 (SID collapse) 对比必须改进
- ckpt 强制保存 (R12)

**GPU 分配**: 4×L40S, 选择空闲卡 (R7)
**预计时长**: 200 epoch × ~10 min = 估算 ~33 h

**STOP 条件**:
- 任一层 utilization < 90% @ epoch 30 → KILL (USAGE-KILL per Issue #55/#56)
- NaN/Inf 出现 → KILL
- collision > 50% @ epoch 30 → KILL

## 5. R11.5 决策

- Gate 0 数值 sanity 8/8 PASS, 进入 Gate 1
- Gate 1 训练使用 HG-Rec Stage 1 标准脚本 (Task #84 baseline 配方)
- num_hierarchies=3, Musical_Instruments dataset, seed=42

---

result: Issue #63 Gate 0 PASS 8/8. κ-Stereographic 距离数值 sanity 全部 PASS (κ=0 Euclidean 退化, κ=max bounded, 对称性, triangle inequality, codebook baseline 健康). 进入 Gate 1 Stage 1 GPU 训练.