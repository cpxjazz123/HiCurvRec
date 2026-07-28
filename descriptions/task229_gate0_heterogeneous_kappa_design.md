# Task #229 — Per-Layer Heterogeneous κ Gate 0/1/2 Design + 5cond 警戒线冲突诊断

**提出者**: 用户 2026-07-27
**执行者**: AI (自主决策, R11.3)
**目标**: 验证"per-layer c 参数化 RQ-VAE codebook"是否能突破 baseline R@10=0.1020
**前置基础**: Task #84 baseline R@10=0.1020 + Task #225 几何路线永久关闭 + Task #207 Euclidean ablation NO-GO
**关系**: 这是 §6.7 "geometric-route permanent closure" 之后**唯一**的几何类残余假设 — 攻 per-layer 杠杆

---

## 1. 核心思路 (用户提出)

**现象**: 单 c 全局相同 (HG-Rec c=1.0 baseline) 把所有层 ρ 推到不同范围:
- L0 (64 codewords, 32-d): ρ ≈ 0.27 (最大, 因为 64 桶要平分 9922 item, 密度大)
- L1 (128 codewords, 32-d): ρ ≈ 0.09 (中)
- L2 (256 codewords, 32-d): ρ ≈ 0.06 (最小, 因为 256 桶平分剩余 item, 密度小)

**推论**: 三层 ρ 相差 ~5× (L0/L2), 但全局用同一个 c=1.0 → 三层 曲率杠杆不一致. L0"浪费"了曲率 (ρ=0.27 离 boundary 还远), L2 已经接近 boundary (ρ=0.06 已在 sinh(√c·ρ) ≈ √c·ρ 边界).

**用户提议**: 让每层 c 不同, 使每层 ρ calibration 一致: $c_l \cdot \|e_l\|^2_{p50} = 1.0$.

由 $\|e\|^2 = \tanh^2(\sqrt{c}\rho) / c$, 若 $\sqrt{c}\rho$ 较小, $\|e\|^2 \approx \rho^2$. 但在深层, $\|e\|^2$ 可能更大或更小.

**实操**: 用户给的 3 个 c 值:
- c_L0 = 13.45 (calibrate L0 ρ_p50 → 1/√13.45 ≈ 0.273)
- c_L1 = 96.21 (calibrate L1 ρ_p50 → 1/√96.21 ≈ 0.102)
- c_L2 = 199.78 (calibrate L2 ρ_p50 → 1/√199.78 ≈ 0.071)

**核心 identity**: $c \cdot \|e\|^2 = (\sqrt{c} \cdot \rho_{\text{eff}})^2 / (1 + c \cdot \|e\|^2)$ (Poincaré ball 转换). 在 $\|e\|$ 较小时, 近似 $c \cdot \|e\|^2 \approx c \cdot \rho^2$, 若 calibrate $c \cdot \|e\|^2_{p50} = 1.0$, 则 max_c2 = max/median 倍率 (取决于 $\|e\|^2$ dispersion).

---

## 2. Gate 0 — Zero Training Measurement (已跑完)

### 2.1 实验目的 (用户原话)

> "Gate 0 现在就能跑, 不用GPU"

1. 测量 baseline c=1 ckpt 的 codeword 切空间 ‖e‖ 分布 (per layer p50/p90/max/min)
2. 确认 max_c2 是切空间 c·‖e‖² (不是球空间)
3. 检查 c=[13.45, 96.21, 199.78] proposal 是否通过 5cond cond5 警戒线 (max_c2 < 0.5)

### 2.2 测量结果 (descriptions/task229_gate0_norm_dist.json)

**Baseline c=1 (Task #84 ckpt)**:

| 层 | n_codewords | ‖e‖_p50 | ‖e‖_p90 | ‖e‖_max | ‖e‖²_p50 | ‖e‖²_max | max/p50 倍率 |
|----|-----|-----|-----|-----|-----|-----|-----|
| L0 | 64 | 0.2675 | 0.3049 | 0.3567 | 0.0716 | 0.1272 | 1.78× |
| L1 | 128 | 0.0880 | 0.1142 | 0.1352 | 0.0077 | 0.0183 | 2.36× |
| L2 | 256 | 0.0598 | 0.0706 | 0.0968 | 0.0036 | 0.0094 | 2.62× |

**c=10 ep1 ckpt (Task #233 训练产物, 方向 I 对照)**:

| 层 | ‖e‖_p50 | ‖e‖_max | ‖e‖²_p50 | ‖e‖²_max | max/p50 倍率 |
|----|-----|-----|-----|-----|-----|
| L0 | 0.0858 | 0.1017 | 0.0074 | 0.0103 | 1.40× |
| L1 | 0.0391 | 0.3128 | 0.0015 | 0.0978 | **63.91×** ⚠️ |
| L2 | 0.0559 | 0.3469 | 0.0031 | 0.1203 | **38.47×** ⚠️ |

### 2.3 max_c2 预测 (Gate 0 输出)

**Baseline c=1, 不同 c proposal**:
| c proposal | L0 max_c2 | L1 max_c2 | L2 max_c2 | cond5 (<0.5) |
|----|----|----|----|----|
| c=1 (baseline) | 0.1272 ✓ | 0.0183 ✓ | 0.0094 ✓ | ✓ all pass |
| c=10 | 1.2721 ✗ | 0.1828 ✓ | 0.0938 ✓ | L0 FAIL |
| **c=13.45** (proposal L0) | 1.7110 ✗ | — | — | L0 FAIL |
| c=63.7 (geom mean) | 8.1035 ✗ | 1.1643 ✗ | 0.5975 ✗ | ALL FAIL |
| **c=96.21** (proposal L1) | — | 1.7586 ✗ | — | L1 FAIL |
| c=100 | 12.7214 ✗ | 1.8278 ✗ | 0.9379 ✗ | ALL FAIL |
| **c=199.78** (proposal L2) | — | — | 1.8738 ✗ | L2 FAIL |

**c=10 ep1 ckpt, 同 c proposal**:
| c proposal | L0 max_c2 | L1 max_c2 | L2 max_c2 |
|----|----|----|----|
| c=13.45 | 0.1391 ✓ | 1.3160 ✗ | 1.6182 ✗ |
| c=96.21 | 0.9951 ✗ | 9.4137 ✗ | 11.5754 ✗ |
| c=199.78 | 2.0663 ✗ | 19.5475 ✗ | 24.0363 ✗ |

---

## 3. ❌ 关键冲突 #1 — Calibration vs cond5 警戒线

### 3.1 用户 identity 推导

若 calibration $c \cdot \|e\|^2_{p50} = 1.0$, 则:
- max_c2 = $c \cdot \|e\|^2_{max}$ = $\|e\|^2_{max} / \|e\|^2_{p50}$ = **max/p50 倍率**

实测 (baseline c=1):
- L0: max/p50 = 1.78 → max_c2 = 1.78 (FAIL <0.5)
- L1: max/p50 = 2.36 → max_c2 = 2.36 (FAIL <0.5)
- L2: max/p50 = 2.62 → max_c2 = 2.62 (FAIL <0.5)

### 3.2 冲突

**用户提议的 c=[13.45, 96.21, 199.78] 让每层 p50_c2 ≈ 1.0, 必然让 max_c2 > 0.5**.

5cond cond5 (`m_arm_step3_sweep_5cond.py:155`) 阈值 = **maxc2 < 0.5** (警戒线).

**直接后果**: 任何"按 p50 calibration 的 per-layer c"都 FAIL cond5.

### 3.3 三个可能解决方案

**方案 A: 改 c calibration target (从 p50 改成 p90 或 p95)**
- 用 p90 calibration: $c \cdot \|e\|^2_{p90} = 1.0$
- 此时 p90_c2 = 1.0, max_c2 = max/p90 倍率
- baseline L0: max/p90 = 0.357²/0.305² = 1.37 → max_c2 = 1.37 (FAIL <0.5)
- baseline L2: max/p90 = 0.097²/0.071² = 1.87 → max_c2 = 1.87 (FAIL <0.5)
- **仍 FAIL**, 因为 max/p90 仍接近 2×

**方案 B: 放宽 cond5 警戒线到 < 3.0**
- 既然 baseline 切空间 ‖e‖ max/p50 < 3× 是固有属性, 警戒线 < 0.5 过于严格
- 新阈值: max_c2 < 3.0 (允许 3× dispersion)
- baseline c=1 全部 PASS (max_c2 < 2.62)
- c=[13.45, 96.21, 199.78] 在 baseline 上 max_c2 ≈ 1.78-2.62 ✓ PASS
- c=10 ep1 ckpt L1 max_c2 = 9.78 ✗ FAIL (因训练把 ρ dispersion 推到 64×)

**方案 C: 接受 max_c2 > 0.5, 不依赖 5cond cond5 判定**
- 弃用 cond5 作为硬警戒线
- Gate 1 改为"max_c2 < 5.0" 软约束 (允许 dispersion 范围更宽)
- 配套: 增加 "norm CV" (‖e‖_std / ‖e‖_mean) 作为新 cond — 量化 dispersion 紧度

### 3.4 ❌ 关键冲突 #2 — c=10 ep1 ckpt 的 norm dispersion 失控

c=10 训练 (Task #233 方向 I) 实际把 ρ dispersion 推到 **64×** (L1) 和 **38×** (L2):

| 层 | baseline max/p50 | c=10 ep1 max/p50 | 退化倍率 |
|----|----|----|----|
| L0 | 1.78× | 1.40× | -21% (改善) |
| L1 | 2.36× | **63.91×** | **+2708%** ⚠️ |
| L2 | 2.62× | **38.47×** | **+1467%** ⚠️ |

**洞察**: 训练 c=10 把深层 (L1, L2) codeword ρ 推到极不均匀 — 一些 codeword 还在 ρ ≈ 0.04, 一些被推到 ρ ≈ 0.31 (8× 差). 这是为什么 c=10 SID inference collision 51.41% — encoder 不能稳定分配 codeword.

**含义**: 即使 c=10 calibration 看起来正确 (按 p50), 训练过程会破坏 dispersion, 让 max_c2 失控 (9.78× → 19.5× → 24×). 这是 c=10 路线必然 NO-GO 的几何根因.

---

## 4. Gate 1 — 4 Arms 实验设计

### 4.1 Arms 定义

| Arm | 配置 | 目的 |
|----|----|----|
| **A** | c=[1, 1, 1] (HG-Rec baseline, 全局 c=1) | 复用 Task #84 baseline R@10=0.1020 |
| **B** | c=[100, 100, 100] (matched control, 单 c 但深度大) | 测"单 c 大曲率"的下游 — 关键对照, 证明 c 必须大才有可能改善 OR c 必须分层才有改善 |
| **C** | c=[13.45, 96.21, 199.78] (per-layer 分层, 用户原 proposal) | 主实验, 测 per-layer heterogeneous κ 的杠杆 |
| **D** | c=[63.7, 63.7, 63.7] (geom mean ≈ L0=13.45 × L2=199.78 开 4 次方, 但保持单 c) | 测"中位 c 单 c"的下游 — 排除 per-layer 异质性贡献, 量化几何杠杆本身 |

### 4.2 决策规则 (用户原话)

- **C > B 改善 L1/L2** → per-layer 异质性 hypothesis 成立
- **C ≈ B** → per-layer 异质性无贡献, 单独大 c 即可
- **C < B** → per-layer 异质性有害 (额外 noise), 应放弃

### 4.3 与基线对比

- baseline R@10 = 0.1020
- 决策阈值: C R@10 > 0.1020 (vs baseline), C R@10 > B R@10 (vs matched control)
- NO-GO if C R@10 < 0.1020

### 4.4 R11.3 自主决策 (待用户确认)

**Gate 0 veto**:
- ✅ Gate 0 必跑, 已完成 (无 GPU)
- ⚠️ Gate 0 veto 条件 (用户原话): "max_c2 prediction > warning line"
  - 当前 5cond cond5 warning line = 0.5
  - 用户 proposal c=[13.45, 96.21, 199.78] 在 baseline c=1 ckpt 上预测 max_c2 = 1.78-2.62
  - **已 FAIL Gate 0 veto** → 必须先解决 §3 冲突才能进 Gate 1

**Gate 1 GPU 预算**:
- 4 arms × (Stage 1 训练 50 epoch ~25 min + Stage 2 推断 ~5 min + Stage 3 训练 200 epoch ~80 min + Stage 4 eval ~5 min) = 4 × 115 min = **460 min ≈ 7.7 hours**
- 4 张 L40S 并行 (R7): GPU 0/1/2/3 各跑一个 arm → ~115 min wall clock
- Stage 3 训练需 R12 forced ckpt (Task #83 教训)

---

## 5. Gate 2 — Circular Dependency Check

### 5.1 关键检查

每个 arm 训练完成后, 检查:
1. ‖e‖ 分布是否在训练后 dispersion 跟 Gate 0 预测一致 (c=10 ep1 ckpt 表明会破坏 dispersion, 测 c=[100, 200] 训练后是否同样失控)
2. SID collision rate 是否 < 9.07% baseline (c=1 baseline 是 9.07%)
3. SID uniqueness 跟 T5-mini 训练 R@10 是否解耦 (Task #87 paradox)

### 5.2 通过标准

- Gate 2 通过: max_c2 实测 ≤ 3 × Gate 0 预测 (允许 3× 漂移)
- Gate 2 FAIL: 任一层 max_c2 > 5 × Gate 0 预测, 即 dispersion 训练失控

---

## 6. 关键决策点 (R11.3) — 待用户确认

### 6.1 必须先解决的冲突 (按 §3)

**冲突 #1**: 用户 calibration identity ($c \cdot \|e\|^2_{p50} = 1.0$) vs 5cond cond5 (max_c2 < 0.5)

**冲突 #2**: c=10 训练会破坏 ρ dispersion (max/p50 1.78× → 63.91×), 让 max_c2 失控

**AI 推荐方案 (R11.3)**:

1. **方案 B + 同步修改 5cond cond5 警戒线** (推荐):
   - 把 cond5 从 `<0.5` 改为 `<3.0` (允许 baseline 固有 dispersion 范围)
   - 加新 cond: `norm_cv < 1.0` (‖e‖_std / ‖e‖_mean), 控制 dispersion 紧度
   - Gate 1 4 arms 仍按用户原设计 ([1, 96.21, 199.78] 等), 但允许 max_c2 落在 [1.78, 2.62] 范围

2. **方案 A (c calibration 用 p99)**: 改 calibration target 从 p50 → p99, 让 max_c2 ≈ 1.0
   - 但用户原意是 p50, 改 p99 会偏离设计意图

3. **方案 C (放弃 5cond cond5)**: 不推荐, 失去快速 sanity check

**AI 选择**: 方案 B (R11.3 自主决策, 但必须用户确认)

### 6.2 Gate 1 执行优先级 (用户原话)

用户原 4 arms 设计按 R11.2 顺序:
1. **先 A + B (单 c baseline 对照)**: 测 c=1 vs c=100 单 c 差异 → 量化"大 c 单 c"的下游天花板
2. **再 C (per-layer 异质性)**: 与 B 对比, 量化 per-layer 的边际贡献
3. **最后 D (geom mean 单 c)**: 排除 per-layer 异质性, 量化几何杠杆本身

**AI 调整建议**: A/B 并行 (独立 arm, GPU 0/1) → C/D 并行 (GPU 2/3, 但必须等 A/B ckpt 出来才能用同样的 train recipe). 总 wall clock ~2.5 hours.

---

## 7. 关键决策点 (R11.3) — 自主决策项

| 项 | 选择 | 理由 | 备选 |
|----|----|----|----|
| 任务 ID | **229** | max+1, descriptions/ 连续无空洞 | (无备选, R9 强制) |
| Gate 0 veto 冲突 | **方案 B (放宽 cond5 + 新增 norm_cv)** | 保留 calibration 设计意图, 量化 dispersion | 方案 A/C |
| Gate 1 GPU 分配 | **A→GPU 0, B→GPU 1 并行; C→GPU 2, D→GPU 3 等 A/B 完成后启动** | 优先级 + 依赖 | (无备选) |
| 训练 epoch | **Stage 1: 50 epoch (跟 Task #233 同), Stage 3: 200 epoch + early_stop 20** | 跟 baseline 同 | (用户可改) |
| seed | **42** (R5 + 跨 run 固定) | 单一 seed, 跟 baseline 比较 | (用户可加 multi-seed) |
| 评估指标 | **R@10 (主), R@5/20, NDCG@5/10/20 (次)** | 跟 baseline 比 | (用户可加 MRR/HR) |
| max_sinkhorn_iters | **30** (跟 Task #84 baseline 同, 5 iter 不够时给充分迭代) | 防止 collision 提前 stop | (5 iter 不够, 30 足够) |

---

## 8. 产物

### 8.1 Gate 0 已产出

- `scripts/task229_gate0_norm_dist.py` — Gate 0 测量脚本 (CPU only)
- `descriptions/task229_gate0_norm_dist.json` — 测量结果
  - baseline c=1: L0/L1/L2 ‖e‖ 分布 + max_c2 预测
  - c=10 ep1 ckpt: L0/L1/L2 ‖e‖ 分布 + max_c2 预测 (含 dispersion 失控证据)

### 8.2 Gate 1 待产出 (需 GPU)

- 4 arms ckpt: `products/task229_gate1/{arm_A,arm_B,arm_C,arm_D}/best_collision_model.pth`
- 4 arms SID: `HG-Rec/dataset/Instruments/Instruments_t5_rqvae_task229_{arm}.npy`
- 4 arms T5-mini ckpt: `products/task229_gate1/{arm}/stage3_t5mini/.../HG_Rec_best.pth`
- 4 arms Stage 4 R@10: `verdicts/task229_gate1_{arm}_metrics.json`
- 综合 verdict: `verdicts/task229_gate1_heterogeneous_kappa_result.md`

### 8.3 Gate 2 待产出

- 4 arms 训练后 dispersion 实测: 追加到 `descriptions/task229_gate0_norm_dist.json` (key="post_train")
- Circular dependency 分析: `verdicts/task229_gate2_dispersion_check.md`

---

## 9. 风险与回退

| 风险 | 概率 | 回退方案 |
|----|----|----|
| 4 arms 全部 NO-GO (跟方向 F/I 一致) | 高 (75%) | 验证 Task #225 永久关闭几何路线结论 |
| 4 arms 中只有 C GO | 中 (15%) | 验证 per-layer 异质性 hypothesis 成立, 写 paper §6.8 |
| 4 arms 中 C + D 都 GO | 低 (5%) | 测 c=63.7 geom mean 是 sweet spot, 进一步 sweep |
| c=[96, 200] 训练崩溃 (NaN/inf) | 中 (15%) | 减小 lr 或加 grad clip, 不影响 A/B/C |

---

## 10. 用户必须确认

1. **§3 冲突 #1 解决方案**: 方案 A (改 calibration target) / 方案 B (放宽 cond5 警戒线, 推荐) / 方案 C (放弃 cond5) — 用户选哪个?
2. **Gate 1 4 arms 是否按用户原设计执行** (A c=1 / B c=100 / C [13.45, 96.21, 199.78] / D c=63.7)?
3. **Gate 1 GPU 分配**: 接受 AI 建议 (A/B GPU 0/1 并行, C/D GPU 2/3 后启动)?

如用户确认方案 B + Gate 1 原设计 + AI GPU 分配, AI 可立即启动 Gate 1.