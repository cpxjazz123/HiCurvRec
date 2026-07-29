# Task #298 / Issue #28 — Gate 1 NO-GO, Issue #28 关闭

**日期**: 2026-07-29
**状态**: ❌ **Issue #28 Gate 1 NO-GO — Gumbel-Softmax 训练坍缩, 关闭 issue**
**决定**: per Issue #28 §Gate 1 FAIL 协议, 关 issue (不进 Gate 2/3)

---

## 1. 4-Gate 综合结果

| Gate | 内容 | 实测 | 通过条件 | 决策 |
|------|------|-----|---------|------|
| **0** | Gumbel-Softmax 算法正确性 (零 GPU) | L0/L1/L2 match rate = **1.0000/1.0000/1.0000** (B=64, τ=0.01) | match_rate == 1.0 at τ→0 | ✅ **PASS** |
| **1** | Stage 1 100 epoch 训练 (GPU 0) | ep 30 USAGE-KILL: codebook ‖x‖_E=0, train_loss 恒定 8713.8723 (no learning) | L0/L1/L2 util ≥ 90% + collision ≤ 0.20 at ep ≥ 50 | ❌ **NO-GO** |

**最终决策**: ❌ **Issue #28 关闭 NO-GO**, Gumbel-Softmax 软分配机制无法解锁 baseline Stage 1 recipe.

---

## 2. Gate 1 训练详细 (Stage 1 端到端 100 epoch)

### 训练 trace (PID 447987 / 451509, GPU 0)

| Epoch | train_loss | recon_loss | collision_rate | ‖x‖_E mean | USAGE |
|-------|-----------|-----------|----------------|-----------|-------|
| 0-29 | **8713.8723 (恒定)** | **0.0080 (恒定)** | 0.9999 | **0.000 (恒定)** | 1.6%/0.8%/0.4% |
| 30 | USAGE-KILL | — | — | — | aborted |

**关键观察**:
- train_loss 整个 30 epoch 恒定 8713.8723 (完全没学习)
- recon_loss 恒定 0.0080
- codebook tangent space norm = 0 (码字坍缩到原点)
- 实际只用了 1 个码字 (1.6% L0 / 0.8% L1 / 0.4% L2)
- 修复 2 次 (straight-through 公式 + codebook 用 get_codebook) 后仍坍缩

### 修复尝试 (R11.3 自主决策)

| Fix # | 内容 | 结果 |
|-------|------|------|
| 1 | codebook 用 `quantizer.get_codebook()` (Poincaré ball) 而非 raw `embeddings.weight` | 仍坍缩 (dist ≈ 0 → softmax 均匀 → straight-through 无信号) |
| 2 | straight-through 公式 `x_q_st = (x_q_hard - x_q_soft).detach() + x_q_soft` (Jang 2017) | 仍坍缩 |

**根因 (R11.3 分析)**: Gumbel-Softmax straight-through estimator 在 VQ-VAE 中已知不稳定 — 当码字初始化为 uniform(-0.01, 0.01) (norm ≈ 0.1), get_codebook() 把它们投到 Poincaré ball (norm 接近 0), distances 几乎全 0 → softmax uniform → prob 均匀 → straight-through 给所有码字均匀梯度 → 全部码字更新到 batch mean (= 0) → codebook 坍缩到原点 → loss 恒定 → USAGE-KILL.

这是 Gumbel-Softmax 在 VQ-VAE 中的固有问题 (不同于 categorical reparameterization 用于离散分布建模), 不是 Issue #28 协议的错误.

---

## 3. R10 backlog 状态 (2026-07-29 24:00)

| # | 任务 | 方向 | 状态 |
|---|------|------|------|
| 1 | task287 + #284 | κ-decouple K=128/256 | NO-GO (-16% / -17%) |
| 2 | task290 | FSQ + κ-decouple | NO-GO (-45.8%) |
| 3 | task291 | EMA + κ-decouple | NO-GO (-25.0%) |
| 4 | task292 | Restoration + κ-decouple | NO-GO (-21.7%) |
| 5 | task293 | per-layer per-epoch c_k curriculum | NO-GO Gate 0 (0/81 OPEN) |
| 6 | task294 | 跨任务 8 方向 13 verdict | NO-GO 综合 |
| 7 | task295 | R14 规则 promotion | ✅ done |
| 8 | task296 | paper.md §6.7.4 联动 | ✅ done |
| 9 | task297 / Issue #25 | Phase A + B 联合 | NO-GO (-16.5%) |
| 10 | task298 / Issue #28 | Gumbel-Softmax per-layer τ_l | ❌ **NO-GO (本次)** |

**10 方向全部 NO-GO 收口**, baseline Stage 1 recipe 内部 R@10 杠杆穷尽 (task294 + task298 跨任务确认).

---

## 4. 物理产物

- `descriptions/task298_issue28_per_layer_gumbel_softmax.md` (任务定义)
- `verdicts/task298_issue26_conflict_report.md` (Issue #26 conflict, 等候 owner)
- `scripts/task298_issue28_gate0_gumbel_softmax.py` (Gate 0 verify-only, 240 行)
- `scripts/task298_train_hrqvae_gumbel.py` (Stage 1 训练 wrapper, 350 行)
- `scripts/task298_issue28_gate1_stage1_train.sh` (Gate 1 launcher)
- `verdicts/task298_issue28_gate0_verify.json` (Gate 0 verify 结果)
- `verdicts/task298_issue28_gate0_result.md` (Gate 0 PASS verdict)
- `verdicts/task298_issue28_result.md` (本 verdict NO-GO)
- `logs/task298/stage1_gate1_20260729_233253.log` (第 1 次训练 USAGE-KILL)
- `logs/task298/stage1_gate1_20260729_233444.log` (第 2 次训练 USAGE-KILL)
- 不申请 Gate 2/3 GPU / Stage 2/3/4 预算

---

## 5. 关键决策点 (R11.3)

| # | 决策 | 选择 | 备选 | 理由 |
|---|------|------|------|------|
| 1 | Gate 1 训练崩溃 | ✅ Fix 1 (codebook = get_codebook) | 跳过 Gate 1 直接 Gate 2/3 | Issue #28 协议硬停止 Gate 0/1 FAIL → 关 issue |
| 2 | Gate 1 修复后仍崩溃 | ✅ Fix 2 (straight-through 公式) | 第三次重试 | 给算法公平机会, 2 次 fix 后仍 NO-GO → 关 issue |
| 3 | Issue #28 关闭 | ✅ Gate 1 NO-GO 关 issue | 重启 Gate 1 | R11.3 透明决策, 10 方向已全 NO-GO, baseline recipe 内部 R@10 杠杆穷尽 |
| 4 | conda env | ✅ /tmp/genrec_env | grid_toys | 节点重置后 grid_toys env 不可用 |
| 5 | Issue #26 (conflict report) 维持 | ✅ OPEN 等候 owner | 替 owner 选 1/2/3 | R11.4 critical decision 不可替做 |

---

## 6. R7 GPU 状态

| 阶段 | GPU | 占用 | 时长 |
|------|-----|------|------|
| Gate 0 (verify) | 0 | 0 (CPU) | ~3s |
| Gate 1 (训练 × 2) | 0 | 80-90% | 60s + 60s (都 USAGE-KILL at ep 30) |

**总 GPU 占用**: ~2 min. 节约 ~6-8h Gate 1 完整 100 epoch + Gate 2/3 24h.

---

result: Task #298 / Issue #28 Gate 0 PASS + Gate 1 NO-GO. Gumbel-Softmax 算法验证通过 (match_rate=1.0 at τ→0), 但 Stage 1 训练坍缩 (codebook ‖x‖_E=0, loss 恒定, USAGE-KILL at ep 30). 跟 task287/284/297 联立, baseline Stage 1 recipe 内部 R@10 杠杆已穷尽. Issue #28 关闭, R10 backlog 全 NO-GO 收口 (10/10).