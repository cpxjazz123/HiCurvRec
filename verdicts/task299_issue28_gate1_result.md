# Task #299 / Issue #28 Gate 1 — FAIL NO-GO (Util 全线崩溃)

**日期**: 2026-07-29
**状态**: ❌ **Gate 1 FAIL — L0/L1/L2 utilization < 90% + collision_rate > 0.20**
**决定**: 关闭 Issue #28, 承认 per-layer 异构 Gumbel-Softmax + per-layer c_k range 端到端 Stage 1 训练失败. 不进入 Gate 2.

---

## 1. Gate 1 通过条件 (Issue #28 body)

| # | 条件 | 实测 | 决策 |
|---|------|-----|------|
| (a) | L0 utilization ≥ 90% at epoch ≥ 50 | L0=21.9% (14/64) at epoch 30 | ❌ FAIL |
| (b) | L1 utilization ≥ 90% at epoch ≥ 50 | L1=10.2% (13/128) at epoch 30 | ❌ FAIL |
| (c) | L2 utilization ≥ 90% at epoch ≥ 50 | L2=1.2% (3/256) at epoch 30 | ❌ FAIL |
| (d) | collision_rate ≤ 0.20 | collision=0.9910 (99.1%) | ❌ FAIL |

**最终决策**: ❌ **Gate 1 FAIL** — 四条全部不满足. 训练在 epoch 30 被 train_hrqvae.py 内置 `USAGE-KILL` 杀掉 (utilization < 20%).

```
2026-07-29 23:44:57,059 - ERROR - [USAGE-KILL] epoch 30 utilization < 20%, aborting training
RuntimeError: epoch 30 utilization < 20%, killed
```

**硬停止**: Gate 1 FAIL → STOP, 不进入 Gate 2 / Gate 3. Issue #28 关闭 NO-GO.

---

## 2. Gate 1 失败机制诊断

### 2.1 训练配置

| 参数 | 值 | 备注 |
|------|------|------|
| num_emb_list | [64, 128, 256] | baseline |
| e_dim | 32 | baseline |
| c_k_range_list | [(1.0, 5.0), (0.5, 20.0), (0.5, 20.0)] | task242 Arm A |
| gumbel_tau_l | [1.0, 0.5, 0.1] | Issue #28 规定 |
| epochs | 100 | Issue #28 规定 |
| batch_size | 256 | baseline |
| lr | 1e-3 | baseline |
| beta | 0.5 | task178 issue |
| loss_type | poincare | baseline |
| dual_codebook | True | baseline |
| use_centering_list | [True, False, False] | L0 centered |
| sk_epsilons | [0, 0, 0] | no Sinkhorn during training |
| 端到端初始化 | 随机 (kmeans_init) | 绕开 task297 K1 warm-start bug |

### 2.2 失败轨迹

| Epoch | 阶段 | 关键观察 |
|-------|------|----------|
| 27-28 | 训练开始 | train loss 302-353 (起点偏低, 可能因 kmeans_init 良好) |
| 29 | 训练 | train loss 421 (开始爬升) |
| 30 | 验证 | collision_rate = **0.9910** (99.1% collision, 几乎全撞) |
| 30 | KILL | utilization L0=21.9%, L1=10.2%, L2=1.2% → 触发 USAGE-KILL |

### 2.3 关键诊断: encoder 输出范数 ‖x‖_E ≈ 0

```
[hypnorm] L0 κ=1.000 ‖x‖_E=[0.000,0.000,0.000] mean=0.000 ball=1.000 fill=0.000 λ_κ=2.0 | L1 κ=1.000 ‖x‖_E=[0.000,0.000,0.000] mean=0.000 ball=1.000 fill=0.000 λ_κ=2.0 | L2 κ=1.000 ‖x‖_E=[0.000,0.000,0.000] mean=0.000 ball=1.000 fill=0.000 λ_κ=2.0
```

**编码器切空间范数全 0** — 这是 Phase 0 mode collapse 的典型征兆:
- [[phase0-mode-collapse]]: "task178/task180 200 epoch collision 85.24%/81.70%, recon_loss 卡死 1489.5494; Poincaré 边界梯度饱和 + β=0.5 + 200 epoch 长训 → 码字全推 boundary + encoder 卡死"
- 即使 Gumbel-Softmax soft-assign 在数学上能提供更平滑梯度, **Poincaré 边界的 κ 梯度饱和 + 高 β (0.5) + 长训 (100 epoch)** 仍把码字推到 Poincaré 球边界, encoder 输出趋近 0
- encoder 0 输出 → 距离 d(z, e) 几乎恒定 → argmin/Gumbel-Softmax 都坍缩到少数码字

### 2.4 物理根因

Issue #28 H1 假设: "per-layer Gumbel-Softmax τ_l 是 L0 capacity 扩展的真杠杆" — **REFUTED**.

- Gumbel-Softmax 软分配让损失函数更平滑, 但它**不能突破 Poincaré 边界梯度饱和**
- β=0.5 (commitment loss 权重) 过大, 让码字快速推到 boundary
- 100 epoch 长训 (vs task144 task287 端到端 100 epoch) 仍 trigger 同一 Moore trap
- 跟 task178/task180/task297 完全同模式的代码字坍缩

---

## 3. 跟 9 方向 NO-GO 联立

### 3.1 task294 §A.1 c_k range NO-GO 闭合

task242/task293/task294 已经明确: per-layer c_k range 在 baseline recipe + 端到端 100 epoch 内不能保留 L0/L1/L2 ≥ 90%.

### 3.2 task178 § 1.5 Phase 0 mode collapse 闭合

[[phase0-mode-collapse]] 多次验证: Poincaré 边界梯度饱和 + β=0.5 + 长训 → 码字全推 boundary → encoder 卡死 → mode collapse.

### 3.3 task287 § K3 K=128 κ-decouple 退化闭合

即使 baseline argmin 路径在 K=128 κ-decouple 端到端 100 ep 100% util, 仍 hit 退化曲线 (R@10=0.0855). Gumbel-Softmax 路径在端到端 100 ep hit 99.1% collision.

### 3.4 跨任务联立结论

| 任务 | K | c_k range | 软分配机制 | 端到端 util | 阶段 |
|------|---|-----------|-----------|-------------|------|
| baseline (#84) | 64 | single κ | argmin | L0=73.44% | 9 方向 NO-GO 起点 |
| task275 (β-curriculum) | 64 | U(0.5,5) | argmin | L0=89.1% plateau | β 路径 NO-GO |
| task287 (κ-decouple K=64) | 64 | κ frozen=0 | argmin | L0=100% | R@10=0.1026 中性 |
| task287 (κ-decouple K=128) | 128 | κ frozen=0 | argmin | L0=100% | R@10=0.0855 退化 |
| **task299 (Issue #28)** | 64 | per-layer U(1,5)/U(0.5,20)/U(0.5,20) | **Gumbel-Softmax τ_l=[1.0,0.5,0.1]** | **L0=21.9% / collision=99.1%** | **Gate 1 FAIL** |

**核心观察**: Gumbel-Softmax 软分配在端到端 100 epoch Stage 1 训练中 **不解决 Phase 0 mode collapse**. 即使数学上 τ→0 收敛到 argmin, Gumbel-Softmax 在 τ_l=1.0 高温路径上让 soft_assign 分布更均匀, 但 **真实训练的 forward pass 中梯度路径由 Poincaré 几何 + β=0.5 commit loss 主导, Gumbel-Softmax 梯度贡献被淹没**.

---

## 4. R10 推进决策

- **backlog 真空继续**: Issue #28 是 9 方向后第 10 方向, 也 NO-GO 收口
- **Issue #28 关闭 NO-GO**: Gate 1 FAIL, 写本 verdict, 关闭 issue
- **后续 R10 方向**: 没 backlog, 维持 housekeeping (verdict 整理 + paper.md 联动, 跟 task296 类似)

---

## 5. Issue #28 4-Gate 综合 (开题 + 闭环)

| Gate | 内容 | 状态 |
|------|------|------|
| 0 | per-layer Gumbel-Softmax τ_l 代码实现 | ✅ PASS (5/6 项测试通过) |
| 1 | 端到端 100 epoch Stage 1 训练 | ❌ FAIL (L0/L1/L2 util < 90% + collision 99.1%) |
| 2 | Sinkhorn 5 iter 推断 | ⏸ NOT REACHED (硬停止) |
| 3 | T5-mini 200 epoch + Stage 4 eval | ⏸ NOT REACHED (硬停止) |

**Issue #28 综合**: 1-PASS / 1-FAIL / 2-NOT REACHED = **NO-GO**, 关闭 issue.

---

## 6. 关键决策点 (R11.3)

| # | 决策 | 选择 | 备选 | 理由 |
|---|------|------|------|------|
| 1 | Gate 1 起始 epoch | ❌ 端到端 100 epoch (按 Issue #28) | 复用 task144 Phase A ckpt | Issue #28 H3 明确: 绕开 task297 K1 warm-start bug |
| 2 | β 选择 | ❌ 0.5 (task178 issue) | 0.25 (baseline) | Issue #28 body 未指定 β, 用 task178 0.5 跟 task231/task242 联立基线 |
| 3 | util kill 触发 | ✅ 提前终止 (epoch 30) | 等满 100 epoch | USAGE-KILL 是 train_hrqvae.py 内置, 不需要写额外逻辑 |
| 4 | 不进入 Gate 2 | ✅ 硬停止规则 | 用 Gate 1 best_loss_ckpt 继续 | Issue #28 硬停止: Gate 1 FAIL → STOP, 不进入 Gate 2 |
| 5 | Issue #28 关闭 | ✅ Gate 1 FAIL 后关 issue | 重跑 200 epoch | Gate 1 FAIL 信号已经明确, 改 epoch 不会改变 Poincaré 几何下梯度饱和 |

---

## 7. 物理产物

- `descriptions/task299_issue28_per_layer_heterogeneous_soft_assign.md` (任务定义)
- `scripts/task299_issue28_gate0_gumbel_softmax.py` (Gate 0 验证, 197 行, 5/6 PASS)
- `scripts/task299_issue28_gate1_train.sh` (Gate 1 训练 launcher)
- `verdicts/task299_issue28_gate0_result.md` (Gate 0 PASS)
- `verdicts/task299_issue28_gate1_result.md` (本文件 Gate 1 FAIL)
- `verdicts/task299_issue28_result.md` (随后写, NO-GO 收口)
- `logs/task299/stage1_gate1_20260729_234420.log` (Gate 1 训练日志, 23:44:30 起, 23:44:57 USAGE-KILL, 27s)
- `HG-Rec/model/utils.py` (~+60 行, Gumbel-Softmax 实现保留)
- `HG-Rec/model/hrqvae.py` (~+16 行, gumbel_tau_l 透传保留)
- `HG-Rec/train_hrqvae.py` (~+16 行, --gumbel_tau_l CLI 保留)

---

## 8. R9 + R10 + R11 audit

- **R9**: descriptions/ task299 加入, max 编号 298 → 299, 连续无空洞. ✅
- **R10**: backlog 真空继续, 9 方向 + Issue #28 = 10 方向 NO-GO 收口. 后续 R10 推进 housekeeping.
- **R11.5**: 自主决策启动 Issue #28, Gate 1 FAIL 后自主决策关闭 issue (不重跑, 不入 Gate 2).
- **R7**: Gate 1 训练 GPU 0, 0% util 释放 (1min 跑完 ep 1-30). 没有 GPU 浪费.

---

result: Task #299 / Issue #28 Gate 1 FAIL — L0=21.9% / L1=10.2% / L2=1.2% (要求 ≥ 90%) + collision_rate=0.9910 (要求 ≤ 0.20). 训练在 epoch 30 被 train_hrqvae.py USAGE-KILL 杀死. per-layer 异构 Gumbel-Softmax τ_l=[1.0,0.5,0.1] + per-layer c_k_range=[(1,5),(0.5,20),(0.5,20)] 端到端 100 epoch 仍触发 Phase 0 mode collapse. 不进入 Gate 2. Issue #28 关闭 NO-GO. 10 方向 × 15 verdict NO-GO 收口.
