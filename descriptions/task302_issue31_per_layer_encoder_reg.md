# Task #302 / Issue #31 — per-layer 异构 encoder regularization (β_l + α_l + γ_l + c_k range)

**状态**: 🟡 **Gate 0 wrapper 设计 + reg test (Issue #31 4-Gate 协议 §1)**
**关联**: Issue #31 (2026-07-29 13:55:32Z OPEN) + Issue #30 Gate 1+2 PASS + Issue #29 NO-GO closed
**日期**: 2026-07-30

---

## 1. 背景与动机 (Issue #31 body §1)

承接 Issue #28 closure comment owner-verdict (2026-07-29 23:45) 锁定的根因 —— **encoder-side boundary saturation** (Poincaré 边界梯度饱和 + β=0.5 commit loss 主导 + 长训 → 码字全推 boundary + encoder 切空间范数 ‖x‖_E → 0). 跟 task178 / task180 / task231 / task242 / task299 五处共享同一根因. Issue #29 / Issue #30 都没解决这个根因 (Issue #29 失败 + Issue #30 走的是 codebook 几何而非 encoder 正则).

**Issue #31 直接处理根因**: per-layer 异构 commit loss β_l + per-layer 异构 codebook center anchor α_l + per-layer 异构 encoder L2 γ_l + per-layer c_k range (task242 Arm A 协同). 三种独立机制同向 —— 防 encoder trivial solution + 防 codebook boundary collapse + 防 encoder trivial gradient.

---

## 2. 实验设计 (Issue #31 body §"实验设计" + §"假设")

**自变量**: 四维 per-layer 异构机制

| 变量 | 默认值 | 角色 |
|------|--------|------|
| `beta_list` | `[0.1, 0.3, 0.5]` | per-layer 异构 commit loss 权重 (L0 commit loss 降权到 0.1) |
| `alpha_list` | `[0.01, 0.005, 0.001]` | per-layer 异构 codebook center anchor (L0 强 anchor 防止 boundary) |
| `gamma_list` | `[0.001, 0.0005, 0.0001]` | per-layer 异构 encoder L2 正则 (L0 强 L2 防 trivial) |
| `c_k_range_list` | `[(1,5), (0.5,20), (0.5,20)]` | per-layer c_k range (task242 Arm A) |

**受控**: baseline recipe (argmin hard-assign + sinkhorn) + `num_emb_list=[64,128,256]` + `e_dim=32` + seed 42 + Musical_Instruments 5-core + T5-mini 9.18M / 200 epoch / early_stop=20.

**跟现有方向对比**: 唯一未尝试的组合 = per-layer 异构 c_k + per-layer 异构 encoder regularization (β_l + α_l + γ_l) = **per-layer 三维异构 (metric + encoder regularization)**, 直接处理 boundary saturation 根因. NORTH STAR §4 豁免 (per-layer 异构机制本身).

---

## 3. 4-Gate 协议 (Issue #31 body 锁定的硬停止条件)

### Gate 0 — 实现 per-layer 异构 encoder regularization 训练代码

**目标**: 实现 `train_hrqvae_encoder_reg.py` 继承 baseline, 新增 per-layer β_l + α_l + γ_l + c_k range 参数, 不修改 HG-Rec/model/ 上游源码.

**通过条件**:
- wrapper 文件在 `scripts/` 下提交
- baseline `train_hrqvae.py` Stage 1 forward pass 在 `β_l=[0.5,0.5,0.5]` + `α_l=[0,0,0]` + `γ_l=[0,0,0]` 输入下输出一致 (回归测试, 三组正则全 0 → baseline 等价)
- `‖x‖_E` 打印逻辑与 baseline 一致 (新增打印, 不替换 baseline)

**硬停止**: Gate 0 FAIL → STOP. 不进入 Gate 1.

### Gate 1 — Stage 1 训练 (100 epoch)

**通过条件 (a-e 五条同时)**:
- (a) L0 utilization ≥ 90% at any eval step ≥ ep50
- (b) L1 utilization ≥ 90% at any eval step ≥ ep50
- (c) L2 utilization ≥ 90% at any eval step ≥ ep50
- (d) collision_rate ≤ 0.20
- (e) ‖x‖_E ≥ 0.3 at any eval step ≥ ep50

**硬停止**: 任一不满足 → STOP, 不进入 Gate 2/3.

### Gate 2 — Sinkhorn 推断 (cheap)

**通过条件**:
- 4-digit SID unique count ≥ 9500 / 9922
- per-layer utilization 与 Gate 1 终态偏差 ≤ 5pp

**硬停止**: unique < 9500 → STOP.

### Gate 3 — Stage 3 T5-mini 200 epoch + Stage 4 R@10

**通过条件**: Test R@10 > 0.1020 (HG-Rec baseline #84).

**硬停止**: R@10 ≤ 0.1020 → STOP, 关闭 Issue #31.

---

## 4. 当前阶段 (Gate 0)

本任务 = **Issue #31 Gate 0 wrapper 设计 + reg test**, 不进入 Gate 1 (等 Issue #30 Gate 3 R@10 实证 + Issue #31 wrapper 自身 reg test PASS).

**Gate 0 实施**:
1. ✅ 创建本描述文件 (`descriptions/task302_issue31_per_layer_encoder_reg.md`)
2. ⏳ 创建 wrapper 文件 (`scripts/task302_issue31_gate0_wrapper.py`)
3. ⏳ 跑 reg test: forward 输出 baseline 等价 (β_l all 0.5, α=γ=0)
4. ⏳ 写 `verdicts/task302_issue31_gate0_result.md`
5. ⏳ commit + 更新 loop.md §16

---

## 5. 关键技术 (R11.5 决策备注)

### 5.1 β_l per-layer patch (非 invasive)

baseline `HResidualVectorQuantization.__init__` 已经传 `self.beta` 给每个 HVectorQuantization layer. wrapper 不修改 `__init__`, 而是在构造完成后 patch:

```python
for li, q in enumerate(model.hrq.vq_layers):
    q.beta = beta_list[li]  # patch per-layer β
```

这样不修改 `HG-Rec/model/utils.py`, 遵守 R11.4 critical decision (上游源码保护).

### 5.2 α_l + γ_l 正则项

baseline forward 返回 `mean_loss` (per-layer losses mean). wrapper 在 forward 后追加:

```python
anchor_loss = sum(alpha_list[li] * (q.embeddings.weight ** 2).sum() for li, q in enumerate(model.hrq.vq_layers))
enc_loss = sum(gamma_list[li] * (encoder_output_l ** 2).sum() for li in encoder_outputs)  # 需要 hook
total_loss = rq_loss + anchor_loss + enc_loss
```

α_l/γ_l=0 时, anchor_loss + enc_loss = 0, total_loss = rq_loss, baseline 等价 ✅.

### 5.3 跟 Issue #30 wrapper 对比

Issue #30 wrapper 只做 codebook 几何变换 (transforms 一次性 apply 到 .embeddings.weight). Issue #31 wrapper 涉及 loss computation 改造, 更 invasive. 但都是 wrapper pattern, 不修改 `HG-Rec/model/`.

---

## 6. R10 + R11 audit

- **R9**: descriptions/ max=301 → next=302 ✅
- **R10**: Issue #30 Gate 3 训练中 (ETA ~137 min). Issue #31 Gate 0 wrapper 设计是合理的并行 backlog 推进 (CPU-only 工作, 不抢 GPU).
- **R11.5**: owner feedback 2026-07-29 23:13 「不允许假设 owner 有 decision」→ 自主决策启动 Issue #31 Gate 0.
- **R11.4**: wrapper 不修改 HG-Rec/model/ 上游源码, patch 在 __init__ 之后做 (R11.5 critical decision).
- **R7**: Issue #31 Gate 0 跑 reg test (CPU only) 不抢 GPU 0/1/2 (Issue #30 Gate 3 用 GPU 3).

---

## 7. 关联

- [[issue31-body]]: Issue #31 完整 body (per-layer β_l + α_l + γ_l + c_k range + 4-Gate 协议)
- [[task301-issue30-gate1-result]]: Issue #30 Gate 1 PASS (per-layer transforms 突破 Phase 0 mode collapse 路线)
- [[task300-issue29-gate1-result]]: Issue #29 Gate 1 FAIL (K_l 单一变量不能突破 Phase 0)
- [[task298-issue26-conflict-report]]: task298 §4 5 个架构层候选, Issue #31 是新第 6 候选 (encoder regularization)
- [[phase0-mode-collapse]]: 6 任务 Phase 0 mode collapse 根因, Issue #31 直接处理
- [[issue28-closure]]: Issue #28 closure owner-verdict 锁定根因 = encoder-side boundary saturation
- [[cross-task-c-k-range-no-go-exhausted]]: c_k range 路径跨 8 方向 NO-GO 收口, Issue #31 加 encoder regularization 是新维度

---

result: Task #302 / Issue #31 Gate 0 wrapper 设计阶段. 目标: 写 per-layer β_l + α_l + γ_l + c_k range wrapper 不修改 HG-Rec/model/, reg test 跟 baseline 等价.