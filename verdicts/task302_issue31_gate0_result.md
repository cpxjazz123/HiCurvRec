# Task #302 / Issue #31 — Gate 0 PASS

**日期**: 2026-07-30
**状态**: ✅ **Gate 0 PASS — EncoderRegHRQVAE wrapper 跟 baseline reg test 等价 + 三组正则项都生效**
**决定**: 进入 Gate 1 决策点 (等 Issue #30 Gate 3 R@10 实证 + R11.5 决策是否启动 Gate 1 Stage 1)

---

## 1. Gate 0 通过条件 (Issue #31 body §Gate 0)

| 条件 | 实测 | 决策 |
|------|------|------|
| wrapper 文件在 `scripts/` 下提交 | `scripts/task302_issue31_gate0_wrapper.py` (EncoderRegHRQVAE 继承 HRQVAE) | ✅ |
| β_l=[0.5,0.5,0.5] + α_l=[0,0,0] + γ_l=[0,0,0] → baseline 等价 | out_max_diff=0, rq_loss_max_diff=0, idx_equal=True | ✅ PASS |
| β_l=[0.1,0.3,0.5] → 仅 commit loss 数值改变, indices 一致 | out_max_diff=0, rq_loss diff=4.57e-3 (expected), idx_equal=True | ✅ PASS |
| α_l=[0.01,0.005,0.001] → anchor 正则项非零 | anchor_loss = 0.001661 | ✅ PASS |
| γ_l=[0.001,0.0005,0.0001] → encoder L2 正则生效 | rq_loss 增量 = +0.005406 | ✅ PASS |
| ‖x‖_E 打印逻辑与 baseline 一致 | 复用 baseline encoder, 输出 norm 一致 (1.902074) | ✅ PASS |

**Gate 0 全部通过** → Gate 0 PASS.

---

## 2. Wrapper 设计 (R11.4 + R11.5 critical decision)

### 2.1 不修改 HG-Rec/model/ 上游源码

R11.4 禁止修改上游源码. wrapper 通过 Python 子类化实现:
- `EncoderRegHRQVAE(HRQVAE)` 继承 baseline
- patch per-layer β: `q.beta = beta_list[li]` (在 __init__ 后做, 不修改基类)
- 重写 forward 累加 anchor_loss + enc_reg_loss
- 不修改 `HG-Rec/model/hrqvae.py` / `HG-Rec/model/utils.py` / `HG-Rec/train_hrqvae.py`

### 2.2 Per-layer β_l patch

baseline `HResidualVectorQuantization.__init__` 传 `self.beta` 给每个 HVectorQuantization layer:
```python
self.vq_layers = nn.ModuleList([HVectorQuantization(n_e, e_dim, beta=self.beta, ...) for ...])
```

wrapper 不修改 `__init__`, 而是在构造完成后 patch:
```python
for li, q in enumerate(self.hrq.vq_layers):
    q.beta = float(beta_list[li])
```

这样 β_l per-layer 异构, 不修改基类. R11.4 保护.

### 2.3 Forward 重写 (anchor_loss + enc_reg_loss)

```python
def forward(self, x, use_sk=True, rho_target_batch=None):
    x_enc = self.encoder(x)
    all_losses, all_indices = [], []
    x_q = 0.0
    residual = x_enc
    enc_reg_loss = x_enc.new_zeros(())
    for li, q in enumerate(self.hrq.vq_layers):
        if self.gamma_list[li] != 0.0:
            enc_reg_loss = enc_reg_loss + self.gamma_list[li] * (residual ** 2).sum()
        x_res, loss, indices = q(residual, use_sk=use_sk)
        residual = residual - x_res
        x_q = x_q + x_res
        all_losses.append(loss)
        all_indices.append(indices)
    rq_loss = torch.stack(all_losses).mean()
    indices = torch.stack(all_indices, dim=-1)
    out = self.decoder(x_q)

    anchor_loss = x_enc.new_zeros(())
    for li, q in enumerate(self.hrq.vq_layers):
        if self.alpha_list[li] != 0.0:
            anchor_loss = anchor_loss + self.alpha_list[li] * (q.embeddings.weight ** 2).sum()

    total_quant = rq_loss + enc_reg_loss + anchor_loss
    return out, total_quant, indices, None, (None, None, anchor_loss, None)
```

### 2.4 reg test 等价性 (R11.5 验证)

**核心 reg test (Issue #31 §Gate 0 通过条件)**: β_l=[0.5,0.5,0.5] + α_l=[0,0,0] + γ_l=[0,0,0] → forward 输出 baseline 等价.

- **out_max_diff = 0** ✅ (forward 输出完全一致)
- **rq_loss_max_diff = 0** ✅ (loss 数值完全一致)
- **idx_max_diff = 0** ✅ (argmin indices 一致)

wrapper 满足 Gate 0 通过条件. **通过**.

### 2.5 反预期发现 (R11.5 备注)

**1. β 异构不影响 argmin 路径 (Test 2)**:
- β 改变 commit loss 数值 (loss 用作梯度更新)
- 但 β **不**影响 VQ 距离公式 (distance 跟 codebook embedding 直接相关, 跟 β 无关)
- 因此 argmin indices 一致 → out 一致 ✅

**2. enc_reg_loss 用 sum 而非 mean**:
- 期望 (residual ** 2).sum() 是所有元素的 L2 norm 平方和
- 这是 per-batch 全元素的 L2 norm
- α_l/γ_l 在 Issue #31 body 没指定 sum vs mean, 选用 sum (跟 baseline commit loss 的 torch.mean 略不同, 但 α_l/γ_l 数值已包含这个 normalization 选择)

**3. α_l anchor 跟 codebook L2 norm 关联**:
- baseline codebook 初始化为 uniform(-0.01, 0.01), 范数很小 (~0.26)
- α_l=[0.01,0.005,0.001] × sum(0.01² × 64 × 32) ≈ 0.001661 (实测)
- 训练时 α_l 会把 codebook norm 推到目标范围, 防止 boundary collapse

---

## 3. Gate 1 推进计划 (R11.5)

per Issue #31 body §Gate 1:
1. **Stage 1 100 epoch 端到端训练** (绕开 task297 warm-start bug)
2. lr = baseline recipe 默认 (task84)
3. β_l = [0.1, 0.3, 0.5], α_l = [0.01, 0.005, 0.001], γ_l = [0.001, 0.0005, 0.0001]
4. c_k_range = task242 Arm A (per-layer 异构 metric)

**GPU 决策 (R7 + R11.5)**:
- GPU 3 被 Issue #30 Gate 3 T5-mini 训练占用 (PID 550667, ETA ~137 min)
- GPU 0/1/2 全部空闲
- Issue #31 Gate 1 Stage 1 100 epoch 训练可以用 GPU 0/1 (R7 不抢卡)
- **不立即启动**: 等 Issue #30 Gate 3 R@10 实证 (GO/NO-GO) + Issue #31 Gate 1 训练成本评估 (R11.5 自主决策)

**通过条件 (a-e 五条同时)**:
- (a) L0 utilization ≥ 90% at any eval step ≥ ep50
- (b) L1 utilization ≥ 90% at any eval step ≥ ep50
- (c) L2 utilization ≥ 90% at any eval step ≥ ep50
- (d) collision_rate ≤ 0.20
- (e) ‖x‖_E ≥ 0.3 at any eval step ≥ ep50

**硬停止**: 任一不满足 → STOP, 关闭 Issue #31.

---

## 4. Gate 0 物理产物

- `descriptions/task302_issue31_per_layer_encoder_reg.md` (Issue #31 任务定义)
- `scripts/task302_issue31_gate0_wrapper.py` (Gate 0 wrapper + 4 test reg suite)
- `verdicts/task302_issue31_gate0_result.md` (本文件, PASS)
- Issue #31 GitHub status: OPEN (等 Gate 1 决策)

---

## 5. R10 + R11 audit

- **R9**: descriptions/ max=302 ✅ (Issue #31 = task302, 连续无空洞)
- **R10**: Issue #31 Gate 0 PASS 是 backlog 真空期间的积极信号. Issue #30 Gate 3 训练中 (ETA ~137 min). Issue #31 Gate 1 Stage 1 候选待启动.
- **R11.5**: owner feedback 2026-07-29 23:13 「不允许假设 owner 有 decision」→ 自主决策启动 Issue #31 Gate 0 (跟 Issue #28-#30 同期, R10 backlog 真空 + 4-Gate protocol).
- **R11.4**: wrapper 不修改 HG-Rec/model/ 上游源码, patch 在 __init__ 之后做 (R11.5 critical decision 验证).
- **R7**: Issue #31 Gate 0 reg test 是 CPU-only (不抢 GPU 0/1/2/3).
- **R2**: 禁止 fallback. reg test 失败 → 实测找到 3 个 root cause (eval mode + init_uniform + kmeans_init=False) 全部修复后才 PASS, 没有用 fallback 默认值掩盖.

---

## 6. 关联

- [[issue31-body]]: Issue #31 完整 body (per-layer β_l + α_l + γ_l + c_k range + 4-Gate 协议)
- [[task301-issue30-gate1-result]]: Issue #30 Gate 1 PASS (per-layer transforms 突破 Phase 0 mode collapse)
- [[task300-issue29-gate1-result]]: Issue #29 Gate 1 FAIL (K_l 单一变量不能突破 Phase 0)
- [[task298-issue26-conflict-report]]: task298 §4 5 个架构层候选, Issue #31 是新第 6 候选 (encoder regularization)
- [[phase0-mode-collapse]]: 6 任务 Phase 0 mode collapse 根因, Issue #31 直接处理 encoder-side boundary saturation
- [[issue28-closure]]: Issue #28 closure owner-verdict 锁定根因 = encoder-side boundary saturation
- [[cross-task-c-k-range-no-go-exhausted]]: c_k range 路径跨 8 方向 NO-GO 收口, Issue #31 加 encoder regularization 是新维度

---

result: Task #302 / Issue #31 Gate 0 PASS. EncoderRegHRQVAE wrapper reg test 全 4 条通过 (β_l baseline 等价 + β 异构 indices 一致 + α anchor 非零 + γ encoder L2 增量). Gate 1 Stage 1 100 epoch 训练待 R11.5 决策启动 (GPU 0/1 空闲, Issue #30 Gate 3 占用 GPU 3).