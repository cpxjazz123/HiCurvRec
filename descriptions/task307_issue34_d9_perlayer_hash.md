# Task #307 / Issue #34 / D9 — per-layer 异构 hash 函数族 on #30 GO 配置

**日期**: 2026-07-30
**状态**: ⏳ Gate 0 启动中
**承接**: Issue #34 (D9 多样 hash, owner #26 R3 暗示 + #32 R11.5 / #33 R11.5 综合判决 D9 = 唯一剩余方向)
**关联**: [[task301-issue30-per-layer-transforms]] [[task303-issue32-dual-axis-synergy]] [[task304-d6-r-l-s-l-ablation]] [[task305-issue33-commit-loss]] [[task306-issue33-d8-forward]]

---

## 1. 任务定义

**目的**: 在 **Issue #30 唯一 GO 配置 (r_l=[0.1, 1.0, 10.0] + s_l=[2, 2, 2] + hard argmin commitment, R@10=0.1022)** 上, **新增 per-layer 异构 hash 函数族维度** —— 每层用不同 hash 函数族将码字映射到多个候选 SID slots (L0 top-3 / L1 top-5 / L2 top-7), 验证能否在 #30 GO 端点上贡献 R@10 增益 (>0.1022).

**触发条件**:
- owner #26 R3 暗示: "后续必须在架构层 (Gumbel-Softmax / 多样 hash / per-item soft-assign)"
- owner #32 R11.5: D7/D8/D9 候选清单, D9 多样 hash 是 owner 锁定候选路径
- owner #33 R11.5: expected-loss 形式禁试, D9 = 多样 hash 是 owner 暗示唯一剩余方向
- 24 方向 × 24 verdict NO-GO 收口 (含 #28 Gumbel-Softmax / #29 K_l / #31 encoder reg / #32 c_k range / #33 per-item soft 2 变体)
- owner feedback 2026-07-29 23:13 「不允许假设 owner 有 decision. 每次 loop. AI 必须自行决策做出可以推进目的的决定」

**否决假设** (per Issue #34 body §反证 / 压力测试):
- ❌ 不重复 FSQ/EMA/Restoration (task290/291/292 NO-GO)
- ❌ 不走 Gumbel-Softmax (#28 NO-GO)
- ❌ 不走 K_l 变化 (#29 NO-GO)
- ❌ 不走 encoder regularization (#31 NO-GO)
- ❌ 不走 c_k_range 双轴 + 温和 r_l + s_l (#32 灾难 NO-GO -99.88%)
- ❌ 不走 per-item soft-assign (#33 task305 + task306 跨 2 变体 NO-GO)
- ❌ 不申请 multi-seed verification (D7 禁试, owner #32 closure 明示)
- ❌ 不修改 HG-Rec/model/ 上游源码 (R11.4 critical decision)
- ❌ 不跨 Gate 取数 (任一 Gate 失败即在该 Gate 处写 verdict 结束)

---

## 2. 5-Gate 硬停止协议 (按 Issue #34 body 严格执行)

### Gate 0 —— 代码实现 + 双回归测试 (零 GPU, ~30-60s)

**目标**: 实现 per-layer 异构 hash 函数族训练代码, 不修改 HG-Rec/model/ 上游源码.

**关键设计决策 (R11.5 自主, owner #34 body 没指定实现细节)**:
- 方案 A: monkey-patch HVectorQuantization.forward, 加 per-layer hash 后处理
- 方案 B: 新增独立 wrapper class `PerLayerHashHRQVAE`, 继承 baseline HRQVAE, 在 forward 后注入 hash
- **首选方案 B** (R11.2 兜底顺序 1: 不动 upstream, 2: 简单实用)
  - 不动 upstream HVectorQuantization
  - 创建 `scripts/task307_issue34_gate0_perlayer_hash.py` 实现 PerLayerHashHRQVAE
  - per-layer hash 函数族:
    - L0 (K=64): sparse random projection hash + binary collision check → top-3 candidates
    - L1 (K=128): LSH multi-probe on residual → top-5 candidates
    - L2 (K=256): k-means bucket hash + multi-bucket assignment → top-7 candidates

**双回归测试**:
1. **R1 退化到 baseline**: per-layer hash OFF + 恒等 r_l/s_l → 输出与 baseline 一致 (max diff < 1e-3)
2. **R2 复现 #30 端点**: per-layer hash OFF + r_l=[0.1,1,10] + s_l=[2,2,2] → 输出与 #30 task301 端点一致 (max diff < 1e-3)

**Integration 验证**:
- PerLayerHashHRQVAE.forward 返回 trainer 兼容 5-tuple (out, idx, loss, path_loss, div_ent)
- per-layer hash candidates 数量 (L0: 3, L1: 5, L2: 7) 实际生效

**硬停止**: 双回归测试任一 FAIL → STOP, 不进入 Gate 1.

### Gate 1 —— Stage 1 100 epoch 训练 (GPU)

- **配置**: per-layer r_l=[0.1, 1, 10] + s_l=[2, 2, 2] + per-layer hash 开启 (L0 top-3 / L1 top-5 / L2 top-7) + hard argmin commitment
- **执行**: 端到端 Stage 1 训练 100 epoch

**通过条件 (六条同时满足)**:
- (a) L0 utilization ≥ 90% at any evaluation step ≥ ep50
- (b) L1 utilization ≥ 90% at any evaluation step ≥ ep50
- (c) L2 utilization ≥ 90% at any evaluation step ≥ ep50
- (d) collision_rate ≤ 0.25 (per-layer hash 多候选 SID 可能略增 collision, 放宽到 0.25 vs #30 端点 ≤ 0.20)
- (e) **norm 健康区 ‖x‖_E ∈ [0.7, 0.95]** for L0/L1/L2 at any evaluation step ≥ ep50 (新增 H3 反证闸门)
- (f) **hash 函数族有效性**: L0 top-3 candidates ≥ 3 个不同 SID, L1 top-5 ≥ 5 个不同 SID, L2 top-7 ≥ 7 个不同 SID (新增 H4 反证闸门)

**硬停止**: 任一不满足 → STOP. 不得为补 Gate 1 通过而进入 Stage 2/3/4.

### Gate 2 —— Sinkhorn 5 iter 推断 (cheap)

- 用 Gate 1 best ckpt 跑 Stage 2 Sinkhorn 推断, max_iters=5
- Sinkhorn 推断时仍用 hard argmin commitment (与 #30 同)
- argmin 后应用 per-layer 异构 hash 给每层多个候选 SID slot

**通过条件**:
- 4-digit SID unique count ≥ 9500 (majority SID per item)
- per-layer utilization 与 Gate 1 终态偏差 ≤ 5pp
- per-layer hash candidates 平均数 ≥ L0 top-3 + L1 top-5 + L2 top-7

**硬停止**: 任何条件不满足 → STOP.

### Gate 3 —— Stage 3 T5-mini 200 epoch 训练 (GPU)

- 按 #30 task301 Stage 3 recipe + 200 epoch + early_stop=20
- **Stage 3 输入** = 每个 item 的 per-layer 多候选 SID slot (而非 #30 单 SID)
- commitment 公式 hard argmin (与 #30 一致)

**通过条件**: Stage 3 训练稳定 (no NaN, no early_stop before ep100, final ckpt loss 收敛).

**硬停止**: NaN / 早期 early_stop → STOP.

### Gate 4 —— Stage 4 Test eval (GPU)

- 按 task278 batch Stage 4 eval recipe + test eval R@10 / R@5 / R@20 / NDCG

**通过条件**:
- **Test R@10 > 0.1022** (HG-Rec baseline #84 0.1020 + #30 端点 0.1022 双重对照)

**硬停止**: R@10 ≤ 0.1022 → STOP, 写 verdict NO-GO, 关闭本 issue. **不得以"再调一下 hash 函数族"或"再调一下候选 slot 数"为由重跑 Gate 0-3**.

---

## 3. 关键决策点 (R11.3 自主决策)

| # | 决策 | 选择 | 备选 | 理由 |
|---|------|------|------|------|
| 1 | per-layer hash 实现位置 | 方案 B: 独立 PerLayerHashHRQVAE wrapper class | A: monkey-patch forward | R11.4 critical decision, 不动 upstream HVectorQuantization |
| 2 | per-layer hash 函数族 | L0 sparse random projection / L1 LSH / L2 k-means multi-bucket | 单一 hash (e.g., LSH for all layers) | issue #34 body §3 自创设计, per-layer 异构才能贡献 R@10 增益 |
| 3 | per-layer 候选 slot 数 | L0=3 / L1=5 / L2=7 | L0=5 / L1=5 / L2=5 (对称) | issue #34 body §3 显式指定, 跟码字几何层数正相关 |
| 4 | commitment 公式 | hard argmin (与 #30 一致) | per-item soft / sinkhorn-soft | owner #33 closure 禁试 expected-loss 形式 |
| 5 | r_l / s_l 配置 | 保留 #30 极端值 [0.1,1,10] + [2,2,2] | 温和 r_l/s_l | #32 灾难 NO-GO 实证温和值毁坏 #30 杠杆 |
| 6 | GPU 选择 | GPU 0/1/2/3 中 R7 空闲任意一张 | 强制 GPU 0 | R7 不抢卡, 选空闲卡 |
| 7 | seed | 42 (单 seed, R11.5 禁 multi-seed) | multi-seed | owner 2026-07-29 反馈 + memory 记录 |

---

## 4. 反证 / 压力测试 (H1/H2/H3/H4, Issue #34 body 列出)

- **H1 反证**: per-layer 异构 hash 在 #30 GO 端点不贡献 R@10 增益. 若 Gate 4 R@10 ≤ 0.1022, 承认本机制 NO-GO, 关闭 issue, 不申请后续预算.
- **H2 反证**: per-layer 异构 hash 重蹈 #33 collapse. 保留 hard argmin commitment, 任何 hash candidates 选择**必须**用 top-k argmin 而非 soft 概率. Gate 0 (f) hash candidates 有效性是硬停止闸门.
- **H3 反证**: per-layer 异构 hash 引入 norm drift 毁坏 norm 健康区. Gate 1 (e) ‖x‖_E ∈ [0.7, 0.95] 是硬停止闸门.
- **H4 反证**: per-layer 异构 hash 数学上不兼容 #30 端点. Gate 0 双回归测试 + Gate 1 Stage 1 训练稳定是硬停止闸门.

---

## 5. 物理产物 (待启动, 部分产物继承 #30)

- `descriptions/task307_issue34_d9_perlayer_hash.md` (本文件, ✅)
- `scripts/task307_issue34_gate0_perlayer_hash.py` (Gate 0 wrapper + 双回归 + Integration)
- `scripts/task307_issue34_gate1_stage1_train.sh` (Gate 1 Stage 1 100 epoch launcher)
- `scripts/task307_issue34_gate2_sinkhorn_inference.py` (Gate 2 Sinkhorn 5 iter)
- `scripts/task307_issue34_gate3_stage3_train.sh` (Gate 3 Stage 3 T5-mini 200 epoch)
- `scripts/task307_issue34_gate4_stage4_eval.sh` (Gate 4 Stage 4 R@10 评估)
- `verdicts/task307_issue34_gate0_*.md` + `task307_issue34_gate1_*.md` + ... (各 Gate 独立 verdict)

---

## 6. 与 D7 / D8 / D9 的关系

- **D7 (multi-seed verification)**: ❌ **禁试** (owner #32 closure 明示, memory `[[user-no-multiseed-override]]` 适用)
- **D8 (per-item soft-assign)**: ❌ **task305 + task306 跨 2 变体 NO-GO 关闭** (codebook collapse)
- **D9 (多样 hash)**: ✅ **本 issue 处理** (owner verdict, R11.2(1) 最高优先级)

---

## 7. R10 + R11 + R12 audit

- **R7 GPU 不抢卡**: Gate 1/3/4 启动前必跑 `nvidia-smi`, 选空闲卡.
- **R9 编号连续**: max+1 = 307 ✅ (R9 layer 2 audit pass).
- **R10 主动推進**: Issue #34 OPEN + R11 backlog 真空, 立即启动 D9.
- **R11.2 owner preference 最高优先级**: owner #26 R3 + #32 R11.5 + #33 R11.5 综合锁定 D9.
- **R11.5 自主决策**: 关键决策点 1-7 (实现位置 / hash 函数族 / 候选 slot 数 / commitment 公式 / GPU 选择 / seed 等) 全部自主决策, 不抛回 owner.
- **R12 ckpt 强制保存**: Gate 1/3 训练结束 → torch.save best_ckpt (save_limit=1 删旧).
- **R13 禁止 Worktree**: 在共享 checkout 直接修改.
- **R14 Issue 自动监控**: Issue #34 OPEN, 本 task 完成后 close.

---

result: Task #307 / Issue #34 / D9 per-layer 异构 hash 函数族 on #30 GO 配置. **5-Gate 硬停止协议**, Gate 0 = 代码实现 + 双回归测试 (退化到 baseline + 复现 #30 端点 R@10=0.1022). Gate 4 通过条件 = R@10 > 0.1022 (HG-Rec baseline 0.1020 + #30 端点 0.1022 双重对照). **D7 (multi-seed) 禁试, D8 (per-item soft) task305/#306 NO-GO 关闭, D9 (多样 hash) 本 issue 处理**. 立即启动 Gate 0 代码实现 (PerLayerHashHRQVAE wrapper class, 不动 HG-Rec/model/ 上游).
