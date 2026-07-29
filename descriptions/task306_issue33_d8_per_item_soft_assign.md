# Task #306 / Issue #33 / D8 — per-item 软分配 on #30 GO 配置 (Codebook Transforms 赋值规则维度)

**日期**: 2026-07-30
**状态**: ⏳ 未启动 (Gate 0 调研中)
**承接**: Issue #33 (owner 2026-07-29 #32 closure comment 锁定 D8 verdict, R11.2(1) 最高优先级)
**关联**: [[task301-issue30-per-layer-transforms]] [[task303-issue32-dual-axis-synergy]] [[task304-d6-r-l-s-l-ablation]]

---

## 1. 任务定义

**目的**: 在 **Issue #30 唯一 GO 配置 (r_l=[0.1, 1.0, 10.0] + s_l=[2, 2, 2] 极端值, R@10=0.1022)** 上, **替换 argmin hard-assign 为 per-item 软分配** —— 验证 per-item 个性化赋值能否在 #30 GO 端点上贡献 R@10 增益.

**触发条件**:
- Issue #32 closure comment (2026-07-29 18:01 owner-verdict 5-Gate NO-GO + D6 ablation + **D7/D8/D9 候选列表**)
- D8 = per-item soft-assign (owner 明确指定)
- owner feedback 2026-07-29 23:13 「不允许假设 owner 有 decision. 每次 loop. AI 必须自行决策做出可以推进目的的决定」
- 20 方向 × 21 verdict 全 NO-GO 收口 (含 #28 #29 #30 #31 #32 + task298/299/300/301/302/303/304 D6 ablation)

**否决假设**:
- ❌ 不重复 FSQ/EMA/Restoration (task290/291/292 NO-GO)
- ❌ 不走 Gumbel-Softmax 全局扰动路径 (#28 NO-GO)
- ❌ 不走 K_l 变化 (#29 NO-GO)
- ❌ 不走 encoder regularization (#31 NO-GO)
- ❌ 不走 c_k_range 双轴 + 温和 r_l + s_l (#32 catastrophic NO-GO -99.88%)
- ❌ 不做 r_l alone / s_l alone ablation (task304 D6 NO-GO)
- ❌ 不申请 multi-seed verification (D7 禁试, owner #32 closure 明示)
- ❌ 不修改 HG-Rec/model/ 上游源码 (R11.4 critical decision)
- ❌ 不跨 Gate 取数 (任一 Gate 失败即在该 Gate 处写 verdict 结束)

---

## 2. 5-Gate 硬停止协议 (按 Issue #33 body 严格执行)

### Gate 0 —— 代码实现 + 双回归测试 (零 GPU, ~30s)

**目标**: 在不修改 HG-Rec/model/ 上游源码前提下, 实现 per-item 软分配.

**关键设计决策 (R11.5 自主, owner #33 body 没指定实现细节)**:
- 方案 A: monkey-patch HVectorQuantization.forward, 把 argmin hard-assign 改成 per-item soft-assign (用 softmax temperature + 每个 item 一个 temperature 参数)
- 方案 B: 新增独立 wrapper class `PerItemSoftVQ`, 继承 baseline HVectorQuantization, 重写 forward (forward 末尾替换 lookup)
- 方案 C: 保持 VQ 不变, 只在 commit loss 阶段引入 per-item 软分布 (Stage 1 forward 仍用 argmin, commit loss 用 soft target)

**首选方案 B** (R11.2 兜底顺序 1: 上游 default, 2: 简单实用, 3: 论文原始方案):
- 不动 upstream HVectorQuantization
- 创建 `scripts/task306_issue33_gate0_peritem_soft_vq.py` 实现 PerItemSoftVQ
- 双回归测试:
  1. **退化到 baseline**: temperature → 0 (sharp softmax) 应逼近 argmin (max diff ≤ 1e-3)
  2. **复现 #30 端点**: 用 #30 r_l + s_l 极端值 + 软分配 temperature 极低, R@10 Stage 4 应 = 0.1022 ± 0.005
- norm 健康区: ‖x‖_E ∈ [0.7, 0.95]

**硬停止**: 双回归测试任一 FAIL → STOP, 不进入 Gate 1.

### Gate 1 —— Stage 1 100 epoch 训练 (GPU)

- 配置: per-layer r_l=[0.1, 1, 10] + s_l=[2, 2, 2] + **per-item temperature** (R11.3 自主默认: τ=1.0, 单一温度)
- 端到端 Stage 1 训练 (per-item 软分配在 forward 中替换 argmin)
- 评估: L0/L1/L2 utilization ≥ 90% + collision_rate ≤ 0.20 + norm 健康区 ‖x‖_E ∈ [0.7, 0.95]

**硬停止**: 任一不满足 → STOP.

### Gate 2 —— Sinkhorn 5 iter 推断 (cheap)

- 用 Gate 1 best ckpt 跑 Sinkhorn max_iters=5
- 4-digit SID unique count ≥ 9500 + per-layer util 偏差 ≤ 5pp

**硬停止**: unique < 9500 → STOP.

### Gate 3 —— Stage 3 T5-mini 200 epoch 训练 (GPU)

- 沿用 task301 Stage 3 recipe
- early_stop=20
- Stage 3 输入仍是 4-digit SID (argmin 输出), commit loss 在 Stage 1 已按 per-item 软分配计算

**硬停止**: NaN / 早期 early_stop → STOP.

### Gate 4 —— Stage 4 Test eval (GPU)

- Test R@10 > 0.1022 (**双重对照**: HG-Rec baseline 0.1020 + #30 端点 0.1022)
- 不允许"再调一下 per-item 软分配温度"或"再调一下 r_l" 为由重跑 Gate 0-3

**硬停止**: R@10 ≤ 0.1022 → STOP, 写 verdict NO-GO, 关闭本 issue.

---

## 3. 关键决策点 (R11.3 自主决策)

| # | 决策 | 选择 | 备选 | 理由 |
|---|------|------|------|------|
| 1 | per-item 软分配实现位置 | 方案 B: 独立 PerItemSoftVQ wrapper class | A: monkey-patch forward / C: 只改 commit loss | R11.4 critical decision, 不动 upstream HVectorQuantization |
| 2 | per-item temperature | τ=1.0 (softmax 平滑) | τ=0.5 (sharp) / τ=2.0 (smooth) | 单一温度, 默认中性, 后续可调 |
| 3 | commit loss 处理 | 软分配输出 + argmax 替代码字 (straight-through estimator) | KL divergence to soft target | 简单 + 跟 baseline codebook 优化路径兼容 |
| 4 | 是否保留 dead_revive | ❌ 不保留 | 保留 task242 A+ | task283 已证实在 L0 健康 ckpt 上是 no-op, 多余 |
| 5 | r_l / s_l 是否微调 | ❌ 保留 #30 极端值 [0.1,1,10] + [2,2,2] | 试 [0.5,1,5] / [3,3,3] | Issue #33 body 明示「保留 #30 r_l + s_l 极端值」 |
| 6 | GPU 选择 | GPU 0/1/2/3 中 R7 空闲任意一张 | 强制 GPU 0 | R7 不抢卡, 选空闲卡 |
| 7 | seed | 42 (单 seed, R11.5 禁 multi-seed) | multi-seed | owner 2026-07-29 反馈 + memory 记录 |

---

## 4. 反证 / 压力测试 (H1/H2/H3, Issue #33 body 列出)

- **H1 反证**: per-item 软分配在 #30 GO 端点不贡献 R@10 增益. 若 Gate 4 R@10 ≤ 0.1022, 承认本机制 NO-GO, 关闭 issue, 不申请后续预算.
- **H2 反证**: per-item 软分配重蹈 #28 失败根因 (commit loss 跨 per-item 个性化主导梯度). Gate 1 (a)(b)(c) 利用率 + norm 健康区是该反证的硬停止闸门.
- **H3 反证**: per-item 软分配破坏 norm 健康区 (引入额外 norm drift). Gate 1 (e) ‖x‖_E ∈ [0.7, 0.95] 是该反证的硬停止闸门.

---

## 5. 物理产物 (待启动)

- `descriptions/task306_issue33_d8_per_item_soft_assign.md` (本文件, ✅)
- `scripts/task306_issue33_gate0_peritem_soft_vq.py` (Gate 0 软分配 VQ wrapper + 双回归测试)
- `scripts/task306_issue33_gate1_stage1_train.sh` (Gate 1 Stage 1 100 epoch launcher)
- `scripts/task306_issue33_gate2_sinkhorn_inference.py` (Gate 2 Sinkhorn 5 iter)
- `scripts/task306_issue33_gate3_stage3_train.sh` (Gate 3 Stage 3 T5-mini 200 epoch)
- `scripts/task306_issue33_gate4_stage4_eval.sh` (Gate 4 Stage 4 R@10 评估)
- `verdicts/task306_issue33_gate0_*.md` + `task306_issue33_gate1_*.md` + ... (各 Gate 独立 verdict)

---

## 6. 与 D7 / D9 的关系

- **D7 (multi-seed verification)**: ❌ **禁试** (owner #32 closure 明示, memory `[[user-no-multiseed-override]]` 适用)
- **D8 (per-item soft-assign)**: ✅ **本 issue 处理** (owner verdict, R11.2(1) 最高优先级)
- **D9 (多样 hash)**: 后续 issue 候选 (本 issue 不处理)

---

## 7. R10 + R11 + R12 audit

- **R7 GPU 不抢卡**: Gate 1/3/4 启动前必跑 `nvidia-smi`, 选空闲卡.
- **R9 编号连续**: max+1 = 306 ✅ (R9 layer 2 audit pass).
- **R10 主动推進**: Issue #33 OPEN + R11 backlog 真空, 立即启动 D8.
- **R11.2 owner preference 最高优先级**: owner #32 closure verdict 锁定 D8.
- **R11.5 自主决策**: 关键决策点 1-7 (温度 / commit loss / GPU 选择 / seed 等) 全部自主决策, 不抛回 owner.
- **R12 ckpt 强制保存**: Gate 1/3 训练结束 → torch.save best_ckpt, 删旧.
- **R13 禁止 Worktree**: 在共享 checkout 直接修改.
- **R14 Issue 自动监控**: Issue #33 OPEN, 本 task 完成后 close.

---

result: Task #306 / Issue #33 / D8 per-item 软分配 on #30 GO 配置. **5-Gate 硬停止协议**, Gate 0 = 代码实现 + 双回归测试 (退化到 baseline + 复现 #30 R@10=0.1022). Gate 4 通过条件 = R@10 > 0.1022 (HG-Rec baseline 0.1020 + #30 端点 0.1022 双重对照). **D7 (multi-seed) 禁试, D9 (多样 hash) 后续候选**. 立即启动 Gate 0 代码实现 (PerItemSoftVQ wrapper class, 不动 HG-Rec/model/ 上游).
