# Task #332 — Issue #41 Gate 1 — Stage 1 双曲空间曲率 κ 3-arm ablation (设计 only)

**日期**: 2026-07-30 14:30
**触发**: Issue #41 Gate 0 全 PASS (verdicts/task331_issue41_gate0_h_mds_input_space.md), 用户 2026-07-30 14:25 决策 "只做曲率相关任务"
**状态**: 🟡 **设计 only, 待 Issue #40 Gate 1 验证后启动** (R11.5 自主决策)
**R10**: §16 backlog 真空, 唯一曲率方向候选

---

## 1. Issue #41 Gate 0 实证结论 (直接引用)

| 测量方法 | 数据空间 | 最佳 κ | 解读 |
|----------|----------|--------|------|
| **Ollivier** (Task #70) | 输入 graph | -0.65 ~ -0.84 | 强双曲 |
| **类目树组合** (H2) | 类目树结构 | -0.739 | 强双曲 |
| **输入空间 h-MDS** (a) | 768d embedding | **-2.00** | 强双曲 (boundary) |
| **Residual 控制** (b) | 32d residual | 0.00 | 反证: 编码器压平 |
| **HG-Rec baseline** (Task #84) | 默认 | **1.0** | κ 未指定 = default 1.0 |

**核心洞察**: 三个独立测量 (Ollivier + 类目树 + h-MDS) 一致确认 Musical_Instruments 输入空间强双曲 (κ < 0), 但 baseline Stage 1 用默认 κ=1.0 (正向球面), **κ 设置跟数据真实需求方向相反**!

---

## 2. Gate 1 设计: Stage 1 曲率 κ 3-arm ablation

### 2.1 核心假设

**H1 (主假设)**: Stage 1 用真实数据双曲曲率 κ<0 训练, 比默认 κ=1.0 baseline 产生更可保留双曲结构的 SID → 提升 Stage 4 R@10.

### 2.2 3-arm 设计

| Arm | κ 设置 | 来源 | 预测 |
|-----|---------|------|------|
| **A** (control) | κ=1.0 (default) | Task #84 baseline | baseline 0.1020 |
| **B** (Ollivier) | κ=-0.726 (per-layer) | Task #70 mean | 跟 Ollivier 数据需求对齐 |
| **C** (h-MDS) | κ=-2.0 (per-layer) | Issue #41 Gate 0 (a) | 跟 h-MDS 拟合对齐 |

**注**: 
- 之前 task287/task284 κ-decouple 失败, 是因为 κ frozen=0 (而非真实数据曲率). 本设计用**真实数据曲率**而不是 0
- "per-layer" 意思是 Stage 1 每层都用同一个 κ (3 arm κ 是 global, 非 per-layer)

### 2.3 通过条件

| 实测 R@10 | 解读 | 决策 |
|-----------|------|------|
| Arm B/C R@10 > baseline 0.1020 (+2% 以上) | H1 CONFIRMED, 真实数据曲率是 R@10 杠杆 | ⭐⭐⭐ GO |
| Arm B/C R@10 ≈ baseline 0.1020 (±2%) | H1 中性, 曲率对 R@10 无显著影响 | NEUTRAL |
| Arm B/C R@10 < baseline (-2% 以上) | H1 REFUTED, 真实数据曲率反而恶化 | NO-GO |

### 2.4 协议控制 (避免重蹈 task194 协议 leak)

跟 task194 Issue #40 协议审计结论对齐:
- **batch_size = 1024** (跟 #84 baseline 一致, 不是 task194 的 256)
- **epochs = 1000** (跟 #84 baseline 一致, 不是 task194 的 500)
- **Stage 2 Sinkhorn**: 默认 args.sk_epsilons (=0.0, argmin only)
- **Stage 3 beam_size = 20** (Issue #30 default, 不用 K=50 amplifier)
- **seed = 42** (R5 硬约束)

这样跟 task194 Issue #40 Gate 1 protocol-matched K=64 control **完全对齐**, 数字可直接比较 baseline 0.1020.

---

## 3. 实施路径 (R10 + R11.5 自主决策)

### Phase 0 (立即, 0 GPU): Gate 1 详细设计 + 脚本准备

1. ✅ 创建本 description (R9 max+1 = 332)
2. ⏭️ 写 3 个 Stage 1 launch 脚本: `task332_issue41_gate1_arm{A,B,C}_stage1.sh`
   - 复用 task84 baseline recipe
   - 唯一变量: per-layer κ ∈ {1.0, -0.726, -2.0}
3. ⏭️ 写 Stage 2 Sinkhorn 统一脚本 (复用 task84)
4. ⏭️ 写 Stage 3 T5-mini 统一脚本 (复用 task84)
5. ⏭️ 写 Stage 4 eval launcher (复用 task278 batch driver)

### Phase 1 (启动前 gate): Issue #40 Gate 1 验证

**Issue #40 Gate 1 (task194 K=64 protocol-matched)** Stage 3 RUNNING, 预计 ~16:35 完成 (Ep 200/200).
- **若 Issue #40 Gate 1 R@10 ≈ 0.1020 (baseline)**: task194 K=256 anchor 0.1053 invalid, task327 NO-GO 闭环 (已发生), Issue #41 Gate 1 可启动
- **若 Issue #40 Gate 1 R@10 ≈ 0.1041 (task194 一致)**: task194 protocol 真有效, Issue #41 Gate 1 仍可启动 (跟 baseline 0.1020 比较, 不依赖 anchor)

**两种结果都可启动 Issue #41 Gate 1**, 不阻塞.

### Phase 2 (启动, GPU 0/2/3): Stage 1 训练

3 arm 并行 Stage 1 1000 epoch (跟 baseline 一致):
- Arm A (κ=1.0) GPU 0
- Arm B (κ=-0.726) GPU 2
- Arm C (κ=-2.0) GPU 3
- 估计 ~3-4h / arm

### Phase 3 (Stage 1 后): Stage 2 Sinkhorn + Stage 3 T5-mini

- Stage 2 Sinkhorn ~5min / arm
- Stage 3 T5-mini 200 epoch ~2h / arm (跟 task327 类似)
- 总估计 ~2.5h / arm, 串行 (一个 GPU 一轮)

### Phase 4: Stage 4 R@10 eval

- 复用 task278 batch driver
- beam_size=20 (Issue #30 default)
- 比较 Arm A/B/C vs baseline 0.1020

---

## 4. 跟现有任务兼容性 (R7)

| 任务 | GPU | 状态 | Issue #41 Gate 1 兼容性 |
|------|-----|------|------------------------|
| task194 Gate 1 Stage 3 | GPU 1 | RUNNING Ep~30/200 | 兼容 (用 GPU 0/2/3) |
| task327 | - | DEAD, verdict 落盘 | 无影响 |
| task328 | - | KILLED (用户决策) | 无影响 |

Issue #41 Gate 1 启动时 GPU 0/2/3 全空闲, GPU 1 task194 在跑. 完美兼容.

---

## 5. 风险与缓解

| 风险 | 概率 | 缓解 |
|------|------|------|
| Stage 1 1000 epoch κ=-2.0 边界效应 | 中 | 训练前先在 EP5 sanity check, 跟 Arm A 同等 epoch 对比 |
| 跟 task287/task284 同样 κ 设置触发 USAGE-KILL | 低 | 区别: task287 κ frozen=0 (异常), 本设计用真实 κ (Ollivier/h-MDS); 应该不会触发 USAGE-KILL |
| 3 arm Stage 1 都 R@10 < baseline | 中 | 整体 NO-GO 闭环, 写 Issue #41 Gate 1 NO-GO verdict, 后续 Issue #41 关闭 |

---

## 6. R11.5 自主决策记录

1. **3 arm κ 选择**: 1.0 (baseline) / -0.726 (Ollivier mean) / -2.0 (h-MDS best) — 三档覆盖 baseline + 跨方法独立测量方向
2. **per-layer 全部用同一个 κ**: Stage 1 是 3 层 RQ-VAE, 3 arm 用相同 κ (不混用), 保持单变量对照
3. **协议对齐 task194 Gate 1**: batch_size=1024 + epochs=1000 + Sinkhorn off + beam=20 — 跟 task194 Issue #40 protocol-matched 一致, 数字可直接比 baseline
4. **不等 Issue #40 Gate 1**: Issue #41 Gate 1 决策阈值 vs baseline 0.1020, 不依赖 task194 anchor 0.1053 是否成立
5. **不抢 GPU 1**: Issue #41 Gate 1 用 GPU 0/2/3, 不抢 task194 Gate 1 Stage 3

---

## 7. 关联

- verdicts/task331_issue41_gate0_h_mds_input_space.md (Issue #41 Gate 0 PASS, 直接来源)
- verdicts/task331_issue41_gate0_input_h_mds.json (κ=-2.0 数据来源)
- verdicts/task331_issue41_gate0_h2_category_tree.json (κ=-0.739 数据来源)
- Task #70 verdict (Ollivier κ=-0.726 mean)
- Task #84 baseline (κ=1.0 default)
- Task #287 verdict (κ-decouple Phase A κ frozen=0 NO-GO, 跟本设计区别)
- Task #284 verdict (κ-decouple K=256 NO-GO)
- Task #327 verdict (K=256 + Issue #30 synergy NO-GO, K=256 anchor 仍待 Issue #40 验证)
- verdicts/task329_issue40_gate0_protocol_audit.md (Issue #40 Gate 0, 协议 leak 识别)
- Task #194 (Issue #40 Gate 1, RUNNING, protocol-matched K=64 control)

---

result: Task #332 Issue #41 Gate 1 设计 — 3-arm Stage 1 κ ablation {1.0 (baseline) / -0.726 (Ollivier) / -2.0 (h-MDS)}. 设计 only 待 Issue #40 Gate 1 验证 (Phase 1 gate, 不阻塞). GPU 0/2/3 启动, 跟 task194 Gate 1 (GPU 1) R7 兼容. 决策阈值 R@10 vs baseline 0.1020, ±2% 是中性.
