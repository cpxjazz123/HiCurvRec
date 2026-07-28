# Task #238 — Issue #10 Redesign: 找 Sinkhorn 之外的 R@10 杠杆

## 来源
- GitHub Issue #10 (2026-07-28): [Validation] 3-arm converged collision 设计 + collision 指标口径统一
- Issue #10 Gate 0 (Task #236, 2026-07-28) PASS: 锁定 collision_rate 权威定义
- Issue #10 Gate 1 (Task #237, 2026-07-29) **PARTIAL FAIL**: Arm B (Sinkhorn=10) R@10=0.1021 跟 Arm A (Sinkhorn=0) 0.1020 持平; Arm C (Sinkhorn=30) 0.1058 仅高 3.7pp. B-C 碰撞率差仅 5pp (< 15pp 阈值). **3-arm 曲线退化为 2-arm**.

## 背景

**Arm A→C 的 +3.7pp R@10 增益的真正变量仍未知**:
| Arm | max_sinkhorn_iters | PRE-collision | R@10 |
|-----|-------------------|---------------|------|
| A (HG-Rec #84) | 0 | 0.99 | 0.1020 |
| B (Task #237) | 10 | 0.1005 | 0.1021 |
| C (phonism) | 30 | ~0.05 | 0.1058 |

Sinkhorn iters (0→10→30) 唯一变量无法解释 R@10 (0.1020→0.1021→0.1058) 的非单调跳变. 候选解释:
- (1) **架构差异**: Arm A 是 Poincaré hyp RQ-VAE, Arm C 是 vanilla RQ-VAE (Euclidean?). hyp vs euc 几何本身可能影响 R@10
- (2) **码字熵**: Sinkhorn 平滑码字概率分布, 增加码字使用均匀性 → 编码多样性 → 下游 T5 学到更可分 SID
- (3) **训练协议**: Arm C 可能用更长 / 不同 lr 协议 / Sinkhorn during training (vs Sinkhorn only at inference)
- (4) **初始化**: kmeans_init vs random 码字初始化
- (5) **SID 后处理**: 4th-digit dedup vs no dedup

## Issue #10 Redesign 提议

**核心思路**: 不再调 Sinkhorn iters (因 Sinkhorn 已饱和), 改为 **架构对比 (hyp vs vanilla RQ-VAE)** 单变量 + **Sinkhorn during training** 单变量. 这两个独立维度都是 Arm A → Arm C 增益的可能来源.

### 设计: 单变量 4 臂 (NEW)

| Arm | 架构 | Sinkhorn during train | Sinkhorn during infer |
|-----|------|------------------------|-----------------------|
| A (HG-Rec baseline, #84) | Poincaré hyp | ❌ | max_iters=0 |
| **D (new)** | **vanilla Euclidean** | **❌** | **max_iters=0** |
| **E (new)** | **vanilla Euclidean** | **✅** | **max_iters=30** |
| **F (new)** | **Poincaré hyp** | **✅** | **max_iters=30** |

### 4-arm 因果分解

| 比较 | 含义 |
|------|------|
| **A vs D** | 单变量架构 (hyp vs euc). 隔离 hyp 的 R@10 效应 |
| **D vs E** | 单变量 Sinkhorn-during-train (euc baseline). 隔离 Sinkhorn 训练时的 R@10 效应 |
| **A vs F** | 单变量 Sinkhorn-during-train (hyp baseline). 隔离 Sinkhorn 训练时对 hyp 的 R@10 效应 |
| **D vs A→C** | 重建 Arm C 路径: A + euc 替换 + Sinkhorn=30 |

### 决策阈值 (R11.3 自主推荐, 待用户决策)

| Gate | 指标 | 阈值 | Pass 含义 |
|------|------|------|-----------|
| Gate 0 (零 GPU) | 4-arm Stage 1 训练 + Stage 2 SID uniqueness | ≥ 90% unique | 码本健康 |
| Gate 1 (GPU) | 4-arm Stage 4 R@10 spread | \|max-min\| ≥ 5pp | 4 臂有真实 R@10 差异 |
| Gate 2 | 单变量隔离清晰 | D vs A Δ, D vs E Δ, A vs F Δ 三者无混淆 | 因果链成立 |
| Gate 3 | R@10 ≥ 0.1020 | Arm D/E/F 任一 ≥ baseline | 方向采纳 |

### 资源预算

- 4 臂 × 50 epoch Stage 1 = ~30 min × 4 = 2h (1 GPU)
- 4 臂 Stage 2 + Stage 3 200 epoch = ~10h × 4 = 40h (max 4 GPU parallel = 10h wall)
- 4 臂 Stage 4 eval = ~3 min × 4 = 12 min (1 GPU)
- 总计 ~13h wall clock + 12 min eval

### R11.3 自主决策

- **架构选择**: Issue #10 redesign 4-arm 设计 (NEW 提议, 不是 Issue #10 原文)
- **备选 A**: 单跑 Arm D (vanilla + no Sinkhorn), 仅验证架构效应 (~3h)
- **备选 B**: 单跑 Arm E (vanilla + Sinkhorn during train), 直接对比 Arm C path (~3h)
- **备选 C**: 接受 Issue #10 当前结论关闭 Issue, 把精力转到 Issue #9 Gate 1

### 阻塞项

- Issue #10 redesign **需要用户确认** (R11.4 关键决策: 改变 Issue #10 实验设计)
- 用户可选项: (a) 批准 4-arm 设计 (b) 选备选 A/B/C (c) 关闭 Issue #10 暂搁

## 步骤

1. **本 tick**: 写 Issue #10 redesign description (本任务)
2. **下 tick**: 等用户决策 (R10: 不主动启动, 等明确信号)
3. 用户批准后: 写 Stage 1 launcher × 4 + chain scripts (10min)
4. 启动 4-arm Stage 1 (parallel, ~2h)
5. Stage 2 SID + Stage 3 训练 (parallel, ~10h)
6. Stage 4 eval (sequential, ~12min)
7. Verdict + Issue #10 close

## 产物

- descriptions/task238_issue10_redesign_collision_lever.md (本文件)
- 候选: scripts/task238_stage1_arm_{D,E,F}.sh, scripts/task238_chain_arm_*.sh (待用户决策后)
- 候选: verdicts/task238_*_result.md (待用户决策后)
- Issue #10 comment: 提议 redesign 4-arm, 等用户反馈

## 依赖

- Issue #10 Gate 0 (Task #236) PASS — collision_rate 定义已锁定
- Issue #10 Gate 1 (Task #237) PARTIAL FAIL — 3-arm design 前提崩塌
- HG-Rec baseline #84 ckpt (作为 Arm A + Arm F 的 Stage 1 输入)
- Phonism baseline (作为 Arm C 的参照)

## Status

R10 自主决策: 本 tick 写 description + Issue #10 comment 提议 redesign, **不启动 Stage 1** (等用户决策, R11.4 关键决策点). GPU 全空闲 (4× L40S, 0% util).

