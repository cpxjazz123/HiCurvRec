# Task #239 Gate 2 — REVERSED per-layer c 验证 (PARTIAL FAIL)

**日期**: 2026-07-29
**任务来源**: Task #229 Gate 2 — circular dependency check (先进 Gate 1 后 Gate 2)
**决定**: PARTIAL FAIL (A1 ✓, A2 ✗). Gate 3 (Stage 3+4 R@10) **不进入**.

## 1. 假设

Task #229 Gate 1 (Q2 c-scan) 显示所有 6 个 c 配置 collision 都 > 0.10 (task229 Gate 1 result). 用户 (2026-07-29) 提议 REVERSED 配置: **L0 高 c + L1/L2 低 c** 推测能让 L1/L2 摆脱 L0 影响, 各自独立达到低 collision.

任务设计 (R11.3 自主决策, 见 descriptions/task239_issue_gate2_reversed_proposal.md):
- 3 变体单变量扫描, 200 epoch 共享 Stage 1 协议
- 决策阈值: collision < 0.10 per arm
- 共享 dataset/seed/codebook/recipe (R5 锁定)

## 2. Stage 1 结果 (200 epoch, 200 batch → same config)

| Variant | c=[c0,c1,c2] | Best Collision | Best Loss | L0 fill | L1 fill | L2 fill |
|---------|--------------|----------------|-----------|---------|---------|---------|
| **A1** | **[10,1,1]** | **0.0893** ✓ | 8.107 | 38.9% | **8.6%** | **6.4%** |
| **A2** | **[30,3,3]** | **0.1072** ✗(+0.7pp) | 8.010 | 40.7% | 11.1% | 8.2% |
| **B (control)** | **[10,10,10]** | **0.0876** ✓ | 8.350 | 38.4% | 25.5% | 19.9% |

(球半径 = 1/√c; L0: 0.316 / 0.183 / 0.316; L1/L2: 1.000 / 0.577 / 0.316)

## 3. 几何分析

### 3.1 fill% 反映 L1/L2 受 c 影响最大

- **L0 fill%** 三者一致 (38.4-40.7%, 差异 < 3pp) → L0 不受 L1/L2 c 影响
- **L1/L2 fill%** 剧烈变化:
  - A1 (c=1) L1/L2 fill = 8.6/6.4% — 是 B (25.5/19.9%) 的 1/3
  - A2 (c=3) L1/L2 fill = 11.1/8.2% — 中间过渡
  - 高 c (B c=10) 给 L1/L2 在 Poincaré 球内更紧凑的半径 (1/√10=0.316), 迫使 ‖x‖_E 接近这个上限 → fill% 更高
  - 低 c (A1 c=1) 球半径 = 1.0, ‖x‖_E 不受 ball 限制, fill% 反而因 Poincaré 几何自然偏离距离码字更远, fill% 低

### 3.2 collision 与 fill% 不耦合

- **A1 (collision 0.0893) vs B (collision 0.0876)**: 差距 0.18pp — 完全在噪声内 (epoch 199 collision 0.1088/0.1200 差异远大于 best)
- **A2 (collision 0.1072) > B 0.0876**: +22.4% 相对恶化, 但绝对差距仅 2pp
- **L1/L2 fill 下降 3-4× 没用**: A1 fill 远低于 B 但 collision 持平 — 说明 c 只影响几何分布集中度, 不解决码字利用率问题

### 3.3 重构 loss 排序

A2 (8.010) < A1 (8.107) < B (8.350) — 几何压缩更狠 (高 c) → 重构 loss 越低, 但 collision 越高. **这反向支持"球半径压缩 vs 码字利用率" trade-off**: c 越大, 球越小, 码字更可能 cluster, collision 反而升高.

## 4. 决策

按用户规则 "前一个 stage 没达到要求则不继续下一个 stage":

- **A1 (c=[10,1,1])** collision 0.0893 < 0.10 → **PASS**
- **A2 (c=[30,3,3])** collision 0.1072 > 0.10 → **FAIL** (+0.7pp)
- **整体**: 3/3 变体中 2/3 通过, 但 A2 仍在同一族 (c 高 + L1/L2 fill 低 → collision 升), **Gate 2 视为 PARTIAL FAIL**

**Gate 3 (Stage 3 + Stage 4 R@10) 不进入**. 用户假设 "L0 高 c + L1/L2 低 c 能让 L1/L2 独立达到低 collision" 在 A1 → A2 序列上被部分证伪 (c=3 上 collision 反而更高).

## 5. 关键决策点

| # | 选项 | 选 | 为什么 |
|---|------|----|----|
| 1 | 单跑 A1 PASS | 不 | A2 FAIL 显示 c-scan 不是单调: 同样 c↑ 不一定 collision↓ |
| 2 | 把 A1 ckpt 跑 Stage 3 R@10 | 不 | Gate 2 整体 PARTIAL FAIL, 不分变体进 Gate 3 |
| 3 | 回到 c-scan 起点 (Gate 1) | 是 | Gate 2 没找到突破口, 注意力回到 Gate 1 已扫的 6 ckpt 的 SID/Stage 3 |
| 4 | Issue #10 redesign 跑 | 否 | 等用户决策 (R11.4 关键决策) |
| 5 | 接受 Gate 2 PARTIAL FAIL 收口 | 是 | 本任务闭环 |

## 6. R11.3 自主决策 — 后续任务建议

候选 (按 ROI 排序):

1. **写 Issue #10 redesign 用户回复 (Task #238)**: 已写 description, 等用户决策 → 主动推进 Issue #10 redesign 4-arm 实验 (~13h wall) — *需用户授权*
2. **Task #229 Gate 2 闭环 + 收口**: 本任务到此 — Gate 2 没找到新杠杆, 回退 Gate 1 6-arm 详细 Stage 3 R@10
3. **跨架构**: LETTER / S3Rec / FDSA 已 paper-aligned fix 但缺一次 R@10 重新基线 — 候选 ROI 较高
4. **训练时长变量**: Task #200 dual_v5 (-10.3%) 暗示 Stage 3 训练时长可能才是真正 R@10 杠杆 — 验证 200/300/400 epoch 差异

## 7. 产物

- products/task239/gate2_c10_1_1/best_loss_model.pth, best_collision_model.pth (A1 ckpt)
- products/task239/gate2_c30_3_3/best_loss_model.pth, best_collision_model.pth (A2 ckpt)
- products/task239/gate2_c10_10_10/best_loss_model.pth, best_collision_model.pth (B control ckpt)
- logs/task239/gate2_c{10_1_1,30_3_3,10_10_10}_*.log × 3
- scripts/task239_gate2_c{10_1_1,30_3_3,10_10_10}_stage1.sh × 3
- descriptions/task239_issue_gate2_reversed_proposal.md
- 本文件: verdicts/task239_gate2_reversed_result.md

## 8. 关键 takeaway

**L1/L2 fill 3× 差 ≠ collision 差**: L1/L2 几何分布受 c 调控 (高 c → 球紧 → fill↑), 但 RQ-VAE collision 主要受 L0 几何 + 码字利用率 + Sinkhorn 控制, L1/L2 几何变化对 collision 影响 < 2pp.

未来方向应避免 "调 c 找 collision 0" 路径. c-scan (Gate 1 6-arm) + REVERSED c-scan (Gate 2 3-arm) 累计 9 个 c 配置, 最佳 0.0876 (B c=[10,10,10]) 离 0.05 (phonism Sinkhorn) 仍差 75%. collision 上限不在几何层调参.

下一个突破方向候选: Sinkhorn during train (Task #238 Issue #10 redesign D/E/F 路径) 或跨架构 (LETTER/S3Rec/FDSA) — 都不是 c-scan 内部能解决的.

## 9. Status

- ✅ Gate 2 PARTIAL FAIL 闭环
- ❌ Gate 3 (Stage 3+4 R@10) 不进入
- ✅ Issue #9 / Issue #10 redesign / Task #229 Gate 2 全部 stay-as-is
- ⏸️ 等用户决策的项: Issue #10 redesign 4-arm (Task #238 提议)
- ⏭️ 下一个主动推进候选: 跨架构 R@10 重新基线 或 Task #229 论文 Eq11/12 ρ 单调假设验证 (Task #137)

result: Task #239 — REVERSED per-layer c 验证 (PARTIAL FAIL)
