# Task #144 — κ + codebook 解耦训练调度 (用户提议)

> **任务目的**: 验证用户提议的训练调度修复: κ 冻结在 0 → 训练 codebook 健康 → 解冻 κ + 小 lr_theta → 监控 utilization → 掉则 freeze. 解决 Task #137/#142 发现的 free-curv HG-Rec 架构 NO-GO 问题 (κ 和 codebook 从训练一开始就纠缠, 互相放大不稳定).

> **完成日期**: in progress
> **状态**: 🟡 待启动 (等 Task #143 FDSA 闭环)

---

## 1. 背景

Task #137 + Task #89 + Task #142 三重证据确认 free-curv HG-Rec 架构 NO-GO:
- Task #137: 3 变体 (M=1/2 × κ_max=0.1/0.5) 全部坍缩 (1-15 unique SID vs baseline 8936)
- Task #142: Launch 4 geodesic kmeans init 单独 insufficient (unique SID = 2, 反而比 v1=12 更差)
- **共同根因**: κ 和 codebook 从训练一开始就纠缠在一起, 互相放大对方的不稳定 (用户 2026-07-24 反馈)

**用户提议方案 (2026-07-24, 推荐"第一个试")**:
> 训练最开始, 把 κ 强制冻结在 0 (纯欧式, 已反复验证稳定, 100% utilization).
> 先把 codebook 单独训练到健康、稳定为止.
> 确认 codebook 健康后, 再解冻 κ, 用一个很小的学习率、缓慢放开, 同时持续监控 utilization.
> 一旦 utilization 开始往下掉, 立刻把 κ 的学习率降回 0 或者直接冻结住, 不让它继续往前冲.
> **这个方案不需要改动量化器本身的实现, 只是调整训练调度, 工程成本最低**.

**优势 vs Task #142 Launch 4 (geodesic kmeans init)**:
- 训练调度层面干预, 不动 quantization 实现 (工程成本最低)
- Phase A (κ=0 freeze) 复用 Task #84 baseline 已知稳定的训练动力学
- Phase B (解冻 + 监控) 利用 baseline 已稳定的 codebook 作为 anchor, κ 只能在"稳定 codebook"附近微调
- 主动 freeze-on-collapse 保护机制, 防止 κ 学过头

## 2. 实验设计

**两阶段训练调度 (per 用户提议)**:

### Phase A — κ 冻结 (0 → codebook 健康)
- `kappa_max=0.0` (冻结 κ, 等价于 Task #84 baseline 纯欧氏 RQ-VAE)
- `theta_init=0.0` (κ 初始为 0)
- **训练 epochs**: 100 (足够让 codebook 健康; Task #84 baseline 100 ep 已 100% util)
- **monitor**: 每个 epoch 末 eval unique SID count + per-layer utilization
- **停止判据**: utilization ≥ 95% in all 3 layers → 进入 Phase B
- **fallback**: 100 ep 后 utilization < 95% → 报告训练不稳定 (unlikely, Task #84 已知稳定)

### Phase B — κ 解冻 (缓慢放开 + 监控)
- 解冻 `θ_m` 参数, 设 `lr_theta=1e-5` (极小, 缓慢放开)
- 监控每 10 epoch:
  - unique SID count (target: ≥ Phase A baseline - 5%)
  - per-layer utilization (target: ≥ 95% per layer)
- **触发 freeze**: 任意 layer utilization 跌 5% → 立刻 `lr_theta=0` (freeze θ_m)
- 继续训练到 200 epoch (跟 Task #137 v2 同 epoch budget)

### 实现 (R11.3 决策)

复用 `scripts/task89_stage1_train_rqvae.py` + 增加 3 个新 flag:
1. `--kappa_freeze_epochs=N` (Phase A 长度, default 100)
2. `--lr_theta_post_unfreeze=1e-5` (Phase B 解冻后 θ_m 学习率, default 1e-5)
3. `--utilization_freeze_threshold=0.05` (utilization 跌多少触发 freeze, default 5%)

Phase B 的 freeze-on-collapse hook: 监控每个 epoch 末 unique SID, 跌过阈值就 freeze θ_m optimizer state.

**复用 Task #138 已经实现的 `geodesic_kmeans` init**:
- Phase A 训练用 `kmeans_init=True` (Task #84 baseline 已有)
- 不用 geodesic kmeans (Task #142 已证 geodesic init 单独 insufficient)
- Phase A 用 vanilla euclidean kmeans init 即可

## 3. 决策触发 (vs Task #84 baseline R@10=0.1020 + 100% util)

| 观察条件 | 结果 | 决策 |
|---------|------|------|
| Phase A 100 ep 后 utilization ≥ 95% all layers | codebook 健康 ✓ | ✅ 进入 Phase B 解冻 |
| Phase B 200 ep 后 util 保持 ≥ 95% + R@10 ≥ 0.1020 | κ 解冻后稳定 + 维持 baseline 性能 | ✅ **解耦方案成功, free-curv 可行** |
| Phase B 200 ep 后 util ≥ 95% + R@10 显著下降 (<0.09) | κ 微调破坏 codebook 但 util 没掉 | ⚠️ 部分成功, 报结果 |
| Phase A 100 ep 后 utilization < 95% | 即使 κ=0 也不能让 codebook 健康 | ❌ **架构根因不是 κ**, NO-GO 更确认 |
| Phase B util 跌过阈值触发 freeze, util 恢复 | freeze-on-collapse 机制生效 | ✅ freeze 保护成功 |
| Phase B util 持续跌 (freeze 也阻止不了) | κ 已有梯度泄漏 → free-curv NO-GO | ❌ 终极 NO-GO |

## 4. 预算

| 阶段 | 估算时间 |
|------|---------|
| 代码改动 (task89 加 3 flag + freeze hook) | ~30 min (复用 Task #138 design) |
| Phase A 训练 100 ep | ~15 min (Task #89 v2 rate ~9s/ep) |
| Phase B 训练 100 ep | ~15 min |
| Stage 2 inference + utilization diagnostic | ~5 min |
| **总计** | **~65 min** |

不并行多臂 (A 臂 only): GPU 1 (FDSA fix 跑完后空闲).

## 5. 风险与缓解

**风险 1**: Phase A 即使 κ=0 也坍缩 (跟 Task #142 Launch 4 一样). 缓解: 复用 Task #84 baseline setup, 已验证稳定.

**风险 2**: Phase B κ 解冻 → utilization 跌 → freeze → 但 κ 已有非零值 → freeze 后仍有影响. 缓解: freeze 时同时 reset θ_m → 0 (回到纯欧氏).

**风险 3**: lr_theta=1e-5 太慢, 200 epoch 仍几乎不动. 缓解: 每 50 epoch log κ_m 终值, 如仍 < 0.01 报"κ 未能 escape 0", 这是 NO-GO 信号.

**风险 4**: 修改 task89_stage1_train_rqvae.py 是上游代码 (Task #89 自建, 不是 src/data/ 上游), 但属于自己写的 launcher, 修改灵活. R11.3 决策: 加 flag 而非改核心逻辑 (跟 Task #138 同 pattern).

## 6. 完成度跟踪

- [x] R9 audit (max=143, next=144)
- [x] Task #144 description 落盘 (per 用户 2026-07-24 提议)
- [ ] task89_stage1_train_rqvae.py 加 3 flag (kappa_freeze_epochs + lr_theta_post_unfreeze + utilization_freeze_threshold)
- [ ] Phase A 100 ep 完训, utilization ≥ 95% ✓
- [ ] Phase B 200 ep 完训 + freeze-on-collapse 监控
- [ ] Stage 2 forward pass + unique SID count
- [ ] 写 `verdicts/task144_kappa_codebook_decoupled_training_result.md` 含 `result:` 行
- [ ] loop.md §16 cleanup

## 7. R11.3 自主决策

- **不复用 Task #142 launcher**: 那是 init-only 方案 (geodesic kmeans init), 不含训练调度 freeze. Task #144 需要新训练脚本.
- **Phase A 100 epoch (vs Task #137 v2 的 200 ep)**: Task #84 baseline 100 ep 已稳定 100% util, 100 ep 足够 Phase A; Phase B 再 100 ep 让 κ 缓慢放开.
- **lr_theta=1e-5 (vs Task #137 v2 的 5e-3)**: 用户明确说"很小的学习率", 1e-5 比 5e-3 小 500x, 给 κ 极慢调整空间.
- **freeze-on-collapse 阈值 5%**: 适中阈值 (不能太敏感 1%, 也不能太迟钝 20%).
- **不立即 launch**: 等 Task #143 FDSA fix 跑完释放 GPU 1 (R7 优先).

## 8. 关联

- 用户 2026-07-24 提议 (R11.3 接收 + 立即登记为 backlog 候选)
- Task #142 verdict — `verdicts/task142_free_curv_codebook_recovery_result.md` (4 launch 全部 collapse, 推翻需新方案)
- Task #137 verdict — `verdicts/task137_kappa_stereographic_fix_retrain_result.md` (架构根因声明)
- Task #89 verdict — Stage 0/1 NO-GO 闭环
- Task #84 baseline R@10=0.1020 — κ=0 已知稳定 reference
- Task #138 — 提供 geodesic_kmeans + dead_code_reset hook (Task #144 不直接复用, 但 pattern 同)

---

## 9. Phase 2 备选: 软量化退火 (用户 2026-07-24 提议, 若 Task #144 不够再上)

用户补充提议: "如果第一步 (κ 解耦) 还不够, 再上软量化退火 — 直接针对'赢家通吃'这个训练动力学机制".

**核心思路**: 当前 hard-assignment VQ (每次精确选一个最近 codeword) + straight-through 梯度天生赢家通吃. 改成:
- **训练早期**: 软分配 (对所有 codeword 算带温度 softmax 权重, 每个 codeword 都能拿梯度, 不会有人一开始完全拿不到信号而"死掉")
- **训练后期**: 慢慢降低温度, 软分配逐渐逼近硬分配

**参考**: VQ-VAE/VQ-GAN 社区标准做法 (Gumbel-Softmax / softmax with temperature annealing).

**与 Task #144 Phase A/B 关系**:
- Task #144 解决"κ 和 codebook 互相纠缠放大"问题
- 软量化退火 解决"硬分配 VQ 赢家通吃 + dead code"问题
- 两个问题可叠加: Phase A (κ 解耦) + Phase B (软量化退火) 联合
- 工程成本: 软量化退火需改量化器 forward 传播逻辑 (per 用户: "改动涉及量化器前向传播, 比第一步工程量大, 但更直接对准机制")

**R11.3 决策**: 不立即登记 Task #145, 等 Task #144 verdict 决定是否需要叠加软量化退火.
**触发上 Phase 2 条件** (R10 backlog 候选, 待 Task #144 闭环后判断):
- Task #144 Phase A κ=0 freeze 仍坍缩 → 软量化退火必须上
- Task #144 Phase B κ 解冻后 util 跌 → 软量化退火叠加 (稳定 + 解耦双管齐下)
- Task #144 完整成功 (util ≥ 95% + R@10 ≥ baseline) → 软量化退火可省略, 但仍是备选 (可能提升质量)

**实现伪代码** (R11.3 草案, 等 Task #145 启动时落实):
```python
# 替换 hard assignment:
# distances: (B, num_emb)
# τ: temperature, annealed from τ_high (e.g. 1.0) → τ_low (e.g. 0.01)
weights = softmax(-distances / τ)            # (B, num_emb)
quantized = weights @ codebook               # soft assignment
# straight-through:
quantized_st = z_e + (quantized - z_e).detach()
# anneal: τ = max(τ_low, τ * decay_rate^epoch)  # exponential anneal
```