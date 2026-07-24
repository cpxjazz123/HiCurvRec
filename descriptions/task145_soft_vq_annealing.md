# Task #145 — 软量化退火 (Soft VQ-VAE/VQ-GAN 风格)

> **任务目的**: 若 Task #144 κ-decouple 不足以解决 free-curv codebook collapse, 引入软量化 (带温度 softmax 权重) + 温度退火, 直接对症"赢家通吃"训练动力学. 这是用户 2026-07-24 提议的 **Phase 2 修复方案**, 接续 Task #144 (Phase 1 κ-decouple).

> **完成日期**: (in progress)
> **状态**: 🟡 待启动 — 等 Task #144 verdict

---

## 1. 背景

Task #137/#89/#142 三重证据证 free-curv 架构 NO-GO; Task #144 (Phase 1) 尝试 κ-decouple. 若 Phase 1 仍不够, 启用本任务 (Phase 2 软量化退火).

**根因机制 (用户 2026-07-24 反馈)**: 当前 VQ hard assignment (每次精确选最近 codeword) + straight-through 梯度, 天生赢家通吃 — 早期 batch gradient 集中在少数 codeword, 其余 codeword 一开始就完全拿不到信号 → 永久"死码". 即使 κ 解耦 (Phase 1), 硬分配本身的赢家通吃动力学仍在.

**Phase 2 直接对症**: 训练早期用 softmax-weighted soft assignment (每个 codeword 拿到梯度), 后期温度 τ → 0 渐进逼近 hard assignment (跟 VQ-VAE/VQ-GAN 社区标准做法一致).

**关联**:
- [[free-curv-user-2026-07-24-proposals]] (Phase 1 + Phase 2 用户提议)
- [[free-curv-codebook-collapse]] (Task #137 架构根因)
- Task #144 description §9 (软量化退火伪代码 + 触发 Phase 2 条件)
- Task #144 verdict (待 Task #144 完, 若 R@10 < 0.0973 baseline 即触发本任务)

---

## 2. 实验设计

**变量**: 量化器 forward 内部从 hard argmin assignment 改成 softmax-weighted soft assignment (带温度 τ, τ annealing schedule)

**保持不变**:
- HG-Rec FreeCurvHRQVAE 架构 (Task #89 baseline)
- M=1, κ_max=2.0, seed=42
- Task #144 κ-decouple 调度 (Phase A 冻结 100ep + Phase B 解冻 100ep)
- data, optimizer, batch size
- Stage 2/3/4 eval 流程

**改动范围** (最小化, 只 patch 量化器 forward):
1. `hrqvae_free_curv.py:HVectorQuantization.forward` 中:
   ```python
   # 旧 (hard assignment):
   distances = compute_distances(z_e, codebook)  # (B, K)
   encoding_indices = torch.argmin(distances, dim=-1)  # (B,)
   quantized = codebook[encoding_indices]  # gather
   # commitment loss / codebook loss via SG/STE

   # 新 (soft assignment with temperature τ):
   soft_weights = F.softmax(-distances / tau, dim=-1)  # (B, K)
   quantized_soft = soft_weights @ codebook  # (B, D), weighted sum
   # straight-through estimator (still needed for gradient):
   quantized = quantized_soft + (quantized_hard - quantized_soft).detach()
   # commitment loss: ||sg[z_e] - z_q||² + ||z_e - sg[z_q]||² (不变)
   ```
2. τ annealing schedule:
   - `tau_start=1.0` (训练开始: 完全均匀分配, 每 codeword 拿到梯度)
   - `tau_end=0.01` (训练后期: 趋近 hard assignment, 但保留微小梯度信号避免再次坍缩)
   - 退火方式: linear over 200 epochs (`tau = tau_start + (tau_end - tau_start) * epoch / total_epochs`)
   - 备选: cosine annealing
3. `task89_stage1_train_rqvae.py` 新 flag:
   - `--soft_vq_enabled` (default False)
   - `--soft_vq_tau_start` (default 1.0)
   - `--soft_vq_tau_end` (default 0.01)
   - `--soft_vq_anneal_type` (linear | cosine, default linear)

**启动命令** (主臂):
```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task145_main \
  nohup python3 scripts/task89_stage1_train_rqvae.py \
    --M=1 --kappa_max=2.0 --lr_theta=0.0 --theta_init=0.0 \
    --kappa_freeze_epochs=100 --lr_theta_post_unfreeze=1e-5 \
    --utilization_freeze_threshold=0.05 \
    --soft_vq_enabled --soft_vq_tau_start=1.0 --soft_vq_tau_end=0.01 \
    --epochs=200 --seed=42 \
    --ckpt_dir=products/task145/train/main_phase1plus2 \
    > logs/task145/main_phase1plus2.log 2>&1 &
```

**3 臂并行** (按 R10 + R11.3 自主决策):
| Arm | κ 调度 | soft_vq | 目的 |
|---|---|---|---|
| Main | Phase A + B (Task #144 调度) | ✅ τ: 1.0 → 0.01 | 验证 soft VQ 是否独立于 κ-decouple 解决 collapse |
| Phase 2 only | κ 全程冻结 0 (Task #144 Arm A) | ✅ τ: 1.0 → 0.01 | 隔离 soft VQ 贡献 (vs Phase 1 only baseline) |
| Phase 1+2 对照 | Phase A + B | ❌ (Task #144 Arm B baseline) | 复测 Task #144 Arm B 看是否复现 collapse |

主臂最关键 — 验证 Phase 2 是否独立改善. Phase 2 only arm 是隔离变量. Phase 1+2 对照是 sanity check.

---

## 3. 决策触发 (vs Task #144 baseline)

| 指标条件 | 结果指标 | 决策 |
|----------|----------|------|
| Main R@10 ≥ 0.0973 (Task #84 baseline) + util ≥ 95% | ✅ Phase 2 工作 | free-curv + soft VQ 可行, paper Section 5.4 update |
| Main R@10 < 0.0973 (跟 Task #144 Arm B 类似 collapse) | ❌ Phase 2 不够 | NO-GO 终极确认, free-curv 三层修复 (geodesic kmeans + κ-decouple + soft VQ) 全部失败, 任务结束 |
| Phase 2 only R@10 ≥ 0.0973 + Main R@10 < 0.0973 | 部分 | 说明 κ-decouple 干扰 soft VQ 效果, 但 soft VQ 单独能修复 — 改用 Phase 2 only 调度 (Phase 1+2 叠加不优) |
| Main R@10 ≥ 0.0973 but util < 50% | ❌ 异常 | soft VQ 让 R@10 高但 codebook 利用率低 — 训练动力学修复但表达力差, 需要降 τ_end (e.g. 0.001) 重新跑 |

**核心判据**: 跟 Task #84 baseline R@10=0.0973 (item-level, musical_instruments) + utilization ≥ 95% 对齐.

---

## 4. 预算

| 阶段 | 估算时间 |
|------|---------|
| 改 hrqvae_free_curv.py 量化器 (~30 行 patch) | ~30 min |
| 改 task89_stage1_train_rqvae.py 新 flag (~20 行) | ~20 min |
| Main arm Stage 1 (200 ep) | ~3 min |
| Phase 2 only arm Stage 1 | ~3 min |
| Phase 1+2 对照 arm Stage 1 | ~3 min |
| Stage 2 (RKMeans inference, 3 arms × 2 min) | ~6 min |
| Stage 3 (T5 train, 3 arms × 30 min) | ~1.5 h |
| Stage 4 (R@10 eval, 3 arms × 5 min) | ~15 min |
| 总计 (含 3 臂并行 on GPU 0/2/3, GPU 1 留给 Task #143 FDSA) | **~2 h** |

注: 训练本身 (Stage 1) 极快 (~3 min), 主要耗时在 Stage 3 T5 训练.

---

## 5. 风险与缓解

**风险 1**: τ annealing schedule 不当 → 训练动力学震荡
- 缓解: 用 R137 NaN guard, 若 loss 震荡 > 5× baseline variance → 自动 raise + 调度降级

**风险 2**: soft VQ 让 expressivity 下降 (τ 趋近 0 但永远 > 0, 残差细微噪声)
- 缓解: 决策触发器 "util < 50% but R@10 高" 已识别此风险, 触发 τ_end 降低

**风险 3**: Stage 1 成功但 Stage 3 T5 训练不稳定 (跟 Task #144 类似的 collapse 模式迁移到 T5 阶段)
- 缓解: Stage 3 monitor 沿用 Task #144 验证成功的 R12 ckpt saving + R137 NaN guard

**风险 4**: 3 臂 Stage 3 跑 T5 占满 GPU 1/2/3, 跟 Task #143 FDSA 冲突
- 缓解: 先让 Task #143 FDSA 收敛完 (valid R@10 已 0.028 ep 0+, 应该 ~30 min), 再启动 Stage 3

**风险 5**: Gumbel-Softmax 替代 (备选实现) 跟 standard softmax temperature 在数值上差异
- 缓解: 主方案用 standard softmax (跟 VQ-VAE paper 一致), Gumbel-Softmax 仅作备选 (R11.3 决策)

---

## 6. 完成度跟踪

- [ ] 等 Task #144 verdict (判定 R@10 ≥ 0.0973 ?)
- [ ] 若 Task #144 失败: 启动本任务
  - [ ] Patch `hrqvae_free_curv.py` 量化器 soft assignment (含 τ param)
  - [ ] Patch `task89_stage1_train_rqvae.py` 加 4 个 soft_vq flag
  - [ ] py_compile PASS + sanity check (small test)
  - [ ] Launch 3 臂并行 (Main + Phase 2 only + Phase 1+2 对照)
  - [ ] Stage 2 RKMeans inference (3 arms)
  - [ ] Stage 3 T5 train (3 arms, ~30 min each)
  - [ ] Stage 4 R@10 eval + utilization diagnostic
  - [ ] 写 verdict `verdicts/task145_soft_vq_annealing_result.md`
- [ ] 若 Task #144 成功: 本任务不启动, 标记 "不适用 — Phase 1 修复成功"

---

## 7. 决策触发条件 (相对 Task #144 verdict)

| Task #144 Arm B 主指标 | 本任务决策 |
|--------------------------|-----------|
| R@10 ≥ 0.0973 (baseline) + util ≥ 95% | ❌ 不启动本任务. Phase 1 修复成功, free-curv 可行 |
| R@10 < 0.0973 or util < 50% | ✅ 启动本任务. Phase 2 软量化退火 |
| R@10 接近 baseline 但 util < 95% | 🤔 模糊: R11.3 自主决策是否启动, 看 Task #144 verdict 推荐 |

**为什么不是默认启动**: 软量化退火改动面大 (量化器 forward), 工程成本比 Phase 1 高. 若 Phase 1 已足够, 不应引入额外复杂度. Phase 2 仅在 Phase 1 失败时启用.

---

## 8. 用户 Phase 2 提议原文 (2026-07-24, 引用)

> "如果第一步还不够, 再上软量化退火——直接针对'赢家通吃'这个训练动力学机制. 现在的量化用的是硬分配 (每次都精确选一个最近的 codeword) 加 straight-through 梯度, 这套机制天生容易赢家通吃. 改成训练早期用软分配 (不是精确选一个, 而是对所有 codeword 算一个带温度的 softmax 权重, 每个 codeword 都能拿到一点梯度, 不会有人一开始就完全拿不到信号而'死掉'), 训练后期再把温度慢慢降低, 让软分配逐渐逼近硬分配. 这是标准 VQ-VAE/VQ-GAN 社区对付 codebook 坍缩最常见的做法, 改动会涉及量化器的前向传播逻辑, 比第一步的工程量大一些, 但更直接对准这次诊断出来的真正机制."

---

## 9. 关联

- [[Task #144 κ-decouple description §9]] (本任务触发条件 + 伪代码设计)
- [[free-curv-user-2026-07-24-proposals]] (用户提议 memory)
- [[free-curv-codebook-collapse]] (Task #137 verdict, 架构根因)
- Verdicts: `verdicts/task145_soft_vq_annealing_result.md` (待写)
- Description: `descriptions/task145_soft_vq_annealing.md` (本文件)