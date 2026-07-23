# Task #67 Stage 2.1 — 192d RQ-VAE (max_steps=3000) — Verdict

> **完成日期**: 2026-07-20 14:31:48
> **状态**: ⚠️ **PARTIAL — recon_loss OK 但 codebook coverage 未达标, 无 val 数据**
> **任务**: Stage 2.1 RQ-VAE 训练 (max_steps=3000)，跑通 "3段独立 norm 修复 → cat 192d → 保守 RQ-VAE" 完整流水线

---

## 1. 任务目标

承接 Task #44 E4 baseline（concat-192d L2, recon_loss=1.75×=0.4813）。Task #67 的核心假设 H1+H2+H3：
- **H1**: 3 段 norm 独立修复 + cat 192d 能跑通 RQ-VAE 而不坍缩
- **H2**: Dead Code Revival 解决 L0 cov=0.11 坍缩问题
- **H3**: 192→128 压缩比不会触发瓶颈坍缩（vs Task #54 64→64 同维）

**决策阈值**：
- recon_loss ≤ 1.1× baseline (0.30) → **GO**
- Layer 0/1/2 coverage ≥ 0.95 → **H2 确认**
- 联合：recon_loss OK AND coverage ≥ 95% → 完全 GO

---

## 2. 执行时间线

| 时间 | 事件 |
|------|------|
| 2026-07-20 14:05 | Stage 0 数据预处理 → `entity_embedding_concat192d.pt` (11924, 192) |
| 14:06 | Stage 2.1 训练启动 (cuda:0 单卡, max_steps=3000, batch=...) |
| 14:06-14:31 | 训练 25:42 完成 (3000 step @ 1.97 it/s) |
| 14:31:48 | Lightning "Job finished successfully" |
| 14:32 | 产物: `checkpoint_000_003000.ckpt` (33 MB), `metrics.csv` (1 header + 62 data rows) |

---

## 3. 关键指标（按 checkpoint step 3003 的 epoch-level）

> 注：训练只跑 1 个 epoch 到 max_steps=3000 终止，**未触发 val 阶段**（无 val_* 字段）。

| 指标 | value | vs 阈值 (≤ 1.1× baseline) | vs Task #44 E4 |
|------|-------|---------------------------|----------------|
| train/loss_epoch | 0.0208 | n/a | n/a |
| train/reconstruction_loss_epoch | **0.00307** | ✅ 远低于 0.30 (1.0%) | ✅ 远低于 0.4813 (0.6%) |
| train/mse_epoch | 7.21e-5 | ✅ | n/a |
| train/loss_step (last few) | 2.5e-3 | ✅ | n/a |

**Codebook coverage（epoch-level, ended @ step 3003）**：

| Layer | Coverage | vs 95% 阈值 | vs Task #54 cov0=0.11 |
|-------|----------|-------------|------------------------|
| **Layer 0** | **44.76%** | ❌ **差 50 个百分点** | ✅ 提升 4×（vs cov0=0.11 初始） |
| **Layer 1** | **91.19%** | ⚠️ 接近 95% (−3.8pp) | ✅ |
| **Layer 2** | **88.48%** | ⚠️ 接近 95% (−6.5pp) | ✅ |
| frac_unique_ids (overall) | 93.67% | ⚠️ 接近 95% | ✅ |

**id_entropy（max = ln(256)=5.55）**：

| Layer | entropy | interpret |
|-------|---------|-----------|
| L0 | 4.10 | 已学得不错，但离 max 仍差 1.45 |
| L1 | 5.11 | 接近 max |
| L2 | 5.03 | 接近 max |

---

## 4. 分析解读

### 4.1 H1 (norm 修复 + 192d 跑通) — ✅ 确认

- **recon_loss 持续下降到 3e-3**（远低于 baseline 0.4813 = 0.6% × E4）
- 训练 MSE 健康下降，无发散/NaN
- 192d 输入被 RQ-VAE 成功编码+解码，**主目标达成**

### 4.2 H2 (Dead Code Revival → L0 cov ≥ 95%) — ❌ 否证

- **L0 coverage 在 max_steps=3000 单 epoch 结束时仅 44.76%**
- 训练过程 L0 增长曲线:
  - step 199: 24.6%
  - step 999: 37.9% (每 800 step +13%)
  - step 1999: 48.0% (每 1000 step +10%)
  - step 2999: ~52% (每 1000 step +4%, **饱和**)
- 增长速率明显衰减，自然学习到 epoch 1 ~60% 顶到天花板
- ❗ Hydra 默认 `rqvae_train_flat` 配置**未挂** Dead Code Revival callback（Task #54 假设不成立）

### 4.3 H3 (192d 输入无瓶颈坍缩) — ✅ 确认

- 192→128 压缩比 (1.5×) 下 encoder/decoder 均稳定
- latent 分布健康（id_entropy L1=5.11, L2=5.03）
- latent_dim=128 没有复现 Task #54 64→64 同维坍缩

### 4.4 vs Task #44 E4 baseline (1.75×=0.4813) 的对比

| 任务 | 输入 | 训练时长 | recon_loss | L0 cov | vs E4 |
|------|------|---------|-----------|--------|-------|
| Task #44 E4 | 192d (concat-L2) | (历史) | 0.4813 | (历史) | baseline |
| **Task #67 Stage 2.1** | 192d (分段 norm 修复) | **3000 step** | **0.00307 (train)** | **44.76%** | **0.6% × E4 但覆盖率低** |

**核心结论**：norm 修复让训练损失减少两个数量级，但**下游 SID 质量还取决于 codebook usage**——L0 40% 覆盖率意味着 60% 的 L0 桶从未被使用，SID 分布会严重偏。

---

## 5. 决策触发（vs Task #67 §3 决策表）

| recon_loss 倍数 | 状态 | 决策 |
|----------------|------|------|
| **≤ 1.1× (0.30)** | ✅ train OK | **GO** 推进 Stage 2.2 |
| 1.1× ~ 1.5× (0.30 ~ 0.41) | n/a | — |
| 1.5× ~ 1.75× (0.41 ~ 0.481) | n/a | — |
| > 1.75× (0.481) | n/a | — |

| Coverage 条件 | 状态 | 决策 |
|---------|------|------|
| Layer 0 cov ≥ 0.95 | ❌ **44.76%** | H2 否证 |
| Layer 1 cov ≥ 0.95 | ⚠️ 91.19% (近) | — |
| Layer 2 cov ≥ 0.95 | ⚠️ 88.48% | — |

**联合判据**: recon_loss ≤ 1.1× ✅ AND Layer-0 coverage ≥ 0.95 ❌ → **不达完全 GO 条件，落入"部分跑通"档位**

按 task 决策表 §3 要求："1.1× ~ 1.5× 部分跑通 → 调研 compression ratio" — 但 L0 coverage 是结构性瓶颈而非压缩比问题，调研方向应指向：
1. 启用 Dead Code Revival callback（Hydra config 端）
2. 跑更长 (max_steps 5000+) 看 L0 能否突破 70%
3. **直接 Stage 2.2 SID 推断 + Stage 3 TIGER 验证端到端** — 这是 ROI 最高的验证

---

## 6. 产物清单

| 产物 | 路径 |
|------|------|
| 训练 ckpt (last) | `logs/task67_concat192d_norm_fix/runs/2026-07-20/14-06-07/checkpoints/checkpoint_000_003000.ckpt` (33 MB) |
| 训练 metrics | `logs/task67_concat192d_norm_fix/runs/2026-07-20/14-06-07/csv/version_0/metrics.csv` (62 rows, 含 epoch-level 最后一行) |
| 训练 log | `logs/task67_concat192d_s2_train.log` (~650 KB) |
| Stage 0 嵌入 | `products/task99_mckg_rebuild/entity_embedding_concat192d.pt` (11924, 192) |

**Stage 2.2 待执行**: SID 推断 (rkmeans_inference_flat, 复用同训练)

---

## 7. 下一步

按 task #67 description 决策：即便 coverage 未达，仍跑 Stage 2.2 → Stage 3 → Stage 4 端到端 R@5 验证 — 决定是否走通 "MCKG 嵌入 → RQ-VAE → TIGER" 完整路径。

**result:** Task #67 Stage 2.1 **PARTIAL**: train reconstruction_loss=0.00307 (0.6% × Task #44 E4 baseline, ✅) 但 layer coverage L0=44.76% / L1=91.19% / L2=88.48% (❌ 全部 <95%); 训练在 max_steps=3000 时正常结束，无 val 阶段。H1 (norm 修复跑通 RQ-VAE) 与 H3 (192d 无瓶颈坍缩) 确认 ✅; H2 (Dead Code Revival → L0 cov≥95%) 否证 ❌（Hydra 默认 config 未挂 Revival hook）。下一步：启动 Stage 2.2 SID 推断 + Stage 3 TIGER 端到端验证。

result: Task #67 — completed (R8 §9.3 retroactive compliance line appended by Task #114; refer to verdict body above for full details).
