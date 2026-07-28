# Task #138 — Free-Curv codebook collapse 修复 (A 方案 geodesic kmeans + B 方案 dead-code reset)

> **完成日期**: 2026-07-24
> **状态**: ✅ 已闭环 (A 臂训练 ep 50 crash + Stage 2 codebook health check 完成, NO-GO 判定)
> **核心目的**: 承接 Task #137 决定性诊断 (free-curv 3 变体全坍缩到 1-15 unique SID, 根因 = kmeans_init euclidean vs VQ argmin geodesic metric mismatch), 验证 **A 方案 (geodesic-space kmeans_init) + B 方案 (dead-code reset)** 是否解决 codebook collapse.
> **决定性发现**: **A+B 方案未解决坍缩**. A 臂 (M=1, κ_max=0.5, geodesic_kmeans=ON) ep 50 best ckpt Stage 2 codebook = **64 unique SID, 99.35% collision, max conflict 9602/9922**. B 方案 (dead-code reset) 触发 NaN (ep 3), A 方案独立跑也在 ep 51 触发 NaN. κ_m 这次是**干净的中间值** (L0=+0.069, L1=+0.043, L2=+0.042, 非饱和), 但 codebook 依然坍缩 → **确认 κ 饱和从来不是坍缩主因**, metric mismatch 修复方向无效. **Task #89 free-curv RQ-VAE 框架终审: NO-GO, 永久放弃.**

---

## 1. 设计 — A 方案 + B 方案 patch 说明

### 1.1 承接 Task #137 根因诊断

Task #137 决定性结论 (见 `verdicts/task137_kappa_stereographic_fix_retrain_result.md` §4):
- R137 fix (4 处 hard-branch → torch.where 统一算子) 接通 κ<0/κ>0/κ=0 autograd, **必要但不充分**
- free-curv 3 变体 (M=1 κ_max=0.5 / M=1 κ_max=0.1 / M=2 κ_max=0.5) **全部坍缩** 到 1-15 unique SID
- 根因 (用户诊断确认): **kmeans_init 在 raw euclidean latent 空间, VQ argmin 用 per-component geodesic 距离 → metric mismatch**
- M=1 也坍缩 → 跟切分无关, 是 init metric vs argmin metric 不一致

### 1.2 A 方案 — geodesic-space kmeans init

**patch 位置**: `HG-Rec/model/hrqvae_free_curv.py` (加 `--geodesic_kmeans` + `--re_kmeans_every` flag)

**思路**: init centroid 不在 raw euclidean latent 撒点, 而是在**当前 κ_m 对应的 manifold space** 跑 kmeans, 让 init metric 与 VQ argmin metric 一致. 并且每 `re_kmeans_every=50` epoch 用当批 latent 重新 kmeans, 追踪 codebook drift.

**执行验证**:
- ep 1 首批 geodesic init done (sklearn KMeans, L2 层 179/256 distinct clusters — duplicate points warning, 正常)
- ep 50 re-kmeans on 2048 latents (L1 127/128, L2 251/256 distinct)

### 1.3 B 方案 — dead-code reset

**patch 位置**: `HG-Rec/model/hrqvae_free_curv.py` (加 `--dead_code_reset_every` + `--dead_code_reset_threshold` + `replace_ratio` flag)

**思路**: 训练中每 `dead_code_reset_every=100` batches, 把 utilization 低于 threshold 的 dead codeword 重置为 batch 内 random latent (replace_ratio=0.1), 强制复活死码防坍缩.

**执行结果**: ❌ **NaN 崩溃**. 首次 A+B 全开跑: ep 3 batch 25 `[B方案] batch 100: reset 25 dead codes` 之后立即 `NaN/Inf loss ... kappa=[[nan],[nan],[nan]]`. dead-code reset 把 random latent 塞进 codebook 后, geodesic distance 在 poincare ball 边界产生 NaN, 传染到 κ.

**R11.4 自决**: 禁用 B 方案 (`dead_code_reset_every=0`), 只跑 A 方案独立验证其健康性 (见 launcher log `R11.4: B方案禁用`).

**py_compile**: ✅ PASS (Task #137 R137 + Task #138 A/B flag 均编译通过)

---

## 2. 执行 — A 臂训练 + Stage 2

### 2.1 A 臂训练 (M=1, geodesic_kmeans=ON, B 禁用)

- **Launcher**: `scripts/task138_A_arm_geodesic_init.sh` (等 Task #137 C 臂 PID 2932710 退出后启动)
- **Patched script**: `scripts/task89_stage1_train_rqvae.py` (加 `--geodesic_kmeans` / `--re_kmeans_every` / `--dead_code_reset_every` flag)
- **PID**: 3028711 (GPU 1, seed=42, θ_init=0.01, κ_max=0.5, 1000 epoch 计划)
- **实际结果**: ❌ **ep 51 batch 1 NaN 崩溃** (R137 NaN guard 触发, `total_loss=nan, quant_loss=nan, kappa=[[nan],[nan],[nan]]`)
- **保留 ckpt**: ep 50 `best_loss_model.pth` (loss=1.4264, R12 强制保存生效)

**训练轨迹** (log `task138_arm_A_M1_jul-24-2026_15-39-01.log`):
```
ep   1 loss=3.7090 recon=1.6576 quant=2.0514 L0κ=[+0.0203] L1κ=[+0.0168] L2κ=[+0.0173]
ep  10 loss=2.4965 recon=1.2676 quant=1.2289 L0κ=[+0.0548] L1κ=[+0.0367] L2κ=[+0.0376]
ep  20 loss=1.7613 recon=1.2822 quant=0.4791 L0κ=[+0.0644] L1κ=[+0.0416] L2κ=[+0.0411]
ep  50 loss=1.4264 recon=1.3105 quant=0.1160 L0κ=[+0.0690] L1κ=[+0.0432] L2κ=[+0.0422]  ← best ckpt
ep  51 NaN crash (re-kmeans at ep 50 后触发)
```

### 2.2 Stage 2 codebook 生成 (ep-50 best ckpt)

- **Script**: `scripts/task138_stage2_geodesic_init.py` (GPU 0, R7 空闲卡)
- **Output**: `HG-Rec/dataset/Instruments/Instruments_t5_hrqvae_geodesic_init.npy` (9922×4) + `_meta.json`
- **含 30 iter Sinkhorn collision resolution** (每 iter ~49-50 collision groups 锁死不减少)

---

## 3. 关键指标 — unique SID / utilization / κ_m 终值

### 3.1 A 臂 (ep 50 best ckpt) codebook 健康度

| 指标 | 值 |
|------|-----|
| 总 item 数 | 9922 |
| **3-token unique SID** | **64** |
| **collision rate** | **0.9935 (99.35%)** |
| **max conflict** (单一 SID 塞入 item 数) | **9602 / 9922** |
| L0 (64) utilization | **35/64 (54.7%)** |
| L1 (128) utilization | **36/128 (28.1%)** |
| L2 (256) utilization | **37/256 (14.5%)** |
| 4-token dedup 后 unique | 9922 (靠 append digit L3=9602 强行去重) |

### 3.2 κ_m 终值 (ep 50, R137 verify 干净读数)

| Layer | κ_m 终值 | 分支 | 是否饱和 |
|-------|----------|------|----------|
| L0 | **+0.0690** | spherical | ❌ 中间值 (非 κ_max=0.5) |
| L1 | **+0.0432** | spherical | ❌ 中间值 |
| L2 | **+0.0422** | spherical | ❌ 中间值 |

**关键**: κ_m 全在 +0.04~+0.07 (远离 κ_max=0.5 边界), 是 **Task #89/#137 全程首次干净的中间值读数** (Task #137 v1 A 臂饱和到 +0.4997). 但 codebook 依然坍缩 → **κ 饱和从来不是坍缩主因, 修复 metric mismatch 也无效**.

---

## 4. 对比 — baseline vs Task #137 v1 A-arm vs Task #138 A-arm

| Model | L0 util | L1 util | L2 util | Unique SID | Collision | κ_m 模式 |
|-------|---------|---------|---------|------------|-----------|----------|
| **Task #84 baseline** (fixed κ=-1 poincare, euclidean kmeans_init) | **100%** | **100%** | **100%** | **8936** | 9.94% | 固定 κ=-1 |
| Task #137 v1 A-arm (free-curv, euclidean init, κ_max=0.5, 1000 ep) | collapsed | collapsed | collapsed | **12** | 99.88% | κ 饱和 +0.4997 |
| Task #137 v2 A-arm (free-curv, euclidean init, κ_max=0.1, 200 ep) | collapsed | collapsed | collapsed | **1** | 99.99% | κ 弱 +0.026 |
| **Task #138 A-arm** (free-curv, **geodesic init**, κ_max=0.5, ep 50) | **54.7%** | **28.1%** | **14.5%** | **64** | 99.35% | **κ 中间值 +0.04~+0.07** |

**解读**:
- Task #138 A 臂 unique SID **64** > Task #137 v1 (12) > v2 (1), 略有改善但 **仍远低于 baseline 8936** (仅 0.72%)
- 64 unique = L0 层 utilization (35 个 codeword) × 极少 L1/L2 组合, 本质仍坍缩 (99.35% item 挤在 top-SID)
- geodesic init 让 L0 从 v1 的 ~1% 提到 54.7%, 但 L1/L2 (28%/14.5%) 仍严重坍缩 → **init 一次性对齐不足, 训练 dynamics 重新拉回坍缩**
- **κ 干净中间值 + 仍坍缩** = 决定性反证: 坍缩不是 κ 饱和造成, geodesic metric alignment 也不能救

---

## 5. 决策 — A+B 方案是否解决坍缩? κ 终值是否干净?

### 5.1 决策触发表 (vs description §3 阈值)

| 决策条件 | 实际 | 决策 |
|----------|------|------|
| L0/L1/L2 utilization 均 ≥ 80% | L0 54.7% / L1 28.1% / L2 14.5% ❌ | **修复失败** |
| 任一层 < 80% | 全部 < 80% | 部分修复 → 不满足 |
| **任一层 < 50%** | **L1 28.1%, L2 14.5%** ❌ | ✅ **切 Task #84 baseline, 永久放弃 free-curv** |
| κ_m ∈ [-0.5,+0.5] 非饱和 | +0.04~+0.07 ✅ | κ 读数干净 (双保险) |

### 5.2 两个核心问题的回答

**Q1: A+B 方案是否解决了 codebook 坍缩?**
❌ **没有.**
- B 方案 (dead-code reset) 直接 NaN 崩溃, 不可用
- A 方案 (geodesic init) 独立跑到 ep 50 (然后也 NaN), Stage 2 codebook = **64 unique / 99.35% collision / L1 28% L2 14.5% utilization** → 仍是坍缩
- 相比 Task #137 v1 (12 unique) 有边际改善, 但离 baseline 100% (8936 unique) 差 2 个数量级

**Q2: κ 终值是否干净 (中间值非饱和)?**
✅ **干净.** κ_m = L0 +0.069 / L1 +0.043 / L2 +0.042, 全在 spherical 分支中间值, 未饱和到 κ_max=0.5.
- 这是 Task #89/#137 全程**首次拿到干净的中间 κ 读数** (证明 R137 fix + geodesic init 让 κ 学习不再病态饱和)
- **但正因为 κ 干净仍坍缩**, 反证 "坍缩 ≠ κ 饱和" — 坍缩是 free-curv VQ 架构与 codebook init/更新机制的根本冲突

### 5.3 决定性结论

- **R137 fix**: 必要 (接通 autograd), 已闭环
- **A 方案 geodesic init**: metric mismatch 假设方向正确 (L0 从 1% → 54.7%), 但**不充分** — L1/L2 仍坍缩, 且训练不稳定 (ep 51 NaN)
- **B 方案 dead-code reset**: ❌ 不可用 (poincare ball 边界 random latent 触发 NaN)
- **Task #89 free-curv product manifold RQ-VAE 框架**: **终审 NO-GO, 永久放弃**
- **SOTA 上限**: Task #84 baseline (固定 κ=-1 poincare) R@10=0.1020, utilization 100% = 当前最佳可用方案

---

## 6. 后续

### 6.1 C 方案 (EMA codebook) 评估

Task #137 memory 列的第 3 修复方向 (EMA codebook + 死码重置) **不再单独启动**:
- B 方案 (hard dead-code reset) 已证实在 poincare ball 触发 NaN, EMA 软更新虽可能更稳但需改 HG-Rec 上游 VQ 核心 (R11.4 critical, 高 risk)
- 即使 EMA 能稳住数值, geodesic init 已证 L1/L2 层坍缩来自架构 (init 对齐后训练 dynamics 重新坍缩), EMA 不解决 init/argmin metric 一致性
- **ROI 判定**: baseline R@10=0.1020 已够好, C 方案预期最好也只是接近 baseline, 边际价值 < GPU 成本 → **不启动**

### 6.2 Task #89 verdict 收尾

- Task #89 retro caveat 已更新 (见本 verdict §7 + `task89_free_curv_product_manifold_result.md` §10)
- Task #89 "数据本质欧氏" 结论定性: R137 bug 污染 → Task #137 推翻 (κ 学到非零) → **Task #138 终审 free-curv 框架 NO-GO** (κ 干净但坍缩)
- paper Section 5.4 结论: **free-curvature product manifold RQ-VAE 在 HG-Rec 实现下不可行**, 用固定 κ=-1 baseline (R@10=0.1020) 作为 geometry 章节的落点

### 6.3 R8 §16 归档

Task #138 已闭环 (verdict 写完 + result 行 + PID 已不存在), 从 loop.md §16 删除任务描述行, verdict 保留在 `verdicts/task138_*.md`.

---

## 7. 产物清单

- 训练 ckpt: `products/task138/train/arm_A_M1/best_loss_model.pth` (ep 50, loss=1.4264)
- crash 备份: `products/task138/train/arm_A_M1/best_loss_model_Ep2_crash.pth.bak` (B 方案首跑 ep 2)
- κ history: `products/task138/train/arm_A_M1/kappa_history.json`
- Stage 2 codebook: `HG-Rec/dataset/Instruments/Instruments_t5_hrqvae_geodesic_init.npy` (9922×4, 64 unique SID) + `_meta.json`
- Launcher: `scripts/task138_A_arm_geodesic_init.sh`
- Stage 2 脚本: `scripts/task138_stage2_geodesic_init.py`
- Quick kmeans check: `scripts/task138_A_arm_quick_kmeans_check.py`
- Patch: `HG-Rec/model/hrqvae_free_curv.py` (geodesic_kmeans + dead_code_reset flag), `scripts/task89_stage1_train_rqvae.py` (flag 透传)
- 日志: `logs/task138_arm_A_M1_jul-24-2026_15-39-01.log` (A-only NaN ep 51), `logs/task138_arm_A_M1_jul-24-2026_15-36-28.log` (A+B NaN ep 3)
- Memory: `free-curv-codebook-collapse.md` (Task #138 发现追加)

---

## 8. 关联

- 前置: Task #135 (κ grad bug 诊断), Task #137 (R137 fix + 3 臂 retrain, 决定性坍缩诊断), Task #89 (原 free-curv 框架 + retro caveat)
- 关联: Task #84 baseline R@10=0.1020 utilization 100% (SOTA 上限), Task #88 c555 R@10=0.1051
- 后续: 无 (free-curv 方向永久放弃, C 方案 EMA 不启动)

---

result: Task #138 — completed. A+B 方案**未解决** free-curv codebook collapse. B 方案 (dead-code reset) 在 poincare ball 触发 NaN (ep 3) 不可用; A 方案 (geodesic kmeans_init) 独立跑 ep 50 best ckpt Stage 2 codebook = **64 unique SID / 99.35% collision / L0 54.7% L1 28.1% L2 14.5% utilization**, 仍坍缩 (vs baseline 8936 unique 100% util, vs Task #137 v1 12 unique). **κ_m 终值干净中间值** (+0.069/+0.043/+0.042, 非饱和) → 反证坍缩 ≠ κ 饱和, metric mismatch 修复不充分. **Task #89 free-curv product manifold RQ-VAE 框架终审 NO-GO, 永久放弃**; C 方案 (EMA) ROI < 成本不启动; SOTA 上限 = Task #84 baseline (固定 κ=-1 poincare, R@10=0.1020).
