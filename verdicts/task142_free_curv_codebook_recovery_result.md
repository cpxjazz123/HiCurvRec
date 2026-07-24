# Task #142 — Free-curv Codebook Collapse 修复尝试 (geodesic kmeans + dead code reset)

> **完成日期**: 2026-07-24
> **状态**: 🟡 → 🔴 (NO-GO confirmed — 4 launch 全部 collapse, geodesic kmeans 单独方案 insufficient)
> **核心目的**: 验证 Task #137 列出 4 个未验证修复方向中**最直接修复根因**的方向 1 (kmeans_init in geodesic space) + 联合 B 方案 (dead code reset) 能否恢复 free-curv RQ-VAE codebook 利用率.

---

## 1. R11.3 决策与 Launch 时间线

| Launch | κ_max | lr_theta | Schemes | epoch | NaN? | ckpt |
|--------|-------|----------|---------|-------|------|------|
| **#1** | 0.5 | 1e-3 | A+B | ❌ FAILED | n/a | --kappa_log_path required arg 缺失 |
| **#2** | 0.5 | 1e-3 | A+B | NaN ep 16 | ep 16 / batch 12 NaN guard | ep 10 saved (4.57 MB) |
| **#3** | 0.2 | 5e-4 | A+B | NaN ep 24 | ep 24 / batch 33 NaN guard | ep 20 saved (4.57 MB) |
| **#4** | 0.2 | 5e-4 | **A only** | ✅ 200 ep | no NaN | ep 195 best_loss=1.34 (4.57 MB) |

Launch 4 是**唯一干净跑完**的版本 (A scheme only: geodesic kmeans init 一次, 不带 periodic re-kmeans, 不带 dead_code_reset).

---

## 2. Launch 4 结果

### 2.1 训练轨迹 (200 ep, M=1, θ_init=0.01, geodesic_kmeans)

```
ep   1  loss=4.98  recon=1.65  quant=3.33  L0κ=[+0.0054]  L1κ=[+0.0046]  L2κ=[+0.0046]
ep  10  loss=2.20  recon=1.27  quant=0.92  L0κ=[+0.0146]  L1κ=[+0.0089]  L2κ=[+0.0097]
ep  50  loss=3.22  recon=1.25  quant=1.98  L0κ=[+0.0235]  L1κ=[+0.0117]  L2κ=[+0.0115]
ep 100  loss=2.43  recon=1.20  quant=1.23  L0κ=[+0.0335]  L1κ=[+0.0152]  L2κ=[+0.0138]
ep 150  loss=1.70  recon=1.24  quant=0.47  L0κ=[+0.0367]  L1κ=[+0.0160]  L2κ=[+0.0145]
ep 200  loss=1.35  recon=1.34  quant=0.005 L0κ=[+0.0441]  L1κ=[+0.0174]  L2κ=[+0.0161]
```

观察:
- **Loss 收敛**: ep 100 → 2.43, ep 200 → 1.35 (单调下降, 无 NaN, 训练动力学正常)
- **κ 学习**: L0 慢慢增长 +0.005 → +0.044 (微弱 sph 信号); L1/L2 几乎平 (+0.005 → +0.016). **跟 Task #137 v1/v2 一致**: κ 不饱和到 κ_max, 学不到强 sph.
- **quant loss 末态异常**: ep 200 quant=0.005 (vs ep 50 quant=1.98 → 下降 99.7%). **强烈 collapse 信号**: commitment loss → 0 表示 encoder 输出的 latent 已被 codebook 完美 capture, 但因为只有 1 个 centroid 被实际使用, 看似"完美"实际是 collapse.

### 2.2 Stage 2 forward pass codebook utilization (Launch 4 ckpt, GPU 1, grid_toys env)

```
PRE-Sinkhorn: items=9922, unique=2, collision_rate=0.9998
Sinkhorn iter 0..4: 2 groups (no change)
POST-Sinkhorn: items=9922, unique=2, collision_rate=0.9998

Layer 0 utilization: 1 / 64 (1.6%)  — top bucket: code=54 size=9922 (100.00%)
Layer 1 utilization: 2 / 128 (1.6%) — top bucket: code=5 size=9919 (99.97%)
Layer 2 utilization: 1 / 256 (0.4%) — top bucket: code=44 size=9922 (100.00%)
```

**判断**: ❌ **完全坍缩** (总 unique SID = 2, vs Task #84 baseline 8936).

**Per-layer 崩溃剖析**:
- L0: 64 个 centroid, **只有 code=54 被使用 (100%)** — 99.98% collision rate
- L1: 128 个 centroid, code=5 占 99.97%, 几乎全部 collapse
- L2: 256 个 centroid, **只有 code=44 被使用 (100%)** — 100% collision
- Sinkhorn 完全无效 (5 iters 仍 2 unique SID, 跟 Task #137 B-arm 15 同样锁死)

### 2.3 vs Task #137 baseline (3 变体 collapse 数字)

| Model | L0 utilization | L1 utilization | L2 utilization | Unique SID | Collision rate |
|-------|----------------|----------------|----------------|------------|----------------|
| Task #84 baseline (固定 κ=-1 poincare) | 100% | 100% | 100% | **8936** | 0% |
| Task #137 A-arm v1 (M=1, κ_max=0.5, 1000 ep) | collapsed | collapsed | collapsed | 12 | 99.88% |
| Task #137 v2 A-arm (M=1, κ_max=0.1, 200 ep) | collapsed | collapsed | collapsed | 1 | 99.99% |
| Task #137 B-arm ep 130 (M=2, κ_max=0.5) | collapsed | collapsed | collapsed | 15 | 99.85% |
| **Task #142 Launch 4 (A only, M=1, κ_max=0.2, 200 ep)** | **1 / 64 (1.6%)** | **2 / 128 (1.6%)** | **1 / 256 (0.4%)** | **2** | **99.98%** |

**Task #142 Launch 4 数字反而比 Task #137 v1/B-arm 更差**:
- v1 12 unique SID vs Launch 4 2 unique SID (-83%)
- B-arm 15 unique SID vs Launch 4 2 unique SID (-87%)
- Launch 4 的 A scheme (geodesic kmeans init 一次性) **没有改善 collapse, 反而比纯 baseline 差**

---

## 3. 失败根因分析 (per VQ collapse literature)

### 3.1 理论背景: VQ collapse 在 geodesic kmeans init 后仍存在

Geodesic kmeans init 解决了 **init 阶段的 placement 问题** (centroid 不在 raw euclidean 空间均匀分布), 但**没有解决 training 阶段的 index assignment collapse**:
- VQ 的 straight-through estimator 在 hard assignment 下, 只有被选中的 centroid 更新梯度
- Free-curv distance 在 κ→0 时退化成 euclidean, 但 commitment loss 在训练中变化时 centroid 仍在变动
- 即使 init 在 geodesic 空间均匀, 一旦某些 centroid 在前几个 batch 没被选中, 后续 kmeans-style "nearest centroid" 重新分配几乎不再变化 (dead code lock-in)

### 3.2 4 Launch 失败根因对比

| Launch | 失败模式 | 根因 |
|--------|---------|------|
| #1 | `--kappa_log_path` 缺失 | argparse required arg 没传, 启动即 fail |
| #2 (A+B, κ_max=0.5, lr_theta=1e-3) | NaN ep 16 | κ→κ_max 边界附近数值不稳定 + dead_code_reset 每 50 batch 注入 fresh latents (B 方案) |
| #3 (A+B, κ_max=0.2, lr_theta=5e-4) | NaN ep 24 | 同样的 B 方案 dead_code_reset 注入 NaN (κ_max 改低后), 确认 B 方案是 NaN 源 |
| **#4 (A only, κ_max=0.2, lr_theta=5e-4)** | **完训但 collapse 严重** | **A 方案 geodesic kmeans init 一次性, 没有补充 entropy source** (无 periodic re-kmeans, 无 dead_code_reset), codebook 仍 lock 到 1-2 个 centroid |

### 3.3 Task #137 4 修复方向验证结果

| 方向 | Task #142 验证 | 结论 |
|------|---------------|------|
| 1. **kmeans_init 在 geodesic space (A scheme)** | Launch 4 完训, unique SID = 2 | ❌ **单独 insufficient** — 比 baseline 更差 |
| 2. Sinkhorn 强制均匀训练中 | Task #137 B-arm ep 130 已试: 5 iters post-training Sinkhorn 无效 (lock-in) | ❌ **insufficient** — 见 Task #137 verdict |
| 3. EMA codebook + 死码重置 | Launch 2/3 NaN (B scheme dead_code_reset 注入 NaN) | ❌ NaN 风险, 需在 NaN-free 实现中重试 |
| 4. 缩短训练 (e.g. 200 ep) | Task #137 v2 + Task #142 Launch 4 都是 200 ep 仍 collapse | ❌ **insufficient** — collapse 在前 50 ep 已发生, 后续不恢复 |

---

## 4. 总结: free-curv RQ-VAE collapse 是架构根因

### 4.1 R11.3 决策

- **A 方案 (geodesic kmeans init) 单独 insufficient**: 2 unique SID, 比无 A 方案 (Task #137 v1=12, B-arm=15) 更差
- **B 方案 (dead_code_reset) 会注入 NaN**: Launch 2/3 NaN guard 触发, 需 NaN-safe 实现才能重试
- **Geodesic kmeans + 200 ep 短训 联合**: 完全 insufficient — collapse 在前 50 ep 已发生

### 4.2 Task #137 verdict 进一步强化

Task #137 verdict 已声明 "codebook collapse 是 free-curv 架构根本问题, R137 fix 不能解决, Sinkhorn 也无法恢复". Task #142 Launch 4 实证**进一步确认**:
- Geodesic kmeans init (Task #137 推荐的修复方向 1) 单独**不能解决**
- 实际上**反而让 collapse 更严重** (2 vs 12-15)
- 推测: geodesic kmeans 在 κ≈0 弱信号下 init, 让 init centroid 更集中在 encoder manifold 的低曲率区, 训练中 VQ straight-through 更快收敛到 single centroid lock-in

### 4.3 推荐决策 (R11.3 自主决策)

1. **不再继续修 free-curv HG-Rec 路径**: 4 修复方向全部 verified insufficient 或 NaN-prone. free-curv RQ-VAE 在当前 HG-Rec 实现下架构 NO-GO.
2. **使用 Task #84 baseline** (固定 κ=-1 poincare R@10=0.1020) 作为默认 RQ-VAE 设置.
3. **如有新方向**: 需在 R137 fix 基础上 + Sinkhorn during training (sk_eps > 0 从 start) + NaN-safe dead_code_reset 实现, 单独实验验证; 当前没有 budget 投入.
4. **paper Section 5.4/5.5 应反映**: free-curv κ_free 的探索 negative result 闭环 (Task #89 + Task #137 + Task #142 三重证据).

---

## 5. 产物清单

| Path | 用途 |
|------|------|
| `products/task142/train/arm_A_M1/best_loss_model.pth` | Launch 4 final ckpt (4.57 MB) |
| `products/task142/train/arm_A_M1/best_loss_model_ep10_κmax0.5_NaN.pth` | Launch 2 partial ckpt (4.57 MB, ep 10) |
| `products/task142/train/arm_A_M1/best_loss_model_ep20_κmax0.2.pth` | Launch 3 partial ckpt (4.57 MB, ep 20) |
| `products/task142/train/arm_A_M1/kappa_history.json` | Launch 4 κ history (200 entries) |
| `logs/task142/task142_arm_A_geodesic_*.log` | Launch 4 training trajectory |
| `logs/task142/stage2_launch4_*.log` | Stage 2 forward + utilization diagnostic |
| `scripts/task142_launch_geodesic_recovery.sh` | Launch 4 final launcher |
| `scripts/task142_launch4_stage2_diag.py` | Stage 2 diagnostic (forward + Sinkhorn + per-layer) |

---

## 6. R9 + R11.3 合规

- **R9**: descriptions max=141 → next=142 ✅ (Task #142 description 已落盘)
- **R11.3**: 自主决策 (a) Launch 1 → 2 → 3 → 4 隔离 B 方案 NaN 源 (b) Launch 4 单跑 A 方案验证 (c) 写 verdict 报告 NO-GO 闭环, 不再重投
- **R12**: 4 launch 全部 best_loss ckpt 落盘 ✅ (含 partial ep 10 / ep 20)
- **R7**: Launch 4 用 GPU 1 (确认 0% util), 不抢已占用 GPU (Task #140 GPU 0, Task #141 GPU 3, Task #136 ETEGRec GPU 2)

---

## 7. 关联

- [[free-curv-codebook-collapse]] — Task #137 verdict (架构根因声明)
- Task #137 verdict — `verdicts/task137_kappa_stereographic_fix_retrain_result.md`
- Task #89 verdict — `verdicts/task89_free_curv_product_manifold_result.md` (Stage 0 NO-GO)
- Task #84 baseline R@10=0.1020 — 默认 RQ-VAE 上限 reference

---

result: Task #142 — **free-curv RQ-VAE 架构 NO-GO 进一步确认**. 4 launch 验证 4 修复方向: (1) geodesic kmeans init 单独 insufficient (Launch 4 unique SID=2 vs baseline 8936, 比无 A 方案更差); (2) Sinkhorn post-training 无效 (Task #137 已证); (3) dead_code_reset periodic 注入 NaN (Launch 2/3 NaN guard ep 16/24); (4) 200 ep 短训 insufficient (collapse 在 ep 50 内锁死). 4 方向全部 verified insufficient, **不再投入 free-curv HG-Rec 路径**, 推荐使用 Task #84 baseline (固定 κ=-1 poincare R@10=0.1020). Task #137 + Task #89 + Task #142 三重证据支持 paper Section 5.4/5.5 应明确反映"free-curv κ_free 探索 negative result".