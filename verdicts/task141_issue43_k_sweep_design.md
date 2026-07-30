# Task #141 — Issue #43 K-sweep ablation 设计 (zero-GPU prep)

**日期**: 2026-07-30 23:25
**状态**: 📝 DESIGN — 等 owner 拍板启动 (R10/R11.5)
**Anchor**: Issue #43 HypPreEncoder K=256 当前 R@10=0.10425 ceiling

## 1. 目的

Issue #43 HypPreEncoder Stage 4 ceiling R@10=0.10425 (vs baseline 0.1020, +2.4%) 是 K=256 fixed recipe 闭环. K-sweep K ∈ {128, 192, 256, 320, 384} 在 HypPreEncoder 配方下未探索. 验证 K-sweep 邻居点是否能推 ceiling 进一步突破.

## 2. 现有 K-sweep 数据 (vanilla RQ-VAE, 不带 HypPre)

| K | R@10 | Δ vs baseline 0.1020 | 来源 |
|---|------|---------------------|------|
| 32 | 0.1006 | -1.4% | task278 |
| 64 | 0.1041 | +2.1% | task278 |
| 128 | 0.1027 | +0.6% | task278 |
| **256** | **0.1053** ⭐ | **+3.3%** | task278 (但 anchor 撤销, 真实值 = baseline) |
| 512 | 0.0824 | -19.2% | task279 |
| 1024 | 0.0847 | -16.9% | task279 (ckpt 不可信) |

**vanilla K-sweep 趋势**: K=256 是 trade-off 顶峰 (跟 K-sweep K=256 anchor 矛盾 — task194 anchor 撤销后, K=256 跟 baseline 持平 0.1020, 不再是 anchor). K-sweep 6-arm 在 vanilla recipe 上没有杠杆.

## 3. Issue #43 K-sweep 设计 (待 owner 拍板)

### 3.1 候选 K ∈ {128, 192, 256, 320, 384}

- **K=128**: vanilla K-sweep R@10=0.1027 (+0.6% vs baseline), HypPreEncoder 联合预期 ≥ 0.10425
- **K=192**: vanilla K-sweep 无数据, 插值 (K=128 ↔ K=256 中点)
- **K=256 (anchor)**: Issue #43 R@10=0.10425 ceiling ✅
- **K=320**: vanilla K-sweep 无数据, K=256 ↔ K=384 中点
- **K=384**: vanilla K-sweep task326 K=384 Gate 0 FAIL (L0=16.9% < 20%), HypPreEncoder 配方下未测试

### 3.2 Stage 1 训练成本估算

| K | Stage 1 时长 | GPU 占用 |
|---|------------|---------|
| 128 | ~2.5h (相比 K=256 快 ~30%) | 单 GPU 0/1/2/3 |
| 192 | ~2.8h (相比 K=256 快 ~10%) | 单 GPU |
| 256 (anchor) | ~3.2h (实测 16:44:46 → 15:53:10) | 单 GPU |
| 320 | ~3.5h (相比 K=256 慢 ~10%) | 单 GPU |
| 384 | ~3.8h (相比 K=256 慢 ~20%) | 单 GPU |

**总成本**: 5 K × ~3.2h 平均 = ~16h Stage 1. 5 GPU 串行需要 16h, 4 GPU 并行需要 ~4h (用 GPU 1/2/3 + 0 if 空闲).

**注意**: Issue #43 Stage 1 1000 epoch recipe 跟 task84 baseline 对齐 (batch_size=1024 + epochs=1000 + sk_eps=0.0). 每个 K 都重训 = 16h GPU.

### 3.3 决策阈值 (R10)

| R@10 实测 | 决策 |
|-----------|------|
| > 0.10425 (突破 Issue #43 ceiling) | 🟢 GO - 新 ceiling 确认 |
| 0.1020 < R@10 ≤ 0.10425 (-2.4% to 持平) | 🟡 NEUTRAL - 跟 K=256 anchor 持平, 没有杠杆 |
| ≤ 0.1020 (-baseline) | ❌ NO-GO - HypPreEncoder 在该 K 失效 |

### 3.4 R10/R11.5 ROI 评估

| 候选 | ROI | 风险 |
|------|-----|------|
| (a) 5 K 全做 K-sweep (~16h GPU) | 中 (5 K × Stage 1 重训 = 16h) | 高 (K=128/192/320 是未探索区, 跟 vanilla K-sweep 趋势不直接对应) |
| (b) 减半 K-sweep: {192, 256, 320} 3 K (~9.6h GPU) | 中-低 (覆盖 K=256 邻居 ±25%) | 中 (K=192/320 是 K-sweep 插值, 边际信号可能弱) |
| (c) 跟 Issue #30 r_l+s_l 联合: 复用 K=256 + 加 Issue #30 transforms (~6h, 跟 Task #135 重叠) | 高 (Issue #30+#43 联合理论上 +0.2-0.5pp over Issue #43 ceiling) | 高 (HRQVAE trainer patch, R11.4 critical) |
| (d) Owner 拍板接受 K=256 ceiling 0.10425 | 零 GPU (paper §5.x 转写) | 零 |

## 4. 推荐 (R11.5 + R10)

**候选 (d) 优先**: owner 拍板接受 K=256 ceiling 0.10425 (跟 Issue #30 0.1022 + Issue #320 0.1034 一起作为 project final ceiling), 立即转写 paper §5.x. 后续 K-sweep 留作 "future work".

**如果 owner 要推 ceiling**: 候选 (c) 联合 Issue #30+#43 (Task #135 已 pending) > 候选 (b) 3 K 减半 > 候选 (a) 5 K 全做.

## 5. 产物路径 (R12 强制)

每个 K 闭环落盘:
- Stage 1 ckpt: `products/task336_ksweep/K{K}/stage1/.../best_collision_model.pth`
- Stage 2 SID: `HG-Rec/dataset/Instruments/Instruments_t5_hrqvae_hyp_pre_K{K}.npy`
- Stage 3 ckpt: `products/task336_ksweep/K{K}/ckpt_hgrec/.../HG_Rec_best.pth`
- Stage 4 eval: `verdicts/task336_issue43_ksweep_K{K}_beam{20,50}.json`

Reproducibility triangle (Task #139 invariant C16): 每个 K 的 SHA256 hash 落盘 + eval script 复用 `scripts/task336_issue43_gate2b_stage4_eval.py`.

## 6. R11.3 决策记录

**自主选择**: 不启动 K-sweep, 默认 owner 拍板接受 K=256 ceiling. R10 backlog 候选 (a/b/c/d) 排序 = (d) > (c) > (b) > (a). 候选 (c) 跟 Task #135 重叠, owner 决策优先级最高.

**备选**: 如果 owner 明确指示推 ceiling, 启动候选 (c) 联合 Issue #30+#43 (R10 + R11.4 critical decision). Task #135 pending 已就绪.

result: Issue #43 K-sweep ablation 设计就绪. 候选 (a) 5 K 全做 ~16h GPU, 候选 (b) 3 K 减半 ~9.6h, 候选 (c) 联合 Issue #30+#43 (跟 Task #135 重叠, 优先), 候选 (d) 接受 ceiling 0.10425 转写 paper. R10 决策 = 候选 (d) 优先, 等 owner 拍板.