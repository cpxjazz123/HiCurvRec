# Task #80 — Per-Layer κ Distortion Grid Search + 3-Way SID/TIGER 对比 (Idea 1)

> **任务目的**: 验证 Idea 1 — phonism RQ-VAE **逐层曲率** (per-layer κ) 是否优于单一全局曲率 (single κ) 与欧氏距离, 在 **Musical_Instruments** 数据集上做 3-way SID/TIGER 对比
>
> **完成日期**: (in progress)
> **状态**: 🟡 待启动 (数据集 2026-07-23 从 Toys 改为 Instruments, 因 Toys sentence-t5-768d.npy 缺失 + phonism Toys ckpt 已丢)

---

## 1. 背景

承接 Task #79 (phonism RQ-VAE δ_95 单调递减) 的核心发现:
- L0→L1: δ_95 **-60.3%** (变化最剧烈)
- L1→L2: δ_95 **-26.5%**
- L2→L3: δ_95 **-30.5%**

δ_95 反映"残差空间整体的树状偏离度", 逐层下降说明 RQ-VAE 越深层越树状。但 δ_95 只给**曲率该变负的相对强度排序**, 不是具体 κ 值。

**Idea 1 假设**: 不同层适合不同 κ 值, 因为每层的 residual 几何不同。
- L0 (raw encoded): 残差最分散, κ 应温和
- L1/L2/L3: 残差更聚类, κ 应更负

**预期产出**:
- C 组 (per-layer κ): 4 层各自独立选最优 κ
- B 组 (global κ): 4 层共享同一 κ
- A 组 (Euclidean): 现状 baseline (κ=0)

下游对比 Recall@5/10、NDCG@5/10:
- **C vs A** (曲率 vs 欧氏): 验证"曲率空间是否比欧氏好"
- **C vs B** (per-layer κ vs single κ): **Idea 1 真正要证明的**

**数据集变更理由** (2026-07-23):
- Toys phonism ckpt 丢失 (genrec 仓库已删, ckpt 一并消失)
- Toys sentence-t5-768d.npy 也不存在 (LLM-RecSys-ID/data/ 下只有 instruments/beauty/sports/yelp)
- Instruments phonism ckpt **存在** (Task #78/79 用过, `products/task78_phonism_rqvae_768d/...`), 立即可用
- Task #79 δ_95 数据本身就是 Instruments, 排序可直接复用

---

## 2. 实验设计

**变量**: 量化阶段的距离度量 (Euclidean / global-κ / per-layer-κ)
**保持不变**:
- RQ-VAE encoder/decoder/codebook: 完全冻结 (不重训)
- TIGER backbone (t5-small): 完全相同超参 + seed=42
- **Musical_Instruments 数据集** (24588 items, 5-core RecBole format)
- SID 序列长度: 3 (num_hierarchies=3, 对应 L0/L1/L2, L3 是 raw)
- 冻结 RQ-VAE: `products/task78_phonism_rqvae_768d/rqvae_ckpt/Jul-23-2026_16-23-23/best_loss_model.pth`
- 冻结 sentence-t5 embedding: `external/LLM-RecSys-ID/data/instruments/sentence-t5-768d.npy`

**Stage 0 (前置)**: ✅ **跳过** — phonism Instruments ckpt 已存在, 直接复用

**Stage 1 — 离线 κ 网格搜索** (不训练, ~5-10 min):
- 输入: 4 层 residual 点云 (从冻结 RQ-VAE 提取, 每层 24588 × 32 维)
- 候选 κ 范围 (按 δ_95 排序约束):
  - L0 温和: κ ∈ {0, -0.3, -0.5}
  - L1/L2/L3 偏负: κ ∈ {-0.5, -1, -1.5, -2}
- 指标: Sarkar-style stress / MDS 拟合误差 (不训练, 纯几何距离)
- 产出:
  - **C 组 (per-layer)**: L0/L1/L2/L3 各自 distortion 最小的 κ
  - **B 组 (global)**: 4 层 distortion 求和后, 整体最小的 κ

**Stage 2 — 3 组 SID 重生成** (~1 min × 3):
- 冻结 RQ-VAE encoder/decoder/codebook 完全不动
- 改量化阶段的 cost matrix:
  - A 组 (现状): Euclidean 距离
  - B 组 (global κ): κ-stereographic 距离 (全 4 层共享同一 κ)
  - C 组 (per-layer κ): κ-stereographic 距离 (4 层各用各自的 κ)
- SINKHORN 仍然用 (sk_epsilons=[0,0,0.003]), 只是 cost matrix 元素换了
- codebook 向量位置 **完全不变**, 不重新训练
- 输出: 3 组 SID tensors (N=24588 × 3 digit)

**Stage 3 — 3 次 TIGER 训练** (~1-2h × 3 = 3-6h):
- 3 组 SID 各训一次, 完全相同超参 + 数据切分
- backbone: t5-small (与 Task #78 一致)
- epoch=200, patience=10 early stop, seed=42
- 输出: 3 个 best ckpt + valid/test Recall/NDCG

**Stage 4 — 对比评估** (~30 min):
- 3 组 test R@5/R@10/NDCG@5/NDCG@10 对比表
- 核心: C vs A, C vs B

---

## 3. 决策触发 (vs baseline A = Euclidean)

| 指标条件 | ΔR@10 区间 | 决策 |
|----------|------------|------|
| **C vs A** ΔR@10 ≥ +5% | 大幅高于欧氏 | ✅ 曲率空间显著优于欧氏 |
| **C vs A** ΔR@10 ∈ [0, +5%) | 略高欧氏 | ⚠️ 曲率有帮助但不大, 需 follow-up |
| **C vs A** ΔR@10 < 0% | 低于欧氏 | ❌ 曲率不优于欧氏, Idea 1 弱否证 |
| **C vs B** ΔR@10 ≥ +2% | 大幅高于单一 κ | ✅ **Idea 1 成立**: per-layer κ 显著优于 global κ |
| **C vs B** ΔR@10 ∈ [0, +2%) | 略高于 single κ | ⚠️ **Idea 1 部分成立**, 需更多层数/更细 κ 网格 |
| **C vs B** ΔR@10 < 0% | 不优于 single κ | ❌ **Idea 1 否证**: per-layer 无收益 |

注: 用户指定宽松阈值 (Δ +2% / +5%)。即使 Idea 1 部分成立也算 positive signal。

---

## 4. 预算

| 阶段 | 估算时间 |
|------|---------|
| ~~Stage 0 训练 phonism Toys RQ-VAE~~ | ~~跳过~~ (Instruments ckpt 已有) |
| Stage 1 离线 distortion 网格搜索 | ~5-10 min (CPU 即可) |
| Stage 2 3 组 SID 生成 | ~3 min total on GPU 2 |
| Stage 3 3 次 TIGER 训练 | ~1-2h × 3 = 3-6h on GPU 2 |
| Stage 4 评估 + 写 verdict | ~30 min |
| **总计** | **~3-5h** (含 GPU 训练时间, Stage 0 跳过) |

---

## 5. 风险与缓解

**风险 1**: ~~Toys 原 phonism RQ-VAE checkpoint 丢失~~ → 已规避, 用 Instruments ckpt

**风险 2**: κ-stereographic 距离需满足 Sarkar 不等式约束 (‖x‖ < 1/√|κ|)
- 缓解: 对 Instruments 32d residual 点云, 选定 κ ∈ [-2, 0] 之前先验证 norm 上界 (phonism L3 residual norm ~0.025, 远小于 1/√2 ≈ 0.707, 满足)

**风险 3**: SINKHORN 用 κ-stereographic cost matrix 可能不收敛 (cost matrix 元素非负假设)
- 缓解: 先在 n=100 小子集验证 SINKHORN 收敛性, 再跑全量

**风险 4**: Instruments 24588 items 训练数据量大, 3 次 TIGER 训练时间 ~3-6h, 期间 task82 P5-CID 也需 GPU 1
- 缓解: Task #80 Stage 3 用 GPU 2 (task82 用 GPU 1, task83 wait daemon 等 task82), 不冲突

**风险 5**: δ_95 排序约束 κ 候选是经验性, 可能漏掉最优组合
- 缓解: 如果 C vs B Δ < 0%, 加跑粗网格 (4 层都用 {-0.5, -1, -1.5} 共 81 组合) 验证

---

## 6. 完成度跟踪

- [ ] Stage 0: 重训 phonism Toys RQ-VAE (ckpt 落盘)
- [ ] Stage 1: 离线 distortion 网格搜索 (JSON 落盘, 含 C/B 两组候选)
- [ ] Stage 2a: A 组 SID 生成 (Euclidean, baseline)
- [ ] Stage 2b: B 组 SID 生成 (global κ)
- [ ] Stage 2c: C 组 SID 生成 (per-layer κ)
- [ ] Stage 3a: A 组 TIGER 训练
- [ ] Stage 3b: B 组 TIGER 训练
- [ ] Stage 3c: C 组 TIGER 训练
- [ ] Stage 4: 3 组 R@5/10, NDCG@5/10 对比表
- [ ] 写 verdict: `verdicts/task80_per_layer_kappa_idea1_result.md`
- [ ] 更新 loop.md §16 (按 R8 清理)