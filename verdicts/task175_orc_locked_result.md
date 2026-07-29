# Task #175 — κ-Stereographic + κ LOCKED at Ollivier ORC 实测值

> **任务目的**: 验证 hypothesis "如果直接把 κ 锁定在真实 ORC 测出来的值 (-0.65~-0.84) 上, 不学习, 跑一次, 能不能超过 HG-Rec baseline (0.1020) / phonism baseline (0.1058)"
> **完成日期**: 2026-07-25
> **状态**: ⛔ NO-GO

---

## 1. 关键结论

**Test Recall@10 = 0.0894, 比 HG-Rec baseline (Task #84, R@10=0.1020) 差 -12.4%, 比 phonism baseline (R@10=0.1058) 差 -15.5%. Hypothesis FALSIFIED.**

锁定 κ 在真实 ORC 实测值下, 下游不仅没超过任何 baseline, 反而显著恶化. 这是 9th κ-Stereographic variant 失败, 也是第一个明确验证 "**κ→ORC 实测值本身就不是 Musical_Instruments 数据需要的曲率**" 的实验.

---

## 2. 实验结果 (Musical_Instruments dataset, T5-mini 9.18M, κ-Stereographic)

| 指标 | Task #175 (κ LOCKED ORC) | HG-Rec baseline #84 (κ learned) | phonism baseline (vanilla RQ-VAE + SINKHORN) |
|------|-------------------------|---------------------------------|----------------------------------------------|
| **Recall@5** | 0.0722 | 0.0816 | - |
| **Recall@10** | **0.0894** | **0.1020** (-12.4% ⛔) | **0.1058** (-15.5% ⛔) |
| **Recall@20** | 0.1110 | 0.1279 | - |
| **NDCG@5** | 0.0611 | 0.0690 | - |
| **NDCG@10** | 0.0667 | 0.0755 | - |
| **NDCG@20** | 0.0722 | 0.0821 | - |

**两个 baseline 都是 NO-GO**: 跟 HG-Rec (-12.4%) 和 phonism (-15.5%) 都恶化.

完整 metrics: `verdicts/task175_orc_locked_metrics.json`
HG-Rec baseline: `verdicts/task84_hgrec_main_repro_instruments_result.md`

---

## 3. Timeline

| Stage | 时间 | 状态 |
|-------|------|------|
| Stage 1 (RQ-VAE 训练) | 2026-07-25 12:53 - 13:01 | ✅ best_ckpt 30.4 MB @ ep 0 |
| Stage 2 (Codebook inference) | 2026-07-25 13:09 | ✅ _t5_rqvae_orc_locked.npy (9922, 4) |
| Stage 3 (T5-mini 训练) | 2026-07-25 13:23 - 13:31 | ✅ HG_Rec_best.pth 落盘 |
| Stage 4 (Test eval) | 2026-07-25 13:31 | ✅ R@10 = 0.0894 |
| Verdict write | 2026-07-25 13:48 | ✅ |

---

## 4. 关键决策点 (R11.3 自主决策)

| 决策 | 选择 | 备选 | 理由 |
|------|------|------|------|
| κ LOCKED 取值 | Task #70 实测 [-0.653, -0.829, -0.840] (per-layer per-component) | paper 报告 c=1.0 等价 κ=-1.0 | 直接用 ORC 实测最真实, 不用理论值 |
| M (分量数) | 3 | 2 (HG-Rec 默认) | 跟 Task #145 baseline 一致, 保持 Stage 3/4 兼容性 |
| κ_max 初始化 | 不适用 (LOCKED) | - | 用户假设 "不学习" 强约束 |
| Stage 4 配置 | 复用 Task #174 v3 pattern (config dict + GenRecDataset positional args) | 老 pattern (kwargs) | v3 已验证可跑通, 避免 GenRecDataset.__init__() TypeError |
| 评估 beam_size | 20 (跟 #174/#176 同步) | 10 / 50 | 标准 HG-Rec 默认, 平衡速度与准确度 |

---

## 5. Analysis & Interpretation

### 5.1 Musical_Instruments 数据本质是欧式最优的又一旁证

- Task #71 init κ ablation 证明 Musical_Instruments 上纯欧式 (κ=0) 最优 (HR@20=0.845, +46% over baseline)
- Task #145 baseline 同样证明 κ→0.0~0.05 是 Musical_Instruments 数据最终学到的曲率
- Task #175 进一步证明: **即使把 κ 强制拉到 ORC 实测的双曲值 -0.65~-0.84, 下游也是恶化而非改善**. Musical_Instruments 数据的 ε-ORC 双曲性不是 "需要被 RQ-VAE 拟合的几何信号"

### 5.2 跟其他 κ-Stereo variants 的横向对比

| Variant | κ 状态 | Test R@10 | Δ vs baseline |
|---------|--------|-----------|----------------|
| #84 baseline (κ learned) | learned | 0.1058 | baseline |
| #165 v3 (κ learned) | learned | 0.0964 | -8.9% |
| **#175 (κ LOCKED ORC)** | **frozen at ORC** | **0.0894** | **-15.5%** |
| #169 (Sinkhorn) | learned | 0.0863 | -18.4% |

#175 的 -15.5% 不是最差, 但明确表明 "ORC 实测双曲性" 在 Musical_Instruments 上不是下游需要的信号. **ORC 计算的几何是数据局部图结构 (边邻域扩展性), 跟 RQ-VAE 的 latent quantization 几何需求不同维度**.

### 5.3 ORC κ 的几何含义

Ollivier Ricci Curvature 反映的是 "沿边走时邻居的 overlap 程度":
- κ<0: 邻居迅速发散 (双曲特征)
- κ>0: 邻居聚集 (球面特征)
- Musical_Instruments 上 κ≈-0.7 是因为 item-item graph 的局部结构是树状的 (Musical_Instruments 长尾稀疏)

但 RQ-VAE 的 latent quantization 是另一个问题: encoder 把 item embedding 投影到 latent, 然后用 codebook 离散化. 这个过程的几何约束是 "latent 空间 → codebook entry 的距离", 跟 "item-item 邻域 overlap" 不是同一回事.

**结论**: ORC 是 **图结构** 的几何, RQ-VAE 是 **embedding 空间** 的几何, 两者强行耦合失败是 expected outcome.

---

## 6. 产物清单

| 文件 | 大小 | 内容 |
|------|------|------|
| `products/task175/t5mini_orc_locked/jul-25-2026_13-30-47/Instruments/Jul-25-2026_13-31-17/HG_Rec_best.pth` | 30.4 MB | T5-mini 训练 best ckpt |
| `products/task175/t5mini_orc_locked/jul-25-2026_13-30-47/Instruments/Jul-25-2026_13-31-17/HG_Rec_best.json` | 28 B | best config |
| `verdicts/task175_orc_locked_metrics.json` | - | Stage 4 metrics |
| `verdicts/task175_orc_locked_result.md` | - | 本 verdict |

Stage 1 RQ-VAE best_ckpt 因 Stage 2 已跑完, 不再保留.

---

## 7. 后续建议

1. **9 个 κ-Stereo variants 全部 NO-GO, 应该考虑:**
   - (a) Musical_Instruments 数据根本不适合 κ-Stereographic, 切换数据集 (Beauty/Sports) 重新验证?
   - (b) 上游 src/ 框架的 RQ-VAE 实现有问题, 改用 GRID 原版 src/train.py 重跑?
   - (c) κ-Stereographic 是错误的几何假设, 改用其他 manifold (e.g., spherical, Euclidean+Riemannian adaptive)?

2. **Task #176 + #177 位置依赖 β(x) 实验**: 已启动, 关注是否能绕过 #175 #165 等失败模式. 初步 Stage 1 信号: best ckpt 在 ep 15-31 (κ≈0 时期), 后期 κ 推到双曲极端 + quant loss 崩溃. 跟 #175 类似的 "欧式最优" 模式.

3. **不再开新的 κ-Stereo 变体**: 在 Musical_Instruments 数据上穷尽 κ-Stereo variants 已达 9 个, 边际价值接近 0.

---

## 8. R8 §16 cleanup

- ✅ 已从 loop.md §16 删除 #175 活跃任务行
- ✅ verdict 文件保留在 verdicts/task175_orc_locked_result.md (本文件)
result: Task #175 — κ-Stereographic + κ LOCKED at Ollivier ORC 实测值
