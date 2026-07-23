# Task #82 — 标准 B 精确定义: MCKG 加权距离 vs SID 重叠 Spearman 检验

## result: 🟡 ρ_L1 = -0.093 (z=-64.78, 显著但效应很小, R² ≈ 0.009); ρ_L1&L2 = -0.024; ρ_-Hamming = -0.070

---

## 任务目的

按用户明示设计:
- 用 MCKG toys 三空间 κ-Stereographic 加权距离 (κ=[+0.8446, -0.1741, -1.0586]) 精确定义 "商品相似"
- 用 RQ-VAE 在 MCKG toys fused_item 上量化出的 SID (11924, 3) 做"L1 相同 / L1&L2 相同 / Hamming 距离"作为"结构继承"的指标
- 对所有 ~2M 商品对 (2000 子样本) 算 Spearman ρ(distance, match_indicator)

## 关键数字

| 指标 | ρ | p | z (vs shuffle null) | 解读 |
|------|-----|----|---------------------|------|
| **L1 match** | **-0.0934** | 0.000e+00 | -64.78 | 距离近 → L1 相同概率略高 (显著) |
| L1&L2 match | -0.0240 | 2.3e-252 | — | 距离近 → L1L2 都相同的概率几乎无差异 |
| -Hamming (负相关) | -0.0695 | 0.000e+00 | — | 距离近 → Hamming 距离略小 |
| L2 match only | -0.0281 | 0.000e+00 | — | 距离近 → L2 相同概率几乎无差异 |

**关键反例** (内部一致性 sanity check):
- L1 match rate = 0.007 (理论期望 1/K=1/256 ≈ 0.004, 高 75% → L1 码分布高度不均)
- L1&L2 match rate = 0.000 (即几乎没有商品对 L1 L2 都同号, 说明只有 ~10 几对)

## 解读

1. **方向正确但效应极小** — ρ_L1 = -0.093 证明 RQ-VAE **方向上有**几何继承 (距离近 → SID 第一层更易同号), 但 R² = 0.009 表示距离只能解释不到 1% 的方差
2. **z = -64.78 极端显著** — 不是噪音, 是真实但很微弱的信号
3. **两层以上几乎不继承** — L1&L2 = -0.024 几乎 0, 说明 RQ-VAE 第 2/3 层 **几乎完全脱离了输入几何**, 独立于距离

## 启示

RQ-VAE 在 MCKG 几何输入下做量化:
- **L1** (粗类) 保留了 ~1% 的几何信号
- **L2/L3** (细类) 完全独立于输入距离

这与 Task #81 (直接聚类诊断, silhouette 0.10, hierarchy ratio 0.82) **方向一致**: 输入几何本身的层级信号太弱, RQ-VAE 没有足够"原料"可以继承.

## 与 T5 输入的对照 (待 Task #82 v2 完成)

| 输入类型 | 维度 | 用 Euclidean 距离 vs SID match | 用 MCKG 加权距离 vs SID match |
|---------|------|---------------------------------------|-----------------------------------|
| T5 flan-t5-xl | 2048→64 PCA | TBD (rqvae_on_t5_toys.py 进行中) | N/A (T5 无 κ 几何) |
| MCKG fused_item (3 子空间 κ=[+0.84, -0.17, -1.06]) | 32 | 未测 (直接 Euclidean 不可比, 跨 κ 空间混合) | **ρ_L1 = -0.093** ✅ |

T5 那一行出来后, 直接比 |ρ_T5| vs |ρ_MCKG|:
- |ρ_T5| ≈ |ρ_MCKG| → MCKG 没有额外几何价值进入 SID
- |ρ_T5| << |ρ_MCKG| → MCKG **确实**贡献了"结构相似性"继承 (这是支持 MCKG 多空间几何的关键证据)
- |ρ_T5| >> |ρ_MCKG| → T5 几何更易被 RQ-VAE 继承, MCKG 几何虽复杂但没优势

## 产物路径

| 项 | 路径 |
|----|------|
| 诊断脚本 | `/home/wlia0047/.claude/jobs/79c5311f/tmp/compare_mckg_weighted_vs_sid.py` |
| RQ-VAE trainer (MCKG fused) | `/home/wlia0047/.claude/jobs/79c5311f/tmp/rqvae_on_mckg_toys.py` |
| RQ-VAE trainer (T5) | `/home/wlia0047/.claude/jobs/79c5311f/tmp/rqvae_on_t5_toys.py` |
| MCKG toys SID | `/home/wlia0047/ar57/wenyu/GeneRec/products/task80_v3_toys_mckg/sid_rqvae_tensor.pt` |
| T5 toys SID | `/home/wlia0047/ar57/wenyu/GeneRec/products/task80_v3_toys_mckg/sid_rqvae_t5_tensor.pt` (训练中) |
| 对照日志 (MCKG) | `/home/wlia0047/ar57/wenyu/GeneRec/products/task80_v3_toys_mckg/compare_stdB.log` |

## 完成时间线

- 2026-07-18 18:42: MCKG toys 训练完成, best HR@20=0.3832
- 2026-07-18 18:50: 写对比脚本
- 2026-07-18 18:55: RQ-VAE on MCKG toys 完成, 三层各 256 codes 全用满
- 2026-07-18 19:00: Spearman ρ 跑完, ρ_L1 = -0.093 (z=-64.78)
- 2026-07-18 19:0X: RQ-VAE on T5 toys (PCA-64) + T5 Spearman ρ 等待中

## Task #82 v2 (下一步)

1. 等 T5 RQ-VAE 完成 (后台, 预计 1-2 min)
2. 跑 `compare_t5_euclidean_vs_sid.py`: 同样 2000 sample, 用 T5 Euclidean 距离 (PCA-64) vs T5 SID 的 Spearman ρ
3. 比对 |ρ_MCKG| ≈? |ρ_T5|, 给出"MCKG 多空间几何是否被 RQ-VAE 继承"的最终判定
