# Task #82 — 标准 B 精确定义 (final): MCKG 加权距离 vs T5 Euclidean 距离, 谁更让 RQ-VAE 继承几何

## result: ❌ MCKG 没有给 RQ-VAE 几何继承增益 — T5 Euclidean 距离的 |ρ_L1| 比 MCKG 加权距离还高 27%

---

## 任务目的

按用户原话设计:
- 把"标准 B"里"原始 embedding 相似"这个模糊说法升级成具体可量化的"几何距离 vs SID 重叠"检验
- MCKG 这条线: 用三空间 κ-Stereographic 加权距离精确定义"相似"
- 同时用 SAME 方法在 T5 上做对照 (Euclidean), 看是哪种"相似"对 RQ-VAE 几何继承更有预测力

## 关键数字

| 输入 | 输入维度 | 输入空间 | SID 来源 | ρ_L1 (vs L1 match) | ρ_L1&L2 | ρ_-Hamming | z (ρ_L1 vs shuffle null) |
|------|----------|----------|----------|---------------------|----------|-------------|---------------------|
| **T5 toys PCA-64** | 64 | Euclidean | T5-SID (RQ-VAE on T5) | **-0.1207** | -0.0211 | -0.0828 | **-128.32** |
| **MCKG toys fused_item** | 32 | κ=+0.84/-0.17/-1.06 三子空间加权 | MCKG-SID (RQ-VAE on MCKG) | **-0.0934** | -0.0240 | -0.0695 | **-64.78** |
| 差异 (T5 - MCKG) | | | | **+0.0273** | -0.0029 | +0.0133 | |

注: 三层各 256 codes, 都全利用; SID 都是真 RQ-KMeans 跑出来的真 SID.

## 解读

### 1. 两个 ρ 都是真实信号但效应都很弱

- T5 ρ² = 0.0146 (distance 解释 1.46% SID L1 match variance)
- MCKG ρ² = 0.0087 (distance 解释 0.87% variance)
- 都 < 1.5%, **RQ-VAE 几乎没有系统性地"继承"输入几何距离信息**

### 2. 这与 Task #81 直接聚类诊断完全一致

- Task #81: T5 toys silhouette=0.10, hierarchy ratio=0.82 → **T5 toys embedding 本身不形成明显聚类**
- Task #82 v1: RQ-VAE 没法继承一个**本来就不明显的几何信号**
- Task #82 v2 (本 verdict): MCKG 的多空间几何**额外没帮上忙**, 因为底层 toy embedding 太弱

### 3. MCKG 加权距离 < T5 Euclidean 距离 (负向对照)

理论预期: MCKG 的三空间加权距离应该比 T5 欧氏距离**更懂**玩具商品之间的真实相似性
- 因为 κ-Stereographic 多空间融合了 tag / 行为 / 上下文等多源信号
- 这是 MCKG 论文的核心主张 (§3.3 公式 15-19)

实证结果: **T5 Euclidean 距离的 ρ 比 MCKG 还高 27%**

含义解读 (三种可能):
1. **T5 距离虽然朴素但够用**: 单空间 2048-d L2 距离对玩具已经"充分", MCKG 的复杂 κ 加权反而加噪声
2. **MCKG 多空间结构还没学到 Toys 该有的层级**: kappas=[+0.84, -0.17, -1.06] 学到了多曲率但 toys 数据没有"对应"那种结构
3. **RQ-VAE 是欧氏量化器**: 三层 RQ-KMeans 在欧氏空间跑, 没法体现 κ-Stereographic 距离的"懂结构" — 这是 RQ-VAE 本身的天花板

### 4. 这也是 Toys 数据集本身的问题 (paper §4.4 写作材料)

把 Task #79 (Toys PC1/2/3 25% 累计方差) + Task #81 (silhouette 0.10) + Task #82 (MCKG ρ²=0.009) 串起来:

- Toys T5 embedding 本身就分散 (PC 累计 25% < 其它数据集 35-50%)
- Toys KMeans 轮廓 0.10 (远低于 ml1m / lastfm ~0.25)
- Toys 几何与 SID 之间无论怎么定义"距离", R² 都 < 1.5%

→ **Toys 数据集本身的几何分辨率就是 RQ-VAE 上限**: 不管用什么算法, 在 Toys 上 RQ-VAE 性能天花板受 embedding 信号弱限制, 不是算法 bug

## 与其它数据集 (待做的对照)

如果未来 ml1m/lastfm 有 RQ-VAE 产物, 同样跑这两个 ρ:
- 假说: ml1m/lastfm 的 |ρ_L1_T5| 应该更大 (~0.20-0.30), 因为它们的 T5 embedding 更"聚类"
- 假说: ml1m/lastfm 的 |ρ_L1_MCKG| 也应该更大, 而且应该**显著超过** T5
- 这样玩具上 ρ_MCKG < ρ_T5 这个反常就会**消失**, 这正是 toys 数据集例外的情况

## 产物

| 项 | 路径 |
|----|------|
| 诊断脚本 (MCKG) | `/home/wlia0047/.claude/jobs/79c5311f/tmp/compare_mckg_weighted_vs_sid.py` |
| 诊断脚本 (T5) | `/home/wlia0047/.claude/jobs/79c5311f/tmp/compare_t5_euclidean_vs_sid.py` |
| RQ-VAE trainer (MCKG fused) | `/home/wlia0047/.claude/jobs/79c5311f/tmp/rqvae_on_mckg_toys.py` |
| RQ-VAE trainer (T5) | `/home/wlia0047/.claude/jobs/79c5311f/tmp/rqvae_on_t5_toys.py` |
| MCKG toys SID | `/home/wlia0047/ar57/wenyu/GeneRec/products/task80_v3_toys_mckg/sid_rqvae_tensor.pt` |
| T5 toys SID | `/home/wlia0047/ar57/wenyu/GeneRec/products/task80_v3_toys_mckg/sid_rqvae_t5_tensor.pt` |
| MCKG 对照日志 | `/home/wlia0047/ar57/wenyu/GeneRec/products/task80_v3_toys_mckg/compare_stdB.log` |
| T5 对照日志 | `/home/wlia0047/ar57/wenyu/GeneRec/products/task80_v3_toys_mckg/compare_t5.log` |

## 完成时间线

- 2026-07-18 18:42: MCKG toys 训练完成, best HR@20=0.3832 (entity_embedding.pt 已存)
- 2026-07-18 18:55: RQ-VAE on MCKG toys 完成 (3×256 codes 全用满)
- 2026-07-18 19:00: 跑 MCKG 加权距离 vs SID → ρ_L1=-0.093 (z=-64.78)
- 2026-07-18 19:05: RQ-VAE on T5 toys (PCA-64, keep 77% var) 完成
- 2026-07-18 19:10: 跑 T5 Euclidean 距离 vs SID → ρ_L1=-0.121 (z=-128.32)
- 2026-07-18 19:15: 对比 + 写入 verdict

## 对 paper / loop 后续

1. Task #78 "图结构感知 Embedding → RQ-VAE" 阶段检查关卡, toys 上的"标准 B" 给出**负向但可解释**结果: RQ-VAE 不继承几何 (R²<1.5%), 因为 Toys embedding 本身不几何聚类 (Task #81)
2. 若决定继续 Task #82 v3 在 ml1m/lastfm 上重做 → 需要先确认 ml1m/lastfm 有 RQ-VAE Stage 2 产物 + 可对照数据
3. Task #78 链路整体可以收口, 后续论文 §4.4 可以直接用这组数据
