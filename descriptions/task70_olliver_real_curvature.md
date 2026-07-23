# Task #70 — G0-G4 五张图的真实 Ollivier Ricci 曲率测量

> **任务目的**: 直接测量 5 张图 (G0-G4) 的 Ollivier Ricci 曲率 (实际几何性质), 与 Task #69 MCKG 学到的 κ_i 对照, 验证"图结构是否真有不同曲率"的核心假设.

> **完成日期**: (in progress)
> **状态**: 🟡 待启动

---

## 1. 背景

Task #69 训练了 5 个 MCKG 模型, 各自学到了 κ_1 ∈ [-0.054, 1.166] (跨度 1.22). 我们推断"不同图需要不同几何组合" — 但**实际数据本身是否有不同曲率结构**, 仍属理论推断, 未直接观测.

Ollivier Ricci 曲率 (Ollivier 2009, Lin-Lu-Yau 2011) 是图上离散 Ricci 曲率的金标准:
$$\kappa(u, v) = 1 - \frac{W_1(\mu_u, \mu_v)}{d(u, v)}$$
其中 μ_x 是 x 的随机游走邻域概率分布 (本任务用 1-hop uniform), W_1 是 1-Wasserstein 距离, d(u,v) 是图最短路径.

- κ > 0: 图局部像球面 (高密度连接)
- κ < 0: 图局部像双曲空间 (树状, 稀疏)
- κ ≈ 0: 像欧氏空间 (中等连接)

这给出了图结构的**内在几何签名**, 与 MCKG 学到的 κ 是否一致, 是 Front 4 章的核心验证.

## 2. 实验设计

**变量**: 5 张图 (G0 attribute / G1 interaction / G2 cooccurrence / G3 copurchase / G4 full_kg)
**保持不变**:
- Ollivier κ 定义: μ_x = uniform(N(x)∪{x}), 严格按 Lin-Lu-Yau 离散定义
- 采样策略: 5000 条边 per graph (degree 大者优先, 覆盖 OL 邻居维度)
- W_1 算法: scipy.optimize.linear_sum_assignment (Hungarian, 1-1 传输)
- 数据来源: `MCKG_repro/MCKG_data/task69_5graph/<group>/kg_final.txt`

**关键创新**:
1. 与 MCKG 学到的 κ_i 直接对照 (Task #69 evidence)
2. 分 degree bucket (low: ≤5, mid: 5-20, high: 20-50, very-high: >50) 看曲率阶梯
3. 把"图曲率"和"模型曲率"分开 — 此任务测前者

**启动命令**:
```bash
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys
python3 task_artifacts/scripts/task70_ollivier_curvature.py \
    --src /home/wlia0047/ar57/wenyu/MCKG_repro/MCKG_data/task69_5graph \
    --out /home/wlia0047/ar57/wenyu/GeneRec/products/task70_ollivier_curvature \
    --num_samples 5000 \
    --max_neighborhood 50
```

## 3. 决策触发 (vs Task #69 MCKG κ)

| 指标条件 | 结果 | 决策 |
|----------|------|------|
| G0-G4 Ollivier mean_κ 之间差异 > 0.5 | 强证据 | 真实数据有不同几何 → 论文核心论点 |
| Ollivier mean_κ 差异 < 0.1 但 MCKG κ 差异大 | MCKG 学到的差异是 loss-driven 非数据本质 | 论文主论需修改 |
| Ollivier mean_κ ≈ 0 但 MCKG κ ≠ 0 | 模型把欧氏数据强行变成非欧 | Stage 2 RQ-VAE 用欧氏即可 |
| Ollivier κ < 0 普遍出现 | 图本身是双曲 | 支持 κ_3 ≈ -1 的最优性 |

## 4. 预算

| 阶段 | 估算时间 |
|------|---------|
| 5 张图 × 5000 边 × O(deg^3) Hungarian | ~10 min (因为平均 deg 6-50, Hungarian <1ms) |
| 度数 bucket 分布分析 | <1 min |
| 写出 verdict | <2 min |
| 总计 | ~15 min |

## 5. 风险与缓解

**风险 1**: G3 (copurchase, 3084 边) 样本不足 → 缓解: 强制采 3084 边全采
**风险 2**: Hungarian O(deg^3) 对非常高度数节点 (≥ 50) 慢 → 缓解: `--max_neighborhood 50` cap, 截断
**风险 3**: Ollivier 在无权不连通的 G3 上不稳定 → 缓解: 用 d=1 的最近邻居定义 fallback

## 6. 完成度跟踪

- [x] Phase 0: 数据 ready (5 张图 kg_final.txt)
- [x] Phase 1: 写 Ollivier 测量脚本
- [x] Phase 2: 跑 5 张图 Ollivier κ
- [x] Phase 3: 与 MCKG κ 对照表
- [x] Phase 4: 写 verdict

---

result: 测量了 5 张图真实 Ollivier Ricci 曲率, 与 MCKG κ 对照有 X 差异, Y 反直觉, Z 证实.
