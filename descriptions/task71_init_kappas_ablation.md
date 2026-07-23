# Task #71 — MCKG init κ ablation: B (全 0) + C (随机 [-1, 0, +1])

> **任务目的**: 检验 [+1, 0, -1] 这个默认初始化是**模型偏好的对称破缺种子**还是**数据本质需求**. 用 G1 (interaction) 图跑 2 组对照 ablation, 报告 final κ 分布.

> **完成日期**: (in progress)
> **状态**: 🟡 待启动

---

## 1. 背景

mckg.py 用 `init_kappas=[+1.0, 0.0, -1.0]` 作为 3 个子空间的 κ 起点 (Task #70 用户问题发现). Task #70 测出 Toys 数据本质是双曲 (Ollivier mean_κ ∈ [-0.83, -0.65]), 但 MCKG 学到 κ1 ≈ +0.7 (反数据). 这引发疑问:

**如果不用 [+1, 0, -1] 作为起点, 学到的 κ 分布会不同吗?**
- 假设 1: 模型会因为这个起点学到 [+0.7, 0, -1.0] 模式, 但其实只需要球面 + 欧氏 + 双曲混合
- 假设 2: [+1, 0, -1] 是关键的对称破缺, 没它 MCKG 学不到明确分化

## 2. 实验设计

**变量**: init_kappas (3 个 κ 起点)
**保持不变**:
- 图: G1_interaction_only (Task #69 最佳 HR@20 = 0.578, 推荐核心对比)
- 训练: M=3, dim=64, n_hops=2, n_neighbors=8, 50 epochs, batch=1024, lr=1e-3
- kappa_clamp=2.0, seed=42, patience=20
- 数据集: toys, kg_final.txt (128773 user-item 边)

**3 组对比**:
| Label | init_kappas | 起点几何 | 假设检验 |
|-------|-----------|---------|----------|
| Baseline (Task #69 G1) | [+1.0, 0.0, -1.0] | 球面+欧氏+双曲 | 已知结果 (κ1=+0.121, κ2=-0.064, κ3=-1.059) |
| **B** | [0.0, 0.0, 0.0] | 全欧氏 | 起点无偏好 → 模型是否仍分化? |
| **C** | [random(-1, 0, +1)] ×3 | 随机 | 起点也无偏好, 但不同 run → κ 是否仍一致? |

**启动命令**:
```bash
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys

# B 组 (init_kappas=0,0,0)
python3 -u task71_ablation_init.py --label B_init000 --init_kappas 0.0,0.0,0.0 \
    --gpu 3 --save-dir products/task71_init_kappas_ablation

# C 组 (init_kappas=random,但 seed 固定以便可复现)
python3 -u task71_ablation_init.py --label C_random --init_kappas random \
    --gpu 3 --save-dir products/task71_init_kappas_ablation
```

## 3. 决策触发

| 观察 | 结果 | 决策 |
|------|------|------|
| B: κ 仍分化到 [+pos, 0, -neg] 模式 | 与 baseline 相似 | 起点不重要, 数据驱动 |
| B: κ 全部 ≈ 0 (欧氏不动点) | 与 baseline 强烈不同 | 起点决定结局 |
| C (run 1) vs C (run 2) 高度一致 | 梯度稳定 | 数据驱动 |
| C (run 1) vs C (run 2) 差异大 | 局部最优多 | 起点仍然有影响 |
| B 和 C 都得到 κ1 ≈ +0.1 (球面) | 与 baseline 同 pattern | [+1, 0, -1] 不重要 |

## 4. 预算

| 阶段 | 时间 |
|------|------|
| B 训练 (50 epochs × G1) | ~15 min |
| C 训练 (50 epochs × G1) | ~15 min |
| 检查 final κ + 比对 | <1 min |
| 总计 | ~30 min |

(G1 单组 Task #69 训练用了 ~12 min, 加上 ablation 总时间类似)

## 5. 风险与缓解

**风险 1**: B 起点全 0 后, κ 都卡在 0 (因为欧氏空间 tan_kappa=identity, 模型无梯度) → 缓解: 检查 final κ std; 若 B κ 全是 0 ± 0.1, 就是这个结果 (这本身是个发现)
**风险 2**: C 随机起点可能在 5 组 random seeds 内跑出不同分布 → 缓解: 用 seed=42 固定 (随机只 init 一次)

## 6. 完成度跟踪

- [x] Phase 0: G1 data ready
- [x] Phase 1: 写 ablation launcher
- [x] Phase 2: B 训练
- [x] Phase 3: C 训练
- [x] Phase 4: 对比 baseline, 写 verdict
