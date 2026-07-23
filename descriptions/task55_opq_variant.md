# Task #55 — OPQ (Orthogonal Procrustes + RQ) 变体

> **任务目的**: 在 RQ 前对 embedding 做 Orthogonal Procrustes 旋转, 让 data 与码本坐标对齐, 减小 RQ 失真
> **完成日期**: (in progress)
> **状态**: 🟡 待启动

---

## 1. 背景

承接 Task #46 (G1 PASS, log1p 是 winner) + Task #53/162 主线. OPQ 是 RQ-VAE 文献中常见的预处理增强, 通过学习一个正交矩阵 R 使 ||e - R^T x||² 最小化, 然后 RQ 在 R^T x 上跑. 适合 norm 健康但 orientation 不对齐的 embedding (S4 AE / S5 T5).

## 2. 实验设计

**变量**: 在 RQ 前插入 OPQ 学习到的正交变换 R
**保持不变**:
- RQ 架构 (K=256, 3-layer, StandardRQ)
- 输入 embedding (S4 AE + log1p / MCKG + log1p)
- 训练 steps

**实现要点**:
```python
# OPQ 训练: 交替优化 R (固定 codebook) 和 codebook (固定 R)
R = nn.Parameter(torch.eye(dim, dim))  # 初始为单位矩阵
for step in range(2000):
    R_orth = project_orthogonal(R)  # SVD-based orthogonal projection
    x_rot = x @ R_orth
    # standard RQ loss on x_rot
    loss = rq_loss(x_rot, codebook)
    loss.backward()
    # 同时更新 R 和 codebook
```

## 3. 决策触发

| RQ recon 改进 | 判定 |
|--------------|------|
| OPQ_recon < 0.9 × RQ_recon | ✅ OPQ 显著有效, 加入 Stage 2 流水线 |
| 0.9-1.0 × | ⚠️ 边际改进, 可选 |
| ≥ 1.0 × | ❌ OPQ 无效 (orientation 已对齐), 跳过 |

## 4. 预算

~1 h (2000 steps × 4 源变体)

## 5. 风险与缓解

**风险 1**: OPQ 与 RQ 联合训练不稳定 → 缓解: 交替优化 (alt-opt) 而非端到端
**风险 2**: R 退化 (学到 permutation 而非 rotation) → 缓解: R 投影到 Stiefel manifold (SVD-based)
**风险 3**: 4 源变体需 GPU 显存 → 缓解: 单卡串行 (1 h 总时间可接受)

## 6. 完成度跟踪

- [ ] OPQ 模块实现 (含 Stiefel projection)
- [ ] S4 AE + log1p × OPQ × RQ 对照
- [ ] MCKG + log1p × OPQ × RQ 对照
- [ ] 写 verdict (OPQ 路线推荐与否)
