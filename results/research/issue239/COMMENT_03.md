## 节点 3：smoke PASS（C9/C10），正式 Stage2 已启动

### smoke 证据

```
SMOKE_PASS   passed=True  steps=6  launch_mode=single-gpu
all_step_norms_positive=True   dead_terms=[]   nonfinite_terms=[]
applied_grad_norms = [1.448e-3, 1.714e-3, 1.400e-3, 1.253e-3, 1.206e-3, 1.127e-3]
curv_weight = 5.3108e-6   recon_grad = 1.4474e-3   curv_grad = 13.6265
```

被探针命中的项（说明曲率项确实在计算图里，而不是挂着）：

| 项 | grad | shape |
|---|---|---|
| `train_rqvae.DISTANCE`（欧氏代价，未改） | 8.85e5 | [4096, 256] |
| `distances.mixed.distances`（仅用于 loss） | 9.63e5 | **[64, 128]** = 参考数 × 本步三元组 |
| `distances.mixed.component_sq` | 5.97e4 | [128] |
| `distances.mixed.pairwise_sq` | 2.26e4 | [128] |
| `train_rqvae.curvature_fidelity_loss` | 3.547 | — |

**C9 的三条都成立**：梯度有限且非零、无 `dead_terms`、无 `nonfinite_terms`；硬 q 前向与 STE 路径用的是
模型自身的 forward（不是 argmin 复制品），因此不存在软/硬分配不一致。

### 与 #238 的关键差别

| | #238 | #239 |
|---|---|---|
| 量化器代价 | 混合距离 | **欧氏**（`DISTANCE` 的 sha256 与基线一致，5 项测试锁定） |
| 常数矩阵守卫 | 被触发 | **不经过**（矩阵来自欧氏代价） |

### 正式训练

- 命令：`stage2_start`（MCP），树 `stage2_RQ-VAE/TIGER_curvloss_rq`
- 预算：3000 epochs、四卡、全局 batch 4096、seed 42（与冻结 TIGER 同）
- 启动后 2 分钟内到 epoch 528；预计约 10 分钟完成

训练结束后我会贴：最终碰撞率、产物路径、评估表（C12/C13），并按 §89 门槛决定 Stage3。

*（`results/research/issue239/COMMENT_03.md`）*
