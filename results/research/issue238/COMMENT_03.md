## 节点 3：实现完成、51 项测试通过，但**真实入口 smoke 阻塞**（issue 保持 OPEN）

证据：`results/research/issue238/VERDICT.md`

### 已完成

| 项 | 状态 |
|---|---|
| 图曲率目标 `K_G`（测地三元组、固定参考集、不可达≠0、分层） | 9 项测试 |
| 混合曲率距离 `D_q`（12/10/10 分块、exp/log、Σw=1、注入尺度） | 9 项测试 |
| 曲率保真 `L_curv`（Huber、冻结 s 与 δ、行为开关） | 6 项测试 |
| 训练前标定（早于任何训练，`created 2026-11-23T01:32:18Z`） | 191,984 三元组；s=3.3386；δ=1.2109 |
| 全部机制测试 | **51 passed** |
| 真实入口 smoke（C24） | **FAILED** |

### 阻塞点

冻结量化器的 Sinkhorn 守卫（`model/layers.py:83-91`，基线 sha256 `0d5eacf6…`）
在训练**第 2 步**拒绝退化代价矩阵：

```
model.forward -> rq -> vq_layer -> _indices -> center_distance
ValueError: Cannot center a constant distance matrix.
```

smoke 共失败 5 次，每次前进一步：设备索引 → 有界径向映射 → `acosh` 上界 →
标定改用 **L3 残差**（而非编码器输出）→ 监督路径绕开 Sinkhorn 用 argmin 硬量化。
第 5 次之后是**主前向自身**退化。

### 已排除的解释（都有实测）

- 学习率：压到冻结 warmup 首步（≈8e-7）仍在第 2 步失败；
- 数值饱和：换成无平坦区的有理映射后，scale ∈ {0.01,1,100}、输入放大 1000 倍仍有 std 1e6，无 NaN；
- 小批退化：n = 450/512/2048/4096 前向 span 6.6–7.1，无 NaN；
- 标定尺度：改用 L3 残差尺度后主前向第 0–1 步正常。

### 为什么这是框架阻塞而非实现缺陷

`center_distance` 是**基线自带**的语义，issue 要求「保持原均衡规则」且「只允许一个 L_curv」。
因此我不能改它、不能关 Sinkhorn（§19h 指定的机制位置）、也不能加第二个损失去绕过。
只要 `L_curv` 的梯度进入同一张图，第 2 步的该层代价矩阵就退化。

### 请您决定的最小放宽（推荐 A）

| 选项 | 内容 |
|---|---|
| **A（推荐）** | 允许 `center_distance` 在常数矩阵时退化为行内中心化（或该层回退 argmin），并把它记录为诊断 |
| B | 允许 `L_curv` 只在**冻结 argmin 量化**上监督（分配 detached，梯度只进编码器与码本） |
| C | 允许混合距离只用于 `L_curv`、不进 Sinkhorn 代价矩阵（需改 §19h，不推荐） |

### 遵守 issue 的两点

- **没有做任何额外重复训练**：Stage2 一次都没启动成功；
- **没有改判据**：47 条清单原样保留。

*（`results/research/issue238/COMMENT_03.md`；issue 保持 OPEN，等待 §3 的放宽决定）*
