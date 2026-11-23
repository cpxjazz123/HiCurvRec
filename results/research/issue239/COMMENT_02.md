## 节点 2（C7）：#238 阻塞根因 = 码本梯度非有限，不是尺度/分配退化

诊断脚本：`results/research/issue239/diagnose_238.py`
原始数据：`results/research/issue239/diagnose_238.json`

### 修复前的逐步证据（#238 配置：混合距离作为 Sinkhorn 代价）

| step | encoder 有限 | L3 残差有限 | 码本 unique | 混合代价矩阵 | k_q 有限 | **grad_total_norm** | 参数更新 |
|---|---|---|---|---|---|---|---|
| 0 | ✓ std 0.0154 | ✓ std 0.0055 | 256 | min 0.217 / std 0.702 / 行跨度 3.29 | ✓ | ✓ 1.48e-3 | ✓ 0.631 |
| 1 | ✓ std 0.0137 | ✓ std 0.0127 | 256 | min 0.381 / std 0.791 / 行跨度 3.60 | ✓ | **✗ NaN，3 个张量** | **NaN** |
| 2 | — | — | — | — | — | — | 抛 `Cannot center a constant distance matrix` |

非有限的张量**点名**：`rq.vq_layers.0.embed.weight`、`rq.vq_layers.1.embed.weight`、`rq.vq_layers.2.embed.weight`（三个码本）。
梯度是 **NaN 而非 Inf**（inf=0，nan=20/40/30），值本身始终有限。

### 结论

- **不是尺度退化**：混合矩阵的 min/std/行跨度在每一步都健康（std 0.70–0.79，行跨度 3.3–3.9）。
- **不是分配退化**：码本始终 256 个唯一码字，`k_q` 有限。
- **是梯度非有限**：第 1 步三个码本的梯度含 NaN → `optimizer.step()` 把 NaN 写进权重 →
  第 2 步编码器输出退化 → 代价矩阵才变成常数。**常数矩阵是下游症状，不是根因。**

### 两条 0/0 的来源（已修）

1. 球面块 `sqrt(‖u−v‖²)` 在两点重合时导数为 ∞（`d√d/dd → ∞`）；
2. 球映射 `cap·x/(‖x‖+s)` 在 `‖x‖=0` 处对 `‖x‖` 求导未定义（缺范数下界）。

修复：`sqrt` 内加 `1e-12` 下界、球映射恢复范数下界 `EPS`、`asin` 参数上界 `1−1e-4`。
修复后 `test_mixed_distance.py` 的零向量/重复点后向实测全部有限。

### 修复后的同一诊断（三次 step 全绿）

| step | grad_total_norm | 非有限张量 | 参数更新 |
|---|---|---|---|
| 0 | ✓ 1.48e-3 | 0 | 0.630 |
| 1 | ✓ 1.96e-3 | 0 | 0.507 |
| 2 | ✓ 1.40e-3 | 0 | 0.480 |

`failure: None`。**#238 的阻塞在根因层面已消除**；按本 issue 的设计（欧氏量化），该路径本就不会被触发，
但 `L_curv` 复用的混合距离需要这次修复。

*（`results/research/issue239/COMMENT_02.md`）*
