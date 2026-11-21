# 现有 relation 目标能否作为结构监督？—— 数值审计

**问题**：不重新生成 Teacher。先检查现有 128 维 `relation`（C_residual GNN 的商品级状态）是否
(1) 稳定、(2) 比纯语义 768 维多出真实结构信息、(3) 能被只看 768 维的离散 SID 保留。

**产物**：`audit_relation.json`（全部数值 + 输入 sha256），`audit_relation.py`（只读，不训练），
`../tests/test_relation_audit.py`（11 个度量自检）。

## 0. 身份与来源

| 项 | 值 |
|---|---|
| relation 目标 | `teacher/structure/v3/structure_targets_v3.npz`，sha256 `7af2d165…`，(24587,128)，已 L2 归一 |
| 生成路径 | `C_residual_edge_main_seed226.pth` → `variant.encode(features, adjacency)`，需图 |
| 复核 | 重算 seed226 状态与表内 relation 余弦 = **1.000**；seed227 = 0.071（见 §1） |
| `teacher_structure_v3.pth` | 是**不含图的 768→目标 carrier**，meta 指向 seed226 checkpoint（sha 一致）；graph/split sha 与契约一致 |
| relation 覆盖 | 24094 / 24587 商品（`coverage_neighbour`）；own-orbit 覆盖仅 12266 |
| 测试对 | 46586 个 held-out 边对，其中两端都覆盖 46057 |

## 1. 稳定性：可旋转对齐，但近邻集不可复现

| 指标（seed226 vs seed227） | 值 |
|---|---|
| 原始逐商品余弦 | 0.071（中位 0.060）——对照学习空间允许整体旋转，此值无意义 |
| Procrustes 对齐后余弦 | **0.962**（中位 0.967） |
| top-10 近邻 Jaccard | **0.270**（随机 ≈ 0.0004） |
| 各 seed 自身 raw-cosine 边 AUC | 0.496 / 0.499（均≈随机） |

读法：两个 seed 学到的是**同一几何的旋转**（0.962），但用这个几何做 10 近邻，只有 27% 重合。
也就是说逐点坐标稳定，**局部近邻结构不稳定**——任何 kNN/相似度配对目标会把这 73% 的差异当标签。

## 2. 是否比语义多出结构信息：在边判别上没有

| 打分方式（46586 held-out 边对，AUC） | AUC | 95% bootstrap |
|---|---|---|
| `cos(relation)` | **0.496** | 0.493–0.499 |
| `cos(semantics 768)` | 0.503 | 0.500–0.506 |
| `cos(residual_after_ridge)`（去掉 768 可线性预测部分） | 0.500 | 0.498–0.503 |
| `cos(neighbour_relation)` | 0.498 | — |
| **Teacher 的 pair head（C_residual）** | **0.747** | 见既有报告，本次复算一致 |
| **Teacher 的 pair head（A_embedding，无图）** | 0.711 | 同上 |

kNN 与训练图的重合（2000 探针，precision@10，随机基线=平均度/24586≈0.032%）：

| 空间 | precision@10 |
|---|---|
| relation | 2.39% |
| neighbour_relation | 2.77% |
| semantics 768 | 2.42% |
| residual after ridge | 0.45% |

768→relation 的线性可预测性：训练内 R² **0.753**，留出 4819 商品 **0.650**。

**结论**：现有 `relation` 的**余弦几何**在 held-out 边上与语义同水平（≈随机），去掉语义可解释部分后
（residual）仍为随机。0.747 的结构优势存在于**用边标签训练过的 pair head**里，不在 embedding 几何里；
C 相对 A 的 +0.036 AUC 是唯一测到的结构增量，且它只能通过该 head 读取。

## 3. 只看 768 维的离散 SID 能否保留它

| 臂 | prefix-length AUC | relation-top10 同首码率 | semantics-top10 同首码率 | 首码随机命中率 |
|---|---|---|---|---|
| `TIGER_RQ-VAE` | 0.535 | 34.2% | 73.3% | 1.09% |
| `TIGER_RQ-VAE__canonical` | 0.535 | 35.3% | 75.2% | 1.19% |
| `TIGER_mixed_rq_F0` | 0.557 | 49.0% | 80.7% | 3.05% |

现有 SID 把**语义近邻**大量收进同一首码（73–81%，随机 1–3%），但只收进 34–49% 的
relation-cosine 近邻。这与 §2 一致：relation 近邻里有一半以上不是语义近邻，而那部分不指向真实边，
因此“SID 没保留它”并不构成损失。prefix-length 对 held-out 边有微弱方向（0.535–0.557），说明
SID 前缀已带一点结构信息，但其量级远小于语义分组。

## 4. 对“用 GNN 结构 Embedding 监督 RQ-VAE”的裁决

- **不可用**：把 `cos(relation)`（或 neighbour_relation）当 KNN/相似度目标。它近邻不稳定（27%）、
  与语义不可分（AUC 0.496 vs 0.503）、残差为随机（0.500）。这样监督等于喂噪声。
- **可用但必须换目标**：Teacher 的 **pair head 分数**才是唯一超过语义的东西（0.747 vs 0.711）。
  若走用户的第二条路（匹配商品对相似度），目标应取冻结 head 在采样对上的分数，而不是 relation 余弦。
- **前提检查已通过**：65% 的 relation 方差可由 768 线性预测，说明 SID 的输入容量足够承载被蒸馏的
  Teacher 分数；瓶颈不在 768 维输入，而在“被匹配的对象是否是结构性的”。
- **若要“GNN 自身几何就是结构”**，需要以**余弦打分的边目标**重训 GNN（当前 C 的 head 是 MLP 读取
  拼接对，几何并未被约束），然后**用本审计同一套指标复检**；验收线建议：
  relation-cosine 边 AUC 的 bootstrap p05 > semantics AUC 的 p95，
  且 residual-after-ridge 的 p05 > 0.5。

## 复现

```
cd results/research/directed_graphlet_teacher
/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 relation/audit_relation.py   # ~45 s
/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 -m pytest tests/test_relation_audit.py -q
```
