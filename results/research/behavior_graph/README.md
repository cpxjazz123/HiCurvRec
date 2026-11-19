# 行为图契约(唯一一份无向图)

**决策(2026-11-19,owner)**:HiCurvRec 只维护这一份**无向**商品行为图。AF3 曲率、graphlet、结构保真度与 Stage2 结构监督实验统一从本目录读取,任何实验不得再自建一份。

方向不是当前研究的建模变量:三角形、开放楔形、社区内/桥接边与 AF3 都定义在无向图上。若将来需要研究有向转移,从 `results/stage0_build_parquet/train.parquet` 的**有序**序列重建即可,现在不额外存储。

## 目录内容

| 文件 | 内容 |
|---|---|
| `edges.parquet` | `source, target, weight`;每条**无向边一行**(`source < target`),`weight` = 两个方向的转移次数之和 |
| `graph.npz` | `nodes/rows/cols/af3/median/p90`(与 `af3_structure.load_graph` 兼容)+ **对称** CSR(`indptr/indices/data`)+ `degree`(每边两端度数的元组口径见下)+ `triangles`(每边三角数)+ `triangles_per_node` + `weight` + `symmetric=True` |
| `node_features.npy` | 24587×768 float32;第 k 行 = 商品 k 的 stage1 embedding |
| `metadata.json` | 构图规则、`policy` 块、三个源文件的 sha256、计数、AF3 摘要、三个输出文件的 sha256 |

## 图的定义

```
E = { {i_t, i_{t+1}} | (i_1..i_T) ∈ S_train }
```

- 每个用户的序列 = 该用户最长 `history` + 其 `target`;
- 只连**相邻**两项:A-B-C-D → A-B、B-C、C-D;
- 无向、去重;
- `af3 = κ_ij = 4 − d_i − d_j + 3·T_ij`,度数与三角数都取自本图。

## 当前实测

| 量 | 值 |
|---|---|
| 序列 | 57439 |
| 有边商品 | 24556 / 24587 |
| 无向边 | 315391 |
| 三角形 | 546438 |
| 度数 | 中位 13 / p90 50 / 最大 2220 |
| AF3 | 最小 −2511 / 中位 −128 / p90 −28 / 最大 0 |

## 读取方

| 脚本 | 用途 |
|---|---|
| `analysis_sid_recall/triangle_sid_fidelity.py` | 三角形 vs 度数匹配非三角形,SID 各级共享率 |
| `analysis_sid_recall/local_structure_fidelity.py` | 闭合(三角形 vs 开放楔形)与 AF3 分层 |

## 重建

```bash
/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 \
  analysis_sid_recall/build_behavior_graph.py
```

重建会覆盖本目录四个文件;`metadata.json` 里的 sha256 随之更新,下游可用它校验读到的是同一版图。
