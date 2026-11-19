# SID 质量 → Stage3 召回:73 个臂上的实测

数据:`results/research/sid_utilization/vector_features.json`(脚本 `vector_stats_all.py`、`vector_features.py`)。
口径:73 个「Stage3 有 `test_recall@10` + Stage2 `rqvae_best.pth` 与 `item_sids.json` 同时在盘上」的臂;全库 165 个 Stage3 臂中其余 92 个的 Stage2 产物已被清理。
SID 一律取**前三码**;文件里的第四项是导出时的碰撞计数器(768/769/…),不参与任何统计。

## 1. 三个测量量

| 量 | 定义 | 依赖 |
|---|---|---|
| collision | `1 − 不同三码三元组数 / 商品数` | 只要 `item_sids.json` |
| quant_error | `mean ‖z − Σ码字‖`,`z` = 该臂编码器对全部商品的前向输出 | checkpoint + stage1 向量 |
| effective_rank (PR) | 协方差特征值的 `(Σλ)² / Σλ²` | 同上,复用同一次前向 |

## 2. 单独看:只有 PR 与 collision 有一点信号

| 关系 | Pearson | Spearman |
|---|---|---|
| recall ~ collision | −0.37 | −0.26 |
| recall ~ quant_error | −0.38 | −0.06 |
| recall ~ PR | +0.08 | −0.03 |
| recall ~ `|log10(qerr/0.3)|` | −0.55 | −0.40 |

quant_error 与召回不是单调关系,而是**中间好、两头坏**;因此对它的「离最优值 0.3 的距离」求相关才有 −0.40。

## 3. 二维格子(中位召回)

| quant_error \ collision | < 0.05 | 0.05–0.20 | ≥ 0.20 |
|---|---|---|---|
| < 0.10 | 0.0427 (1) | 0.0544 (8) | 0.0557 (11) |
| 0.10–0.20 | 0.0375 (2) | 0.0555 (8) | 0.0549 (14) |
| 0.20–0.60 | **0.0634 (8)** | 0.0523 (2) | 0.0552 (6) |
| > 0.60 | 0.0631 (1) | 0.0361 (2) | 0.0472 (10) |

区域 `(collision<0.10, qerr 0.13–0.65)` 放行 14 个臂、12 个过 0.06(86 %);其余 59 个只有 1 个过线。

## 4. 两个假阳性是「照抄」形态

| 臂 | recall | collision | qerr | PR |
|---|---|---|---|---|
| `TIGER_curvature-embedding-rq` | 0.0490 | 0.010 | 0.198 | **14.08** |
| `TIGER_dul_rq` | 0.0260 | 0.016 | 0.160 | **19.14** |

12 个过线臂的 PR 是 3.57–11.47,两个假阳性是 14.08 / 19.14 —— **中间 11.47–14.08 无臂**。
机制:好臂把 768 维输入压成 4–11 个有效方向,三级码(≤24 bit)能覆盖;这两个臂保留了 14–19 个方向,即输入未经组织地被搬运穿过,三级码覆盖不了。

向量空间的其它候选(同样能分开这 2 个,但均只有 2 个反例支撑):
`norm_z`(好 0.052–0.105,坏 0.163/0.199)、`rel_qerr`(好 2.6–6.5,坏 0.81/1.21)、collision 下限(好 ≥0.030,坏 0.010/0.016)。

## 5. 最终门槛与实测

```
collision < 0.10          # 商品分得开
0.13 ≤ quant_error ≤ 0.65 # 码字有损但不背答案
effective_rank ≤ 12       # 编码器真的压缩过
```

| 规则 | 放行 | 其中 >0.06 | 精确率 | 漏掉的高召回臂 |
|---|---|---|---|---|
| collision<0.10 + qerr 0.13–0.65 | 14 | 12 | 86 % | 0 |
| **+ PR ≤ 12** | **12** | **12** | **100 %** | **0** |

放行的 12 个:0.0648 `r48k2_rq`、0.0647 `r48k1_rq`、0.0643 `crowd_posattract_w3`、0.0642 `behavior_residual_geometry_transport`、0.0637 `r48k2u_rq`、0.0636 `resample_fn_mask`、0.0635 `crowd_couser_w3`、0.0634 `resample`、0.0631 `BEHAVIOR_R48`、0.0630 `resample_s43`、0.0628 `resample_l1only`、0.0624 `resample_s44`。

实现在 `tools/research/stage2/sid_quality.py`,由 `tools/genrec-mcp/server.py::tool_stage3_start` 以 `genrec_env_v2` 子进程执行;测量失败一律拒绝。

## 6. 局限

- 三个阈值都由**同一批 73 个臂**、且 PR 那条只有 **2 个反例**支撑,存在拟合成分;`r48k2`(0.0648)与 0.065 门槛仍差 0.0002。
- 全部是相关性证据;真正检验只能在下一个新臂上盲判后跑 Stage3 对答案。
- 召回区间内的差距很小(0.0624–0.0648 对 0.055),这套门槛的作用是**筛掉明显坏的**,不是保证成功。

## 7. 复现

```bash
/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 \
  results/research/sid_utilization/vector_stats_all.py     # PR/sep/qerr/collision per arm
/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 \
  results/research/sid_utilization/vector_features.py      # norm_z / rel_qerr / hubness
/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 -m tools.research.stage2.sid_quality <item_sids.json>
```
