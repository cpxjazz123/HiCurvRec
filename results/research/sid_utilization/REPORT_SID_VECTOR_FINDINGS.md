# SID 质量 → Stage3 召回:73 个臂上的实测(修正版)

数据:`results/research/sid_utilization/metrics_corrected.json`、`gate_verdicts_corrected.json`(脚本 `vector_stats_all.py`、`vector_features.py`)。
口径:73 个「Stage3 有 `test_recall@10` + Stage2 `rqvae_best.pth` 与 `item_sids.json` 同时在盘上」的臂;全库 165 个 Stage3 臂中其余 92 个的 Stage2 产物已被清理。
SID 一律取**前三码**;文件里的第四项是导出时的碰撞计数器(768/769/…),不参与任何统计。

## 0. 修正记录(2026-11-19)

首版把编码器前向写成「只乘 `weight`、层间 ReLU」,漏掉了 `model/layers.py::MLP` 里每个 `nn.Linear` 的 **bias**。checkpoint 确实带 bias(`encoder.mlp.1.bias` 等,mean|b| 0.006–0.054),漏掉后 `z` 小了约 5.7 倍(`resample`:mean|z| 0.014 → 0.080),量化误差实际上变成了 `‖q‖` 的读数,而不是重建误差。

| 臂 | 旧 qerr | **新 qerr** | 旧 PR | **新 PR** |
|---|---|---|---|---|
| `resample` | 0.512 | **0.078** | 9.19 | **6.65** |
| `r48k2_rq` | 0.148 | **0.041** | 3.91 | **6.49** |
| `dul_rq` | 0.160 | **0.096** | 19.14 | **23.22** |

同时补上两处与真实前向一致的处理:SimVQ 的码表必须经 `embed_proj`(`SimVQLayer.get_code_embs`),EMA 码本键是不带 `.weight` 的 `embed`。混合几何无需特殊相加——`RQLayer.forward` 里 `quantized_x` 就是各层码字之和,双曲距离只决定分配(分配直接取自导出的 SID)。

## 1. 三个测量量

| 量 | 定义 | 依赖 |
|---|---|---|
| collision | `1 − 不同三码三元组数 / 商品数` | 只要 `item_sids.json` |
| quant_error | `mean ‖z − Σ码字‖`,`z` = 该臂编码器(含 bias)对全部商品的前向输出 | checkpoint + stage1 向量 |
| effective_rank (PR) | 协方差特征值的 `(Σλ)² / Σλ²` | 同上,复用同一次前向 |

## 2. 12 个高召回臂(>0.06)与其余 61 个

| 量 | 12 个 >0.06 | 其余 61 个 |
|---|---|---|
| collision | **0.030 – 0.087** | 0.010 – 0.998(中位 0.216) |
| quant_error | **0.041 – 0.090** | 0.007 – 16.68(中位 **0.049**) |
| effective_rank | **5.99 – 6.93** | 1.00 – 24.94(中位 11.85) |

注意:quant_error 两组的**中位数几乎相同**(好臂 0.078,其余 0.049),它是靠**窗口**而不是单调性起作用的;真正把两组拉开的是 effective_rank(好臂全部落在 6–7,其余中位 11.9)。

## 3. 门槛的最终形态:碰撞率硬卡,两个向量数只记录

`.035–.10` 与 `PR ≤ 8` 的窗口在 73 个臂上能做到 12/12,但按家族拆开就露馅了:

| 家族 | n | collision | quant_error | PR | 最高召回 |
|---|---|---|---|---|---|
| `r48` | 4 | 0.041–0.087 | 0.041–0.076 | 6.49–6.73 | 0.0648 |
| `resample` | 7 | 0.037–0.078 | 0.032–0.083 | 1.89–6.93 | 0.0636 |
| `crowd` | 2 | 0.030–0.041 | 0.067–0.090 | 5.99–6.50 | 0.0643 |
| `community` | 6 | 0.107–0.224 | 0.046–0.061 | 1.06–17.56 | 0.0558 |
| `mixed_rq` | 7 | 0.111–0.562 | 0.028–0.455 | 1.84–22.86 | 0.0579 |
| `successor` | 5 | 0.170–0.674 | 0.030–0.072 | 6.40–24.51 | 0.0544 |
| `curvature` | 6 | 0.010–0.613 | 0.007–0.163 | 3.46–12.70 | 0.0540 |

12 个高召回臂的 PR 全在 5.99–6.93,只是**行为采样家族的指纹**;`mixed_rq` 与 `successor` 的 quant_error 与好臂同区间,只是 PR/collision 不同。把向量窗口当硬门槛,等于禁止混合曲率整条线进入 Stage3(冻结基线 `TIGER_RQ-VAE` 的 collision 0.169 同样会被拒)。

因此最终形态:

| 条件 | 作用 | 数值 |
|---|---|---|
| **collision** | **硬门槛**(拒绝即不启动 Stage3) | **< 0.10** |
| quant_error | 记录 + `warnings` 告警 | 建议区间 0.035–0.10 |
| effective_rank | 记录 + `warnings` 告警 | 建议 ≤ 8 |
| checkpoint / 向量 / 码本宽度缺失 | 硬拒绝(无法检查即不放行) | — |

实现:`tools/research/stage2/sid_quality.py`;由 `tools/genrec-mcp/server.py::tool_stage3_start` 以 `genrec_env_v2` 子进程执行,判决里的 `quant_error`/`effective_rank`/`warnings` 一并写入启动上下文。反坍缩工具 `anticollapse/artifact.py::COLLISION_GATE` 与硬门槛保持一致(测试断言两侧相等)。

按此形态,73 个臂中 collision < 0.10 的 18 个全部放行(含 12 个高召回臂与 6 个低召回臂),碰撞率 ≥ 0.10 的 55 个全部拒绝。

## 5. 复现

```bash
/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 \
  results/research/sid_utilization/vector_stats_all.py     # PR/sep/qerr/collision per arm
/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 \
  results/research/sid_utilization/vector_features.py      # norm_z / rel_qerr / hubness
/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 -m tools.research.stage2.sid_quality <item_sids.json>
```
