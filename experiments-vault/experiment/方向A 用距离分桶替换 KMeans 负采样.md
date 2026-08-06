---
type: experiment
category: "experiment"
zh: "方向A 用距离分桶替换 KMeans 负采样"
direction: "方向A"
status: "NO_GO"
topic: "hyp_routes"
created: 2026-08-05
tags:
  - status-no_go
  - topic-hyp-routes
  - experiment
---

# 方向A 用距离分桶替换 KMeans 负采样

> **情况**: [[稳定化 κ 路径全部放弃方法|稳定化 κ 路径全部放弃方法]] (NO_GO)
> **主题**: [[Hyp 路线方法 (per-item radius + 自由 κ)|Hyp 路线方法 (per-item radius + 自由 κ)]] (hyp_routes)

## 用什么方法

方向A 用 cosine 距离分位区间负采样 + false-neg 排除 (NO-GO)

## 方法描述

**方向 A 做了什么**: 把 Hyp 路线最佳配方的负采样从 KMeans cluster 换成 cosine 距离分位区间 +top-20 false-neg exclusion，单变量设计，test R@10 = 0.0968 (-0.0056)。

**方法描述**:
- **L0** (P85-P100, far, 跨粗簇): 距离区间 [0.190, 0.388]
- **L1** (P40-P75, mid, 类别边界): 距离区间 [0.148, 0.178]
- **L2** (P10-P40, near, hard 但排除 top-20 false-neg): 距离区间 [0.117, 0.148]
- 每 anchor 池大小 = 128，排除 false-neg 20

**单变量设计成功**: 仅替换 neg_sampler，其他 (REC_LAYER_W/REL_STRUCT/自由 κ/stage3/4) 维持 Hyp 路线最佳配方

**失败原因**: 距离分桶 (连续信号) 比 KMeans 簇分桶 (稀疏信号) 更激进，驱动 κ 学到比最佳配方高 3-10x 的极端值 (final κ = [0.106, 0.244, 0.403])，T5 端在 test 集 R@10 = 0.0968 (-0.0056) 9 次确认中最低，valid/test ratio = 0.81。

**结论**: 整体收线，收口。

## 关键指标

- **test_r10**: 0.0968
- **delta_r10**: -0.0056
- **ndcg20**: 0.0804
- **valid_r10**: 0.12

## 相关实验

- [[EXPERIMENT_GUIDE|Vault 入口]]
