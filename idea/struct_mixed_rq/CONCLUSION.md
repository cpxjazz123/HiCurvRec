# struct_mixed_rq — 结论：三层前缀 + 曲率回读已跑通全链路，但无增益

**判定：负结果。** `test_recall@10` 未过 0.065，且低于合法基线的较好一次。按 §19g 该臂只跑一次、不调参、不做消融、不换种子，因此就此封存。

## 1. 两次完整运行（均实测，非引用）

| 版本 | Stage2 | Stage3 `test_recall@10` |
|---|---|---|
| `TIGER_struct_mixed_rq`（末层前缀 + 旧 STE） | 3000 epoch | **0.05790490781524748** |
| `TIGER_struct_mixed_rq_ste_fix`（三层前缀 + 新 STE） | 3000 epoch，全局 batch 4096，seed 42，1038 s，SID `1839d1ce…` | **0.05459705078431031** |
| 基线 `TIGER_RQ-VAE` | — | 0.05553717857205035 / 0.04019916781281011 |
| 门槛 | — | 0.065 |

## 2. SID 质量指标与召回方向相反

| 指标 | 旧臂（R@10 0.0579） | 新臂（R@10 0.0546） | resample 参照 |
|---|---:|---:|---:|
| L1 使用 | **5 / 256** | **224 / 256** | 238 / 256 |
| L1 Hard ER | 1.69 | **7.55** | 16.14 |
| L2 前缀数 | 538 | **9289** | 15860 |
| L3 三元组数 | 10760 | **21862** | 23675 |
| L3 collision（item-weighted 三码） | 0.7265 | **0.1855** | — |

新臂把坍缩基本修好了（L1 5→224、ER 1.69→7.55、前缀 538→9289、碰撞 0.7265→0.1855），**召回反而降了 0.0033**。这说明"把 SID 的分布/碰撞指标修好"不是瓶颈本身；真正决定 recall 的是**哪些商品被分到一起**与下一跳行为是否对齐，而不是分层统计是否漂亮。

## 3. 两次尝试共同的结构性局限

两个臂都把曲率放在**同一侧**：`mixed_sq` 作为读回距离，加上一个作用在「已选码字累计位置」上的结构 stress。**Sinkhorn 的代价矩阵没有变**——选码仍按码本空间的欧氏距离 + 基线均衡规则决定。也就是说，曲率只影响"事后评分与回读"，从不影响"哪个商品拿到哪个码"。§19h 明确允许的是：**机制的打分进入 Sinkhorn 的代价矩阵内部，而均衡规则本身仍是基线的**。

## 4. 数据与凭据

- Stage2：`results/stage2_RQ-VAE/TIGER_struct_mixed_rq_ste_fix/`
  - `item_sids.json 1839d1cea74dbc0eb5789c88b3fda1d69ff87586c561ea7b1c19ca41a5ec61d3`
  - `rqvae_best.pth 9da6aa4baa3f12bf992b48d4f06d0d48755945d30d1b8a42335487f0749f0e06`
  - `logs/training_metrics.jsonl 9efc6e0d782116e9cedda66f75f6b1e68ea7aa93f3c9cc4fcb54d722215e9d57`（3004 行，3000 条 `train`）
  - `stage2_metrics/1839d1…json 0bbe9a16ce1e234a5cf03ad029a1fdde3bd0d6f5145dbc4394a27528703ece22`
- Stage3：`results/stage3_T5Train/TIGER_struct_mixed_rq_ste_fix/logs/Amazon_2023_Instruments/Nov-10-2026_12-53-45/`
  - `test_final.json 415507a2ff607066c76d2e1ec33750a2153f691f67cb8cb422900c1bae7d49a2`（`n_eval=57439`）
  - `HG_Rec.log cf3ff9362b1a9b091d89438781f074e94323ab279bc5141f6a927ccb352e504d`
  - `training_metrics.jsonl d40baf6db5d036b83b8c644b47d30ce6c7028ab169228afe3d8db050234fa2f8`
- R2：`r2:genrec/58/`，`results/research/issue139/r2_ledger.json` 条目 `58`（四件齐全，11 个文件，recall 键 6 个）。

## 5. 对下一轮的结论（写进新 idea）

曲率要影响召回，必须影响**选码本身**：让 `mixed_sq`（E16×H16 乘积几何）进入 Sinkhorn 的代价矩阵，并让行为信号在代价里带一个相对尺度的偏移量；保持基线的均衡分配规则不变（§19h），不做普通 `argmin` 替换。
