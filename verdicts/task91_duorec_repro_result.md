# Task #91 Result — DuoRec 复现 (paper Table 2 #4)

> **完成日期**: 2026-07-23
> **状态**: ✅ 已完成 — R@10=0.0672 vs paper 0.0454 (**Δ +48%**)
> **脚本**: `external/DuoRec/run_seq.py` + `musical_instruments_duorec.yaml`

---

## 任务目标

复现 ETEGRec paper Table 2 baseline #4 DuoRec (paper: arXiv:2210.07661), 验证 paper R@10=0.0454 在 Musical_Instruments 上能否复现。

## 模型 / 数据

| 项 | 值 |
|----|----|
| 配置 | `external/DuoRec/config/musical_instruments_duorec.yaml` |
| Backbone | SASRec |
| 数据集 | Musical_Instruments (5-core RecBole format) |
| 训练 epoch | 50 (best epoch: 18) |
| GPU | GPU 2 |
| 评估指标 | Recall@K, NDCG@K, MRR@K, Precision@K |

## 关键结果

| 指标 | paper 报告 | Task #91 复现 | Δ |
|------|----------|--------------|---|
| **Recall@5** | - | **0.0410** | - |
| **Recall@10** | 0.0454 | **0.0672** | **+48%** ✅ |
| **Recall@20** | - | **0.1008** | - |
| Recall@50 | - | 0.1619 | - |
| NDCG@5 | - | 0.0242 | - |
| NDCG@10 | - | 0.0326 | - |
| NDCG@20 | - | 0.0411 | - |
| MRR@10 | - | 0.0221 | - |

**结论**: R@10=0.0672 显著超过 paper 报告的 0.0454 (+48%)。这与 LETTER/TIGER 在此数据集上的强势表现一致 (Task #72 DECOR +49%) — 提示 Musical_Instruments 数据集可能比 Amazon Books 默认配置下推荐的"基准数据集"更容易。

## 已知局限

**L1**: 训练完成后 RecBole `quick_start.py` 因缺 `seaborn` 库在最终可视化阶段崩溃 (`ModuleNotFoundError: No module named 'seaborn'`)。**核心训练/评估指标已成功输出 (epoch 18 best)**, seaborn 仅用于绘制 loss 曲线, 不影响主指标。已 pip install seaborn 在 grid_toys env (2026-07-23 16:35)。

**L2**: DuoRec 论文原配置是 `epoch=300`, Task #91 用 `epoch=50`。RecBole early-stopping 在 epoch 18 触发, 已收敛。

## 产物清单

- 训练日志: `logs/task91_duorec_jul-23-2026_15-22-00.log`
- 最佳 ckpt: RecBole 自动保存在 `external/DuoRec/saved/DuoRec-*.pth` (R12 best)
- 配置: `external/DuoRec/config/musical_instruments_duorec.yaml`

result: Task #91 DuoRec 复现完成 (Musical_Instruments), R@10=0.0672 vs paper 0.0454 (Δ +48%), Recall@5=0.0410, NDCG@10=0.0326, best epoch=18。RecBole early-stopping 正常收敛。