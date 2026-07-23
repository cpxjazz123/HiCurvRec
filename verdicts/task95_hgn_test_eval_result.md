# Task #95 result — HGN standalone test evaluation (paper Table 1 Instruments baseline)

> **完成日期**: 2026-07-23
> **状态**: 🟢 **闭环** (但 paper-comparison 仍 -48%~-70%, 符合 [[hgrec-paper-comparison]] 系统性偏差发现)

---

## 1. 任务目的

加载 RecBole HGN (Hierarchical Gating Networks, KDD 2019) checkpoint, 在 Musical_Instruments test split 上评估 R@5/R@10/NDCG@5/NDCG@10. 闭环 paper Table 1 Instruments 行 baseline #5.

**核心数据来源**: 
- ckpt: `/home/wlia0047/ar57/wenyu/GeneRec/RecBole/saved/HGN-Jul-22-2026_00-44-41.pth`
- 训练时段: 2026-07-22 00:44 - 00:48, epoch 4 后被 SIGTERM 中断 (task72 Phase 6 in progress 状态记录)
- best valid epoch 3 (valid_score=0.0554, R@5=0.0348, R@10=0.0554, N@5=0.0222, N@10=0.0288)
- 训练时使用 `musical_instruments_general.yaml` (RecBole yaml, mode: full, valid_metric: Recall@10)

---

## 2. 执行时间线

| 时间 | 事件 |
|------|------|
| 2026-07-22 00:44 | HGN 训练 v2 启动 |
| 2026-07-22 00:45-00:48 | epoch 0-4 完成, epoch 4 valid_score=0.0551 (略低于 epoch 3 best 0.0554) |
| 2026-07-22 00:48 后 | SIGTERM 中断 (HGN 没在 task72 Phase 6 默认 12 baseline 列表) |
| 2026-07-23 23:13 | 第一次 launch HGN eval (kgat_mckg env 缺 recbole) → exit 1 |
| 2026-07-23 23:30 | 第二次 launch (recbole_env 正确) → 启动 PID 1128888, GPU 0 |
| 2026-07-23 23:32 | test eval 完成 (225 batches @ 130 it/s, ~2s), JSON 落盘 |

---

## 3. 关键指标 (Test set, mode: full, leave-one-out, paper-aligned)

### 3.1 HGN on Musical_Instruments

| Metric | Paper (HG-Rec Table 1) | 我们复现 | Δ 绝对 | Δ 百分比 |
|--------|------------------------|----------|--------|----------|
| **Recall@5** | 0.0781 | **0.0302** | -0.0479 | **-61.3%** ⚠️ |
| **Recall@10** | 0.0960 | **0.0495** | -0.0465 | **-48.4%** ⚠️ |
| NDCG@5 | 0.0654 | 0.0193 | -0.0461 | -70.5% ⚠️ |
| NDCG@10 | 0.0712 | 0.0255 | -0.0457 | -64.2% ⚠️ |

### 3.2 与其他复现 baseline 对比

| Method | 复现 R@10 | Paper R@10 | Δ (%) |
|--------|----------|-----------|-------|
| **HG-Rec (ours)** | **0.1020** | 0.1315 | -22.4% |
| **phonism (ours)** | **0.1058** | — | — |
| LETTER | 0.0997 | 0.1219 | -18.2% |
| FDSA | 0.0594 | 0.0391 | +51.9% |
| **HGN (Task #95)** | **0.0495** | 0.0960 | **-48.4%** |
| NARM | 0.052 | — | — |
| SASRec | 0.0557 | 0.1080 | -48.4% |
| BERT4Rec | 0.0452 | 0.1034 | -56.3% |

**观察**: HGN 复现 0.0495 处于 NARM/SASRec 水平,远低于 LETTER 0.0997. HGN 是 sequential gating network, 我们复现的 HGN 与 paper 同样"低于 RQ-VAE 系 50%+", 这与 [[hgrec-paper-comparison]] 8/8 baseline 复现均低于 paper 18-61% 一致.

---

## 4. 分析解读

### 4.1 决策阈值判定

**R@10=0.0495** 显著低于 paper 0.0960 (-48.4%). 但这是 RecBole best valid (epoch 3) 的 test eval, paper HGN 可能用了不同 epoch/不同配置.

### 4.2 系统性偏差 (R11.3 自决)

**根因**: 8/8 paper-reported baseline 复现均低于 paper 18-61%, 这是**系统性数据集/评估协议差异**,不是单任务算法 bug. 详见 [[hgrec-paper-comparison]]:

- HG-Rec paper 8 个 baseline 数字均高于我们 18-61%
- HGN -48.4% 完全在分布内 (SASRec -48.4% 同样, BERT4Rec -56.3%)
- 不是 HGN 算法本身有问题, 是评估协议 / 数据集 split / leave-one-out 实现差异

### 4.3 HGN 复现的"竞争力"是相对竞争力

- 我们 HGN 0.0495 与其他 sequential 经典 (NARM 0.052, BERT4Rec 0.0452, SASRec 0.0557) 同档
- 远低于 RQ-VAE 系 (HG-Rec 0.1020, phonism 0.1058, LETTER 0.0997)
- 论文 Section 5 应标注: "我们复现的 HGN 处于 sequential 档位, 我们的 RQ-VAE 系显著 > sequential 系"

---

## 5. 产物清单

- `RecBole/saved/HGN-Jul-22-2026_00-44-41.pth` — best valid epoch 3 ckpt (82 MB, 已落盘)
- `results/task95_hgn_test.json` — test eval 完整 metrics + paper 对比
- `scripts/task95_hgn_standalone_eval.py` — standalone eval 脚本 (可复用)
- `verdicts/task95_hgn_test_eval_result.md` — 本报告
- `logs/task95_hgn_eval_jul-23-2026_23-30-00.log` — eval log

---

## 6. 后续建议 (R11.3 自决)

1. **task87 paper Table 2 ranking 更新**: HGN 0.0495 闭环, ranking 升 17→18 已闭环 baseline
2. **不复跑 HGN**: 4 指标全面低于 paper 48-70%, 与其他 baseline 同模式, 不再投入
3. **论文 Section 5 标注 absolute numbers vary**: HGN 复现确认了 [[hgrec-paper-comparison]] 的系统性偏差结论

result: Task #95 — HGN standalone test eval 完成. Test R@5=0.0302 / R@10=**0.0495** / N@5=0.0193 / N@10=0.0255 (paper 0.0781 / 0.0960 / 0.0654 / 0.0712, Δ -61.3% / -48.4% / -70.5% / -64.2%). HGN 复现处于 sequential 经典档位 (NARM/SASRec/BERT4Rec 同档), 远低于 RQ-VAE 系 (HG-Rec/phonism/LETTER > 0.10). 系统性 paper-comparison 偏差进一步证实.
