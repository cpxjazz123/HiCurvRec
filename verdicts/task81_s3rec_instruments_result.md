# Task #81 Result — S³Rec RecBole 训练 (paper Table 2 baseline #8)

> **完成日期**: 2026-07-23
> **状态**: 🟢 **修复重启后训练中 (PID 625025, GPU 1, epoch 0 step 10/1327)**
> **下游任务**: paper Table 2 baseline #8 (S³Rec) 在 Musical_Instruments 数据集上

---

## 1. 重要修正 (2026-07-23 用户反馈)

**用户反馈原文**:
> "S³Rec 如果有问题, 怎么会被放在 DECOR.md 里跟主方法比较作为 baseline, 还使用同一个数据集呢?"

**我之前的错误判定** (verdict 之前的版本):
- 报告 S³Rec "数据限制 skip", 因为 `Field [class] not defined in dataset`
- 这导致我跳过了 S³Rec 训练

**根因** (2026-07-23 修复发现):
- `data/recbole/Musical_Instruments/Musical_Instruments.item` 文件**本来就有**, 含
  `class:token_seq` 字段 (24588 items × class label)
- **缺失的是 yaml 配置**: `musical_instruments_sequential_paper.yaml` 漏了
  `load_col.item: [item_id, class]` 这一行 (在 FDSA 用的 `musical_instruments.yaml` 有这行)
- 这是**配置错误**, 不是数据结构问题
- 修复: 在 `musical_instruments_sequential_paper.yaml` line 23 后添加:
  ```yaml
  load_col:
    inter: [user_id, item_id, timestamp]
    item: [item_id, class]    # FDSA/S3Rec 需要 class 做 feature embedding (R-fix 2026-07-23)
  ```

**结论**: 我之前的 "数据结构限制 skip" 判定**完全错误**. S³Rec 应该可以训练, 且 DECOR paper
Table 2 baseline 列表里明确包含 S³Rec, 同数据集 (Musical_Instruments 5-core) 完全合理.

## 2. 训练配置

| 项 | 值 |
|----|----|
| 框架 | RecBole |
| 模型 | S³Rec (Self-Supervised Sequential Recommendation) |
| 数据集 | Musical_Instruments (5-core, 24588 items) |
| Item features | `class:token_seq` (449 unique categories) |
| GPU | L40S (sm_89) GPU 1 (cuda:1) |
| 总 epoch | 200 (best valid + stopping_step=20) |
| pretrain_epochs | 500 (S³Rec 内部两阶段: pretrain + finetune) |
| 训练启动 | 2026-07-23 18:58 (PID 625025) |
| Test result | 训练完成后输出 |

## 3. S³Rec 模型结构 (启动 log 确认)

```
Trainable parameters: 1,720,448
FLOPs: 1,986,224.0

S3Rec(
  (item_embedding): Embedding(24589, 64, padding_idx=0)
  (position_embedding): Embedding(20, 64)
  (feature_embedding): Embedding(449, 64, padding_idx=0)  # class field 加载成功
  (trm_encoder): TransformerEncoder(2 layers, 2 heads)
  (aap_norm/mip_norm/map_norm/sp_norm): Linear(64, 64)  # 4 个 SSL 任务
)
```

**关键确认**: `feature_embedding: Embedding(449, 64)` — class 字段有 **449 unique tokens**,
模型成功用 class 字段做 feature embedding. 之前因 yaml 漏配置 `load_col.item` 而报
`Field [class] not defined in dataset` 错误.

## 4. 当前训练状态 (2026-07-23 18:58)

```
Train epoch 0: step 10/1327, 2.12 it/s, GPU RAM: 0.39 G / 44.39 G
```

- 1 epoch 1327 step (按 batch_size=256, 339k interactions → 1327 batches)
- 当前 2.12 it/s, 1 epoch ≈ 10 min
- 200 epoch ≈ 33 h (过夜训练)
- S³Rec 内部还有 500 epoch pretrain → 实际总时长会更长

## 5. 与 paper Table 2 baseline #8 (S³Rec) 对比

| 数据集 | Paper hit@10 | 复现 hit@10 | Δ |
|--------|--------------|-------------|----|
| Amazon Beauty | 0.0644 | (无 Beauty 复现) | — |
| Amazon Toys | 0.0627 | (无 Toys 复现) | — |
| Amazon Sports | 0.0501 | (无 Sports 复现) | — |
| **Musical_Instruments** | **未报** | **训练中** | new reference |

## 6. 关键时间线

| 时间 | 事件 |
|------|------|
| 2026-07-23 16:36 | S³Rec 启动 (PID ~ 未捕获), 因 yaml 缺 load_col.item 失败 |
| 2026-07-23 18:30 | S³Rec 重启 (PID 601727), 同样 yaml 错误失败 |
| 2026-07-23 18:48 | 用户质疑: "S³Rec 怎么会在 DECOR paper 跟主方法比较" |
| 2026-07-23 18:58 | 修复 yaml (加 `item: [item_id, class]`), 启动成功 (PID 625025) |
| 2026-07-23 ~19:30 | S³Rec 训练中 (PID 625025, GPU 1, epoch 0 step 10) |

## 7. 产物清单

- 修复后的 yaml: `/home/wlia0047/ar57/wenyu/GeneRec/RecBole/musical_instruments_sequential_paper.yaml`
  (line 24 新增 `item: [item_id, class]`)
- 训练启动 log: `logs/task81_s3rec_v2_jul-23-2026_19-30-00.log`
- 训练 PID: `products/task81/_TRAINING_PID` = 625025
- ckpt 输出目录: `RecBole/saved/S3Rec-Musical_Instruments-*-Jul-23-2026_19-30-00-*/

## 8. Pipeline 完整性

| Stage | 状态 | 产物 / verdict |
|-------|------|----------------|
| Stage 1 yaml 修复 | ✅ (2026-07-23) | 本任务 |
| Stage 2 模型训练 | 🟢 进行中 (PID 625025) | epoch 0 step 10 |
| Stage 3 test evaluation | ⏳ (RecBole 自动) | 待训练完成后 |
| Stage 4 paper Table 2 综合 | ⏳ (Task #87) | 待综合 |

result: Task #81 S³Rec RecBole 训练 (paper Table 2 baseline #8) 修复重启后进行中。
之前因 yaml 配置漏 `load_col.item: [item_id, class]` 导致 "Field [class] not defined in dataset"
错误而 skip, 经用户质疑后定位为配置错误 (非数据限制), 已修复并启动训练 (PID 625025, GPU 1,
epoch 0 step 10/1327, 2.12 it/s)。item feature 加载成功: `feature_embedding: Embedding(449, 64)`
(class field 有 449 unique tokens)。预估 ~33h 训练完成, 待完成后产出 test result 供 Task #87
paper Table 2 综合排名使用。