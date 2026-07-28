# Task #148 — S³Rec paper-aligned finetune (Stage 2 of Task #140 pretrain)

> **任务目的**: 承接 Task #140 S³Rec pretrain ckpt → 跑 finetune 推荐任务 (BPR loss), 验证落在 DECOR paper Table 2 S³Rec Instruments R@10=0.0538 ±10% [0.048, 0.059].

> **完成日期**: (in progress)
> **状态**: 🟡 待启动

---

## 1. 背景

Task #140 S³Rec NO-GO 根因: yaml `train_stage='pretrain'` 错配, 只跑了 self-supervised 4 个辅助 loss (masked item + segment + order + attribute), 无 valid/early stop/best ckpt, 训练 30 epoch 4h7min 后 kill.

承接 **DECOR paper Table 2 Instruments**:
- Caser R@10=0.0392
- GRU4Rec R@10=0.0537
- SASRec R@10=0.0530
- **S³Rec R@10=0.0538**
- FDSA R@10=0.0557
- P5-SID R@10=0.0438
- P5-CID R@10=0.0507
- TIGER R@10=0.0574
- LETTER R@10=0.0581
- CoST R@10=0.0570
- ETEGRec R@10=0.0609
- **DECOR R@10=0.0617**

DECOR paper §4.1.1 明确 "full-ranking evaluation over the entire candidate item set without sampling" (跟 RecBole `mode: full` 一致).

**核心假设 R1**: Task #140 pretrain ckpt (`S3Rec-Musical_Instruments-10.pth`) 已落盘并保留自监督学习信号, finetune 阶段能利用这些信号提升推荐性能.

## 2. 实验设计

**变量**: 仅 `train_stage` + `pre_model_path` (其余沿用 paper-aligned yaml)
**保持不变**:
- `musical_instruments_sequential_paper.yaml` (lr=0.001, MAX_ITEM_LIST_LENGTH=20, valid_metric=NDCG@10, stopping_step=20)
- Musical_Instruments 5-core 数据集
- seed=2025, epochs=200
- Task #140 pretrain ckpt 已落盘无需重训

**启动命令** (`scripts/task148_s3rec_finetune.py`):
```python
config_dict = {
    "train_stage": "finetune",
    "pre_model_path": "/home/wlia0047/ar57/wenyu/GeneRec/products/task140/train/pretrain/S3Rec-Musical_Instruments-10.pth",
    "learning_rate": 0.001,
    "weight_decay": 0.0,
    "stopping_step": 20,
    "epochs": 200, "seed": 2025, "gpu_id": 0,
    "checkpoint_dir": ".../products/task148/train/",
    "valid_metric": "NDCG@10",
}
run(model="S3Rec", dataset="Musical_Instruments", config_file_list=[yaml_path], config_dict=config_dict)
```

GPU 1 100% 空闲 (R7 验证, 0% util, 0 MiB).

## 3. 决策触发 (vs DECOR paper R@10=0.0538)

| 观察条件 | R@10 区间 | 决策 |
|---------|-----------|------|
| **test R@10 ∈ [0.048, 0.059]** | ±10% paper | ✅ **paper-aligned 闭环**, 替换 Task #140 NO-GO |
| test R@10 ∈ [0.040, 0.048] | -25% ~ -10% | ⚠️ 部分 fair, paper-aligned 略偏差, 接受 |
| test R@10 ∈ [0.030, 0.040] | -45% ~ -25% | ⚠️ pretrain 信号未充分 transfer, 需调查 |
| test R@10 < 0.030 | <-45% | ❌ paper S³Rec 在我们数据集上不能复现, 报结果 |

## 4. 预算

| 阶段 | 估算时间 |
|------|---------|
| Pretrain ckpt load + init | ~2 min |
| Finetune 训练 200 epoch (paper stopping_step=20) | ~2-3h |
| Test eval | ~1 min |
| **总计** | **~2-3h** |

## 5. 风险与缓解

**风险 1**: Task #140 pretrain ckpt 可能未充分收敛 (10 epoch vs paper 100+ epoch). 缓解: 接受作为 warm init, finetune 阶段可补偿.

**风险 2**: pretrain → finetune transfer gap (自监督 loss 跟推荐 BPR loss 优化目标不同). 缓解: lr=0.001 paper 默认 + stopping_step=20 patience.

**风险 3**: 修改 RecBole properties/model/S3Rec.yaml 会污染上游. 缓解: 不改 yaml, 仅在 launcher 层 config_dict override.

## 6. 完成度跟踪

- [x] R9 audit (max=147, next=148)
- [x] Task #148 description 写完
- [x] scripts/task148_s3rec_finetune.py 写完 + py_compile PASS
- [ ] GPU 1 launch finetune
- [ ] Stage 2 finetune 收敛 (early stop or 200 epoch)
- [ ] Test eval R@10
- [ ] 写 verdict `verdicts/task148_s3rec_finetune_result.md` vs DECOR paper R@10=0.0538
- [ ] 更新 §16 + TASKS_INDEX + CHANGELOG + README counts

## 7. 关联

- Task #140 S³Rec pretrain NO-GO (Stage 1 完成, Stage 2 没跑)
- DECOR paper Table 2 Instruments S³Rec R@10=0.0538
- Task #141 Caser paper-aligned fix (成功样本, 反例: Task #143 FDSA paper-aligned fix 三层错)
- Task #143 FDSA NO-GO 撤回 (paper source 错 + class feature 错 + protocol 误读)
- verdicts: `verdicts/task140_s3rec_no_go_pretrain_only_result.md` (NO-GO 历史)
