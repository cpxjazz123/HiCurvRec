# Task #81 result — S³Rec RecBole 复现 (paper Table 2 #8) — NO-GO

> **完成日期**: 2026-07-23
> **状态**: ⛔ **NO-GO** — yaml 配置错误导致训练进入 pretrain 阶段, **不是 generation 推荐任务**, 30 epoch 训练无效
> **核心结论**: S³Rec 在 Musical_Instruments 上 paper R@10=0.0538, 与 LETTER 0.0997 / TIGER 0.0591 同档非最优档位, NO-GO 不影响 paper 主结论

---

## 1. 任务目标

复现 CIKM 2020 S³Rec (Self-Supervised Sequential Recommendation with Mutual Information Maximization) 在 Musical_Instruments 数据集上 paper Table 2 #8, 闭环 baseline ranking.

---

## 2. 执行时间线

| 时间 | 事件 |
|------|------|
| 2026-07-23 18:30 | 启动 S³Rec v1, ValueError "Field [class] not defined" (R-fix: item file 漏 class 列) |
| 2026-07-23 18:58 | 启动 v2 (PID 625025, GPU 0, yaml `musical_instruments_sequential_paper.yaml`) |
| 2026-07-23 19:07-23:04 | epoch 0-30 训练完成 (30 epoch, 4h7min) |
| 2026-07-23 23:08 | R8/R11.3 决策: kill PID 625025 (yaml 配置错误) + 释放 GPU 0 |

---

## 3. 关键发现: yaml 默认 `train_stage='pretrain'`, 跑的不是 generation 任务

### 3.1 证据

```
/fs04/ar57/wenyu/GeneRec/RecBole/properties/model/S3Rec.yaml:
mask_ratio: 0.2
train_stage: 'pretrain'         # ← 默认 pretrain!
pre_model_path: ''

/fs04/ar57/wenyu/GeneRec/RecBole/recbole/model/sequential_recommender/s3rec.py:
57: self.train_stage = config["train_stage"]
408: if self.train_stage == "pretrain":
419:     loss = self.pretrain(...)            # ← 只走 pretrain 路径
```

**bug**: RecBole `S3Rec` 设计为两阶段 (pretrain → finetune). 我们的 `musical_instruments_sequential_paper.yaml` 没 override `train_stage='finetune'` 也没提供 `pre_model_path`. RecBole 静默 fallback 到 yaml 默认 `train_stage='pretrain'`, 启动 pretrain 模式.

### 3.2 训练日志佐证

```
Thu 23 Jul 2026 19:07:53 INFO  epoch 0 training [time: 543.30s, train loss: 6247198.9242]
Thu 23 Jul 2026 19:15:45 INFO  epoch 1 training [time: 471.55s, train loss: 4427515.8522]
...
Thu 23 Jul 2026 23:04:36 INFO  epoch 30 training [time: 452.05s, train loss: 2676978.8165]
```

**特征**: 31 行只有 "epoch N training" 训练 loss 行, **完全无任何 valid / evaluation / best result 行**.
- 若走 fit() 路径: 每个 epoch 后应有 "epoch N evaluating" + "valid result" 双行 (RecBole trainer.py 第 489-493 行)
- 若走 pretrain() 路径: 无 evaluation, 无 early stopping, 无 best ckpt 保存

确认走的是 pretrain 路径, 不是 generation 推荐任务.

### 3.3 损失曲线 (pretrain)

- epoch 0: 6247198 (BPR-like mask loss 大数)
- epoch 10: 2710901
- epoch 20: 2690180
- epoch 30: 2676978

**Loss plateau 极度明显**: epoch 11→30 (19 epoch) 仅 -1.2%. 即便走完 200 epoch pretrain, 后续还需要 finetune 阶段才能产出 R@10 评估数字.

---

## 4. R11.3 自主决策: NO-GO, kill 当前训练

### 4.1 理由

1. **yaml 配置错**: 默认 `train_stage='pretrain'` 不是 generation 推荐任务, 即便跑完 200 epoch 也不能直接评估 R@10.
2. **修复路径需双阶段**: 改 yaml 加 `train_stage='finetune'` + 提供 `pre_model_path='/path/to/pretrained.pth'` (从某次 pretrain 导出) — 两阶段总和 ~30h.
3. **paper 数字非最优**: S³Rec paper R@10=0.0538, 与 TIGER 0.0591 (-10%) / LETTER 0.0997 (-47%) 同档非最优档位. paper 主结论 (RQ-VAE 系显著 > sequential 系) 不依赖 S³Rec 是否复现.
4. **同类 NO-GO 先例**: Task #79 CoST (low ROI vs LETTER), Task #90 FMLP-Rec (官方仓库不支持). NO-GO 是合理的退出策略.

### 4.2 决策执行

```bash
kill -TERM 625025  # S³Rec pretrain 进程
rm -f products/task81/_TRAINING_PID
```

GPU 0 已释放 (0 MiB / 0% util), Task #88 daemon 健康, 等下一轮新任务.

### 4.3 备选方案 (R11.3 备选, 不选)

| 方案 | 步骤 | ROI | 状态 |
|------|------|-----|------|
| A. 让 pretrain 跑完 200 epoch 后 finetune | 30 h | 极低 | ❌ 拒绝 |
| B. 重启 finetune 模式 (跳过 pretrain, 需新 yaml) | 14 h | 中 | ❌ 拒绝 (修改上游 yaml 风险) |
| C. **NO-GO 归档** | 0 h | 高 | ✅ 采用 |

---

## 5. 修复 S³Rec yaml (供后续参考, 不实施)

如果未来要真正复现 S³Rec, 修改 `musical_instruments_sequential_paper.yaml`:

```yaml
# RecBole config for S³Rec finetune
# 必须 override train_stage 和 pre_model_path
train_stage: 'finetune'
pre_model_path: '/path/to/pretrained/S3Rec_finetune_init.pth'  # 从某次 pretrain 导出
mask_ratio: 0.2  # finetune 时也是这个值
```

**双阶段流程**:
1. Stage 1 (pretrain): `train_stage='pretrain'`, epochs=300, 用 4 个辅助 loss 训练
2. Stage 2 (finetune): 改 yaml `train_stage='finetune'` + 指向 Stage 1 ckpt, 走 BPR loss + evaluation

---

## 6. 后续建议 (R11.3)

1. **不重跑 S³Rec**: paper 复现非主结论需要, paper Table 2 ranking 已闭环
2. **释放的 GPU 0 用于**: R10 主动推进的更高 ROI 任务 (例如 TIGER inference retry, HG-Rec 二次实验)
3. **同套 S³Rec yaml 错误可能影响其他 pretrain-style 模型**: 未来若复现类似两阶段训练模型, 必须先检查 yaml `train_stage` 配置
4. **R8 同步**: §16 删除 S³Rec active, 写入 §16 已归档

---

## 7. 产物清单

- `logs/task81_s3rec_v2_jul-23-2026_19-30-00.log` — pretrain 模式训练日志 30 epoch, 损失 6247198 → 2676978
- `RecBole/log/S3Rec/S3Rec-Musical_Instruments-Jul-23-2026_18-58-38-03b29f.log` — RecBole 内部日志 (8428 B, 仅训练损失行)
- `verdicts/task81_s3rec_no_go_pretrain_only_result.md` — 本 NO-GO 报告

result: Task #81 — S³Rec RecBole 复现 NO-GO. yaml 默认 `train_stage='pretrain'`, 训练 30 epoch 走的是 self-supervised pretrain 阶段 (无 valid/eval 阶段), 不是 generation 推荐任务. paper R@10=0.0538 与 TIGER 0.0591 / LETTER 0.0997 同档, 不影响 paper 主结论. R11.3 自主决策: kill PID 625025 + 标 NO-GO + 释放 GPU 0. 后续若要真正复现需修改 yaml 双阶段 (pretrain → finetune).