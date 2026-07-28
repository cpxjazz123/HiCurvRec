# Task #140 — S³Rec NO-GO 修复:两阶段 pretrain + finetune

> **任务目的**: 关闭 Task #81 (S³Rec paper Table 2 row) ⛔ NO-GO → ✅ GO, 用 paper-aligned 两阶段训练机制在 Musical_Instruments 上闭环 R@10 评估.

> **完成日期**: in progress
> **状态**: 🟡 待启动

---

## 1. 背景

Task #81 (2026-07-23) S³Rec paper-aligned baseline 复现报 ⛔ NO-GO:
- yaml 默认 `train_stage='pretrain'` 导致训练走 self-supervised 阶段 (4 个辅助 loss: aap+mip+map+sp), 无 valid/early stopping, 无 best ckpt
- 训练 30 epoch 4h7min 后 kill PID 625025
- root cause: yaml 隐藏的 two-stage 状态机未切换

用户 2026-07-24 显式 ask "S³Rec NO-GO ... 这个可以解决吗" — 本任务 = 修复路径.

**Pre-task 状态**:
- `RecBole/recbole/model/sequential_recommender/s3rec.py` line 408-446 `calculate_loss` 分支支持 `train_stage='pretrain'` (4 SSL losses) 和 `'finetune'` (CE/BPR)
- s3rec.py line 109-122 finetune 通过 `load_state_dict` 从 `pre_model_path` 加载 pretrain ckpt
- `RecBole/recbole/properties/model/S3Rec.yaml` 含 `train_stage: 'pretrain'` + `pretrain_epochs: 500` + `save_step: 50` + `pre_model_path: ''` 完整字段
- paper S³Rec Instruments R@10=**0.0538**

## 2. 实验设计

**变量**: train_stage ∈ {pretrain, finetune} + loss_type=CE
**保持不变**:
- 数据集: Musical_Instruments (RecBole atomic file)
- 配置: musical_instruments_sequential_paper.yaml
- seed: 2025
- gpu_id: 0 (R7: GPU 0 空闲, 不抢 #136 ETEGRec GPU 2)

**Stage 1 (pretrain) 设计**:
- `train_stage='pretrain'`, `pretrain_epochs=50`, `save_step=10`
- 4 SSL losses (aap+mip+map+sp, yaml 默认权重 0.2/1.0/1.0/0.5)
- 50 epoch 成本预估: ~7h (经验值, paper 报告 loss plateau by ep 30)
- 产物: `products/task140/train/pretrain/S3Rec-Musical_Instruments-{ts}.pth` 每 10 epoch + final (R12 自动)
- R12 强制: RecBole built-in save_step 机制 + TrainerCheckpoint 自动覆盖

**Stage 2 (finetune) 设计**:
- `train_stage='finetune'`, `pre_model_path=<latest_pretrain_ckpt>`, `loss_type='CE'`, `epochs=200`, `stopping_step=20`
- 加载预训练 encoder weights → 在 CE 监督下微调推荐任务
- Early stopping on valid NDCG@10 (RecBole built-in best_metric tracking)
- 产物: `products/task140/train/finetune/S3Rec-Musical_Instruments-{ts}.pth` (best valid)

**启动命令**:
```bash
# Stage 1
python scripts/task140_s3rec_stage1_pretrain.py --epochs 50 --save_step 10 --seed 2025 --gpu_id 0

# Stage 2 (after Stage 1 ckpt available)
python scripts/task140_s3rec_stage2_finetune.py --epochs 200 --stopping_step 20 --loss_type CE --seed 2025 --gpu_id 0

# Stage 4 eval (RecBole built-in test.py, will follow finetune)
python scripts/task140_s3rec_test_eval.py  # 标准 RecBole 模式: load_data_and_model + test
```

## 3. 决策触发 (vs paper + vs Task #81 NO-GO)

| 观察条件 | 结果 | 决策 |
|---------|------|------|
| Stage 1 50 epoch loss 4.4M → ~2.7M by ep 5, plateau by ep 30 | 正常 self-supervised 收敛 | ✅ 继续 |
| Stage 2 finetune R@10 ≥ 0.045 (paper 0.0538 的 -17% 偏差内, 与 HG-Rec/-22.4% 同档) | 部分确认 paper RQ-VAE 在 sequential 的位置 | ✅ 论文 Section 5 闭环 baseline |
| Stage 2 finetune R@10 < 0.025 | pretrain ckpt 质量或 finetune 超参错 | ❌ kill + 调 pretrain_epochs=100/200 重跑 |
| Stage 2 finetune R@10 ≥ 0.06 (反超 paper) | 数据集偏差或过拟合 | ⚠️ 报告, 调 epochs=100 + multi-seed 不可 (per [[user-no-multiseed-override]]) |

## 4. 预算

| 阶段 | 估算时间 |
|------|---------|
| Stage 1 pretrain 50 epoch | ~7h |
| Stage 2 finetune 200 epoch (early stop) | ~30min |
| Stage 4 test eval | ~1min |
| **总计** | **~7.5h** |

## 5. 风险与缓解

**风险 1**: RecBole `run_recbole.py` argparse 未包含 `--train_stage`. 缓解: 用 programmatic launcher (task140_s3rec_stage1_pretrain.py) 直接调用 `recbole.quick_start.run()` 通过 `config_dict` 传 `train_stage` (config 优先级高于 yaml).

**风险 2**: Stage 1 ckpt file naming 不规律 (RecBole 内部 `${MODEL_NAME}-${DATASET}-${TIMESTAMP}.pth`), Stage 2 难以自动发现. 缓解: launcher 内 `find_latest_pretrain_ckpt()` 按 mtime 排序自动选最新 pretrain .pth.

**风险 3**: pretrain loss 实测 plateau by ep 30 但 yaml 默认 500 epoch, 50 epoch 可能不够. 缓解: 监控 loss 曲线, 若 ep 50 仍在下降则 Stage 2 前置跑 pretrain_epochs=100 (over-budget, 但不超 GPU 0).

**风险 4**: dual run conflict with Task #136 GPU 2. 缓解: 严格 GPU 0 (R7 confirmed, 0% util), 不动 #136.

## 6. 完成度跟踪

- [x] R9 audit (max=139, gap-free, next=140)
- [x] scripts/task140_s3rec_stage1_pretrain.py (py_compile PASS)
- [x] scripts/task140_s3rec_stage2_finetune.py (py_compile PASS)
- [x] description 落盘 (task140_s3rec_two_stage_fix.md)
- [ ] TaskList + loop.md §16 update (R8 + R10)
- [ ] Stage 1 pretrain launch GPU 0
- [ ] Stage 1 50 epoch 跑完 + loss plateau 验证
- [ ] Stage 2 finetune launch
- [ ] Stage 2 finetune 收敛 + early stop
- [ ] Stage 4 test eval (R@5/10, NDCG@5/10)
- [ ] 写 `verdicts/task140_s3rec_two_stage_fix_result.md` 含 `result:` 行
- [ ] 更新 TASKS_INDEX + CHANGELOG via dispatcher (Round 4)

## 7. R11.3 自主决策记录

- **pretrain_epochs=50** (vs yaml default 500): 推荐 by 陈敏雷 2024 经验值 + 不超 GPU 0 单卡 8h; 备选 paper-aligned 500 epoch (~70h). 决策: 50 epoch 足够, 因 4 个 SSL loss 浅层信号 plateau 快.
- **loss_type='CE'** (vs 'BPR'): paper-aligned + RecBole S3Rec original 推荐 CE for ranking. 备选 BPR (pairwise). 决策: CE, 复现 paper.
- **stopping_step=20** (vs 默认 10): paper-aligned (FDSA 同) 留更多 early-stop margin. 备选 默认 10. 决策: 20.
- **save_step=10** (vs yaml 默认 50): 让 Stage 2 有更多可选 ckpt + 监控 checkpoint 演变速率. 备选 yaml 默认 50 (5 个 ckpt 在 50 epoch 内). 决策: 10.
- **skip yaml 默认 `selected_features: ['class']`**: S3Rec 论文实验用 raw item id 不加 class feature. 备选 FDSA-style 加 class. 决策: 沿用 yaml 默认 (无 class feature), 让 S3Rec 跑 paper-aligned setting.
