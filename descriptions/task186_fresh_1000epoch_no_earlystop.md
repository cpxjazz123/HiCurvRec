# Task #186 — 从 0 开始训练 1000 epoch, 关闭早停 (Phase 0.6 paper-aligned)

> **任务目的**: 用户 2026-07-25 18:00 指令——完全从 0 开始训练 1000 epoch, 不要用之前的 checkpoint 恢复, 不要加早停。看完整 1000 epoch 训练能跑出什么样的 checkpoint, 是否能超过 Task #181 R@10=0.1057。

> **完成日期**: 待启动
> **状态**: 🟡 待启动

---

## 1. 背景与动机

**Task #181 (Phase 0.6 paper-aligned, early stop at epoch ~92)**:
- Test R@10 = **0.1057** (+3.6% vs baseline 0.1020)
- Best val R@10=0.1262, NDCG@20=0.1003
- Early stop counter=9 触发, 训练在 epoch ~92 结束

**Task #185 (resume from Task #181 + disable early stop, 1000 epoch)**:
- 启动 17:52, 跑到 epoch 8 后被用户叫停
- val R@10=0.1062, NDCG@20=0.0825 (epoch 8) — 在恢复中
- **问题**: 优化器状态重置导致前几个 epoch val 指标明显下降, 用户认为这种方式"不干净"

**用户 2026-07-25 18:00 新指令**:
- 完全从 0 开始训练 1000 epoch
- **不要**用 Task #181 checkpoint 恢复
- **不要**加早停
- 看 1000 epoch 完整训练能跑出什么结果

**预期**:
- 完整 1000 epoch 训练可能让 val 指标爬到比 Task #181 (epoch 92) 更高的水平
- 没有早停 = 不会被 counter=9 提前终止 = 100% 跑完 budget
- best_ckpt 由 val NDCG@20 提升触发覆盖, 所以最终 ckpt 仍是 val 最优的

---

## 2. 实验设计

**变量**:
- 训练从 epoch 0 开始 (无 --resume_from)
- 1000 epoch 上限 (--num_epochs 1000)
- 无早停 (--early_stop 99999, --disable_early_stop)

**保持不变 (跟 Task #181 完全一致)**:
- Stage 2 codebook: `Instruments_t5_rqvae_paper_fix.npy` (paper-aligned)
- T5-small 5.5M (6 enc + 4 dec, d_model=128, d_ff=1024)
- β=1.0, loss = Poincaré dist² on raw + logmap0 (Phase 0.6 官方)
- sk_eps=[0,0,0] (Sinkhorn OFF), seed=42
- batch=1024 (train), infer_size=96 (val)
- lr=1e-4, Adam optimizer
- beam_size=20, topk=[5,10,20]

**启动命令**:
```bash
python3 /home/wlia0047/ar57/wenyu/GeneRec/scripts/task84_hgrec_stage3_train.py \
    --dataset_name Instruments \
    --batch_size 1024 \
    --infer_size 96 \
    --lr 1e-4 \
    --num_epochs 1000 \
    --num_layers 6 \
    --num_decoder_layers 4 \
    --d_model 128 \
    --d_ff 1024 \
    --num_heads 6 \
    --d_kv 64 \
    --vocab_size 1025 \
    --max_len 20 \
    --pad_token_id 0 \
    --eos_token_id 0 \
    --device cuda:0 \
    --mode train \
    --save_path /home/wlia0047/ar57/wenyu/GeneRec/products/task186/t5small_fresh_1000epoch/ \
    --log_path /home/wlia0047/ar57/wenyu/GeneRec/logs/ \
    --seed 42 \
    --early_stop 99999 \
    --disable_early_stop \
    --beam_size 20
```

### 决策触发 (vs HG-Rec baseline R@10=0.1020, vs Task #181 R@10=0.1057)

| val NDCG@20 (best) | test R@10 (Stage 4) | 决策 |
|-------------------|---------------------|------|
| > 0.1003 | > 0.1057 | ⭐ **GO** —— 1000 epoch 训练找到更好 ckpt, 超过 Task #181 |
| 0.0950-0.1003 | 0.1020-0.1057 | ✅ 跟 Task #181 持平 (disable early stop 没有显著增益) |
| 任何 | < 0.1020 | 🔴 1000 epoch 训练导致过拟合 |

---

## 3. 修改文件清单

| 文件 | 修改 |
|------|------|
| `scripts/task186_fresh_1000epoch.sh` (新) | Task #186 launcher, 无 --resume_from, 1000 epoch, disable early stop |
| `verdicts/task186_fresh_1000epoch_result.md` (新) | 训练完后写 verdict |

---

## 4. 预算

| 阶段 | 估算时间 |
|------|---------|
| 1000 epoch 训练 (56s/epoch) | ~15.6 h |
| Stage 4 test eval | ~5 min |
| 总计 | ~15.6 h |

**GPU 占用**: 1 张 L40S (GPU 0)

---

## 5. 风险与缓解

**风险 1**: 1000 epoch 训练时间过长 → 预估 15.6 h
**风险 2**: 长训过拟合 → val 指标在 epoch ~100 后开始下降, 但 best_ckpt 只在 val NDCG@20 提升时覆盖, 最终 ckpt 仍是 val 最优
**风险 3**: 训练崩溃 → R12 已强制 best_ckpt 每个提升 epoch 落盘, 重启可继续 resume (但用户指令"不要 resume", 所以需要决定是否重新从 0 开始)

---

## 6. 完成度跟踪

- [ ] 写 task186_fresh_1000epoch.sh launcher
- [ ] 启动 Task #186 训练 (GPU 0, PID 待分配)
- [ ] 监控 val 指标趋势 (epoch 0→1000)
- [ ] 1000 epoch 训练完成 (约 15.6 h)
- [ ] Stage 4 test eval (vs Task #181 R@10=0.1057)
- [ ] Verdict 写完 + §16 清理

---

## 7. 关键决策点

### 决策 1: 训练从 0 开始 vs resume?
**选了**: 从 0 开始 (用户指令明确)
**为什么**: 用户认为 resume + optimizer reset 不干净, 想要完整 1000 epoch 训练
**备选**: resume from Task #181 best ckpt (上一版 Task #185) — 已被用户否决

### 决策 2: 是否监控 val 指标, 在指标饱和时报告用户?
**选了**: 是. 定期报告 val R@10/NDCG@20, 标记是否超过 Task #181 早停点 (val R@10=0.1262)
**依据**: 用户想知道长训是否能突破 short-train 早停点的天花板

---

## 8. 修订记录

| 日期 | 修订内容 |
|------|---------|
| 2026-07-25 | 首次创建 (用户 2026-07-25 18:00 指令) |
