# Task #139 — LETTER-TIGER paper-aligned baseline reproduction (DECOR Table 2)

> **任务目的**: 验证 DECOR paper §4.2 "we re-run the official open-sourced code under identical experimental conditions" — 拉 LETTER 官方仓库 (HonghuiBao2000/LETTER @ 8d0154e) 跑 LETTER-TIGER vanilla baseline, 看 paper R@10=0.0574 是否能复现 (vs 我们的 snap-research/GRID fork R@10=0.1029, +79.3% gap).

> **完成日期**: (in progress)
> **状态**: 🟡 训练待启动 (GPU 0 已分配合规, TASK #137 C-arm ep 840+ / ETEGRec in GPU 2)

---

## 1. 背景

承接 [Task #136 verdict](../verdicts/task136_decor_5_baselines_table2_result.md):
- DECOR paper Table 2 Instruments 列报告 TIGER R@10=0.0574
- 我们的 snap-research/GRID fork TIGER R@10=0.1029 (seed=2025, +79.3% 高于 paper)
- 我们 LETTER R@10=0.0997 vs paper 0.0581 (+71.6% 高于 paper)
- 关键问题是: **fork 升级 vs paper baseline 训练不充分 vs dataset 差异**

**假设**: 拉 LETTER 官方仓库 (含 LETTER-TIGER vanilla baseline, 是 DECOR 论文 §4.2 "official open-sourced code" 的字面指代), 用相同的乐器 Amazon 数据集, 应该:
- 若 paper baseline 训练充分 → 我们 LETTER-TIGER R@10 ≈ 0.0574 (gap 验证)
- 若 paper baseline 训练不充分 → 我们 LETTER-TIGER R@10 > 0.0574 (paper 数字低估)
- 若根本原因在 dataset → 我们 LETTER-TIGER R@10 ≠ 0.0574 且 ≠ 0.1029 (third config 揭示)

---

## 2. 实验设计

**变量**:
- 唯一改动的量: TIGER 代码实现 (从 snap-research/GRID fork 切到 LETTER 官方 LETTER-TIGER pipeline)
- 不动: 数据集 (Musical_Instruments/Amazon Review), Sentence-T5 embedding, codebook config

**保持不变**:
- Stage 1 dataset: `external/LETTER/data/Instruments.*` (LETTER 处理过的乐器 Amazon 数据, 跟我们 `data/amazon_data/toys/Instruments` 等价)
- Stage 2 codebook: LETTER 内置 RQ-VAE (256 codebook × 3 levels)
- Stage 3 T5: t5-base backbone (LETTER-TIGER finetune.py 默认), 200 epoch, per_device_batch_size 256, learning rate 5e-4 (源自 `LETTER-TIGER/run_train.sh`)
- Stage 4 evaluation: 跟 LETTER-TIGER 自己的 `test.py` (full ranking Recall@5/10, NDCG@5/10)

**启动命令**:
```bash
cd /home/wlia0047/ar57/wenyu/GeneRec/external/LETTER/LETTER-TIGER
export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=disabled
export CUDA_LAUNCH_BLOCKING=1
bash run_train.sh
```

`run_train.sh` 内容 (来自 LETTER 仓库):
```bash
DATASET=Instruments
torchrun --nproc_per_node=1 --master_port=2314 ./finetune.py \
    --output_dir ./ckpt/$DATASET/ \
    --dataset $DATASET \
    --per_device_batch_size 256 \
    --learning_rate 5e-4 \
    --epochs 200 \
    --index_file .index.json
```

---

## 3. 决策触发 (vs DECOR paper Table 2 / vs Task #136 fork)

| LETTER-TIGER R@10 区间 | 解读 | 决策 |
|-----------------------|------|------|
| [0.0574, 0.0740] (~paper+Δ0.02) | ✅ paper §4.2 字面 reproduction 成立 | paper Table 2 baseline 数字可对照 (DECOR 增益可信) |
| [0.0741, 0.0929] (paper+Δ0.04) | ⚠️ partial | paper 训练不充分但未达 fork gap, 需要进一步隔离 (run_to_run 噪声 vs paper bias) |
| [0.0930, 0.1150] (fork ±noise) | ❌ fork 仍优 0.0574 | snap-research/GRID fork 配置优于 original, paper baseline 完全低估 |
| > 0.1150 | ❓ fork 仍优 1.5×+ | 强烈说明 paper baseline 训练不充分到无法对齐 |

---

## 4. 预算

| 阶段 | 估算时间 |
|------|---------|
| LETTER-TIGER 训练 200 epoch | ~30 min (T5-base + per_device 256 单卡) |
| Stage 4 test eval | ~5 min |
| Total | ~35-60 min |

---

## 5. 风险与缓解

**风险 1**: LETTER 数据 (Instruments.inter.json + .item.json) 跟我们 internal data/amazon_data/toys/Instruments 是否真的等价?
  → 缓解: 先检查 n_user/n_item 是否跟 paper Table 1 报告 (~24k 用户, ~10k items) 一致, 不一致即停

**风险 2**: LETTER finetune.py 依赖 torch==1.13.1+cu117, 跟我们 grid_toys env 不兼容
  → 缓解: 试 grid_toys (torch 2.13+cu130) 是否能直接跑 (T5-base trainer 是 standard 的), OOM 再调 per_device_batch_size

**风险 3**: LETTER 仓库 commit 8d0154e 自己也是跟 paper Table 2 同期的 LETTER 加成 (含 contrastive alignment + diversity loss)
  → 不能直接当 "vanilla TIGER" baseline. LETTER-TIGER 的 vanilla TIGER 用 `--dataset=$DATASET --epochs 200 --per_device_batch_size 256 --learning_rate 5e-4` 默认参数, **不应该启用 LETTER 的 contrastive/diversity regularization** — 这是 LETTER 论文的修改. vanilla TIGER 配置需要 verify finetune.py 是否自动开 contrastive.
  → 缓解: 跑前 grep finetune.py 看是否包含 "contrastive" / "diversity" 关键字 (如果开了, 这其实跑的是 LETTER-TIGER 不是 vanilla TIGER, 结论需要分两层)

**风险 4**: LETTER 内置的 `Instruments.emb-bge.npy` 是 BGE embedding 不是 Sentence-T5 embedding
  → paper §4.1.2 明确说用 Sentence-T5; LETTER 处理时可能用 BGE (更新的 embedding). 看 `data_process/amazon_text_emb.py` 验证

---

## 6. 完成度跟踪

- [x] Description 写盘 (R9 layer 1+2 audit PASS)
- [x] task138 retroactive fill (R9 layer 2 修复, max=138 现在连续)
- [x] external/LETTER 仓库已克隆 (commit 8d0154e, 5 commits)
- [x] LETTER-TIGER/run_train.sh 已读取 (DATASET=Instruments ✓)
- [ ] **风险核查** (这是 R11.4 critical decision sign-off): 跑前 grep LETTER-TIGER/finetune.py 看是否含 contrastive/diversity (避免被错认成 vanilla TIGER)
- [ ] 验证 LETTER Instruments data n_user/n_item 是否对齐 paper Table 1 (~24k/~10k)
- [x] GPU 分配: GPU 0 (空闲, R7 合规, 不抢 Task #137 GPU 1 / Task #136 GPU 2)
- [ ] 启动 LETTER-TIGER 训练 on GPU 0 (background, PID file)
- [ ] Stage 4 test eval → Recall@5/10, NDCG@5/10
- [ ] 写 verdict `verdicts/task139_letter_tiger_paper_baseline_result.md` (含 §3 决策表完整填写)

result: Task #139 — LETTER-TIGER paper-aligned baseline reproduction ready, R9 contiguous 1-139, GPU 0 allocation ready, 待风险核查后启动
