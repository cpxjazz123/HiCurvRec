# Task #188 — Paper Table 7 复现 + 3 种子方差报告

> **任务目的**: 按用户 2026-07-25 18:42 提案, 完整复现 HG-Rec paper Table 7 (不同碰撞率 codebook 的下游 Recall@10) + 增加 3 种子方差报告. 这是 paper 没做的, 是项目建立可信度最关键的一步.

> **完成日期**: (in progress)
> **状态**: 🟡 待启动

---

## 1. 背景与动机

**用户 18:42 提案摘要**:
- HG-Rec 流水线三个环节的 checkpoint 选择:
  - ✅ Stage 1 取碰撞率最优 (自动, `best_collision_model.pth`)
  - ✅ Stage 3 取 best-val checkpoint (自动, `HG_Rec_best.pth`)
  - ❌ Stage 2 选哪个 ckpt 进 codebook 生成 (手动改 `gen_codebook.py` 路径)
- 默认 `--save_limit=5` 会删除早期 (碰撞率 80%) 的 ckpt — Table 7 复现不出来
- **复现建议**: `--save_limit 50` 保留完整碰撞率谱系, 跑 4 个碰撞率档 (≈20%/40%/60%/80%) + 3 个种子 = 12 个 Stage 3 实验, 报告方差

**Task #181 现状** (单一组合):
- Stage 1: β=1.0, Sinkhorn OFF, 1000 epoch, save_limit=5 (默认)
- 现有 ckpt: epoch 54/59/64/69/79 (coll 0.09-0.094) + epoch 979-999 (coll 0.12)
- **缺**: 80%/60%/40% 碰撞率档的 ckpt (早就被 save_limit 删了)
- 单 seed (42) → 单 test R@10=0.1057, 没法评估方差

**Task #188 假设**:
- R1: paper Table 7 显示"碰撞率越低, R@10 越高"趋势 — 我们能在 Task #188 复现这个趋势
- R2: 3 种子的 R@10 标准差 < paper 报的 5% 提升 — paper 提升是显著的, 不是种子偶然
- R3: (如果 R2 失败) HG-Rec paper 的 5% 提升落在种子间方差范围内 → 论文结论站不住脚

---

## 2. 实验设计

### Phase 1 — Stage 1 重训 (`--save_limit 50`)

**目的**: 单次 Stage 1 训练, 但保留完整碰撞率谱系 (epoch 4-999 全留)

**启动命令**:
```bash
cd /home/wlia0047/ar57/wenyu/GeneRec/HG-Rec
python3 -u train_hrqvae.py \
    --lr 1e-3 --epochs 1000 --batch_size 1024 \
    --num_workers 4 --eval_step 5 \
    --learner AdamW --lr_scheduler_type linear --warmup_epochs 20 \
    --data_path ./dataset/Instruments/item_emb.parquet \
    --weight_decay 0 --dropout_prob 0.0 --bn False \
    --loss_type poincare --kmeans_init True --kmeans_iters 1000 \
    --sk_epsilons 0.0 0.0 0.0 --sk_iters 50 \
    --device cuda:0 \
    --num_emb_list 64 128 256 --e_dim 32 \
    --quant_loss_weight 1.0 --beta 1.0 \
    --layers 512 256 128 64 \
    --save_limit 50 \
    --ckpt_dir /home/wlia0047/ar57/wenyu/GeneRec/products/task188/hrqvae_save_limit50
```

**预期产物** (~200 个 epoch_X_collision_Y.pth):
- epoch 4-9: collision ~0.85-0.99 (≈80% 档)
- epoch 14-29: collision ~0.18-0.71 (≈60% 档)
- epoch 39-79: collision ~0.09-0.13 (≈20% 档)
- epoch 999: collision ~0.12

### Phase 2 — Stage 2 codebook (4 个碰撞率档)

**选 4 个目标 ckpt**:
| 档位 | 目标 collision | 选择策略 |
|------|----------------|----------|
| ≈80% | 0.85 ± 0.10 | 从 epoch 4-9 选最接近 |
| ≈60% | 0.60 ± 0.05 | 从 epoch 14-29 选最接近 |
| ≈40% | 0.40 ± 0.05 | 从 epoch 19-29 选最接近 (coll 0.41) |
| ≈20% | 0.13 ± 0.05 | 从 epoch 59-79 选最接近 (coll 0.09-0.13) |

**Stage 2 命令** (复用 `task181_stage2_codebook.py`):
```bash
python3 scripts/task188_stage2_codebook.py \
    --ckpt_epoch <N> --ckpt_collision <C> \
    --output_path dataset/Instruments/Instruments_t5_rqvae_paper_fix_coll<C>.npy
```

**输出**: 4 个 `.npy` codebook (9922, 4) 各给一个下游 Stage 3 用

### Phase 3 — Stage 3 训练 (4 codebook × 3 seed = 12 个 run)

**Stage 3 命令** (复用 `task84_hgrec_stage3_train.py`):
```bash
for SEED in 42 123 2024; do
  for COLL in 0.85 0.60 0.40 0.13; do
    python3 scripts/task84_hgrec_stage3_train.py \
        --batch_size 1024 --infer_size 96 --num_epochs 200 \
        --lr 1e-4 --num_layers 6 --num_decoder_layers 4 \
        --d_model 128 --d_ff 1024 --num_heads 6 --d_kv 64 \
        --vocab_size 1025 --max_len 20 --pad_token_id 0 --eos_token_id 0 \
        --dataset_name Instruments \
        --codebook_size 64 128 256 1 \
        --code_path _t5_rqvae_paper_fix_coll${COLL}.npy \
        --mode train --early_stop 20 \
        --save_path /home/wlia0047/ar57/wenyu/GeneRec/products/task188/t5small_coll${COLL}_seed${SEED}/ \
        --seed ${SEED} --beam_size 20
  done
done
```

### Phase 4 — Stage 4 test eval (12 个)

```bash
for SEED in 42 123 2024; do
  for COLL in 0.85 0.60 0.40 0.13; do
    python3 scripts/task188_stage4_eval.py \
        --ckpt products/task188/t5small_coll${COLL}_seed${SEED}/.../HG_Rec_best.pth \
        --code_path _t5_rqvae_paper_fix_coll${COLL}.npy \
        --output verdicts/task188_coll${COLL}_seed${SEED}_metrics.json
  done
done
```

**输出格式**: 每个 run 一个 JSON, 然后汇总成表.

---

## 3. 决策触发 (vs HG-Rec baseline 0.1020, vs Task #181 0.1057)

| 条件 | 决策 |
|------|------|
| **R1 趋势成立**: R@10 随 collision 升高而单调下降 | ✅ GO — 复现 paper Table 7 趋势 |
| R1 趋势不单调 (e.g. 80% 反而 R@10 最高) | ⚠️ 反直觉 — 报告异常, 不强行解释 |
| **R2 方差 < 5%**: std(R@10) / mean(R@10) < 0.05 across 3 seeds | ✅ paper 5% 提升是显著 |
| R2 方差 ≥ 5% | 🔴 paper 5% 提升落在方差内, 结论不可信 |

**报告矩阵** (verdict 主输出):
```
                  seed=42  seed=123  seed=2024  mean   std
collision≈80%     R@10=?   R@10=?    R@10=?     ?      ?
collision≈60%     ?        ?         ?          ?      ?
collision≈40%     ?        ?         ?          ?      ?
collision≈13%     ?        ?         ?          ?      ?
```

---

## 4. 预算

| 阶段 | 估算时间 |
|------|---------|
| Phase 1 Stage 1 重训 (1000 epoch, save_limit=50, GPU 0) | ~5 min |
| Phase 2 Stage 2 × 4 | ~8 min |
| Phase 3 Stage 3 × 12 (并行 GPU 0/1/2/3, 各 ~12 min) | ~40 min (3 串行 × 4 coll) |
| Phase 4 Stage 4 × 12 | ~10 min |
| **总计** | **~65 min** (如 GPU 满载并行可压到 ~25 min) |

**GPU 占用**:
- Phase 1: 1× GPU
- Phase 3-4: 4× GPU 并行 (12 run / 4 GPU = 3 串行 / GPU)

---

## 5. 风险与缓解

**风险 1**: `--save_limit 50` 仍不够 — 实际碰撞率档可能不落在 [80%, 60%, 40%, 20%]
→ 缓解: 选 ckpt 时允许 ±5-10% 浮动, 不强求精确

**风险 2**: 12 个 Stage 3 run 占用 GPU 时间长 (~40 min), 用户中途可能叫停
→ 缓解: 每个 run 用 R12 强制 best_ckpt 保存, 即使中途停也有产物

**风险 3**: Stage 2 codebook 选错 ckpt (碰撞率 ≠ 目标值)
→ 缓解: 选 ckpt 时用 `collision` 字段精确匹配, 不要凭 epoch 推算

**风险 4**: 多 seed 训练中 Stage 3 早停阈值不一致 — 同一 collision 档下, 不同 seed 可能在 epoch 50 / 80 / 120 各自早停
→ 接受, 不强制 fixed epoch. 早停本身是 valid 训练方式, 记录每个 seed 早停 epoch

**风险 5**: Stage 1 重训跟 Task #181 用的 Stage 1 ckpt 不同, 没法跟 Task #181 直接对比
→ 缓解: 复用 Task #181 现有 collision 0.094 ckpt 跑一次 Stage 3 当 baseline (seed=42), 跟 Task #181 R@10=0.1057 对齐

---

## 6. 完成度跟踪

- [ ] Phase 1: Stage 1 重训 (--save_limit 50)
- [ ] Phase 2: 选 4 个 collision 档的 ckpt, 生成 4 个 codebook
- [ ] Phase 3: Stage 3 × 4 codebook × 3 seed = 12 个 run
- [ ] Phase 4: Stage 4 × 12 = 12 个 test eval JSON
- [ ] 汇总矩阵 (mean/std per collision 档)
- [ ] 写 verdict + 更新 paper section

---

## 7. 关键决策点 (R11.3 自主决策)

### 决策 1: 种子选哪 3 个?
**选了**: seed=42 (Task #181 baseline) + seed=123 + seed=2024
**为什么**: 42 是主 seed, 123/2024 是常用 distinct primes. 避免选 0/1 (容易让模型退化)
**备选**: 完全连续 (40/41/42) — 不够 distinct, R11.2 不用

### 决策 2: 4 个 collision 档还是 5 个?
**选了**: 4 档 (≈80/60/40/13%)
**为什么**: 跟 paper Table 7 一致 (paper 也是 4 档)
**备选**: 5 档加 ≈30% — ROI 低, paper 没做, 跳过

### 决策 3: Stage 3 训练并行策略?
**选了**: 4 GPU 各跑一个 (Stage 3 × 12 = 4 GPU × 3 串行)
**为什么**: 充分利用 4× L40S, 节省时间
**备选**: 单 GPU 串行 12 个 — 太慢 (~2.5 h), R7 鼓励并行

### 决策 4: 复用 Task #181 现有 ckpt (collision 0.094) 还是 Phase 1 重训?
**选了**: Phase 1 重训 (跟 Task #181 Stage 1 同 recipe, 但 save_limit=50)
**为什么**: 用户明确建议 "把 --save_limit 调大", 重训是干净实验
**备选**: 复用现有 ckpt — collision 0.094 已经在 (≈20% 档), 但缺 40%/60%/80% 档, 必须重训

---

## 8. 修订记录

| 日期 | 修订内容 |
|------|---------|
| 2026-07-25 | 首次创建 (用户 18:42 提案) |
