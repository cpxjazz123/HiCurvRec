# Task #187 — 4 层 Encoder 实验 (vs Task #181 6 层)

> **任务目的**: 用户 2026-07-25 18:25 指令 — 试一下 encoder 用 4 层 (vs Task #181 的 6 层). 验证减少 encoder 深度对 R@10 的影响.

> **完成日期**: 待启动
> **状态**: 🟡 待启动

---

## 1. 背景与动机

**Task #181 现状** (6 层 encoder):
- Encoder: 6 层, d_model=128, d_ff=1024, 6 heads
- Decoder: 4 层
- 总参数: 5,508,864
- Test R@10 = 0.1057 (新高, +3.6% vs baseline 0.1020)
- Test R@5 = 0.0848, R@20 = 0.1293

**Task #187 假设**:
- 减少 encoder 层数 → 模型容量 ↓ → 可能 test R@10 ↓ (因为 capacity 限制)
- 但训练更快, 也可能 early stop 提前收敛
- 单变量对照: --num_layers 6 → 4, 其他不变

**用户决策依据** (R11.3 自主决策):
- 如果 4 层 R@10 ≥ 0.1057: encoder 6 层是浪费, 用 4 层即可 (更快训练, 更小 ckpt)
- 如果 4 层 R@10 < 0.0850: 6 层是必须的, 不要继续减少
- 如果 0.0850 ≤ R@10 < 0.1057: 中间地带, 性价比 (R@10 / params) 衡量

---

## 2. 实验设计

**变量** (单变量):
- `--num_layers 4` (vs Task #181 6)

**保持不变**:
- `--num_decoder_layers 4`
- `--d_model 128 --d_ff 1024 --num_heads 6 --d_kv 64`
- `--vocab_size 1025 --max_len 20`
- `--batch_size 1024 --infer_size 96 --lr 1e-4`
- `--codebook_size 64 128 256 1 --code_path _t5_rqvae_paper_fix.npy`
- `--seed 42 --early_stop 20 --beam_size 20`
- Recipe: Phase 0.6 paper-aligned (Poincaré dist² + logmap0 + β=1.0, Sinkhorn OFF)
- Stage 2 codebook: `Instruments_t5_rqvae_paper_fix.npy` (跟 Task #181 同一个)

**参数估算**:
- 每层 encoder: ~328k params
- 6 → 4 层 = -656k params
- 新总参数: ~4.85M (vs Task #181 5.51M)
- 训练时长: 估计 ~8 min/epoch → 早停在 epoch ~80-100 (跟 Task #181 类似)

**启动命令**:
```bash
python3 /home/wlia0047/ar57/wenyu/GeneRec/scripts/task84_hgrec_stage3_train.py \
    --batch_size 1024 \
    --infer_size 96 \
    --num_epochs 200 \
    --lr 1e-4 \
    --num_layers 4 \                # ← 单变量
    --num_decoder_layers 4 \
    --d_model 128 --d_ff 1024 --num_heads 6 --d_kv 64 \
    --vocab_size 1025 --max_len 20 --pad_token_id 0 --eos_token_id 0 \
    --dataset_name Instruments \
    --codebook_size 64 128 256 1 --code_path _t5_rqvae_paper_fix.npy \
    --mode train \
    --save_path /home/wlia0047/ar57/wenyu/GeneRec/products/task187/t5small_4layer/ \
    --log_path /home/wlia0047/ar57/wenyu/GeneRec/logs/task187/ \
    --seed 42 --early_stop 20 --beam_size 20
```

### 决策触发 (vs HG-Rec baseline 0.1020, vs Task #181 0.1057)

| Test R@10 | 决策 |
|-----------|------|
| ≥ 0.1057 | ⭐ **GO** — 4 层够了, 训练更快, 节省 12% 参数 |
| 0.1020 - 0.1057 | ✅ 持平 — 单变量不重要, 6 层是历史选择 |
| 0.0850 - 0.1020 | ⚠️ 退化但有信号 — encoder 6 层是必要的 |
| < 0.0850 | 🔴 4 层严重不足 |

---

## 3. 修改文件清单

| 文件 | 修改 |
|------|------|
| `scripts/task187_encoder4_train.sh` (新) | launcher |
| `descriptions/task187_encoder4_layers.md` (新) | 本文件 |
| `verdicts/task187_encoder4_result.md` (新) | 训练完后写 verdict |

---

## 4. 预算

| 阶段 | 估算时间 |
|------|---------|
| Stage 3 训练 (200 epoch, 早停 ~80-100) | ~15 min |
| Stage 4 test eval | ~5 min |
| 总计 | ~20 min |

**GPU 占用**: 1× L40S (GPU 0)

---

## 5. 完成度跟踪

- [ ] 写 task187_encoder4_train.sh launcher
- [ ] 启动 Task #187 训练 (GPU 0)
- [ ] 监控 val 指标, 早停后 Stage 4 eval
- [ ] Verdict 写完 + §16 清理

---

## 6. 关键决策点 (R11.3 自主决策)

### 决策 1: 早停阈值?
**选了**: --early_stop 20 (跟 Task #181 一样)
**为什么**: 跟 Task #181 单变量对照, 早停阈值不变

### 决策 2: num_epochs 上限?
**选了**: 200 (跟 Task #181 一样)
**为什么**: 跟 Task #181 单变量对照, 上限不变. 早停会提前终止

---

## 7. 修订记录

| 日期 | 修订内容 |
|------|---------|
| 2026-07-25 | 首次创建 (用户 18:25 指令) |
