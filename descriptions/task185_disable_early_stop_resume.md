# Task #185 — 取消早停 + 从 Task #181 best ckpt 继续训练到 1000 epoch

> **任务目的**: 用户 2026-07-25 指令——取消早停机制，从 Task #181 best ckpt (val R@10=0.1262, test R@10=0.1057) 继续训练到 1000 epoch，看是否能在长训中找到更好的 checkpoint。

> **完成日期**: 待启动
> **状态**: 🟡 待启动（等 Task #184 Stage 2 评估后决定，或直接并行启动）

---

## 1. 背景与动机

**Task #181 Stage 3 训练现状**:
- 启动 2026-07-25 16:29，best ckpt 落盘 17:16 (47 min)
- Best validation R@10=0.1262, NDCG@20=0.1003
- Stage 4 test R@10=**0.1057** (+3.6% vs HG-Rec baseline 0.1020)
- Early stop counter=9 (~epoch 92 触发)
- 训练进程消失（用户/系统 kill 或 early stop 后退出）

**用户决策 (2026-07-25)**:
- 早停可能在 epoch ~92 切断训练，可能错过 92→1000 epoch 之间更好的 checkpoint
- 取消早停 → 训练到完整 1000 epoch → 看是否能找到超过 R@10=0.1057 的更好 ckpt

**预期收益**:
- 长训可能让 val/test 指标继续提升（经验上 T5 类模型在 200→500 epoch 之间常有第二段提升）
- 即便没有 val 提升，Stage 4 test 上可能有惊喜（val 和 test 不是严格单调相关）

---

## 2. 实验设计

**变量**:
- 取消 early stop (`--disable_early_stop`)
- 从 Task #181 best ckpt resume (`--resume_from <path>`)
- 训练 epoch 上限 = 1000 (`--epochs 1000`)

**保持不变**:
- Stage 2 codebook: `Instruments_t5_rqvae_paper_fix.npy` (paper fix)
- T5-small 5.5M (6 enc + 4 dec, d_model=128, d_ff=1024)
- β=1.0, loss = Poincaré dist² on raw + logmap0 (Phase 0.6 官方)
- sk_eps=[0,0,0] (Sinkhorn OFF), seed=42
- batch=1024 (train), infer_size=96 (val)
- lr=1e-4, Adam optimizer
- beam_size=20, topk=[5,10,20]

**关键决策点**:
- `resume_start_epoch`: 当前 Task #181 已经跑到 epoch ~92，我们继续 epoch 92→1000（即剩余 ~908 epoch）
- `num_epochs = 1000`（不是 908，因为任务文件说"1000 epoch 总量"）
- early stop patience 设为 99999（disable early stop 实际效果）

**启动命令**:
```bash
python3 /home/wlia0047/ar57/wenyu/GeneRec/scripts/task84_hgrec_stage3_train.py \
    --dataset_name Instruments \
    --train_batch_size 1024 \
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
    --save_path /home/wlia0047/ar57/wenyu/GeneRec/products/task185/t5small_resume/ \
    --log_path /home/wlia0047/ar57/wenyu/GeneRec/logs/ \
    --seed 42 \
    --disable_early_stop \
    --resume_from /home/wlia0047/ar57/wenyu/GeneRec/products/task181/t5small_phase0.6/Instruments/Jul-25-2026_16-29-42/HG_Rec_best.pth \
    --beam_size 20
```

### 决策触发 (vs HG-Rec baseline R@10=0.1020, vs Task #181 R@10=0.1057)

| val R@10 | test R@10 (Stage 4) | 决策 |
|---------|---------------------|------|
| 任何 | ≥ 0.1057 (=Task #181) | ✅ 不退化（最低门槛） |
| 任何 | > 0.1057 | ⭐ **GO** —— 长训找到更好 ckpt，超越 Task #181 |
| < 0.10 | < 0.1057 | 🔴 长训导致过拟合 / 不稳定 |

---

## 3. 修改文件清单

| 文件 | 修改 |
|------|------|
| `scripts/task84_hgrec_stage3_train.py` | 加 `--disable_early_stop` / `--resume_from` / `--resume_start_epoch` CLI 参数；resume 时 load_state_dict；disable 时 early_stop 设为 99999 |
| `scripts/task185_resume_train.sh` (新) | Task #185 launcher，调用上面的 trainer |
| `verdicts/task185_resume_result.md` (新) | 训练完后写 verdict |

---

## 4. 预算

| 阶段 | 估算时间 |
|------|---------|
| 1000 epoch 训练 | ~6 h (22 it/s × 515 batch × 1000 = 6.5h, 但 R12 epoch save + 每 epoch evaluate 加 ~30s/eval × 1000 = 8h total) |
| Stage 4 test eval | ~5 min |
| 总计 | ~6-8 h |

**GPU 占用**: 1 张 L40S（GPU 0 空闲可用）

---

## 5. 风险与缓解

**风险 1**: 长训导致过拟合 → val R@10 在 epoch ~150 后开始下降 → 但 disable early stop 后仍会继续训练；ckpt 只在 val NDCG@20 提升时覆盖（best_ckpt），所以最终 ckpt 是 val 上最优的
**风险 2**: Resume 后 optimizer 状态丢失 → 重新初始化 Adam，lr=1e-4 跟原来一样，不会破坏训练
**风险 3**: 长训找不到更好 ckpt → val 指标饱和在 0.1262 / NDCG@20=0.1003 → Stage 4 test 仍是 0.1057 → 平局（无 GO 也不 NO-GO，记"未超越 Task #181"）
**风险 4**: 训练中途崩溃 → R12 已强制每个 epoch 末保存 best ckpt → 重启可继续 resume
**风险 5**: GPU 占用冲突 → Task #184 Stage 2 也需要 GPU 0 → 串行执行：先 Task #185 跑完，再 Task #184 Stage 2/3/4

---

## 6. 完成度跟踪

- [ ] 修改 task84_hgrec_stage3_train.py（已加 --disable_early_stop + --resume_from）
- [ ] py_compile 验证（已通过）
- [ ] 写 task185_resume_train.sh launcher
- [ ] 启动 Task #185 训练 (GPU 0, PID 待分配)
- [ ] 监控 val 指标趋势 (epoch 92→1000)
- [ ] Stage 4 test eval (vs Task #181 R@10=0.1057)
- [ ] Verdict 写完 + §16 清理

---

## 7. 关键决策点（R11.3 自主决策明示）

### 决策 1: 是否用 resume_from 还是从头开始?
**选了**: `resume_from` Task #181 best ckpt。
**为什么**: 用户明确指令"接着task181的checkpoint训练"。从头训违背指令。
**备选**: 从头训 1000 epoch — 浪费之前 47 min 训练结果，且不满足"接着 task181 的 checkpoint"语义。

### 决策 2: 取消早停但保持 best_ckpt 覆盖逻辑?
**选了**: 保持 best_ckpt 覆盖逻辑 + 取消 early_stop 触发退出。
**为什么**: best_ckpt 是 val NDCG@20 提升时保存的，长期训练 val 可能上下波动，best_ckpt 仍是 val 最优的。disable_early_stop 让训练跑到 1000 epoch，但 best_ckpt 不会被差 epoch 覆盖。
**备选**: 完全禁止任何 ckpt 保存（仅 last epoch 保存）— 风险大，违反 R12。

### 决策 3: 是否并行 Task #185 + Task #184?
**选了**: 串行（先 Task #185 1000 epoch → 然后 Task #184 Stage 2/3/4）。
**为什么**: Task #184 Stage 2 需要 GPU 0 inference；Task #185 训练需要 GPU 0 训练。两者争同一卡。
**备选**: Task #185 用 GPU 0 训练 + Task #184 Stage 2 inference 用 GPU 2（inference 显存小，可能可行）→ 留作后续选择，先串行保险。

---

## 8. 修订记录

| 日期 | 修订内容 |
|------|---------|
| 2026-07-25 | 首次创建（用户 2026-07-25 17:50 指令） |
