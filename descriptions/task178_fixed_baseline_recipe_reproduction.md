# Task #178 — 修正后的 HG-Rec baseline (论文 Table 6 Instruments 原配)

> **任务目的**: 用 Phase 0 修复后的 HG-Rec 量化器代码 (utils.py 4 处 bug + train_hrqvae.py 论文原配默认值), 跑完整 Stage 1+2+3+4 流水线, 拿到修复后的 HG-Rec baseline R@10. 这是后续 Idea 1/2 决策的新对照基线.

> **完成日期**: in progress
> **状态**: 🟡 待启动 (Phase 0 修复已验证, GPU 0 已释放)

---

## 1. 背景

**2026-07-25 用户关键发现**: 本地 HG-Rec 量化器代码 (`HG-Rec/model/utils.py`) 与论文 Eq (6)(7)(8) + Table 6 配置**多处不匹配**, 很可能就是连续 9 个 κ-Stereo variant 全部 NO-GO 的核心外因. Phase 0 已修复 6 处不匹配 (utils.py 4 处 + train_hrqvae.py 默认值), 单元测试 4/4 通过.

**当前基线对照**:
- 旧 HG-Rec baseline (Task #84): R@10 = 0.1020, **建立在 broken 代码上**, 跟论文 Table 6 不对齐
- 新目标: 论文 Table 6 Instruments 原配 ([64,128,256], β=0.5, poincare loss, sk_epsilons=0.003, epoch=200), 修正后的代码

**假设 R1**: 修复后 HG-Rec baseline 在 0.10-0.13 之间, 跟旧 0.1020 持平或略高 (修复主要消除坍缩倾向, 不显著改变几何信号).
**假设 R2**: 若 R1 失败 (新 baseline 显著 < 0.1020), 说明旧代码存在"过拟合特殊路径", 此时应抛弃旧 baseline 数字.

## 2. 实验设计

**变量**: 仅 HG-Rec 量化器代码 (Phase 0 修复后) + train_hrqvae.py 默认值 (论文 Table 6)
**保持不变**:
- T5-small 5.5M (跟 Task #84 完全一致: 6 enc + 4 dec, d_model=128, num_heads=6, d_kv=64)
- Stage 2 SID tensor shape `(9922, 4)` (论文 3 层 + 1 dedup = 4 digits)
- Stage 4 eval 配置 (Recall@5/10/20, NDCG@5/10/20, beam_size=20, seed=42)
- Item embeddings (`sentence-t5-base` 输出, 9922 × 768)

**Stage 1 启动命令** (论文 Table 6 + Phase 0 修复):
```bash
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys
cd /home/wlia0047/ar57/wenyu/GeneRec/HG-Rec
export TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task178
mkdir -p $TRITON_CACHE_DIR
python3 train_hrqvae.py \
  --lr 1e-3 \
  --epochs 200 \
  --batch_size 1024 \
  --num_workers 4 \
  --learner AdamW \
  --lr_scheduler_type linear \
  --warmup_epochs 20 \
  --data_path ./dataset/Instruments/item_emb.parquet \
  --weight_decay 0 \
  --dropout_prob 0.0 \
  --bn False \
  --loss_type poincare \
  --kmeans_init True \
  --kmeans_iters 1000 \
  --sk_epsilons 0.003 0.003 0.003 \
  --sk_iters 50 \
  --device cuda:0 \
  --num_emb_list 64 128 256 \
  --e_dim 32 \
  --quant_loss_weight 1.0 \
  --beta 0.5 \
  --layers 512 256 128 64 \
  --save_limit 5 \
  --ckpt_dir /home/wlia0047/ar57/wenyu/GeneRec/products/task178/hgrec_fixed_baseline/jul-25-2026_XX-XX-XX
```

**Stage 2 启动命令**:
```bash
cd /home/wlia0047/ar57/wenyu/GeneRec
python3 scripts/task178_hgrec_fixed_stage2_codebook.py \
  --ckpt_path /home/wlia0047/ar57/wenyu/GeneRec/products/task178/hgrec_fixed_baseline/<TS>/HRQVAE_best.pth \
  --output_path /home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/Instruments_t5_rqvae_fixed_hgrec.npy \
  --device cuda:0
```

**Stage 3 启动命令** (T5-small 5.5M, 跟 Task #84 一致):
```bash
cd /home/wlia0047/ar57/wenyu/GeneRec
export TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task178_stage3
mkdir -p $TRITON_CACHE_DIR
nohup python3 scripts/task84_hgrec_stage3_train.py \
  --dataset_name Instruments \
  --dataset_path /home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/ \
  --code_path _t5_rqvae_fixed_hgrec.npy \
  --codebook_size 64 128 256 1 \
  --num_epochs 200 \
  --batch_size 256 \
  --lr 1e-4 \
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
  --save_path /home/wlia0047/ar57/wenyu/GeneRec/products/task178/t5small_fixed_hgrec/jul-25-2026_XX-XX-XX \
  --log_path /home/wlia0047/ar57/wenyu/GeneRec/logs/task178 \
  --seed 42 \
  --early_stop 20 \
  --beam_size 20 \
  --infer_size 96 \
  > logs/task178_stage3_train.out 2>&1 &
echo $! > /home/wlia0047/ar57/wenyu/GeneRec/products/task178/_TRAINING_PID
```

**Stage 4 eval 命令**:
```bash
cd /home/wlia0047/ar57/wenyu/GeneRec
bash scripts/task178_hgrec_fixed_stage4_eval.sh
```

## 3. 决策触发 (vs 旧 HG-Rec baseline R@10=0.1020)

| 测得 R@10 | 与 0.1020 比较 | 解读 + 下一步 |
|-----------|-----------------|-------------|
| **≥ 0.1100** | 显著超过 | HG-Rec 修正版本确实超越旧 baseline, 修复消除了坍缩倾向, 之前 9 个 κ-Stereo NO-GO 大概率部分源自代码 bug. |
| **0.0900 - 0.1100** | 持平 ±0.008 | HG-Rec 几何在 Instruments 上中性, 修复代码不显著改变基线. 进入 Phase 2/3 (Idea 1/2) 探索新方向. |
| **< 0.0900** | 显著恶化 | 旧代码存在"过拟合特殊路径", 抛弃旧 0.1020 baseline, 用新测得数字作为决策基线. |

## 4. 预算

| 阶段 | 估算时间 |
|------|---------|
| Stage 1 RQ-VAE 训练 (200 ep) | ~30-60 min (CPU/GPU 0) |
| Stage 2 SID codebook 推断 | ~5-10 min (GPU 0) |
| Stage 3 T5-small 5.5M 训练 (200 ep + early_stop=20) | ~2-4 h (GPU 0) |
| Stage 4 test eval | ~2-3 min (GPU 0) |
| **总计** | **~3-5 h** |

## 5. 风险与缓解

**风险 1**: Stage 1 训练中可能因为 β 挂载方式修正 + sk_epsilons 打开后, early stage loss 数值比旧 baseline 大 → 训练不稳定
**缓解**: 沿用 200 epoch + warmup_epochs=20 + early_stop=20, R12 强制 best_loss_model.pth on improvement + 删旧

**风险 2**: 修正后 HG-Rec baseline 数字比 0.1020 低 → 用户对"修复反而变差"不满
**缓解**: verdict 中明示"修正版本更接近论文原配, 旧 0.1020 可能建立在数据过拟合特殊路径上, 不强行追旧数字"

**风险 3**: GPU 0 在训练中, Phase 2/3 (Idea 1/2) 启动被阻
**缓解**: Phase 2/3 安排在 GPU 3 (空闲), 与 Phase 1 (GPU 0) 完全并行, R7 互不干扰

**风险 4**: Stage 2 推断时 CUDA OOM (sinkhorn 30 轮 + 9922 items)
**缓解**: 复用 Task #174 pattern, batch_size 256, device=cuda:0, ~5 min 跑完

## 6. 完成度跟踪

- [ ] Stage 1 launch (HRQ-VAE 训练)
- [ ] Stage 1 finish (200 epoch / early stop + best_loss_model.pth 落盘)
- [ ] Stage 2 launch (Sinkhorn codebook inference)
- [ ] Stage 2 finish (npy shape (9922, 4) + unique=9922 + 4 层利用率 ≥85%)
- [ ] Stage 3 launch (T5-small 5.5M 训练, GPU 0)
- [ ] Stage 3 finish (best ckpt 落盘 + 早停触发 / 200 epoch 完成)
- [ ] Stage 4 launch + finish (eval on test set, JSON 写入)
- [ ] verdict 写完 (`verdicts/task178_fixed_baseline_result.md`)
- [ ] CLAUDE.md 更新 (新 baseline 数字替换 0.1020)
- [ ] loop.md §16 cleanup (R8)

## 7. R-RQ-VAE framework 集成

### R7 GPU 隔离
- Phase 1 占 GPU 0 全程
- Phase 2 (Idea 1) 安排 GPU 3, 完全并行不抢卡
- Phase 3 (Idea 2) 安排 GPU 3, 完全并行不抢卡
- 实际 Phase 2/3 也可分开: Idea 1 GPU 3, Idea 2 GPU 0 (Stage 1) → GPU 3 (Stage 3) — 视实际 GPU 状态动态调整

### R8 §16 cleanup
- Phase 0 完成 → #176/#177 行已删, 但 verdicts 已写 (broken baseline termination 标记)
- Phase 1 完成 → 删 #178 行, verdict 保留

### R9 编号连续
- descriptions max = 177 → 新编号 178 ✅ 已 pre-creation check
- 无空洞 (1..177 连续)

### R10 主动推进
- Phase 1 launch 后立即启动 Phase 2 (Idea 1) — 不空闲等待 Phase 1 完成

### R11 自主决策
- Stage 1 β=0.5 论文原配 (备选: 1.0, 不选)
- Stage 1 epochs=200 论文原配 (备选: 300/500, 不选)
- Stage 1 sk_epsilons=0.003 打开 Sinkhorn (备选: 0 关掉, 不选 — 论文继承自 TIGER/LETTER)
- Stage 3 num_layers=6/num_decoder_layers=4/d_model=128 跟 Task #84 完全一致 (备选: T5-base 220M, 不选 — R12 强制存 + 单 seed 已够)

### R12 强制存 ckpt
- Stage 1 trainer 默认保存 best_loss_model.pth (R12 验收)
- Stage 3 trainer (Task #84) 已 R12 化

### R13 不用 worktree
- 全部修改直接落在共享 checkout