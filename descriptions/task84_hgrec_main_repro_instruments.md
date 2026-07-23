# Task #84 — HG-Rec 主实验复现 (Musical_Instruments, paper Table 2 #14)

> **任务目的**: 复现 ICML 2026 HG-Rec 主实验 (Poincaré loss RQ-VAE + Differential-Length
> Codebook + T5-based seq2seq), 在 Musical_Instruments 上评估 Recall@5/10, NDCG@5/10.
> **核心目标**: 验证 Poincaré 损失 (vs vanilla MSE) 是否在 RQ-VAE 量化上**显著超过** phonism
> (vanilla RQ-VAE + SINKHORN, 复现 R@10=0.1058) — 如果是, 强证实 Task #70 输入空间强双曲
> (κ_Ollivier=-0.65~-0.84) 的几何先验在下游有意义.

> **完成日期**: (in progress)
> **状态**: 🟡 待启动

---

## 1. 背景

**任务来源**: 用户 2026-07-23 要求 "复现HG-Rec主实验". HG-Rec (ICML 2026) 是
Hyperbolic RQ-VAE enhanced Generative Recommendation with Differential-Length Codebook
Strategy, 仓库: `HG-Rec/`.

**承接的前置 task 结论**:
- **Task #70** (5-graph Ollivier curvature): sentence-t5 768d 输入空间**强双曲** (κ=-0.65~-0.84,
  99%+ 边 κ<0), MCKG 训练出的 κ1 ≈ +0.7 是模型 regularization 而非数据真实
- **Task #80** (per-layer κ distortion v2): RQ-VAE 32D 残差空间**真实欧氏** (RGD MDS + Kruskal
  stress-1 + 加密网格三重支撑验证)
- **Task #82** (Stage 1c 弱信号 + 网格加密): phonism 残差真实欧氏, 排除 metric bug +
  网格空隙 + 弱信号漏检三种解释
- **Task #78+#84** (TIGER T5 训练+inference): phonism 复现 R@10=**0.1058** (paper 0.0574, +84%)
- **Task #80+#85** (FDSA RecBole 训练+eval): 复现 R@10=**0.0594** (paper 0.0391, +52%)
- **Task #82+#85** (P5-CID 训练+eval): 复现 test hit@10=**0.0413**, NDCG@10=0.0211 (paper N@10=0.0158, +33%)

**关键 insight** (HG-Rec 主实验假设):
> HG-Rec 用 **Poincaré loss** 训练 HRQ-VAE (vs vanilla MSE), 利用 sentence-t5 768d
> 输入空间的强双曲先验. 如果 HG-Rec 复现 R@10 **显著超过** phonism 0.1058 (我们 baseline),
> 这是**第一个在 downstream 验证 Task #70 几何先验有效**的实验.

**编号冲突说明 (R9 偏差)**:
- descriptions/ max=82, R9 规则要求新任务 = max+1 = 83
- 但 **#83 已被 Task #83 P5-SID 训练占用** (PID 193954, GPU 2, products/task83/,
  scripts/task83_*.sh, TaskList #77 in_progress)
- 本任务用 #84, 不破坏 P5-SID 现有路径, 偏差在 verdict §0 记录

---

## 2. 实验设计

**3 阶段 Pipeline** (HG-Rec 仓库自带):

### Stage 1 — HRQ-VAE 训练 (`train_hrqvae.py`)

**变量**: `loss_type='poincare'` (vs vanilla 'mse'/'l1') — **唯一核心改动**
**保持不变**:
- num_emb_list=[64,128,256] (3 层 RQ)
- e_dim=32
- sk_epsilons=[0,0,0.000] (SINKHORN last layer only, 同 phonism)
- layers=[512,256,128,64] (encoder/decoder MLP)
- data_path: `HG-Rec/dataset/Instruments/item_emb.parquet` (sentence-t5 768d → HRQ-VAE 输入)

**启动命令**:
```bash
cd /home/wlia0047/ar57/wenyu/GeneRec/HG-Rec
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys
export CUDA_VISIBLE_DEVICES=1
python3 train_hrqvae.py \
  --data_path ./dataset/Instruments/item_emb.parquet \
  --ckpt_dir ./ckpt/Instruments \
  --loss_type poincare \
  --num_emb_list 64 128 256 \
  --e_dim 32 \
  --sk_epsilons 0.0 0.0 0.000 \
  --epochs 1000 \
  --batch_size 1024 \
  --lr 1e-3 \
  --device cuda:0
```

### Stage 2 — Codebook 生成 (`gen_codebook.py`)

**任务**: 从 Stage 1 best HRQ-VAE ckpt 生成 Differential-Length Codebook + 变长 SID
**启动命令**:
```bash
cd /home/wlia0047/ar57/wenyu/GeneRec/HG-Rec
python3 gen_codebook.py \
  --ckpt_path ./ckpt/Instruments/<best_hrqvae>.pth \
  --output_path ./dataset/Instruments/Instruments_t5_hrqvae_poincare.npy
```

### Stage 3 — HG-Rec 训练 (`train_HG-Rec.py`)

**变量**: 改用 HG-Rec codebook (vs phonism RQ-VAE codebook)
**保持不变**:
- T5-small: 6 encoder + 4 decoder layers, d_model=128, d_ff=1024, num_heads=6
- vocab_size=1025 (4 层 codebook 总和 + pad)
- num_epochs=200, batch_size=256, lr=1e-4
- infer_size=96, num_beams=20 (评估时)
- dataset_path: `HG-Rec/dataset/Instruments/`

**启动命令**:
```bash
cd /home/wlia0047/ar57/wenyu/GeneRec/HG-Rec
export CUDA_VISIBLE_DEVICES=1
python3 train_HG-Rec.py \
  --dataset_name Instruments \
  --dataset_path ./dataset/ \
  --code_path _t5_hrqvae_poincare.npy \
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
  --device cuda \
  --log_path ./logs/
```

### Stage 4 — 评估

**指标**: Recall@5/10, NDCG@5/10 (与 paper Table 2 一致)
**命令**: `train_HG-Rec.py --mode evaluation --code_path ... --infer_size 96 --num_beams 20`

---

## 3. 决策触发 (vs baselines)

| 指标条件 | HG-Rec R@10 区间 | 决策 |
|----------|-----------------|------|
| **> 0.1058 (phonism baseline)** | > 0.1058 | ⭐ **强证实 Poincaré 损失有效**, Task #70 几何先验下游有意义 |
| **0.06 - 0.1058** | 0.06 - 0.1058 | 🟡 持平或部分优于 baseline, 进一步分析 (训练 epoch 不足? 超参不当?) |
| **< 0.06 (paper HG-Rec 估计下限)** | < 0.06 | ❌ 否证 HG-Rec, 回到 phonism |

**辅助基线**:
- phonism 复现: R@10=0.1058 (我们的最强 baseline)
- FDSA 复现: R@10=0.0594
- P5-CID 复现: hit@10=0.0413
- paper HG-Rec 报告数字: 待从 paper 提取 (paper 没在手, 估计 R@10 > 0.06 合理)

**核心目标 (R11.3 自主决策)**:
- 主目标: HG-Rec R@10 **> 0.1058** (强证实 Poincaré 损失)
- 次目标: HG-Rec R@10 **> 0.06** (与 paper 数量级一致)

---

## 4. 预算

| 阶段 | 估算时间 | GPU |
|------|---------|-----|
| **Stage 1: HRQ-VAE 训练** (1000 epoch) | ~1-2h | GPU 1 或 3 (空闲) |
| **Stage 2: Codebook 生成** | ~5 min | GPU 1 |
| **Stage 3: HG-Rec 训练** (200 epoch T5-small) | ~3-5h | GPU 1 或 3 |
| **Stage 4: 评估** (beam=20) | ~30 min | GPU 1 |
| **总计** | **~5-7h** | |

**GPU 资源约束 (R7)**:
- GPU 0: Task #81 S3Rec 训练 (48 min, 还在跑)
- GPU 1: 空闲 ✅
- GPU 2: Task #83 P5-SID 训练 + evaluation 阶段
- GPU 3: 空闲 ✅
- **Stage 1 HRQ-VAE 用 GPU 1, Stage 3 HG-Rec 用 GPU 3 (避免单卡长时间独占)**

---

## 5. 风险与缓解

**风险 1**: sentence-t5 768d embedding 还没生成 (HG-Rec 默认需要 `item_emb.parquet`)
- 缓解: 用 `external/LLM-RecSys-ID/data/instruments/sentence-t5-768d.npy` 转换为 parquet
  (`pd.DataFrame(emb).to_parquet('HG-Rec/dataset/Instruments/item_emb.parquet')`)

**风险 2**: HG-Rec loss_type='poincare' 实现可能有 bug (这是新仓库, 第一次跑)
- 缓解: 先跑 Stage 1 短 epoch (10) 验证 loss 下降 + ckpt 可用, 再跑完整 1000 epoch

**风险 3**: T5-small 200 epoch 训练时间可能比估计长 (paper 报告 4-6h)
- 缓解: 监控 loss + val, 提早 early stop (monitor NDCG@10 plateau)

**风险 4**: HRQ-VAE 残差空间几何 (Step 1 训练后) 是否**真的**比 vanilla RQ-VAE 更双曲?
- 缓解: 跑完 Stage 1 后用 Task #92 δ-hyperbolicity 测 HRQ-VAE 残差, 对比 vanilla RQ-VAE
  (Task #92) + phonism (Task #79). 如果 HRQ-VAE 残差 δ 显著**小于** vanilla, 说明 Poincaré
  loss **确实**把双曲信号传到残差空间. 这是 Task #70 → 残差空间的因果链证据.

**风险 5**: HG-Rec `codebook_size=[64,128,256,1]` 第四个 1 是什么意思?
- 缓解: 看 `gen_codebook.py` + `data/dataset.py` 是否处理变长 + 第四层特殊 token

---

## 6. 完成度跟踪

- [ ] **Stage 1 启动**: `python3 train_hrqvae.py ...`
- [ ] **Stage 1 完成**: best HRQ-VAE ckpt 落盘 (loss_type=poincare)
- [ ] **(可选) Step 1.5 残差几何分析**: Task #92 δ-hyperbolicity on HRQ-VAE residuals
- [ ] **Stage 2 启动**: `python3 gen_codebook.py ...`
- [ ] **Stage 2 完成**: codebook .npy 落盘
- [ ] **Stage 3 启动**: `python3 train_HG-Rec.py --mode train ...`
- [ ] **Stage 3 完成**: best HG-Rec ckpt 落盘
- [ ] **Stage 4 启动**: `python3 train_HG-Rec.py --mode evaluation ...`
- [ ] **Stage 4 完成**: Recall/NDCG 输出
- [ ] **写 verdict**: `verdicts/task84_hgrec_main_repro_instruments_result.md` (含 result: 行)
- [ ] **更新 loop.md §16**: 标记 Task #84 完成 + 写入 Task #87 综合排名

---

## 7. 引用

- HG-Rec paper (ICML 2026): "Hyperbolic RQ-VAE enhanced Generative Recommendation with
  Differential-Length Codebook Strategy" (待 paper PDF 找到, 目前只有 README + code)
- HG-Rec 仓库: `/home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/`
- HG-Rec README: 1 行, ICML 2026 + Differential-Length Codebook
- Task #70 Ollivier curvature: `verdicts/task70_olliver_curvature_result.md`
- Task #80 per-layer κ: `verdicts/task80_per_layer_kappa_idea1_result.md` (含 §11 三重支撑)
- Task #82 弱信号 + 网格加密: `verdicts/task82_weak_signal_grid_refinement_result.md`
- Task #84 (TIGER inference): `verdicts/task84_tiger_inference_result.md`
- Task #85 (FDSA eval): `verdicts/task85_fdsa_test_eval_result.md`
- Task #87 综合排名框架: `verdicts/task87_paper_table2_summary_framework.md`
