# Task #194 — K0 码本容量扫描 (机制因果验证 #3)

> **任务目的**: 验证 K0 码本容量是否是真 controlling variable — Task #192+#193 已经否证 β 跟 encoder, 下一候选是 K0. 单数据集 (Instruments) 剂量-反应: K0 ↑ → collision 应该单降, final/min 比值应该单降.

> **完成日期**: (in progress)
> **状态**: 🟡 待启动

> **R9 说明**: 严格按 R9, max+1 = 193, 但 #193 在 descriptions/task192_beta_dose_scan.md §1.2 + verdicts/task192_193_mechanism_causal_verdict.md 已用于 encoder freeze 实验, 不再重用. 新 ID = 194.

---

## 1. 背景

Task #192 (β 剂量) 跟 Task #193 (encoder freeze) 否证了 β 跟 encoder 这两条调节通道. 综合 Task #190 (码字/残差比值 +24%) 跟 Task #191 (L0_err 单调↑ r=0.861), mechanism 重写为 **decoder-driven**:

> 当 encoder 权值固定后, residual ‖r‖ 仍持续增长 (从训练 loss 是 decoder 重建 + quant loss; 即使 encoder 不变, decoder 重建需要 z 更宽分布来拟合不同 item). 这意味着 decoder 推动 encoder 输出分布扩张 → L0_err ↑ → collision ↑.

下一个 controlling variable 候选: **L0 码本 K0 容量**. 假设:
- K0 ↑ → 码字更密集 → 跟 z 距离更短 → L0_err ↓ → collision ↓

判据:
- K0 ↑ → final collision 单调↓
- K0 ↑ → final/min 比值单调↓ (不同 K0 退化比例不同, K0 大退化慢)

---

## 2. 实验设计

**变量**: K0 容量 (L0 码本大小)
- 4 臂: K0={32, 64, 128, 256}
- 默认 (K0=64) 是 Task #188 / #192 / #193 baseline

**保持不变**:
- num_emb_list: 改 K0, 保持 L1=128, L2=256 (跟 baseline 一致)
  - K0=32 → `--num_emb_list 32 128 256`
  - K0=64 → `--num_emb_list 64 128 256` (baseline)
  - K0=128 → `--num_emb_list 128 128 256`
  - K0=256 → `--num_emb_list 256 128 256`
- e_dim=32, β=0.5, loss_type=poincare, kmeans_init=True
- epochs=500 (足够看 collision 演化, 不需 1000)
- --save_limit 50 (保留 collision 谱系)
- 其他 recipe 跟 Task #188 phase 1 完全一致

**启动命令** (per 臂):

```bash
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys

REPO=/home/wlia0047/ar57/wenyu/GeneRec
export PYTHONPATH=$REPO/HG-Rec:$PYTHONPATH
export HF_HOME=/home/wlia0047/ar57_scratch/wenyu/hf_models
mkdir -p $REPO/logs/task194 $REPO/products/task194

cd $REPO/HG-Rec
GPU=0  # 每个臂分配到 cuda:0/1/2/3
CACHE_DIR=/home/wlia0047/.triton/cache_task194_k0${K0}
OUT_DIR=$REPO/products/task194/hrqvae_k0${K0}

mkdir -p $CACHE_DIR $OUT_DIR
export CUDA_VISIBLE_DEVICES=$GPU
export TRITON_CACHE_DIR=$CACHE_DIR
mkdir -p $TRITON_CACHE_DIR

python3 -u train_hrqvae.py \
    --lr 1e-3 \
    --epochs 500 \
    --batch_size 256 \
    --num_workers 4 \
    --eval_step 5 \
    --learner AdamW \
    --lr_scheduler_type linear \
    --warmup_epochs 20 \
    --data_path ./dataset/Instruments/item_emb.parquet \
    --weight_decay 0.0 \
    --dropout_prob 0.0 \
    --loss_type poincare \
    --kmeans_init True \
    --kmeans_iters 1000 \
    --sk_epsilons 0.0 0.0 0.0 \
    --sk_iters 50 \
    --num_emb_list ${K0} 128 256 \
    --e_dim 32 \
    --quant_loss_weight 1.0 \
    --beta 0.5 \
    --layers 512 256 128 64 \
    --save_limit 50 \
    --device cuda:0 \
    --ckpt_dir $OUT_DIR
```

---

## 3. 决策触发 (vs baseline K0=64)

| K0 | final collision (预计) | 假设验证 | 决策 |
|----|------------------------|----------|------|
| 32  | > 0.15 (高 collision) | K0 容量小 → collision 高 ✅ | K0 单调性通过 |
| 64  | ≈ 0.13 (baseline) | 对照 | — |
| 128 | < 0.10 (中 collision) | K0 容量大 → collision 降 ✅ | K0 单调性通过 |
| 256 | < 0.06 (低 collision) | K0 大 → collision 接近 0 ✅ | K0 单调性通过 |

| 比值指标 | K0=32 | K0=64 | K0=128 | K0=256 | 趋势 |
|---------|-------|-------|--------|--------|------|
| final/min 比值 | ≥1.6 | ≈1.48 | ≤1.3 | ≤1.15 | **单调↓** → K0 是 controlling variable ✅ |

**否证条件**:
- K0 ↑ 但 collision 不单降 → ❌ K0 不是 controlling variable
- final/min 比值对 K0 无反应 → ❌ K0 不是 controlling variable

---

## 4. 预算

| 阶段 | 估算时间 |
|------|---------|
| 4 臂 Stage 1 训练 (500 epoch × 4 GPU 并发) | ~50 min |
| Stage 2 SID 生成 (4 臂 × Sinkhorn) | ~5 min |
| Stage 3 T5-mini 训练 (4 臂 × 200 epoch) | ~30 min |
| Stage 4 评估 (4 臂 × Stage 4) | ~5 min |
| K0 L0_err 诊断 (复用 task191) | ~5 min |
| **总计** | **~95 min** (4 GPU 全占) |

---

## 5. 风险与缓解

**风险 1**: K0=256 时码字过多, kmeans_init 慢 → K-means 1000 iter 顶到极限
**缓解**: 已设置 `--kmeans_iters 1000`. 若失败, 降到 500 iter.

**风险 2**: K0=32 容量太小, 训练初期 collision 极高 (>0.30), 早停后没保存 ckpt
**缓解**: 不设 early stop, --save_limit=50 保证有 snapshot.

**风险 3**: 4 臂 × 500 epoch 跑下来 ~50 min, 万一某臂 早期发散需要重跑 → 时间翻倍
**缓解**: 先 fire K0={64, 128} 两个最有信号的臂, 看到趋势再决定跑 K0={32, 256}.

**风险 4**: 复用 task191 诊断脚本时, 4 臂 ckpts 路径不在原路径上 → 需要改 path
**缓解**: 复用 task191 脚本骨架, 把 CKPT_DIR 改成 products/task194/hrqvae_k0${K0}/ 的子目录.

---

## 6. 完成度跟踪

- [ ] Stage 1 launch (4 臂并发, 4 GPU)
- [ ] Stage 1 finish (collision 收敛判据)
- [ ] Stage 2 SID 生成 (4 臂 Sinkhorn + 4th-digit dedup)
- [ ] K0 L0_err 诊断 (4 臂 × 多 epoch)
- [ ] Stage 3 T5-mini 训练 (4 臂, 200 epoch)
- [ ] Stage 4 评估 (4 臂, R@10 落盘)
- [ ] 写 verdict (task194_k0_capacity_result.md) + 更新 paper section

---

## 7. 关联任务

| 任务 | 结论 |
|------|------|
| Task #191 | L0_err 单调↑ r=0.861 — 机制坐实 |
| Task #192 | β dose-response ❌ — β 不控 |
| Task #193 (encoder freeze) | encoder freeze ❌ — encoder 不控 |
| **Task #194 (K0 容量)** | K0 容量是否单因素控制 ← **本任务** |

---

**决策原则 (R11.3)**: 4 臂同 recipe, 单变量 K0, GPU 0-3 各 1 臂并发. 复用 task191 诊断, 不重新设计. 跑完直接写 verdict.