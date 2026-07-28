# Task #138 — A-arm geodesic kmeans init (post-collapse root-cause fix)

> **任务目的**: 验证 free-curv codebook collapse 的修复方向 — 在 geodesic space (不是 euclidean) 做 kmeans_init + dead-code reset,看 A 臂 M=1 κ_max=0.5 是否仍坍缩 (root cause 已是 metric mismatch 而非 split).

> **完成日期**: (in progress)
> **状态**: 🟡 在跑 (GPU 1 launcher PID 2973935 等 C-arm 退出后启动; A 臂用 geodesic 距离 kmeans_init, 待启动)

---

## 1. 背景

承接 [Task #137 v2 failure analysis](../verdicts/task137_v2_failure_analysis.md) + [free-curv codebook collapse](../memory/free-curv-codebook-collapse.md):

- **根因**: kmeans_init 在 raw euclidean latent 空间, 但 VQ argmin 用 per-component geodesic 距离 → metric mismatch
- **三臂全部坍缩** (A M=1 12 unique, B M=2 15 unique, C M=3 1-15 unique) — 跟切分无关 (用户假设确认)
- **Sinkhorn 不恢复**: B-arm quick Stage 2 跑 5 iters 后仍 15 unique SID 锁死

**假设**: 如果 kmeans init 改在当前 κ_m 对应的 geodesic space, 应该跟 VQ argmin metric 一致, 不会再坍缩.

---

## 2. 实验设计

**变量**:
- A. **init_emb 改成 geodesic kmeans**: 在当前 κ_m 对应的 manifold space 跑 kmeans (而不是 raw euclidean)
- B. **dead code reset**: 训练过程中把 utilisation 接近 0 的 codeword 重置成 batch 内 random latent
- C. **保留** (可选项): Temperature Sinkhorn during training (sk_eps > 0 训练时也开)

**保持不变**: Task #137 baseline (θ_init=0.01, lr_theta=5e-3, M=1, κ_max=0.5, 1000 epoch, 64+128+256 codebook, 32d e_dim)

**启动命令**:
```bash
python3 /home/wlia0047/ar57/wenyu/GeneRec/scripts/task89_stage1_train_rqvae.py \
    --M 1 --epochs 1000 --batch_size 256 \
    --lr 1e-3 --lr_theta 5e-3 \
    --theta_init 0.01 --kappa_max 0.5 \
    --seed 42 --num_emb_list 64 128 256 \
    --e_dim 32 --layers 512 256 128 \
    --loss_type poincare --beta 1.0 --quant_loss_weight 1.0 \
    --sk_epsilons 0.0 0.0 0.0 --sk_iters 50 \
    --kmeans_init --kmeans_iters 1000 \
    --geodesic_kmeans_init --re_kmeans_every 100 \
    --dead_code_reset_every 100 --dead_code_reset_threshold 1.0 \
    --data_path /home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/item_emb.parquet \
    --device cuda:0 \
    --ckpt_dir /home/wlia0047/ar57/wenyu/GeneRec/products/task138/train/A_arm_geodesic \
    --kappa_log_path /home/wlia0047/ar57/wenyu/GeneRec/products/task138/train/A_arm_geodesic/kappa_history.json \
    --log_interval 10 --save_every 50
```

---

## 3. 决策触发 (vs Task #84 baseline)

| Stage 1 codebook 健康度 | 结果 | 决策 |
|------------------------|------|------|
| L0 utilization ≥ 80% AND L1 ≥ 80% AND L2 ≥ 80% | ✅ 修复生效 | 进入 Stage 2/3/4 下游 |
| 任一层 < 80% | ⚠️ 部分修复 | 加 C 方案 (Sinkhorn during train) 重跑 |
| 任一层 < 50% | ❌ 修复失败 | 切到 Task #84 baseline c555, 永久放弃 free-curv 方向 |

最终 κ_m 稳定区间:
- κ_m ∈ [-0.5, +0.5] (非边界饱和): ✅ 干净读数,可对照 Task #88 c555 网格
- κ_m → 0 + small jitter: ✅ "数据本质欧氏" 复证 (Task #89 已确认, 双保险)
- κ_m → κ_max 边界饱和: ❌ 跟 v2 一样坍缩, 修复无效

---

## 4. 预算

| 阶段 | 估算时间 |
|------|---------|
| A 臂 1000 epoch 训练 | ~50 min (Task #137 A 臂速度参照) |
| Stage 2 codebook + Stage 3 + 4 评估 | ~30 min |
| 总计 | ~1.5 h |

---

## 5. 风险与缓解

**风险 1**: geodesic kmeans 数学实现可能跟 euclidean 不同 (expmap0 后跑 KMeans vs 直接 expmap), 需要 unit test 验证.
  → 缓解: scripts/task138_geodesic_kmeans_unit_test.py (10 min sanity check)

**风险 2**: A 臂跑完后可能跟 C-arm κ 趋势一致,仍不能解决"κ 选择饱和到 κ_max"问题.
  → 缓解: C 方案 (Sinkhorn during train) 留作 follow-up, 不混到本任务结果里

**风险 3**: Task #137 C 臂占用 GPU 1 (ep 840/1000, ~50 min 后完成), 本任务必须等.
  → 缓解: 等 C 臂自然退出后启动, 不抢 GPU

---

## 6. 完成度跟踪

- [x] Description (R9 retroactive fill, 2026-07-24)
- [x] R9 contiguous 1-138 (含 task138 description retroactive fill)
- [x] scripts/task138_A_arm_geodesic_init.sh launcher (脚本已写)
- [ ] scripts/task138_geodesic_kmeans_unit_test.py (代码改动前的 sanity check)
- [x] Patch hrqvae_free_curv.py 加 geodesic_kmeans_init + dead_code_reset 标志
- [x] py_compile PASS
- [ ] C-arm 退出后启动 A 臂 geodesic_init 训练 (GPU 1)
- [ ] 训练完成 + Stage 2/3/4 下游 (等)
- [ ] 写 verdict `verdicts/task138_a_arm_geodesic_kmeans_result.md`

result: Task #138 — A 臂 geodesic_kmeans_init launcher ready, 等待 C 臂退出 (Task #137 ep 840/1000 ~50min ETA)
