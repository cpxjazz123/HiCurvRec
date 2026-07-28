# Task #169 — κ-Stereographic + Sinkhorn 组合 (T5-mini 9.18M)

> **任务目的**: 验证 κ-Stereographic 距离 + Sinkhorn 正则化组合是否能让 test R@10 > baseline 0.1058

> **完成日期**: (in progress)
> **状态**: 🟡 待启动

---

## 1. 背景

承接 Task #165/166/167/168 ablation across capacities (4 档全部 test R@10 < baseline):
- #168 T5-5.5M 0.0997 (-5.8%)
- #166 T5-mini 9.18M 0.1019 (-3.7%)
- #165 v3 T5-small 60M 0.0965 (-8.8%)
- #167 T5-base 220M [待]

**关键发现**: 当前 κ-Stereographic Phase A/B 训练使用 `--sk_epsilons 0.0 0.0 0.0` (**完全禁用 Sinkhorn**)。但 baseline (vanilla+Sinkhorn phonism R@10=0.1058) 使用 `--sk_epsilons 0.0 0.0 0.003` (L2 only)。

参考 `task79_phonism_delta_hyperbolicity.py:79` 注释:
```python
print(f"  sk_epsilons (inferred): {[0.0 if i < num_layers - 1 else 0.003 for i in range(num_layers)]}")
```

**Root cause 假设**: κ-Stereo distance + argmin (no Sinkhorn) 可能丢失 baseline 的 entropy regularization 优势。

**假设**: 在保持 κ-Stereographic 距离公式不变前提下, 加 Sinkhorn (L2 only) 能修复 downstream test R@10 退化。

## 2. 实验设计

**变量**: RQ-VAE Stage 1 `--sk_epsilons` 从 `[0.0, 0.0, 0.0]` 改为 `[0.0, 0.0, 0.003]`
**保持不变**:
- κ-Stereographic 距离公式 (在 hrqvae_free_curv.py 中不变)
- Phase A/B 训练调度 (100 ep κ frozen + 100 ep κ unfrozen lr_theta=1e-5)
- Stage 3: T5-mini 9.18M (4+4 layers, d_model=256, d_ff=1024, 4 heads × d_kv=64)
- 其他所有超参 (θ_init=[0,0,0], κ_max=2.0, β=0.5, lr=1e-3, num_emb_list=[32,64,256], seed=42)

**Stage 1 命令** (相对于 task164_part2_kappa_decouple.sh):
```bash
python3 /home/wlia0047/ar57/wenyu/GeneRec/scripts/task89_stage1_train_rqvae.py \
    --M 1 \
    --epochs 200 \
    --batch_size 256 \
    --lr 1e-3 \
    --lr_theta 1e-5 \
    --theta_init_list 0.0 0.0 0.0 \
    --kappa_max 2.0 \
    --kappa_freeze_epochs 100 \
    --seed 42 \
    --num_emb_list 32 64 256 \
    --e_dim 32 \
    --layers 512 256 128 \
    --loss_type poincare \
    --beta 0.5 \
    --quant_loss_weight 1.0 \
    --sk_epsilons 0.0 0.0 0.003 \  # ← 关键改动: L2 启用 Sinkhorn
    --sk_iters 50 \
    --kmeans_init \
    --kmeans_iters 1000 \
    --data_path /home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/item_emb.parquet \
    --device cuda:0 \
    --ckpt_dir $PROD_DIR
```

**Stage 3 命令**: 复用 #166 T5-mini launcher, 但 code_path 改用新 SID npy 文件.

## 3. 决策触发(vs baseline)

| 条件 | 结果指标 | 决策 |
|------|----------|------|
| test R@10 > 0.1058 | PASS | Stop hook 条件满足 |
| 0.1019 ≤ test R@10 ≤ 0.1058 | 持平 baseline | Sinkhorn 微小影响, κ-Stereo 本身不支持 test gain |
| test R@10 < 0.1019 | 退化 | κ-Stereo + Sinkhorn 比纯 κ-Stereo 更差, NO-GO |

## 4. 预算

| 阶段 | 估算时间 |
|------|---------|
| Stage 1 Phase A/B 训练 | ~2 min (per #164 actual) |
| Stage 2 codebook 推断 | ~1 min |
| Stage 3 T5-mini 训练 (counter=20) | ~40 min |
| Stage 4 评估 | ~3 min |
| 总计 | **~45 min** |

## 5. 风险与缓解

**风险 1**: κ-Stereo + Sinkhorn 在 L2 可能让 codebook 退化到 vanilla 模式, κ 没学到东西.
**缓解**: 监控 κ_history.json, 看最终 κ_m 是否仍 ≈ -0.09 (跟 #164 接近). 如果 κ 跑到 0 附近, 说明 Sinkhorn 把 κ 的几何信号拉平了.

**风险 2**: Stage 2 SID npy 跟 #164 不同, Stage 3 重新训练.
**缓解**: 这是预期行为, 直接复用 T5-mini 训练 launcher.

**风险 3**: hrqvae_free_curv.py 改 `--sk_epsilons` 后 Stage 1 不能启动 (e.g., 类型不匹配).
**缓解**: 已验证 task89_stage1_train_rqvae.py 接受 `nargs='+'` 列表参数, 应兼容.

## 6. 完成度跟踪

- [ ] Stage 1 Phase A/B 训练 (with Sinkhorn L2)
- [ ] Stage 2 codebook 推断 (产出新 SID npy)
- [ ] Stage 3 T5-mini 9.18M 训练
- [ ] Stage 4 评估
- [ ] 写 verdict + 更新 loop §16

## 7. R11.3 自主决策记录

| 决策点 | 选择 | 理由 |
|--------|------|------|
| T5 容量 | T5-mini 9.18M | 4 档中最接近 baseline 的容量 (-3.7% gap), 最有可能突破 |
| Sinkhorn 强度 | sk_eps=0.003 (跟 baseline 一致) | 用户选了"试 Sinkhorn 组合", 复用 baseline 配置 |
| Sinkhorn 作用层 | 仅 L2 (跟 baseline 一致) | baseline 配置 `[0.0, 0.0, 0.003]`, 保持一致 |
| Phase A 时长 | 100 ep (跟 #164 一样) | 不改变 κ-decouple 调度, 隔离 Sinkhorn 变量 |