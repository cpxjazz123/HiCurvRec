# Task #149 — 3-层异质 κ 学习验证 (per-layer diverse κ INIT)

> **任务目的**: 验证用户 2026-07-24 明确 "我希望三层学习到三个不一样的k, 而不是0"。给 3 个 RQ-VAE 层用显著不同的 θ_init (-0.5, 0.0, +0.5), 让 κ 自由学习 (lr_theta=5e-3, no freeze), 看 ep 200 终态 3 层是否学习到 3 个不一样 κ 值 (而不是都掉回 0).
> 
> **承接**: Task #89 NO-GO (全 θ=0 起点 + 全 M=1 κ 冻结 → 18 κ_m 都 = 0); Task #144 Phase B (θ 解冻但 init 同样 → κ 不动, lr 太小); Task #145 (κ 冻结 + 软量化). 这三条都验证了 "θ_init=0 起点 = κ 全 0 终点", 无法分离 "数据本身是欧氏" vs "起点太偏导致不动" 两个 confounder. Task #149 用 3 个不一样的起点打破对称, 直接验证自然训练动力学是否能保持 κ 异质性.

> **完成日期**: in progress
> **状态**: 🟡 待启动

---

## 1. 背景 (用户 2026-07-24 反馈)

用户在 Task #145 Stage 1 完成后看到 κ 全程 = 0, 明确指出原假设错误:

> "不对, 我现在的要求是: 我希望三层学习到三个不一样的k, 而不是 0"

- Task #89 + Task #137 + Task #144 Arm A 三组实验 κ 全 = 0 是因为 θ_init=0 + (lr=0 or lr 极小), "起点对称 + 没法动" → 自然不分裂.
- 但如果 **给 3 层非对称起点** (-0.5, 0.0, +0.5), κ 在自然 lr 下是否能保持 3 个不同值? → 数据几何信号 vs 模型正则化倾向 的对决.

## 2. 实验设计

**变量** (单变量改动):
- `--theta_init_list = -0.5 0.0 0.5` (3 layers, **唯一改动**)

**保持不变** (跟 Task #144 Arm A baseline 对照):
- M=1, num_emb_list=[64,128,256], e_dim=32
- `--kappa_max=2.0`
- `--lr_theta=5e-3` (Task #89 默认, 不是 Task #144 Phase B 的 1e-5)
- `--kappa_freeze_epochs=0` (**不冻结**, 自然学习)
- `--geodesic_kmeans` (Task #89 A方案防 collapse)
- `--dead_code_reset_every=100`, threshold=0.0 (Task #89 B方案)
- `--utilization_freeze_threshold=0.05` (Task #144 保护: 但若 κ 漂移致 util 暴跌, 永久冻结 κ)
- epochs=200, seed=42
- soft_vq **不启用** (本任务跟软量化正交)
- GPU 3 (R7 检查空闲)

**启动命令**:
```bash
CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task149 \
  python3 scripts/task89_stage1_train_rqvae.py \
    --M=1 \
    --num_emb_list=64 128 256 \
    --e_dim=32 \
    --layers=512 256 128 \
    --kappa_max=2.0 \
    --theta_init_list=-0.5 0.0 0.5 \
    --lr_theta=5e-3 \
    --kappa_freeze_epochs=0 \
    --utilization_freeze_threshold=0.05 \
    --geodesic_kmeans --re_kmeans_every=50 \
    --dead_code_reset_every=100 \
    --epochs=200 --seed=42 \
    --ckpt_dir=products/task149/train/main_heterokappa \
    --kappa_log_path=products/task149/train/main_heterokappa/kappa_history.json \
    --phase_a_baseline_util_path=products/task149/train/main_heterokappa/phase_a_baseline.json
```

**Stage 2 → 3 → 4**: 跟 Task #144 同模板但换路径:
- Stage 2: `Instruments_t5_hrqvae_heterokappa.npy`
- Stage 3 T5: `task84_hgrec_stage3_train.py --code_path _t5_hrqvae_heterokappa.npy --save_path products/task149/ckpt`
- Stage 4 eval: `task144_stage4_eval.py` fork, R@5/10/20 + N@5/10

## 3. 决策触发 (vs Task #89/Task #144 baseline)

| 条件 | 结果 κ 模式 | 决策 |
|------|----------|------|
| ep 200 终态 3 κ_m 仍然 3 个**显著不同** (e.g. [-0.3, 0.0, +0.4] > 0.05 跨度) | κ 异质性保持 | ✅ **数据几何信号存在**, 写 verdict 表明 "代码 + 起点能学到差异化 κ", 论文 Section 5.4 加 "起点对称性问题" 讨论 |
| ep 200 终态 3 κ_m **都收敛到 ~0** (abs(κ) < 0.05) | 起点被正则化拉回 | ⚠️ **算法/正则化倾向欧氏**, 跟 Task #89 同结论但新证据: 即便起点偏, optimizer 仍把 κ 拉回 0. 写 verdict 表明 "这是模型 regularization, 不是数据缺信号" |
| ep 200 终态 3 κ_m 跟其他冲突 (冻结过早/NaN/空 codebook) | 实现 bug 或 instability | NO-GO, 撤回, 跟 Task #89/Task #145 verdict 合并 |

## 4. 预算

| 阶段 | 估算时间 | GPU |
|------|---------|-----|
| Stage 1 | ~5 min (跟 Task #89 量级一致, 200 epoch) | GPU 3 |
| Stage 2 codebook | ~5 min | GPU 3 |
| Stage 3 T5-small | ~30-60 min (early stop 期望 ep 30-50) | GPU 1 |
| Stage 4 eval | ~1 min | GPU 1 |
| **总计** | **~45-75 min** | GPU 3→GPU 1 |

## 5. 风险与缓解

**风险 1**: κ 自由学习可能触发 R137 NaN (acos 在 κ→κ_max 边界不稳定).
**缓解**: `--utilization_freeze_threshold=0.05` (Task #144 保护, lr_theta=0 永久冻结), 加 R137 NaN guard raise.

**风险 2**: 用 geodesic_kmeans (Task #89 A方案) 让 init 已经偏向 hyp 流形, 跟本任务 "起点 -0.5/0/+0.5" 叠加可能让 layer 0 更强 hyp bias.
**缓解**: 这是合理叠化 (geodesic init 是初始化手段 + θ_init 是偏置起点, 两者不冲突), 不调整.

**风险 3**: 异质 κ 初始化让 layer 0/2 的 cluster 几何跟 Task #144 Arm A (κ=0) 完全不同, SID 维度分离度可能更好.
**缓解**: 这是 goal — 若真如此, 下游 R@10 应高于 baseline 0.1020, 验证成功.

## 6. 完成度跟踪

- [x] Stage 1 launch on GPU 3
- [ ] Stage 1 跑完 200 epoch + κ_history 终态确认 (≠ 0 vs 全 = 0)
- [ ] Stage 2 codebook → SID .npy
- [ ] Stage 3 T5-small Stage 3 训练
- [ ] Stage 4 test eval → R@5/R@10/R@20 + N@5/N@10
- [ ] 写 verdict: `verdicts/task149_heterogeneous_kappa_three_layers_result.md`
- [ ] 更新 loop.md §16 (R8 旧清理), 更新 §15 backlog 状态
