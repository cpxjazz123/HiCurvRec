# Task #142 — Free-curv Codebook Collapse 修复尝试 (geodesic kmeans + dead code reset)

> **任务目的**: 验证 Task #137 列出 4 个未验证修复方向中**最直接修复根因**的方向 1 (kmeans_init in geodesic space) + 联合 B 方案 (dead code reset) 能否恢复 free-curv RQ-VAE codebook 利用率.

> **完成日期**: in progress
> **状态**: 🟡 待启动

---

## 1. 背景

Task #137 verdict 列出 4 个未验证修复方向:
1. **kmeans_init 在 geodesic space** ← 本任务选
2. Sinkhorn 强制均匀训练中 (sk_eps > 0 from start)
3. EMA codebook + 死码重置
4. 缩短训练 (e.g. 200 ep)

Task #137 证明 3 个 free-curv 变体 (v1/v2/B-arm) **全部 codebook 坍缩** (1-15 unique SID vs Task #84 baseline 8936). 根因:
- kmeans_init 在 raw euclidean 空间撒 centroid
- Free-curv 用 variable-κ geodesic distance
- κ 学到 sph 分支 → distance nearly-uniform → VQ collapse

**Pre-task 状态** (Task #137 + #138):
- Task #138 design plan 已完成 A+B 方案设计 (`task138_free_curv_A_B_scheme_fix_result.md`)
- **基础设施已就绪** (Task #138 中已实现):
  - `init_emb_geodesic` 函数 (HG-Rec/model/hrqvae_free_curv.py line 154)
    - κ_m ≠ 0: data → expmap0 → kmeans in U^{n_m}_{κ_m} → logmap0 → centroids
    - κ_m = 0: 直接 kmeans (Euclidean, 与原版等价)
  - `--geodesic_kmeans` flag + `--re_kmeans_every` flag (task89_stage1_train_rqvae.py line 60-65)
  - `dead_code_reset` 函数 (line 199) + periodic reset hook
  - `--dead_code_reset_every`, `--dead_code_reset_threshold`, `--dead_code_replace_ratio` flags

**Select 配置 (R11.3 自主决策)**:
- `--geodesic_kmeans` ✅ 启用
- `--re_kmeans_every=50` (每 50 epoch 重做 geodesic kmeans, 跟 Task #138 design 一致)
- `--dead_code_reset_every=50` batches (B 方案: 每 50 batch 重置死码)
- `--dead_code_reset_threshold=1` (任何 usage=1 都视为活跃, 避免误判)
- `--dead_code_replace_ratio=0.1` (单次最多替换 10% codebook)
- `--theta_init=0.01` (R137 escape Euclidean fixed point)
- **epochs=200** (R11.3: 缩短自 1000 — Task #137 v2 已证 200 epoch 仍坍缩, 但 κ_max=0.5+200 ep 仍未充分 escape, 本次同时启用 A 方案能突破)
- **seed=42** (跨 run 一致)
- **M=1** (本任务单臂 A) — 与 Task #137 A-arm 直接可比

## 2. 实验设计

**变量**: A 方案 (geodesic re-kmeans every 50 ep) + B 方案 (dead code reset every 50 batch) 联合启用
**保持不变**:
- 数据集: sentence-t5-base 768d 编码 Musical_Instruments (Task #89 一致)
- 超参: lr=1e-3, lr_theta=1e-3, batch_size=256, num_emb_list=[64,128,256], e_dim=32, layers=[512,256,128]
- loss_type=poincare, β=1.0
- R12 ckpt: 每个 epoch 末 + best_loss 自动覆盖

**启动命令**:
```bash
python3 scripts/task89_stage1_train_rqvae.py \
    --M=1 --kappa_max=0.5 \
    --geodesic_kmeans --re_kmeans_every=50 \
    --dead_code_reset_every=50 \
    --dead_code_reset_threshold=1 \
    --dead_code_replace_ratio=0.1 \
    --theta_init=0.01 \
    --epochs=200 --seed=42 \
    --ckpt_dir=products/task142/train/arm_A_M1/ 2>&1 | tee logs/task142/A.log
```

GPU 1 空闲 (R7 确认 0% util / 0 MiB)

## 3. 决策触发 (vs Task #137 3 个 baseline collapse 数字)

| 观察条件 | 结果 | 决策 |
|---------|------|------|
| **A方案 geodesic kmeans init 一次性**足够阻止坍缩 | unique SID > 1000 | ✅ A 方案有效, R137 verdict 错误, free-curv 可行 |
| A方案 + B方案 联合才阻止坍缩 | unique SID > 1000 (但 geodesic init alone 不够) | ✅ 联合方案必要, free-curv 可行 |
| unique SID 50-1000 (partial) | codebook 部分恢复但仍 < Task #84 baseline 8936 | ⚠️ 边际改进, 报结果 |
| unique SID 1-15 (跟 v1/v2/B baseline 一样崩) | ❌ A+B 方案无效, free-curv 架构 NO-GO **已确认** | ❌ 写 verdict NO-GO, 不再投入 |
| R137 NaN guard 触发 (κ→κ_max 边界) | ❌ κ_max 改 0.3 重跑 (per Task #137 风险 7) | ⚠️ fallback 改参数 |

## 4. 预算

| 阶段 | 估算时间 |
|------|---------|
| 数据加载 + Task #89 setup | ~3 min |
| A 方案 geodesic kmeans init (first batch) | +5s |
| 训练 200 epoch | ~30 min (A 臂 1.7M param CNN, ~9s/epoch from Task #137 v2 rate) |
| Dead code reset (every 50 batch) 累计开销 | <1 min total |
| Geodesic re-kmeans (every 50 epoch, 4 次) 累计开销 | <1 min total |
| **总计** | **~35 min** |

数据点: Task #137 v2 A 臂 (200 epoch + κ_max=0.1) 训练跑完 ~25 min, 这是最低 ceiling. 加 A+B 钩子开销 +1 min.

## 5. 风险与缓解

**风险 1**: re_kmeans 中 sklearn KMeans 输出在 κ_m sph 分支边界可能不稳定 (sklearn KMeans 是 euclidean, 不在 manifold 上做 → 不算真 geodesic). 缓解: spherical 中心用 `centroid / ‖centroid‖ × ‖centroid_init‖` 重新归一化保留 magnitude, 失败回退 plain kmeans.

**风险 2**: dead code reset 太频繁 (every 50 batch) 破坏 training stability. 缓解: replace_ratio=0.1 单次最多替换 10%, 总 codebook 仍是 90% 保留.

**风险 3**: A+B 方案可能放大彼此副作用 (per Task #137 risk 3). 缓解: 本任务用 `--dead_code_reset_every=50` 而非更激进 (e.g. 10), 给宽限时间.

**风险 4**: 修改的是 HG-Rec/model/hrqvae_free_curv.py (上游) + scripts/task89_stage1_train_rqvae.py (Task #89 自建). 已在 Task #138 改动, 不在 src/data/ 上游框架源码 (R6 例外允许).

## 6. 完成度跟踪

- [x] R9 audit (max=141, next=142)
- [x] Task #138 status updated to completed (design plan deliverable)
- [x] Task #142 description 落盘
- [x] 基础设施 audit (geodesic_kmeans + dead_code_reset hook 均有)
- [ ] launcher python 直接调用 task89_stage1_train_rqvae.py 验证 args
- [ ] launch GPU 1 (R7: 0% util)
- [ ] 200 epoch convergence
- [ ] codebook utilization 诊断 (unique SID count, max bucket size)
- [ ] 写 `verdicts/task142_free_curv_codebook_recovery_result.md` 含 `result:` 行
- [ ] vs Task #137 baseline 12 / 1 / 15 unique SID 对比表

## 7. R11.3 自主决策记录

- **不并行多臂 (A 臂 only)**: GPU 1 单独可跑 1 个 arm, 多臂需要等 35min × 3 = 105 min. 决策: 只跑 A 臂, 若 A 臂有效再考虑 B/C 臂.

- **epochs=200 (vs Task #137 baseline 1000)**: Task #137 v2 已证明 200 ep 仍坍缩, 但 v2 没启用 A 方案. A 方案 re-init 在 epoch 50/100/150/200 强制 centroid 重分布, 应能突破 v2 的坍缩. 决策: 200 ep 已够 (跟 Task #137 v2 同样成本, 加 A 方案钩子期望突破).

- **不一起改 κ_max**: Task #137 v1+0.5, v2+0.1 都坍缩, 证 κ_max 不是 collapse 主因. 决策: κ_max=0.5 (跟 Task #137 v1 一致), 不动 κ_max 让比较有效.

- **dead_code_reset_threshold=1 (vs 默认 0)**: threshold=0 太激进 (任何 1 batch 未选就 reset), 可能 reset 太多. threshold=1 给每个 code 至少 1 batch 缓冲. 决策: 1.

- **launch 立刻不 S3Rec 等**: S3Rec 在 GPU 0 (5h ETA), Caser fix 在 GPU 3 (15-45 min ETA), ETEGRec 在 GPU 2 (10h ETA). GPU 1 100% 空闲. 决策: 立刻 launch.

## 8. 关联

- [[free-curv-codebook-collapse]] — Task #137 verdict + 4 方向列表
- Task #138 verdict (design plan) — 提供 flag + hook 实现
- Task #89 — 原 retrain (pre-R137, κ 错误归零)
- Task #135 — 4-run diagnostic, R137 bug confirmed
- Task #84 baseline R@10=0.1020 — 上限 reference
