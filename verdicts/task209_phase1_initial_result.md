# Task #209 Phase 1 初次尝试 (5-epoch run) — NO-GO

> **完成日期**: 2026-07-26
> **状态**: ❌ FAIL (硬投影夹断方向梯度, 4 臂全部 mode collapse)

---

## §1 实验条件

5 臂 × 1000 epoch × 4 GPU 并行跑:
- A0: 不跑 (用 #181 baseline 已有 ckpt)
- A1: `--radii 1.0 1.35 1.70` (硬 norm 投影)
- A2: A1 + `--scale_norm poincare`
- A3: A2 + `--w_path 1.0 --path_geometry hyp` (完整方法)
- A4: A3 with `--path_geometry euc` (关键对照)

发现的问题: **硬 norm 投影 (`tangent_norm * F.normalize(codebook)` + `tangent_norm * F.normalize(latent)`)** 在纯 hyp forward 路径中干扰了 kmeans_init 的正常方向分散, 造成 mode collapse.

## §2 跨 4 臂结果

| 臂 | Best Train Loss | Best Collision Rate | hypnorm L0/1/2 (raw → projected fill) |
|----|------|------|--------------------------------------|
| A1 (radii only) | 21.71 | 0.994 | mean=0.109 / 0.396 / 0.261 |
| A2 (+scale_norm) | 74.88 | 0.996 | mean=0.097 / 0.397 / 0.261 |
| A3 (+w_path hyp) | 95.51 | **0.9999** | mean=0.085 / 0.393 / 0.255 |
| A4 (+w_path euc) | 93.17 | 0.995 | 同 A3 |

#181 baseline (无 --radii, epoch ~1000):
- Best CollRate ≈ **0.06** (6%) ← baseline 应有水平
- hypnorm L0/1/2 mean = 0.263 / 0.147 / 0.122

## §3 根因诊断

### §3.1 关键观察
- 即使纯 A1 (只加 `--radii`, 无 --w_path), collision_rate 也卡在 99.5% — 这意味着 **硬 norm 本身就是塌陷源**, 跟 path reg 无关.
- A3 collision_rate 0.9999 (perfect collapse), 说明 hard norm + path reg 让 encoder 更快被锁死.
- #181 baseline 没硬 norm, 允许 codebook 自由优化方向, 所以 collision 正常下降到 6%.

### §3.2 物理机理
硬 norm 投影 `codebook_n = tangent_norm * F.normalize(codebook)` 的雅可比矩阵具有切空间投影结构 (无径向梯度). 在 `latent_n = tangent_norm * F.normalize(latent)` 的对称投影下, 整个 forward 退化为**纯方向编码问题**:
- 32-dim 球面上, kmeans 初始化给出 64 个方向 (L0).
- 由于球面 vs 32-dim 的容量不足 + 切空间投影迫使所有向量同径, kmeans 早期微弱的随机性不足以把方向拉散.
- argmin 在相似的方向簇中收敛, ckpt 训练整个 epoch 都不动.

### §3.3 验证证据
A1 epoch 4 时 collision 已是 0.994, 一开始就是 mode collapse; ckpt 没动;
A3 epoch 0-1000 collision 一直 0.9999 不变 — 完全无学习.

## §4 修复方向 (Phase 1 retry)

### §4.1 改用 `--norm_target` 软约束代替 `--radii` 硬投影
```bash
--norm_target 1.0 1.35 1.70 \
--gamma_norm 0.5 \
```
这是 Task #196 的设计 (cutting 空间软 norm), 不夹断方向梯度, 让 kmeans_init 自然分散, 然后用 norm_loss 推逐步接近 r_target.

γ=0.5 是中等强度: 太弱 norm 不动 (跟不加一样), 太强又接近硬投影. 待 sweep.

### §4.2 同步移除硬投影 patch
我之前给 HVectorQuantization.forward 加的 `if self.rho is not None and ...` 硬投影段需要撤回 (只保留 product_manifold 分支原有行为, 不再扩到 pure hyp).

### §4.3 重跑 4 臂
预计 5 分钟 wall clock, 验证 collision 能否降到 ≤ 11%.

### §4.4 若软约束仍不行
进一步调查: 
- 是否 kmeans_init 数据集归一化不对 (latent norm 太大导致投影后方向混乱)
- 是否需要在 init_emb 阶段对球面 kmeans 加 namespace-aware 方向初始化

## §5 R11.3 决策

**主决策**: 立即回滚硬投影 + 换软约束, 不修改 task209 description 5 臂设计 (只是把 "--radii" CLI 改写成 "--norm_target + --gamma_norm" 同语义).

**备选**: 如果软约束仍 no-go, 则 Phase 1 直接进入失败模式, 在 verdict 里记录 "硬 norm 失败的物理证据", 此后 Phase 3 切片评估 + 机制指标叙事 — 任务目标从 "5 臂消融" 收窄为 "硬 vs 软 norm 行为差异" 单点论证.

## §6 当前产物 (保留供后续分析)

- `products/task209/phase1_arm_A{1,2,3,4}/` 4 个 ckpt (含 best_loss_model + best_collision_model + epoch_{N}_*.pth)
- `logs/task209/phase1_arm_*.log` 4 个训练日志
- `/tmp/launch_*.out` 启动日志
- `products/task209/_TRAINING_PID_arm_*` (PID 已停, 可清理)

**result:** ❌ Phase 1 FAIL — 4 臂 collision ≈ 0.99 (vs baseline 0.06), 根因: HVectorQuantization hard-norm 投影夹断 kmeans 方向, 触发 mode collapse. 修复: 改 `--norm_target + --gamma_norm` 软约束, 撤回硬投影 patch, 重跑 4 臂.
