# Task #211 Phase 1 — C1/C2/C3 三臂 Stage 1 训练结果

> **完成日期**: 2026-07-26
> **状态**: ❌ **NO-GO** (3 臂全失败, 跟 Phase B B1/B2/B3 同根因)
> **训练时长**: ~3 分钟 (C1/C2/C3 + C2'/C3' 重启均快速 fail, 提前 abort)
> **关键发现**: **product_manifold + norm_target=ρ/2 路径本身就是 NO-GO**, 跟 w_path 无关

---

## §1 实验条件 (用户 2026-07-26 方案)

| 臂 | d_hyp | ρ | norm_target | w_path | path_geometry | GPU |
|---|-------|---|-------------|--------|----------------|-----|
| C0 (baseline) | - | - | - | - | - | (复用 #181) |
| **C1** | 4 | [2.0, 2.7, 3.4] | [1.0, 1.35, 1.70] | **0.0** | - | 0 |
| **C2** | 4 | [2.0, 2.7, 3.4] | [1.0, 1.35, 1.70] | 1.0 | hyp | 2 |
| **C2'** | 4 | 同上 | 同上 | **0.1** (降 w 重试) | hyp | 2 |
| **C3** | 4 | [2.0, 2.7, 3.4] | [1.0, 1.35, 1.70] | 1.0 | euc | 3 |
| **C3'** | 4 | 同上 | 同上 | **0.1** (降 w 重试) | euc | 3 |

e_dim = 36 (= 4 hyp + 32 euc), num_emb_list=[64,128,256], 其余对齐官方.

## §2 失败结果

| 臂 | epoch | collision | L0 ‖x‖_E range | 失效模式 | user gate hit |
|---|-------|-----------|------------------|----------|---------------|
| C1 | 181/1000 | **0.119** | [0.131, 0.999, **1.000**] | **boundary saturation** | max > 0.98 ❌ |
| C2 | 19/1000 | **0.9999** | mean=0.139 | **mode collapse** (立即) | - (col 失控) |
| C2' | 35/1000 | **0.874** | mean=0.241 | **mode collapse** (降 w 不够) | - |
| C3 | 19/1000 | **0.9995** | mean=0.142 | **mode collapse** (立即) | - |
| C3' | 119/1000 | **0.494** (持续涨) | [0.120, 0.138, **1.000**] | **mode collapse + boundary** | max > 0.98 ❌ |

**全部 5 个变体 (C1/C2/C2'/C3/C3') 都 NO-GO**.

## §3 关键发现 (R11.3)

**product_manifold + norm_target=ρ/2 路径本身就是 NO-GO, 跟 w_path 无关**:

1. **C1 (w_path=0)** collision 健康 0.12, 但 L0 码字被推到 ‖x‖_E=1.000 boundary → 用户 gate "max ≤ 0.95 全程" 失败 (B1 失效模式)
2. **C2/C2' (path_reg hyp)** w_path ∈ {1.0, 0.1} 都立即 mode collapse → 路径正则化让 hyp subspace 饱和更快
3. **C3/C3' (path_reg euc)** w_path ∈ {1.0, 0.1} 同样 mode collapse → path_geometry 不是关键, 是 product_manifold 本身

## §4 跟 Phase B B1/B2/B3 完全同根因

| 维度 | Phase B | Phase 1 (Task #211) |
|------|---------|---------------------|
| e_dim | 40 (8 hyp + 32 euc) | 36 (4 hyp + 32 euc) |
| norm_target | (无, Task #209 retry 配置) | [1.0, 1.35, 1.70] |
| B1/C1 (no path_reg) | ‖x‖_E=[0.077, 1.000, 1.000] | ‖x‖_E=[0.131, 0.999, 1.000] |
| B2/C2 (path_reg hyp) | collision=0.9999 | collision=0.9999 |
| B3/C3 (path_reg euc) | collision=0.9999 | collision=0.9995 |

**d_hyp 从 8 降到 4 没改变失效模式**. **product_manifold 架构本身触发 boundary saturation + path_reg 触发 mode collapse**, 跟 d_hyp, norm_target, w_path 无关.

## §5 用户 gate 失败清单

| Gate | 用户标准 | Phase 1 结果 |
|------|----------|---------------|
| hyp_norm_max ≤ 0.95 全程 | C1 L0 max=1.000 ❌, C3' L0 max=1.000 ❌ |
| dyn_range ≥ 2.0 全程 | (无法测, monitor 未在 hypnorm 输出) |
| 碰撞率 ≤ 12% | C1 ✅, C2/C2'/C3/C3' ❌ |
| 真实利用率 ≥ 90% (vs C0) | (无法测, 训练未跑完) |
| λ 三层 ≥ 4.0 | (无法测, 训练未跑完) |
| flip_rate 10-30% | (无法测, 训练未跑完) |
| 无 NaN | C2/C3 有 train loss 2114/97 (无 NaN, 但数值爆炸) |

## §6 关键决策点 (R11.4)

**主决策 (R2 禁止 fallback, R11.4 critical)**:
1. **不再花 GPU 跑 product_manifold 变体**. Phase B B1 已验证产品 manifold 撞 boundary; Phase 1 C1/C2/C3 三次确认.
2. **path_reg 不再是核心机制**. 5 个变体全部 NO-GO, 跟 w_path ∈ {0, 0.1, 1.0} 无关.
3. **下一方向**: 不再叠层叠 loss 叠 manifold. 回到 baseline #181 + 探索单变量改动 (例如只改 kmeans_init, 只改 sk_eps).

## §7 R10 + 主动推进建议

按用户 "follow loop.md" 原则 + R10 主动推进:

**Option A (R11.3 推荐)**: 
- 报告用户 Phase 1 NO-GO
- 用户决策下一步 (继续叠 vs pivot vs 接受 baseline)
- 同时监控 B1 Stage 3 训练 (GPU 1, epoch 68+, best NDCG=0.098, 预测 test R@10 0.085-0.095)

**Option B (R11.3 备选)**: 
- 假设用户希望"完成 Phase 1 4 臂"，再试 d_hyp=2 (Phase 0 显示 d=2 dyn 最大 35-165, 但 L2 angle 0.7° 太窄)
- 风险: d=2 在用户预测中可能同样失败 (跟 d=4/8 同根因)

**Option C (R11.3 备选)**: 
- 假设用户希望"放弃 product_manifold 路径"，直接做 Phase 2 下游 (用 B1 SID 跑 T5 → test eval)
- 风险: B1 的 SID 也是从 boundary-saturated codebook 推出来的, 下游泛化可能也不好

## §8 当前产物

- `products/task211/hrqvae_C1/Jul-26-2026_20-39-40_*/epoch_*_collision_*.pth` (best ckpt @ epoch ~39 collision 0.10)
- `products/task211/hrqvae_C2/Jul-26-2026_20-39-40_*/epoch_19_collision_0.9999_model.pth` (mode collapse ckpt)
- `products/task211/hrqvae_C2p/Jul-26-2026_20-40-35_*/epoch_*_collision_*.pth` (mode collapse ckpt)
- `products/task211/hrqvae_C3/Jul-26-2026_20-39-40_*/epoch_19_collision_0.9995_model.pth` (mode collapse ckpt)
- `products/task211/hrqvae_C3p/Jul-26-2026_20-40-35_*/epoch_*_collision_*.pth` (mode collapse ckpt)
- `scripts/task211_phase1_C{1,2,3,C2p,C3p}_stage1_train.sh` (5 个 launcher, 可复用)
- `scripts/task211_phase1_monitor.sh` (提前中止监控, 已知 epoch 跑太快 monitor 错过窗口, 待修)
- `logs/task211/{C1,C2,C2p,C3,C3p}_stage1_train.out` (5 个训练日志)

**result:** ❌ Task #211 Phase 1 NO-GO — 5 个 product_manifold 变体全部失败. 跟 Phase B B1/B2/B3 同根因 (boundary saturation + mode collapse). product_manifold 架构本身不可行. 等待用户决定下一步方向.


result: Task #211 — C1/C2/C3 三臂 Stage 1 训练结果
