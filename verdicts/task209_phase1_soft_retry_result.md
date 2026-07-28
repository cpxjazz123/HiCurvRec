# Task #209 Phase 1 软约束重试 — NO-GO (双失败)

> **完成日期**: 2026-07-26
> **状态**: ❌ FAIL (软约束让几何激活但 argmin 仍 99% 坍缩, 与硬投影症状一致)
>
> **覆盖**: `verdicts/task209_phase1_initial_result.md` §4 "修复方向: 改用 --norm_target" 已被证伪. 两次尝试 (硬 + 软) 都 NO-GO.

---

## §1 实验条件 (软约束 retry)

回滚硬投影 patch (HVectorQuantization.forward 中的 `tangent_norm * F.normalize(...)`), 改用 Task #196 软 norm 约束:

```bash
# A1 --norm_target + --gamma_norm
--norm_target 1.0 1.35 1.70 \
--gamma_norm 0.1 \
```

四臂差异 (其余 100% 对齐 #181 baseline, batch=1024, epochs=1000, kmeans_init=True, kmeans_iters=1000, beta=0.5):

| 臂 | 与 baseline 差异 |
|---|---|
| **A1** | + --norm_target 1.0 1.35 1.70 --gamma_norm 0.1 |
| **A2** | A1 + --scale_norm poincare |
| **A3** | A2 + --w_path 1.0 --path_geometry hyp --rho_targets_path 2.0 2.7 3.4 |
| **A4** | A3 but --path_geometry euc |

A0 不跑 (用 #181 baseline ckpt).

## §2 跨 4 臂最终 collision

| 臂 | Final CollRate (ep 1000) | Min CollRate | ckpt 路径 |
|---|---|---|---|
| A1 | 0.9904 | 0.9903 | `products/task209/phase1_arm_A1/.../best_loss_model.pth` |
| A2 | 0.9899 | 0.9899 | `products/task209/phase1_arm_A2/.../best_loss_model.pth` |
| A3 | **0.9997** | 0.9985 | `products/task209/phase1_arm_A3/.../best_loss_model.pth` |
| A4 | 0.9976 | 0.9964 | `products/task209/phase1_arm_A4/.../best_loss_model.pth` |
| #181 baseline | **0.06** | — | (不重跑) |

**Phase 1 出口条件 (CollRate ≤ 11%): 4 臂全 FAIL**. 主判据二次失败.

## §3 几何激活成功 (软约束生效)

A3 hypnorm 最终状态 (epoch 1000):

```
[hypnorm] L0 κ=1.000 ‖x‖_E=[0.762,0.762,0.762] mean=0.756 λ_κ=4.7
         L1 κ=1.000 ‖x‖_E=[0.874,0.874,0.874] mean=0.870 λ_κ=8.4
         L2 κ=1.000 ‖x‖_E=[0.935,0.935,0.927] mean=0.927 λ_κ=15.8
```

逐层 norm 精准匹配 `tanh(√c·ρ/2)` (target ρ=2.0/2.7/3.4):
- tanh(1.0)=0.762 ✓ L0
- tanh(1.35)=0.874 ✓ L1
- tanh(1.70)=0.935 ✓ L2

`λ_κ` (conformal factor) 升到 4.7 / 8.4 / 15.8, 比 #181 的 0.57 / 0.19 / 0.13 高一个数量级. **κ-Stereographic 几何确实在球面 shell 上激活了**, 不是 dead weight.

## §4 失败根因 (与硬投影不同)

### §4.1 软约束 vs 硬投影的对比

| | 硬投影 (初次) | 软约束 (重试) |
|---|---|---|
| 实现 | `e_n = r · F.normalize(e)` (无径向梯度) | `γ·(‖e‖ - r)²` loss (有梯度) |
| 几何激活 | ✅ norm 强制到目标 | ✅ norm 软拉到目标 |
| Collision | 0.994-0.9999 | 0.990-0.9997 |
| 方向梯度 | ❌ 全夹断 | ✅ 保留 |

### §4.2 软约束特有的失败机制

`λ_κ = 2/(1 - c‖x‖²)`. 当 ‖e‖_poincare ≈ 0.762, λ_κ ≈ 4.7. 这意味着 **码字点离球面边界还有 23.8% 距离, 几何"中段"**.

但软约束 + kmeans_init 联用产生新问题:
1. kmeans_init 在 latent 空间跑, latent ‖x‖ ≈ 0.05-0.15 (小).
2. kmeans 给出的 64 个聚类中心, 经 expmap0 + proj_to_ball 后在球面分散.
3. 但 `norm_loss` 一开始就把所有码字拉到 target shell ‖e‖_poincare = 0.762 (远大于 latent norm 0.1).
4. 此时 `argmin` 距离: `d²(emapper(latent), emapper(codebook))` 在切空间大致 ∝ ‖latent - codebook‖² ≈ ‖codebook‖² 常数 + 方向差.
5. 因为 codebook norm 都相等, 距离退化到方向差, 但**所有 64 个码字方向被 norm_loss 的径向梯度忽略, 等价于同径问题**.
6. 跟硬投影一样 → mode collapse.

### §4.3 关键差异

硬投影是直接消除径向梯度; 软约束是 norm_loss 让码字同径后, 让方向问题在球面更高 λ_κ 处重新涌现. **但本质同问题**: 码字 norm 相同时, 32 维方向空间 + 球面 shell 不足以保证 kmeans 充分分散.

## §5 R11.3 决策: 两次失败后路线图

### §5.1 不再投入 GPU 尝试第三种 norm 方案
- 硬投影: 100% mode collapse
- 软约束: 99% mode collapse (但几何激活)
- 推论: 单纯码字范数约束不够, 需要**方向多样性正则 + 范数约束联用**, 或**根本修改 kmeans_init 在球面上的初始化方式** (后者属于 "核心算法改动", 超出 Phase 1 范畴, 需用户授权)

### §5.2 pivot 到 §7 退路 2 (机制叙事)

description §7 已明确"整体 R@10 不动是很可能的结果", 给出两条退路 (切片 + 机制叙事). 既然 Phase 1 阶段 codepath 已不可能进 Phase 2 多 seed, 调整成:

#### Phase 2 极简版 (替代原 3-seed)
- **A0 (=#181 baseline, 不重跑) + A3 (软约束 ckpt, 已存在)**: 各 1 seed 跑 Stage 2 (SID 推断) + Stage 3 (T5-mini 训练) + Stage 4 (eval R@10)
- 不再做 Phase 2b (3 seed 重复) — 没意义, 因为 Phase 1 已经 no-go, 不需要方差
- A3 用 `best_loss_model.pth` (跟 #181 baseline 比较)

#### Phase 3 不变 (机制 + 切片)
- **Phase 3.1**: A3 vs A0 的机制对比 (绕路成本曲线 / √c·ρ / argmin 一致率 / 角分辨率) — 重点证明 **A3 几何激活 (λ_κ=4.7+) 真实存在于球面**, 即使 coll 99%
- **Phase 3.2**: 切片评估 (Head/Body/Tail R@10, NDCG@10) — 即使整体持平, 看 Tail 上 A3 是否优于 A0

#### 不进入 Phase 1b
- Phase 1b 是 "在能收敛的前提下扫 w_path / ρ", 但 Phase 1 都没收敛, Phase 1b 没有入口. 取消.

### §5.3 决策依据 (R11.3 自主决策留痕)

(a) **选哪个**: pivot 到机制叙事 (跳过 Phase 1b 和 Phase 2 多 seed)
(b) **为什么**: 两次 norm 约束尝试都失败, 继续投入 GPU 修代码成本边际递减; description §7 退路已明示
(c) **备选方案**:
  - 选项 A: pivot 到机制叙事 ← **选这个**
  - 选项 B: 修改 kmeans_init 在球面重新实现 (3-4 天 GPU 时间) — 风险高, 收益不确定
  - 选项 C: 直接关闭 Task #209, 退回 #208 双码本 — 浪费 Phase 0 已做的 6 项验证

## §6 产物保留 (供 Phase 3 分析)

- A1/A2/A3/A4 ckpt + 训练日志 — 用于 Phase 3 机制分析
- A3 几何激活数据 (λ_κ=4.7/8.4/15.8) — 核心叙事证据
- A3 mode collapse 数据 — 描述 norm 约束的根本限制, 不是 bug

## §7 关键超参变化 (下一步用)

A3 100% 对齐 #181 baseline 的 launcher, 只加 3 个 flag:

```bash
--norm_target 1.0 1.35 1.70 \
--gamma_norm 0.1 \
--scale_norm poincare \
--w_path 1.0 --path_geometry hyp --rho_targets_path 2.0 2.7 3.4 \
```

`best_loss_model.pth` 路径:
`products/task209/phase1_arm_A3/Jul-26-2026_18-26-17_beta_0.500_codebook_[64,128,256]_sk_0.000/best_loss_model.pth`

A0 直接用 #181 ckpt.

---

**result:** ❌ Phase 1 软约束重试 FAIL — 4 臂 collision 0.99 (vs baseline 0.06), 但 A3 几何激活成功 (λ_κ=4.7/8.4/15.8). 不再投入 GPU 修 norm 约束, pivot 到 description §7 退路 2 (机制叙事 + 切片评估), Phase 2 简化为 A0/A3 各 1 seed 下游, Phase 1b 取消.
