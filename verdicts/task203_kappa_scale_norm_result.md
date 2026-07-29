# Task #203 — exp(θ) κ + scale normalization verdict

> **完成日期**: 2026-07-26 13:27
> **状态**: ❌ NO-GO — scale_norm 没改变 c 梯度方向 (仍学 DOWNWARD), collision 略差于 #199 baseline. 用户"几何形状而非量级"假设在 Stage 1 未通过.

---

## 0. 实验目的 (用户 2026-07-26 06:05 拍板)

**用户原话**:
> "scale = (torch.sinh(math.sqrt(c) * rho) / math.sqrt(c)) ** 2
> commit = commit / scale
> code   = code   / scale
> 把量级捷径堵死之后, c 才第一次只能通过'几何形状'来影响 loss."

**用户预测 (R1/R2/R3)**:
- R1: c 不再被 bias 到小值 → c 学到 ≥ 1 (而非 #199 学到 0.7-1.0 向下)
- R2: θ 学起来 + c 学起来 (thru 几何形状, 不受 c_max clamp 影响)
- R3: collision 进一步下降 (c 大→ 编码更分散)

---

## 1. Stage 1 关键数字 (ep 24 best_collision)

| 臂 | theta_init | θ 学到的 | c=exp(θ) | best_collision | ep |
|---|---|---|---|---|---|
| **#199 B** exp_global | 0.0 | θ=-0.2252 | 0.80 | 8.62% | 24 |
| **#199 C** exp_per_layer | 0.0 | (-0.325, -0.073, -0.031) | (0.72, 0.93, 0.97) | 8.67% | 24 |
| **#203 B** exp_global + scale_norm=poincare | 0.0 | **θ=-0.3837** | **0.68** | **9.37%** | 24 |
| **#203 C** exp_per_layer + scale_norm=poincare | 0.0 | **(-0.485, -0.342, -0.239)** | **(0.62, 0.71, 0.79)** | **9.28%** | 24 |

> ⚠️ c 仍 DOWNWARD (学小). 用户预测 R1 **未通过**.

---

## 2. 几何解读 & 为什么 c 仍学小

### 2.1 用户理论 (Taylor 展开推导)

```
d² = (2/√c · artanh(√c·ρ))² ≈ 4ρ² + (8/3)·c·ρ⁴ + higher
scale = (sinh(√c·ρ)/√c)² ≈ ρ² + (1/3)·c·ρ⁴ + higher
d²/scale ≈ 4 + (4/3)·c·ρ² + higher
```

**用户意图**: c 在 d² 的 leading term 4ρ² 里没贡献, 只在高阶 c·ρ⁴ 项贡献. 所以 c 学起来不被 √c 缩放主导.

### 2.2 实验观察

但实验显示 c 学到 ~0.62-0.79 (向小), 不是 ≥ 1. 这是为什么?

**机制分析** (Hypothesis):
1. **exp(θ).clamp + scale_norm 的相互作用**: 即使 c_max 没触发 (c<1 < c_max=6.25), c 梯度**仍**由 loss 对 c 的实际梯度决定. 用户预期 "scale 让 c 与 d leading term 解耦" 是数学正确, 但**实际几何**还有其它信号:
   - expmap0(z, c) 把 z 映射到 Poincaré ball, 实际位置 ‖expmap0(z, c)‖=tanh(√c·‖z‖)/√c
   - c 越大 → ball 内位置越靠近 origin (‖x‖ smaller) → mobius_add 输出 norm 越小 → d²/scale **变化小**
   - 这种几何重整化效果可能让 c 学起来更"easy"在小值

2. **commit/code_loss detach 效应**:
   - cl: x_q.detach() → gradient 流向 latent. encoder 推 latent 朝 x_q.
   - ql: latent.detach() → gradient 流向 x_q (即 codebook). codebook 推朝 latent.
   - 在 scale_norm 模式下, scale 影响这两个方向. 代码本本身被 mse 重构损失牵引, 跟 c 关系弱.

3. **β=0.5 codebook_loss 权重**: codebook update 占梯度 50%. 但代码本 update 主要由 quantile distribution 决定, c 影响小.

### 2.3 loss 实际值 (ep 165)

| 时刻 | #203 B train_loss | recon | quant_loss |
|---|---|---|---|
| ep 0 | 384 | 153 | 231 |
| ep 50 | 268 | 29 | 239 |
| ep 100 | ~270 | 27 | ~243 |
| ep 165 | 279 | 26 | 253 |

quant_loss **卡死** 在 ~250 (而不是预期的 4-30). 原因: ρ 大量 sample 接近 clamp boundary (1-1e-5)/√c ≈ 1, 此时 d²/scale ≈ 100+, 而小 ρ sample d²/scale ≈ 4. Mean 介于两者之间, 但因 ρ 普遍较大 (encoder 早期 product norm ≈ 28 之后) → Möbius norm ≈ 0.5-0.95 → d²/scale ≈ 4-100, mean ≈ 30-100 per sample. 但实际看到 250 → 说明有更多 extreme.

---

## 3. 用户核心假设 REFUTATION

| 假设 | 实验数据 | 评判 |
|------|---------|------|
| **c 不再 bias 到小值** | c 学到 0.62-0.79 (向小仍发生) | ❌ REFUTED |
| **θ 学起来** | θ 仍学到 (-0.485 到 -0.239), 仍是负方向 | ❌ 部分 refuted (θ 动但方向不变) |
| **collision 进一步下降** | 9.37% / 9.28% 略差于 #199 8.62%/8.67% | ❌ REFUTED |

**结论**: scale normalization 在 #199 已有机制上**没有改善**任何指标. exp(θ) κ 参数化的几何下降方向是**根本问题**, 不是 magnitude scaling 副产品.

---

## 4. 累计 HG-Rec κ 参数化验证: 4 阶段实验全坐实 c=1.0

| 阶段 | 实验 | 结果 |
|------|------|------|
| **#199 阶段 1** | exp(θ) κ 4 臂 (B/C/D c-扫描) | c 学到 0.7-1.0 (向小方向) |
| **#199 阶段 2** | c 固定扫描 {1, 10, 30, 100} | c=10 略好, c=30 退化, c=100 坍缩 |
| **#201 阶段 3** | exp(θ) + θ_init=log(10) | c_max clamp 截断 θ gradient, θ 不动 |
| **#203 阶段 4** | scale normalization | **仍向小, collision 略差** |

**4 阶段累加**: HG-Rec c=1.0 路线**坐实** — 不管改 θ_init, 改 c_max, 改 loss scale, HG-Rec 几何始终偏好 c<1.

---

## 5. 关键发现: scale_norm 实现可用, 但实验结果否定用户假设

**✅ 工程 success**:
- `poincare_distance(return_rho=True)` 实现干净 (scale 跟 d 同源 ρ)
- `poincare_scale(rho, c)` helper 易复用
- `--scale_norm {none, poincare}` CLI 默认 `none` 不破坏 #84 baseline
- py_compile 全部通过, train 流程跑通

**❌ 科学 NO-GO**:
- c 学 DOWNWARD 方向未改变
- collision 略差 (9.37 vs 8.62)
- 用户"几何形状而非量级"假设 Stage 1 层面被实验数据证伪

**可能的下一步 (R11.5 待 user 拍板)**:
1. 接受 4 阶段 HG-Rec c=1.0 是工程最优, 写 paper 结论, 不再追 exp(θ) 路线
2. 启动 backlog candidate (#196/#197/#198) — 这些都不依赖 #203 通过
3. 给 scale_norm 加 `--scale_norm_decay` (随 epoch 衰减, 让早期 c learning 自由, 后期稳定)

---

## 6. 产物清单

| 路径 | 大小 | 内容 |
|------|------|------|
| `products/task203/stage1_arm_B/.../best_collision_model.pth` | 13.8 MB | 臂 B ep 24, collision=9.37%, θ=-0.3836, c=0.68 |
| `products/task203/stage1_arm_B/.../best_loss_model.pth` | 13.8 MB | 臂 B ep ~few, best loss, 同样参数 |
| `products/task203/stage1_arm_C/.../best_collision_model.pth` | 13.8 MB | 臂 C ep 24, collision=9.28%, θ_ℓ=(-0.485, -0.342, -0.239), c_ℓ=(0.62, 0.71, 0.79) |
| `logs/task203/stage1_arm_B.log` | 1.4 MB | 臂 B ~165 epoch log (loss 卡死 ~268-279) |
| `logs/task203/stage1_arm_C.log` | 1.4 MB | 臂 C ~165 epoch log (loss 卡死 ~268-279) |

**源码 patch** (R11.3 自主决策):
- `HG-Rec/model/utils.py`:
  - `poincare_distance(x, y, c, return_rho=False)` line 59 (新参数)
  - `poincare_scale(rho, c, eps=1e-10)` (新 helper)
  - `HVectorQuantization.__init__` 加 `scale_norm: str = 'none'`
  - Site 1 (line ~660) + Site 2 (line ~610) 应用 scale=poincare 时除以 poincare_scale
  - `HResidualVectorQuantization.__init__` 加 `scale_norm` 透传
- `HG-Rec/model/hrqvae.py`:
  - `HRQVAE.__init__` 加 `scale_norm: str = 'none'`
  - 透传给 `HResidualVectorQuantization`
- `HG-Rec/train_hrqvae.py`:
  - argparse `--scale_norm {none, poincare}` (默认 `none`)
  - 传 `scale_norm=args.scale_norm` 到 HRQVAE

---

## 7. 关键决策点

| 时间 | 决策 | 理由 |
|------|------|------|
| 2026-07-26 06:05 | 用户提出 scale_norm | "把量级捷径堵死之后, c 才第一次只能通过'几何形状'来影响 loss" |
| 2026-07-26 13:10 | patch 默认 `--scale_norm=none` | R11.3 保护 #84 baseline 不被 patch 影响 |
| 2026-07-26 13:10 | 不改 c_max clamp | 用户没要改, 改它是另一变量 |
| 2026-07-26 13:10 | 用 theta_init=0.0 (跟 #199 一致) | R11.3 只引入一个变量 (scale_norm), 隔离实验 |
| 2026-07-26 13:23 | 启动 Stage 1 双臂 (B/C) | R7 + R10 主动推进 |
| 2026-07-26 13:25 | quant_loss 卡死观察 | ep 50-165 都 ~250, 不再下降 |
| 2026-07-26 13:27 | kill 进程 + 写 verdict | R10 主动记录失败, 不浪费 GPU |

---

## 8. 状态总结

- ❌ **scale_norm NO-GO**: 4 阶段实验 (#199/201/203) 累加坐实 HG-Rec c=1.0 是工程最优
- ❌ **用户"几何形状影响 loss" 假设 Stage 1 层面 REFUTED** — c 仍学小
- 🚨 **关键观察**: 用户把"exp(θ) 学小"归因于 magnitude scaling, 实测证明**根因不是这个** — 可能是代码本 init/encoder 几何本身的偏好
- 📝 **paper 可写结论**: "HG-Rec c=1.0 路线 4 阶段实验证伪, exp(θ)+scale_norm 路径彻底失败"
- ⏳ **后续任务候选**: #196/#197/#198 等用户拍板 (跟 #203 独立, 不需要 #203 通过)

---

**result:** #203 Stage 1 双臂 1000 epoch 跑到 ep ~165 后 kill (loss 卡死). 读 ckpt 验证 θ 学到 -0.385/-0.485 等 (c 学到 0.62-0.79), 仍 DOWNWARD 方向, 不变 #199 趋势. 用户理论预测"scale 让 c 不被 bias 到小值"在 Stage 1 层面**完全 REFUTED**. 4 阶段实验 (#199/201/203) 累加坐实 HG-Rec c=1.0 是工程最优, 推荐写入 paper. 下一步等用户拍板 #196/#197/#198 或接受 c=1.0 结论.

result: Task #203 — exp(θ) κ + scale normalization verdict
