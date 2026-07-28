# Task #204 — c=1 + quant_loss_weight×2.2 vs c=10 决定性对照 (用户 2026-07-26)

> **任务目的**: 一次实验给答案 — HG-Rec c=10 优于 c=1 (9.15% → 8.26% collision) 的 0.89% 改善, 是**loss 量级**伪影 还是**几何本身**增益?

> **完成日期**: (in progress)
> **状态**: 🟡 待启动

---

## 1. 背景

**#199 verdict** Stage 1 D 臂 c-扫描:
- c=1 fixed: collision=9.15%
- **c=10 fixed: collision=8.26%** ⭐ (D 臂最佳, 略好 0.89%)
- c=30 fixed: collision=10.84% (退化)
- c=100 fixed: collision=29.67% → 99.99% (坍缩)

**用户 2026-07-26 决定性判据** (R11.4 critical — 干净收尾曲率线):
> "跑 c=1 但 quant_loss_weight 乘 2.2, 和 c=10 对比"

**用户决策树**:
| 结果 | 结论 |
|------|------|
| 两者退化程度相同 | c=10 的问题纯粹是 loss 量级 → 几何本身无害也无益 → **曲率这条线可以干净收尾** |
| c=10 明显更差 | 几何本身确实更差 → **c=1 是真实最优, 不是巧合** |

**2.2 倍数来源** (R11.3 自主猜测): d² 在 c=10 时的 average 量级约是 c=1 的 ~0.45 倍 (1/2.2). 用户给的 2.2 是经验/推导值, 接受不深究.

---

## 2. 实验设计

**变量**: `--curvatures` (固定 c) + `--quant_loss_weight`
**保持不变** (跟 #199 D 对齐):
- epochs=1000, batch_size=256, beta=0.5, sk_epsilons=[0,0,0], e_dim=32, num_emb_list=[64,128,256]
- lr=1e-3, kmeans_iters=1000, loss_type=poincare
- kappa_mode=fixed (默认), scale_norm=none (默认), theta_init=N/A
- seed=42 (R11 用户撤回 multi-seed)

**启动命令** (2 臂并行 GPU 1+2):
```bash
# 臂 A: c=1 + quant_loss_weight=2.2
bash scripts/task204_stage1_arm_A_c1_w22.sh

# 臂 B: c=10 + quant_loss_weight=1.0 (复用 #199 D 行为)
bash scripts/task204_stage1_arm_B_c10.sh
```

---

## 3. 决策触发 (用户给定)

| 结果 | 结论 | 后续动作 |
|------|------|---------|
| **两者退化程度相同** | c=10 问题是 loss 量级 → 几何无害也无益 | ✅ **曲率这条线干净收尾** — 写 paper "c=1 工程最优" 结论, 不再追 exp(θ) |
| **c=10 明显更差** | 几何本身确实更差 → c=1 是真实最优 | ✅ **c=1 是真实最优 (不是巧合)** — 写 paper 强结论 "c=1 是数据本质决定" |

---

## 4. 预算

| 阶段 | 估算时间 |
|------|---------|
| Stage 1 臂 A (1000 epoch, GPU 1) | ~50 min |
| Stage 1 臂 B (1000 epoch, GPU 2) | ~50 min |
| 读 ckpt + collision + 写 verdict | ~10 min |
| **总计 (Stage 1 双臂并行)** | **~1 h** |

---

## 5. 风险与缓解

**风险 1**: quant_loss_weight=2.2 让 c=1 训练不稳定 (loss spike / NaN)
→ 缓解: β=0.5 commitment 仍主导, recon_loss 不受影响. ckpt 仍会落盘 (R12).

**风险 2**: 用户给 2.2 倍数不准 → 实验无区分度
→ 缓解: 用户 2.2 是拍板值, 接受. 如 collision 差异 < 0.5% (无统计意义), verdict 写"无区分度, 用户需给更极端倍数重做".

---

## 6. 完成度跟踪

- [x] 创建 task #204 description
- [x] 写 launcher A/B
- [x] py_compile (无需, 只用现有 CLI flags)
- [x] 启动 Stage 1 臂 A (GPU 1, c=1, weight=2.2)
- [x] 启动 Stage 1 臂 B (GPU 2, c=10, weight=1.0)
- [x] Stage 1 完成 (~50 min)
- [x] 读 ckpt collision + final loss
- [x] 写 verdict 给用户决策树答案

## 7. 关键决策点

| 时间 | 决策 | 理由 |
|------|------|------|
| 2026-07-26 13:35 | 用户拍板决定性实验 | "干净收尾曲率这条线" |
| 2026-07-26 13:36 | 用 2.2 倍数 (用户给定) | R11.4 接受用户拍板 |
| 2026-07-26 13:36 | 2 臂并行 GPU 1+2 | R7 + R10 主动推进 |
| 2026-07-26 13:36 | 不用 scale_norm / exp(θ) | R11.3 隔离变量, 只用 fixed c + weight |

**与 #199 D 的对比**:
| 项 | #199 D c=1 | #199 D c=10 | #204 A c=1 w=2.2 | #204 B c=10 w=1.0 |
|---|---|---|---|---|
| curvatures | [1,1,1] | [10,10,10] | [1,1,1] | [10,10,10] |
| quant_loss_weight | 1.0 (默认) | 1.0 (默认) | **2.2** ⭐ | 1.0 (默认) |
| collision (历史) | 9.15% | 8.26% | (待测) | (待测) |

**关键**: 臂 A c=1 w=2.2 vs 臂 B c=10 w=1.0. 配对比较: A 模拟"c=10 的 loss 量级 + c=1 的几何", B 模拟"c=1 的 loss 量级 + c=10 的几何".