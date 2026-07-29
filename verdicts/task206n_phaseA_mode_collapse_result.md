# Task #206n — Phase A: usage-target_r 半径语义项扫描 verdict

> **完成日期**: 2026-07-26 16:15
> **状态**: ❌ **全部 3 臂均发生模式坍缩 (collision ≥ 93.6%)**
> **根本原因**: 强度对齐 per-layer c (93/604/702) 过高, 导致码字被推到 Poincaré 球边界, 共形因子 λₖ 爆炸 → 梯度不稳定 → 全码本退化到同一位置

---

## 1. 实验设计回顾

Phase A (用户 2026-07-26 批准) 意在测试三层都加 usage-target_r (w_rad ∈ {0, 0.1, 1.0}) + per-layer 强度对齐曲率 (c=93/604/702, 目标强度 0.85). 对照基线是 HG-Rec baseline (c=1.0, collision ~13%).

| 臂 | w_rad | per-layer c |
|----|-------|-------------|
| 0 | 0.0 (对照) | 93/604/702 |
| 1 | 0.1 (轻) | 93/604/702 |
| 2 | 1.0 (重) | 93/604/702 |

## 2. 核心结果

| 臂 | w_rad | Best Loss | Best Collision | Final Collision | 完成 |
|----|-------|-----------|----------------|-----------------|------|
| 0 | 0.0 | 12.38 | **93.6%** | 99.99% | ✅ |
| 1 | 0.1 | 13.52 | **99.7%** | 99.99% | ✅ |
| 2 | 1.0 | 13.48 | **99.9%** | 99.99% | ✅ |

**所有 3 臂全部坍缩.** 对照组 (w_rad=0, 无 usage-target_r) 也无法幸免.

## 3. 坍缩时间线 (w_rad=0.0 典型)

| Epoch | Collision | 状态 |
|-------|-----------|------|
| 4 | 99.1% | 已经坍缩 |
| 9 | 95.7% | 轻微恢复 |
| **14** | **93.6%** | **最佳 epoch** |
| 19-49 | 96-98% | 徘徊 |
| 54+ | 99.99% | **永久坍缩** |

即使在最佳 epoch (epoch 14), collision 93.6% 也远超 HG-Rec baseline 的 ~13%.

## 4. 坍缩机制诊断

**根因**: per-layer c 过高 (93/604/702) → Poincaré 球体积不足:
- L0: ball radius r=1/√c=0.104, codebook fill=64%
- L1: ball radius r=1/√c=0.041, codebook fill=75-80%
- L2: ball radius r=1/√c=0.038, codebook fill=78-89%

**关键爆点 — 共形因子 λₖ**:
- L0: λₖ = 1/(1-c·‖x‖²) ≈ 3.5 (小幅放大, 安全)
- L1: λₖ ≈ 9388 (放大近万倍 → 梯度爆炸)
- L2: λₖ ≈ 105812 (放大十万倍 → 彻底梯度爆炸)

当码字接近球边界 (c‖x‖² → 1), λₖ 发散. 这造成:
1. 边界附近码字被"重力井"吸引
2. 所有码字向边界聚集 → 范数趋同 (min=max)
3. 方向区分度丧失 → argmin 退化 → 全部映射到同一码字
4. Sinkhorn 已关闭 (sk_eps=0), 无"人为利用率假象"掩盖

**比较**: HG-Rec baseline (c=1.0) 的 λₖ ≈ 1.1 (几乎不变形), collision ≈ 13%.

## 5. 关键教训

**Task #206m-v4 的静态诊断不能替代动态训练验证.** 即使强度对齐的 c 在 argmin 扫描中显示"几何主导", 实际训练时高曲率导致码本坍缩不可逆.

**§五 预测依然成立** (c↑ → util↓), 但 c 的可操作窗口远小于预期:
- c=1 → 几何≈欧氏 (argmin 一致率 99.9%), 但训练稳定 (collision 13%)
- c≥93 → 几何主导 (argmin 一致率 55-83%), 但训练不稳定 (collision 94-99.9%)

**中间窗口 (c ∈ [1, 93]) 需要探索** — 那里几何开始显现但训练还未崩溃.

## 6. 后续建议

| 方向 | 理由 | 优先级 |
|------|------|--------|
| **1) c ∈ [1, 10] 扫描** (Task #199 已有) | c=1→10 不坍缩, 且有 Δutil -22.7% (L2) 信号 | **高** |
| **2) 放弃高 c 策略** | c≥93 不可训练, 几何激活路线本质此路不通 | 高 |
| **3) 把曲率用在 margin loss 而非 argmin** (用户 2026-07-26 D 方向) | sort_clarity 1.93→1.03 证实 argmin 敌对, margin loss 回避 argmin | 用户决定 |
| **4) 使用 Sinkhorn 稳定训练** (与 #178 相同模式) | Sinkhorn 可掩盖坍缩 → 不可靠 | 低 |

## 7. 产物清单

| 产物 | 路径 |
|------|------|
| 日志 (w_rad=0.0) | `logs/task206n/arm_wrad0.0_stage1.log` |
| 日志 (w_rad=0.1) | `logs/task206n/arm_wrad0.1_stage1.log` |
| 日志 (w_rad=1.0) | `logs/task206n/arm_wrad1.0_stage1.log` |
| Launcher | `scripts/task206n_phaseA_launcher.sh` |
| 代码 Patch (w_rad) | `HG-Rec/model/utils.py` (HVectorQuantization), `HG-Rec/model/hrqvae.py` (HRQVAE), `HG-Rec/train_hrqvae.py` (CLI) |

## 8. 保留的代码修改

w_rad CLI + pass-through 保留在 `train_hrqvae.py`, `hrqvae.py`, `utils.py` 中, 默认关闭 (w_rad=None → [0.0,0.0,0.0]), 未来如有低 c 场景可复用.

---

**一句话结论**: 强度对齐 per-layer c (93/604/702) 导致全部 3 臂坍缩, 即使 w_rad=0 (无 usage-target_r) 也无例外. 根本原因是高曲率压缩 Poincaré 球体积 → λₖ 共形因子爆炸 (L2 达 10⁵ 倍) → 训练梯度不稳定. 几何激活路线 (Task #206 主线) 确认 c ∈ [1, ~10] 是安全窗口, c≥93 不可训练.

result: Task #206 — Phase A: usage-target_r 半径语义项扫描 verdict
