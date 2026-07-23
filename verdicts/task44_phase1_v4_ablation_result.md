# Task #44 Phase 1 V4 — Flat-3seg 对照矩阵 (终止前最后检验)

> **完成日期**: 2026-07-20
> **任务**: 用户怀疑 PM-RQ V1-V3 退化可能源于 "fused 比 3 sub 干净" + "norm long tail" 而非几何机制. 跑对照矩阵分离变量.

---

## 1. 对照矩阵 (toy toy: K=64, 10K items, 3 层, 2000 steps, seed=42)

| 实验 | 输入处理 | 模型 | final_loss | vs B |
|------|---------|------|------|------|
| **B-L2 (已有)** | fused L2 normalized | StandardRQ 64d | **0.275** | ref |
| E1 (新) | fused raw (max=381) | StandardRQ 64d | **0.815** | 3.0× |
| E2 (新, 关键) | 3 sub NORM_CAP=5.0 clip | **Flat-3seg 独立欧氏 RQ** | **1.020** | 3.7× |
| E3 (新, 关键) | 3 sub 各自 L2 norm | **Flat-3seg 独立欧氏 RQ** | **1.160** | 4.2× |
| E4 (新) | 3 sub concat → 192d, L2 norm | StandardRQ 192d | **0.482** | 1.75× |
| PM-RQ V3 (已有) | 3 sub NORM_CAP=5.0 clip | PM-RQ 曲率 + learnable κ + fusion | **3.512** | 12.8× |

---

## 2. 关键判定 (基于 V4 数据)

### 判定 (1): 数据/管线本身有问题, 不能简单归罪 PM-RQ
- E3 (Flat-3seg-L2, 纯欧氏 + L2 norm) = **1.16**, 比 B (0.275) 高 **322%**
- 即去掉所有 PM-RQ 机制(曲率 / learnable κ / K³ 联合搜索 / fusion logits), 单纯 "3 段独立 L2 norm" 仍然差于 baseline
- **结论**: 用户怀疑 ✅ **成立** — 数据/管线层有结构性问题

### 判定 (2): norm long tail 不是 root cause
- E2 (clip) = 1.020 vs E3 (L2 norm) = 1.160 → L2 norm 在 3 sub 上**反而差 12%**
- E1 (fused raw) = 0.815 vs B (fused L2) = 0.275 → L2 norm 对 fused **帮助 3×**
- **结论**: norm 影响 **因数据而异**:对 fused 帮助大, 对 3 sub 反而有害. norm 不是 root cause.

### 判定 (3): 维度 / 容量是核心变量
- E4 (concat 192d L2) = **0.482**, 比 E3 (3 个独立 64d) 好 **2.4×**, 接近 baseline (0.275) 的 **1.75×**
- 数学解读: 3 段独立 norm² loss = 3 × 单段 loss, 所以理论下界是 3 倍于单段 RQ
- fused (0.275) vs concat (0.482) 的 1.75× 差距, 是 fused 平均的 "信号整合 + 去噪" 收益
- **结论**: "3 段独立 RQ + norm² 求和" 这个架构选择**本身就差**, 与几何机制无关

### 判定 (4): PM-RQ 几何机制在 toy toy 仍然有害
- E2 (Flat-3seg-raw clip, 欧氏) = 1.020 vs PM-RQ V3 = 3.512
- 即使同样脏输入, PM-RQ 几何机制让 loss 进一步**变差 3.4×**
- 这是 learnable κ + K³ 搜索 + commitment loss 的额外惩罚
- **结论**: 即使数据干净, 当前 PM-RQ 几何机制在 toy 设定下**仍然过拟合/不稳定**

### 判定 (5): fused 原始数据也有 norm 问题
- E1 (fused raw) = 0.815 vs B (fused L2) = 0.275 → fused raw 退化 3×
- 但 baseline 用 L2 norm 屏蔽了 fused 自身的 norm 问题
- **结论**: fused 也不是天然干净, 是 baseline 的 L2 norm 让它"看起来干净"

---

## 3. PM-RQ 否证的真正原因 (修订)

| 误读 (V3 verdict) | 真实 (V4 verdict) |
|---|---|
| "PM-RQ 假设被否证" | **"3 段独立 norm² loss + 几何机制"双重失败** |
| "几何假设错了" | 几何假设无法验证, 因为 **3 段独立 loss 架构本身就不公平** |
| "sphere/hyperbolic 是反信号" | 不能定论, 因为 3 段架构让 sphere/hyperbolic 段被迫和 euclid 段独立 norm, 拖累整体 |

**修订结论**:
- ✅ **当前 PM-RQ 架构否证成立** (C 3.51 ≫ B 0.275)
- ⚠️ **但否证的是架构, 不是假设** — "3 段独立 norm²" 这一选择本身就把 loss 下界抬高到 3× baseline
- ✅ **几何机制在 toy 仍然有害** (同输入下 3.51 vs 1.02, 3.4× 退化) — 即便架构修好, 当前 κ / fusion 实现仍有问题

---

## 4. 后续决策 (修订)

| 方案 | 描述 | ROI |
|------|------|---|
| **A. 终止 PM-RQ** | 当前架构否证, ROI 低, 推荐终止 | ✅ 推荐 |
| **C-1. 改架构: concat 192d + 几何重参数化** | 在 E4 (0.482) 的基础上, 对 192d 整体做几何重参数化, 看能否进一步降到 0.275 以下 | ⚠️ 中, 需新设计 |
| **C-2. 改架构: 共享 codebook + 段专属残差** | 1 个 codebook (K=64) 处理主要信号, 3 段只在残差阶段独立 norm | ⚠️ 中, 需新设计 |
| **B. Phase 2 全规模** | 在 toy 否定后, 全规模 ROI 低 | ❌ 不推荐 |

---

## 5. 产物清单

| 路径 | 内容 |
|------|------|
| `scripts/task44_flat3seg_ablation.py` | V4 主脚本 (4 组对照 + 1 reference) |
| `logs/task44_phase1_v4.log` | V4 训练日志 (5 min, E1-E4 全完成) |
| `products/task44_pmrq_phase1/task44_phase1_v4_ablation.json` | V4 完整 summary + judgment |
| `verdicts/task44_phase1_v3_result.md` | V3 旧 verdict (本次修订) |
| `verdicts/task44_phase1_v4_ablation_result.md` | **本 verdict (含 result: 行)** |

---

## 6. 任务时间线

- **V1 (原始 argmin)**: 2026-07-19 — C=1.926, 7× 退化
- **V2 (STE)**: 2026-07-20 — C=2.394, 8.7× 退化
- **V3 (commitment)**: 2026-07-20 — C=3.512, 12.8× 退化, fusion_logits 学会 trivial solution
- **V4 (Flat-3seg 对照矩阵)**: 2026-07-20 — 揭示 root cause = "3 段独立 norm² loss + 几何机制" 双重失败

---

**result:** Task #44 Phase 1 V4 对照矩阵揭示 root cause: 当前 PM-RQ 否证来自 **(a) 3 段独立 norm² loss 架构本身就把 loss 下界抬到 3× baseline, (b) 几何机制在 toy 进一步恶化 3.4×**。用户对 norm long tail 的怀疑部分正确(L2 对 fused 帮助 3×, 但对 3 sub 反而 -12%);fused 比 3 sub 干净的怀疑成立(0.275 vs 1.160)。**推荐 A 终止当前 PM-RQ 架构**,如要继续则需要改架构(C-1: concat 192d + 几何重参数化,或 C-2: 共享 codebook + 段专属残差)。

result: Task #44 — completed (R8 §9.3 retroactive compliance line appended by Task #114; refer to verdict body above for full details).
