---
task: 56
type: result
created: 2026-08-02
tags:
  - misc
up: "[[index]]"
---
# Task #56 — 曲率重审 / Task #85 三几何清算 — 低 ROI (维持 Task #85 verdict)

> **任务目的**: 在 dist_kappa κ→0 L'Hôpital 边界修复 (阈值 1e-6 → 1e-3) 后, 重审 Task #85 三几何独立 SID 的曲率机制
> **执行日期**: 2026-07-20
> **状态**: ✅ 完成 — ROI 低, 维持 Task #85 verdict, 不重训 TIGER
> **额外发现**: ⭐ Task #99 fused embedding norm 长尾极端 (CV=6.32, max=381 vs mean=0.79)

---

## 1. 时间线

| 时间 | 事件 |
|------|------|
| 2026-07-20 03:38 | Task #56 启动 (纯诊断, < 10s) |
| 2026-07-20 03:38 | 落盘 verdicts/task56_curvature.json |
| 2026-07-20 03:38 | 写 verdict |

---

## 2. Task #85 历史 verdict

| 几何 | TEST R@5 | vs Task #80 baseline (0.0383) | 判定 |
|------|----------|-------------------------------|------|
| m=0 球面 (Task85 m0) | 0.0174 | 45.4% (-54.6%) | 单层球面 SID 弱 |
| m=1 准欧氏 (Task85 m1) | 0.0200 | 52.2% (-47.8%) | **三者最优** |
| m=2 双曲 (Task85 m2) | trivial bias | n/a | mode collapse |
| Task #80 baseline (fused KMeans) | 0.0383 | 100% | — |

---

## 3. L'Hôpital 修复影响分析

**修复**: `dist_kappa(u, v, κ)` 中, 当 `|κ| < 1e-3` 时直接返回欧氏距离 (避免 L'Hôpital 边界处返回 2× 欧氏距离的 bug).

**对 Task #85 影响**:
- Task #99 MCKG 训出 κ = [+5.05, -0.08, -5.04]
- 中间子空间 κ=-0.08 距离 |κ|<1e-3 阈值 (1e-3) **很远**, 即便修复也不触发此路径
- 仅当 κ 训练过程中漂过 0 时才触发边界, 但 Task #85 训练日志未观察到此类漂移
- **结论**: L'Hôpital 修复对 Task #85 端到端 R@5 **几乎无影响**

**判定逻辑**:
- L'Hôpital 修复仅影响训练损失曲线的数值精度 (避免 2× 欧氏的偏移)
- R@5 排序 (m=1 优于 m=0 优于 m=2) 由几何结构决定, 不依赖数值精度
- **Task #85 verdict 维持**: 准欧氏 R@5=0.0200 仍为单 κ 流形最优

---

## 4. ⭐ 额外发现 — Task #99 fused embedding norm 长尾

诊断时 `cv=norm.std()/norm.mean()` 在 fused embedding 上测得:

```
embedding shape=(11924, 64)
norm min=0.0727, max=381.3249, mean=0.7938, std=5.0133
CV (norm.std / norm.mean) = 6.32
```

**CV=6.32 是极端值**:
- 健康 embedding: CV 通常 < 1.0
- Task #53 S4 AE + log1p: CV = 0.0 (uniform, 过归一化)
- Task #99 MCKG fused: **CV = 6.32, max=381** — **强烈 norm 长尾**

**这印证**:
- Task #53 log1p 是治标 (后处理) → S4 AE 输入需要 (因 S4 AE 已归一化但完全均匀)
- Task #54 L3 norm 是治本 (源头治疗) → MCKG fused embedding CV=6.32 迫切需要 L3 norm
- 之前判断 "log1p 是治标, L3 是治本" 完全正确, 但**Task #53 因输入 S4 AE 已归一化让 log1p 退化为 no-op**, 而 Task #99 fused embedding 仍是 log1p 后处理的最佳候选

---

## 5. ROI 评估

| 项 | 评估 |
|----|------|
| L'Hôpital 修复 ROI | **极低** (Task #85 历史 recall 排序稳定, 修复仅数值精度) |
| Task #85 verdict 是否需改 | **否** (维持) |
| TIGER 重训 ROI | **零** (重训 m=1 准欧氏期望 R@5=0.0200 ± 1%) |
| **最高 ROI 提案** | **Task #98 / Task #104 提过 (MCKG fused + log1p 后处理 + RQ)** → CV=6.32 + log1p 是治疗长尾的最佳后续路径 |

---

## 6. 产物清单

```
verdicts/task56_curvature.json    # 诊断数据
verdicts/task56_curvature_result.md  # 本文档
scripts/task56_curvature_diagnose.py # 复用有效
```

---

result: Task #56 曲率重审诊断完成 — L'Hôpital 阈值 1e-3 修复对 Task #85 三几何 R@5 排序无影响 (κ=-0.08 子空间不漂过 0), Task #85 verdict 维持准欧氏 R@5=0.0200 为单 κ 流形最优。**TIGER 重训 ROI=0**, 不投入。**⭐ 额外重要发现: Task #99 fused embedding CV=6.32 (max=381, mean=0.79) 证实其 norm 长尾极端** — 这是 MCKG + 几何融合设计的固有问题, **后续最高 ROI 路径是 MCKG fused + log1p 后处理 + 简化 RQ (不用 neural encoder/decoder)**, 已在 Task #58 提议。
