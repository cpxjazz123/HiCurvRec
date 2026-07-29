# Task #273 — Task #178 §6.7.4 stop-loss (i) 复核

> **完成日期**: 2026-07-29
> **状态**: 🟡 **BORDERLINE FAIL** — L0=89.06% 差 §6.7.4 stop-loss (i) 阈值 90% 共 0.94pp, 严格判定 FAIL. L1/L2 彻底坍缩 (0.78% / 0.39%).

---

## 1. 直接测量结果

**ckpt**: `products/task178/hgrec_fixed_baseline/Jul-25-2026_04-30-51/.../best_collision_model.pth`

| 层 | unique | total_capacity | fraction | §6.7.4 stop-loss (i) 阈值 | 判定 |
|---|---|---|---|---|---|
| **L0** | **57** | **64** | **89.06%** | ≥ 90% | ⚠️ **差 0.94pp** (borderline) |
| L1 | 1 | 128 | 0.78% | (未规定) | ❌ **彻底坍缩** (单码字) |
| L2 | 1 | 256 | 0.39% | (未规定) | ❌ **彻底坍缩** (单码字) |

**collision_rate_pre_resolve**: 0.9943 (99.43% items 撞到同一 SID)

**n_unique_sid**: 57 (跟 L0 unique count 吻合, 因为 L1/L2 各只有 1 个码字 → 整个 SID 仅由 L0 决定)

**utilization_avg_over_layers**: 30.08% (平均下来偏坍缩)

## 2. §6.7.4 stop-loss (i) 判定

按 §6.7.4 hard stop-loss (i): L0 utilization < 90% → **STOP**.

| 条件 | threshold | 实测 | 判定 |
|------|-----------|------|------|
| L0 ≥ 90% | ≥ 58/64 | **57/64 = 89.06%** | ❌ **严格 FAIL (borderline -0.94pp)** |

**结论**: Task #178 R@10=0.1135 ckpt **违反** §6.7.4 stop-loss (i), 严格判定 over-闸. 但**数值极 borderline** (差 < 1pp), 不是显著越闸.

**注意时序**: Task #178 训练时 (2026-07-25) §6.7.4 stop-loss (i) **还没硬实施** (Issue #17 Gate 1 修复 2026-07-29). 当时跑完才知. 这是**潜在 historical 越闸**, 跟 Issue #13 类似但 borderline 不是 99.6% 崩塌.

## 3. 重要反预期发现

### 3.1 L1/L2 严重坍缩但 R@10 仍然最高

Task #178 的 Stage 3 R@10=0.1135 居然是历史最高 (vs HG-Rec baseline 0.1020, +11.3%), 但其 Stage 1 ckpt 是:
- L0 borderline 89.06% (勉强够)
- **L1 只用 1 个码字**
- **L2 只用 1 个码字**

**含义**: T5-mini Stage 3 只用 L0 这一层的码字就能生成足够多样的 SID 序列. L1/L2 在 Stage 3 中**没有提供判别能力**, 几乎是无意义层.

### 3.2 后续 baseline 跟 task178 差距

| baseline | R@10 | L0 | L1 | L2 |
|---|---|---|---|---|
| Task #84 (HG-Rec baseline) | 0.1020 | 73.44% | 100% | 100% |
| Task #253 (直接测量) | 0.0938 (Task #225 Stage 4) | 73.44% | 100% | 100% |
| **Task #178 (fixed baseline)** | **0.1135** | **89.06%** | **0.78%** | **0.39%** |

**Task #178 vs Task #84**:
- L0 高 15.62pp (89 - 73)
- L1 塌陷到 0.78% vs 100% (-99pp)
- L2 塌陷到 0.39% vs 100% (-99pp)
- R@10 高 11.3%

**含义**: L0 utilization 提升对 R@10 至关重要, 比 L1/L2 利用率更重要. 这是 §6.7.4 stop-loss (i) 设 L0 ≥ 90% 而不看 L1/L2 的**经验性正确**.

### 3.3 候选 5 (L0 ≥ 90% 阈值重审) 浮现

Task #178 案例让 §6.7.4 stop-loss (i) 的 90% 阈值有了反例:
- 89.06% < 90% 严格 FAIL, 但实际跑出 R@10=0.1135
- 73.44% (baseline) > 任务 #84 R@10=0.1020

**是否应放松阈值?**: 这是 R11.4 critical decision (影响项目 §6.7.4 rule). **不建议** AI 自主决策. 候选 5 留待用户拍板.

## 4. 物理产物 + 量化数字

```
products/task273/task178_bestcollision_utilization.json  (verifier 输出)
{
  "ckpt_path": "...task178/.../best_collision_model.pth",
  "n_items": 9922,
  "n_unique_sid": 57,
  "collision_rate_pre_resolve": 0.9943,
  "utilization_avg_over_layers": 0.3008,
  "per_layer_utilization": {
    "layer_0": {"unique": 57, "total_capacity": 64, "fraction": 0.8906},
    "layer_1": {"unique": 1, "total_capacity": 128, "fraction": 0.0078},
    "layer_2": {"unique": 1, "total_capacity": 256, "fraction": 0.0039}
  }
}
```

## 5. 关键决策点 (R11.3)

- **共享 verifier 补丁**: 加 `bn=ckpt_args.get("bn", False)` (默认 False 不破坏 task253 兼容). R12 严格存.
- **判定 borderline FAIL**: 严格按 §6.7.4 stop-loss (i) 数字判, 不放宽. 但在 verdict §3 标注 borderline 性质.
- **不立 issue 但提到候选 5**: 阈值重审是 R11.4 critical decision, AI 不立 issue.
- **不重跑 Task #178**: 已存 ckpt 不动, 物理产物保留作为 §6.7.4 实际判定依据.

## 6. 物理产物

```
descriptions/task273_task178_utilization_audit.md
verdicts/task273_task178_utilization_audit_result.md  (本文件)
products/task273/task178_bestcollision_utilization.json
scripts/task263_issue17_gate2_task253_direct_utilization_meas.py  (patched with bn kwarg)
```

result: Task #273 — task178 L0=57/64=89.06% (差 §6.7.4 stop-loss (i) 0.94pp borderline FAIL), L1=1/128=0.78% 严重坍缩, L2=1/256=0.39% 严重坍缩. 关键反预期: L1/L2 坍缩但 R@10=0.1135 仍是项目最高 → L0 utilization 是真正 R@10 杠杆, §6.7.4 设阈值正确但 90% 边界有 89% 反例 (候选 5: 阈值重审 R11.4 不立).
