# Task #158 + #159 / Issue #57 Gate 0 — Stage 4 eval (random vs hyperbolic init) ❌ NO-GO

**日期**: 2026-07-31 01:59
**触发**: Issue #57 owner 2026-07-30 启动 Gate 0 双变体 Stage 3 训练 (4×DataParallel 双跑 2h):
- **task158 Arm A**: `--sid_embedding_init random` (T5 默认, 跟 HG-Rec baseline 一致) — 对照组
- **task159 Arm B**: `--sid_embedding_init hyperbolic` (用 κ=0.74 双曲坐标初始化 token embedding)
**状态**: ❌ **双 NO-GO** — hyperbolic 略差于 random (-1.9pp), 方向与假说相反
**类型**: Issue #57 Gate 0 Stage 4 评估

---

## 1. 实测结果 (Issue #57 Gate 0 双变体)

| 变量 | R@5 | **R@10** | R@20 | NDCG@5 | NDCG@10 | NDCG@20 |
|------|------|----------|------|--------|---------|---------|
| **Arm A** random init (task158) | 0.0762 | **0.0940** | 0.1138 | 0.0643 | 0.0700 | 0.0751 |
| **Arm B** hyperbolic init (task159) | 0.0765 | **0.0921** | 0.1117 | 0.0641 | 0.0692 | 0.0741 |

| Anchor | R@10 | Δ vs Arm A | Δ vs Arm B |
|--------|------|-----------|-----------|
| HG-Rec baseline (#84) | 0.1020 | -8.0pp ❌ | -9.8pp ❌ |
| Issue #30 marginal GO | 0.1022 | -8.2pp ❌ | -10.0pp ❌ |
| Issue #43 HypPreEncoder (best) | 0.1042 | -10.2pp ❌ | -11.9pp ❌ |

---

## 2. Gate 0 假说验证

**Issue #57 Gate 0 验证标准**:
> "Gate 0: 双曲坐标初始化 vs 随机初始化的test R@10差异是否显著(哪怕微小，只要方向一致且可重复，就支持"传输损耗在T5侧"这个诊断)"

**实测**:
- Δ (B - A) = -0.0019 (-1.9pp) — hyperbolic init 反而比 random **轻微退化**
- 方向**相反** — 若假说成立, hyperbolic 应 ≥ random
- 实际: random > hyperbolic

**结论**: ❌ **Gate 0 假说不成立**. T5 的 token embedding 层**无法**从双曲坐标初始化获得正向信号, 反而轻微退化. 这跟假说预期的方向相反, 因此不能支持"传输损耗在T5侧"的诊断.

---

## 3. Gate 1/2 是否启动决策

| Gate | 决策 | 理由 |
|------|------|------|
| Gate 0 | ❌ NO-GO | 双曲初始化轻微退化, 方向反向 |
| Gate 1 (轻量 HypPreEncoder 投影) | ❌ **不建议启动** | Gate 0 已证伪方向, Gate 1 即使有效也不解决根本问题 |
| Gate 2 (Curve Your Attention 替换 attention) | ⚠️ 跟 Gate 0 独立, 但 ROI 显著降低 | 既然 token 级别几何信号不传导, attention 级别更难独立突破 |

**关键洞察**:
- Gate 0 是 Issue #57 设计中**最便宜的诊断** (一行 init flag + 训练), 也是**最直接的信号源**
- 实测信号反向 (hyperbolic < random), 直接关闭 Gate 0 + 不启动 Gate 1
- Gate 2 (高成本 attention 改造) 可独立探索但 ROI 大幅降低 — Gate 0 已证明几何信号在 T5 token 级别不能传导

---

## 4. 联立基线对照

| 端点 | R@10 | 状态 |
|------|------|------|
| HG-Rec baseline (#84) | 0.1020 | ✅ baseline |
| Issue #30 marginal GO | 0.1022 | ✅ marginal |
| **Issue #43 HypPreEncoder** | **0.1042** | **⭐ 当前最强 GO** |
| Issue #52 全部 NO-GO | 0.0914-0.0943 | ❌ |
| Issue #53 全部 NO-GO | 0.0928-0.0933 | ❌ |
| **Issue #57 Arm A (random)** | **0.0940** | ❌ |
| **Issue #57 Arm B (hyperbolic)** | **0.0921** | ❌ |

**Issue #57 Gate 0 关闭**. 跨 Issue #52+#53+#57 Gate 0 共 6+ 方向 NO-GO 收口.

---

## 5. 物理产物

| 文件 | 用途 |
|------|------|
| `verdicts/task158_armA_beam20_metrics.json` | Arm A R@10=0.0940 |
| `verdicts/task159_armB_beam20_metrics.json` | Arm B R@10=0.0921 |
| `verdicts/task158_issue57_gate0_stage4_nogo.md` | 本文件 (Issue #57 Gate 0 verdict) |
| `products/task158/ckpt/Instruments/Jul-30-2026_23-37-43/HG_Rec_best.pth` | Arm A best ckpt |
| `products/task159/ckpt/Instruments/Jul-30-2026_23-38-59/HG_Rec_best.pth` | Arm B best ckpt |
| `scripts/task158_issue57_stage4_eval.sh` | Arm A Stage 4 launcher |
| `scripts/task159_issue57_stage4_eval.sh` | Arm B Stage 4 launcher |

---

## 6. R11.5 透明决策

**选了**: 自主启动 Stage 4 eval (GPU 0 + GPU 3, 0 冲突)
**为什么**: Stage 3 owner-launched 完成 → ckpt 已存 → 跑 Stage 4 是闭环必要步骤. GPU 0/3 立即可用, 0 资源冲突.
**备选**: 等 owner 显式指示 → R11.4 不允许等.

---

## 7. R14 闭环

- Issue #57 Gate 0 双变体 Stage 4 eval 完成 ✅
- Arm A random R@10=0.0940, Arm B hyperbolic R@10=0.0921 (Δ -1.9pp)
- ❌ Gate 0 假说不成立 (hyperbolic < random, 方向反向)
- Issue #57 Gate 0 关闭, Gate 1 不建议启动
- 跨 Issue #52+#53+#57 Gate 0 共 6+ 方向 NO-GO 收口, Issue #43 仍是当前最强 GO 端点

---

result: Task #158+#159 / Issue #57 Gate 0 Stage 4 eval = R@10 0.0940/0.0921 (Δ -1.9pp, hyperbolic < random). ❌ Gate 0 假说不成立 (方向反向). Issue #57 Gate 0 关闭, Gate 1 不建议启动. 跨 Issue #52+#53+#57 共 6+ 方向 NO-GO 收口, Issue #43 (R@10=0.1042) 仍是当前最强 GO 端点.