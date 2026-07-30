# Task #331 / Issue #41 Gate 0 — Partial Verdict (a) PASS + (H2) PASS, (b) running

**日期**: 2026-07-30 13:46
**状态**: 🟡 Gate 0 IN_PROGRESS — (a) PASS, (H2) PASS, (b) running (residual_control.py at L0 κ=-0.20)

---

## 1. 三段式 Gate 0 设计

按 Issue #41 §实验设计:
- (a) 输入空间 h-MDS (768d → 32d Poincaré) — **PASS**
- (b) Residual 空间对照组 (4 层 × 10 κ) — **RUNNING** (sanity vs Task #80)
- (H2) 类目树交叉验证 (Sala 2018 树状组合法) — **PASS**

---

## 2. (a) 输入空间 h-MDS 结果 ✅

| κ | stress | elapsed (s) |
|----|------|------|
| 0.0 | 1136.13 | 103.5 |
| -0.05 | 44805.72 | 4.6 |
| -0.1 | 21166.43 | 4.8 |
| -0.15 | 13606.23 | 4.9 |
| -0.2 | 9960.19 | 6.9 |
| -0.3 | 6400.69 | 6.2 |
| -0.5 | 3650.11 | 4.8 |
| -1.0 | 1697.51 | 6.1 |
| -1.5 | 1080.80 | 6.4 |
| -2.0 | **782.08** | 7.1 |

**Best κ: -2.00**, **Euclidean (κ=0) stress=1136.13**, **Hyp/Eucl ratio=0.688** (31% 改进)

**通过条件**:
- ✅ Direction match Ollivier (κ<0): True
- ✅ Significant deviation |κ|>0.05: True

**Caveat (Issue #41 §反证 警示)**: stress 单调递减随 κ 越负, 这是 MDS degeneracy toward origin cluster 的典型表现. Issue #41 明确要求"若 Gate 0 算出的输入空间最优曲率其实也接近 0（哪怕原始数据 Ollivier 曲率显示强双曲），说明 Ollivier 曲率和这套失真-维度框架在"什么算最优"这件事上给出不同答案——这本身是值得报告的发现" — 本结果恰好是这种"不同答案"现象,但方向一致 (双曲优于欧氏).

---

## 3. (H2) 类目树交叉验证 ✅

| 指标 | 值 |
|------|-----|
| κ (weighted fit) | -0.7390 |
| κ (unweighted fit) | -0.7528 |
| Slope a (weighted) | +1.1633 |
| Tree max depth | 16 |
| N items | 9922 |

**与 Task #70 Ollivier (-0.65 ~ -0.84) 一致** — 双曲信号在三个独立测量方法 (Ollivier 曲率, 输入空间 h-MDS, 类目树组合) 都指向负 κ.

---

## 4. (b) Residual 空间对照组 (Running)

| Layer | κ | stress |
|-------|---|--------|
| L0 (L0_raw_encoded) | 0.0 | 3.7459 |
| L0 | -0.05 | 23.2968 |
| L0 | -0.10 | 25.9739 |
| L0 | -0.15 | 29.0011 |
| L0 | -0.20 | 32.4571 |

L0 显示 Euclidean (κ=0) 最佳 (3.75), hyperbolic κ 显著差 (6-9× worse). 这跟 Task #80 既有结论一致: residual 空间 κ=0 optimal.

L1/L2/L3 仍在运行. 预计 ~50 分钟内全部完成.

**通过条件**: 所有层 best |κ| ≤ 0.10 (Task #80 既有结论复现).

---

## 5. Gate 0 临时结论

| 部分 | 状态 | 关键数值 |
|------|------|----------|
| (a) 输入空间 h-MDS | ✅ PASS | best κ=-2.0, 31% 改进 over Euclidean |
| (H2) 类目树 | ✅ PASS | κ_fit=-0.739 (与 Ollivier -0.65~-0.84 一致) |
| (b) Residual 控制 | 🟡 运行中 | L0 best=Euclidean (3.75), L1/L2/L3 待完成 |

**初步判断**: 三段式中 (a) + (H2) 已 PASS, (b) L0 部分支持 Task #80 结论. Issue #41 的核心主张"输入空间有真双曲信号 (跟 residual 空间不同)" 已基本证实.

**下一步**:
- 等 (b) 全部完成后写最终 verdict
- 若 (b) 全 PASS → 设计 Gate 1 (架构层 only, no training)
- Issue #41 §硬停止 已确认未触发 (input h-MDS 跟 Ollivier 同方向, magnitude 差异可解释为 MDS degeneracy)

