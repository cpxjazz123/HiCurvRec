# Task #163.5 — 真 Ollivier-Ricci Curvature 实现 + 重测 (用户反馈: avg_cc 不够)

> **任务目的**: 用户 2026-07-24 反馈 avg_cc ≠ ORC, 必须实跑真 ORC (Wasserstein-1 距离 + 测地距离比) 才能判 ORC 思路本身. 本任务替换 #163 Phase B 的 avg_cc proxy 实现.
> **完成日期**: 待启动
> **状态**: 🔴 待登记 (用户 23:30 反馈后必启动)

---

## 1. 背景

**Task #163 Phase B 选择**: avg_clustering coefficient 作为 Ollivier-Ricci Curvature (ORC) 的代理.
- avg_cc ∈ [0, 1] = 三元闭包数 / 三元组数 (networkx 实现 `nx.average_clustering`)
- 真 ORC ∈ ℝ = 1 - W_1(μ_x, μ_y) / d(x, y), μ 是 lazy random walk on neighbors (Lin et al. 2011)

**用户 2026-07-24 23:30 反馈**:
> "实际测的东西, 跟最初想测的 ORC 不是一回事... 这两个指标只是 '感觉有点像', 不是同一个数学概念. 所以就算最后结果不好, 也不能说 'ORC 这个思路不行', 只能说 '这次用的这个简化版指标不够好'."

**核心修正**: verdict §分析 必须分两层:
- **A. proxy 结果**: avg_cc → θ_init (这次跑出来的)
- **B. 真 ORC 结果**: W_1 + d_geodesic → θ_init (Task #163.5 跑出来的)
- 不允许用 §A 结论覆盖 §B 思路. 即使 §A 失败, §B 思路仍然开放.

---

## 2. 实验设计

### Stage 0 — 复检 Task #156 codeword + Instruments.inter.json (CPU, 5 min)
- 复用 Phase A 产物 (task163_codeword_L0/L1/L2.npy)
- 复用 Phase B item-item graph builder (G: 9922 nodes, 943112 edges)

### Stage 1 — 真 ORC 实现 (CPU, ~1-2 hr)

**算法 (Lin et al. 2011 "Ollivier Ricci Curvature", Ollivier 2009 "Coarse Ricci Curvature")**:
- 对无向图 G(V, E) 的边 (x, y):
  1. **μ_x (lazy random walk distribution)**: μ_x(z) = α · 1/(d(x)+1) if z = x, else α / (d(x)+1) · 1/d(x) for neighbor. 通常 α = 0.5.
  2. **W_1(μ_x, μ_y)** (Earth Mover's Distance / Wasserstein-1): 用 `scipy.stats.wasserstein_distance` 或 linear programming (LP) 算.
  3. **d(x, y)** (测地距离): networkx shortest_path_length 即可.
  4. **κ(x, y) = 1 - W_1(μ_x, μ_y) / d(x, y)**
- 边曲率可以是负数 (hyperbolic 桥接) 或正数 (spherical 抱团).

**实现策略** (针对 L0 32 组 × ~300 nodes):
- L0 子图 (max 527 nodes): 用 scipy.sparse + linear programming, 单组 ~30-60 sec
- 32 组总时间: ~30 min
- L1 prefix (296 groups, ≥10 items): 每组 ~10 sec → ~1 hr (但可以并行 8 worker)
- L2 prefix (跳过 per 用户风险 #1)

**优化方向** (R137 类似思路 — 简单实用):
- 不并行: 用 networkx 简化版 + `scipy.stats.wasserstein_distance` (近似)
- 完整 LP: 用 `ot.emd2` (Optimal Transport Python package, 需 pip install POT)

### Stage 2 — Per-group ORC stats + θ_init 推导 (CPU, 5 min)
- 输出 `verdicts/task163_5_real_orc_per_layer.json`
- 推导 θ_init 公式 (跟 Phase C 一致):
  - θ_init(l) = sign(mean_ORC_l) × κ_max × min(|mean_ORC_l|, 0.5)
  - |mean_ORC| < 0.05 → ±0.1 偏移

### Stage 3 — 比较 avg_cc vs ORC 在 (a) mean, (b) variance, (c) sign 三个维度 (CPU, 5 min)
- 输出 `verdicts/task163_5_orc_vs_proxy.json`
- 关键对比: "ORC l=0 vs avg_cc l=0 sign 一致率 / magnitude 比例"
- **决策触发** (三档):
  - **ORC = avg_cc (sign 同, magnitude 比例 ~1)** → proxy 实际上够用, verdict §A 结论可以覆盖 §B 思路
  - **ORC 跟 avg_cc sign 反** → proxy 误导, 必须 ORC verdict 是权威
  - **ORC 全接近 0 (||ORC_l=0 < 0.05)** → 数据无曲率信号, ORC 思路本身 OR proxy 思路都被判 fail

---

## 3. 决策触发 (vs Task #163 Phase B)

| ORC 实测 vs avg_cc | 含义 | verdict 处理 |
|---------------------|------|--------------|
| **ORC l=0 sign 一致 OR 全接近 0** | proxy (avg_cc) 跟 ORC 评估方向一致, 都是 "数据偏负" | verdict §B 可弱化, proxy §A 可信 |
| **ORC sign 跟 avg_cc 反 (l=0 avg_cc>0.5 ↔ ORC<0)** | proxy 跟真指标评估方向相反 | verdict §B 才是权威, §A 推翻, 必须 ORC 重跑 paper-free-curv Stage 1 |
| **ORC 跟 avg_cc 都 = 0** | 数据无曲率信号 | verdict §B 推迟, 跟 Task #89 NO-GO 一致 |
| **ORC 跟 avg_cc 都非 0 但 magnitude 差 ≥2 倍** | proxy 跟 ORC 仅 weak correlated | 两个都给, 备注 "ORC 才是权威, proxy 仅 sanity check" |

---

## 4. 预算

| Stage | 时间 |
|-------|------|
| Stage 0 (复用) | 5 min CPU |
| Stage 1 (真 ORC 测量) | 1-2 hr CPU (depending on L1 prefix 并行) |
| Stage 2 (推导) | 5 min CPU |
| Stage 3 (proxy vs ORC 比对) | 5 min CPU |
| **总计** | **~1.5-2.5 hr CPU** (no GPU) |

---

## 5. 风险与缓解

**风险 1**: 真 ORC 实现 bug. → 缓解: 用 Lin et al. 2011 paper Figure 2 toy graph (4-cycle 边 κ=0.5) 验证实现正确性 (unit test).

**风险 2**: scipy.stats.wasserstein_distance 在大图 (300+ nodes) 上慢. → 缓解: 用 `ot.emd2` (POT package, 优化版本), 或抽样 50 节点近似.

**风险 3**: ORC 测量 sign 跟 avg_cc sign 反 → 需要 Stage 4 paper-free-curv 重跑 (跟 Phase D 同样的 θ_init). → 缓解: 这是 low-frequency outcome, Stage 3 比对就是 dry-run.

---

## 6. R11.3 决策明示 (写入 loop.md §16 备注)

**(a) 选了哪个**: Task #163.5 真 ORC 实现 + 重测 + vs proxy 比较 (4 Stage: 复用 + 测量 + 推导 + 比对)

**(b) 为什么**:
- 用户 2026-07-24 23:30 反馈 avg_cc ≠ ORC (proxy ≠ 真指标), verdict 必须分两层
- 复用 avg_cc 测出来的 θ_init (proxy) 不能判 ORC 思路本身 (真)
- 边际成本: 1.5-2.5 hr CPU only (无 GPU)
- 跟 Task #163 Phase D GPU pipeline 不冲突 (并行推进)

**(c) 备选方案**:
- **A**: 不做真 ORC, verdict 仅用 avg_cc 结论 (违背用户反馈)
- **B**: 立即 kill Phase D, 用真 ORC 重跑 Stage 1 (Phase D 已跑 7 min, 浪费)
- **C**: 跑 Task #163.5 真 ORC 全流程 (并行, 不浪费 Phase D, verdict 双层化) (本次决策)

---

## 7. 完成度跟踪

- [ ] Stage 0: 复用 Task #163 codeword + graph
- [ ] Stage 1a: 写 ORC 实现脚本 (`scripts/task163_5_real_orc_measurement.py`)
- [ ] Stage 1b: Lin et al. 2011 toy graph 验证 (4-cycle κ=0.5 unit test)
- [ ] Stage 1c: 跑 L0 (32 组) + L1 prefix (296 组) 真 ORC 测量
- [ ] Stage 2: θ_init 推导 (sign + 截断 + 偏移)
- [ ] Stage 3: proxy vs ORC 比对 (sign / magnitude / variance)
- [ ] 写 verdict §B (真 ORC): `verdicts/task163_5_real_orc_followup_result.md`
- [ ] 更新 #163 verdict §A 引用真 ORC follow-up

---

result: Task #163.5 — 真 ORC Wasserstein-1 实现 + 重测 + vs avg_cc proxy 比对, 跟 #163 Phase D GPU pipeline 并行推进 (CPU only). 重点看 ORC sign 跟 avg_cc sign 是否一致 (决定 verdict 权威性).
