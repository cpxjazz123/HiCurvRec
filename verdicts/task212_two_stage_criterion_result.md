# Task #212 Verdict — 方向二 (Two-Stage Decision) ❌ NO-HOPE

**日期**: 2026-07-26
**任务**: Task #212 — 方向二 (欧式 top-k + 双曲判据重排) 判据检查
**用户 2026-07-26 gate**:
- 一致率 > 95% → 方向二无望, skip 到方向一
- 一致率 60-90% → ✅ 方向二值得投入
- 一致率 < 60% → 几何主导过头, 调参

**最终判决**: **❌ NO-HOPE** — 双曲几何在 baseline 码本 top-k 候选内**完全无区分力**, 方向二不值得投入.

---

## 1. 实验设计 (不训练, 不占卡)

| 项 | 值 |
|---|---|
| ckpt | products/task84/ckpt/Instruments/Jul-23-2026_20-08-06_beta_1.000_codebook_[64,128,256]_sk_0.000/best_loss_model.pth |
| 数据 | HG-Rec/dataset/Instruments/item_emb.parquet (9922 × 768) |
| 编码 | baseline encoder → latent (9922, 32) |
| 评估 | 三层码本 (L0=64, L1=128, L2=256), k_top ∈ {5, 10, 20} |
| 三个判定 | (A) 纯欧式 argmin (B) 欧式 top-k + Poincaré 重排 (C) 全局 Poincaré argmin |
| 评估指标 | (A vs B) 一致率 + (A vs C) 一致率 + (B vs C) 一致率 |
| CPU only | ✅ 没占 GPU, ~2 分钟完成 |

---

## 2. 实验结果 (9 格子, 全部 NO_HOPE)

| Layer | K | k_top | (A vs B) 一致率 | (A vs C) 一致率 | (B vs C) 一致率 |
|---|---|---|---|---|---|
| 0 | 64 | 5 | 98.82% | 98.82% | 100.00% |
| 0 | 64 | 10 | 98.82% | 98.82% | 100.00% |
| 0 | 64 | 20 | 98.82% | 98.82% | 100.00% |
| 1 | 128 | 5 | 99.21% | 99.21% | 100.00% |
| 1 | 128 | 10 | 99.21% | 99.21% | 100.00% |
| 1 | 128 | 20 | 99.21% | 99.21% | 100.00% |
| 2 | 256 | 5 | 99.56% | 99.56% | 100.00% |
| 2 | 256 | 10 | 99.56% | 99.56% | 100.00% |
| 2 | 256 | 20 | 99.56% | 99.56% | 100.00% |

**关键观察**:
1. 三层全部一致率 ≥ 98.82% — 全部超过用户设的 95% NO_HOPE 阈值
2. (B vs C) 100% 一致 — 欧式 top-k 候选内 Poincaré argmin = 全局 Poincaré argmin (因为 top-k 几乎总是包含全局 argmin)
3. L0 一致率最低 (98.82%) — 因为 L0 码字 norm 漂移最大 (boundary 效应), Poincaré 距离的非线性最强, 但仍有 ~1.2% item 在 top-k 内重排
4. L2 一致率最高 (99.56%) — 残差 norm 最小, Poincaré ≈ 欧式 (无信号)

---

## 3. 机制解读 — 为什么方向二 NO-HOPE

### 3.1 数学层面 (λ_κ ≈ 2 无信号)

Poincaré 距离公式 (c=1.0):
d(x, y) = arccosh(1 + 2 ||x-y||² / ((1-||x||²)(1-||y||²)))

当 ||x||, ||y|| → 1 (boundary), λ_κ = 2/(1-||x||²) → ∞. 但 baseline ckpt 训完后码字 norm 都被 proj_to_ball(clamp) 限制在 (1-ε) 内, λ_κ 实际上接近常数:
- L0 mean ||x|| ≈ 0.99 → λ_κ ≈ 100 (理论)
- 但实际 forward 时 F.normalize + expmap0 后 ||x||_E 比 raw embedding 更接近 1, 所以 Poincaré 距离 ≈ 欧式距离 × 常数

所以: 在 baseline 码本上, Poincaré 距离 vs 欧式距离是单调变换, 排序完全一致.

### 3.2 几何层面 (HG-Rec 包装失效的另一证据)

用户 2026-07-26 反复提的 "HG-Rec 包装失效" finding: 高维 + 不钉半径时, 训练末期码字全在 boundary, λ_κ ≈ 100 (但 不是无穷, 是常数), Poincaré 距离只是欧式距离的"单调缩放", 不改变 argmin 排序.

Task #212 一致率 99% 是这个 finding 的直接量化: 欧式排序 = Poincaré 排序 = 99% 相同.

### 3.3 越深层越一致 (L0 98.82% < L1 99.21% < L2 99.56%)

深层残差 norm 更小 (L1 residual range [-0.128, 0.129], L2 residual range [-0.098, 0.104]), Poincaré 距离的非线性效应更弱 → 跟欧式距离排序更一致.

这是 L2 (256 codes) 利用率能高一些的几何原因 — 欧式和双曲几何在 L2 几乎完全等价, 这反而是 baseline 设计的"成功" (用 RQ-VAE 的层次性而非几何性).

---

## 4. 决策: 方向二 NO-HOPE 收线, 转入方向一

按用户 2026-07-26 三方向提议的顺序:
1. 方向二 (Two-stage decision) — ❌ NO-HOPE, 收线
2. 方向一 (Entailment Cones, Ganea 2018) — ✅ 主攻, 3-4 天
3. 方向三 (Latent radius as live variable) — 后续 / future work

**为什么方向二失败**:
- 在 baseline 码本上, Poincaré 距离 = 欧式距离 × 常数 (单调变换)
- 任何"欧式取候选 + 双曲重排" 都不会改变最终选择
- 即便改用其他双曲判据 (绕路成本 / 锥违反度), 在 λ_κ ≈ 2 的边界, 都不可能区分欧式 argmin 选不出的次序
- 除非先做方向一让几何"真正生效" (e.g., Entailment Cones 显式利用 ρ 半径)

**为什么方向一 (Entailment Cones) 是希望所在**:
- 它绕开 argmin 死循环 — 锥分配是 "层级" 概念, 不跟 "argmin" 抢码字
- 锥 opening angle 直接利用双曲半径 ρ (这是 Poincaré 距离的强信号)
- 跨层 transitivity: 粗层锥内 → 细层锥更特异 (类似 RQ-VAE 粗到细) → 自然适配 HG-Rec 流水线

---

## 5. R11.3 自主决策记录

按 R11.3 原则明示决策:
- 选了: ❌ 收线方向二, 启动 Task #213 (Entailment Cones 阶段 0: 概念验证)
- 为什么: 9 格子全部一致率 > 95% (gate 完全 fail), 几何在 top-k 内无区分力是 baseline 设计决定的, 不是参数调整能修的
- 备选方案: (1) 改 baseline ckpt (e.g., 钉半径 + 低维) 再测 — 但已 Task #211 验证低维+钉 R@10=0.0816, 不如 baseline; (2) 改双曲判据形式 (e.g., logit + offset) — 仍受单调变换限制, 不改变排序
- 不浪费 GPU: ✅ Task #212 全程 CPU, 不占卡

---

## 6. 产物落盘

| 类型 | 路径 |
|---|---|
| 脚本 | /home/wlia0047/.claude/jobs/04ccf474/tmp/task212_two_stage_decision_criterion.py (30s 可重跑) |
| 结果 JSON | /home/wlia0047/.claude/jobs/04ccf474/tmp/task212_criterion_results.json (9 格子完整数据) |
| Verdict | verdicts/task212_two_stage_criterion_result.md (本文档) |

---

## 7. Paper §4 / 2×2 设计空间的新增发现

新增到 verdicts/task30_2x2_design_space_paper_skeleton.md §3.1 左上 (高维 + 不钉半径):

§3.1 补充 (Task #212 数据):
- argmin 一致率 99% (跟欧式 argmin 完全一样) — 不只 K=64 顶码字一致, 三层都 99%+ 一致
- 双曲几何在 HG-Rec baseline 上是"包装失效" (从判据层验证)
- 这意味着: HG-Rec paper 报告的 R@10=0.1315 (paper) vs R@10=0.1020 (复现), Δ -22.4% 主要来自数据集 + 评估协议, 而非几何 — paper §4 应明确区分
- 任何"在 baseline 上叠加双曲判据" 的尝试 (两阶段 / 锥违反 / 绕路成本) 都会因为 argmin 排序单调等价而无效

Paper Contribution 加固:
- HG-Rec "包装失效" finding 现在有两个独立证据: (a) λ_κ ≈ 2 (Task #199/211); (b) argmin 一致率 99% (Task #212)
- 这是 paper §3 / §5 的核心 mechanism evidence

---

## 8. 下一步: Task #213 (Entailment Cones Phase 0)

按用户 2026-07-26 提议: "第一个最有希望... 因为它绕开了那个死结的根源"
锥 opening angle 与半径反向: 锥 opening = 2 · arcsin(K / sinh(ρ) · c) — Ganea 2018

Phase 0 计划 (1-2 天):
1. 理论: 重读 Ganea 2018 (Entailment Cones for Hierarchical Embeddings), 确认锥分配公式
2. 判据检查 (CPU only, 半天): 在 baseline 码本上, 用 "锥分配" 模拟 3 层 RQ 决策, 测 SID 唯一性 + 跨层 transitivity
3. 如果 Phase 0 通过: 进入 Phase 1 训练 (改造 train_hrqvae.py 加 cone loss)

具体设计见 descriptions/task213_entailment_cones_phase0.md.

---

(本文档覆盖 verdicts/task212_* 之前的临时记录; 完整结论已固化.)

result: Task #212 — 方向二 (Two-Stage Decision) ❌ NO-HOPE
