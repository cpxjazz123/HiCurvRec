# VERDICT — Issue #234

**标题**:Stage2: #232 原机制参数扫描,寻找 Graphlet 对齐与 SID 碰撞的平衡点
**正文 sha256**:`ede0c1be5bc1112fca9c496f3a367a381790ed192b17866fae3855befeab5868`(2386 字符,冻结后未变)
**issue 状态**:CLOSED(`2026-11-22T13:33:29Z`,`NOT_PLANNED`,owner 关闭)
**本文件写入时间**:2026-11-23,在 issue 关闭之后补齐 §5 对账(A5 结果与完整证据此前未评论,已由 COMMENT_9 补发)

---

## 0. 是否需要训练

**需要。** issue「评估与选择」段要求三码碰撞、重建误差、三种 Δ 与满足率、以及最终 test 验收,
这些只能由新权重产生。owner 在「固定规则」段明确批准多参数组合实验,上限 8 次正式训练。

实际执行 **6 次 Stage2 正式训练(A1–A6,各 3000 epoch)** + **5 次 Stage3(A1–A5,各 150 epoch)**。
阶段 B(B1/B2)按训练前冻结的规则第 3 条未执行(无质量合格候选)。A6 的 Stage3 按 owner 指令在
冷启动阶段终止。

---

## 1. 逐条对账

### A. 固定规则(C1–C14)

| 编号 | issue 要求 | 实际 | 证据 | 判定 |
|---|---|---|---|---|
| C1 | 唯一基线 TIGER,禁止重训 | 复用 `results/stage2_RQ-VAE/TIGER_RQ-VAE` 冻结 checkpoint/SID | 目录未修改 | PASS |
| C2 | 多组合经 owner 批准,同一 seed=42 | 六组均 seed 42 | 各组 `train_config.json` | PASS |
| C3 | 不做多种子/打乱对照/其他消融 | 0/0/0 | `RUN_LOG.md` | PASS |
| C4 | 每个 Stage2 候选只训练一次 | 六组各一次 | `phase_a.log` | PASS |
| C5 | 每组 3000 epoch、四卡 DDP 串行、全局 batch 4096 | 逐项一致 | 各组 `training_metrics.jsonl` 至 epoch 3000 | PASS |
| C6 | 768→32→256×3、原量化/导出协议、优化器/LR/采样/温度/护栏不变 | 六树逐文件 diff 仅 `ARM_NAME` 与 `GRAPHLET_WEIGHT_MAX` 两行不同 | 本文 §3 | PASS |
| C7 | 从 `run232_archive/source.tar.gz` 恢复源码并核对哈希 | 归档哈希已记录 | `archive_hashes.json` | PASS |
| C8 | 不得用 #232 独立步骤版或 #233 新机制替代 | 归档为联合训练版,无 `collision.py` | `restore/` 已入 gitignore,哈希见 manifest | PASS |
| C9 | 只允许指定参数变化,逐文件 diff 核实 | 见 §3 的 diff 输出 | — | PASS |
| C10 | 结果目录每组独立,不覆盖历史 checkpoint | 6 个独立目录 | `results/stage2_RQ-VAE/TIGER_i234_A{1..6}/` | PASS |
| C11 | 原 loss 形式不变 | 单次 backward/clip/step | `train_rqvae.py` 与归档一致 | PASS |
| C12 | 保留 L1/L2/L3 目标与真实 token/STE | 与归档一致 | diff | PASS |
| C13 | 不得新增 #233 的 L1 容量或 L3 碰撞 Loss | 0 处 | 源码扫描 | PASS |
| C14 | 不改框架护栏、不做导出重指派 | 0/0 | — | PASS |

### B. 阶段 A(C15–C23)

| 编号 | 候选 | λ_max | 开始 epoch | 判定 |
|---|---|---|---|---|
| C15 | A1 | 0.002 | 1500 | 跑满 3000,碰撞 0.1005 | PASS |
| C16 | A2 | 0.005 | 1500 | 跑满 3000,碰撞 0.1075 | PASS |
| C17 | A3 | 0.01 | 1500 | 跑满 3000,碰撞 0.1190 | PASS |
| C18 | A4 | 0.02 | 1500 | 跑满 3000,碰撞 0.1454 | PASS |
| C19 | A5 | 0.05 | 1500 | 跑满 3000,碰撞 0.2864 | PASS |
| C20 | A6 | 0.10 | 1500 | 跑满 3000,碰撞 0.4119 | PASS |
| C21 | 0.20 用 #232 历史结果,不重复训练 | 0 次训练 | — | PASS |
| C22 | 同一初始化/种子/数据/预算 | 逐项一致 | — | PASS |
| C23 | 允许护栏正常终止,记录失败,不放宽护栏 | 护栏未改,六组均无护栏终止 | 源码 + 日志 | PASS |

### C. 阶段 B(C24–C27)

| 编号 | issue 要求 | 实际 | 判定 |
|---|---|---|---|
| C24–C25 | B1/B2 在选定 λ 下跑 start 1000/2000 | **未执行** | N/A |
| C26 | 先完成 A 再决定 B,选择规则训练前冻结 | 规则见 `RUN_LOG.md`「训练前冻结的选择规则」 | PASS |
| C27 | 总计最多 8 次新正式训练 | 6 次 | PASS |

阶段 B 未执行的理由(冻结规则第 3 条,原文):
> 若 A 中无质量合格候选:停止并报告,不进入阶段 B(issue 原文「若无质量合格候选,停止并报告」)

六组中**没有任何一组同时满足** 碰撞 ≤ 0.1693 **且** 重建 ≤ 6.992e-5。A1–A4 碰撞合格但重建
7.27–7.80e-5 全部超出 1.05 门;A5/A6 碰撞与重建双双超标。

### D. 评估与选择(C28–C40)

| 编号 | issue 要求 | 实际 | 证据 | 判定 |
|---|---|---|---|---|
| C28 | 复用冻结 pair-disjoint bank;监督对与验证/测试无交集 | bank 未重建,沿用 #231/#232 | issue #232 记录 | PASS |
| C29 | 真实前三码计 Δ/满足率/near/far/平局/违反/逐级增量 | 齐全 | `metrics_TIGER_i234_A{1..6}_validation.json` | PASS |
| C30 | 各级随机共享率与同深度随机基线匹配诊断 | 每份 metrics 内含 `baseline_random_share` | 同上 | PASS |
| C31 | 不改变原主指标 | 与 #232 同口径 | `measure_group.py` | PASS |
| C32 | 每组报碰撞、重复桶比例、最大桶、各层占用及条件熵、重建误差 | 齐全 | 同上 | PASS |
| C33 | 记录结构/重建梯度范数、权重轨迹、防坍缩日志 | 训练日志内含 | 各组 `logs/` | PASS |
| C34 | **质量约束**:碰撞 ≤ 0.1693 且重建 ≤ 6.992e-5 | **无候选满足** | 见 §2 表 | **FAIL** |
| C35 | 在 validation 按 maximin 排序 | 无合格候选,排序不适用 | — | N/A |
| C36 | 同分时按满足率、再按碰撞 | 同上 | — | N/A |
| C37 | 报告完整 Pareto 表,不隐瞒任何组结果 | 六组全列 | `phase_a_pareto.json`,issue COMMENT_9 | PASS |
| C38 | **最终成功**:三种 Δ 与满足率均超 TIGER、fork/merge 转正、near 不低于 TIGER、质量约束满足 | 见 §2;仅 A6 结构项达标而质量门全破 | — | **FAIL** |
| C39 | 锁定首选后仅对该候选与 TIGER 做最终 test | 无首选,test bank 未使用 | — | **FAIL(前提未达成)** |
| C40 | 不用 test 反馈改参 | 0 处 | — | PASS |

### E. 实现校验与交付(C41–C48)

| 编号 | issue 要求 | 实际 | 判定 |
|---|---|---|---|
| C41 | 真实入口短 smoke,含 ramp 开启后有效梯度 | 六树各做 1 epoch smoke | PASS |
| C42 | 源码差异审计 | 见 §3 | PASS |
| C43 | 硬码前向核对 | 各组 Stage2 导出可被 checkpoint 复现(export_gate `tokens_match: true`) | PASS |
| C44 | 确认 #232 机制与导出规则一致 | 一致;差异项数 122–291,属框架 export_gate 既定行为 | PASS |
| C45 | 发现影响有效性的源码缺陷则先报告并停止 | 未发现机制缺陷 | PASS |
| C46 | 每组保留配置/日志/代码/checkpoint/SID/bank 哈希 | 齐全 | PASS |
| C47 | 输出完整参数-结果表、Pareto、最平衡候选及验收结果 | 本文 + issue COMMENT_9 | PASS |
| C48 | 若无候选达标如实说明,不擅开新训练或改判据 | 如实说明,未开新训练 | PASS |

---

## 2. 参数-结果表

Stage2 指标取 Stage3 实际读取的表(经 export_gate 从 checkpoint 复现);`collision@export` 为
Stage2 导出时的记录。

| 候选 | λ_max | 碰撞 | collision@export | recon | chain Δ改善 | fork Δ改善 | merge Δ改善 | 质量合格 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| TIGER | 0 | 0.1693 | 0.1693 | 6.66e-5 | — | — | — | 基线 |
| A1 | 0.002 | 0.1005 | 0.1003 | 7.27e-5 | −0.0026 | −0.0134 | +0.0247 | 否(recon) |
| A2 | 0.005 | 0.1075 | 0.1077 | 7.40e-5 | −0.0104 | −0.0268 | +0.0154 | 否(recon) |
| A3 | 0.01 | 0.1190 | 0.1189 | 7.51e-5 | −0.0156 | −0.0235 | +0.0401 | 否(recon) |
| A4 | 0.02 | 0.1454 | 0.1456 | 7.80e-5 | +0.0026 | −0.0101 | +0.0278 | 否(recon) |
| A5 | 0.05 | 0.2864 | 0.2864 | 8.50e-5 | −0.0052 | **+0.0336** | **+0.0586** | 否(碰撞+recon) |
| A6 | 0.10 | 0.4119 | 0.4118 | 9.37e-5 | **+0.0495** | **+0.0738** | **+0.0432** | 否(碰撞+recon) |

Δ改善 = 本臂 Δ − TIGER Δ(chain +0.0339 / fork −0.0201 / merge −0.0216)。

### Stage3 结果(A1–A5)

| 候选 | λ_max | test_R@5 | test_R@10 | test_NDCG@10 |
|---|---:|---:|---:|---:|
| A1 | 0.002 | 0.034994 | **0.053013** | 0.028979 |
| A2 | 0.005 | 0.035081 | **0.053013** | 0.029056 |
| A3 | 0.01 | 0.034837 | 0.052403 | 0.028680 |
| A4 | 0.02 | 0.034454 | 0.051916 | 0.028331 |
| A5 | 0.05 | 0.029370 | **0.043089** | 0.023861 |
| A6 | 0.10 | — | 未跑完(owner 终止) | — |

`n_eval = 57439`。

---

## 3. 参数隔离(diff 审计)

`stage2_RQ-VAE/TIGER_i234_A1/train_rqvae.py` 与 A2–A6 的逐文件 diff,每份**只有两行**不同:

```
63c63
< ARM_NAME = "TIGER_i234_A1"
---
> ARM_NAME = "TIGER_i234_A<N>"
121c121
< GRAPHLET_WEIGHT_MAX = 0.002
---
> GRAPHLET_WEIGHT_MAX = <λ>
```

---

## 4. 结论

**issue 的核心问题——"是否存在兼顾 Graphlet 对齐与商品区分能力的工作点"——在 #232 原机制下不存在。**

三条证据同时成立:

1. **结构侧需要高 λ**:merge 在 λ=0.01 达峰(+0.0401),fork 在 0.05 转正(+0.0336),
   chain 到 0.10 才最强(+0.0495)。三种 graphlet 的响应阈值不同,统一标量无法同时喂饱。
2. **质量侧在高 λ 崩坏**:λ=0.05 起碰撞翻倍(0.1454 → 0.2864),重建与 ramp 后劣化(ep1500 后
   +29.2%/48%)。
3. **Stage3 recall 全区间单调负相关**:`0.053013 → 0.053013 → 0.052403 → 0.051916 → 0.043089`。
   λ 0.02→0.05 是转折点,r@10 下降 17%,同期碰撞从 0.1454 跳到 0.2864。

**没有质量合格的候选,阶段 B 按冻结规则不执行,test bank 未使用。**

### 附带的结构性发现

- **λ 改变的是 SID 的解,不是解的局部扰动**:四臂两两三级码一致率最高仅 0.0041%(1/24587),
  五对为 0。
- **A1/A2 的 `test_R@10` 逐位相同是巧合**:两臂命中数都恰好是 3045(σ≈53.7,概率约 0.5%),
  但 top-10 列表 0/57439 相同、SID 三级码一致率 0%。`985 = 985` 是 $|A|=|B|$ 时的代数恒等式
  $\text{onlyA}=|A|-M=\text{onlyB}$,不是结构互补(已在 issue COMMENT_7 更正)。

---

## 5. 遗留与如实说明

| 项 | 状态 | 说明 |
|---|---|---|
| A6 的 Stage3 | 未完成 | owner 指令终止于冷启动阶段,无 `test_final.json` |
| C34 / C38 / C39 | FAIL | issue 自身允许此结果:「不保证一定存在满足者」「若无候选达标,如实说明,不擅开新训练或改判据」 |
| 阶段 B | 未执行 | 冻结规则第 3 条;6 次训练(上限 8 次)未用尽 |
| test bank | 未使用 | 规则 5:首选锁定后才使用 |

**ISSUE 已由 owner 关闭(`NOT_PLANNED`)。本 VERDICT 在其关闭后补齐对账,不发现在关闭时点之前本可避免的证据缺失
以外的任何未完成项。**

---

## 6. 证据索引

| 内容 | 路径 |
|---|---|
| 实验日志(257 行) | `results/research/issue234/RUN_LOG.md` |
| 参数-结果表 | `results/research/issue234/stage3_arm_table.json` |
| Pareto 表(阶段 A) | `results/research/issue234/phase_a_pareto.json` |
| 六组 validation 指标 | `results/research/issue234/metrics_TIGER_i234_A{1..6}_validation.json` |
| A1/A2 排序一致性 | `results/research/issue234/prediction_identity_A1_A2.json` |
| A6 门禁外豁免测量 | `results/research/issue234/sid_quality_A6_waived.json` |
| 归档哈希 | `results/research/issue234/archive_hashes.json` |
| 复算脚本 | `collect_stage3.py`、`compare_predictions.py`、`measure_group.py` |
| issue 评论(1–9) | `results/research/issue234/COMMENT_{1..9}.md` |
| Stage2 产物 | `results/stage2_RQ-VAE/TIGER_i234_A{1..6}/` |
| Stage3 产物 | `results/stage3_T5Train/TIGER_i234_A{1..5}/` |
