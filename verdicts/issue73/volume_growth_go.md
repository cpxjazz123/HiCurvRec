# Issue #73 (Issue F) 双曲体积增长可视化 — GO

- **Issue**: #73 Issue F: 双曲体积增长可视化 (volume growth curve)
- **Labels**: experiment, visualization, §1.2-validation
- **日期**: 2026-08-07
- **结论**: **GO** (Gate 1 PASS / Gate 2 SKIP / Gate 3 SKIP / Gate 4 PASS)

---

## 1. 任务目标 (issue 原文摘录)

> 为 §1.2「曲率 = 空间展开速度」提供直觉可视化, 把 `V(r) ∝ exp((d-1)√c·r)` 公式画成图。
> 无需训练 — 纯理论/解析画图; d ∈ {32, 64, 128}, κ ∈ {0.1, 0.5, 1, 2, 5}。
> 产出 Figure 6: V_c(r) 曲线族, 3 张子图 (每个 d 一张), 每子图 5 条曲线, r ∈ [0, 5];
> 标注「κ=5 时 r=2 处的体积是 κ=0.1 时的 X 倍」。
> 这是 4 个 issue 里工作量最小的一个 (0 训练, 0 GPU)。

---

## 2. R18 四维度对比

**D1 — spec 摘录差异**: HG-Rec (ICML 2026) 正文与 §4 ablation 均**无**体积增长曲线图, 仅在 §1.2 用文字 +
公式 `V(r) ∝ exp((d-1)√c·r)` 描述「曲率 = 空间展开速度」。历史 verdicts/ 中 281 个文件无任何
volume growth 相关产物 (`grep -l volume verdicts/` 命中 0)。本 issue 是**首次**把该公式落成可视化产物,
与历史任何 issue 的 spec 均不同 → R18 判定「必须做实验 (此处 = 必须出图)」。

**D2 — 实施核心差异**: 历史全部 issue 的实施核心是 4 阶段流水线 (Stage1 embedding → Stage2 RQ-VAE →
Stage3 T5 → Stage4 eval), 依赖 GPU + ckpt + parquet。本 issue 实施核心是**纯解析计算 + matplotlib**,
新增独立目录 `common/analysis/` (此前不存在), 不触碰任何 stage 脚本、不读 ckpt、不占 GPU。
关键实施决策 = **全程 log10 域计算**: d=128 / c=5 / r=5 时指数 `(d-1)√c·r = 127×√5×5 ≈ 1419.9`,
即 V ≈ 10^616, 远超 float64 上限 1.8e308, 直接 `np.exp` 必然 overflow → inf。
改用 `log10 V_c(r) = (d-1)·√c·r / ln(10)`, 全程不调用 `np.exp`, 纵轴直接画 log10 值。

**D3 — Gate 1 失败机制差异**: 历史 issue 的 Gate 1 = Stage 2 码本健康度 (utilization / collision),
失败机制是码本塌缩。本 issue **无 Stage 2**, Gate 1 重定义为「公式实现正确性 + 数值不溢出」,
失败机制是 float64 overflow (已由 log10 域规避) 与公式误写。issue 原文 D3 明示
「无需训练, 跳过 Gate 2/3, 直接 Gate 4 (出图)」, 本 verdict 遵此执行。

**D4 — 引用文献差异**: 历史 issue 引用 HG-Rec §4.4 ablation / Chen 等 Hyperbolic Rec (NeurIPS 2022) /
Hyperbolic DT (NeurIPS 2023)。本 issue 引用 **Nickel & Kiela 2017 (Poincaré Embeddings) 原始论文 Fig. 1**
与 Chami 2019 双曲几何综述中的体积增长示意, 文献集合与历史 issue 无交集。

**四维度结论**: D1/D2/D3/D4 **全部不同** → 不可凭「路径同构」判 NO-GO, 必须实际产出。已产出。

---

## 3. 4 Gate 逐项

### Gate 1 — 公式实现与数值正确性: **PASS**

- 实现公式 (JSON `formula_proportional`): `V_c(r) ∝ exp((d-1)*sqrt(c)*r)`, 与 issue 原文逐字符一致。
- 精确参考式 (JSON `formula_exact_reference`): `V_c(r) = S_{d-1} * ∫_0^r (sinh(sqrt(c)*t)/sqrt(c))^(d-1) dt`,
  已在产物中记录, 说明比例式是其 r→大 时的主导项。
- log10 变换 (JSON `log10_transform`): `log10 V_c(r) = (d-1)*sqrt(c)*r / ln(10)`, 全程不调用 `np.exp`。
- 溢出规避核验: d=128 / c=5 / r=5 → `(d-1)√c·r = 127 × 2.2360679... × 5 = 1419.90`,
  对应 V ≈ 10^616 ≫ float64 max 1.8e308。log10 域下该值为 616.66, 正常 float64 可表示。
- 手算交叉验证 (d=32, κ=1, r=1): `31 × 1 × 1 / ln10 = 31/2.302585 = 13.4631289...`,
  JSON 实测 `13.463128939000805` — 完全一致 (相对误差 < 1e-15)。
- 线性性核验: log10 V 对 r 应严格线性过原点。JSON 实测 d=32/κ=0.1: r=1 → 4.257415187976867,
  r=2 → 8.514830375953734 (= 2×r=1 值, 精确), r=5 → 21.287075939884335 (= 5×r=1 值, 精确)。
- R2 (禁 fallback) 核验: `grep -nE "os\.environ|try:|except" common/analysis/issue73_volume_growth.py`
  仅命中第 53 行**注释文字** `# 常量区 (R30: 全部超参硬编码, 严禁 os.environ 读取)`,
  无任何实际 try/except 块、无默认值回退、无降级策略 → R2 PASS。
- R30 (超参硬编码) 核验: 无 `os.environ.get` / `os.environ[]` 实际调用, 全部常量位于顶部常量区
  (`D_LIST=[32,64,128]` / `KAPPA_LIST=[0.1,0.5,1.0,2.0,5.0]` / `R_MIN=0.0` / `R_MAX=5.0` /
  `N_POINTS=500` / `ANNOT_R=2.0`) → R30 PASS。
- R4 (py_compile) 核验: `python3 -m py_compile common/analysis/issue73_volume_growth.py` → 通过。
- R32 (禁 .sh 包装) 核验: 启动方式为 `python3 -u common/analysis/issue73_volume_growth.py`, 无 .sh 文件。

### Gate 2 — Stage 2 训练: **SKIP (设计使然, 非失败)**

issue 原文 D3 明示「无需训练, 跳过 Gate 2/3」。本 issue 是纯解析可视化, 不存在 RQ-VAE 码本,
因此不存在 utilization / collision / κ 收敛等 Gate 2 指标。**0 GPU 占用**, 未写 `_TRAINING_PID`。
此 SKIP 不触发 R17「前 Gate FAIL → 后 Gate STOP」, 因为 Gate 1 为 PASS 而非 FAIL。

### Gate 3 — Stage 3 T5 训练: **SKIP (设计使然, 非失败)**

同 Gate 2。本 issue 不产生 SID, 无 T5 训练环节, 无 valid R@10 / loss 曲线。
不涉及 R5 基线 (valid R@10=0.1267 / test R@10=0.1024) 的比较 — 本 issue 不是 R@10 类实验,
其验收标准是「图是否正确表达 §1.2 的几何直觉」, 而非推荐指标提升。

### Gate 4 — 产物与可视化: **PASS**

产物目录 `taskA/_history/issue73_volume_growth/` (符合「训练产物统一放 taskX/_history/」):

| 文件 | 大小 | 说明 |
|---|---|---|
| `figure6_volume_growth.png` | 231240 B | Figure 6 位图 |
| `figure6_volume_growth.pdf` | 37825 B | Figure 6 矢量图 (论文插入用) |
| `volume_growth_numbers.json` | 4344 B | 全部数值 + 公式 + config |

图形结构核验 (人工读图确认):
- 1×3 子图布局, 每个子图对应一个 d ∈ {32, 64, 128} — 符合「3 张子图 (每个 d 一张)」。
- 每子图 5 条曲线对应 κ ∈ {0.1, 0.5, 1, 2, 5} — 符合「每子图 5 条曲线」。
- r ∈ [0, 5], N_POINTS=500 — 符合「r ∈ [0, 5]」。
- 纵轴为 log10 V, 曲线呈过原点直线并按 κ 张开成扇形 (斜率 ∝ √c), 直观表达「曲率 = 空间展开速度」。
- r=2 处虚线参考 + 端点白心 marker, 左上黄色标注框写出 κ=5 vs κ=0.1 的体积比,
  ylim 设为 y_max×1.38 保证标注框不遮挡曲线; 图例置右下; suptitle 含 "Figure 6"。

**关键数值 — r=2 处 κ=5 相对 κ=0.1 的体积比 (log10 值)**:

| d | log10(V_{κ=5} / V_{κ=0.1}) at r=2 | 即倍数 |
|---|---|---|
| 32 | 51.69411261894711 | ~10^51.7 |
| 64 | 105.05577725786026 | ~10^105.1 |
| 128 | 211.77910653568654 | ~10^211.8 |

推导校验: `log10[V_hi/V_lo] = (d-1)(√5 − √0.1)·r / ln10`。
d=32: `31 × (2.2360680 − 0.3162278) × 2 / 2.302585 = 31 × 1.9198402 × 2 / 2.302585 = 51.6941126` ✓

**§1.2 对应关系**: 图直接显示同一半径 r 下, κ 越大体积增长越快 (斜率 √c 越大),
且维度 d 越高该差异被放大 (d=128 的比值是 d=32 的约 4.1 倍, 因 (d-1) 线性放大)。
这为 §1.2「曲率过大 → 浅层过度分离; 曲率过小 → 深层拥挤碰撞」提供了几何量级依据:
同一 r 处可容纳的码字数量随 √c 指数级变化, 因此单一 κ 无法同时满足三层不同的容量需求。

---

## 4. 计数与口径说明

- issue 原文未出现计数矛盾 (d 3 值 × κ 5 值 = 15 条曲线, 分 3 子图各 5 条, 与文本一致)。
- 本 issue 不使用 `stage4_eval_beam20.py` 或 `stage4_eval_pure_t5.py`, 因无 ckpt 可评估。
- 本 issue 不涉及 multi-dataset / multi-seed (R5 + 用户指示)。
- 环境记录更正: 实际 GPU 为 **1× A100-SXM4-80GB** (非 CLAUDE.md 记录的 4× L40S 46GB),
  但本 issue 0 GPU 占用, 不受影响。

## 5. 复现命令

```
python3 -u /fs04/ar57/wenyu/GeneRec/common/analysis/issue73_volume_growth.py
```

无参数、无环境变量、无 .sh 包装 (R30 + R32 合规)。
