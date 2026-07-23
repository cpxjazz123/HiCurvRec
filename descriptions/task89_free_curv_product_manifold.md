# Task #89 — 自由曲率乘积流形 RQ-VAE (基于 MCKG 框架)

> **任务目的**: 验证 **per-layer × per-component 可学习曲率 κ_m** 是否能让 RQ-VAE 学到非零曲率, 并测其对下游 recall 和 codebook 利用率的真实贡献. 这是对 Task #88 (固定曲率网格) 的扩展: 不预设曲率符号/大小, 让模型自己决定几何.

> **完成日期**: (in progress)
> **状态**: 🟡 待启动

---

## 1. 背景

承接 Task #88/#116/#117/#118 系列结论:

| 前置任务 | 关键结论 |
|----------|----------|
| Task #88 | 6 网格 R@10 跨度仅 0.0998-0.1051 (5.3%), per-layer curvature 边际效应弱 |
| Task #116 | δ_95/d 跨 c111/c555 差异 <5%, 改造版反而**略不树状** (+1-5%) |
| Task #117 | 8 (层 × 版本) 组合 best κ=0 全胜, 应力比 κ<0 高 4-50× |
| Task #118 | 6 curvature codebook 利用率都是 100%, 3-token SID 碰撞 ~10% 几乎相同 |

**核心矛盾**: Task #82 (phonism) + Task #88 (HG-Rec) + Task #70 (Ollivier κ_real≈0.7) 三重证据都指向"Musical_Instruments 数据集**几何上接近欧氏**", 但用户仍想测**"完全不预设曲率, 让模型自己选"**这个终极对照. 若 κ_m 学完仍接近 0, 强烈支持"机制而非几何"; 若某个分量 κ_m 显著非零, 则重新审视固定 c 网格的扫描范围.

**假设**:
- **R1**: 即便不预设曲率符号/大小, 大部分 κ_m 会收敛回接近 0 → 数据本质是欧氏
- **R2**: 乘积流形 M=2 或 M=3 的某个分量可能长出非零 κ_m → 子空间分解后局部结构和整体结构不同
- **R3**: κ_m 偏离 0 但 recall 不变 → 可能是过拟合/伪影, 应加 L2 正则

---

## 2. 实验设计

### 2.1 Stage 0 — 零训练成本纯几何检验 (~30 min, CPU only)

**目的**: 在不训练的前提下, 对 Task #84 (c=[1,1,1]) HRQ-VAE 的 4 层残差做**分块** stress-metric 网格搜索. 如果分块后最优 κ 仍清一色在 κ=0 附近, 基本可以预判 Stage 1 也是同样结果 (省 GPU).

**做法**:
- 加载 Task #84 ckpt
- 提取 4 层 residual (n × 32 维)
- 把 32 维按 M=2 切成 16+16, 按 M=3 切成 11+11+10
- 对每块单独跑 Task #80 `compute_true_distortion` κ ∈ {0, -0.05, ..., -2.0} 网格
- 报告 best κ per block per layer

**启动命令**:
```bash
python3 scripts/task89_stage0_block_stress.py \
    --ckpt /home/wlia0047/ar57/wenyu/GeneRec/products/task84/ckpt/Instruments/Jul-23-2026_20-08-06_beta_1.000_codebook_[64,128,256]_sk_0.000/best_loss_model.pth \
    --emb /home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/dataset/Instruments/item_emb.parquet \
    --M_list 1 2 3 \
    --output verdicts/task89_stage0_block_stress.json
```

### 2.2 Stage 1 — A/B/C 三臂完整训练 (~18 h GPU)

**修改范围**: 仅替换量化器 (`HVectorQuantization` / `HResidualVectorQuantization`) 内部的流形定义和距离计算. encoder/decoder/训练循环/Sinkhorn/Stage 2/Stage 3/Task #118 利用率脚本全部原样保留.

**核心改动** (per-quantizer layer `m`):

```python
# 重参数化: θ_m -> κ_m ∈ [-κ_max, κ_max], κ_max=2
self.theta_m = nn.Parameter(torch.zeros(1))   # init → κ_m=0

@property
def kappa_m(self):
    return self.kappa_max * torch.tanh(self.theta_m)   # init 0 → κ=0

# 编码后: 欧式 → U^{n_m}_{κ_m}
e_m = expmap_k(x_m, self.kappa_m)            # κ=0 时为恒等映射

# 量化: 用 Table 1 统一测地距离 (κ<0→Poincaré, κ=0→Eucl, κ>0→球面)
d = geodesic_dist_k(e_m, c_j, self.kappa_m)

# 残差更新: Möbius 加法 (κ=0 时退化为普通减法)
x_residual = mobius_add_k(x_m, -c_j, self.kappa_m)
```

**跨分量距离合成 (Baseline)**:
```python
dist(x, c) = sqrt(Σ_m d_{κ_m}(x_m, c_m)²)
```

**优化**: codebook/θ_m 都是 `nn.Parameter`, 用标准 Adam (不走 geoopt 黎曼优化器), 沿用 HG-Rec 训练稳定性经验.

### 2.3 Stage 2 (可选) — D 臂门控融合

```python
w_m = softmax(MLP(concat([x_1, ..., x_M])))
dist(x, c) = Σ_m w_m · d_{κ_m}(x_m, c_m)
```

仅当 Stage 1 中 B 或 C 出现有意义的 κ_m ≠ 0 信号时启动. 否则跳过以省 GPU.

### 2.4 实验臂

| 臂 | M | 组合方式 | 说明 |
|----|---|----------|------|
| **A** (对照) | 1 | — | κ_m 自由学习单一统一空间 (32 维整块), 等价于 task84/task88 c 可学习版 |
| **B** | 2 | baseline (√Σ d²) | 16+16 切分 |
| **C** | 3 | baseline (√Σ d²) | 11+11+10 切分 |
| **D** (可选) | 2 | MCKG 式门控 | 隔离"融合机制"新变量 |

每层 (L0/L1/L2) **独立**训练, κ_m 不共享 (每层每分量各自学习) — 呼应 phonism L0-L3 δ 趋势问题.

### 2.5 启动命令 (Stage 1 模板, A 臂示例)

```bash
# A 臂: M=1, κ 自由学习
python3 scripts/task89_stage1_train_rqvae.py \
    --M 1 \
    --kappa_max 2.0 \
    --codebook_sizes 64 128 256 \
    --num_epochs 1000 \
    --batch_size 256 \
    --seed 42 \
    --output_dir products/task89/train/arm_A_M1

# B 臂: M=2
python3 scripts/task89_stage1_train_rqvae.py \
    --M 2 --kappa_max 2.0 \
    --codebook_sizes 64 128 256 \
    --num_epochs 1000 \
    --batch_size 256 \
    --seed 42 \
    --output_dir products/task89/train/arm_B_M2

# C 臂: M=3
python3 scripts/task89_stage1_train_rqvae.py \
    --M 3 --kappa_max 2.0 \
    --codebook_sizes 64 128 256 \
    --num_epochs 1000 \
    --batch_size 256 \
    --seed 42 \
    --output_dir products/task89/train/arm_C_M3
```

每臂 Stage 1 训练 ~6h (与 task84 量级相同, 1000 epoch HRQ-VAE), ×3 臂 = ~18h GPU, 4 卡并行 (单臂单卡).

Stage 2/3/4 流程复用 task84/task88 现有脚本 (`task88_stage2_codebook.py` + `task84_hgrec_stage3_train.py` + `task88_stage4_eval.py`), 仅 codebook/code_path 后缀改为 `curv_free_M{m}` 形式.

---

## 3. 决策触发 (vs baseline)

| 指标条件 | 结果 | 决策 |
|----------|------|------|
| **所有 κ_m 收敛回 |κ|<0.1** + B/C recall 仍优于 A | 强烈支持 "机制而非几何" — 提升来自多分量融合, 不是曲率 |
| **某个分量 κ_m 显著偏离 0 (|κ|>0.5)** + 该臂 recall/碰撞率改善 | 推翻 "数据本质欧氏" — 子空间结构不同, 启动 Stage 2 (D 臂) 做进一步拆解 |
| **κ_m 偏离 0 但 recall/利用率无相应提升** | 伪影/过拟合 — 加 θ_m L2 正则 (penalty=0.01) 重跑 A/B/C 做稳健性检验 |
| **R@10 跨 A/B/C/D 跨度 >5%** | 假设 H2 部分确认 (机制有效) |
| **R@10 跨 A/B/C/D 跨度 <2%** | 假设 H2 否证 (机制无效), 归档 NO-GO |

vs baseline = Task #84 R@10=0.1020 (HG-Rec 单 c=1.0).

| R@10 区间 | 决策 |
|-----------|------|
| > 0.115 | 强确认 → 写 P5 paper section, 启动 Stage 2 D 臂 |
| 0.105-0.115 | 部分确认 → 跑第二种子验证, 看 run-to-run 噪声 |
| 0.099-0.105 | 与 baseline 持平 → 写 NO-GO verdict, 归档 |
| < 0.099 | 改造破坏 → 立即停止后续训练 |

---

## 4. 预算

| 阶段 | 估算时间 |
|------|---------|
| Stage 0 (block stress) | ~30 min CPU |
| Stage 1 A 臂 (HRQ-VAE 1000 epoch) | ~6 h GPU |
| Stage 1 B 臂 | ~6 h GPU |
| Stage 1 C 臂 | ~6 h GPU |
| Stage 1 合计 (3 臂并行) | **~6 h GPU wallclock** (4 卡并行) |
| Stage 2/3/4 (每臂 × 3 臂) | ~3 × 1.5 h = ~4.5 h GPU |
| Stage 2 D 臂 (可选, 仅 Stage 1 有意义信号时) | +6 h |
| Task #118 利用率分析 (每臂 × 6 臂) | ~10 min × 6 = 1 h GPU |
| **总计** | **~12-18 h GPU** |

4 卡 L40S 全空闲, 3 臂并行 ~6 h wallclock 即可完成 Stage 1.

---

## 5. 风险与缓解

**风险 1 — 分量坍缩**: 某分量训练中 codebook 利用率掉到 0 (dead code), κ_m 读数无意义
→ 缓解: 每臂训练中每 100 epoch 打印 per-component 利用率; 任一分量 <50% 立即 kill 该臂, 报告 "分量坍缩 NO-GO"

**风险 2 — run-to-run 噪声**: κ_m 训练终值不稳, 单 seed 不可靠
→ 缓解: A/B/C 每臂留 1 重复种子 (seed=42 + seed=123), 看 κ_m 终值 std 是否 <0.1

**风险 3 — 边界数值不稳定**: κ 过大时 1/√(-κ) 边界急剧收缩 → 梯度爆炸
→ 缓解: `κ = κ_max · tanh(θ)` 有界重参数化, κ_max=2 (用户指定) 不会触发极端值

**风险 4 — 训练初始化差异**: θ_m=0 起点看似"无偏", 但和 task88 c=1.0 起点不同, recall 差异未必来自"几何"
→ 缓解: A 臂作为必要基准 (M=1, κ 自由学习), 用 A vs Task #88 c111 比对"自由学习 vs 固定 c=1.0" 本身的差异

**风险 5 — Stage 0 全 κ=0 但 Stage 1 仍可能长出非零 κ**: 训练优化可能突破零训练态的局部最优
→ 缓解: Stage 0 仅做"提前止损"判据, 不阻塞 Stage 1

---

## 6. 完成度跟踪

### Stage 0 (block stress 分块预检)
- [ ] task89_stage0_block_stress.py 写完 + py_compile 通过
- [ ] Stage 0 跑完 (Task #84 ckpt × 4 layer × M∈{1,2,3} × 11 κ)
- [ ] verdicts/task89_stage0_result.md 写出 (Stage 0 决策: 进入 Stage 1 / 提前归档)

### Stage 1 (3 臂完整训练)
- [ ] task89_stage1_train_rqvae.py 写完 + py_compile 通过 (支持 M=1/2/3 + κ_m 可学习)
- [ ] A 臂 (M=1) Stage 1 训练完成 (epoch 1000)
- [ ] B 臂 (M=2) Stage 1 训练完成
- [ ] C 臂 (M=3) Stage 1 训练完成
- [ ] 每臂的 (a) 最终 κ_m 值 per layer, (b) θ_m 训练曲线落盘
- [ ] verdicts/task89_stage1_arm_{A,B,C}_result.md 各自写出

### Stage 2/3/4 (下游 T5 训练 + 评估)
- [ ] 每臂 Stage 2 codebook 生成 (`_curv_free_M{m}_t5_hrqvae_poincare.npy`)
- [ ] 每臂 Stage 3 T5 训练完成 (early stop)
- [ ] 每臂 Stage 4 test eval 完成 (R@5/R@10/R@20/NDCG)
- [ ] verdicts/task89_stage4_eval_*.json 落盘

### 综合分析
- [ ] 6 臂 R@10 跨臂对比表
- [ ] κ_m 终值 vs R@10 散点图 (验证 R1/R2/R3)
- [ ] Task #118 利用率对比 (6 臂 codebook 健康度)
- [ ] verdict: verdicts/task89_free_curv_product_manifold_result.md 写出 (含可证伪预期表 + 决策触发结果)
- [ ] 更新 loop.md §16 (R8 归档)

---

## 7. 参考资料

- MCKG Table 1 统一算子表 (expmap_k / logmap_k / geodesic_dist_k / mobius_add_k)
- HG-Rec 论文 §3.3 HResidualVectorQuantization
- Task #80 compute_true_distortion (Stage 0 复用)
- Task #118 codebook utilization (Stage 1 后复用)
- Task #82 B 加密网格 κ ∈ {0, -0.05, -0.10, -0.15, -0.20, -0.25, -0.30, -0.50, -1.00, -1.50, -2.00}

---

## 8. 关键决策点 (R11.3 自决)

1. **Stage 0 提前止损**: 用户原始建议"如果 Stage 0 全 κ=0 可考虑不投入完整训练", 我保留为"提前止损信号"而非硬阻塞 — 因训练优化可能突破零训练态
2. **κ_max=2**: 用户给定, 不追问
3. **θ_m=0 初始化**: 用户给定, κ_m=0 起点 = 真正的"无偏"
4. **M=2 切 16+16, M=3 切 11+11+10**: 用户给定 (≈ 均分), 不重新设计
5. **跳过黎曼优化器 (geoopt)**: 用户指定沿用 HG-Rec 经验, 不引入复杂度
6. **A 臂作为必要基准**: 用户提到 "M=1 κ 自由学习", 我把它从"对照"提升为"必要基准" — 这样能隔离"曲率自由学习 vs 多分量融合"两个变量
7. **每臂重复种子**: 用户提到"每臂留 1 重复种子", 我直接采用 seed=42 + seed=123 两组