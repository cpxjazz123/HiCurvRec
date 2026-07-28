# Task #199 — exp(θ) κ 参数化 4 臂 Stage 1 (用户 2026-07-26 根因 + 修正方案)

> **任务目的**: 验证 κ 失败 8 次的根因 (κ_max 小 50 倍, 搜索范围不足), 改 exp(θ) 参数化覆盖 [1, 400], 验证 idea 本身是否成立 (vs 双码本降级版).
> **完成日期**: (in progress)
> **状态**: 🟡 待启动 (3 臂并行 GPU 1/2/3)

---

## 1. 背景

用户 2026-07-26 核心发现:

| c | λ (= sinh²(ρ_median)·c) |
|---|---|
| 1 | 2.15 (现状) |
| **2** | **2.32** ← 历史 κ_max 天花板 (tanh(θ) ∈ [-2, 2]) |
| **30** | **10.5** ← 健康窗口 |
| 100 | 109 |

**8 次 κ-Stereo NO-GO 原因**: 搜索范围差两个数量级, 不需要"半径没钉住".

**新参数化**: `c = exp(θ).clamp(max=(5/r_median)²)`, 覆盖 [1, 400+], 梯度不饱和.

**逐层 κ 的硬理由**: 三层半径不同 (ρ0≈2.0, ρ1≈2.7, ρ2≈3.4), 达到同样 λ≈10 需要的 c 差 18 倍:

| 层 | 需要 c |
|---|---|
| L0 | ≈ 30 |
| L1 | ≈ 250 |
| L2 | ≈ 550 |

**安全性**: 切空间参数化下不可能撞球壁 (码字 + 球壁一起往里收), 安全上限 L0 c=345, L1=2860, L2=6260.

**双码本不废弃, 降级成"更干净的版本"** — 先用最便宜方式验证 idea 本身.

---

## 2. 实验设计

### 4 臂 Stage 1 (RQ-VAE, GPU 并行)

| 臂 | 配置 | κ 行为 | GPU | 备注 |
|---|------|-------|-----|------|
| **A** | 官方基线 c=1.0 固定 | constant | (复用 #181) | 历史 baseline |
| **B** | 单个可学习 c, exp(θ) 参数化 | learn scalar | GPU 1 | 用户首要验证: c 学到 > 1 且 λ 进 [5, 75]? |
| **C** | 逐层可学习 c_ℓ, exp(θ_ℓ) 参数化 | learn per-layer | GPU 2 | 验证三层 c 明显不同 |
| **D** | c 固定扫描 {1, 10, 30, 100} | constant (4 子臂) | GPU 3 (4 顺序跑) | "几何不影响性能" 正面证据 |

**保持不变 (跟 #181 baseline 对齐)**:
- 数据: `HG-Rec/dataset/Instruments/item_emb.parquet` (9922 items × 2048d)
- epochs=1000 (paper Table 6 Instruments)
- batch_size=256
- β=0.5
- sk_epsilons=[0,0,0] (paper HG-Rec 默认关)
- e_dim=32, num_emb_list=[64,128,256]
- lr=1e-3, kmeans_iters=1000
- loss_type=poincare

**变量**:
- B: 单个可学习 scalar `c_global = exp(θ).clamp(max=c_max)`
- C: 逐层可学习 `c_ℓ = exp(θ_ℓ).clamp(max=c_max_ℓ)` where `c_max_ℓ = (5/r_target_ℓ)²`
- D: c 固定 (4 个子臂, 顺序跑)

### 启动命令

```bash
# 臂 B (GPU 1)
bash scripts/task199_stage1_arm_B.sh

# 臂 C (GPU 2)
bash scripts/task199_stage1_arm_C.sh

# 臂 D (GPU 3, 4 子臂顺序)
bash scripts/task199_stage1_arm_D.sh
```

---

## 3. 决策触发 (vs HG-Rec #181 baseline)

| 检查 | 通过标准 |
|------|---------|
| **臂 B**: c 学到 > 1 且 λ 进 [5, 75] | ✅ 原始 idea 成立 |
| **臂 C**: 三层 c 明显不同 (factor > 5×) | ✅ 逐层自适应成立 |
| **臂 D**: 性能不随 c 变 (R@10 波动 < ±0.005) | ✅ "几何不影响性能" 正面证据 |
| **臂 B/C**: c 还是不动 (θ ≈ 0) | ❌ 那时才需要双码本 (回 #200 双码本方案) |

---

## 4. 预算

| 阶段 | 估算时间 |
|------|---------|
| 3 臂并行 Stage 1 (1000 epoch) | ~2.5 h per 臂 |
| 4 子臂 D 顺序 | ~4 × 2.5 h = ~10 h (1 GPU) |
| Stage 2 推断 (4 臂) | ~5 min × 4 |
| Stage 3+4 评估 (如果 Stage 1 通过) | ~1.5 h × 4 |
| 总计 (Stage 1 验证 idea) | ~2.5 h |
| 总计 (含 Stage 2-4) | ~10 h |

---

## 5. 风险与缓解

**风险 1**: c 学到 NaN / Inf (θ 爆炸) → clamp(max=c_max_ℓ) 安全
**风险 2**: exp(θ) 初始梯度太大 → 初始化 θ=0 → c=1 (等同 baseline)
**风险 3**: 三层 c 互相竞争 → 用 per-layer θ_ℓ 独立参数
**风险 4**: Stage 1 完成后 ρ_median 偏离 r_target → c_max clamp 自动收紧

---

## 6. 完成度跟踪

- [ ] patch HVectorQuantization exp(θ) 参数化 (model/utils.py)
- [ ] patch train_hrqvae.py argparse 加 --kappa_mode / --kappa_init / --kappa_per_layer
- [ ] 写 3 臂 launcher
- [ ] 启动 3 臂并行 (GPU 1/2/3)
- [ ] 监控: 每 100 epoch 打印 c, ρ_median, λ
- [ ] Stage 2 推断 + Stage 3+4 评估 (idea 通过后)
- [ ] 写 verdict + 更新论文 §5

---

## 7. 与 #200 双码本关系

按用户 2026-07-26 指示:
- #200 双码本不废弃, 降级成"更干净的版本"
- #199 优先 (cheaper verification of idea itself)
- #200 Stage 3 dual_v5 继续做 (不暂停), 完成后跑 Stage 4 eval
- #199 + #200 verdict 合并: 决定下游 Stage 3+4 用 exp(θ) κ 还是 dual codebook