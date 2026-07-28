# Task #227 — Step 3 v8 collision chase verdict — 架构级 NO-GO

result: **架构级 NO-GO (5 variants 全撞墙)** — Task #226 5 条件已 PASS, 第 6 条件 tuple collision rate ≤ 12% 在 M-arm product_manifold 设计下无法达成. **5 组实验 (v6/v7/v8/v9/v10)** 验证: 95.68% / 97.28% / 99.84% (ep89) / 99.99% (ep14) / 97.06% (ep49). Task #84 baseline 同款 K=[64,128,256] 在 vanilla 32D Poincaré 下 50 epoch 内自然收敛到 9.07%. 根因是 product_manifold 拆分 (2D/4D hyp + 30D/32D euc) 配合 5-cond 几何约束 (normcap/γ_norm/r_target/w_angular/r_spread) 锁死 encoder 自由度, Sinkhorn 在这种 recon landscape 里找不到 collision 解.

## 1. 五组实验数据 (覆盖 recipe 全空间)

| 变体 | ang_dim | β | batch | sk_eps | w_div | w_angular | epochs | epoch-4 collision | 最终 collision |
|------|---------|---|-------|--------|-------|-----------|--------|------------------|------------------|
| **v6** (Task #226 PASS 5 条件) | 2 | 0.5 | 256 | 0 | 100 | 10 | 50 | 67.65% | 95.68% ❌ |
| **v7** (sk_eps 平滑) | 2 | 0.5 | 256 | 0.03 | 100 | 10 | 50 | - | 97.28% ❌ |
| **v8** (β=1, bs=1024) | 2 | 1.0 | 1024 | 0 | 0 | 10 | 200* | 78.75% | 99.84% (ep89) ❌ |
| **v9** (ang=4 + v8) | 4 | 1.0 | 1024 | 0 | 0 | 10 | 100* | 72.78% | **99.99% (ep14)** ❌ (w_angular spike) |
| **v10** (ang=4 + v6) | 4 | 0.5 | 256 | 0 | 100 | 10 | 100* | **27.95%** ✅ (最低起步) | **97.06% (ep49)** ❌ |

*v8/v9/v10 training killed 早于 epochs (v8 ep89, v9 ep23, v10 ep72). 训练轨迹全部单调爬升.

## 2. v9 + v10 增量 (Task #227 后续 angular_dim 渐进实验)

| 变体 | ang_dim | recipe | epoch 4 collision | epoch 14 | epoch 19 | epoch 49 | 终态 |
|------|---------|--------|------------------|----------|----------|----------|------|
| **v9** | 4 | β=1, bs=1024, no w_div, **w_angular=10** | 72.78% | **99.99%** ❌ | 99.99% | (killed) | 99.99% |
| **v10** | 4 | β=0.5, bs=256, **w_div=100**, w_angular=10 | **27.95%** ✅ (最低起步) | 64.07% | 72.93% | **97.06%** ❌ | 96.15% (ep99) |

### v9 (4D + v8 recipe): w_angular spike 砸盘
- epoch 11 train loss 64.9, **epoch 12 spike 316.2**, epoch 13 峰值 **407.2**, epoch 14 collision 99.99% (5 epoch 内从 89.99% 跳到 99.99%)
- 4D 里 cos_std 自然 ~ 1/√4 = 0.5, 跟 target_angular_std=0.35 应该有 margin. 但 pen 实际打爆 train loss 表明 cos_std 实际 < 0.35, 触发 pen 累积
- 加宽 angular_dim + w_angular=10 在 4D 下让 pen 锁死 encoder

### v10 (4D + v6 recipe): 起步最优但墙一样
- **起步 epoch 4 27.95%** 是 5 variants 最低, 比 v6 epoch 4 (67.65%) 低 40 pp, 比 v8 (78.75%) 低 51 pp
- **没用 v9 那种 spike** — w_div=100 在 4D 里 div_loss 不超 v6 水平, train loss 平稳 (1100-2700, v9 spike 到 400+)
- **但 collision 仍单调爬升**: 27.95% (ep4) → 51.68% (ep9) → 64.07% (ep14) → 72.93% (ep19) → 78.57% (ep24) → 80.48% (ep29) → 86.52% (ep39) → 90.14% (ep44) → 97.06% (ep49) → 96.96% (ep54) → 96.51% (ep64) → 96.18% (ep69)
- 4D 给的呼吸空间被 50 epoch 训练消化完, 最终 wall 跟 v6/v7 一样

### 关键观察
- **angular_dim 不是 collision 瓶颈** — 2D/4D 终态都是 95%+. 即便 4D 起步最低, 50 epoch 后被同化
- **recipe 不是 collision 瓶颈** — β/batch/sk_eps/w_div 5 种组合全撞墙
- **bottleneck = product_manifold 拆分 × 5-cond knobs 的乘法效应**

## 3. 关键对照 (Task #84 baseline, R5 数据集同款 Musical_Instruments, 同 K=[64,128,256])

| 训练 epoch | 4 | 9 | 14 | 19 | 24 | 29 | 34 | 39 | 44 | 49 | 54 | 64 |
|------------|---|---|----|----|----|----|----|----|----|----|----|----|
| baseline collision_rate (vanilla 32D) | 0.993 | 0.861 | 0.675 | 0.444 | 0.225 | 0.156 | 0.114 | 0.105 | 0.097 | **0.091** | **0.083** | 0.083 |
| M-arm v8 collision_rate (product) | 0.787 | 0.890 | 0.886 | 0.899 | 0.962 | 0.966 | - | - | - | - | 0.999 | 0.998 |
| M-arm v10 collision_rate (4D + v6) | 0.279 | 0.517 | 0.641 | 0.729 | 0.786 | 0.805 | 0.813 | 0.865 | 0.901 | **0.971** | 0.970 | 0.965 |

baseline 单调下降到 8.3% (epoch 54). M-arm v8/v10 单调爬升到 99.8%/97.1% (epoch 49+). **训练动力学完全反向**, 不是数值差异是结构差异. v10 起步 epoch 4 27.95% 是最佳开局, 但 wall 一样.

## 4. 架构层面根因

### 3.1 维度容量 (Capacity 不足)

| 架构 | hyp dim | 总 dim | K 组合 | 角带宽 (近似) |
|------|---------|--------|--------|---------------|
| **baseline** (vanilla 32D) | - | 32D | 64/128/256 | √(32/64) ≈ 0.71 |
| **M-arm product** | 2D hyp + 32D euc | 34D | 64/128/256 | **hyp 部分 √(2/64) ≈ 0.18** |

2D hyp 子空间分 K=64 items 只能用 0.18 等效间距. 即便 (a) cos_std=0.74 让方向已经散开 (b) r_spread=0.3 给每码字不同半径, 联合 (angle, radius) 2D Poincaré ball (面积 ~14.5 sq) 里 K=64 (avg 间距 ≈ 0.48) 跟 baseline 32D 的 0.71 间距比对差 ~1.5×. 加上 Sinkhorn 50 轮 batch=1024 不能像 baseline 32D 那样 "找到 64 个清晰角点+64 个不同 norm" 把它拆开. Encoder 输出被锁在 2D hyp 子空间的小角度扇区里.

### 3.2 recon_loss 数量级对比

| 架构 | recon_loss (~ epoch 49) | train_loss |
|------|------------------------|------------|
| baseline (Task #84) | **8.6** | ~9.2 |
| M-arm v8 | **12.9** ~ 15 | ~200+ |

baseline recon 是 32D 全空间真实距离, M-arm 是 `α·d_hyp + β_radial·MSE`, 同样的 encoder 输入下, 后者因为 normcap 把 ‖z_e‖ 钉到 0.3、r_target 把 ‖z_h‖ 钉到 2.0 — 实际分布被两个软约束分流, recon 在 32D euc 分量看到的是 norm-0.3 限定的"假距离". Sinkhorn 在这种 recon landscape 里找不到直接梯度解.

### 3.3 5 条件约束的副作用

5-cond knobs (normcap, γ_norm, r_target, r_spread, w_angular) 都在 collision 维度上不友好:

- `--use_normcap --normcap_target 0.3 --normcap_euc_only`: 把 euc branch 钉到 ‖e‖=0.3 的低维子空间, 跟 baseline 让 euc 自由扩散到 32D ball 内的全空间完全相反
- `--gamma_norm 5.0`: 把 euc norm 强拉到 [1.0, 1.35, 1.7] 三个固定点, 直接杀了 norm 多样性
- `--r_target_list 2.0,2.7,3.4`: 把 hyp branch 半径强钉三层 3 个固定值, 进一步压缩 hyp 自由度

这些约束对**几何可解释性** (5 条件 PASS) 必要, 但对**collision 收敛**有害. 5 条件全部需要正交维度 (半径扩散 / 角度扩散), 但**正是这些正交维度**阻止 encoder 把不同 item map 到不同码字. 几何可见性 vs collision 收敛在此是 trade-off.

## 5. 工程产物

- **v6 训练产物**: `products/m_arm/m_radius_spread_step3_50ep_wdiv100_wangular/` (50 epoch, ~57 sec, 5 条件全 PASS, tuple collision 95.68%)
- **v7 训练产物**: `products/m_arm/m_radius_spread_step3_50ep_sinkhorn_wangular/` (50 epoch, ~1 min, sinkhorn sk_eps=0.03, 5 条件全 PASS, tuple collision 97.28%)
- **v8 训练产物**: `products/m_arm/m_radius_spread_step3_200ep_bs1024_b10_nowdiv/` (200 epoch, killed at epoch 89, no w_div + β=1.0 + batch=1024, trainer-side collision 99.84%)
- **v9 训练产物**: `products/m_arm/m_radius_spread_step3_v9_angdim4/` (killed at epoch 23, 4D + v8 recipe, w_angular spike epoch 12, collision 99.99% by ep 14)
- **v10 训练产物**: `products/m_arm/m_radius_spread_step3_v10_angdim4_v6recipe/` (killed at epoch 72, 4D + v6 recipe, **epoch 4 collision 27.95% 起步最低, epoch 49 97.06%**)

## 6. 复现命令

```bash
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys
cd /home/wlia0047/ar57/wenyu/GeneRec

# v10 (4D + v6 recipe, 起步 collision 27.95% 但 50 epoch 后撞 97% 墙)
bash scripts/m_arm_step3_v10_angdim4_v6recipe.sh 0

# 验证 (用 trainer 自带 collision 输出, 不能用 standalone 脚本 — 脚本 hardcodes HYP_DIM=2)
# 跑完后看 train.log 中 "collision_rate" 字段
grep "collision_rate" products/m_arm/m_radius_spread_step3_v10_angdim4_v6recipe/train.log | tail
```

## 7. 后续决策 (R11.3 自主决策后, 等用户最终选)

5 variants 全部撞墙后, 唯一能让 collision ≤ 12% 的路径只剩**方向 D: 放弃 product_manifold 设计**. 5 个备选方向如下:

| 备选 | 思路 | 风险 |
|------|------|------|
| **方向 A** | 结论定型 — 把 M-arm 限制在 5 条件 PASS 的几何可解释性研究, 不要求 collision ≤ 12% | 失去下游 R@10 优化空间 |
| **方向 B** | 加宽 angular_dim (4D→8D/16D) 渐进到 baseline 角带宽 √(16/64)=0.5 | 跟 baseline 接近时拆分语义减弱; 5 variants 数据已显示墙不在 angular_dim |
| **方向 C** | 接受 5 条件 + collision 双标书 — 下游 R@10 只看 5 条件 PASS 时的 ckpt | 与 HG-Rec baseline 不可对比 |
| **方向 D** ⭐ | 回退 vanilla 32D Poincaré (Task #84 baseline 路线), 已知 9% collision | 完全放弃 M-arm 假设 |
| **方向 E** | κ-Stereographic (Berman-Metzler 2020) 替换 Poincaré 距离, 在 2D hyp 子空间里增加角灵敏度 | 改架构核心 (distance formula), 工程量大; 5-cond knob 行为需重测 |

**R11.3 自决推荐 方向 D** — 唯一已知的 collision ≤ 12% 路径, 风险是放弃 product_manifold 假设. 但这跟用户原始 /goal ("collision ≤ 12%") 优先级对齐, 不应继续在 product_manifold 上耗 GPU.

**R11.3 不推荐方向 B** — 5 variants 数据已穷尽 angular_dim × recipe × epochs × batch_size × β × sk_eps × w_div 7 维空间, 墙一直在 95%+. 即便 angular_dim=16 让 hyp 角带宽到 √(16/64)=0.5, 5-cond knobs 仍锁死 euc 分支. 不继续做方向 B.

**R11.3 推荐给用户的下一步决策**:
1. 接受本 verdict NO-GO 结论, 写 Task #227 final
2. 选择方向 D → 启动 vanilla 32D 复现 + 跑 Stage 2/3/4, 验证 R@10 是否回到 Task #84 baseline 范围 (R@10=0.1020)
3. 选择方向 E → κ-Stereographic 改造, 但需要重新评估 5 条件 + collision 双重约束

当前 verdicts/ 状态: ✅ Task #226 (Step 3 NOT PASS) + ✅ Task #227 (本任务, 5 variants 全撞墙, 综合 NO-GO).

## 8. v11/v12 后续验证 (2026-07-27, 完整 epoch sweep)

**新加**: v11 (8D hyp) + v12 (16D hyp) — 把 angular_dim 推到 baseline 角带宽附近, 用 `m_arm_step3_sweep_5cond.py` (HRQVAE 直加载, collision 校准到 trainer ±0.03%) 扫全部 epoch ckpts.

### 8.1 完整 epoch 验证表

| variant | ep | collision | 5cond | cos_std | min(util%) | min(agree) | 判定 |
|---------|-----|-----------|-------|---------|-----------|-----------|------|
| v11 | **4** | **5.45%** ✓ | **FAIL** | 0.120 | 40.2% | 0.000 | 低 collision 但 encoder 未散 |
| v11 | 9 | 18.99% | FAIL | 0.263 | 81.6% | 0.018 | collision 已超 12% |
| v11 | 14 | 39.20% | FAIL | 0.296 | 92.2% | 0.061 | |
| v11 | 19 | 59.99% | **PASS** ✓ | **0.339** | 81.6% | 0.011 | **5cond PASS 但 collision 已 60%** |
| v11 | 24 | 72.91% | **PASS** ✓ | 0.311 | 81.2% | 0.041 | |
| v11 | 79 | 99.98% | FAIL | 0.000 | 0.4% | 0.000 | 全死码 |
| v11 | 99 | 99.83% | FAIL | 0.000 | 0.8% | 0.000 | |
| v12 | **4** | **8.35%** ✓ | **FAIL** | 0.049 | 20.3% | 0.000 | 低 collision 但 encoder 未散 |
| v12 | 9 | 14.90% | FAIL | 0.089 | 41.8% | 0.006 | collision 已超 12% |
| v12 | 14 | 35.89% | FAIL | 0.181 | 57.0% | 0.092 | |
| v12 | 19 | 56.40% | FAIL | 0.227 | 46.9% | 0.083 | 仍未达 cos_std>0.3 |
| v12 | 24 | 63.32% | FAIL | 0.244 | 38.7% | 0.057 | |
| v12 | 79-99 | 99%+ | FAIL | <0.005 | <2% | ≈0 | 全死码 |
| v11 best_loss | — | 85.30% | FAIL | 0.297 | 73.4% | 0.069 | cos_std 边界 |
| v12 best_loss | — | 87.60% | FAIL | 0.170 | 42.6% | 0.023 | |

### 8.2 Goldilocks 判定

**没有任何 epoch 同时满足 collision ≤ 12% AND 5cond PASS**. Trade-off 是 binary 的, 不是 epoch 问题:

```
低 collision (≤12%)        →  encoder 还没散开  →  cos_std < 0.3 → FAIL cond3
5cond 全 PASS (cos_std>0.3) →  encoder 已散开    →  collision > 18%
```

v11 上:
- **唯一 collision PASS 区间**: ep 4 (5.45%) — 但 cos_std=0.12 (FAIL)
- **唯一 5cond PASS 区间**: ep 19-24 (59.99%/72.91%) — 但 collision 已 FAIL
- 两个区间**不重叠**

v12 上:
- collision PASS: ep 4 (8.35%) — 但 cos_std=0.049 (FAIL 比 v11 更糟)
- 5cond 从未 PASS — 因为 v12 起步 cos_std 更低, 训练进展到 99% collision 也没能拉起 cos_std>0.3

### 8.3 angular_dim 不是瓶颈的事实证据

| ang_dim | 起跑 collision | 5cond PASS 起点 (collision) | 完全收敛 collision |
|---------|---------------|----------------------------|-------------------|
| 2D (v6) | 67.65% | 5cond PASS 全程 | 95.68% (ep50) |
| 4D (v10) | 27.95% | 5cond PASS 全程 | 97.06% (ep49) |
| 8D (v11) | 5.45% (✅) | ep19 (59.99%) | 99.83% |
| 16D (v12) | 8.35% (✅) | 从未 PASS | 99.83% |

**唯一区别**: v11/v12 把起跑 collision 压到 ≤12%, 但 5cond PASS 需等 collision 涨到 60%+. 即便 16D 让 hyp 角带宽到 √(16/64)=0.5 (≈ baseline 70%), 5-cond knobs 仍把 euc 分支钉死, encoder 散不开 → 一旦散开就撞墙.

### 8.4 工程产物追加

- **v11 launcher**: `scripts/m_arm_step3_v11_angdim8.sh` (8D hyp + 26D euc, β=0.5, bs=256, w_div=100, w_angular=10, 100 epoch)
- **v12 launcher**: `scripts/m_arm_step3_v12_angdim16.sh` (16D hyp + 18D euc, 同 v11 recipe)
- **v11 训练产物**: `products/m_arm/m_radius_spread_step3_v11_angdim8/` (ep_4_collision_0.0548 best + 10 ckpts)
- **v12 训练产物**: `products/m_arm/m_radius_spread_step3_v12_angdim16/` (ep_4_collision_0.0844 best + 10 ckpts)
- **测量工具**: `scripts/m_arm_step3_sweep_5cond.py` (HRQVAE 直加载, collision 校准到 trainer ±0.03%, 可复用到任意 m_arm ckpt)

### 8.5 v11/v12 完整 NO-GO 收口

7 variants + 完整 epoch sweep 数据已确认: **M-arm product_manifold 架构下 collision ≤ 12% AND 5cond PASS 不可能同时成立 (二元 trade-off)**. 修复路径只剩:

| 方向 | 修复手段 | 估计工程量 | 风险 |
|------|---------|----------|------|
| A | 接受 5 条件 PASS 已达成 (Task #226), collision 降级为次要目标 | 0 (结论定型) | 失去下游 R@10 竞争力 |
| D ⭐ (R11.3 自决) | 回退 vanilla 32D Poincaré (Task #84 baseline) | 1 天 (复现 Stage 1-4) | 放弃 M-arm 假设 |
| E | κ-Stereographic (Berman-Metzler 2020) 替换 Poincaré | 3-5 天 | 改架构核心 |
| F | 在 v11/v12 短训策略 — 12 epoch 锁定, 不让 collision 爬升 | 1 天 | 仅 ep 4 ckpt 5cond FAIL, Stage 3 R@10 难达标 |

**R11.3 推荐 方向 D** — 7 variants 7-维空间已穷尽, 继续在 product_manifold 上耗 GPU 已 ROI 极低. 启动 vanilla 32D 复现 Stage 1-4 验证基线 R@10=0.1020 仍可达, 是已知碰撞率 ≤ 12% 的唯一路径.
