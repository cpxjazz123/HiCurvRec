# Task #184 — 半径设计 Phase 1a: Product Manifold (球面 × 双曲) 解耦码字

> **任务目的**: 修复 Task #183 v2 暴露的"角向坍缩"问题——把码字显式分解为 (θ ∈ S^(d-1), r ∈ ℝ⁺) 两部分，argmin 同时考察两轴，避免角向 argmin 退化。

> **完成日期**: 待启动
> **状态**: 🟡 待启动（等 Task #181 Stage 3 early stop 后启动）

---

## 1. 背景与动机

**Task #183 v2 暴露的核心问题**:
- 软正则化 `(‖codebook‖ − ρ)²` 成功把码字范数推到 [0.762, 0.873, 0.934]（fill ∈ [0.70, 0.85, 0.93] ✓）
- 但 collision_rate 稳定在 96.25%——96% 的项目坍缩到同一码字
- 根因：argmin 是径向 × 角向 的复合自由度，**只修复径向，角向仍可坍缩**

**几何诊断**:
双曲距离公式 `d_H(x,y)² = ‖x‖² + ‖y‖² − 2·(1+⟨x,y⟩/c)⁻¹·⟨x,y⟩` 在 ‖x‖ ≈ ‖y‖ 时退化为纯角向 argmin：
```
argmin_y d_H(x,y) ≈ argmin_y ⟨x,y⟩  (‖x‖ ≈ ‖y‖)
```
即码字范数被控到 ρ 后，argmin 只剩角向竞争——若没有显式角向约束，k-means init 后码字倾向于聚集到同一方向。

**Product manifold 解耦思路** (Phase 1a):
- 把码字显式分解为 (θ ∈ S^(d-1), r ∈ ℝ⁺)
- θ: 球面角向（保证 anti-podal 区分）
- r: 双曲径向（沿用 Task #183 v2 软正则化）
- argmin 距离公式: `d(x, codeword) = α · d_S(θ_x, θ_codeword) + β · |r_x − r_codeword|`
- α, β 平衡两轴贡献；α > 0 强制球面区分 → 角向坍缩不可能

---

## 2. 实验设计

**变量**: RQ-VAE 码字 = product (θ ∈ S^(d-1), r ∈ ℝ⁺)，α=1.0, β=1.0

**保持不变**:
- 数据集: Musical_Instruments (9922 items)
- RQ-VAE 架构: num_emb_list=[64,128,256], e_dim=32, M=3
- seed=42, sk_eps=[0,0,0]
- 超参: β=1.0, batch=1024, lr=1e-3, warmup=20, epoch=1000
- ρ_ℓ = [1.0, 1.345, 1.69] (同 Task #183 v2)
- rho_reg_weight=0.1 (同 Task #183 v2，约束 r 部分)
- Stage 3: T5-small 6enc+4dec d=128 (5.5M)
- Stage 4: 同 task84 eval protocol

**新增超参**:
- `--angular_dim 16`: θ 部分维度（占 e_dim=32 一半）
- `--radial_dim 16`: r 部分维度（r ∈ ℝ^16，不是单标量）
- `--alpha 1.0`: 球面距离权重
- `--beta_radial 1.0`: 径向距离权重
- 距离公式: `d²(x, codeword) = α · (1 − cos(θ_x, θ_codeword)) + β_radial · MSE(r_x, r_codeword)`
- kmeans_init: 角度做球面 k-means（用余弦相似度）+ 径向做欧氏 k-means

**启动命令模板**:
```bash
python3 train_hrqvae.py \
    --data_path ./dataset/Instruments/item_emb.parquet \
    --num_emb_list 64 128 256 \
    --e_dim 32 \
    --layers 512 256 128 64 \
    --beta 1.0 \
    --loss_type poincare \
    --sk_epsilons 0.0 0.0 0.0 \
    --epochs 1000 \
    --batch_size 1024 \
    --lr 1e-3 \
    --learner AdamW \
    --weight_decay 0 \
    --warmup_epochs 20 \
    --lr_scheduler_type linear \
    --num_workers 4 \
    --kmeans_init True \
    --kmeans_iters 1000 \
    --product_manifold True \
    --angular_dim 16 \
    --radial_dim 16 \
    --alpha 1.0 \
    --beta_radial 1.0 \
    --radii 1.0 1.345 1.69 \
    --rho_reg_weight 0.1 \
    --ckpt_dir /home/wlia0047/ar57/wenyu/GeneRec/products/task184/hrqvae_product_manifold \
    --device cuda:0
```

### 决策触发 (vs HG-Rec baseline R@10=0.1020)

| Stage 1 指标 | Stage 3 R@10 | 决策 |
|-------------|--------------|------|
| angular_util > 80% + radial_util > 80% + collision ≤ 15% (final) | ≥ 0.1020 | ✅ Product manifold 解耦成功 |
| angular_util > 90% | ≥ 0.1315 | ⭐ 球面码字利用率极高，超预期 |
| angular_util < 30% (球面坍缩) 或 collision > 50% (final) | 任意 | 🔴 Product 设计有问题 |

---

## 3. 修改文件清单

| 文件 | 修改 |
|------|------|
| `HG-Rec/model/utils.py` | HVectorQuantization: 新增 product_manifold 模式；codebook 拆分为 angular_codebook (K, angular_dim) + radial_codebook (K, radial_dim)；forward 用距离公式 `α·(1−cos) + β_radial·MSE`；get_codebook 重组回 d_dim |
| `HG-Rec/model/hrqvae.py` | HRQVAE.__init__: 加 product_manifold / angular_dim / radial_dim / alpha / beta_radial 参数 |
| `HG-Rec/train_hrqvae.py` | 加 --product_manifold / --angular_dim / --radial_dim / --alpha / --beta_radial CLI 参数 |

---

## 4. 预算

| 阶段 | 估算时间 |
|------|---------|
| Phase 1a 实施 + py_compile | ~30 min |
| 50-epoch 验证（无坍缩） | ~5 min |
| Stage 1 1000 epoch | ~30 min |
| Stage 2 Codebook inference | ~5 min |
| Stage 3 T5-small 200 epoch | ~1.5 h |
| Stage 4 Eval | ~5 min |
| 总计 | ~2.5 h |

---

## 5. 风险与缓解

**风险 1**: 球面 k-means init 不稳定 → 用 spherical k-means (sklearn or 自实现)
**风险 2**: α/β_radial 不平衡（角向坍缩或径向坍缩）→ 监控 angular_util / radial_util 分别
**风险 3**: codebook 重组回 d_dim 时几何不连续 → 用 `x_θ * ‖r‖` 直接乘积（避免 Cartesian→Poincaré 重投影）
**风险 4**: 50 epoch 后 collision 仍 > 50% → pivot 到 Phase 1b (orthogonal product space, θ ⊥ r 严格正交)

---

## 6. 完成度跟踪

- [ ] Phase 1a 实施（修改 utils.py + hrqvae.py + train_hrqvae.py）
- [ ] py_compile 三文件验证
- [ ] 50-epoch 验证：angular_util > 80% + radial_util > 80%
- [ ] Stage 1 1000 epoch 训练（GPU 0，PID 待分配）
- [ ] Stage 1 完成 + 最终 collision ≤ 15%
- [ ] Stage 2 codebook inference
- [ ] Stage 3 T5-small 训练
- [ ] Stage 4 Eval (R@10 vs 0.1020 baseline)
- [ ] angular_util / radial_util 监控指标验证
- [ ] Verdict + §16 cleanup

---

## 7. 关键决策点（R11.3 自主决策明示）

### 决策 1: 维度分配 (angular_dim + radial_dim = e_dim)
**选了**: angular_dim=16, radial_dim=16（各占一半）
**为什么**: e_dim=32 时 16+16 平分；angular 部分需足够表达 d-1 维球面，radial 部分单标量不够灵活
**备选**: angular_dim=24, radial_dim=8（偏向角向）；angular_dim=8, radial_dim=24（偏向径向）
**回退方案**: 如果 16+16 不够，加 α/β_radial 平衡

### 决策 2: 距离公式权重
**选了**: α=1.0, β_radial=1.0（等权）
**为什么**: Phase 1a 主要验证解耦是否有效，等权 baseline 最直观；如有需要可调
**备选**: α=2.0, β_radial=1.0（强化球面区分）

### 决策 3: codebook 重组方式
**选了**: x = ‖r‖ · θ̂（保留 r 的范数 + θ 的方向）
**为什么**: 几何连续，无 Cartesian→Poincaré 重投影误差；自动保留 r 的 L2 范数
**备选**: x = r_vec（直接把 r 当向量，不取范数）——失去 ‖r‖ 几何含义

---

## 8. 修订记录

| 日期 | 修订内容 |
|------|---------|
| 2026-07-25 | 首次创建（基于 Task #183 v2 NO-GO verdict） |
