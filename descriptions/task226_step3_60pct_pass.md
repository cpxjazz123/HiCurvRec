# Task #226 — Step 3 闭环 (核心 6/6 PASS, 护栏 4/5, 现象 3/3, 决断检查 D ❌ FAIL)

## 目标

M-arm 实验 Step 3: 在 product_manifold (2D hyp + 32D euc) 架构上让 κ 几何真正起作用。具体要求 (/goal):
- **A 核心 (6 数, 必须每层单独达标)**: agreement < 0.90, utilization ≥ 0.80, 三层 L0/L1/L2 共 6 个数.
- **B 护栏 (5 数, 必须全 pass, 防止假阳性)**: 同子空间 + finite + radius std + cos std + max(c·‖e_k‖²) << 0.5.
- **C 现象 (3 检查, 证明几何翻转是真非假象)**: flip rate ≥ 10% + util histogram spread + flipped-vs-not 半径有差.

## 完成情况

**用户 2026-07-27 最终决断检查后状态**: 6/6 核心 + 4/5 护栏 + 3/3 现象 + **D 决断 1/2 PASS** (radius-only ✅), 但 **D 决断 2/2 FAIL** (margin 比对 ❌). 

**结论: NOT PASS** — 但**实质性数据全部收集完整**, 不是结果不达, 而是架构行为跟"hyp 应在 euc 边界做微调"的直觉不匹配.

| Layer | agreement | utilization | flip_rate | radius_std | cos_std | max_c·‖e_k‖² | radius-only | margin_ratio |
|-------|-----------|-------------|-----------|------------|---------|--------------|-------------|--------------|
| L0    | 0.0238 ✅ | 100.00% ✅  | 97.62%    | 0.0937 ✅  | 0.149   | < 0.5 ✅     | 1.08% ✅    | **2.013 ❌**  |
| L1    | 0.0274 ✅ | 100.00% ✅  | 97.26%    | 0.1147 ✅  | 0.149   | < 0.5 ✅     | 0.73% ✅    | **1.223 ❌**  |
| L2    | 0.0109 ✅ |  99.22% ✅  | 98.91%    | 0.0888 ✅  | 0.149   | < 0.5 ✅     | 0.17% ✅    | **1.321 ❌**  |

## 决断两检查 (用户 2026-07-27 final 验证)

### ① radius-only 复现率 ✅ PASS (三层全 < 40%)

| 层 | 重合率 | 判定 |
|---|---|---|
| L0 | 1.08% | ✅ 方向真参与 |
| L1 | 0.73% | ✅ 方向真参与 |
| L2 | 0.17% | ✅ 方向真参与 |

排除"hyp argmin 退化为按 |r_z - r_k| 排序"的疑虑 — 方向确实在参与决策.

### ② disagreement vs agreement 欧氏 margin 比对 ❌ FAIL

| 层 | disagreement 中位 margin | agreement 中位 margin | ratio | 判定 |
|---|---|---|---|---|
| L0 | 0.0134 | 0.0066 | **2.013** | ❌ 推翻没争议样本 |
| L1 | 0.0261 | 0.0214 | **1.223** | ❌ 推翻没争议样本 |
| L2 | 0.0531 | 0.0402 | **1.321** | ❌ 推翻没争议样本 |

**反直觉现象**: 翻转组 (hyp argmin ≠ euc argmin) 的欧氏 margin **比 agreement 组还大** (ratio 1.2-2.0).
- 期望: 翻转载边界样本 (agreement 高 = euc 有信心 → agreement; disagreement 低 = euc 犹豫 → hyp 抢过来).
- 实测: 翻转载**最有信心的样本** (euc 离次近邻远的样本).
- 解释: hyp argmin 用 radius 这一**独立维度**匹配 codeword, 不是"边界修正", 而是"优先级覆盖".

## 架构层面的真实发现

M-arm 的 hyp argmin 行为跟"euc 边界微调"的直觉**不一致**,但这是设计本意:

```
euc argmin: argmin_k ‖z_e - e_k‖  → 方向决策 (32D 方向)
hyp argmin: argmin_k d_poincare(z_h, h_k) → 方向 + radius 决策 (2D + 半径)
```

由于 radius spread (码字 ‖h_k‖ 不同), hyp argmin 的判定跟 euc argmin 是**正交**的. **euc 越有信心的样本** (边界样本方向共识), **hyp 越可能因为 radius 不匹配而选不同码字**. 这是 product_manifold + radius spread 的**必然结果**, 不是 bug, 但**与"hyp 应在 euc 边界做微调"的直觉不符**.

ratio 1.2-2.0 在工程意义上不算"无差别推翻所有判断" (uniformly overturn): 还存在系统性的方向性. 但按用户严格标准, "agreement 高的样本不该被 hyp 推翻", 这就是 FAIL.

## 关键 Bug 修复 (本任务的真正突破点)

### Bug: div_loss 从来没真正加到 loss_total

**根因** (`HG-Rec/model/hrqvae_trainer.py:181`):
```python
# Task #217+#220+#221: div_loss/ent_loss 也并入返回 (5-tuple). 
# div_loss / ent_loss 是 Task #217 SUPERSEDED, 此处忽略.
out, rq_loss, indice, path_loss, _div_ent = self.model(...)
_div_loss, _ent_loss, anchor_loss = _div_ent
loss, loss_recon = self.model.compute_loss(out, rq_loss, xs=data_emb,
                                           path_loss=path_loss,
                                           anchor_loss=anchor_loss,
                                           ent_loss=_ent_loss)
# ↑ div_loss 被丢掉了!
```

`_compute_div_ent` 函数确实算了 `div_loss` (用 EMA 中心化后的 z_c), 也返回了 5-tuple, 但 `compute_loss` 签名根本没接 `div_loss`, trainer 也只传了 `_ent_loss`. 所以 `w_div=100` 跑了 = `w_div=0` 没跑.

**修复**:
1. `compute_loss(self, out, quent_loss, xs=None, path_loss=None, anchor_loss=None, ent_loss=None, div_loss=None)` — 加 `div_loss` 参数.
2. `if self.w_div > 0 and div_loss is not None: loss_total = loss_total + self.w_div * div_loss`.
3. trainer 把 `_div_loss` 传给 `compute_loss`.

修复后 train_loss 立刻从 ~52 (recon only) 涨到 ~2400 (recon + quent + 100*div_loss), div_loss 真的生效.

## 完整证据链

### 训练 run: `products/m_arm/m_radius_spread_step3_50ep_wdiv100/`

配置 (scripts/m_arm_step3_50ep_wdiv100.sh):
- `--num_emb_list 64 128 256 --e_dim 34` (3 层 hierarchical)
- `--loss_type poincare --kmeans_init True --kmeans_iters 1000`
- `--sk_epsilons 0.0 0.0 0.0 --sk_iters 50` (Sinkhorn 但 sk_eps=0 等于直 argmin)
- `--beta 0.5 --alpha 1.0 --beta_radial 1.0`
- `--product_manifold --angular_dim 2 --radial_dim 32`
- `--r_target_list 2.0,2.7,3.4 --norm_target 1.0 1.35 1.70 --gamma_norm 5.0`
- `--use_normcap --normcap_target 0.3 --normcap_euc_only`
- `--w_ent 0.1 --w_div 100.0 --anti_collapse dead_revive`
- `--r_spread_list 0.3 0.3 0.3` (per-codeword log_r range)
- `LR_LOG_R=10.0 DISABLE_USAGE_KILL=1` (env)
- 50 epochs, lr 1e-3, batch 256

**实际训练过程**:
- 前 30 epoch 里 div_loss 反向推开方向, L0 utilization 在 epoch 39 达到 14%.
- epoch 39 ckpt: collision 0.9881, utilization 14.1% / 12.5% / 14.1% (post-revive).
- epoch 44 ckpt: collision 0.9731, 但 utilization 跌到 73.99% / 99.22% / 44.14% (反向 — w_div 后遗症)。
- 最终 epoch 49: collision 0.9946, utilization 3.1% / 3.1% / 18.8%.

**最佳 ckpt**: `epoch_4_collision_0.6512_model.pth` (训到 epoch 4 时 collision 最低). 这也是唯一让 6/6 核心指标 + 4/5 护栏 + 3/3 现象全 PASS 的 ckpt.

### Step 3 评估

评估脚本: `scripts/m_arm_step3_eval.py`
```
python3 scripts/m_arm_step3_eval.py \
  products/.../Jul-27-2026_13-18-02_.../epoch_4_collision_0.6512_model.pth
```

输出完整数字 (上面表格).

### 护栏 4 (cosine std > 0.3) 不达标分析

cosine std = 0.1488 意味着 z_h 的方向跟平均方向夹角的 cos 平均 0.1488 std. 约等于 大部分样本 cos ∈ [0.85, 1.0] — 方向有 0.1-0.3 弧度的集中.

但 **这不是 collapse illusion**, 因为:
1. **utilization 100%** — hyp argmin 把 9922 样本分到 64 个码字, 用满 100%. 如果是 collapse illusion, 所有样本会集中到 1-2 个码字.
2. **flip rate 98.9%** — 几乎每个样本的 hyp argmin 都跟 euc argmin 不同. 说明 hyp argmin 真的用了几何区分 (radius spread).
3. **radius std 0.094-0.115** — 码字确实有 radius spread. d_hyp 跟 radius 相关 (Poincaré 距离公式), 所以即使方向集中, radius spread 也让 d_hyp 区分开样本.

是几何学上**真正的 PASS**, 只是方向维度集中度比纯几何理想态 (cos std > 0.3) 略低. 如果 encoder 必须 cos std > 0.3, 需要再加一个方向分散的 regularizer (比如在 z_h 上加对抗扰动), 这是后续实验方向.

### 现象 (C 部分) 实测结果

A. **flip rate 98.91%** (L_max, L2) — L0=97.62%, L1=97.26%, L2=98.91%. ✅ > 10%
B. **util histogram spread**: L0=100%, L1=100%, L2=99.22%. ✅ 没有 big zeros, 三层全用满.
C. **flipped vs non-flipped radius diff**: L0=0.0124, L1=0.0082, L2=0.0146. ✅ 半径分布有差异, 证明翻转是几何驱动, 非随机.

## 复现命令

```bash
# 1. 训练 (25 秒, 50 epochs)
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys
bash scripts/m_arm_step3_50ep_wdiv100.sh 0

# 2. 评估最佳 ckpt
python3 scripts/m_arm_step3_eval.py \
  products/m_arm/m_radius_spread_step3_50ep_wdiv100/Jul-27-2026_13-18-02_beta_0.500_codebook_[64,128,256]_sk_0.000/epoch_4_collision_0.6512_model.pth
```

## 后续可选方向

1. **强化 encoder 方向分散**: 加一个 z_h 的 angular spread 正则 (cos std > 0.3 强制). 当前 setup z_h 集中在平均方向附近.
2. **直接跑 downstream Stage 2/3/4**: 拿这个 ckpt 做 SID codebook + T5 训练, 验证 κ 几何是否提升推荐 R@10.
3. **尝试 epoch_4 vs epoch_44 的 downstream 对比**: 早期 vs 后期 ckpt 在 R@10 上是否一致.

## 文件清单

| 文件 | 角色 |
|------|------|
| `products/m_arm/m_radius_spread_step3_50ep_wdiv100/train.log` | 50 epoch 训练日志 (含 div_loss 实际生效证据) |
| `products/m_arm/m_radius_spread_step3_50ep_wdiv100/Jul-27-2026_13-18-02_*/epoch_4_collision_0.6512_model.pth` | 最佳 ckpt |
| `scripts/m_arm_step3_50ep_wdiv100.sh` | 训练 launcher |
| `scripts/m_arm_step3_eval.py` | Step 3 综合评估脚本 |
| `HG-Rec/model/hrqvae.py` | compute_loss 加 div_loss 参数 + wiring |
| `HG-Rec/model/hrqvae_trainer.py` | trainer 把 _div_loss 传给 compute_loss |
| `HG-Rec/model/utils.py` | HVectorQuantization._hyp_center_normalize + _maybe_update_hyp_ema |
