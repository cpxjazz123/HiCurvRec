# Task #248 — Issue #13 Gate 0: 残差算子几何一致性核查 (PASS)

## 1. 目的

执行 Issue #13 §阶段闸门 Gate 0 (零 GPU, 零训练, 分钟级代码核查):
- 定位 embedding network 的几何
- 定位 layer 间残差递推算子 (欧式 or Möbius)
- 定位 distance metric 的几何
- 输出三行表, 通过条件 = 残差算子确为欧式减法

## 2. 三行表 (核心结果)

| 项 | 几何 | 源位置 (HG-Rec/model/) | 关键代码 |
|---|------|------------------------|----------|
| **embedding network** | **欧式** | `hrqvae.py:459` (HRQVAE.forward) | `x_lat = self.encoder(x)` — encoder 是纯 MLP (`encode_layer_dims` linear stack), 输出直接当 latent, **无 expmap0 / proj_to_ball**. NormCap 后处理 (`hrqvae.py:463-475`) 仍是欧式范数缩放. |
| **residual computation** | **欧式减法** | `utils.py:1797` (`HResidualVectorQuantization.forward`) + `hrqvae.py:550` (update_c_k_from_spread) | `residual = residual - x_res` (utils.py:1797), `residual = residual - x_res` (hrqvae.py:550). 两处都是欧式减法, 无 `mobius_add(-x, y, c)` / Möbius 加法调用. |
| **distance metric** | **混合**: hyp 维度用 `poincare_distance**2`, euc 维度用 MSE | `utils.py:961-1018` (`_product_manifold_distance`) | `hyp_dist = poincare_distance(x_exp, cb_exp, self.c).squeeze(-1) ** 2` (utils.py:1009); `euc_dist = ((euc_exp - cb_euc_exp) ** 2).mean(dim=-1)` (utils.py:1014). 总距离 `alpha * hyp_dist + beta_radial * euc_dist` (utils.py:1017). 当 `product_manifold=False` 时退化为纯欧式 `euc_qloss` 路径 (`utils.py:1247-1268`). |

## 3. 关键代码引用

### 3.1 Residual computation (欧式减法)

```python
# HG-Rec/model/utils.py:1789-1798 (HResidualVectorQuantization.forward)
residual = x
for li, quantizer in enumerate(self.vq_layers):
    if isinstance(quantizer, HVectorQuantization):
        quantizer.layer_idx = 0
    x_res, loss, indices = quantizer(residual, use_sk=use_sk)
    residual = residual - x_res  # ← 欧式减法 (无 Möbius 加法)
    x_q = x_q + x_res
    ...
```

```python
# HG-Rec/model/hrqvae.py:545-550 (HRQVAE.update_c_k_from_spread, no-Sinkhorn path)
residual = x_lat
for li, vq in enumerate(self.hrq.vq_layers):
    x_res, _, indices = vq(residual, use_sk=False)
    latents_per_layer[li].append(residual.detach())
    indices_per_layer[li].append(indices.detach())
    residual = residual - x_res  # ← 同样欧式减法
```

### 3.2 Distance metric (混合)

```python
# HG-Rec/model/utils.py:961-1018 (HVectorQuantization._product_manifold_distance)
# Hyp part: 硬归一化到 tangent space + expmap0 + poincare_distance²
latent_hyp_h = proj_to_ball(expmap0(latent_hyp_n, self.c), self.c)
cb_hyp_h = proj_to_ball(expmap0(cb_hyp_n, self.c), self.c)
hyp_dist = poincare_distance(x_exp, cb_exp, self.c).squeeze(-1) ** 2  # (B, K)
# Euc part: 原始欧氏 MSE
euc_dist = ((euc_exp - cb_euc_exp) ** 2).mean(dim=-1)  # (B, K)
combined = self.alpha * hyp_dist + self.beta_radial * euc_dist  # 等权 alpha=1.0, beta_radial=1.0
```

### 3.3 Embedding (欧式 MLP)

```python
# HG-Rec/model/hrqvae.py:316, 459
self.encoder = MLP(layers=self.encode_layer_dims, dropout=..., use_bn=...)  # 纯 MLP, 无几何
...
def forward(self, x, ...):
    x_lat = self.encoder(x)  # ← 直接 MLP 输出, 无 expmap0/proj_to_ball
```

## 4. Gate 0 决策

**H1 成立**: HG-Rec 的 `HResidualVectorQuantization` 层间残差用欧式减法计算 (`utils.py:1797`, `hrqvae.py:550`). 

embedding / residual / distance 三处几何状态:
- ✅ embedding: **欧式** (MLP 输出无几何投影)
- ❌ residual: **欧式减法** (H1 成立 — 跟 HRQ 主文献主张不一致)
- ⚠️ distance: **混合** (hyp 维度双曲 + euc 维度欧式, **不是纯双曲**)

**关键解释**: 三处**不一致**, 但更精确的描述是:
- "只双曲化了 hyp 距离度量 + 部分维度" (assignment 判据在 hyp 子空间上, 但 euc 子空间仍欧式)
- "未双曲化残差算子" (residual 流仍是欧式)
- "未双曲化 embedding 网络" (encoder 输出仍是欧式, 后处理 NormCap 是欧式范数缩放)

## 5. Gate 0 通过, 进入 Gate 1

按 Issue #13 §阶段闸门, Gate 0 通过条件 = "残差算子确为欧式减法" ✅. 进入 Gate 1 (零 GPU 纯前向).

Gate 1 设计 (R11.3 自主决策):
- 载入 `products/task222/hrqvae_pck_replay/.../epoch_29_collision_0.3706_model.pth` (task222 ep29 healthy ckpt)
- 逐层比较: 欧式残差 `z - e_k` vs Möbius 残差 `z ⊖_{c_l} e_k` 范数分布 + 下一层 argmin 一致率
- 通过条件: 至少一层 argmin 一致率 ∈ [60%, 90%] OPEN 带
- 硬停止: 三层一致率均 > 95% → STOP + 写 verdict "残差算子不是杠杆"

资源预算: ~10 min wall (载入 ckpt + 前向一次), 零 GPU.

## 6. 反 H3 先验诚实声明

Issue #13 §假设 已声明 H3 先验偏弱的两条理由:
1. **领域不同**: HRQ +20% 来自 WordNet hypernym 树的 hierarchy modeling, 任务是层级建模而非推荐召回
2. **本仓库悲观先验**: `verdicts/task225_pck_stage4_eval_result.md` §4 "几何激活跟下游 Recall 解耦"

Gate 0 通过仅确认**残差算子是欧式**这一结构性事实, 不预测 H3 方向是否能落地. 后续 Gate 1/2/3 才是验证窗口.

## 7. 决策点 (R11.3 自主决策)

| 决策 | 选了什么 | 为什么 |
|------|---------|------|
| 三行表粒度 | embedding / residual / distance 三处 | Issue #13 §Gate 0 明确要求 |
| residual 检查覆盖 | `utils.py:1797` 主路径 + `hrqvae.py:550` no-Sinkhorn update 路径 | 两处都查到, 完整覆盖 |
| distance 检查 | 混合描述 (hyp + euc), 不简化 | 实际 `_product_manifold_distance` 公式就是混合, 简化会失真 |
| 是否启动 Gate 1 | 是, 立即 | Gate 0 通过 = 进入 Gate 1 (Issue #13 阶段闸门明文) |
| Gate 1 ckpt 选择 | task222 ep29 healthy ckpt (L0 20.31% / L1 98.44% / L2 91.02%) | Issue #13 §Gate 1 明文指定 |
| 是否发 Issue #13 GitHub 评论 | 是 | 让 issue 状态对外可见 (Gate 0 PASS + 等 Gate 1) |

## 8. 产物

- `descriptions/task248_issue13_gate0_residual_operator.md`
- `verdicts/task248_issue13_gate0_residual_operator_result.md` (本文件)
- Issue #13 GitHub 评论 (待发, Gate 0 PASS 标记)

## 9. 状态

✅ **Gate 0 PASS**: H1 成立, 残差算子确为欧式减法. 三处几何不一致明确记录. 立即进入 Gate 1.

result: **Issue #13 Gate 0 PASS. 三行表已锁定: embedding=欧式 (MLP), residual=欧式减法 (H1 成立, utils.py:1797 + hrqvae.py:550), distance=混合 (hyp 子空间双曲 + euc 子空间欧式). 结构性判定: 本项目"只双曲化了 hyp 距离度量, 没双曲化残差算子和 embedding 网络". Gate 0 通过, 立即进入 Gate 1 (零 GPU 前向检查)**.