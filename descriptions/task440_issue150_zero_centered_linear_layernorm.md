# Task #440 / Issue #150 [方向C Gate3] 零中心线性几何残差与输入LayerNorm联合适配

## R18 4 维度路径对比 (vs Issue #147 / Task #437)

| 维度 | Issue #147 (Task #437, NO-GO Gate 3) | Issue #150 (本 task, 修复方向C) |
|------|--------------------------------------|------------------------------------|
| **D1 spec 摘录** | MLP-sigmoid adapter (sigmoid 乘法), 全 T5 冻结, 13 adapter params (commit `1a3c623`): Gate 0 max_diff=0 PASS, 10 epoch 后 sigmoid scale 饱和到零, grad 衰减至 1e-9, T5 main loss 卡死 1.886-1.889, NO-GO Gate 3 | **zero-centered bounded-linear residual** (NO sigmoid 乘法饱和), 仅解冻 T5 **input LayerNorm** scale/bias (其他 T5 权重冻结), 初始化必须严格 identity (残差系数 0), linear 系数 + LayerNorm 首步后获稳定梯度 |
| **D2 实施核心** | `scripts/task437_issue147_curvature_conditioned_residual.py`: Adapter = Sequential(Linear, Sigmoid, Linear) residual scale 0.3, full T5 freeze, 13 params | `scripts/task440_issue150_zero_centered_linear_layernorm.py` (本任务, fork from task437): ZeroCenteredLinearResidual = Linear(in→out) residual **WITHOUT sigmoid** (NO 饱和), 仅 T5 input LayerNorm 解冻 (~512 params × 2 = scale + bias, d_model=128), 其他 T5 冻结. 残差参数化: x_out = x + α·conditioner(x) (α=0 init, linear 系数 init), conditioner 仍从 #147 复用 (geometric features) |
| **D3 Gate 3 失败机制** | sigmoid 乘 scale → 饱和到 0/1 → grad 全程过 sigmoid 后被 saturate → adapter grad 衰减至 1e-9 (commit `1a3c623`) | **新机制**: linear (NO sigmoid) → 残差系数=0 时输出严格 identity → linear 系数有直接梯度 (无饱和截断). 仅解冻 input LayerNorm 引入 ~1K params 让 T5 不冻结全部 (验证 "冻结主干" 是否根因). 预期: 10 epoch 内梯度不连续 5 epoch 低于预注册阈值, loss 相对 epoch 0 下降 |
| **D4 引用文献** | 内部 spec (no specific arXiv cited) | arXiv:2309.04082 (mixed/product-stereographic 几何可进入 Transformer 表示路径) — nature-academic-search 检索 arXiv/CrossRef/PubMed 核验 |

**R18 判定**: 4 维度都有差异 (重点在 D1 zero-centered bounded-linear vs MLP-sigmoid 饱和 + D2 仅解冻 input LayerNorm vs 全 T5 冻结 + D3 实证 linear 残差是否有梯度 + D4 arXiv:2309.04082 vs 内部 spec), 必须做新实验, **不允许**套用 #147 判决 (R18 强制).

## 实施
- `scripts/task440_issue150_zero_centered_linear_layernorm.py` (~700 lines, fork from `scripts/task437_issue147_curvature_conditioned_residual.py`)
- Stage 3 框架 (跟 #147 同): Task #84 stage 3 训练协议, musical_instruments, seed=42, T5-mini (num_layers=6 enc, 4 dec, d_model=128, d_ff=1024)
- **改 1 — Zero-Centered Bounded-Linear Residual**:
  - 取消 sigmoid 乘法, 改用 `x_out = x + α · conditioner(x)`, α=0 init (残差起点)
  - linear 系数 init = small (Kaiming normal, fan_in)
  - α 用 bounded linear (NOT sigmoid), 仅约束 α ≥ 0 (NOT 强制 ≤ 1)
- **改 2 — Input LayerNorm 解冻**:
  - 仅解冻 T5 第一层 `layer[0].layer_norm0.weight + bias` (T5 has input layer norm per layer)
  - 总可训练参数: 13 (conditioner linear) + 128 (input LayerNorm scale) + 128 (input LayerNorm bias) = ~269 params
  - 其他 T5 权重全部冻结 (~9.18M params)
- **改 3 — Gate 0 PASS 检查**:
  - 残差系数=0 时输出与原 T5 max_diff=0 (strict identity)
  - 首个优化 step 后: conditioner linear + LayerNorm scale/bias 均有 finite nonzero gradient + delta
  - 仅 conditioner + LayerNorm 更新 (其他 T5 不动)
- **改 4 — R12 强制 ckpt 落盘**: training loop 结束 → evaluation 开始之间强制 `torch.save` (即使后续 eval 失败也能保住产物), 删旧 ckpt
- 8 件套: config + SHA256(item_emb) + ckpt(SHA256) + train_curve (10 epoch) + precheck + LayerNorm_unfreeze_proof + verdict.json + commit

## Precheck 决策阈值 (Issue #150 spec 强制)
- **残差零系数时输出 identity**: max_diff = 0
- **input LayerNorm 解冻**: 仅 `layer[0].layer_norm0.weight/bias` 改 requires_grad=True, 其他 T5 requires_grad=False
- **conditioner 仅 13 linear params**: 跟 #147 一致
- **首步后三组参数均获梯度**: conditioner + linear α + LayerNorm scale + LayerNorm bias 全部 finite nonzero
- **严格 forbidden**: 替换 SID, proxy SID[:3], 改 evaluator, pure-Euclidean bypass, 独立 T5, 解冻完整 T5

## Gate 3 决策阈值 (Issue #150 spec 强制)
- PASS: 10 epoch 内**不连续 5 个 epoch**梯度 < 预注册阈值, loss 相对 epoch 0 **有可审计下降**, save/load missing=0/unexpected=0, forward 一致, 真实 history-SID 可加载, 无 NaN/Inf
- FAIL: 任一不满足即 STOP, 禁止进入 Stage 4

## Gate 4 决策阈值 (Gate 3 PASS 后)
- 双复跑 + 报告六项指标 (R@5/10/20, NDCG@5/10/20)
- 仅 test R@10 > 0.1020 (HG-Rec baseline) = Target reached

## 复现命令
```bash
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/genrec_env
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task440 \
  python3 scripts/task440_issue150_zero_centered_linear_layernorm.py
```