# Task #447 / Issue #156 [方向B Gate1] 无饱和截断单纯形权重与双边配额复核

## R18 4 维度路径对比 (vs Issue #154 / Task #445 MECHANISM PASS + clamp saturation)

| 维度 | Issue #154 (task445 MECHANISM PASS, clamp saturation) | Issue #156 (本 task, 升级修复方向B) |
|------|--------------------------------------------------------|--------------------------------------|
| **D1 spec 摘录** | TruncatedSimplexKappaModel + hard clamp([w_min, w_max]) + renormalize → weight 在 [0.1, 0.8] 边界 saturation, weight_mlp grad=0 | **无饱和仿射截断 softmax**: `alpha_i = 0.1 + 0.7 * softmax(logits)_i`. 每项严格 [0.1, 0.8], 对 logits 可导, 无 clamp. |
| **D2 实施核心** | weight = clamp(softmax(logits), 0.1, 0.8) + renormalize | weight = 0.1 + 0.7 * softmax(logits) (仿射映射, [0, 1] → [0.1, 0.8] 严格). 同样和 = 0.3 + 0.7 = 1.0 ✓ |
| **D3 Gate 1 失败机制** | hard clamp 在边界切断 grad path (类 ReLU), weight_mlp grad=0 | 新机制: 仿射截断是 linear transformation, 完全保 grad path. softmax 输出 [0, 1] 严格内部 (除了 logits 极端时趋近 0/1 但仍 grad ≠ 0). |
| **D4 引用文献** | arXiv:2307.04514 | arXiv:2307.04514 (同文献, 不同实施: 仿射截断 vs hard clamp) |

**R18 判定**: 4 维度都有差异 (重点在 D1 仿射截断 vs hard clamp + D2 linear transform vs clamp + D3 grad 保 path vs grad 切断 + D4 同文献不同实施), 必须做新实验, **不允许**套用 #144/#146/#149/#152/#154 判决 (R18 强制).

## 仿射截断 softmax 数学证明 (替代 hard clamp)
- 给定 logits ∈ R^K (无界)
- softmax(logits)_i ∈ (0, 1), sum = 1
- alpha_i = 0.1 + 0.7 * softmax(logits)_i ∈ (0.1, 0.8) (严格, 因为 softmax 输出严格 (0, 1))
- sum_i alpha_i = K * 0.1 + 0.7 * sum_i softmax_i = K*0.1 + 0.7 = 0.3 + 0.7 = 1.0 ✓ (K=3 情况)
- grad path: ∂alpha_i/∂logits_j = 0.7 * ∂softmax_i/∂logits_j = 0.7 * (δ_ij * softmax_i - softmax_i * softmax_j) → 全程可导
- 跟 hard clamp 对比: hard clamp 在 w=0.1 或 w=0.8 边界切断 grad (类 ReLU at 0)

## 实施
- `scripts/task447_issue156_affine_truncated_softmax.py` (fork from task445)
- AffineTruncatedSimplexKappaModel: per-sample MLP → 3 logits → softmax → **仿射截断** (0.1 + 0.7 * softmax)
- 保留 #154 的: 3 层 κ + eucl codebook + product d_mix + bilateral Hungarian cost
- 替换 hard clamp 为仿射截断
- 同时记录每层 κ, mixing logits, alpha 及其梯度 + step 前后 delta
- 至少 10 个记录点
- 双边配额 (lower=1 + upper=ceil(B/K)) 不变
- SHA256(item_emb) = 1a42341f01537d6d...
- 8 件套: config + SHA256 + precheck + affine_truncated_proof + train_curve + ckpt + verdict.json + commit

## Precheck 决策阈值 (Issue #156 spec)
- 证明 L0 K64/L1 K128/L2 K256 独立可学习 κ 路径仍为主路径
- 产品分量只为扩展, 禁止 global κ, fixed-only, 纯欧氏绕过, 独立模型
- 仿射截断数学验证: alpha ∈ (0.1, 0.8) 严格, sum = 1, grad path 完好

## Gate 1 决策阈值 (Issue #156 spec)
- PASS: 至少 10 个记录点; 每层 κ grad + 每个 mixing-logit grad 均有限非零; 对应参数 delta 非零; alpha 逐点和=1, 每项 [0.1, 0.8] 严格 (无 clamp); 报告熵和各分量贡献, 不能退化为单一分量; 每层 util=100%, max_load<5%, min_load>=1; 无 NaN/Inf; 8 件套齐全
- FAIL: 任一不满足即 STOP

## 复现命令
```bash
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/genrec_env
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task447 \
  python3 scripts/task447_issue156_affine_truncated_softmax.py
```

## 跟 Issue #154 关键差异
- Issue #154: weight = clamp(softmax, 0.1, 0.8) + renormalize → boundary saturation, grad=0
- Issue #156: weight = 0.1 + 0.7 * softmax (仿射截断) → 严格 (0.1, 0.8) 内部, grad path 完好
- 数学等价性: 两种方案都满足 [0.1, 0.8] 和=1, 但 grad path 完全不同 (hard clamp 切断, 仿射保留)