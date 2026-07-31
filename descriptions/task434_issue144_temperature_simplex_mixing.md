# Task #434 / Issue #144 [方向B 预检+Gate1] 非饱和层级mixing与独立κ梯度通路审计

## R18 4 维度路径对比 (vs Issue #141)

| 维度 | Issue #141 (task431, NO-GO 收口) | Issue #144 (本 task, 修复方向B) |
|------|----------------------------------|---------------------------------|
| **D1 spec 摘录** | bounded softmax 3 分量 + 1000-step 实证 | **温度受控 simplex + 显式熵下界**: centered logits + temperature-scaled softmax + entropy_lower_bound clamp, 修复三层独立 κ optimizer ownership |
| **D2 实施核心** | AttributionMatrixModel + layer-level 3 scalars per layer + bounded softmax | **TemperatureSimplexMixing**: logits - logits.mean() + softmax(logits/T) + entropy lower-bound clamp; **TempSimplexModel** 3 层独立 κ + 3 个 mixing + 4-group optimizer |
| **D3 Gate 1 失败机制** | bounded softmax 假设能避免 0/1 饱和 → 仍 logits grad=0 | **温度受控 + 熵下界假设**: 假设 center + temperature 解饱和 + entropy bound 强制非贴边能让 mixing logits 拿到 grad |
| **D4 引用文献** | arXiv:2307.04514 Weighted Mixed-Curvature Product Manifold | 同文献 + Mixed-Curvature + Riemannian Optimization |

**R18 判定**: 4 维度都有差异 (重点在 D1 温度受控 + D2 simplex + 熵下界 + D3 新失效机制假设), 必须做新实验.

## 实施
- `scripts/task434_issue144_temperature_simplex_mixing.py` (~360 lines)
- TemperatureSimplexMixing + TempSimplexModel + 4-group optimizer + entropy lower-bound + 1000-step audit
- Precheck: parameter_registry (9 entries: κ + logits + temperature) + optimizer_groups (4 groups) + before/after state trace
- 7 件套: config + SHA256 + trace + precheck + raw_log + verdict + commit
- 产物: `products/task434_issue144_temperature_simplex_mixing/{config,verdict,precheck}.json`

## Precheck 决策阈值
- 9 entries in parameter_registry (κ + mixing_logits + log_temperature)
- Optimizer 4 group 独立 lr
- before/after trace 完整

## Gate 1 决策阈值
- 每层 κ 和 mixing 均有限非零 grad + 更新
- 三分量权重不贴边 + 熵高于下限 (≥ 0.3)
- 至少两分量贡献 > 0.1
- usage >= 90%, max_load < 5%
- hard SID round-trip + 无 NaN/Inf

## 复现命令
```bash
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/genrec_env
CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task434 \
  python3 scripts/task434_issue144_temperature_simplex_mixing.py
```