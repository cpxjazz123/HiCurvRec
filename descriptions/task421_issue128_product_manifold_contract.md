# Task #421 / Issue #128 [方向B Gate1] Product-space 三分量 Poincaré 域合同

## R18 v2 4 维度路径对比 (vs Issue #125)

| 维度 | Issue #125 (task418, NO-GO 收口) | Issue #128 (本 task, 修复方向B) |
|------|----------------------------------|---------------------------------|
| **D1 spec 摘录** | hard argmin + per-codeword α_l,k + 3 分量 (learn κ hyp / fixed κ hyp / Euclidean) | **同架构** + 每分量独立 ball projection + **d_mix 前域断言** |
| **D2 实施核心** | HG-Rec `poincare_pairwise` (数值不稳, 3 分量叠加放大 NaN) | **每 hyperbolic 分量独立 project to ball** + **d_mix 前 assert_in_ball** + α 自身 finiteness assertion |
| **D3 Gate 1 失败机制** | 3 分量叠加 → 3× NaN 风险放大 | **每分量独立 clamp 边界 + 域断言, 物理隔离 NaN 传播** |
| **D4 引用文献** | α-blending decoder (Gao et al 2020 略) | 同文献 + Nickel-Kiela 2017 ball projection |

**R18 v2 判定**: 4 维度都有差异 (重点在 D2 实施核心 + D3 修复策略), 必须做新实验, 不允许套用 #125 判决.

## 实施
- `scripts/task421_issue128_product_manifold_contract.py` (~440 lines)
- 5-step product audit (域断言嵌入) + 30 epoch main (有 projection+assertion) + control (#125 unstable 形式)
- 3 分量 d_mix = Σ_j α_l,k,j · d_l,j(x, c_k)
- K=[64,128,256], β=0.25, lr=1e-4, seed=42
- 产物: `products/task421_issue128_product_manifold_contract/verdict.json`

## Gate 1 决策阈值
- Main: 每层 util ≥ 0.9, max_load < 0.05, 5-step product audit 全 PASS (含域断言)
- Control: 任何 audit FAIL 立即 STOP

## 复现命令
```bash
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/genrec_env
CUDA_VISIBLE_DEVICES=1 python3 scripts/task421_issue128_product_manifold_contract.py
```