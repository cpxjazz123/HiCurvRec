# Task #420 / Issue #127 [方向A Gate1] Poincaré-ball 域投影 + 稳定 acosh 距离

## R18 v2 4 维度路径对比 (vs Issue #124)

| 维度 | Issue #124 (task417, NO-GO 收口) | Issue #127 (本 task, 修复方向A) |
|------|----------------------------------|---------------------------------|
| **D1 spec 摘录** | hard argmin + soft posterior (per #122 Bridge PreFailure) | 同样 hard argmin + soft posterior + **每层独立 ball projection + 稳定 acosh** |
| **D2 实施核心** | HG-Rec `poincare_pairwise` + mobius_add + artanh (数值不稳) | **每层 projector** ‖x‖<(1-eps)/√c_l + **acosh form** + domain assertion + **κ 更新后 radial rescale** |
| **D3 Gate 1 失败机制** | c≈0.793 + norm≈1 → artanh 越界 NaN | **同根因, 但 #127 在源头修复 ball projection 防止 norm→1** |
| **D4 引用文献** | Berman-Metzler 2020 κ-Stereographic | **同文献 + Nickel-Kiela 2017 Poincaré-ball ball projection** |

**R18 v2 判定**: 4 维度都有差异 (重点在 D2 实施核心 + D3 修复策略), 必须做新实验, 不允许套用 #124 判决.

## 实施
- `scripts/task420_issue127_ball_projection.py` (~390 lines)
- 5-step audit + 30 epoch main (有 projection+acosh 稳定化) + control (#124 形式)
- K=[64,128,256], β=0.25, lr=1e-4, seed=42
- 产物: `products/task420_issue127_ball_projection/verdict.json`

## Gate 1 决策阈值
- Main config: 每层 util ≥ 0.9, max_load < 0.05, 5-step audit 全 PASS
- Control config: 任何 audit FAIL 立即 STOP

## 复现命令
```bash
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/genrec_env
python3 scripts/task420_issue127_ball_projection.py
```