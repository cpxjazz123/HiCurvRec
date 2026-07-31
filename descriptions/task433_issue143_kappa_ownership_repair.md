# Task #433 / Issue #143 [方向A 预检+Gate1] 修复独立κ冻结并审计曲率同步更新

## R18 4 维度路径对比 (vs Issue #140)

| 维度 | Issue #140 (task430, NO-GO 收口) | Issue #143 (本 task, 修复方向A) |
|------|----------------------------------|---------------------------------|
| **D1 spec 摘录** | hard-EMA 更新顺序隔离 + 1000-step 实证 | **预检强制 + 独立 κ 所有权修复**: 参数注册表 + optimizer param-group + update-order trace, 不允许 detach/常量覆盖/κ 漏出 optimizer |
| **D2 实施核心** | HardEMAModel with detached geodesic EMA + #47 scale sync | **RepairedKappaModel**: 3 独立 `kappa_l_raw_0/1/2` 各自 nn.Parameter + 独立 `codebook_0/1/2` + 3-group optimizer (kappa lr=1e-3, codebook lr=1e-4, other lr=1e-4) |
| **D3 Gate 1 失败机制** | 隔离更新顺序假设"encoder 不偷渡到 codebook" → 100% single-codeword collapse | **修 κ 所有权假设**: 假设修参数独立 + optimizer group 独立 lr 能恢复 κ 更新 |
| **D4 引用文献** | arXiv:2405.13979 Robust Hyperbolic Learning | 同文献 + curvature-aware optimizer + fine-tunable hyperbolic scaling |

**R18 判定**: 4 维度都有差异 (重点在 D1 预检强制 + D2 独立 κ 修复 + D3 新机制假设), 必须做新实验, 不允许套用 #140 判决.

## 实施
- `scripts/task433_issue143_kappa_ownership_repair.py` (~270 lines)
- RepairedKappaModel + 3-group optimizer + #47 sync trace + 1000-step audit
- Precheck: parameter_registry + optimizer_groups + before/after state
- 7 件套: config + SHA256 + trace + precheck + raw_log + verdict + commit (R20+R21 强制)
- 产物: `products/task433_issue143_kappa_ownership_repair/{config,verdict,precheck}.json`

## Precheck 决策阈值
- 3 个独立 `kappa_l_raw_0/1/2` 各自 `nn.Parameter(torch.tensor(0.0))` + requires_grad=True
- Optimizer 3 group 独立 lr
- update-order trace 完整

## Gate 1 决策阈值
- 三层 κ 均有有限非零 grad + 非零更新, 互不相同
- #47 同步 trace 完整且有限
- 无越域/NaN/Inf
- hard SID round-trip
- usage >= 90%, max_load < 5%

## 复现命令
```bash
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/genrec_env
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task433 \
  python3 scripts/task433_issue143_kappa_ownership_repair.py
```