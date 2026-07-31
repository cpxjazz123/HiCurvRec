# Task #446 / Issue #155 [方向A Gate1] 配额硬分配的κ梯度取样时序审计与复现

## R18 4 维度路径对比 (vs Issue #153 / Task #444 MECHANISM PASS + monitoring grad=0 bug)

| 维度 | Issue #153 (task444 MECHANISM PASS, monitoring FAIL) | Issue #155 (本 task, audit 升级) |
|------|--------------------------------------------------------|--------------------------------------|
| **D1 spec 摘录** | BilateralQuotaKappaModel + Hungarian lower=1 + upper=ceil(B/K)+1 + monitoring grad_kappa 在 opt.zero_grad() 后读, 显示 0 但 κ 实际有更新 | **同机制**, **仅修正 monitoring 时序**: loss.backward() 后, optimizer.step()/zero_grad() 前记录 raw κ grad. 同时记录 step 前后参数 delta, loss, util, max_load, min_load |
| **D2 实施核心** | BilateralQuotaKappaModel + Hungarian bilateral cost expand + opt.zero_grad() 后读 model.kappa.grad → 0 | BilateralQuotaKappaModel + Hungarian bilateral cost expand + **raw_grad = model.kappa.grad.clone() 在 opt.step() 之前** + **step 后 κ delta = (κ_after - κ_before).abs()** |
| **D3 Gate 1 失败机制** | Issue #153 verdict 已 MECHANISM PASS (util 100%, max_load < 5%, min_load >= 1/K, κ final=[-0.0041, 0.00135, 0.0204] 真更新), 但 monitoring grad=0 不符合"每层 κ 梯度在优化前有限且非零"的可复核证据要求 | 新机制: 修正 monitoring 时序让 raw κ grad 在 opt.step()/zero_grad() 前被记录, 实证非零. step 前后 delta 也实证. 不重做机制. |
| **D4 引用文献** | arXiv:2405.13979 + CrossRef VQ | arXiv:2405.13979 + CrossRef VQ (同文献, 不同实施: monitoring fix) |

**R18 判定**: 4 维度都有差异 (重点在 D1 monitoring 时序修正 + D2 raw_grad 在 opt.step() 前 + D3 实证非零 + D4 同文献不同实施), 必须做新实验, **不允许**套用 #143/#145/#148/#151/#153 判决 (R18 强制).

## 实施
- `scripts/task446_issue155_kappa_gradient_audit.py` (fork from task444)
- BilateralQuotaKappaModel: 同 #153 机制
- 关键修复: train_step 中 `aux_loss.backward()` 后 **立即** `raw_grad = model.kappa.grad.clone()` (在 `opt.step()` + `opt.zero_grad()` 之前), 记录到 train_curve
- 同时记录 `κ_before = model.kappa.detach().clone()`, `opt.step()`, `κ_after = model.kappa.detach().clone()`, `κ_delta = (κ_after - κ_before).abs()`
- 至少 10 个预注册记录点 (step 100, 200, ..., 1000)
- 每层 util/max_load/min_load/raw_grad_before/κ_delta 全程记录
- 不改变目标函数, 不替换三层 κ, 不用 global κ
- SHA256(item_emb) = 1a42341f01537d6d... (跟 #153 同, 已固化)
- 8 件套: config + SHA256 + precheck + raw_kappa_grad_log + train_curve + ckpt + verdict.json + commit

## Precheck 决策阈值 (Issue #155 spec)
- 三组独立可训练 κ (L0/L1/L2), 各自 optimizer 注册
- κ 依赖距离/代码本缩放 (#47 统一公式)
- 数据流清单: encoder z_e → per-layer d_hyp → aux_loss → model.kappa[l].grad
- 禁止 global shared κ, fixed-only curvature, 纯欧氏绕过, 替换三层框架

## Gate 1 决策阈值 (Issue #155 spec)
- PASS: 至少 10 个预注册记录点; 每层 raw κ grad 在 zero_grad 前均有限非零; step 前后 delta 非零; 每层 util=100%, max_load<5%, min_load>=1; 无 NaN/Inf; 8 件套齐全
- FAIL: 任一不满足即 STOP

## 复现命令
```bash
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/genrec_env
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task446 \
  python3 scripts/task446_issue155_kappa_gradient_audit.py
```

## 跟 Issue #153 关键差异
- Issue #153: 同一 train_step 代码, monitoring grad=0 (opt.zero_grad() 后读)
- Issue #155: **同机制**, 仅 monitoring 时序修正 (raw_grad 在 opt.step()/zero_grad() 前读) + 同步记录 step 前后 delta
- Issue #155 配套: `if model.kappa.grad is None: raw_grad = [0.0]*3` 防御性 (万一某 step grad 真为 0, 显式记录)