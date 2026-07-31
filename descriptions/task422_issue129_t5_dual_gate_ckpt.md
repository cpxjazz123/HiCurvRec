# Task #422 / Issue #129 [方向C Gate3] 真实 Stage3 T5-mini 双态 gate adapter + checkpoint 合同

## R18 v2 4 维度路径对比 (vs Issue #126 / Issue #123)

| 维度 | Issue #126 (task419, NO-GO) | Issue #129 (本 task, 修复方向C) |
|------|------------------------------|---------------------------------|
| **D1 spec 摘录** | Gate 4 R@K eval (复用 #123 ckpt) | **Gate 3 真实训练 + ckpt 加载合同** (zero-gate max_logits_diff ≤ 1e-5) |
| **D2 实施核心** | task84 baseline 跑 evaluator, 但 #123 ckpt 不存在 | **集成 #123 dual-gate adapter → task84 HG_Rec T5 input emb**, 2 epoch 真实训练, save+load verify |
| **D3 Gate 1 失败机制** | #123 spec 假设有 ckpt, 实际只有 architecture audit | **#129 直接产出 ckpt, 消除 #126 的 pre-condition 漏洞** |
| **D4 引用文献** | arXiv:2309.04082 | 同文献 + checkpoint contract paper §R12 |

**R18 v2 判定**: 4 维度都有差异 (重点在 D1 spec 跳回 Gate 3 + D2 ckpt 产出 + D3 pre-condition 修复), 必须做新实验.

## 实施
- `scripts/task422_issue129_t5_dual_gate_ckpt.py` (~350 lines)
- 复用 task84 HG_Rec + T5ForConditionalGeneration + 集成 DualGateAdapter
- **Issue #102 SID (task396 learnable-variable-curvature)** — 必须 SHA256 verify
- **pre-registered 有限训练 = 2 epoch** (短期训练, 不是 200 epoch baseline)
- Save+load 双合同: zero-gate mode 时 logits_diff ≤ 1e-5
- 产物: `products/task422_issue129_t5_dual_gate_ckpt/{adapter_ckpt.pt, adapter_trained_2ep.pt, verdict.json}`

## Gate 3 决策阈值
- zero-gate max_logits_diff ≤ 1e-5 ✅
- save+load 后 zero-gate max_logits_diff ≤ 1e-5 ✅
- active-gate 训练后 logits_diff > 1e-3 (验证 adapter 真起作用)
- 2 epoch training loss 全 finite

## 复现命令
```bash
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/genrec_env
CUDA_VISIBLE_DEVICES=2 python3 scripts/task422_issue129_t5_dual_gate_ckpt.py
```