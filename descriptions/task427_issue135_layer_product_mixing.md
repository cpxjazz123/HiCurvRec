# Task #427 / Issue #135 [方向B Gate1] 层级 product mixing 与 anchor 限幅防 per-codeword 坍缩

## 目标

Issue #132 shared Sinkhorn per-codeword α NO-GO (util=0.009 max_load=1.0)。Issue #135 提出新机制: 把 mixing 从 per-codeword α (高维) 改为 **每层 3 个标量** (低维), 减少 codeword 级自由度导致的坍缩等价解; assignment 以 learnable-κ 主路作为 **anchor**, product components 只作为 bounded correction; 加 component contribution 与 hard-count usage 审计。

## 实施

- 脚本: `scripts/task427_issue135_layer_product_mixing.py`
- LayerProductHRQVAE 类: 3-component d_mix + layer-level mixing weights (3 scalars per layer, NOT per-codeword)
- Bounded correction: 每个 component contribution ≤ 30% of anchor distance, anchor 至少 ≥ 40%
- 5-step audit per Issue #135 §Gate1 1:
  1. 三分量 d_mix 有限
  2. layer-level mixing 微扰改变 d_mix/loss
  3. learnable κ 与 mixing 梯度有限非零
  4. bounded correction 不覆盖 anchor
  5. hard assignment/round-trip 存在

## 预期产物

- `products/task427_issue135_layer_product_mixing/verdict.json`
- `verdicts/task427_issue135_gate1_*.md`
- 30 epoch 训练 + 5-step audit 全部落盘

## Gate 1 决策

- PASS: 30 epoch 内 (a) L0/L1/L2 util ≥90% (b) max_load <5% (c) 5-step audit PASS
- FAIL: USAGE-KILL @ any epoch 或 audit FAIL

## 关联

- Issue #135: layer-level mixing + anchor-preserving hard assignment + bounded correction
- 联立 #128 (per-item posterior) + #132 (per-codeword α Sinkhorn) → #135 (layer-level + anchor) 是低自由度对策
- R7: 启动前 nvidia-smi 选空闲 GPU (cuda:1, 4×L40S 全空闲)
- R12: 训练 ckpt 强制保存到 products/task427/
- R17 + R20: commit + verdict 必须详细 4 Gate 回答