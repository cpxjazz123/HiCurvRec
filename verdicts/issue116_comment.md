## Issue #116 R20+R21 强制 4 Gate 详细内容 + commit hash

### Gate 1 (= Stage 1 alternating training): ❌ FAIL per spec
- 关键数据: best_epoch=1, best_avg_util=**0.0091** (USAGE-KILL @ ep 5)
- final metrics: **L0/L1/L2 util=1.6%/0.8%/0.4% (≪ 90%)**, max_load=**100%** (≫ 5%, 完全坍缩)
- gate_weights 三层均 = **[0.665, 0.245, 0.090]** ❌ (跟 init [1.0, 0.0, -1.0]·softmax 完全一致 — gate 零学习)
- gate_gradient_nonzero: **False** (grad_norm=0.000e+00, gate_logits 没在 forward path 中影响输出)
- round-trip max_diff: **2.808** ❌ FAIL (tolerance 1.0)
- 训练时长: 5 epoch × ~0.6s = ~3s 后 USAGE-KILL
- 失败原因: (1) gate_logits 零梯度 (设计 bug, gate 没参与 forward); (2) codebook 冻结 → random 坍缩; (3) gate entropy floor 无效
- verdict 路径: `verdicts/task409_issue116_gate1_fail_v2.md`
- commit: 3f79615

### Gate 2/3/4: ⏸ STOP per spec
- 原因: Gate 1 FAIL

### 关键产物
- commit hash: **3f79615**
- push: origin/main (5aed3c5..3f79615)
- verdict: `verdicts/task409_issue116_gate1_fail_v2.md`
- 实施: `scripts/task409_issue116_continuous_gate_train.py` (492 lines, ThreeComponentVQ + ThreeComponentHRQVAE + alternating phase + 7 项 PASS 验证)
- 训练产物: `products/task409_issue116_continuous_gate/` (ckpt + train.out + verdict.json)
- 整体决策: **❌ NO-GO 收口** (Issue #116 跟 #113 联立 = 三分量 product gate 架构级 NO-GO)

### R18 v2 跨方向联立 (跟 #113 路径对比)
| 维度 | Issue #113 (task407, closed) | Issue #116 (本 task) |
|------|------------------------------|----------------------|
| D1 spec 摘录 | 三分量 product schema (R137 fix) | **连续 distortion + 冻结** |
| D2 实施核心 | 单阶段 hard assignment | **两阶段交替 (gate + codebook)** ✅ 新路径 |
| D3 Gate 失败机制 | 4 轮 v1-v4, gate stuck 在 init | **gate_logits 零梯度 + codebook 随机坍缩** ✅ 新根因 |
| D4 引用文献 | arXiv:2307.04514 | arXiv:2307.04514 ✅ |

新数据 + 新路径, 不允许沿用判决 (R18 v2). 两阶段交替不是 gate 零梯度的解药; Issue #113 跟 #116 联立 = 三分量 product gate 架构级 NO-GO.