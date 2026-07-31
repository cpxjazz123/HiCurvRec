# Task #424 / Issue #132 [方向B Gate1] Weighted product d_mix 共享 Sinkhorn 运输分配

## 目标

验证 Issue #132 [方向B] 的核心假设：**从 3 分量稳定 d_mix (learnable κ hyp + fixed κ=1 hyp + Euclidean) 生成共享 Sinkhorn transport plan**, per-codeword α 同时影响该全局 plan 的 cost; α 梯度来自受列边际约束的匹配; 最终 SID 仍为 d_mix hard argmin.

vs 历史任务（per R18 4 维度对比）:
- D1 spec: Issue #132 假设**共享 transport plan** 能打破 per-item posterior 数学等价灾难 (#128 per-item 后退到 weighted centroid → 码字聚集到几何中心 → 坍缩)
- D2 实施核心: 共享 Sinkhorn transport plan + per-codeword α (影响全局 cost, 不是 per-item 后验) + 稳定 d_mix. 跟 #128 (#128 per-item posterior, 同 #128 等价 task305/306 → 坍缩) 实施完全不同
- D3 Gate 1 失败机制假设: 不同 (#128 是码字加权几何中心 collapse, #132 是假设共享列边际约束避免 weighted centroid)
- D4 引用文献: 同 #128 的 Sinkhorn balanced transport (Cuturi 2013), 但 #132 用 **shared plan** 替代 per-item posterior

→ **4 维度不一致**, R18 强制做实验验证, 不允许沿用 #128 NO-GO 判决

## 实施核心

### `scripts/task424_issue132_shared_sinkhorn.py` (~440 lines)

**SharedSinkhornProductHRQVAE**:
- `project_to_poincare_ball`: 稳定 Poincaré-ball 投影
- `stable_pairwise_hyp`: 稳定双曲距离 (learnable κ)
- `sinkhorn_balanced_transport`: 平衡运输分配 (列边际=均匀)
- `forward_layer`:
  1. 3 分量 d1 (learnable κ hyp) + d2 (fixed κ=1 hyp) + d3 (Euclidean)
  2. per-codeword α = softmax(gate_logits[K, 3])
  3. d_mix = α[:, 0]·d1 + α[:, 1]·d2 + α[:, 2]·d3  (per-codeword, per-component)
  4. 共享 Sinkhorn transport plan on d_mix
  5. z_q_transport = plan @ codebook
  6. hard argmin SID 记录 (assign = d_mix.argmin)
- `radial_rescale_codebook`: κ 更新后码字径向重缩放

**PerItemControlHRQVAE**: 3 分量 + per-codeword α + per-item soft posterior (=F.softmax(-d_mix/T)) → 对照 #128 数学等价灾难

### 5-step 域审计 (per Issue #132 §Gate1 1)
1. 两个双曲域约束通过: c√·‖z_e‖ max < 1-eps (learnable κ) AND c_fixed=1·‖z_e‖ max < 1-eps
2. α 微扰 (per-component offsets [0.5, -0.3, 0.1]) 改变 d_mix, plan, loss (断 softmax invariance, 因为是 per-component 而不是 per-row constant)
3. α/κ 梯度有限非零 (kappa grad + alpha grad 都要 finite non-zero)
4. plan 行/列边际残差受控 (row_res < 1.0, col_res < 1.0)
5. hard SID 已记录 (assign_list[0].numel() > 0)

### 训练配置
- K = [64, 128, 256]
- 30 epoch
- seed=42
- batch_size=256
- lr=1e-4, β=0.25
- κ ∈ [-2.0, -0.1] (learnable) + fixed κ=1 (常量)

### 决策阈值
- 主配 (Shared Sinkhorn): 3 层 usage ≥90% AND max_load <5% AND audit PASS → ✅ GO
- 控制 (per-item posterior): 3 层 usage ≥90% AND max_load <5% AND audit PASS → 对照基线
- 任一不达标 → ❌ NO-GO

## 关键产物
- `scripts/task424_issue132_shared_sinkhorn.py`
- `products/task424_issue132_shared_sinkhorn/verdict.json`
- `verdicts/task424_issue132_gate1_result.md`

## 联立 R 规则
- R17: commit 含 Gate 1 状态 + 失败原因
- R18: 4 维度对比 + 实验强制
- R19: 立即开工不等待
- R20: 4 Gate 详细内容 ≥3-5 行/Gate
- R21: commit hash 具体
- R22: 立即开工
- R7: GPU 1/2 并行
- R4: py_compile 验证

## 决策
- Shared Sinkhorn 30 epoch USAGE-KILL → 方向 B NO-GO 收口
- Shared Sinkhorn 30 epoch util ≥90% + max_load <5% + audit PASS → Gate 1 PASS
- 联立 task305/task306/task418/task421 (#128): 如果 #132 也 FAIL, 进一步证实 per-codeword α 在任一框架 (per-item posterior / shared plan) 都坍缩 → 锁定 "per-codeword α 是坍缩根因"
