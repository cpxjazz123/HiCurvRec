# Task #423 / Issue #131 [方向A Gate1] 稳定双曲 cost 的 Sinkhorn 平衡运输防单码字坍缩

## 目标

验证 Issue #131 [方向A] 的核心假设：**对稳定 hyperbolic cost matrix 执行批内 Sinkhorn 平衡运输分配**, 列边际固定为均匀、行边际为样本质量; 硬 argmin SID 仅用于记录/导出; κ 梯度来自运输损失 + 重构损失; κ 更新后继续按 #47 同步 scale/codebook.

vs 历史任务（per R18 4 维度对比）:
- D1 spec: Issue #131 假设 Sinkhorn 平衡运输能防单码字坍缩, 跟 #127 (#128) 同 root cause 假说 (#127 是 ball projection 修复数值层; #131 是 Sinkhorn 修复运输分配层, 都是非架构层修复)
- D2 实施核心: stable hyperbolic cost + Sinkhorn balanced transport + per-layer κ + transport+recon loss. 跟 #127 (#127 ball projection + VQ argmin) 实施完全不同
- D3 Gate 1 失败机制假设: 不同 (#127 是 max_load=1.0 单码字吞所有 token, #131 是假设多码字均衡分配)
- D4 引用文献: 同 #127 的 Berman-Metzler 2020 κ-Stereographic, 但 #131 走 Sinkhorn 路径

→ **4 维度不一致**, R18 强制做实验验证, 不允许沿用 #127 NO-GO 判决

## 实施核心

### `scripts/task423_issue131_sinkhorn_balanced_transport.py` (~390 lines)

**SinkhornBalancedHRQVAE**:
- `project_to_poincare_ball`: 稳定 Poincaré-ball 投影 ‖x‖<(1-eps)/√c
- `stable_pairwise_hyp`: 稳定双曲距离 acosh(1 + 2c·‖diff‖²/((1-c·‖x‖²)(1-c·‖y‖²)))
- `sinkhorn_balanced_transport`: 平衡运输分配 (行=样本质量均匀, 列=均匀, max_iters=10, tol=1e-3)
- `forward_layer`: 计算稳定 hyp cost → Sinkhorn balanced transport → 软分配 z_q_transport = plan @ codebook + 硬 argmin SID 记录
- `radial_rescale_codebook`: κ 更新后码字径向重缩放 (#47)
- `optimizer_step_hook`: κ 变化时触发 rescale

**ArgminControlHRQVAE**: 同 stable cost + argmin SID, 无 transport (对照组, 用于证伪 #127 不是 cost 不稳定导致坍缩)

### 5-step 域审计 (per Issue #131 §Gate1 1)
1. domain_ok: c√·‖z_e‖ max < 1-eps 约束
2. Sinkhorn 行/列边际残差受控 (row_res < 1.0, col_res < 1.0)
3. κ 微扰 0.5 改变 transport plan + loss (断 κ 不影响分配的 fallback)
4. gradient finite non-zero (κ_l grad 有限非零)
5. plan 有 grad-fn (plan_requires_grad=True, 不是 hard argmin 唯一梯度来源)
6. NaN/Inf 监测

### 训练配置
- K = [64, 128, 256]
- 30 epoch (Issue #131 §Gate1 2, short cycle)
- seed=42
- batch_size=256
- lr=1e-4, β=0.25
- κ ∈ [-2.0, -0.1] (softplus 限幅)

### 决策阈值
- 主配 (Sinkhorn): 3 层 usage ≥90% AND max_load <5% AND audit PASS → ✅ GO
- 控制 (argmin): 3 层 usage ≥90% AND max_load <5% AND audit PASS → 对照基线
- 任一不达标 → ❌ NO-GO

## 关键产物
- `scripts/task423_issue131_sinkhorn_balanced_transport.py`
- `products/task423_issue131_sinkhorn_balanced_transport/verdict.json`
- `verdicts/task423_issue131_gate1_result.md`

## 联立 R 规则
- R17: commit 含 Gate 1 状态 + 失败原因
- R18: 4 维度对比 + 实验强制
- R19: 立即开工不等待
- R20: 4 Gate 详细内容 ≥3-5 行/Gate
- R21: commit hash 具体, 不允许 pending
- R22: 立即开工
- R7: GPU 0/1/2 并行
- R4: py_compile 验证

## 决策
- Sinkhorn 30 epoch USAGE-KILL → 方向 A NO-GO 收口
- Sinkhorn 30 epoch util ≥90% + max_load <5% + audit PASS → Gate 1 PASS, 后续 Gate 2 (Sinkhorn 推断) + Gate 3 (T5) + Gate 4 (R@K) 推进
