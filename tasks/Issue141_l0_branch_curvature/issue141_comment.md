**Issue #141 CLOSED — 机制有效 (MECHANISM_EFFECTIVE)** (commit a682298d6800795e5c56956d9903091bbcd2728a, pushed to gitlab origin/main)

## Gate 1 — 输入与实现一致性: PASS (7/7)
1. A/B 公共源码 (stage1/3/4) 路径归一化后 hash 一致; stage2 唯一差异 = L0_SHARED_CURVATURE (单变量); seed/epochs/batch/c_range/lambda_anchor 全等。
2. Stage1 embedding 逐元素一致 (sha 1a6dd2ac)。
3. 训练前 B 的 64 个 c_0,b 与 A 的 c_0 全等 (c0_ref=1.014055)。
4. 初始 L0 distance (max diff <1e-6) / assignment / loss / SID 完全一致 (0 漂移)。
5. 扰动 B 的 theta_9 → 只有第 9 列距离改变。
6. A (1 θ) / B (64 θ) 全部进 optimizer, 梯度 finite。
7. 无 NaN/Inf、0 边界命中、无尺度捷径 (d_norm 全局标量归一化, 与 Issue140 同设计)。

## Gate 2 — Stage2 几何与 SID: PASS
- A (共享曲率): c_0 学到 1.0068, std=0 (设计语义, 非捷径), 0 边界命中; L0 assignment vs ref 变化 90.04%。
- B (codeword-specific): c_0,b 分化 std=0.00233, range [1.000, 1.011], 0 边界命中, 无尺度捷径; **occupancy Spearman=-0.604 (p=3.8e-7, Holm 显著)** — 高占用 codeword 曲率更低, 与 Issue140 方向一致; L0 assignment vs ref 变化 88.69%。
- 两组 SID 均 4-tuple util 1.0, dead code 0, reload 5/5 一致。

## Gate 3 — Stage3/4 A/B 对照: 机制有效
- Stage3 (DDP 4 卡, EARLY_STOP=20): A best valid R@10=0.1239 (epoch 134); B best valid R@10=**0.1282** (epoch 194)。
- Stage4 (单 ckpt, beam=20, 全量 24772):
  - test R@10: **A=0.0968 vs B=0.0990 (+0.0022)**; R@5 +0.0024, R@20 +0.0027, NDCG 全指标 B 领先。
  - **paired bootstrap (10000 次): ΔR@10 95% CI=[0.00004, 0.00436] 不跨 0, P(H0: Δ≤0)=0.025**。
  - McNemar: only_B=405 vs only_A=351 (p=0.054, 边缘)。
  - first_error=1 比例: 87.98% → 87.93% (改善 -0.0005)。
  - 固定 L0 桶 [200,1000) (reference-0 编码, 固定 item index, n=6717): 0.1699 → 0.1694 (-0.0004, 恶化 < 0.01 容差), 另一项 (first-error) 改善 → 判定通过。
- B 无尺度捷径; TARGET (>0.1020) 未达 (0.0990)。

## 结论
codeword-specific bounded curvature (B) 相对共享曲率 (A) 机制有效: 提升来自 B 的曲率分化 (与 occupancy 显著相关), paired 检验显著, 非 SID 重分配 artifact (固定 reference-0 item index 分桶下无重选)。与 Issue140 (0.0968→0.0989, seed 2024) 结论一致; 本 issue seed=42 下 A=0.0968 (恰好等于 Issue139 锚点) / B=0.0990。

## 附注
- 修复了 baseline/stage4.py raw predictions 收集回归 (5610b1c 覆盖丢失, 已重新注入 baseline + 两臂 stage4.py)。
- 产物: control/ + treatment/ 完整 4-stage 目录、ab_initial_equivalence.json、l0_curvature_trace_{control,treatment}.csv、l0_assignment_paired.csv、scale_shortcut_audit_{control,treatment}.json、stage4_metrics_ab.csv、paired_prediction_analysis.csv/json、issue141_verdict.md/json。
