**Issue #144 CLOSED — 无效 (INVALID)** (commit 26b2d77ddd398eb8c1ec32242272652d33ab11fb, pushed to gitlab origin/main)

## Gate 1 — 输入与初始等价性: PASS (7/7)
1. A/B 公共源码 (stage1/3/4) 路径归一化 hash 一致; stage2 唯一差异 = SHARED_LAYERS (A=[T,T,T] vs B=[T,T,F]); seed/epochs/c_range/c_ref/lambda_anchor 全等。
2. Stage1 embedding 逐元素一致 (sha 1a6dd2ac)。
3. 训练前 B 的 256 个 c_2,b 与 A 的 c_2 全部相等 (三层 c_l,ref 全等, max diff <1e-7)。
4. 初始 L2 distance / assignment / loss / SID 完全一致 (0 漂移)。
5. 扰动 B 的 θ_2,9 → 只有第 9 列改变。
6. A (3 θ) / B (1+1+256=258 θ) 全部进 optimizer, 梯度 finite。
7. 无 NaN/Inf、0 边界命中、无尺度捷径。

## Gate 2 — Stage2 几何与 SID: PASS (但揭示机制抑制)
- A (三层共享): L0/L1/L2 std≈0 (设计语义), c=[1.007, 1.056, 1.073]。
- B: **L2 分化 std=0.00026, range [1.0723, 1.0737]** — 256 个 θ_2,b 塌缩到 c_ref≈1.073! per-codeword anchor (spec 强制) 在 256 codeword 层完全抑制分化 (比 Issue141 B 的 L0 0.0023 更严重 — codeword 越多每个被拉回越强)。
- 两组 SID 均 4-tuple util 1.0, reload 5/5 一致; 无 scale shortcut。

## Gate 3 — Stage3/4 A/B: **无效 (INVALID)**
- Stage3: A best valid R@10=0.1273 (epoch 179) vs B 0.1254 (epoch 44) — A 领先。
- Stage4: **A test R@10=0.0999 vs B 0.0959 (-0.0040)**。
- **paired bootstrap: ΔR@10 CI=[-0.0060, -0.0019] 不跨 0, 但方向为负 (P=1.0)**。
- McNemar: only_A=394 vs only_B=296 (**p=0.0002**) — A 显著更好。
- first_error=1: A=88.55% vs B=88.29% (B 劣化); fixed L0 桶 (reference-0, n=6717): 0.1708 vs 0.1702。
- TARGET 未达; B 无 scale shortcut 但曲率分化被完全抑制。

## 关键结论: 曲率粒度收益集中在 L0 (粗码层), L2 (细码层) 应保持共享
- **L0 per-codeword 有效** (Issue141: +0.0022), **L2 per-codeword 无效甚至负向** (-0.0040)。
- 机制: (a) per-codeword anchor 在 256 codeword 层完全抑制分化 (std 0.00026); (b) 即使不抑制, L2 曲率重排破坏细粒度 SID 语义。
- **三层共享曲率 (A=0.0999) 是全框架当前最优配置** — 超过 Issue142 mean-only L0 (0.0992)。即: 比"L0 per-codeword + mean-only anchor"更好的是"全部共享"。
- 与 Issue140-142 系列结论整合: 曲率框架的收益窗口在 L0 层 (粗码), 扩展 L1/L2 无益。

## 附注
- 修复: control/treatment 路径混淆 (cp 覆盖导致 control 误写 treatment 产物) — 已修复路径并重新完整训练两臂 (A/B 各 1000 epoch × 2 轮); resume 模式 Gate2 κ 检查对全 branch 层跳过 (κ 恒 0 是设计语义)。
- 产物: control/ + treatment/ 完整 4-stage 目录、ab_initial_equivalence.json、l2_curvature_trace_{control,treatment}.csv、l2_assignment_paired.csv、scale_shortcut_audit_{control,treatment}.json、stage4_metrics_ab.csv、paired_prediction_analysis.csv/json、issue144_verdict.md/json。
