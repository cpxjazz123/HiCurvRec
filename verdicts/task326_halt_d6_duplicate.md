# Task #326 — HALT 决策: D6 ablation 重复 task304 + 转向 K=384 probe

**日期**: 2026-07-30
**状态**: 🔴 HALT — 原 task326 (D6 ablation Arm B+C launchers) 删除, 转向 Task #326 K=384 sweet spot probe
**Root cause**: Task #326 D6 ablation 描述与 task304 (D6 ablation 3-arm) 完全重复, task304 已闭环 (Arm A R@10=0.0990 / Arm B R@10=0.0943 / Arm C R@10=0.1022 → synergy CONFIRMED)

## 事件时间线

1. **T0**: 在 D6 ablation backlog 下, 计划 task321/322/326 跑 r_l only / s_l only / r_l+s_l control 3-arm
2. **T1**: 写了 task322 D6 Arm A launcher (r_l only) + task326 D6 Arm B/C launchers (s_l only + r_l+s_l)
3. **T2**: 写完 task326 description 后, 才发现 task304 已经跑过完全相同的 3-arm 实验并闭环
4. **T3**: R11.5 立即 HALT task326, 删除 description + 两个 launcher
5. **T4**: 转向 K=384 sweet spot probe (真正未探索方向) 作为新的 task326

## Task #304 既有结论 (D6 ablation 已闭环)

| Arm | 配置 | R@10 | 判定 |
|-----|------|------|------|
| A (r_l only) | r_l=[0.1,1,10] + s_l=[1,1,1] | 0.0990 | -2.9pp NO-GO |
| B (s_l only) | r_l=[1,1,1] + s_l=[2,2,2] | 0.0943 | -7.5pp NO-GO |
| C (r_l+s_l) | r_l=[0.1,1,10] + s_l=[2,2,2] | 0.1022 | +0.2pp GO marginal |
| **判定** | r_l alone NO-GO, s_l alone NO-GO, **r_l+s_l synergy CONFIRMED** |

## 新 Task #326 方向 (K=384 sweet spot probe)

- 取代浪费方向, 选真正未探索 ROI
- K-sweep K0 ∈ {32, 64, 128, 256} (task194) + K0 ∈ {512, 1024} (task279) → K=256 anchor ⭐⭐⭐
- **K=384 未探索** — K=256 → K=512 插值点
- 决策阈值 R@10 > 0.1053 (vs task194 K=256 anchor)

## R9 audit + cleanup

- task326 description 删除 → 引入空洞 326
- 立即用 renumber (Task #327 K=384 → Task #326) 填补空洞
- R9 layer-2 audit PASS: descriptions 1-326 连续无空洞
- 启动器: `scripts/task326_k384_sweet_spot_stage1.sh`

## 教训 (R11.5 透明)

- 任何创建新 task description 前, **必须** grep verdicts/ 检查是否已有相同实验闭环
- D6 ablation 方向 task304 闭环后, 不应该再创建 task321/322/326 重复 3-arm
- R9 layer-1 + R9 layer-2 之外, **还应该有 R9 layer-3 (verdicts/ 重复实验检查)**

## 关联

- task304 (D6 ablation 3-arm 闭环, synergy CONFIRMED)
- task194 (K-sweep K=256 anchor ⭐⭐⭐)
- task279 (K-sweep K=512/1024 NO-GO)
- verdicts/task326_halt_d6_duplicate.md (本文件)
- descriptions/task326_k384_sweet_spot_probe.md (新方向)

result: Task #326 HALT D6 ablation duplicate (task304 已闭环). 转向 K=384 sweet spot probe (真正未探索), 启动器 scripts/task326_k384_sweet_spot_stage1.sh. 等 task320 完成 GPU 1 释放后启动.