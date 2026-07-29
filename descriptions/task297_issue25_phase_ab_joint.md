# Task #297 — Issue #25 Phase A + B 联合 (κ-decouple + per-layer c_k range)

## 背景

GitHub Issue #25 (2026-07-29 创建, R14 第一步扫描发现): "[Phase A + B 联合] κ-decouple Phase A 锁 L0=100% → per-layer c_k range Phase B 异构曲率"

Issue body 论证: 仓库内 backlog 唯一未尝试组合 = Phase A κ-decouple 锁 L0=100% (task287 已证) → Phase B per-layer c_k range 异构曲率 (task242 / task293 Phase B 在 baseline recipe 上失败因为 L0 没清空, Phase A 修正后 Phase B 才有可能做差异化分配).

## 4-Gate 协议 (Issue #25 body 硬停止规则)

| Gate | 内容 | 通过条件 | 失败动作 |
|------|------|---------|---------|
| **0** | Phase A 复用 task287 Arm A ckpt | ckpt 存在 + 三层 util ≥ 90% | 不重训 (task144 fallback) |
| **1** | Phase B per-layer c_k range 30 epoch (Phase A 起点) | L0≥95% / L1≥90% / L2≥90% / collision≤0.20 | 不进 Gate 2, 关 issue |
| **2** | Stage 2 Sinkhorn 5 iter | unique ≥ 9500 + per-layer util 偏差 ≤ 5pp | 不进 Gate 3, 关 issue |
| **3** | Stage 3 T5-mini 200 epoch + Stage 4 eval | **R@10 > 0.1020** | 关 issue NO-GO |

## Gate 0 实测 (零 GPU, 2026-07-29)

冻结 `products/task287/hrqvae_k0128_armA_phaseA_only/best_loss_model.pth` (Phase A only κ frozen=0, 100 epoch), 在 Phase 0 forward-pass 上测三层 utilization:

```
L0: 128/128 = 100.00%  ✅ ≥ 90%
L1: 128/128 = 100.00%  ✅ ≥ 90%
L2: 256/256 = 100.00%  ✅ ≥ 90%

Gate 0 (Issue #25): 三层 utilization >= 90% = True
```

**附加测量**: 4-digit SID unique = 9918/9922, collision = 0.0004 (基本 0, 跟 task260 vanilla 4-digit dedup 0.0 collision 路径一致).

**Gate 0 决策**: ✅ **PASS** — ckpt 存在 + 三层 util 100% ≥ 90%. 进 Gate 1.

## Gate 1 启动 (下一步)

按 Issue #25 body §Gate 1:
- 起始 ckpt: task287 Arm A Phase A 100 epoch (κ frozen=0)
- Phase B 30 epoch 续训
- κ 解冻 (Phase A frozen=0 → Phase B κ learnable)
- 三层独立 schedule: **Schedule A** (异构时变 U(0.5,20)→U(1,5)→U(2,8), task293 verdict 选 A)
- dead_revive = off (task242 Arm A+ + task283 NO-GO 已证)
- 评估点: 每 5 epoch 测 per-layer util + collision

## 关键决策点 (R11.3 自主决策)

1. **ckpt 选择**: task287 Arm A Phase A 100 epoch (K=128, 不是 K=64). 跟 Issue #25 body "继承 task287 Arm A 的 Phase A 配置" 一致. K=128 配置在 Issue #25 body 里被默认接受 (虽然 §4 表格 task287 Arm A R@10=0.0855 NO-GO, 但 Phase A 只用其 L0=100% 起点, 不继承其 R@10 退化).
2. **Schedule 选择**: Schedule A 异构时变 (跟 task293 verdict §3 "Schedule B/C 表现相似, 选 A" 一致).
3. **κ 解冻**: Phase B κ 不再 frozen (Issue #25 body §Gate 1 "Phase B κ 不再 frozen", 允许 κ 跟 c_k range 协同).
4. **warm-start 实现**: 需要新写 launcher `scripts/task297_issue25_gate1_phase_b.py` (复用 task275 warm-start pattern 但加 per-layer schedule A).
5. **R12 ckpt 保存**: 强制每个 epoch 末保存 (R12 规则, 避免训练崩溃丢失).

## 数据

- 脚本: `scripts/task297_issue25_gate0_phase0.py` (零 GPU, ~30s, 已跑)
- Gate 0 输出: L0/L1/L2 = 100% / 100% / 100%, collision 4-digit = 0.0004
- 起始 ckpt: `products/task287/hrqvae_k0128_armA_phaseA_only/best_loss_model.pth`
- Issue: https://github.com/WENYULIANG123/GeneRec/issues/25

## 不消耗 GPU (Gate 0 only)

Gate 0 = 零 GPU (Phase 0 forward-pass on frozen ckpt). Gate 1-3 需 GPU.

result: Task #297 Gate 0 (Issue #25 Phase A + B 联合) PASS. 三层 util 100% / 100% / 100% + 4-digit collision 0.0004. 启动 Gate 1 (Phase B per-layer c_k range 30 epoch Schedule A 异构时变).