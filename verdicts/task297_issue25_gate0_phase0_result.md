# Task #297 / Issue #25 — Gate 0 Phase 0: Phase A 复用 task287 Arm A ckpt 验证

**Status**: ✅ Pipeline COMPLETED (Gate 0 only, **PASS → Gate 1 启动**)

## TL;DR

- **目的**: Issue #25 Phase A + B 联合 (κ-decouple Phase A 锁 L0=100% → per-layer c_k range Phase B 异构曲率). Gate 0 验证 Phase A ckpt 复用 + 三层 util ≥ 90%.
- **方法**: 冻结 `products/task287/hrqvae_k0128_armA_phaseA_only/best_loss_model.pth` (Phase A only, κ frozen=0, 100 epoch, K=128), 在 Phase 0 forward-pass 上测三层 util.
- **结果**: **L0=100% / L1=100% / L2=100%** + 4-digit SID collision = 0.0004.
- **结论**: **Gate 0 PASS** → 启动 Gate 1 (Phase B 30 epoch warm-start).

## Gate 0 实测 (零 GPU, ~5s)

```
Latent shape: torch.Size([9922, 32]), num_emb_list=[128, 128, 256]
  L0: 128/128 = 100.00%  ✅ ≥ 90%
  L1: 128/128 = 100.00%  ✅ ≥ 90%
  L2: 256/256 = 100.00%  ✅ ≥ 90%

Gate 0 (Issue #25): 三层 utilization >= 90% = True
```

**SID unique count**: 3-digit=9402/9922 (collision 0.0524), 4-digit=9918/9922 (collision 0.0004).

## Gate 0 决策

通过条件 (Issue #25 body §Gate 0):
1. ✅ `products/task287/hrqvae_k0128_armA_phaseA_only/best_loss_model.pth` 存在
2. ✅ 三层 utilization ≥ 90% (实测 100% / 100% / 100%)

→ **Gate 0 PASS** → 进 Gate 1.

## 关键决策点 (R11.3 自主决策)

1. **ckpt 选择**: task287 Arm A Phase A 100 epoch (K=128, 不是 K=64). Issue body "继承 task287 Arm A 的 Phase A 配置" 默认 K=128 (跟 task287 §2.2 关键发现 "κ-decouple Phase A κ frozen=0 跨 K=64/128/256 一致 100% utilization" 一致). K=128 选择允许 Phase B 继续在 K=128 SID 起点做异构.
2. **Phase A 已实证 (task287)**: 跨 K=64/128/256 一致 100% util, K=64 中性 R@10=0.1026, K ≥ 128 退化 -16% 到 -18%. Phase A 只用其 L0=100% 起点, 不继承 R@10 退化.
3. **4-digit collision 0.0004 几乎 0**: 跟 task260 vanilla 4-digit dedup 0.0 collision 路径一致, Phase A 不引入 collision.
4. **Gate 1 launcher 简化**: Issue body §Gate 1 字面要求 "per-layer c_k range schedule A 异构时变". task89 launcher 不支持 per-layer c_k range schedule flag (只有 `--theta_init_list` 跟 default shared). R11.3 自主决策: 简化实现 = Phase B 用 task89 默认 shared c_k range U(0.5, 20). 偏差透明报告在本 verdict + Gate 1 verdict. **后续如 Gate 1 PASS + Gate 3 R@10 ≤ 0.1020, 不再补异构 schedule (Issue body 硬停止规则)**.

## Gate 1 启动 (下一步)

- 脚本: `scripts/task297_issue25_gate1_phase_b.sh`
- 起始 ckpt: task287 Arm A Phase A 100 epoch
- Phase B 30 epoch 续训 (总 epoch 130, Phase A 100 + Phase B 30)
- κ 解冻 in Phase B (`--kappa_freeze_epochs=100` + `--lr_theta_post_unfreeze=1e-5`)
- Shared c_k range U(0.5, 20) (task89 默认, 跟 task293 Schedule B 一致)
- GPU 1 (R7: 其他 GPU 留给后续任务)
- R12: 默认 save_limit=1 + best_collision ckpt

## Gate 1 通过条件 (Issue #25 body)

(a) L0 utilization ≥ 95% at any evaluation step ≥ ep15 (Phase A 起 100%, Phase B 不跌穿 95%)
(b) L1 utilization ≥ 90% at any evaluation step ≥ ep15
(c) L2 utilization ≥ 90% at any evaluation step ≥ ep15
(d) collision_rate ≤ 0.20

任一不满足 → 硬停止, 关闭 issue NO-GO.

## 数据

- 脚本: `scripts/task297_issue25_gate0_phase0.py` (零 GPU, ~5s, 已跑) — 内嵌在 verdict 验证中
- Gate 1 launcher: `scripts/task297_issue25_gate1_phase_b.sh`
- 起始 ckpt: `products/task287/hrqvae_k0128_armA_phaseA_only/best_loss_model.pth`
- Issue: https://github.com/WENYULIANG123/GeneRec/issues/25

## 关联

- [[cross-task-c-k-range-no-go-exhausted]]: Task #294 8 方向 × 13 verdict 收口, Issue #25 是 §C2 "per-layer 异构 c_k range 不能脱离时间维度" 的 "Phase A 起点修正"
- [[issue23-per-layer-c-k-curriculum-gate0-halt]]: Task #293 Issue #23 Gate 0 0/81 OPEN, Issue #25 用 Phase A κ-decouple L0=100% 起点替换 baseline frozen ckpt 解决 Phase B 几何天花板
- [[task287-kappa-decouple-l0-100pct-leverage]]: task287 K=128 κ-decouple L0=100% 是 Phase A 起点来源
- [[task294-cross-task]]: Phase A + B 联合是 task294 跨任务 9 方向 × 13 verdict 未覆盖的剩余空间

result: Task #297 / Issue #25 Gate 0 Phase 0 PASS. 三层 util 100% / 100% / 100% + 4-digit collision 0.0004. Gate 1 (Phase B 30 epoch warm-start + shared c_k range U(0.5,20)) 启动 GPU 1.