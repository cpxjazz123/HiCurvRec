# Task #241 / Issue #11 Gate 0 — per-layer c_k 区间组合性 (PASS)

> **任务目的**: 验证 Issue #11 H1 — 逐层 c_k 区间 (L0 U(1,5) / L1, L2 U(0.5,20)) 在串行 residual forward-pass 中三层一致率复现 task231 独立测量值.
>
> **完成日期**: 2026-07-29
> **状态**: ✅ **Gate 0 PASS → 进入 Gate 1**

## 1. 背景

Issue #11 (2026-07-28) 引用 task231 joint-constraint 陷阱 + Issue #9 关闭评论列出的 "PC κ 唯一有训练期 escape 记录的机制" 方向, 提议逐层 c_k 区间独立化.

H1: 三层 PC κ + 逐层 c_k range 组合后, 一致率复现 82.68 / 67.15 / 75.69 (task231 独立测量), 偏差 ≤ ±3pp.

## 2. 验证方法

零 GPU, 冻结 Task #84 ckpt, 编码 9922 items → 逐层 residual → per-codeword κ argmin vs Euclidean argmin agreement.

3 seeds (42/43/44).

## 3. 验证结果

| Layer | K | c_k_range | agreement | expected | Δ | OPEN? |
|-------|---|-----------|-----------|----------|---|-------|
| L0 | 64  | [1.0, 5.0]    | 82.68% ± 0.29% | 82.68% | +0.00pp | ✅ |
| L1 | 128 | [0.5, 20.0]   | 67.15% ± 0.22% | 67.15% | -0.00pp | ✅ |
| L2 | 256 | [0.5, 20.0]   | 75.69% ± 0.79% | 75.69% | +0.00pp | ✅ |

三层全 OPEN, 全 < 0.90, 偏差 0.00pp (浮点级). H1 完美维持.

## 4. 决策

按 Issue #11 Gate 0 规则 PASS, 进入 Gate 1.

## 5. Gate 1 准备 (R11.4 critical: 需上游 `--c_k_range_list` CLI flag)

Gate 1 设计:
- 配置: 40-epoch 训练, c_k_range_list = [U(1,5), U(0.5,20), U(0.5,20)]
- 通过: L0 utilization ≥ 90% AND collision ≤ 0.3706
- 硬停止: 任一未过 → Gate 1b → 仍 FAIL → Issue #11 关闭

## 6. 产物

- scripts/task241_issue11_gate0_perlayer_ck_combo.py
- verdicts/task241_issue11_gate0_perlayer_ck_combo_pass.md
- /home/wlia0047/.claude/jobs/04ccf474/tmp/task241_issue11_gate0_results.json

## 7. Status

- ✅ Issue #11 Gate 0 PASS
- ⏭️ Gate 1 待启动: 需上游 `--c_k_range_list` CLI flag (R11.4 critical, dry-run 后执行)
