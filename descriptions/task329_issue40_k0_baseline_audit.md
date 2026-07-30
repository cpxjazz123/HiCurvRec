# Task #329 — Issue #40 Gate 0: task194 K0 baseline validation audit

**日期**: 2026-07-30
**状态**: 🔄 IN PROGRESS (Gate 0 only, 0 GPU)
**Stage**: Stage 0 validation audit (zero GPU, hard pre-Stage-1 gate per Issue #40)
**Anchor**:
- Issue #30 GO endpoint: R@10=0.1022 (+0.2% vs baseline 0.1020) — 17 方向首个 GO
- task194_k0256 cited R@10=0.1053 (+3.2%) — 后续多个 verdict (Issue #37 closure) 引用为 "更高 baseline"
- HG-Rec Task #84 baseline: R@10=0.1020

**Issue #40 核心质疑**: task194_k0256 R@10=0.1053 是否可信? verdict 文件是 template placeholder, diagnose JSON 是空 `{}`, 数字未经 gate 校验.

## Issue #40 Gate 0 (硬性前置, 必须先做)

### Gate 0 任务清单 (零 GPU)
1. **读 task194 4 个原始 Stage 4 metrics JSON** (`verdicts/task194_k0{32,64,128,256}_test_metrics.json`) 提取:
   - beam_size, topk_list, model arch, dataset split
   - Stage 3 训练 epoch 数, optimizer, lr schedule
   - checkpoint 选择规则 (best_NDCG@20? last epoch?)
2. **读 task84 baseline Stage 4 脚本** (scripts/task84_hgrec_stage3_train.py + task174_v3_stage4_eval.sh pattern) 提取:
   - 同样的 hyperparameter 列表
3. **读 Issue #30 GO endpoint 训练脚本** (task301 stage 3 训练) 提取:
   - 同样的 hyperparameter 列表
4. **对比 3 套脚本的参数差异**:
   - 如果 task194 vs task84 在 beam_size/checkpoint/Stage 3 epoch 任一项不同 → 协议不一致 → STOP
   - 如果 task194 vs #30 在 same 项不同 → 协议不一致 → STOP
5. **写审计 verdict** (Gate 0 pass/fail):
   - PASS (无差异) → 任务 329 完成, Issue #40 可进入 Gate 1
   - FAIL (有差异) → STOP, 不进 Gate 1, Issue #40 关闭 with verdict (不能 reset task194 anchor)

### Gate 0 决策阈值

- ✅ **PASS**: 3 套脚本在所有 protocol 维度一致 → task194 anchor 可信, Issue #40 close, 后续 task328 vs anchor 0.1053 比较 valid
- ❌ **FAIL**: 任一 protocol 维度不一致 → task194 anchor 不可信, Issue #40 close + Issue #37 / Issue #30 chain 重新评估, Issue #30 GO marginal 0.2pp 恢复为有效 "17 方向首个突破"

## Issue #40 Gate 0 数据点 (4 个 task194 臂 + 2 个对照组)

| K0 | R@5 | R@10 | Δ vs #84(0.1020) |
|---|---|---|---|
| 32 | 0.0817 | 0.1006 | -1.4% |
| 64 | 0.0832 | 0.1041 | +2.0% |
| 128 | 0.0830 | 0.1027 | +0.6% |
| 256 | 0.0845 | 0.1053 | +3.2% |

**Issue #40 关键异常**: K0=64 (跟 baseline #84 L0 码本大小一致) 给 +2.0%, 跟 "K0 单调影响 R@10" 假设矛盾. 最可能解释 = 协议差异.

## Gate 0 期望产出

`verdicts/task329_issue40_gate0_protocol_audit.md` + `verdicts/task329_gate0_pass_fail.json`

## Critical caveats (R11.5 transparency)

1. **Gate 0 是硬性前置**: Issue #40 明确说 "STOP if protocol difference found" — 不能跳过 Gate 0 直接进 Gate 1
2. **零 GPU 任务**: 纯文档审计, 不消耗任何 GPU 资源
3. **跟 task328 并行不冲突**: task328 R-Drop alpha sweep 正在 GPU 0/2/3 跑, Gate 0 不依赖 GPU
4. **R9 compliance**: 本任务编号 #329 = max(328) + 1 ✅
5. **决策影响范围**: 如果 Gate 0 FAIL, Issue #37 关闭时引用 0.1053 作 anchor 的所有后续 verdict 都需重新评估

## Reference verdicts (跟当前 backbone 一致)

- Issue #30 GO marginal 0.1022 (task301 R@10=0.1022 Stage 4 K=20)
- task194_k0256 R@10=0.1053 (cited but unvalidated)
- HG-Rec Task #84 baseline R@10=0.1020 (Stage 4 K=20)
- Issue #37 closure (用 task194 anchor 否决 Issue #30 GO marginal)
- Issue #38 / task320 5-arm verdict (PARTIAL GO with R-Drop α=1.0 +1.4%)
- task328 R-Drop alpha sweep (running, will use task194 anchor 0.1053 as decision threshold)
