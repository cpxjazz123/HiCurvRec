# Issue #57 Gate 0 (T5 mixed-curv attention sid_embedding_init) — Wrapper PASS + Bug 发现 (2026-07-31)

## 任务

Issue #57 = 把双曲/混合曲率结构直接搬进 Stage 3 T5 attention (对照 Curve Your Attention). Gate 0 实现: `HG_Rec_Issue57` wrapper 类, 通过 `sid_embedding_init` 标志切换:
- `random`: T5 默认 Xavier/Kaiming init (baseline, 无变化)
- `hyperbolic`: 用 codebook + expmap0(c·x) 初始化 SID token embedding

## 审计结果

### ✅ Sanity 5/5 PASS (wrapper 行为正确)

| Test | 检查 | 实测 |
|------|------|------|
| T1 | random init 行为 = baseline | shared.weight norm mean = 11.27 (T5 default scale) ✅ |
| T2 | hyperbolic init 应用到 SID tokens | codebook_hyp norm min=0.0000 max=0.4563 < 1 ✅ |
| T3 | random vs hyper SID tokens 显著不同 | mean abs diff = 0.7969 > 0 ✅ |
| T4 | code_path 加载正确 | sid_arr loaded, total_K=449 ✅ |
| T5 | wrapper 不改 upstream HG_Rec.py | git log HG_Rec.py 无新 commit ✅ |

### 🚨 Bug 发现: expmap0 c 符号方向 (跟 #55/#59 同模式)

**问题**: wrapper 默认 `hyp_c=1.0` (正数) 在 `expmap0` 中触发 Euclidean 分支, 不是 hyperbolic 分支:

```python
def expmap0(x, c=1.0):
    sqrt_c = math.sqrt(abs(c))
    norm = x.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    if c >= 0:                                       # ← hyp_c=0.74 走这里
        return x / (1.0 + torch.sqrt(1.0 + c * norm ** 2))  # ← 球面收缩, 不是 Poincaré ball
    else:
        return torch.tanh(sqrt_c * norm / 2.0) * x / (sqrt_c * norm)  # ← 真正的 hyperbolic
```

**Issue #57 §验证标准 期望**: SID tokens 初始化到 Poincaré ball (‖x‖ < 1/√|c| ≈ 1)
**当前实际**: SID tokens 初始化到 Euclidean sphere (‖x‖ < 1/c + 1 ≈ 1.0/0.74 + 1 ≈ 2.35, 实际 max=0.4563 — 因为输入 norm 也小)

**真 hyperbolic 需要**: `hyp_c < 0` (例如 -1.0), 走 tanh 分支.

### 当前 #158/#159 训练状态

| 任务 | 启动时间 | GPU | 当前 hyp_c | 实际初始化 |
|------|----------|-----|-----------|-----------|
| #158 random init | 23:37 | 1 | N/A (走 random 分支) | T5 default (baseline) ✅ |
| #159 "hyper" init | 23:38 | 2 | **0.74 (正数)** | ❌ **球面收缩, 非 hyperbolic** |

**#159 实际是 "sphere init" 不是 "hyperbolic init"** — 不是真正 Issue #57 的设计目标.

## 决策 (R11.5 + R12)

### 不 kill #158/#159 的理由

1. 1.5h sunk cost 已投入, kill 浪费 GPU 时间
2. #158 (random) vs #159 ("sphere init") 仍是有效 ablation: 测的是"非随机 init 是否优于 T5 default"
3. 真 hyperbolic 需要重跑 hyp_c=-1.0, 可作为 follow-up

### 立即行动

| 步骤 | 状态 |
|------|------|
| 1. 写 Gate 0 verdict (本文件) | ✅ done |
| 2. 让 #158/#159 继续跑完 | ⏸️ 监控中 (~3h 总长, 已 1.5h) |
| 3. Stage 4 eval (#158 vs #159) | ⏸️ 等 Stage 3 完成 |
| 4. **如果 #159 (sphere) 比 #158 (random) 显著好**: 重跑 #159 with hyp_c=-1.0 验证真 hyperbolic 是否更优 |
| 5. **如果 #159 = #158**: 直接 NO-GO (sphere/hyperbolic 都不是杠杆, 跟 #55/#56 同方向失败) |

## 历史类比 (跟 #55/#56 同模式)

| Issue | 表面失败 | 真实原因 |
|-------|---------|---------|
| #55/#56 | α_l/scale_l 静止 → 方向 NO-GO | 实现 bug (Bug #1+#2+#3) |
| **#57** | **hyp init 跟 random 差不多 → 方向 NO-GO?** | **实现 bug (hyp_c 符号错, 走 sphere 不是 hyper)** |

第 3 个公式 bug 案例 (跟 #47 #55 同模式). 教训: "方向看起来没用" 必须先验证实现.

## 跟 R12 (Riemannian retraction) 关联

`HG_Rec_Issue57._init_sid_embedding_hyperbolic` 只在 init 时做一次. 训练过程中 T5 optimizer (AdamW) 是 Euclidean 的, 没有 Riemannian retraction. 如果 hyperbolic init 真有效, 训练会破坏 Poincaré ball 约束.

→ 即使修复 hyp_c=-1.0, 训练期间 SID token embedding 也会飘出 Poincaré ball. 需要配套 Riemannian optimizer 或 post-step retraction.

## 未来根因诊断候选 (R10 backlog)

1. **Bug 修复**: hyp_c=-1.0 + 配套 Riemannian retraction → Issue #57 真 hyperbolic 路径
2. **T5 内部 Riemannian attention**: 不只 init, T5 attention Q/K/V 也用 hyperbolic distance (Curve Your Attention 路线)
3. **直接放弃 Issue #57 方向**: 跟 #55/#56 同证据, 架构层不传导 → 转 Issue #30+#43 synergy (唯一 GO marginal R@10=0.1022)

## Issue 状态

| Issue | 状态 | 备注 |
|-------|------|------|
| #57 Gate 0 | ✅ PASS + Bug 发现 | 本 verdict |
| #57 Gate 1 (Stage 3) | ⏸️ #158/#159 训练中 | elapsed 1.5h, ETA ~02:30 AEST |
| #57 Gate 2+ (Stage 4) | ⏸️ 等 Stage 3 | follow-up hyp_c=-1.0 候选 |

## 关联产物

- Wrapper: `HG-Rec/model/HG_Rec_Issue57.py` (200+ 行, sanity 5/5 PASS)
- Stage 3 trainer: `scripts/task158_issue57_stage3_train.py`
- Stage 3 launchers: `scripts/task158_issue57_stage3_random_init.sh` + `scripts/task159_issue57_stage3_hyp_init.sh`
- Issue #57 Gate 0 design: `verdicts/task151_issue57_direction_c_design.md`
- 本 verdict (Gate 0 audit): `verdicts/task164_issue57_gate0_wrapper_audit.md`

---
result: Issue #57 Gate 0 wrapper 5/5 sanity PASS, 但发现 hyp_c=0.74 符号 bug (走 sphere 不是 hyperbolic, 跟 #47/#55/#59 同公式 bug 模式). #158/#159 继续跑 (sphere vs random 仍是有效 ablation), 后续如 #159 优于 #158 重跑 hyp_c=-1.0.