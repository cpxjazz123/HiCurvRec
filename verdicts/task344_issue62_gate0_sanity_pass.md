# Task #344 / Issue #62 Gate 0 sanity test PASS (2026-07-31)

## 任务

承接 Issue #62 (owner 显式 opened, 2026-07-30 16:07) Gate 0 zero-GPU 准备:
- 验证 Issue #30 wrapper + Issue #43 wrapper 顺序 compose 不破坏 baseline
- 5/5 单元测试回归 (T1-T5)

## 5/5 Gate 0 测试结果

```
✅ T1: base HRQVAE forward OK (8, 768) out, (8, 3) indices
✅ T2: HRQVAEWithHypPre wraps cleanly (Issue #43 alone, c=0.74)
✅ T3: + per-layer Codebook Transforms r_l=[0.1,1,10]+s_l=[2,2,2] composes (Issue #62 joint)
✅ T4: gradient flows through both wrappers (L0/L1/L2 grad > 0, no dead params)
✅ T5: identity compose ≡ baseline (max diff = 0.000000, no-op safe)
```

### T4 关键发现
梯度流确认 (pre-warm 后跑真实 grad test):
- L0 grad mean norm = 0.000373
- L1 grad mean norm = 0.000198
- L2 grad mean norm = 0.000087
- HypPre c = 0.74 (fixed scalar, not learnable, as designed)

→ 两个 wrapper 顺序应用不阻挡梯度, 训练可行.

### T5 关键发现
identity compose (HypPre disabled + r_l=s_l=1) 跟 baseline **完全等价** (max diff = 0.000000):
- 证明 wrapper 顺序应用是 "no-op safe"
- 联合 ablation 失败时, 可安全回退 baseline

## Gate 0 关键修复

| 问题 | 修复 |
|------|------|
| HRQVAE.forward 返回 5-tuple (out, rq_loss, indices, path_loss, div_ent) 不是 3-tuple | T1-T5 全部解包 5 个变量 |
| `apply_per_layer_codebook_transforms` 不在 task301 gate0, 而在 gate1 | 修正 import 路径 |
| T4 kmeans init 失败 (n_samples=8 < n_clusters=64) | pre-warm 用 512 sample 触发 init_emb 设置 `initted=True` |
| `model.hyp_pre.c` 是 float 不是 tensor | 区分 float (fixed scalar) vs tensor (learnable) 处理 |

## Issue #62 Gate 1 启动条件确认

| 条件 | 状态 |
|------|------|
| Wrappers 正交 | ✅ (Gate 0 T3) |
| 梯度流通畅 | ✅ (Gate 0 T4) |
| Identity no-op safe | ✅ (Gate 0 T5) |
| 完整代码可运行 | ✅ (Gate 0 全部 5/5 PASS) |
| GPU 空闲 | ✅ (4× L40S 全空闲, 待 Gate 1 启动) |
| Owner 拍板启动 Gate 1 | **PENDING** (R11.4 critical decision) |

## 关联产物

| 类型 | 路径 |
|------|------|
| Gate 0 sanity script | `scripts/task344_issue62_gate0_joint_sanity.py` |
| Issue #62 描述 | `descriptions/task344_issue62_30_43_joint_ablation.md` |
| 复用 #43 wrapper | `scripts/task334_issue43_gate2a_hyp_pre_encoder.py` |
| 复用 #30 apply fn | `scripts/task301_issue30_gate1_stage1_train.py::apply_per_layer_codebook_transforms` |

## 后续步骤

1. **Gate 1 Stage 1 训练** (Arm D #30+#43 joint, ~3h GPU):
   - 复用 #30 训练脚本 + 插入 #43 HypPreEncoder
   - 监控: L0/L1/L2 utilization ≥ 90% @ ep ≥ 50, collision ≤ 0.20
2. **Gate 2 Stage 3 + Stage 4 eval** (~3h GPU):
   - 决策: R@10 > 0.1042 = 突破 #43 单点天花板, 转 [TARGET REACHED]
3. **R9 顺带修复**: 配合 task342/343/345 placeholder 同步填补.

**Gate 1 启动需要 owner 显式确认** (R11.4 critical decision, 大约 6h GPU).

---

result: Task #344 Issue #62 Gate 0 sanity test PASS (5/5). T1 baseline + T2 HypPre + T3 joint compose + T4 grad flow + T5 identity no-op safe. 两个 wrapper 顺序应用正交 + 梯度流通畅 + baseline 等价, 可启动 Gate 1 Stage 1 训练 (Arm D #30+#43 联合, ~3h GPU). Gate 1 启动需 owner 显式确认 (R11.4 critical decision).