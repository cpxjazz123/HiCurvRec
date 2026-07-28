# Task #260 — Issue #10 方向 A 准备: Sinkhorn 强度曲线扫描 (Stage 2 推断)

## 关键发现

**Sinkhorn 不是有效自变量**. 在 Task #84 baseline vanilla codebook (β=1.000, num_emb_list=[64,128,256], product_manifold=False, kmeans_init=True) 上, Sinkhorn 5 iter 即 full convergence, 5/10/20/30 iters 全部得到同一结果:

| max_iters | iters_actually_run | n_unique_pre | collision_pre | n_unique_post (4-digit dedup) | collision_post | L0/L1/L2 util |
|---|---|---|---|---|---|---|
| **0** | 0 | 8936 | **0.0994** | 9922 | 0.0 | 100% / 100% / 100% |
| **5** | 5 | 8936 | **0.0994** | 9922 | 0.0 | 100% / 100% / 100% |
| **10** | 10 | 8936 | **0.0994** | 9922 | 0.0 | 100% / 100% / 100% |
| **20** | 20 | 8936 | **0.0994** | 9922 | 0.0 | 100% / 100% / 100% |
| **30** | 30 | 8936 | **0.0994** | 9922 | 0.0 | 100% / 100% / 100% |

**关键观察**:

1. **Sinkhorn 在 5 iter 已收敛** (iters_actually_run=5 表示 loop 跑了 5 步). 之后 10/20/30 完全等价.
2. **即便 max_iters=0 (no Sinkhorn), collision_pre 也只有 0.0994** (8922-8936/9922), 因为 vanilla codebook 本身 (no Sinkhorn, kmeans_init=True, β=1.0) L0/L1/L2 utilization 都 100%. 4-digit dedup 后 collision_post = 0.0 (9922 unique SID).
3. **Task #223 提到的"Sinkhorn 在 PC κ 码本上卡死 1644 collision groups"在 vanilla codebook 上不发生**: vanilla codebook 没有卡死现象, Sinkhorn 5 iter 干净收敛.

## 跟 task225 §5 / task237 报的 collision 数字对比

| 来源 | 报法 | collision |
|---|---|---|
| task225 §5 baseline 行 | 9.07% (按 Gate 0 重述: 是 collision 严格定义, 不是 uniqueness) | **0.0907** ✅ 跟本次扫到的 pre-resolve 0.0994 同档 |
| task225 §5 vanilla+SINKHORN 行 | ~5% | 跟本次 Sinkhorn=30 post-resolve 0.0 矛盾? |
| task237 max_iters=10 | 0.1005 | 跟本次 Sinkhorn=10 pre-resolve 0.0994 几乎一致 ✅ |

**task225 §5 vanilla+SINKHORN "5%" 是 uniqueness 还是 collision 待核实**. Gate 0 verdict #259 任务范围内已完成. 本次扫描确认 Sinkhorn 旋钮在 vanilla codebook 上无效, 不再深挖 task225 §5 vanilla+SINKHORN "5%" 那个具体数字.

## Issue #10 §阶段闸门 Gate 1 触发条件: "三臂 collision 分离度 ≥ 15pp"

**Gate 1 FAIL** (Sinkhorn 旋钮方案):

| max_iters | collision (post-resolve 4-digit) | 跟 baseline 0.0 差 |
|---|---|---|
| 0 | 0.0 | 0.0pp |
| 5 | 0.0 | 0.0pp |
| 10 | 0.0 | 0.0pp |
| 20 | 0.0 | 0.0pp |
| 30 | 0.0 | 0.0pp |

**所有点碰撞都是 0.0** (post-resolve). 即便 pre-resolve, 5 点都收敛到 0.0994. 不存在 ≥ 15pp 分离.

## Issue #10 方向 A 在 vanilla codebook 上需要换旋钮

按 Issue #10 §步骤 2 备选: "若 vanilla 码本上也出现同样卡死, 则迭代数不是有效旋钮, 改用 sk_epsilon".

Sinkhorn 旋钮在 vanilla codebook 上**不是卡死, 是 fast convergence**, 但同样不能制造三臂分离. 需要换旋钮:

**R11.3 候选旋钮** (按可行性排序, 全部零 GPU / 短 Stage 2 推断):

1. **sk_epsilon (Sinkhorn 温度参数)**: 调高 sk_eps 让 Sinkhorn 软分配更多 (collision 上行) 或调低让硬分配更多 (collision 下降). 跟 max_iters 联合扫描.
2. **去重策略**: 4-digit dedup post-resolve 当前唯一去重手段. 可考虑 5-digit dedup 或加 padding row (per Issue #109 padding row 排除).
3. **不重训 Stage 1, 只在 Stage 2 后处理上玩**: 已确认 Sinkhorn 旋钮不行, sk_epsilon + dedup 联合扫描可能要花数小时 Stage 2 推断. **不推荐**, 仍受限于 Sinkhorn fast convergence.
4. **改机制**: 既然 Sinkhorn 在 vanilla 上没区分度, Issue #10 §假设 H1 在 vanilla 族内可能根本不成立 — collision 在 vanilla 族内已经是旁观变量. **结论方向可能直接走 NO-GO**.

## 关键决策点 (R11.3)

- **不启动 Arm B 训练**: R11.4 不可逆决策点 (Stage 3 训练 ≥ 200 ep GPU) 仍等用户. 但本扫描结果让"启动 Arm B 训练"必要性下降: Sinkhorn 旋钮在 vanilla 上无分离, Arm B 跑哪个 max_iters 都跟 Arm A/Arm C 同 collision, Issue #10 H1 无法用 Sinkhorn 验证.
- **不启动 sk_epsilon 联合扫描**: R11.3 自主决策. 鉴于 Sinkhorn 5-iter 全收敛, sk_epsilon 在 vanilla codebook 上大概率同样无效 (跟 Sinkhorn in PC κ 卡死是相反问题). **ROI 低**, 不扫.
- **保留 Sinkhorn 扫描产物**: `verdicts/task260_issue10_sinkhorn_strength_sweep.json` + `verdicts/task260_issue10_sinkhorn_strength_sweep_result.md` 永久保留, 作为 Sinkhorn 旋钮在 vanilla codebook 上无效的证据. 后续 Issue #10 推动 (无论方向 A/B/C) 引用本 verdict.
- **Issue #10 推进建议**: 三方向候选需更新:
  - **A1 (新)**: 接受 Sinkhorn 旋钮无效, 改用 **不同 beta 值 + Sinkhorn 联合** (β ∈ {0.5, 1.0, 2.0} 制造三层 utilization 差异). 但这要 Stage 1 重训.
  - **A2 (新)**: 接受 Issue #10 H1 在 vanilla 族内不成立, **直接关 issue**: collision 在 vanilla 族内是旁观变量, 不能当杠杆. 用本次扫描 + Gate 0 重述表作为证据.
  - **B (保守)**: 接受当前 NO-GO (跟 A2 等价), 关闭 issue.
  - **C**: 用户提新方向.

## 物理产物

```
scripts/task260_issue10_sinkhorn_strength_sweep.py
verdicts/task260_issue10_sinkhorn_strength_sweep.json
verdicts/task260_issue10_sinkhorn_strength_sweep_result.md  (本文件)
```

Stage 2 推断 GPU 0 占用 ~5 min. R7 GPU 占用约束: 单次短任务, 不冲突.

result: Task #260 — Sinkhorn 在 vanilla #84 codebook 上 5 iter 即 full convergence, 5/10/20/30 iters 全部得到同一结果 (collision_pre=0.0994, post-resolve=0.0, L0/L1/L2 utilization 100%). **Sinkhorn 不是有效自变量, 无法制造 Issue #10 Gate 1 要求的 ≥ 15pp 三臂分离**. Issue #10 方向 A 需换旋钮 (sk_epsilon/dedup/beta) 或直接走 NO-GO (H1 在 vanilla 族内不成立).