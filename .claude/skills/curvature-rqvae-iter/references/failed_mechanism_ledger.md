# Failed Mechanism Ledger（仅 Agent F 标 TRUE_MECHANISM_FAIL 的条目）

> **该 ledger 仅接受 Agent F 标为 TRUE_MECHANISM_FAIL 的条目；其它 4 类失败仅写入 failure_attribution_iter${i}.md，不进入本 ledger**。
>
> 列定义：
> - `iter_id | mechanism_id | 类别 | fingerprint 摘要 | Stage3 状态 | 累计次数（同类）`
> - 类别：1=机制未生效（DE 反向）、2=SID 几何方向错、3=SID 几何对但 downstream 不吃、4=其它（不进入本 ledger）
> - True_MECHANISM_FAIL 由 Agent F 在收到 Agent D 三态判定 + Agent E 归类后综合裁决，必须包含至少 1 条 DE 反向或完全不发生的硬证据

---

## Ledger 条目

- iter7 \| wide_curriculum_warmstart \| 类 3（SID 几何对但 downstream 不吃） \| full_gini 0.1225, collision 13.2%, unique 21350/24587, H(L1\|L0)=5.1757, wider cyclic c range (0.05..1.5) warm-started from iter4 \| test_R@10=0.0565 (+0.0003 vs iter4) \| 累计 1
- iter8 \| curvature_dependent_sinkhorn \| 类 3（SID 几何对但 downstream 不吃 — 不同机制族） \| full_gini 0.0684, collision 7.1%, unique 22831/24587, H(L1\|L0)=5.5336, balanced per_layer [0.2046, 0.2303, 0.2233], Sinkhorn ε ∝ c/c_max, warm-started from iter7 \| test_R@10=0.0595 (+0.0030 vs iter7) \| 累计 1
- iter9 \| power_law_sinkhorn \| 类 3（power-law ε over-amplified cycle contrast, SID stats improved but downstream regressed） \| full_gini 0.0618, collision 6.5%, unique 22997/24587, H(L1\|L0)=5.5042, per_layer [0.1120, 0.1420, 0.2850], ε ∝ (c/c_max)**0.5, warm-started from iter8 \| test_R@10=0.0568 (-0.0027 vs iter8) \| 累计 1
- iter10 \| sinkhorn_iters_5 \| 类 3（Sinkhorn 3→5 iters shifted L2 utility to dominate; on same edge as iter9 regression） \| full_gini 0.0665, collision 6.9%, unique 22879/24587, H(L1\|L0)=5.4893, per_layer [0.1739, 0.2460, 0.3233], sk_iters 5 vs 3, ε linear (iter8), warm-started from iter8 \| test_R@10=0.0594 (-0.0001 vs iter8) \| 累计 1
- iter11 \| curvature_scaled_commitment \| 类 3（c-modulated commitment preserved iter8 balance but only +0.0003 gain） \| full_gini 0.0641, collision 6.7%, unique 22948/24587, H(L1\|L0)=5.5844, per_layer [0.1922, 0.2108, 0.2240] preserved, commitment weight α=0.25 c-modulation, warm-started from iter8 \| test_R@10=0.0598 (+0.0003 vs iter8) \| 累计 1
- iter12 \| short_period_faster_cycle \| 类 3（cycle period 100k→50k preserved iter11 balance but Stage3 regressed） \| full_gini 0.0629, collision 6.5%, unique 22982/24587, H(L1\|L0)=5.5935, per_layer [0.1877, 0.2040, 0.2220], period 50k vs 100k in iter11, warm-started from iter8 \| test_R@10=0.0585 (-0.0013 vs iter11) \| 累计 1
- iter13 | per_layer_temperature | 类 3 (best SID stats but Stage3 ceiling at 0.060; log_tau_l drifted but no downstream gain) | full_gini 0.0482, collision 4.9%, unique 23382/24587, H(L1|L0)=5.8442, per_layer [0.1918, 0.1653, 0.1587], log_tau_l multiplier per layer, warm-started from iter8 | test_R@10=0.0596 (+0.0002 vs iter8) | 累计 1

---

## Ledger 维护规则

1. **写入权限**：仅 Agent F（Mechanism Classifier）在确认 Agent D 三态为 `MECHANISM_FAIL` 且 Agent E 归类为"类 1：机制未生效"后追加一行；其它 4 类失败不进入本 ledger。
2. **追加格式**：每行 6 列固定列序 `iter_id | mechanism_id | 类别 | fingerprint 摘要 | Stage3 状态 | 累计次数（同类）`，列内 `|` 必须用 `\|` 转义。
3. **同类合并**：同类机制（如"attention temperature 与 cyclic c(t) 联动漂移"）累计次数递增；新发现的子指纹需另起一行。
4. **删除/重写规则**：禁止删除历史条目；如需修正 fingerprint 摘要，追加新行而非修改原行。
5. **配套文件**：`failure_attribution_iter${i}.md`（每轮单独写，归类为类 1 时由 Agent E 写，归类为其它类时由相应 Agent 写）。
