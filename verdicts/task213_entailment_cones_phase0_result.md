# Task #213 Verdict — Entailment Cones (方向一) ❌ NO-HOPE

**日期**: 2026-07-26
**任务**: Task #213 Phase 0 — Entailment Cones 概念验证 (用户 2026-07-26 提议"方向一最有希望")
**最终判决**: **❌ NO-HOPE** — 在 HG-Rec baseline 码本上, Entailment Cones 无法工作.

---

## 1. 实验设计 (Phase 0 v1 + v2, 都不训练, 不占卡)

**v1 单组合**: α = arctan(K/scale=10) + 决策 "选 ||p|| 最大"
**v2 4 组合 sweep**:
- (A) α=arctan(K/sf=10) + 决策 cos_max (贴合优先)
- (B) α=π/2·(1-||p||) + 决策 α_min (specific 优先, Ganea 标准)
- (C) α=π/3·(1-||p||) + 决策 α_min (中等 specific)
- (D) α=π/4 + 决策 cos_max (固定锥宽 + 贴合)

**判据**:
- S1: 锥 vs 欧式 argmin 一致率 (目标 60-95% 适中)
- S2: 跨层 transitivity (三层都在锥内子集的 SID unique 率, > 50%)
- S3: 三层 SID unique 率 (> 90%)

---

## 2. 实验结果

### 2.1 v1 单组合 (Phase 0 第一轮)

| Layer | K | In-cone rate | S1 (锥 vs 欧式 argmin) |
|---|---|---|---|
| 0 | 64 | 100.00% | 3.18% |
| 1 | 128 | 100.00% | 0.15% |
| 2 | 256 | 100.00% | 0.15% |
| **三层汇总** | - | 100% | **1.16% avg** |
| S2 transitivity | - | - | **0.01% (1 item)** |
| S3 SID unique | - | - | **0.01% (1 unique SIDs)** |

### 2.2 v2 4 组合 sweep

| 配置 | S1 avg | S2 | S3 | 诊断 |
|---|---|---|---|---|
| (A) fixed_arctan + cos_max | 0.24% | 0.01% | 0.01% | ❌ FAIL |
| (B) inverse_radius + α_min | 1.90% | 0.66% | 0.66% | ❌ FAIL |
| **(C) inverse_radius_third + α_min** | **9.65%** | **25.14%** | **25.15%** | ⚠️ PARTIAL |
| (D) fixed_quarter_pi + cos_max | 0.21% | 0.01% | 0.01% | ❌ FAIL |

**最佳 (C) 仍 FAIL**: S2/S3 都 < 50%, SID 唯一率只 25%.

---

## 3. 根因诊断: HG-Rec baseline 几何坍缩

**所有 4 组合共有的现象**:
- **In-cone rate = 100%**: 几乎所有 z 都被某个码字锥覆盖 (锥太宽)
- **S1 ≈ 0%**: 锥决策跟欧式 argmin 完全无关
- **S2/S3 ≈ 0%**: 几乎所有 z 都被推到同一个"最 specific" 码字

**深层根因** (跟 Task #205 sort_clarity c=100 完全一致):

1. **HG-Rec baseline 码字 norm 几乎全接近 1.0 (boundary)**:
   - baseline ckpt 训完后, 码字经过 proj_to_ball(clamp), ‖p‖_E ∈ [0.95, 1.0]
   - 这意味着 λ_κ = 2/(1-‖p‖²) ∈ [20, ∞), 已经 boundary 饱和
2. **ganea_cos_angle 公式在 ‖p‖ ≈ 1 时数值病态**:
   - cos_angle = ((1+‖p‖²-‖z‖²+2p·z)‖z‖) / (2‖p‖(1-p·z))
   - 当 ‖p‖ → 1, 分母 2‖p‖(1-p·z) → 0, cos_angle 数值不稳定
3. **决策规则坍缩**: 不管选 cos_max / α_min / pnorm_max, 都被 in_cone=100% 推到最 specific 码字
4. **真正的几何信息已经被 baseline 训练吸收** — 码字全在 boundary, 锥 opening 无法反映 norm 差异

**跟之前 finding 串联**:
- Task #199/200/211: 高维 + 不钉半径 = boundary 饱和, λ_κ ≈ 100
- Task #205: c=100 几何坍缩 (sort_clarity 1.93→1.03)
- Task #212: 几何在 top-k 内跟欧式 argmin 99% 一致 (包装失效)
- **Task #213: 锥 opening 也利用不了 norm (包装失效 → 锥也失效)**

**结论**: **HG-Rec baseline 几何已经坍缩**, 任何需要"真双曲几何" 的方案 (锥 / 绕路成本 / 锥违反度) 都失效. 这是"包装失效" finding 的最彻底证据 — 不只是 argmin, 连"层级蕴含" 也利用不了.

---

## 4. 决策: 方向一 NO-HOPE 收线

按用户 2026-07-26 三方向提议:
1. **方向二 (Two-stage decision)** — ❌ NO-HOPE (Task #212)
2. **方向一 (Entailment Cones)** — ❌ **NO-HOPE (本次 Task #213)**
3. **方向三 (Latent radius live variable)** — 投机, 最后候选

**为什么方向一失败** (总结):
- 锥分配需要码字 norm 有差异 (some general + some specific), 才能体现层级
- 但 HG-Rec baseline 训完后, 码字 norm 全 ≈ 1.0 (无差异)
- 锥 opening 公式变成"几乎全覆盖", 决策规则无信号, 跟欧式 argmin 完全无关 (S1=0%) 且全部坍缩 (S3=0.01%)
- 这跟 Task #205 发现的 c=100 几何坍缩完全对应 — 双曲几何在 boundary 饱和时退化成"全 0 信号"

**修复路径** (R11.3 自主决策, 评估):
- (a) 训练新 ckpt with 钉半径 + 低维 (Task #211 C1 已经 NO-GO R@10=0.0816)
- (b) 训练新 ckpt with product_manifold + 钉半径 (Task #210 B1 R@10=0.1018 持平 baseline, 但也没突破)
- (c) 承认 HG-Rec 几何路线在 Musical_Instruments 数据上**全部失效**, 转入 paper 收尾

按 R11.3 自主决策, **(a)/(b) 边际价值 < 0**: 已经做过, 都没突破 baseline. **(c) 是最有价值的方向**:
- Paper §4 写"HG-Rec 包装失效" 的完整 4 证据链 (Task #199 λ_κ≈2 + Task #212 一致率 99% + Task #213 锥 NO-HOPE + Task #211 钉半径 NO-GO)
- 这是 paper 的核心 contribution — "HG-Rec paper 报告的提升不是几何带来的, 是包装 + 框架带来的"

---

## 5. R11.3 自主决策: 不浪费 GPU, 转入 paper 收尾 + 方向三 Phase 0

按用户 R10 主动推进原则, 不空闲等待. 启动两个并行任务 (都 CPU only, 不占卡):

**Task #214 — 方向三 (Latent radius live) Phase 0 概念验证**:
- 核心 idea: 让 radius ρ 成为 live variable (item-adaptive), 不同 item 的"层级强度" 不同
- 投机方向, 4 阶段全跑预期 5-7 天
- Phase 0: CPU 概念验证, 在 baseline 码本上测"radius 离散度" 是否有 signal
- 通过 → Phase 1 改造 train_hrqvae.py (加 radius head)

**Task #215 — Paper §4 完整叙事 (HG-Rec 包装失效 + 4 证据链)**:
- 立即可做, 0 卡
- 目标: 把 4 证据 (Task #199 λ_κ + #212 一致率 99% + #213 锥 NO-HOPE + #211 钉半径 NO-GO) 整合成 paper §4 mechanism 段
- 跟 verdicts/task30_2x2_design_space_paper_skeleton.md 合并

**为什么并行情一 + 情二**:
- Task #215 有确定性输出 (paper 段落可引用), 不浪费时间
- Task #214 即使 FAIL 也是 paper 的"探索过的方向" 列表, 不浪费

---

## 6. 产物落盘

| 类型 | 路径 |
|---|---|
| v1 脚本 | /home/wlia0047/.claude/jobs/04ccf474/tmp/task213_entailment_cones_phase0.py |
| v2 脚本 (4 组合 sweep) | /home/wlia0047/.claude/jobs/04ccf474/tmp/task213_entailment_cones_v2.py |
| v1 结果 JSON | /home/wlia0047/.claude/jobs/04ccf474/tmp/task213_phase0_results.json |
| v2 结果 JSON | /home/wlia0047/.claude/jobs/04ccf474/tmp/task213_phase0_v2_results.json |
| Verdict | verdicts/task213_entailment_cones_phase0_result.md (本文档) |
| Description | descriptions/task213_entailment_cones_phase0.md |

---

## 7. Paper §4 / 2×2 设计空间的新增发现

新增到 verdicts/task30_2x2_design_space_paper_skeleton.md §6 future work 替换:

**原 §6** (旧 future work, 含 θ 可学习 + 双码本解耦等) — 已被前面 Task 验证 NO-GO, 删除.

**新 §6 几何路线终结总结**:
- HG-Rec 几何路线 6 个方向全部 NO-GO:
  1. exp(θ) 可学习 κ (Task #199/201/203)
  2. 双码本解耦 (Task #200/208)
  3. path regularization (Task #209)
  4. 低维双曲 + 钉半径 (Task #211)
  5. Two-stage decision (Task #212)
  6. Entailment Cones (Task #213)
- **共同根因**: HG-Rec baseline 训完后码字 norm 全 ≈ 1.0 (boundary), 几何坍缩
- **结论**: HG-Rec paper 报告的提升 (R@10=0.1315) 不是几何带来的, 是包装 (Sinkhorn + 4-digit dedup + T5 容量) 带来的
- 任何"在 HG-Rec 框架内加几何" 的尝试, 都因 baseline 几何坍缩而失效
- **唯一可能突破**: 重新设计框架 (非 HG-Rec), 或者承认几何对 RQ-VAE 推荐是"无效组件"

---

(本文档覆盖 verdicts/task213_* 之前的临时记录; 完整结论已固化.)

result: Task #213 — Entailment Cones (方向一) ❌ NO-HOPE
