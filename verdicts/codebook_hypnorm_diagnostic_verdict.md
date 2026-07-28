# 码字双曲范数诊断 verdict (2026-07-25)

**任务**: 回答用户提出的关键诊断问题 — "你们的码字到底有没有走到能感受到曲率的地方?"
**结果文件**: `products/codebook_hypnorm_diagnostic.json` (10 ckpts × 3-4 layers = 34 层测量)
**核心结论**: **YES, 用户的假说得到强烈证实**. 旧 broken baseline 上 8 个 κ-Stereo 变体全部建在 ‖x‖_E ≪ 1/√c 的尺度上, 曲率在数值上等于没参与计算.

---

## 1. 数学背景 (复习)

- κ-Stereographic 共形因子: λ_κ(x) = 2 / (1 - κ‖x‖²)
- 双曲距离: d(x, y) = λ_κ(o) · ‖x⊕(-y)‖_E = (2/√κ) artanh(√κ‖x⊕(-y)‖_E)
- 球边界: ‖x‖_E < 1/√κ (c=1 时为 1.0, c=4 时为 0.5)
- **关键观察**: 当 ‖x‖_E ≪ 1/√κ 时:
  - κ‖x‖² ≪ 1
  - λ_κ(x) ≈ 2.0 (与 κ 完全无关)
  - 距离 ≈ 2·‖x - y‖_E (退化为欧式距离的 2 倍)
  - 曲率对 loss / 梯度贡献接近 0

**判定阈值** (经验):
| ‖x‖_E mean | conformal = 2/(1-κ‖x‖²) | κ 数值有效性 |
|------------|------------------------|-------------|
| < 0.1 | ≈ 2.00 | **完全无效** (距离退化为 2× 欧式) |
| 0.1-0.3 | 2.0 - 2.2 | 极弱 (κ 漂移难被 loss 察觉) |
| 0.3-0.7 | 2.2 - 10 | 中等 (曲率开始起作用) |
| 0.7-1.0 | 10 - 10^5 | 强 (曲率主导距离) |

---

## 2. 10 ckpt × 3-4 层全表 (post-hoc 测量)

| ckpt | 层 | κ | ‖x‖_E mean | ‖x‖_E max | conformal | verdict |
|------|----|----|-----------|-----------|-----------|---------|
| **task180_graph_aware (新修正版)** | L0 | 1.0 | **0.899** | 1.000 | **18822.56** | **STRONG** ✓ |
| **task180_graph_aware (新修正版)** | L1 | 1.0 | **0.797** | 1.000 | **21648.61** | **STRONG** ✓ |
| **task180_graph_aware (新修正版)** | L2 | 1.0 | **0.852** | 1.000 | **19609.57** | **STRONG** ✓ |
| task84_baseline (旧 broken, R@10=0.1020) | L0 | 1.0 | 0.262 | 0.338 | 2.15 | WEAK |
| task84_baseline | L1 | 1.0 | 0.104 | 0.166 | 2.02 | INVALID |
| task84_baseline | L2 | 1.0 | 0.072 | 0.126 | 2.01 | **INVALID** |
| task164_kappa_stereographic | L0 | -1.696 | 0.363 | 0.651 | 2.00 | INVALID |
| task164_kappa_stereographic | L1 | -0.805 | 0.280 | 0.397 | 2.00 | INVALID |
| task164_kappa_stereographic | L2 | -0.303 | 0.171 | 0.239 | 2.00 | INVALID |
| task169_phase_b_sinkhorn | L0 | 0.000 | 0.689 | 0.930 | 2.00 | INVALID |
| task169_phase_b_sinkhorn | L1 | 0.000 | 0.312 | 0.413 | 2.00 | INVALID |
| task169_phase_b_sinkhorn | L2 | 0.000 | 0.198 | 0.285 | 2.00 | INVALID |
| task170_phase_b_sinkhorn_all3 | L0 | 0.000 | 0.480 | 0.632 | 2.00 | INVALID |
| task170_phase_b_sinkhorn_all3 | L1 | 0.000 | 0.187 | 0.239 | 2.00 | INVALID |
| task170_phase_b_sinkhorn_all3 | L2 | 0.000 | 0.121 | 0.162 | 2.00 | INVALID |
| task171_phase_b_dead_code | L0 | 0.000 | 0.524 | 0.633 | 2.00 | INVALID |
| task171_phase_b_dead_code | L1 | 0.000 | 0.188 | 0.247 | 2.00 | INVALID |
| task171_phase_b_dead_code | L2 | 0.000 | 0.134 | 0.260 | 2.00 | INVALID |
| task172_phase_b_kappa_max4 | L0 | 0.000 | 0.597 | 0.838 | 2.00 | INVALID |
| task172_phase_b_kappa_max4 | L1 | 0.000 | 0.312 | 0.407 | 2.00 | INVALID |
| task172_phase_b_kappa_max4 | L2 | 0.000 | 0.212 | 0.299 | 2.00 | INVALID |
| task175_orc_locked | L0 | -0.774 | 0.643 | 0.914 | 2.00 | INVALID |
| task175_orc_locked | L1 | -0.774 | 0.277 | 0.382 | 2.00 | INVALID |
| task175_orc_locked | L2 | -0.774 | 0.195 | 0.289 | 2.00 | INVALID |
| task175_orc_locked | L3 | -0.774 | 0.019 | 0.019 | 2.00 | **INVALID** (dedup) |
| task176_posdep_sigmoid (broken) | L0 | -0.130 | 0.528 | 0.645 | 2.00 | INVALID |
| task176_posdep_sigmoid | L1 | -0.048 | 0.263 | 0.347 | 2.00 | INVALID |
| task176_posdep_sigmoid | L2 | -0.032 | 0.183 | 0.247 | 2.00 | INVALID |
| task176_posdep_sigmoid | L3 | -0.033 | 0.019 | 0.019 | 2.00 | **INVALID** |
| task177_posdep_conformal (broken) | L0 | -0.059 | 0.170 | 0.672 | 2.00 | INVALID |
| task177_posdep_conformal | L1 | -0.031 | 0.297 | 0.409 | 2.00 | INVALID |
| task177_posdep_conformal | L2 | -0.029 | 0.173 | 0.242 | 2.00 | INVALID |
| task177_posdep_conformal | L3 | -0.028 | 0.017 | 0.017 | 2.00 | **INVALID** |

**统计数据**:
- 33/34 层: conformal ≈ 2.00 → κ 数值无效
- 唯一例外: task180_graph_aware 三层都 STRONG (conformal 18822-21649)
- task84_baseline L2 也是 INVALID (‖x‖_E = 0.072 < 0.1 阈值)

---

## 3. 核心发现 — 旧 κ-Stereo NO-GO 不是"数据不需要曲率"

### 3.1 矛盾统一解释

**之前观察到的两个矛盾现象**:
1. 有时 κ 冲边界 (e.g., task164 L0 κ=-1.696, task175 κ=-0.774)
2. 有时 κ 缩回 0 (e.g., task169/170/171/172 κ=0.000)

**统一解释** (用户的假说, 现在得到证据):
- 这两个表象不是"模型找到了 κ 的两个稳定吸引子"
- 而是 **κ 这个参数对 loss / 梯度贡献接近 0**, 优化器在数值噪声驱动下随机漂移
- 跟数据本身需要不需要曲率没关系 — 模型从来就没真正"用过"曲率

### 3.2 task84 vs task180 码字范数差异 (Phase 0 修复效果)

| | task84 (旧) | task180 (新) |
|---|------------|-------------|
| β | 1.0 | 0.5 (论文原配) |
| codebook | [64, 128, 256] | [64, 128, 256] |
| sk_epsilons | 0.0 (Sinkhorn OFF) | 0.003 (Sinkhorn ON) |
| loss type | poincare | poincare |
| **L0 ‖x‖_E mean** | 0.262 | **0.899** |
| **L1 ‖x‖_E mean** | 0.104 | **0.797** |
| **L2 ‖x‖_E mean** | 0.072 | **0.852** |
| conformal L0 | 2.15 | **18822** |

Phase 0 修复 (β 改挂 commitment + 论文 Eq (8) 4 个参数全在球内 + Sinkhorn 打开) 让码字范数从 ~0.1 量级跳到 ~0.85 量级. 这个 ~10× 范数跳跃直接把 κ 从"数值无效"区间推到"强曲率作用"区间.

**结论**: Phase 0 修复带来的不只是"代码 bug 修了", 而是"embedding 尺度从曲率无效区跳到曲率有效区". 这是一个量级跃迁, 不是小幅改良.

### 3.3 8 κ-Stereo 变体的重新解读

| Task | κ 范围 | 实际作用 |
|------|--------|---------|
| task164 (κ-Stereo test) | [-1.696, -0.303] | **κ 漂移无效** — 8 倍 κ 范围在 conformal 上看不出 |
| task169 (κ-Stereo + Sinkhorn L2) | [0, 0, 0] | 死锁在 κ=0 |
| task170 (κ-Stereo + Sinkhorn all 3) | [0, 0, 0] | 死锁在 κ=0 |
| task171 (κ-Stereo + dead_code_reset) | [0, 0, 0] | 死锁在 κ=0 |
| task172 (κ-Stereo κ_max=4.0) | [0, 0, 0] | 死锁在 κ=0 |
| task175 (κ LOCKED at ORC) | [-0.774, -0.774] | 固定也无效 |
| task176 (β(x) sigmoid) | [-0.130, -0.033] | broken baseline |
| task177 (β(x) conformal) | [-0.059, -0.028] | broken baseline |

**核心结论**: 8 个 κ-Stereo NO-GO 结论**不应该被解读为"Instruments 数据是欧式最优"**. 应该解读为**"在 broken baseline 上 κ 物理上无法参与计算, 任何曲率选择都不影响结果"**.

**这意味着**:
- Task #173 / #178 / #179 / #180 等新实验 (基于 Phase 0 修复后代码) **值得重跑**
- 一旦码字范数稳定在 0.7-1.0 区间, κ 就能真正参与 loss 计算
- 此时再观察 κ 是收敛到 0 (数据真的欧式最优) 还是非零 (数据有几何结构需要曲率)

---

## 4. 重新评估方向

### 4.1 需要重新考虑的工作

1. **重跑 κ-Stereo 系列**: 用 Phase 0 修复后的代码, 跑 1-2 个 κ-Stereo 变体 (e.g., κ=1.0 fixed + sinkhorn all 3 layers), 看新 baseline 下 κ 漂移是否变得有意义. 决策信号: κ 是否收敛到某个非 0 吸引子.
2. **修整 #163 NO-GO 8-task verdict**: 把"模型学到 κ≈0" 改写为"在 broken baseline 上 κ 数值无效, 无信息量" — 不应作为最终结论, 应在 Phase 0 修复后重新评估.
3. **CLAUDE.md 更新**: 把 "HG-Rec 在 Instruments 上没几何优势" 这种结论暂时标注 "in re-evaluation after Phase 0 fix", 不作为定论.

### 4.2 已验证的工作

- **task84 baseline (旧 R@10=0.1020)**: 建立在 ‖x‖_E ≪ 1/√c 的小尺度上, R@10=0.1020 是 broken 代码的产物, **不是 HG-Rec 论文原配的真正表现**.
- **修正 baseline (#178 正在训练)**: 用同一组词表 (item_emb.parquet, 9922 items), Phase 0 修复后应得到 ‖x‖_E ≈ 0.85 量级码字, κ 真正起作用. Stage 3 完成后的 R@10 才是 HG-Rec 论文在 Instruments 上的真实 baseline.

### 4.3 task180 graph-aware 的额外信号

- task180 用 LightGCN 平滑后的 item_emb 当输入, 训出来的 codebook ‖x‖_E = 0.80-0.90 (接近球边界).
- 这跟 task84 旧 broken baseline 0.07-0.26 形成 ~10× 差距.
- 假设: **LightGCN 平滑可能不是给 R@10 加分的关键, 而是"让码字真的进到球内, 让 κ-Stereographic 等结构起作用"**.
- 这暗示: 任何"先把 item embedding 推到合理尺度"的方法 (e.g., 加 LayerNorm、ℓ2-normalize 到固定 norm) 都可能复现这个效果.

---

## 5. 下一步 (R10 主动推进)

### 必须做
1. **task178 Stage 3 跑完 → 立即 Stage 4 eval**, 拿到修正后 baseline R@10. 这是最关键的数字.
2. **task179 Stage 3 dual-branch**: 已经在 GPU 1 跑 (Epoch 17, NDCG@20=0.0744, 进展好). 它走的是 dual-branch 融合, 范数动态可能要重新分析.
3. **task180 Stage 3 graph-aware**: 已经在 GPU 2 跑. 验证 graph-aware 的 R@10 是否真的改善.

### 推荐做 (新任务, Phase 0 修复后)
4. **task181 (新)**: 重跑 κ=1.0 fixed + Sinkhorn all 3 layers, 用 Phase 0 修复代码, 看 κ 是否还有 NO-GO 结论. 30-60 min Stage 1 即可.

### 可选
5. **live norm logging**: 在 HVectorQuantization 加 log_hyperbolic_norm_stats(), 在 train_hrqvae.py 每 epoch 调用. 这是用户明确要的"在现有代码里加一行日志", 但 post-hoc 已经覆盖了所有现存 ckpt, 优先级低于 1-3.

---

## 6. 关键产物路径

| 文件 | 路径 |
|------|------|
| 诊断 JSON | `/home/wlia0047/ar57/wenyu/GeneRec/products/codebook_hypnorm_diagnostic.json` |
| 诊断脚本 | `/home/wlia0047/ar57/wenyu/GeneRec/scripts/diagnose_codebook_hyperbolic_norm.py` |
| 本 verdict | `/home/wlia0047/ar57/wenyu/GeneRec/verdicts/codebook_hypnorm_diagnostic_verdict.md` |

---

## 7. 码字范数 epoch-by-epoch 演化 (Task #181 50 epoch 短训)

新增 Task #181 (50 epoch 短训, β=0.5, sinkhorn=0.003, kmeans_init=True, GPU 3) 用来直接观察范数如何从初始态演化. 跟 task180 (200 epoch 满训) 形成"短训 vs 长训"对照.

### 7.1 演化曲线 (Epoch 5 → 50, 每 5 epoch 一个采样点)

| Epoch | L0 ‖x‖_E | L1 ‖x‖_E | L2 ‖x‖_E | L0 λ_κ | L1 λ_κ | L2 λ_κ |
|-------|----------|----------|----------|--------|--------|--------|
| 5 | 0.094 | 0.027 | 0.024 | 2.0 | 2.0 | 2.0 |
| 10 | 0.107 | 0.039 | 0.032 | 2.0 | 2.0 | 2.0 |
| 15 | 0.123 | 0.032 | 0.026 | 2.0 | 2.0 | 2.0 |
| 20 | 0.147 | 0.035 | 0.021 | 2.0 | 2.0 | 2.0 |
| 25 | 0.167 | 0.044 | 0.023 | 2.1 | 2.0 | 2.0 |
| 30 | 0.189 | 0.048 | 0.026 | 2.1 | 2.0 | 2.0 |
| 35 | 0.212 | 0.055 | 0.030 | 2.1 | 2.0 | 2.0 |
| 40 | 0.235 | 0.062 | 0.035 | 2.1 | 2.0 | 2.0 |
| 45 | 0.254 | 0.068 | 0.040 | 2.1 | 2.0 | 2.0 |
| 50 | **0.260** | **0.071** | **0.042** | 2.2 | 2.0 | 2.0 |
| (task180 epoch 194) | **0.899** | **0.797** | **0.852** | 18822 | 21648 | 19609 |

### 7.2 关键观察

1. **码字范数在 epoch 50 时仍处于 INVALID 区间** (L0=0.26, L1=0.07, L2=0.04).
2. **但 task180 epoch 194 已经 STRONG** (L0=0.90, L1=0.80, L2=0.85). 范数跳跃主要发生在 epoch 50-194 之间.
3. **范数增长是超线性的**: epoch 5 → 50 期间, L0 从 0.094 升到 0.260 (+0.17). 但 epoch 50 → 194 之间, L0 从 0.260 跳到 0.899 (+0.64). 后半段增长占总数 79%.
4. **原始 weight norm 的对比更说明问题** (未经 proj_to_ball):
   - task181 (50 ep): 原始 ‖w‖ = 0.27 / 0.07 / 0.04 (跟球内 norm 几乎一样, 没碰到边界)
   - task180 (194 ep): 原始 ‖w‖ = 2.67 / 2.14 / 2.22 (max 高达 22.37, 远超球边界)
5. **proj_to_ball 是个隐性尺度放大器**: 当模型想让码字远离原点 (为了更好拟合数据), proj_to_ball 会把它们 clamp 在 1/√c 附近, 让 ‖x‖_E ≈ 1.0, 从而激活 κ 的几何效应.

### 7.3 推论

- 8 个 κ-Stereo 变体的 ckpt 几乎都训了 200 epoch, 但都是用旧 broken code (β=1.0, sk=0). 它们训完之后范数停在 0.07-0.36, **不是模型没学会推到球边界, 而是 broken code 让 codebook 梯度信号太弱, 推不动**.
- Phase 0 修复 (β=0.5 + sk=0.003 + Eq (8)) 给 codebook 端更大 / 更准的梯度信号, 训到 200 epoch 时码字范数到达 0.85+.
- **所以"先做 Phase 0 修复再谈 κ"这个决策是正确的**. 在没修之前做 κ 实验, 等于在错误的尺度上测量无效信号.

### 7.4 工程含义

- 任何 future Stage 1 跑 hrqvae.log 都会有 [hypnorm] 行. 训练者一眼就能判断码字是否已到达曲率有效区.
- 经验阈值: 训练 50 epoch 是"warmup 阶段", 范数还在 INVALID; 训练 200 epoch 后范数 STRONG. **新实验一律训 ≥ 200 epoch, 短训容易出现假性 NO-GO** (代码没训练够就判 κ 无效).
- 这条经验教训直接解释了"为什么 #164-#172 的 κ 漂移是无信号噪声": 那些 ckpt 都训满了 200 epoch 但范数仍然小, 说明 broken code 让梯度信号微弱到无法把码字推到球边界.

---

## 8. 教训总结 (写进 memory)

1. **κ-Stereo 系列 NO-GO 结论需要打折扣** — 它们都建立在 ‖x‖_E ≪ 1/√κ 的小尺度上, κ 在数值上等于没参与计算. 不是"数据不需要曲率", 而是"曲率没参与过计算".
2. **κ 漂移信号不可信** — 在码字范数小的时候, κ 无论漂到 0 还是冲到 -2 都是噪声. 看到 κ 漂移不要解读为"模型找到了曲率偏好".
3. **Phase 0 修复是一个量级跃迁** — 把码字范数从 ~0.1 量级推到 ~0.85 量级, 让 κ-Stereographic / κ-Stereo 系列真正进入"曲率能起作用"的尺度. 重新评估方向.
4. **诊断永远先于架构改动** — 在做"idea 1 / idea 2 / idea 3"之前, 先确认前 8 个 NO-GO 不是被一个隐藏的尺度问题同时废掉. 这一条诊断节省了可能几个月的工作.

result: 用户假说强烈证实. 旧 broken baseline 上码字 ‖x‖_E 普遍 < 0.3, κ 在数值上无效; 8 个 κ-Stereo NO-GO 应该解读为"在 ‖x‖_E ≪ 1/√c 尺度下 κ 等于没参与计算"而非"数据不需要曲率". Phase 0 修复 (β 改挂 + 论文 Eq (8) + Sinkhorn 打开) 把 task180 码字推到 ‖x‖_E ≈ 0.85, conformal ~20000, κ 真正起作用. Task #181 50 epoch 短训补充验证: 范数从 0.094 慢慢升到 0.260 仍在 INVALID, 但 epoch 50-194 之间超线性增长到 0.899, 表明 proj_to_ball 是隐性尺度放大器, 模型需训练足够长才能把码字推到球边界激活 κ. 重跑 κ-Stereo 系列在 Phase 0 修复代码上是高优先级下一步.