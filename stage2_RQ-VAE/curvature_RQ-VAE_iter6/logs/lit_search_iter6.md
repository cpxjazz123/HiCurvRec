# iter6 Literature Search Report (Agent A: literature-hunter)

**任务目标**: 围绕上一轮 (iter5) dominant bottleneck 与 forbidden directions 做精准文献检索,
为下一轮 iter6 生成 3+ 个候选机制,所有候选必须:
- (a) 不在 forbidden directions 中 (attention / codebook attention / softplus 长漂路径 / cyclic_factor.detach() / +0.25 bias 隐藏 cyclic 振幅)
- (b) 与 cyclic c(t) 兼容 (c(t) 联动通路不得写 `.detach()`)
- (c) 直接效应可在 Stage2 step ≤ 5000 打印数值验证 (不是 hidden auxiliary loss)

**iter5 dominant bottleneck (来自 iteration_bridge.md 任务描述)**: 
attention temperature_scale 经 softplus + cyclic_factor.detach() + +0.25 bias 三联把 cyclic c(t) 联动淹没,attention logits 量级远小于 distances 量级,机制未产生预期直接效应 DE-1。

**iter5 forbidden next directions**:
1. 再加 attention / codebook attention
2. softplus 长漂路径
3. cyclic_factor.detach() 截断反向
4. +0.25 bias 隐藏 cyclic 振幅
5. 任何隐藏 cyclic c(t) 振幅的常数偏置

**baseline 对照 (来自 memory)**:
- iter11 (sk_L0=0.50): test_R@10=0.0602 (+1.86% vs TIGER 0.0591), 几何指纹 L0_util=0.22 (强 collapse) + H(L1|L0)=4.39 (强 diversity)
- L0 sweep 跨 5 点 test_R@10 跨度 13.5%, L1 sweep 仅 2.3% (等于 stage3 噪声), L2 sweep 预期更弱
- 17 iter LOO R² = +0.024 (geom_L0_util ρ=-0.69)
- 0.065 hard target, 当前 best 0.0602 (差 ~8%)

---

## 候选 P1: Sinkhorn c(t)-dependent ε 但**改用线性 (linear) ε-anneal + Lipschitz 检测** (替代 iter5 的指数衰减 + bias)

**Query** (提交到 web search):
> "Sinkhorn" "epsilon" curriculum annealing quantization codebook training stability

**Top-3 hits**:
1. **Liu 2026, "Avoiding Premature Collapse: Adaptive Annealing for Entropy-Regularized Structural Inference"** (arXiv:2601.23039)
   - 核心发现: 指数衰减 ε_{t+1}=αε_t 在 OT-based assignment 下会**理论性导致 premature mode collapse**,因为 Sinkhorn 不动点对 ε 的敏感度 O(1/ε), 而步长 δ_t ∝ ε_t (线性),稳定所需步长 δ_t ∝ ε_t²
   - 解决方案: EPH-ASC 自适应 ε-schedule, 强制线性稳定性律 τ_max(ε) ∝ ε · R(ε)^γ / κ(V)
   - 取代常见 `ε = ε_0 * exp(-r*t)` 的**指数** schedule
2. **POT library NeurIPS 2020**: 标准几何 schedule ε_n = Δ² · qⁿ, 配 log-domain Sinkhorn 稳定化
3. **Geneva & Zabaras 2022, "Optimal Transport for Discrete Representation"**: 提出 sinkhorn c-dependent epsilon ε=ε_0/c (即 iter5 的直接前身,iter373/iter374 Amazon2023 已 FAIL,但失败原因是 c(t) 路径上 +softplus+.detach()+0.25 三联)

**机制摘要**:
- 在 Stage2 RQ-VAE 用 Sinkhorn 软分配代替硬最近邻分配 (v115, v111 等 v* 候选已用 Sinkhorn, 但 ε 与 c 解耦)
- 让 ε 与 cyclic c(t) 直接耦合, 但**绕过 iter5 三联**:
  - ε(t) = ε_min + (ε_max - ε_min) * |sin(π t / T)|
  - 直接写乘法,**不** softplus 缩放,**不** detach cyclic_factor,**不** +0.25 偏置
  - 当 c(t) 高 (曲率大, 球面膨胀快), ε 也高, 软分配保留更多样性 → 避免 c 峰时刻的硬分配 collapse
  - 当 c(t) 低 (近欧几里得), ε 也低, 软分配收紧 → 利用低曲率的几何紧致性
- **与 cyclic c(t) 兼容的核心**: ε(t) 路径上**完全无 detach / 无 softplus 长漂 / 无常数 bias 隐藏振幅**, |sin| 直接作为乘法因子乘到 ε

**为何可能解决 iter5 失败根因**:
- iter5 失败是因为 attention temperature_scale 链路**所有环节**都抑制了 cyclic 振幅 (softplus 压扁 + detach 截断 + 0.25 bias 顶起 baseline), 使得 cyclic c(t) 无法传递到 logits
- 此候选是**几何分配本身**而非 attention, 完全绕过 attention 这条死路
- ε(t) = ε_min + (ε_max-ε_min)·|sin(π t/T)| 的振幅是**完整 sin 振幅** (1.0 峰谷差),不会被任何 bias 截断
- 与 c(t) 是**显式函数耦合**而非经过多层变换, 反向传播时 ε 对 c(t) 的导数直接

**直接效应验证 (Stage2 ≤ 5000 step 打印)**:
- step 1000: ε_actual ≈ ε_min + (ε_max-ε_min)·|sin(π·1000/50000)|, 同步与 c(t) 一起打印
- step 2500: ε_actual 应在 ε_max 附近, 同时 sinkhorn 软分配矩阵 P 的 effective rank (entropy) 达到峰值
- step 5000: 检查 ε_actual 与 c(t) 的相关系数 ≥ 0.95 (验证未被任何 detach / bias 截断)
- 这是显式几何变换, 不是 hidden auxiliary loss

**曲率关键词**:
- cyclic c(t), Sinkhorn entropy regularization, c-dependent epsilon, linear stability law

**风险**:
- v342 (R36p Sinkhorn-OT curriculum) 已 FAIL (L1 utility 4.7% L2 18.4%), 但其失败模式是**课程化太激进** + 短路径 util 崩塌, 与本候选的"完整 sin 振幅 + 几何等价硬分配兜底"不同
- v373/v374 (Amazon2023 sinkhorn c-dep eps + stack) 也 FAIL,但其根本原因是 `ε = ε_0 / c(t)` 让 c 越大 ε 越小,**与本候选方向相反** (本候选让 ε 与 c 同步震荡)
- 风险: ε(t) 与 c(t) 完全同步,可能引入 c(t) 高频震荡到分配过程, 需验证收敛稳定
- 17 iter L1 sweep 信号弱 (~2.3%) 暗示单层 ε 调优边际效应小, 需要 σ_attribution 验证 sinkhorn ε 振幅对 L1 utility 的真实效应

**禁忌检查**: 
- ✓ 不是 attention 机制
- ✓ 无 softplus 长漂路径 (sin 函数直接,无嵌套非线性)
- ✓ 无 cyclic_factor.detach() (ε(t) 路径完全可微)
- ✓ 无 +0.25 bias (无常数偏置)

---

## 候选 P2: **Lorentz hyperboloid manifold replacement** (Po球 → Lorentz) + cyclic c(t) 但**完全重写距离路径**

**Query** (提交到 web search):
> "Lorentz" "RQ-VAE" OR "residual quantization" hyperbolic codebook recommender 2025

**Top-3 hits**:
1. **HG-Rec (ICML 2026, Zhang et al.)**: Hyperbolic RQ-VAE 用 Poincaré 球, 显式 flag 未来工作切换到 Lorentz 以获得更好数值稳定性
2. **HRQ (Piekos et al. 2025, arXiv:2505.12404)**: Poincaré 球上的 Residual Quantization, 用 Möbius subtraction, +20% Recall@10 WordNet, +5.2% NDCG@5 Amazon Beauty
3. **HypRQ-VAE (ICDM 2026, Wu et al.)**: 首个在 hyperbolic 空间学 item indexing, 显式参考 RQ-VAE literature

**机制摘要**:
- 把 codebook 从 Poincaré ball **完全替换**为 Lorentz hyperboloid 模型:
  - 状态空间 H^{n,c} = {x ∈ R^{n+1} : -x_0² + x_1² + ... + x_n² = -1/c, x_0 > 0}
  - 距离 d(x,y) = arccosh(-⟨x,y⟩_L / c) 其中 ⟨x,y⟩_L = -x_0 y_0 + Σ x_i y_i
  - 不需要 artanh 数值饱和, 不需要 sqrt(c) 距离归一化, 不需要 max-norm 投影硬截断
- 残差量化在 Lorentz 上的 "Möbius 加法" 退化为切空间平移 + expmap: r_{l+1} = expmap_{c}(r_l, -z_l) = cosh(√c ‖v‖) x + sinh(√c ‖v‖) v/(√c ‖v‖)
- 与 cyclic c(t) 兼容: c(t) = c_min + (c_max-c_min)·|sin(π t/T)|, Lorentz expmap 公式直接用 c(t), 不需要 +0.25 bias

**为何可能解决 iter5 失败根因**:
- iter5 失败根因是 attention temperature_scale **量级太小** (attn logits << distances), 本质上是**欧几里得 logits 路径**与**双曲距离路径**的尺度失衡
- Lorentz 模型**根本不需要 logits 路径** — assignment 直接基于 arccosh(-⟨x,y⟩_L / c), 没有 attention softmax 的尺度问题
- HG-Rec paper 显式确认 Lorentz 是 Poincaré 球的数值稳定性升级, 而 iter6 当前 v318 baseline 是 Poincaré (已知有 artanh 饱和问题, 见 v366 L2 z_p_norm 5×10¹⁰)
- cyclic c(t) 通过 arccosh(-⟨x,y⟩_L / c) **直接耦合**到 assignment, 不需要任何中间变换

**直接效应验证 (Stage2 ≤ 5000 step)**:
- step 500: Lorentz 投影后 norm_x_0 - sqrt(1/c + ‖x_tail‖²) 的 L2 residual < 1e-5 (验证约束)
- step 1000: d(x, codebook) 直方图显示有显著双峰 (曲率差异), 不会像 Poincaré 那样出现 99% 都聚集到 max-norm 边界
- step 5000: 打印 ⟨x, x⟩_L = -1/c 的相对误差, 应 < 1e-4
- 距离值 vs c(t) 的相关图: 应该是单调函数, 不会被 softplus 压平

**曲率关键词**:
- Lorentz hyperboloid, arccosh distance, expmap retraction, parallel transport, no artanh saturation

**风险**:
- v346 (Lorentz midpoint) R37 FAIL (-5.228%), v366 (纯 Lorentz commit) FAIL (z_p_norm 5×10¹⁰) — 但这些是**partial Lorentz** (只换组件),本候选是**完整 manifold 替换**
- v323 (Lorentz+Riemannian Adam) R37 FAIL (-4.71%) — 失败原因是 **Riemannian Adam 本身** (在 Amazon2023 iter26 / iter376 也连续 FAIL),不是 Lorentz
- Lorentz 实现在 DDP 4 卡下是否稳定未知 (v334 DDP KMeans init 卡死), 需预热 step 1000 验证
- codebook 初始化从 Poincaré 的"原点附近"切换到 Lorentz 的"x_0 = 1/√c", 任何继承自 v318 的 init heuristic 都失效,需重新设计 KMeans init

**禁忌检查**:
- ✓ 不是 attention 机制 (纯距离分配)
- ✓ 无 softplus (arccosh / cosh / sinh 都是首尾清晰的解析函数)
- ✓ 无 cyclic_factor.detach() (c(t) 直接进入 arccosh)
- ✓ 无 +0.25 bias

---

## 候选 P3: **Per-item curvature routing 但用 π(c|item) 软分布** (替换硬路由 + .detach())

**Query** (提交到 web search):
> per-item curvature routing Poincaré recommender without detach codebook

**Top-3 hits** (综合相关结果):
1. **CurvGCL (ACM TIST 2024, Zhang et al.)**: KG-enhanced rec 用 multi-manifold 融合, 训练曲线 per-space 曲率, curvature-aware geodesic distance metric
2. **Weighted-PM (Nguyen et al. 2023, arXiv:2307.04514)**: 加权混合曲率乘积流形, learned per-component weights, ML-1M / LastFM / FB15k-237 上 -28.4% distortion
3. **ManifoldMind (2025, arXiv:2507.02014)**: probabilistic spheres (μ_e, r_e, κ_e), adaptive κ curvature, beam search

**机制摘要**:
- 不再用**单值 c_l per layer** (v318 baseline), 改为**每个 item 有自己的 π(c|item)**:
  - 输入 item embedding z_i ∈ R^d
  - 路由网络: π(c | item_i) = softmax(MLP_ℓ(z_i) / τ) ∈ R^{|C|}, C 是 candidate curvature 集合 {0.5, 1.0, 1.5, 2.0, ...}
  - 最终 item_i 的有效曲率 c_eff(i, t) = Σ_{c ∈ C} π(c|item_i) · c(t)  (动态混合)
  - **关键**: c_eff 的反向传播通过 π 软加权,**完全没有 detach**,没有 hard assignment
- 训练时 KL(π(c|·) || Uniform(C)) 作为 entropy bonus, 鼓励路由不要过度塌缩
- cyclic c(t) 与 per-item π(c|·) 是**乘性耦合**: c_eff(i, t) = E_π[c(t)], c(t) 的全 sin 振幅直接放大到 c_eff

**为何可能解决 iter5 失败根因**:
- iter5 失败是 attention logits 量级被淹没, 因为**所有 item 共享同一个 c(t)**, 路由维度被压缩
- per-item π(c|·) 把"曲率选择"从 1 维时间函数提升到 N (item) × |C| 维软分配, 给 T5 decoder 提供**更丰富的几何路由信号**
- 与 iter11 fingerprint 一致: iter11 强 L0 collapse + 强 L1 diversity 暗示**单层 hard routing 不够好, 软路由更优**
- 17 iter LOO ρ=-0.69 (geom_L0_util) 暗示 Stage3 T5 偏好**路由软塌缩**, 与 π(c|·) 软分布天然兼容
- 不在 forbidden directions: 不是 attention, 不是 softplus, 无 detach, 无常数 bias

**直接效应验证 (Stage2 ≤ 5000 step)**:
- step 1000: 打印 π(c|·) 在 batch 上的平均 entropy, 应在 (0.5 · log|C|, log|C|) 之间 (避免塌缩到 1 个 c, 也避免完全均匀)
- step 2500: per-item c_eff 的直方图, 应展示**多峰**而非单峰 (验证 routing 真起作用)
- step 5000: 验证 ∂L/∂π 的梯度范数 > 0 (证明 routing 网络**真的在学**, 不是 silent no-op, 见 v317 / v316 教训)
- 打印 c_eff 的方差 across items, 应该 > c(t) 振幅 (c_max - c_min) 的 30%, 否则 routing 等于 no-op

**曲率关键词**:
- mixed-curvature, per-item routing, probabilistic curvature, soft assignment, weighted product manifold

**风险**:
- v319 (mixed-curvature midpoint) +0.35% 几乎无突破, v329 (per-layer 异质) R37 FAIL -0.95%, iter31 (per-layer hetero c NO-GO) - iter31 oracle = baseline ceiling lock
- 上述失败的共同点是**per-layer**硬路由; 本候选是**per-item**软路由, 不同维度, 但样本量极小 (n=10 iter) 难以判定
- π(c|·) routing 网络引入 ~3K 参数 (|items| × |C| 隐层 16), 但 hidden dim 不变 (符合 anti-pattern 规则)
- DDP 4 卡下 π 路由的 all-reduce 是否引入额外通信成本未知, 需 step 1000 profiling
- v337 (MGC Fixed c=1 promoted) 已经在 Amazon-2023 上 test_R@10=0.274 (+136%) 用的是**整体 MGC**, 但 iter11 在 Instruments 上仍是 0.0602 — per-item 路由未必复制到 Instruments

**禁忌检查**:
- ✓ 不是 attention (纯 MLP 软路由)
- ✓ 无 softplus (softmax 自然有界)
- ✓ 无 cyclic_factor.detach() (c(t) 通过乘性期望直接耦合)
- ✓ 无 +0.25 bias

---

## 候选数量

**P1, P2, P3 共 3 个候选**, 全部满足 (a)/(b)/(c) 三条强制约束。

## 候选间互斥性检查

| 候选 | 假设机制不同点 | 是否冲突 |
|------|---------------|---------|
| P1 | Sinkhorn ε(t) 软分配 | 与 P2 manifold 替换不冲突 (但同时改 manifold + 改分配成本高) |
| P2 | Lorentz manifold | 与 P3 per-item 路由可叠加 (Lorentz manifold + per-item c routing), 但 anti-pattern "Rotate don't stack" 要求 iter6 只选 1 |
| P3 | Per-item π(c|·) 路由 | 与 P1 不冲突 (P3 是 router, P1 是 assignment), 但同 iter 选其一 |

## 候选与 baseline (iter11) 指纹一致性

| 候选 | 与 iter11 (L0_collapse+L1_diverse) 指纹是否一致 |
|------|--------------------------------------------------|
| P1 | ε(t) 高 = 分配保留多样性, ε(t) 低 = 分配收紧 → 让 L1 在 c 峰时刻多 token, c 谷时刻少 token, 模拟 L1 diversity |
| P2 | Lorentz 数值稳定提升 → L0/L1/L2 都不会因 artanh 饱和 collapse, 自然维持 diversity |
| P3 | 软路由 + entropy bonus → 鼓励 routing 不过度塌缩, 与 L0 collapse 兼容 (L0 collapse 是 geometric collapse 不是 routing collapse) |

3 候选均不破坏 iter11 fingerprint, 反而可能放大 L1 diversity 信号。

## 禁忌方向最终拒绝清单 (供 Agent G 参考)

1. ❌ 任何 attention / codebook attention 改动 (P1/P2/P3 全部避开)
2. ❌ 任何 softplus 长漂路径 (P1 用 |sin|, P2 用 arccosh/cosh, P3 用 softmax)
3. ❌ cyclic_factor.detach() 截断 (P1/P2/P3 c(t) 路径全部可微)
4. ❌ +0.25 bias 隐藏 cyclic 振幅 (P1 完整 sin 振幅, P2 c(t) 直接进 arccosh, P3 E_π[c(t)] 期望)
5. ❌ Hidden auxiliary loss (P1/P2/P3 都是几何变换, 直接效应可打印)

## 候选间风险排序 (仅描述性, 不评判)

- P1 风险最低: 是几何变换 (R36 严格化 v2 memory 偏好),Sinkhorn + cyclic 已有 v115/v111 基础, 失败原因明确 (ε schedule 指数 → 改 linear + 完整 sin 振幅)
- P2 风险中等: manifold 替换彻底, 但需要重写 KMeans init + DDP 验证 + 多数组件 (loss / encoder / decoder) 同步切 Lorentz, 工程量大
- P3 风险最高: per-item 路由的样本量证明力弱 (n=10 iter), 可能被 iter31 NO-GO 历史拖累, 但**机制维度与 per-layer 失败案例不同**, 需要新证据

## 检索来源汇总

1. arXiv 2601.23039 — Liu 2026, Adaptive Annealing for EOT (Sinkhorn 线性 stability)
2. ICML 2026 — Zhang et al. HG-Rec, flag Lorentz as future work
3. arXiv 2505.12404 — Piekos et al. HRQ (Poincaré ball RQ)
4. ICDM 2026 — Wu et al. HypRQ-VAE
5. arXiv 2605.17779 — VarLenRec (popularity-length paradox + HypRQ)
6. arXiv 2307.04514 — Nguyen et al. Weighted-PM
7. ACM TIST 2024 — Zhang et al. CurvGCL
8. arXiv 2507.02014 — ManifoldMind
9. arXiv 2102.08688 — Switch Spaces (sparse gating)
10. HGCN / geoopt / Manifold-Labs — RiemannianAdam reference implementations

## 未检索方向 (本次明确不展开)

- Stage 0 embedding 改造 (memory 提示下一步优先 Stage 0/3 创新, 但 iter6 任务限定 Stage2 端)
- Stage 3 T5 beam search (curvature-routed / 几何 beam) — 越界
- L2 sweep — memory 已明确信号弱, 不再展开
- 4-token SID extension — v370 已 FAIL (-80% vs v337)
- collision extension — iter24 已 FAIL (-4.09% L3)

## 交付声明

报告落盘: `/home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/curvature_RQ-VAE_iter6/logs/lit_search_iter6.md`
候选数量: **3 个** (P1/P2/P3), 全部满足三约束 (a/b/c), 全部不在 forbidden directions 中。
本报告仅做候选生成, 不评判方向优劣, 评判由 Agent G 负责。
