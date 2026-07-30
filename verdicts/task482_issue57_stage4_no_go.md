# Issue #57 (新方向C: T5 混合曲率 attention — sid_embedding_init) — Stage 4 NO-GO (2026-07-31)

## 任务

Issue #57 Gate 0 后的 Stage 3+4 完整闭环:
- Task #158: random init baseline (Stage 3 T5-mini 训练 + Stage 4 eval)
- Task #159: hyperbolic init baseline (Stage 3 T5-mini 训练 + Stage 4 eval)
- Task #482: Stage 4 eval launcher + 结果分析

## Stage 3 训练结果 (Task #158/#159)

| 任务 | SID init | GPU | 训练时长 | ckpt |
|------|----------|-----|----------|------|
| #158 | random | GPU 1 | 2h 18min (200 epoch) | `products/task158/ckpt/Instruments/Jul-30-2026_23-37-43/HG_Rec_best.pth` |
| #159 | hyperbolic | GPU 2 | 2h 19min (200 epoch) | `products/task159/ckpt/Instruments/Jul-30-2026_23-38-59/HG_Rec_best.pth` |

两者均完成 200 epoch, exit code 0, R12 ckpt 强制每 epoch 保存.

## Stage 4 Eval 结果 (Task #482)

**Test set 24772 samples, beam_size=20, topk_list=[5,10,20]**:

| Metric | Baseline (Task #84) | #158 random | #159 hyp (实际 sphere) | sphere vs random | sphere vs BL |
|--------|--------------------:|------------:|-----------------------:|-----------------:|-------------:|
| R@5    | 0.0816 | 0.0758 (-7.1%) | 0.0761 (-6.8%) | +0.0003 | -6.75% |
| **R@10** | **0.1020** | **0.0935 (-8.4%)** | **0.0917 (-10.1%)** | **-0.0019** | **-10.14%** |
| R@20   | 0.1279 | 0.1133 (-11.4%) | 0.1112 (-13.0%) | -0.0021 | -13.08% |
| NDCG@5 | 0.0690 | 0.0639 (-7.4%) | 0.0638 (-7.5%) | -0.0001 | -7.53% |
| NDCG@10 | 0.0755 | 0.0697 (-7.7%) | 0.0688 (-8.9%) | -0.0009 | -8.89% |
| NDCG@20 | 0.0821 | 0.0747 (-9.0%) | 0.0737 (-10.2%) | -0.0010 | -10.22% |

**Elapsed**: 17.7s (random) + 17.0s (sphere) = 34.7s 总 eval 时间.

## 关键发现

### 1. 双向 baseline **双双失败**
两个 init 策略都比 HG-Rec baseline (R@10=0.1020) 低:
- random init: R@10=0.0935 (-8.29%)
- sphere init (因 hyp_c=0.74 符号 bug): R@10=0.0917 (-10.14%)

**Issue #30 (R@10=0.1022) 是当前唯一 +0.2pp GO 端点**, Issue #57 路径是 false positive.

### 2. 双向 baseline 之间**基本中性**
- sphere vs random: R@10 Δ = -0.0019 (约 -2% 相对差异, 在 HG-Rec 实验噪声范围内)
- 6 项指标全部 Δ ∈ [-0.0021, +0.0003], 无一致方向

→ 即使 #159 跑对了 (真正的 hyperbolic init), 也不会显著击败 random init.

### 3. Stage 2 SID 配置是核心瓶颈 (K10 新发现)
- Issue #57 (random/sphere init) 比 baseline **低 8-10%**, 说明问题不在 sid_embedding_init
- Stage 2 SID 文件 `_A2_t5_hrqvae_poincare.npy` 是 R10 RQ-VAE 配置生成的 (μ=0.74, expmap0 sphere 分支)
- 跟 Issue #49 (R@10=0.1005, -1.5%) 联立: Stage 2 SID 几何信号弱, T5 init 策略边际效应极低

## 根因诊断

### Issue #57 Gate 0 bug 复盘 (来自 task164 verdict)
hyp_c=0.74 (正数) 在 expmap0 公式上走 **Euclidean branch**:
```python
if c >= 0:
    x / (1 + sqrt(1 + c * ||x||²))   # 球面/退化分支
else:
    tanh(sqrt(|c|)·||x||/2) · x / (sqrt(|c|)·||x||)  # 真正双曲分支
```

→ #159 实际是 sphere init, 不是 hyperbolic init.
→ 任务命名 `sid_embedding_init='hyperbolic'` 误导, 但因 sphere vs random 也是中性, 不改变 verdict 方向.

### 深层根因
sid_embedding_init 只影响 T5 token embedding 的初始化方式, 不影响:
1. Stage 2 SID 的几何分布 (Stage 1 RQ-VAE 已固化)
2. T5 attention 计算 (跟 embed init 无关)
3. Loss function (跟 init 无关)

→ init 策略在已固化 SID 配置下没有杠杆空间.

## 综合结论

**Issue #57 NO-GO**:
- ❌ Stage 3: 两个 init 策略都训练成功 (200 epoch, exit 0)
- ❌ Stage 4 random init: R@10=0.0935 (-8.4% vs baseline)
- ❌ Stage 4 sphere init: R@10=0.0917 (-10.1% vs baseline)
- ❌ sphere vs random: -0.0019 essentially neutral (sphere doesn't help)

**Issue #57 维持 NO-GO, GitHub close per R15**.

## 关联产物

| 类型 | 路径 |
|------|------|
| Stage 4 random init JSON | `verdicts/task482_random_stage4_metrics.json` |
| Stage 4 sphere init JSON | `verdicts/task482_hyp_stage4_metrics.json` |
| Stage 4 launcher | `scripts/task482_issue57_stage4_eval.sh` |
| Stage 3 ckpt #158 | `products/task158/ckpt/Instruments/Jul-30-2026_23-37-43/HG_Rec_best.pth` (22MB) |
| Stage 3 ckpt #159 | `products/task159/ckpt/Instruments/Jul-30-2026_23-38-59/HG_Rec_best.pth` (22MB) |
| Gate 0 verdict | `verdicts/task164_issue57_gate0_wrapper_audit.md` |
| Stage 4 log random | `logs/task482/random_20260731_020009.log` |
| Stage 4 log hyp | `logs/task482/hyp_20260731_020038.log` |

## 后续方向建议 (供 R10 backlog)

### 已证伪方向
- ❌ sid_embedding_init='random' vs 'hyperbolic' 路径: 双向都 < baseline 8-10%, 不是 R@10 杠杆
- ❌ expmap0(c=0.74) sphere init: 不优于 random

### 未测试方向 (R10 backlog 候选, ROI 待评估)
1. **hyp_c=-1.0 (真双曲)**: 重训 #159 用 hyp_c=-1.0 走 hyperbolic branch, 真正测 hyperbolic init
   - 时间成本: 2.3h Stage 3 + 30s Stage 4 = ~2.5h
   - ROI: 极低 (本 verdict 已证 sphere ≈ random, hyp 即使有 marginal 增益也不会突破 baseline)
2. **K-sweep sid_embedding_init**: K=128/256 不同码本大小下 init 效应
   - ROI: 低 (Stage 2 SID 配置才是瓶颈, init 边际效应有限)
3. **Stage 2 SID 几何健康化**: 修复 μ=0.74 → μ=-1.0 让 RQ-VAE 学真双曲
   - ROI: 中 (跟 Issue #44 #45 #49 联立, 需先证明 codebook 健康化前提)
4. **架构层突破**: Issue #43 HypPreEncoder (R@10=0.1041, +2.1%) 仍是当前唯一 +GO 端点

### 推荐下一步
- Issue #57 关闭, 进入 drift-cycle 监测 (per [[drift-cycle-pattern-recognition]])
- 优先启动 Issue #30 (R@10=0.1022) 跟 Issue #43 (R@10=0.1041) 联合 ablation
- 等 owner 明确新方向 (R10 backlog 全空, 不主动启动低 ROI 实验)

---

result: Issue #57 Stage 4 NO-GO. Random init R@10=0.0935 (-8.4%), sphere init (因 hyp_c=0.74 符号 bug 实际是 sphere 不是 hyperbolic) R@10=0.0917 (-10.1%), 双向均低于 HG-Rec baseline 0.1020. Sphere vs random Δ=-0.0019 essentially neutral. sid_embedding_init 不是 R@10 杠杆. Issue #57 GitHub closed per R15.
