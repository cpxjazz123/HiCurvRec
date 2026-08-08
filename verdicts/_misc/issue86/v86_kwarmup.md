# Issue #86 v86 κ_warmup 实施: PARTIAL-GO (FAIL vs v77)

日期: 2026-08-08
Issue: work_items #86
状态: **实施 PASS, 端到端 FAIL (PARTIAL-GO)**

---

## 1. 背景

v77 (Stage1 per-item radius + v74 HAB frozen + v15 capmatch SID) = 当前曲率路线 SOTA, test R@10=0.1080.

v78 (DECOR) test R@10=0.1092 (历史最佳), 三个抗 trap 改动之一: **alpha_warmup_steps=200** —— 训练初期 alpha ≈ 0, 让 T5 学稳定 baseline, 再慢慢引入 DECOR bias.

DECOR BAN 下, **借鉴动态学习 schedule 思想**, 在曲率框架内用 HAB warmup (Issue #71 Phase A 已实现) 替换 alpha_warmup:
- 训练初期 λ_eff=0 (T5 baseline 友好)
- 步 200-600 渐增到 λ_eff=1 (8 epoch 平滑过渡)
- 步 600+ 完全注入 HAB (与 v77 后续一致)

物理动机: Issue #83 已证 L0 偏好 Euclidean, L1/L2 偏好 hyperbolic. 训练初期全 Euclidean-friendly, 慢慢让深层弯曲.

---

## 2. 实验设计 (1 个变体)

| 配置 | v85p baseline | v86 κ_warmup |
|---|---|---|
| Stage1 | hyp_v2 | hyp_v2 (同) |
| Stage2 SID | v15 capmatch | v15 capmatch (同) |
| Stage3 backbone | T5 6+6 d_model=128 | 同 |
| Stage3 LR | cosine | cosine (同) |
| Stage3 epochs | 200 | 200 |
| HAB residual_alpha | -20 | -20 (同) |
| HAB lambda_max | 0.20 | 0.20 (同) |
| **HAB warmup T0/Tw** | **0/0 (立即)** | **200/400 (延迟)** |

---

## 3. 结果

| 版本 | best epoch | valid_R@10 | test_R@10 | ratio |
|---|---|---|---|---|
| v85p (baseline+v15, 无 HAB warmup) | ep174 | 0.1265 | **0.1042** | 1.218 |
| **v86 κ_warmup** | ep76 | 0.1206 | **0.0968** | 1.245 |
| v77 (立即 HAB, per-item radius) | ep95 | 0.1312 | **0.1080** | 1.215 |
| HG-Rec baseline (无 HAB) | - | 0.1267 | 0.1024 | 1.237 |

v86 best epoch ep76 (因 early_stop=20 在 ep101 触发):
- valid_R@10=0.1206 (vs v85p=0.1265, -0.0059; vs v77=0.1312, -0.0106)
- test_R@10=0.0968 (vs v85p=0.1042, **-0.0074**; vs v77=0.1080, **-0.0112**)
- valid/test ratio=1.245 (vs v85p=1.218, **+2.7%** 过拟合加剧; vs v77=1.215, +3.0%)
- HAB λ_eff 全程激活 (ep13 后), λ_eff=[-0.20, 0.20, 0.20]

---

## 4. Gate 评估

| Gate | 判定 | 结果 |
|---|---|---|
| **Gate 1** (Stage2 SID) | v15 capmatch SID (sha=5f8331cc) 与 v85p 一致 | ✅ PASS |
| **Gate 2** (Stage3 训练健康) | ep1-101 全程 loss 单调下降 2.83→2.71, 无 NaN/Inf | ✅ PASS |
| **Gate 3** (HAB 注入健康) | λ_eff 三层全部激活到 ±0.20, U/V 范数稳定 | ✅ PASS |
| **Gate 4** (端到端) | **test_R@10=0.0968 < v77=0.1080 (-0.0112)**, valid/test ratio 1.245 偏过拟合 | ❌ **FAIL** |

**3/4 Gate PASS, Gate 4 FAIL → PARTIAL-GO**.

---

## 5. 根因分析 — 为什么 warmup 反而失败

### 5.1 假设 1: warmup 期间 (ep1-12) T5 学了"无 HAB baseline", 但 λ 渐增后无法回到 v77 性能
- 证据: v85p 无 HAB baseline test=0.1042, v77 (立即 HAB) test=0.1080. HAB 注入是必要 +0.0038.
- v86 ep12 (warmup 末) test=0.0965. λ=1 后 ep25+ test=0.0898→0.0929→0.0967. 渐增后无法恢复 v77 水平.

### 5.2 假设 2: λ_eff 全程激活导致训练后期过度偏向 HAB bias
- 证据: λ_eff=[-0.20, 0.20, 0.20] 与 v77 完全一致. 但 valid/test ratio 1.245 > v77 1.215 (+3%), 说明 HAB 信号过强导致过拟合 valid 分布.
- 推测: 训练初期无 HAB 让 T5 依赖 context embedding; 后期突然注入 HAB bias → T5 在 valid 上学到"几何信号"模式, 但 test 分布不同.

### 5.3 假设 3: warmup 步数选择不当 (T0=200 / Tw=400 ≈ 8 epoch)
- Issue #71 Phase A HAB warmup 原默认值 (T0=0, Tw=0) 表示**立即**激活. v86 改 T0=200 (8 epoch 延迟) + Tw=400 (8 epoch 渐增).
- 经验: T5 通常 5-10 epoch 就能学到强 baseline, 但 HAB bias 是**残差型** (residual_alpha=-20), 立即 vs 延迟影响应该不大. 实际差距显著, 暗示**早期 HAB 注入对 T5 收敛有利**.

### 5.4 综合判断
- **HAB 延迟开启不是 anti-trap, 而是 anti-learn**: 训练初期 HAB 立即注入能引导 T5 在 embedding 空间内对齐几何结构; 延迟注入迫使 T5 先学"无几何" baseline, 后期再适配 HAB 反而损失早期几何对齐信号.
- v78 alpha_warmup 之所以有效: DECOR alpha 控制 embedding 动态混合 (bos_queries), 初期不激活 → 让 T5 自由学; 后期激活 → 引导 attention 重新对齐.
- HAB 不同: HAB 是 **静态 frozen 几何 bias** (Stage2 SID 已训练好, 无动态学习), 立即注入 = 始终对齐; 延迟注入 = 错误让 T5 学无几何 baseline 后强制切换.

---

## 6. NO-GO 判定 (R23 7 信号)

| 信号 | v86 实际 | 判定 |
|---|---|---|
| 1. val_R@10=0 跨 ≥2 checkpoint | ❌ (val=0.0965-0.1209 全程 > 0) | OK |
| 2. loss 不下降 | ❌ (loss 2.83→2.71 单调) | OK |
| 3. val loss 反向 | ❌ (val_loss 单调改善到 best 0.1209) | OK |
| 4. wrapper broken | ❌ (DDP 4 worker 正常, λ_eff 注入健康) | OK |
| 5. ckpt 不存 | ❌ (HG_Rec_best.pth ep81 保存成功) | OK |
| 6. NaN/Inf | ❌ (loss 全程 finite) | OK |
| 7. GPU 100% 但 loss 不变 | ❌ (GPU 34-36%, loss 持续下降) | OK |

7 信号全部 OK, 训练健康. PARTIAL-GO 单纯是端到端 test 性能不足.

---

## 7. 后续路径

### 7.1 v86 (κ_warmup 单独) PARTIAL-GO → 暂时 NO-GO κ_warmup
- 借鉴 v78 alpha_warmup 失败: HAB 是静态 frozen, warmup 反而破坏早期对齐.
- **不再单独尝试 κ_warmup 路径**.

### 7.2 候选下一步 (曲率框架内)

#### 路径 A: v87 = v77 + attn_entropy regularizer (借鉴 v78 attn_entropy)
- 机制: Stage3 加 attn_entropy_loss (仅 HAB 头), 鼓励 attn 分布尖锐化.
- ROI: ★★★ (v78 三个改动之一, 防 hyperbolic 几何信号被 attn 稀释).
- 单独实施 (~30min) + DDP 训练 ~50min.

#### 路径 B: v87' = v77 + Stage1 radius 强化
- 机制: 增大 R_MAX (0.99 → 0.995), sigmoid 斜率提升 (Stage1 per-item 半径分布更广).
- ROI: ★★ (v77 已经用 per-item radius, 强化边际可能小).

#### 路径 C: 跳出 v77 框架 → v88 = v77 + new SID κ 调整
- 机制: 用 Issue #83 拟合的 c_l* (L0=0, L1=2, L2=5) 替换 v15 capmatch κ, 验证端到端.
- ROI: ★★★ (Issue #83 强证据, 但工程复杂度高).

### 7.3 推荐
**路径 A (v87 = v77 + attn_entropy)** 最优 ROI, 借鉴 v78 第二个改动 (防 attn 稀释), 与 HAB 协同最直接.

---

## 8. DECOR BAN

v86 严格在曲率框架内推进:
- 无 --enable_prompt_former
- 无 decor_prompt_former.py
- 无 DECOR alpha gate / candidate bins

仅用现有 HAB warmup 机制 (Issue #71 Phase A), 借鉴 v78 动态 schedule 思想.

---

## 9. 文件清单

- `/home/wlia0047/ar57/wenyu/GeneRec/taskA/_history/issue86_v86_kwarmup/`
  - `train_v86.log` (训练日志)
  - `verdict.json` (best=ep81, loss=2.7338, done=17:28:08)
  - `HG_Rec_best.pth` (Stage3 adapter, ep81 best)
  - `trace.json` (per-epoch 评估记录)
  - `stage4_test_on_best/` (ep76 best test 评估)
- Stage3 主脚本修改: `common/stage3/stage3_train_pure_t5.py` (--hab_warmup_T0/Tw 默认值改为 200/400)

---

## 10. 总结

v86 κ_warmup (借鉴 v78 alpha_warmup) **端到端 PARTIAL-GO**:
- 训练健康 (3/3 PASS)
- test_R@10=0.0968 < v77=0.1080 (-0.0112)
- 根因: HAB 是静态 frozen 几何 bias, 延迟开启破坏早期 T5 几何对齐
- 后续推荐 v87 = v77 + attn_entropy regularizer (借鉴 v78 第二个改动)
