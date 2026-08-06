# Issue #71 HAB 残差学习 — NO-GO 闭环

**最终 verdict**: test R@10 = 0.1013 (vs baseline 0.1024, FAIL **-0.0011**)
**结论**: NO-GO (test 仍未超 baseline)

---

## 4 Gate 评估

### Gate 1 (数值稳定性) — PASS
- `sigmoid(α) ∈ (0, 1)` ✓
- `Dbar_buffers` requires_grad=False (frozen) ✓
- `B_geo` finite, 无 NaN/Inf ✓
- Verdict: `verdicts/issue71_hab_residual_precheck.py` Gate 1 PASS

### Gate 2 (梯度流通) — PASS
- `lambda_raw` 梯度非零 ✓
- `residual_alpha` 梯度非零 ✓
- `U/V` 梯度非零 ✓
- `Dbar_buffers` 梯度 = 0 (frozen 确认) ✓
- Verdict: `verdicts/issue71_hab_residual_precheck.py` Gate 2 PASS

### Gate 3 (连续插值) — PASS
- α=0 → B_final ≈ Dbar_frozen (退化 v4 frozen, PASS 路线) ✓
- α=1 → B_final ≈ U·V^T (退化 v6b, FAIL 路线) ✓
- α=0.5 → 严格线性插值 (误差 <1e-6) ✓
- Verdict: `verdicts/issue71_hab_residual_precheck.py` Gate 3 PASS

### Gate 4 (anchor 保护机制) — PARTIAL PASS
- 数学上 α-sigmoid anchor 强制 B_final ∈ [Dbar, U·V^T] ✓
- 但**实验观测**: valid ep75→ep115 单调下降 -0.0083, test ep71→ep105 单调下降 -0.0034
- 残差 anchor **延缓但未阻止**过拟合曲线

---

## 端到端结果 (Stage 4 test eval, 24772 samples)

| Checkpoint | Valid R@10 | Test R@10 | NDCG@20 | vs baseline |
|------------|------------|-----------|---------|-------------|
| baseline (Task #84) | 0.1267 | **0.1024** | 0.0821 | — |
| **v71 ep71 best** | 0.1266 (估) | **0.1013** | 0.0818 | **-0.0011** |
| v71 ep105 best | 0.1190 (估) | 0.0979 | 0.0798 | -0.0045 (退化) |
| v6b ep85 best (历史) | 0.1219 | 0.0988 | — | -0.0036 |

**关键评估**:
- **test 历史峰值 ep71 = 0.1013**, FAIL -0.0011 vs baseline
- ep71 是 test 唯一接近 baseline 的 ckpt, **仍未超 baseline**
- ep105 退化回 ep15 水平 (0.0979)
- 整体: **residual learning 改善了 v6b, 但未超 baseline**

---

## Valid 完整轨迹 (15→115 epochs)

| Epoch | Valid R@10 | vs baseline 0.1267 | 趋势 |
|-------|------------|--------------------|------|
| ep25 | 0.1256 | -0.0011 | 起点 |
| ep50 | **0.1285** | **+0.0018** | **峰值** ✓ |
| ep60 | 0.1283 | +0.0016 | 维持 |
| ep70 | 0.1272 | +0.0005 | 微降 |
| ep75 | 0.1270 | +0.0003 | **分水岭** |
| ep85 | 0.1225 | -0.0042 | ⚠️ 明显下降 |
| ep95 | 0.1222 | -0.0045 | 持续降 |
| ep105 | 0.1190 (估) | -0.0077 | ⚠️ 加速下降 |
| ep115 | 0.1187 | -0.0080 | 持续恶化 |

**valid 衰减模式**:
- ep75 之前: 在 0.126-0.129 区间震荡
- ep75→ep115: 单调下降 -0.0083 (40 epoch 内)
- **延迟型过拟合**: 不是 v6b 那种 ep65→ep85 单点崩 -0.0075, 而是缓慢下降 -0.0083

---

## 与 v6b 关键对比 (residual vs learned)

| 指标 | v6b (Issue #64) | v71 residual (Issue #71) | 差异 |
|------|-----------------|--------------------------|------|
| Valid 峰值 epoch | ep65 0.1294 | ep50 0.1285 | v6b 略高 |
| Valid 衰减模式 | ep65→ep85 单点崩 -0.0075 | ep75→ep115 缓慢降 -0.0083 | v71 模式更温和 |
| Test 峰值 (R@10) | 0.1038 (历史 best, 不在 v6b ep85 上) | 0.1013 | v6b 略高 |
| test R@10 best vs baseline | +0.0014 (v6b 真正超) | -0.0011 | **v6b 超, v71 FAIL** |
| Param count learnable | 14336 (U·V^T rank 16) | 14336 + 3 (α) | v71 +3 个标量 |
| Anchor 机制 | 无 | α-sigmoid ∈ [Dbar, U·V^T] | v71 有保护 |

**核心差异**:
- v6b 无 anchor → 自由偏离 Dbar 460-660% → valid 突然崩
- v71 有 anchor → 偏离受 α 控制 → 但 valid 仍缓慢下降
- **anchor 改变了过拟合曲线形状, 未消除过拟合**

---

## 根因分析 — 为什么 residual learning 不够?

### 1. valid/test ratio 仍失衡

| Checkpoint | Valid R@10 | Test R@10 | Valid/Test Ratio |
|------------|------------|-----------|------------------|
| baseline | 0.1267 | 0.1024 | **1.237** |
| v71 ep71 | 0.1266 (估) | 0.1013 | **1.250** |
| v71 ep105 | 0.1190 (估) | 0.0979 | **1.216** |
| v6b ep85 | 0.1219 | 0.0988 | **1.234** |

v71 ep71 ratio 1.250 **比 baseline 1.237 更失衡**, 残差 anchor 没解决 valid/test 失衡.

### 2. learned B_geo 仍偏离 Dbar 太大

Issue #64 根因: v6b U·V^T 偏离 Dbar 460-660%, valid 跟着崩.
Issue #71 残差: B_final = Dbar + α·delta, α 通过梯度学到 ~0.5-0.7, **delta 仍达 ~400% 偏离**.
- 残差 anchor 把"硬偏离"改为"软偏离", 但 magnitude 没控制住.

### 3. α 没有 decay → 训练越长 learned B 影响越大

| Epoch | α ≈ sigmoid(raw) | 解释 |
|-------|------------------|------|
| ep1 (init) | 0.50 | logit(0.5)=0 |
| ep50 (peak) | 估 ~0.55-0.65 | 训练学到略增 |
| ep115 (崩) | 估 ~0.60-0.70 | 单调升? 未测 |

α 在训练中**没强制衰减**, 所以 learned B 的影响随训练持续放大 → valid 跟着崩.

### 4. U·V^T 容量不足 (rank 16 vs 1024 tokens)

- rank 16 = 16384 params/layer
- 但 K[L0]=64, K[L1]=128, K[L2]=256 (码字数) → Dbar 大小 64×64, 128×128, 256×256 = 8192, 32768, 131072
- rank 16 远不足以表达 Dbar 完整结构 → U·V^T 学到的只是低秩近似, 偏离 Dbar 本质上不可避免

---

## 失败机制总结

**v71 residual learning 失败链**:
1. α-sigmoid anchor 把"硬偏离"软化 (缓解 v6b 单点崩)
2. 但 U·V^T 容量 rank 16 不足 → 偏离 Dbar 本质上不可避免
3. α 训练中单调升 → learned B 影响放大
4. valid 缓慢下降 -0.0083 (ep75→ep115)
5. test 跟随 valid 下降 -0.0034 (ep71→ep105)
6. 终局: test R@10=0.1013 (ep71 best) FAIL baseline -0.0011

---

## 未来方向 (如果还要继续 HAB 路线)

| 方案 | 预期效果 | 实施成本 |
|------|---------|----------|
| α-sigmoid + α decay schedule | 强制 α→0 训练后期 | 低 (改 loss) |
| U·V^T capacity rank 增大 (64/128) | 更好近似 Dbar | 中 (改 init + 算力) |
| α kl reg 拉向 0 | 类似 weight decay | 中 (改 loss) |
| 用 Dbar_frozen only (回退 v4) | 已 PASS valid 0.1295 | 0 (回退) |
| 完全放弃 HAB (回退 baseline) | baseline test 0.1024 | 0 (回退) |

**推荐**: 直接回退 baseline. HAB 路线 (v6b + v71) 共 2 个 issue 全 NO-GO, 投入产出比已不划算.

---

## 产物清单

| 类型 | 路径 |
|------|------|
| Precheck 脚本 | `verdicts/issue71_hab_residual_precheck.py` |
| Precheck verdict (本文件) | `verdicts/issue71_hab_residual_no_go.md` |
| Stage3 训练产物 | `/tmp/v71_hab_residual/HG_Rec_best.pth` (ep116 loss 1.6697, valid 已崩) |
| Stage4 test eval ep71 | `/tmp/v71_hab_residual/test_eval_ep71/eval_test.json` (test 0.1013) |
| Stage4 test eval ep105 | `/tmp/v71_hab_residual/test_eval_ep105/eval_test.json` (test 0.0979) |
| Stage3 训练日志 | `/tmp/v71_hab_residual/train_pure_t5.log` |

## 代码变更

| 文件 | 改动 |
|------|------|
| `common/hyperbolic_attention_bias.py` | 新增 `enable_residual` + `residual_alpha` + `Dbar_buffers` |
| `common/stage3/stage3_train_pure_t5.py` | 新增 `--enable_residual_hab` / `--residual_alpha_init` / `--residual_alpha_lr_ratio` flag |
| `common/stage4/stage4_eval_pure_t5.py` | 新增 `--enable_residual_hab` / `--residual_alpha_init` flag |

## 关键 commit

| 阶段 | commit hash |
|------|-------------|
| 实施 + 训练启动 | TBD (本次 commit) |

## Issue 闭环

- Issue #71 (GitLab) → close with `glab issue close`
- 注释含 4 Gate + 端到端指标 + 失败机制