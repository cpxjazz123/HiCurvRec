# Issue #14 Verdict — [方向A] Gate4 单seed Task84 全量评估

**Issue #14**: [方向A] Gate4 单seed Task84 全量评估 — beam K=20 六指标产出 + kappa/codebook 同步性取证

**Status**: ✅ PASS (per #14 spec "无论 R@10 是否达标, 本 issue 即可关闭")

---

## Step 1: 单 seed Task84 test 全量评估 (24772 samples × beam K=20, 4 GPU 并行)

| 项 | 值 |
|---|---|
| ckpt | taskA_stage3_issue192_long_run/best_adapter.pt (epoch 48) |
| n_samples | **24772** (full Task84 test set) |
| GPU split | 0/6193 + 6193/12386 + 12386/18579 + 18579/24772 |
| BEAM_SIZE | 20 |
| seed | 42 |
| **elapsed_sec (max)** | **1528.5s (~25.5 min)** |
| **validity_pct** | **100.0% (all 4 GPUs, total tokens 99,088)** |
| R@5 | **0.0575** |
| **R@10** | **0.0724** ❌ (vs baseline 0.1020, -29%) |
| R@20 | **0.0958** |
| NDCG@5 | **0.0478** |
| NDCG@10 | **0.0525** |
| NDCG@20 | **0.0584** |
| 6 指标互不恒等 | ✅ 6/6 unique values |
| R@K 单调非降 | ✅ R@5=0.0575 ≤ R@10=0.0724 ≤ R@20=0.0958 |
| P1 forward path | ✅ PASS (autoregressive 4-step + layer-wise mask) |
| P2 vocab mapping | ✅ PASS (vocab_size=1025, valid 449 tokens, layer_ranges hash=aacb3085) |
| P3 lm_head path | ✅ PASS (t5.model.lm_head, [1025,128], float32) |
| P4 valid SID constraint | ✅ PASS (validity=100%, layer-wise mask) |
| Verdict | **PASS** (per #14 spec, 不依赖 R@10 达标) |

**4 partial JSON → 1 final verdict**: `verdicts/issue14_full_eval_result.json`

---

## Step 2: κ/codebook 同步性取证

| 项 | 值 |
|---|---|
| per-layer κ 定义 | `taskA/stage2/taskA_stage2_kappa_vq_fix.py:104` — `self.kappa_logit = nn.Parameter(...)`, 3 层 × 1 per layer |
| κ → κ_eff 公式 | `taskA/stage2/taskA_stage2_kappa_sync.py:111-128` — `return 1.0 + self.kappa + 1e-3` |
| poincare_distance c=κ 应用 | `HG-Rec/model/hrqvae_orc_locked.py:63` — `ratio = max(min(k_val / self.kappa_max, 0.9999), -0.9999)` |
| 数值验证 | distance 矩阵依赖 κ (diff > 0.19 across κ ∈ {0.5, 1.0, 2.0}) |
| sync_verified | ✅ True (κ 改变 → distance 改变 → codebook 量化结果必然改变) |
| 落盘 | `verdicts/issue14_kappa_codebook_sync.json` |

---

## Step 3: R@10 = 0.0724 < 0.1020 根因分类 (a)/(b)/(c) + 综合判定

| 类别 | 判定 | 数值证据 |
|---|---|---|
| **(a) 映射/约束层残留不一致** | ✅ 排除 | 4 项 protocol_audit 全 PASS + validity=100% (24772 samples, 99088 tokens) |
| **(b) α clamp 削弱 κ 信号** | ✅ 排除 | α_logit=-1.440, α_value=0.213, α/α_max=0.4251 < 0.95 (未触 clamp 上限). 旁证: 前序 κ 终值 -0.0082 量级极小, 跟 α clamp 关系不大 |
| **(c) Stage3 适配器容量不足** | ⚠️ 结构事实 | T5 主干 frozen ~60M 参数, adapter 仅 132609 参数 (0.22%), conditioner 99.1% (131456) |
| **综合** | 主因: 数据规模 (9922 items) + epoch (200) + α 弱信号 + κ 终值量级极小 (-0.0082) 共同决定上限 | Stage3 P4 修复解决了 decode 约束 (validity 25% → 100%), 但没解决上游信号强度 |

R@10 = 0.0724 (-29% vs baseline) 反映 Stage3 适配器在当前数据 + epoch 下的天然上限. **单 P4 修复无法超越 baseline**.

---

## Issue #14 关闭条件验证

| 条件 | 状态 | 证据 |
|---|---|---|
| 全量 Task84 test 六项指标落盘 | ✅ | R@5/10/20=0.0575/0.0724/0.0958, NDCG@5/10/20=0.0478/0.0525/0.0584 |
| 六项指标互不恒等 | ✅ | 6/6 unique values |
| validity=100% | ✅ | 100.0% (99088/99088 tokens) |
| protocol_audit 4 项 PASS | ✅ | P1/P2/P3/P4 全 PASS |
| κ/codebook 同步性证据 | ✅ | `verdicts/issue14_kappa_codebook_sync.json` (distance diff > 0.19) |
| **Issue 关闭条件** | ✅ **PASS** | per #14 spec "无论 R@10 是否达标, 本 issue 即可关闭" |

---

## 跟 #12 #13 关系

- #12 [方向A Step 1-4] (commit 469c99e + 380228f): 修 P4 + canary + beam K=20 (100 samples)
- #14 (本 issue): #12 Step 5 — 单 seed 全 eval (24772 samples) + sync 取证
- #13 [方向B] (commit 061e4cc): taskB 平行 6 步 (audit + fix + mixing diag + beam)
- #13 Step 7: 单 seed Task84 全 eval 待派工 — **#14 即对应方向A 这部分, 现在完成**

---

## R@10 = 0.0724 不达标后续

按 #14 spec "Target reached（方向A整体）：R@10 > 0.1020" 没达成. 后续:
- 不在本 issue 范围
- 方向A 上限突破需要新 issue (例如: 增加训练 epoch / 增大 κ 信号 / 解冻 T5 部分层 / 数据增强)
- 当前基线 HG-Rec Task #84 R@10=0.1020 仍是参考天花板

---

## 文件清单

| 文件 | 状态 |
|---|---|
| `taskA/stage4/taskA_stage4_full_eval_issue14.py` | 新建 (Step 1 multi-GPU eval 脚本) |
| `taskA/stage4/taskA_stage4_kappa_codebook_sync_issue14.py` | 新建 (Step 2 sync 取证脚本) |
| `taskA/stage4/taskA_stage4_root_cause_issue14_step3.py` | 新建 (Step 3 根因分类脚本) |
| `taskA/stage4/issue14_full_eval_gpu{0,1,2,3}.pid` | 临时 (4 GPU PID, eval 完成后可删) |
| `verdicts/issue14_precheck_verdict.md` | 新建 (R18 4 维度对比) |
| `verdicts/issue14_full_eval_gpu{0,1,2,3}_partial.json` | 新建 (4 GPU partial, 单 GPU 6193 samples) |
| `verdicts/issue14_full_eval_result.json` | 新建 (Step 1 聚合 final) |
| `verdicts/issue14_kappa_codebook_sync.json` | 新建 (Step 2 sync verdict) |
| `verdicts/issue14_root_cause.json` | 新建 (Step 3 根因 verdict) |
| `verdicts/issue14_step1_step2_step3_verdict.md` | 新建 (本文件) |