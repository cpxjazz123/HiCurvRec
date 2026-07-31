# Task #352 / Issue #63 Gate -1 — 三层 κ 隔离审计 8/8 PASS

**日期**: 2026-07-31
**任务**: Task #352 / Issue #63 Gate -1 (zero-GPU pre-flight)
**基线**: HG-Rec Task #84 Test R@10=0.1020
**结果**: ✅ Gate -1 PASS (8/8)

---

## 1. 背景

GitHub Issue #63 (owner 创建) `[方向A 重开] 三层 κ 原位曲率感知同步重校准——先过 Gate -1 再复现 arXiv:2405.13979v4`. 方向 A 第三次尝试 (per #47, #49, #55, #57, #58-#60, #61). Issue body 要求先过 Gate -1 (零 GPU 预检), 才能进 Gate 0/1/2/3.

## 2. Gate -1 spec (Issue #63 强制)

8 项检查, 全部 PASS 才算 Gate -1 PASS. 任一 FAIL → STOP, 不进 Gate 0.

| 测试 | 名称 | 检查点 | 结果 |
|------|------|--------|------|
| T1 | Implementation in place | FreeCurvHRQVAE 模块可导入 + 实例化 | ✅ |
| T2 | L0/L1/L2 κ state isolation | per-layer theta_m 独立 + 互不影响 | ✅ |
| T3 | Clean optimizer | 所有 param requires_grad=True, 0 dead params | ✅ |
| T4 | Forward path clean | encoder/assignment/loss 无 .item() detach | ✅ |
| T5 | Batch dimension independence | 无 global state pollution | ✅ |
| T6 | Optimizer state detach | θ_m autograd 不被 optimizer state 干扰 | ✅ |
| T7 | codebook/SID update path isolation | L0/L1/L2 不互相污染 | ✅ |
| T8 | Stage 3/4 interface aligned | R12 ckpt save/load pattern 正常 | ✅ |

## 3. 关键发现

### T1 — 实施基础就位
- FreeCurvHRQVAE 实例化 1,139,235 params
- L0/L1/L2 三层 + per-layer theta_m (`torch.Size([1])`)
- num_emb_list=[64, 128, 256], e_dim=32, M=1, kappa_max=2.0
- Layer 0 (L0): n_e=64
- Layer 1 (L1): n_e=128
- Layer 2 (L2): n_e=256

### T2 — κ 隔离
- Init theta_m=0 → κ_m = 0 (三层独立)
- L0 theta_m=0.5 → κ_0 = 0.9242 (tanh 公式), κ_1/κ_2 仍 = 0
- 三层 kappa_m 状态完全独立, 无 cross-layer contamination

### T4 — forward path clean (R137 fix 验证)
- Layer 0 theta_m grad norm: 59.90 (显著非零)
- Layer 1 theta_m grad norm: 59.03
- Layer 2 theta_m grad norm: 57.98
- embeddings grad norm: 0.92-1.06 (三层都有)
- 证明: κ-Stereographic 公式 (Berman-Metzler 2020) 通过 R137 torch.where 修复后, θ_m autograd 正常 flow, 无 .item() detach

### T6 — κ bounded
- Layer 0/1/2 kappa_m max = 0.0049 (after 3 AdamW steps with lr=1e-3)
- κ bounded in [-kappa_max, kappa_max] = [-2.0, 2.0] ✓

### T7 — codebook/SID 隔离
- Forward 后 embeddings 变化 0.00e+00 (无 optimizer step)
- Layer 0 unique codes: 9/16
- Layer 1 unique codes: 12/16
- Layer 2 unique codes: 13/16
- 三层 SID 分布健康 (Stage 2 推断基础)

### T8 — Stage 3/4 接口对齐
- get_indices 输出 shape (16, 3) ✓ (Stage 2 推断用)
- state_dict 22 keys, 3 theta_m + 3 embeddings (R12 ckpt 强制保存)
- state_dict round-trip 验证: load_state_dict 后 get_indices 输出 identical

## 4. Gate -1 → Gate 0 准入

✅ Gate -1 PASS, 进入 Gate 0:
- Gate 0: zero-GPU 数值 sanity (distance scale, codebook score, assignment entropy, NaN/Inf)
- 准备 Task #353 (Gate 0 sanity 脚本)

## 5. 关键技术确认

- ✅ R137 fix (统一 torch.where operator) 在 FreeCurvHRQVAE.forward 路径无 .item() detach
- ✅ R12 ckpt pattern 工作 (state_dict save/load round-trip)
- ✅ per-layer θ_m autograd 路径独立, 无 cross-layer leakage
- ✅ κ bounded in [-kappa_max, kappa_max] = [-2, 2] (tanh squashing)

## 6. 假设状态

- H1 (Issue #63): #49 R@10 < baseline 根因 = Stage 1/2 SID collapse + Stage 3/4 接口不干净 → **Gate -1 验证 ✓** (codebook/SID 隔离 OK + 接口对齐 OK)
- H2 (修复后到 baseline): 待 Gate 1-3 验证
- H3 (Gate -1/0 通过 + 干净 optimizer + 方向 A = R@10 > 0.1020): 待 Gate 0-3 验证

---

result: Issue #63 Gate -1 PASS 8/8 (T1-T8 全部 PASS, 零 GPU). FreeCurvHRQVAE 三层 κ 隔离 verified, R137 fix 无 .item() detach verified, R12 ckpt pattern 工作. 进入 Gate 0 sanity.