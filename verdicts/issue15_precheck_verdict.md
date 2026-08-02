# Issue #15 Precheck — R18 4 维度对比 + 实施计划

**Issue #15**: [方向B] precheck blocked — Stage3 curvature_meta 常数化 (κ=1.0 / mixing=0) 修复 + 重走 precheck→Gate3→Gate4

**Status**: precheck ❌ BLOCKED (per #15 顶部红线问题: Stage3 curvature_meta 是常数, 违反 fixed-only 曲率 / 混合权重退化固定值红线)

---

## R18 4 维度对比 (vs Issue #13 [方向B 已关 061e4cc] + Issue #14 [方向A 已关 d958f5d])

| 维度 | Issue #13 (方向B, 已关) | Issue #14 (方向A, 已关) | Issue #15 (方向B, 新) | 差异 |
|---|---|---|---|---|
| **D1 spec 摘录** | Step 1-6 + 单 seed 全 eval (Step 7 待派工) | 单 seed 全 eval + κ/codebook sync 取证 | Step 1-6 修复 Stage3 curvature_meta + 重走 precheck→Gate4 | **D1 关键差异**: #15 是**红线修复** (curvature_meta 常数化违反 precheck), 跟 #13 #14 完全不同性质 |
| **D2 实施核心** | 4 个 GPU 跑 4 步 (audit/fix/diag/beam) | 4 GPU full eval 24772 samples | Step 1 改 taskB_stage3_mixed_curv_recontinue L387-389: κ_l → per-layer nn.Parameter (3 layers × 1), mixing_l → learnable (3 layers × 3) | **D2 关键差异**: #15 必须改 training script 源码 + 重训, 不能复用现有 ckpt (curvature_meta 路径已固定) |
| **D3 Gate 1 失败机制** | taskB P2/P4 FAIL → Step 2-3 修复 PASS | 不适用 (评估任务) | **precheck BLOCKED** (Stage3 curvature_meta 常数, 违反 precheck 红线 fixed-only 曲率 / 混合权重退化为固定值) | **D3 关键差异**: #15 命中红线, 必须修源码 + 重训 |
| **D4 引用文献** | arXiv + DOI 多种 (SID design + 乘积流形) | 5 篇 arXiv (SID + test-time scaling) | arXiv 相关性退化, CrossRef 纯微分几何, PubMed 0 | **D4 差异**: #15 文献检索质量不佳, 跟 #13 #14 引用差异显著 |

**D1+D2+D3+D4 都不同**. R18 强制: **必须做实验** (Step 1 改源码 + 数值验证 + 重训, 不是单纯落 verdict).

---

## 失败机制根因分析 (跟 #13 自我复盘一致)

### #13 close 不充分 — 我的诚实复盘
#13 (commit 061e4cc) 在 Step 5 mixing diagnostic 里诚实记录了 `spec_impl_mismatch`:
- Issue #13 spec 期望: 三层独立 learnable κ + 固定双曲/欧氏分量 + 可学习 mixing weights
- 实际实现 (taskB_stage3_mixed_curv_recontinue L387-389): κ_l = torch.ones(B, 3, 1) [constant 1.0], mixing_l = torch.zeros(B, 3, 3) [constant 0]
- learnable: α_logit + curvature_embed + conditioner + first_input_ln

虽然 #13 spec_impl_mismatch 字段已诚实记录, 但 #13 整体仍写 Gate 3 PASS — 这被 #15 owner 判定为 **conditional / not verified**: 协议 4 项虽全 PASS, 但被评估的模型本身不满足 precheck, Gate 判定不成立.

### 红线命中 (per #15 owner 引用)
- **fixed-only 曲率**: curvature_meta κ_l 全部 1.0, 跟固定曲率 baseline 无异
- **混合权重退化为固定值**: mixing_l 全部 0, 没真混合 (固定 0 = 单分支 = 双曲分支)
- 此前 Stage1/Stage2 的 per-layer κ 与 weight_mlp 梯度非零**并不能替代 Stage3 路径的合规性**

### 跟 #13 spec/impl mismatch 区别
#13 spec 写"三层独立 learnable κ" 是 owner期望, 实际没实现. #15 owner 直接把这条作为红线, 必须修.

---

## 实施计划 (6 步, 严格按序)

### Step 1: 修复 Stage3 curvature_meta spec/impl mismatch (本 tick)
- 改 `taskB_stage3_mixed_curv_recontinue.py`:
  - 在 `BoundedWeightedMixedCurvatureConditioner` 类增加 `nn.Parameter`:
    - `self.kappa_logits = nn.Parameter(torch.zeros(3))` — 3 层独立 learnable κ_logit
    - `self.mixing_logits = nn.Parameter(torch.zeros(3, 3))` — 3 层 × 3 mixing weights learnable (固定双曲 + 固定欧氏 + 混合)
  - 加 `get_kappa_per_layer()` 返回 softplus(kappa_logits)
  - 加 `get_mixing_per_layer()` 返回 softmax(mixing_logits, dim=-1) (强制 sum=1)
  - 改 L387-389 用 learnable 参数 (替代 torch.ones + torch.zeros):
    ```python
    kappa_l = model_wrapper.adapter.get_kappa_per_layer().unsqueeze(0).expand(B, -1, -1)  # (B, 3, 1)
    mixing_l = model_wrapper.adapter.get_mixing_per_layer().unsqueeze(0).expand(B, -1, -1)  # (B, 3, 3)
    curvature_meta = torch.cat([kappa_l, mixing_l], dim=-1)  # (B, 3, 4)
    ```
- 数值验证 (前向):
  - 三层 κ 值互不相同 (打破常数 1.0)
  - mixing weights 非全 0 (打破常数 0)
  - requires_grad=True + 梯度非零
- **禁止**: 退化为纯欧氏、删双曲/欧氏分支、固定权重替代 learnable、合并三层 κ 为 global
- py_compile 验证

### Step 2: precheck 重新判定 (本 tick)
- 落 precheck verdict JSON: 三层 κ Parameter 位置 + K 值 + mixing 权重 Parameter + 梯度证据
- precheck 通过 → 进入 Step 3
- precheck 不通过 → 保持开放

### Step 3: Gate1 / Gate2 继承性复核 (本 tick 末或下 tick)
- 确认 gate1_evidence.json sid_sha256 在 Step 1 改动后仍成立
- Stage2 κ 梯度证据继承
- 若 Step 1 改了 SID 产出 → 重取证 (本次预期不改 SID 产出, 仅改 curvature_meta 数值)

### Step 4: Stage3 重训 + Gate3 判定 (下 tick)
- 重训: taskB_stage3_issue193_long_run.py 续训 (从 epoch 50 继续, 或从 0 开始)
- 两套检查表 4+3 项必须逐条 ✅
- protocol_audit 4 项 + val 评估 3 项全 PASS → Gate 3 PASS

### Step 5: Gate4 canary (下 tick)
- 100 样本 canary, R@10 > 0, validity = 100%

### Step 6: Gate4 单 seed 全 eval (后续 tick)
- 4 GPU 并行 24772 samples × beam K=20 (跟 #14 平行)
- 6 指标 + validity + protocol_audit

---

## 跟 #13 #14 关系

- #13 (061e4cc) 是 #15 之前的 Stage3 已关, 但 #15 owner 判定其 Gate 3 conditional / not verified (红线未修)
- #14 (d958f5d) 方向A 已关 (R@10=0.0724 不达标但 issue 自身 PASS)
- #15 是 #13 的红线修复 + 重走 Gate3→Gate4

**已训练 ckpt 重用评估**: 不允许 (现有 ckpt curvature_meta 路径已固定常数 1.0, 改源码后必须重训)

---

## precheck 状态

| 项 | 状态 |
|---|---|
| precheck (Stage 3 curvature_meta 路径) | ❌ BLOCKED (per #15 顶部红线, L387-389 常数化) |
| Stage 1 (data + emb) | ✅ PASS (沿用 gate1_evidence.json, sid_sha256=2dab29) |
| Stage 2 (RQ-VAE + per-layer κ + codebook) | ✅ PASS (per #14 sync 取证 + Stage 2 ckpt 已有) |
| Stage 3 (training) | ❌ BLOCKED (curvature_meta 常数化违反 precheck, 必须修源码 + 重训) |
| Gate 1 状态 | ✅ PASS (sid_sha256=2dab29 沿用) |
| Gate 2 状态 | ✅ PASS (Stage 2 κ 梯度非零, 但 Stage 3 集成路径不通) |
| Gate 3 状态 | ❌ conditional (模型本身不满足 precheck, Gate 判定不成立) |
| Gate 4 状态 | ❌ blocked-no-canary (Step 5 canary 待 Step 1-4 完成) |

**结论**: 4 维度对比 + 红线根因 + 修复路径明确, **precheck blocked 待修**. R19 立即启动 Step 1 源码修改 (本 tick), Step 2-6 后续 tick.

---

## 修复工作量估计

| Step | 范围 | 估计耗时 |
|---|---|---|
| Step 1 源码 + 数值验证 | 改 taskB_stage3_mixed_curv_recontinue.py 加 Parameter, 改 L387-389 调用 | ~15 min |
| Step 2 precheck verdict | 写 precheck JSON 验证 nn.Parameter + grad | ~5 min |
| Step 3 继承复核 | 验证 sid_sha256 不变 + Stage 2 κ 仍 OK | ~5 min |
| Step 4 重训 | taskB_stage3_issue193_long_run 从 0 重训 50 epoch (~2 hours) | ~2 hours |
| Step 4 Gate 3 判定 | 两套检查表 | ~10 min |
| Step 5 canary | 100 样本 canary | ~1 min |
| Step 6 全 eval | 4 GPU 并行 24772 samples | ~25 min |

**本 tick 目标**: Step 1 + Step 2 完成 (源码改 + precheck verdict 落盘). Step 3-6 跨 tick.