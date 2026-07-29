# Task #307 / Issue #34 / D9 — Gate 1 FAIL (e) + H3 反证闸门 falsified

**日期**: 2026-07-30
**状态**: ❌ **Gate 1 FAIL** — 6 条 Gate 1 条件 (e) norm 健康区 FAIL
**关键决策**: Issue #34 新增的 H3 反证闸门 (norm 健康区) 跟 task301 Issue #30 GO 配置 (r_l=[0.1, 1, 10] + s_l=[2, 2, 2]) 实证矛盾 → H3 假设 falsified
**建议**: 修订 Issue #34 Gate 1 (e) 条件 (按实证, norm 0.11-0.26 是 GO 基线), 或承认 H3 假设不成立, 重新审视"

---

## 1. Gate 1 训练结果 (实测)

| 指标 | 实测 | 阈值 | 判定 |
|------|------|------|------|
| (a) L0 utilization ≥ 90% @ ep50+ | **100%** (64/64) | ≥ 90% | ✅ **PASS** |
| (b) L1 utilization ≥ 90% @ ep50+ | **100%** (128/128) | ≥ 90% | ✅ **PASS** |
| (c) L2 utilization ≥ 90% @ ep50+ | **100%** (256/256) | ≥ 90% | ✅ **PASS** |
| (d) collision_rate ≤ 0.25 | **0.0901** (best) / 0.1240 (final) | ≤ 0.25 | ✅ **PASS** |
| (e) norm 健康区 ‖x‖_E ∈ [0.7, 0.95] | **mean L0=0.26, L1=0.14, L2=0.11** | ∈ [0.7, 0.95] | ❌ **FAIL** |
| (f) hash candidates 有效性 | L0=64/64 unique, L1=128/128, L2=256/256 | PASS by construction | ✅ **PASS** |
| loss finite | **35.63** (best_loss) | finite | ✅ **PASS** |

**Gate 1 总判定**: ❌ **FAIL** (e) norm 健康区 FAIL, 硬停止.

---

## 2. H3 反证闸门 falsified (核心发现)

### 2.1 Issue #34 Gate 1 (e) 设计假设

Gate 1 (e) 设计基于 H3 假设: "norm 健康区 ‖x‖_E ∈ [0.7, 0.95] is required for downstream R@10 to be > 0.1022 baseline".

这个假设来自 history:
- Issue #10 Phase 0 fix (task178/180): c=1.0 baseline ‖x‖_E ≈ 0.85 → codebook 健康区
- c=10 baseline ‖x‖_E ≈ 0.27 → norm 偏离健康区但 R@10 HYPER 复原 (task204)
- Issue #25 task287 联合立判据: "norm 健康区 ≠ R@10 杠杆"

### 2.2 task307 Issue #34 实证反证

| 配置 | norm 健康区 | R@10 | 任务 |
|------|------|------|------|
| r_l=[0.1,1,10] + s_l=[2,2,2] | ❌ FAIL (0.11-0.26) | **0.1022 GO** | task301 Issue #30 |
| r_l=[0.1,1,10] + s_l=[2,2,2] + per-layer hash | ❌ FAIL (0.11-0.26) | TBD (Gate 1 STOP) | **task307 Issue #34** |

**核心反证**: 完全相同的 r_l+s_l 配置, task301 实测 R@10=0.1022 GO, 但 norm 0.11-0.26 远低于 [0.7, 0.95] 健康区阈值. H3 假设 (norm 健康区 → R@10 杠杆) **falsified**.

### 2.3 实证矛盾解析

- norm 健康区是 *paper 设计期望* (HG-Rec paper §3.3 推 Poincaré ball 边界分布)
- per-layer r_l+s_l 直接 reshape codebook norm 到更紧致区 (mean 0.11-0.26)
- 但这个"偏离健康区"的下游 SID 仍能达到 R@10=0.1022 GO (task301 实证)
- 机理: norm < 0.7 → codebook 落在 Poincaré ball 内部 → encoder 输出也落在内部 → VQ argmin 仍 OK → Sinkhorn 仍 OK → SID 唯一性 OK → T5 学到 SID 序列
- **结论**: norm 健康区 ≠ R@10 杠杆, 只是 paper 美学偏好

---

## 3. 关键决策 (R11.5 透明)

### 3.1 决策点 1: 严格按 Issue #34 Gate 1 描述硬停止

**选了**: 严格按 §Gate 1 "任一不满足 → STOP" → Gate 1 FAIL
**备选**: 自主 override (e) FAIL → 继续 Gate 2 (考虑 task301 历史 GO)
**理由**: R11.4 关键决策. Issue #34 body 是用户/owner 决策的工程 hard rule. 即使 (e) 是 falsified 假设, 也不能擅自 override issue body hard stop. R11.5 兜底顺序 (1) 项目 CLAUDE.md / memory 已固化的偏好 优先于 (2) 上游 framework 默认值. Issue #34 Gate 1 (e) 是 issue body 显式 hard stop, 优先级最高.

### 3.2 决策点 2: 是否修订 Issue #34 Gate 1 (e)

**选了**: 不修订, 让 verdict 记录 H3 falsified, 等 owner 决策
**备选**: 直接修订 Issue #34 Gate 1 (e) 删掉 norm 健康区条件
**理由**: R11.4 critical decision. Issue body 修改是 owner 决策, AI 不可擅自删 hard stop. 应在 verdict 中 transparent 暴露 falsified 假设, 让 owner 决定是否修订.

### 3.3 决策点 3: per-layer hash 实现是否保留

**选了**: 保留. PerLayerHashHRQVAE wrapper class 实现 (Gate 0 PASS) + post-train 验证 (Gate 1 (f) PASS) 已落盘 scripts/task307_issue34_gate0_perlayer_hash.py + scripts/task307_issue34_gate1_stage1_train.py + best ckpt
**理由**: 即使 Gate 1 FAIL (e), wrapper 实现仍可复用. task308 修订 Issue #34 后可立即重跑.

---

## 4. 物理产物

- `descriptions/task307_issue34_d9_perlayer_hash.md` ✅ (任务定义 + 5-Gate 协议)
- `scripts/task307_issue34_gate0_perlayer_hash.py` ✅ (Gate 0 PASS, PerLayerHashHRQVAE wrapper class)
- `scripts/task307_issue34_gate1_stage1_train.py` ✅ (in-place transforms, 沿用 #30 模式)
- `scripts/task307_issue34_gate1_stage1_train.sh` ✅ (GPU 0, 100 epoch launcher)
- `scripts/task307_issue34_gate1_postverify.py` ✅ (Gate 1 (f) post-verify)
- `verdicts/task307_issue34_d9_gate0_result.md` ✅ (Gate 0 PASS)
- `verdicts/task307_issue34_d9_gate1_result.md` ✅ (本文件 — Gate 1 FAIL (e))
- `products/task307/hrqvae_issue34_gate1/Jul-30-2026_04-50-48_beta_0.500_codebook_[64,128,256]_sk_0.000/best_loss_model.pth` ✅ (best ckpt, 可复用)
- `logs/task307/stage1_gate1_20260730_045042.log` ✅ (训练日志)

---

## 5. R10 + R11 + R12 audit

- **R7 GPU 不抢卡**: GPU 0 空闲, 启动任务. ✓
- **R9 编号连续**: max+1 = 307 ✓
- **R10 主动推进**: Issue #34 OPEN → 启动 task307 → Gate 0/1 完成 → 硬停止 (e) FAIL → 本 verdict
- **R11.2 owner preference**: 严格按 Issue #34 body hard stop, 不擅自 override
- **R11.5 透明决策**: 3 决策点全部明示 (严格停 / 不修订 / 保留实现)
- **R12 ckpt 强制保存**: best_loss_model.pth 已落盘, save_limit=1
- **R13 禁止 Worktree**: 共享 checkout 直接修改, 未触发
- **R14 Issue 自动监控**: Issue #34 OPEN → 启动 task307 → Gate 1 FAIL → 等 owner 决策

---

## 6. 后续建议 (R11.3 透明, 等 owner 决策)

### 6.1 选项 A: 修订 Issue #34 Gate 1 (e)
- 删除 norm 健康区条件 (H3 falsified)
- task307 → Gate 2/3/4 继续推进
- risk: 仍可能 R@10 ≠ 0.1022 (per-layer hash 本身可能是新变量)

### 6.2 选项 B: 承认 Issue #34 Gate 1 hard stop 有效 → 关闭 Issue #34
- task307 DONE → Issue #34 closed
- 后续实验重新设计 (per-layer hash 走新 Issue)
- risk: per-layer hash 思路被 archived, 错失 D9 验证机会

### 6.3 选项 C: 启动 task308 重新设计 Issue #34 Gate 1
- 修订 (e) 条件 (按 task301 history 实证)
- 启动新 task (task308) 走 Stage 2/3/4
- risk: 占用新任务编号

### 6.4 推荐: 选项 A (按 R11.3 兜底顺序)
- 选择: 选项 A (修订 Issue #34 Gate 1 (e) → task308 继续)
- 理由: task301 实证 R@10=0.1022 GO, per-layer hash 加在这个 GO config 上仍未验证, 应该继续评估 D9 是否对 R@10 有 marginal 提升
- 备选: 选项 C (task308 全新 task)
- 是否不可逆: Issue body 修订是 R11.4 critical decision, 必须 owner 决策

---

result: Task #307 / Issue #34 / D9 Gate 1 **FAIL (e)**. 6 条 Gate 1 条件 5 条 PASS (L0/L1/L2 100% util + collision 0.0901 + hash candidates valid + loss finite). 1 条 FAIL: norm 健康区 ‖x‖_E ∈ [0.7, 0.95] 实证 mean 0.11-0.26 (远低于 0.7 下限). **H3 反证闸门 falsified**: task301 相同配置 (r_l=[0.1, 1, 10] + s_l=[2, 2, 2]) 实测 R@10=0.1022 GO, 证明 norm 健康区 ≠ R@10 杠杆. 严格按 Issue #34 Gate 1 hard stop → FAIL. 后续: 等 owner 决策 (选项 A 修订 Gate 1 (e) / 选项 B Issue #34 closed / 选项 C 启动 task308 重新设计).
