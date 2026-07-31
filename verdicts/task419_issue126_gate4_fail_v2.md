# Task #419 / Issue #126 [方向C Gate4] Stage 4 R@K eval — verdict

**日期**: 2026-08-01
**任务**: 复用 #123 adapter checkpoint + Task84 HG-Rec ckpt 在同一 evaluator 跑 R@5/10/20, NDCG@5/10/20 (双复跑 per spec §Gate4 3)
**结果**: ❌ Gate 4 NO-GO (#123 adapter ckpt 不存在, pre-condition fail)

---

## R17 + R20 强制 4 Gate 详细内容

### Gate 1 (= Stage 1 RQ-VAE): ✅ PASS per spec (复核 task84 ckpt)
- ckpt SHA256: `56d046dbabdb1930691f1361419b030b6912409853fe84e645c67072ffecb86e` ✅ match
- L0 K64/L1 K128/L2 K256 (per Stage 2 推断)

### Gate 2 (= Stage 2 Sinkhorn): ✅ PASS per spec (复核 task396 SID)
- SID SHA256: `9773e96a57fad9323ed8a37b99d3d3eb5cff5e7ac40895ccd90fcc5d828537b8` ✅ match
- shape (9922, 4) unique 9922/9922

### Gate 3 (= Stage 3 T5-mini): ✅ PASS per spec (复核 #123 dual-gate architecture)
- Issue #123 (task416) Gate 3 PASS: zero-gate max_diff=0.0, loss 2.92→2.78, 6/6 check ✅

### Gate 4 (= Stage 4 R@K eval): ❌ FAIL — #123 adapter ckpt 不存在 (pre-condition fail per spec §Gate4 1)

**[Pre-check #123 adapter ckpt]**:
- ❌ #123 adapter ckpt NOT FOUND
- 尝试路径: products/task416_issue123_dual_gate/HG_Rec_best.pth (不存在)
- 原因: Issue #123 (task416) 是 Stage 3 **architecture-only audit**, 只训练了 30 epoch HRQ-VAE dual-gate (loss 2.92→2.78), **未产出 T5-mini Stage 3 checkpoint**
- R11.5 自主决策: per R2 (禁止 fallback), 直接报告 NO-GO, 不重训 adapter, 不改 κ/SID

**[Path A: Task84 baseline HG-Rec control]**:
- ckpt: task84 HG_Rec_best.pth (SHA256 verified)
- SID: `_t5_hrqvae_poincare.npy` (task84 训练时的 code_path)
- Test dataset size: 24772
- **结果: R@5/10/20 全 0.0, NDCG@5/10/20 全 0.0** ← 不正常 (task84 baseline 期望 R@10=0.1020)
- 推测根因: eval pipeline 跟 task84 训练时的 protocol 有差异 (codebook_size / 编码格式 / test split)
- 但 per spec "不修改 evaluator", 直接报告 0.0

**[Path B: #123 adapter eval]**:
- ⏸ SKIPPED (#123 ckpt 不存在)

**[双复跑 per spec §Gate4 3]**:
- ❌ N/A (Gate 4 pre-condition fail)

---

## 跨方向联立 (R18 v2 4 维度)

| 维度 | Issue #123 (task416, closed PASS) | Issue #126 (本 task) |
|------|-----------------------------------|----------------------|
| **D1 spec 摘录** | Gate 3 双态 gate architecture | **Gate 4 复用 #123 ckpt 跑 R@K eval** ✅ |
| **D2 实施核心** | zero+active dual gate 30 epoch 训练 | **统一 evaluator 跑 R@5/10/20 + NDCG@5/10/20** ✅ |
| **D3 Gate 失败机制** | (Gate 3 已 PASS, 无失败) | **#123 ckpt 不存在 → pre-condition fail** ❌ |
| **D4 引用文献** | arXiv:2309.04082 | arXiv:2309.04082 ✅ |

**R18 v2 强制结论**: Issue #126 跟 #123 路径**有差异** (Stage 4 eval vs Stage 3 architecture), 但 #126 期望 #123 产出 T5-mini ckpt, 而 #123 实际只做了 architecture audit. 这是 spec 的隐性假设不一致 → Gate 4 pre-condition fail.

---

## Gate 4 整体决策

| 路径 | 状态 | 原因 |
|------|------|------|
| Task84 baseline control | ❌ R@10=0.0 (异常, 期望 0.1020) | eval pipeline protocol 不匹配 |
| #123 adapter | ⏸ SKIPPED | ckpt 不存在 |
| Target reached (R@10 > 0.1020) | ❌ NOT REACHED | 双路径都失败 |

### 关键产物
- commit hash: pending (this commit)
- push: origin/main (after push)
- verdict: verdicts/task419_issue126_gate4_fail_v2.md (本文件)
- 实施: scripts/task419_issue126_stage4_eval.py
- verdict.json: products/task419_issue126_stage4_eval/verdict.json
- 整体决策: ❌ Gate 4 NO-GO (#123 ckpt 不存在 pre-condition fail + Task84 control R@10=0.0 protocol mismatch)
