## Issue #126 R18+R20+R21 实证闭环 — 4 Gate 详细内容回答 (R17 + R20 + R21 强制)

### Gate 1/2/3 复核 (per spec §复核):
- ✅ Gate 1: task84 ckpt SHA256 `56d046db...86e` match
- ✅ Gate 2: task396 SID SHA256 `9773e96a...7b8` shape (9922,4) unique 9922/9922
- ✅ Gate 3: #123 dual-gate 6/6 check PASS (verdict task416)

### Gate 4 (= Stage 4 R@K eval): ❌ FAIL — #123 adapter ckpt 不存在 (pre-condition fail per spec §Gate4 1)

**[Pre-check #123 adapter ckpt]**:
- ❌ #123 adapter ckpt NOT FOUND
- 原因: Issue #123 (task416) 是 Stage 3 **architecture-only audit**, 30 epoch HRQ-VAE dual-gate, **未产出 T5-mini Stage 3 checkpoint**
- R11.5 自主决策: per R2 (禁止 fallback), 直接 NO-GO, 不重训 adapter

**[Path A: Task84 baseline HG-Rec control]**:
- ckpt: task84 HG_Rec_best.pth (SHA256 verified)
- SID: `_t5_hrqvae_poincare.npy` (task84 训练时的 code_path)
- Test dataset size: 24772
- **结果: R@5/10/20 全 0.0, NDCG@5/10/20 全 0.0** ← 不正常 (期望 R@10=0.1020)
- 推测: eval pipeline protocol 不匹配 (codebook_size / 编码格式 / test split)
- per spec "不修改 evaluator", 直接报告 0.0

**[Path B: #123 adapter eval]**: ⏸ SKIPPED (ckpt 不存在)
**[双复跑 per spec §Gate4 3]**: ❌ N/A (Gate 4 pre-condition fail)

### 跨方向联立 (R18 v2 4 维度)
- vs Issue #123: 路径有差异 (Stage 4 eval vs Stage 3 architecture), 但 #126 期望 #123 产出 T5-mini ckpt, 实际 #123 只做了 architecture audit → spec 隐性假设不一致

### 关键产物
- commit hash: 18f5645
- push: origin/main
- verdict: verdicts/task419_issue126_gate4_fail_v2.md
- 实施: scripts/task419_issue126_stage4_eval.py
- 整体决策: ❌ Gate 4 NO-GO 收口 (#123 ckpt 不存在 pre-condition fail + Task84 control R@10=0.0 protocol mismatch)
