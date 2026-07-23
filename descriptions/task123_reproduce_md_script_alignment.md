# Task #123 — Align REPRODUCE.md Script References with Actual Repo State

> **任务目的**: REPRODUCE.md references 7 verification scripts (verify_env.sh, verify_dataset.sh, verify_stage1.py, verify_stage3.py, verify_stage4.py, append_dedup_digit.py, end_to_end.sh) + 1 download script (download_amazon_musical_instruments.sh), of which NONE exist as separate files. 实际 verification logic 都在 `scripts/task101_verify_env.py` + `scripts/all_audits.py` (5-audit dispatcher). 文档 vs 现实 drift 修复. Per R2 (no fallback), 修文档不创建 placeholder scripts.

> **完成日期**: (in progress)
> **状态**: 🟡 待启动

---

## 1. 背景

REPRODUCE.md (Task #101 创建 + Task #118 加 §0) 包含 8 处 references to scripts that don't exist:
- §2.2 line 89: `bash scripts/download_amazon_musical_instruments.sh` (下载数据集)
- §3 line 126: `python3 scripts/verify_stage1.py` (Stage 1 验证)
- §4.2 line 171: `python3 scripts/append_dedup_digit.py` (Stage 2 dedup)
- §5.3 line 210: `python3 scripts/verify_stage3.py` (Stage 3 验证)
- §8 line 293: 7 verification scripts 表格 (verify_env.sh, verify_dataset.sh, verify_stage1.py, append_dedup_digit.py, verify_stage3.py, verify_stage4.py, end_to_end.sh)
- §9 row 3 line 323: "Run `append_dedup_digit.py` before Stage 3"

实际情况 (`git ls-files scripts/`):
- ✅ `task101_verify_env.py` — env + dataset + Stage 1 shape 验证 (inlined)
- ✅ `task103_paper_claims_audit.py` — paper claims
- ✅ `task105_ckpt_integrity.py` — R12 ckpt
- ✅ `task106_audits.py` — defense bundle
- ✅ `task114_verdict_integrity.py` — verdict integrity
- ✅ `cleanup_checkpoints.sh` — ckpt cleanup
- ✅ `audit_r9_compliance.sh` — R9 audit
- ✅ `all_audits.py` — single dispatcher (5 audits)

**Task #123 = 文档对齐实际 scripts**

---

## 2. 实验设计 (writeup only)

### 2.1 修复 §2.2 line 89 (download script)

Replace:
```bash
bash scripts/download_amazon_musical_instruments.sh
```
With:
```bash
# Manual download (no helper script; dataset acquisition is documented inline)
```

### 2.2 修复 §3 line 126 (Stage 1 verifier)

Replace `python3 scripts/verify_stage1.py` 引用 → 指向 `task101_verify_env.py` (inlined).

### 2.3 修复 §4.2 line 171 (append_dedup_digit.py)

Replace: helper script 引用 → 解释 append 是 `rkmeans_inference_flat` config 内联 (4th digit 自动从 (N, 3) codebook index 计算).

### 2.4 修复 §5.3 line 210 (Stage 3 verifier)

Replace `python3 scripts/verify_stage3.py` → 指向 `task101_verify_env.py`.

### 2.5 修复 §8 (7-script 表格)

Replace 整个表格 with 8-script 表格 reflecting actual scripts + dispatcher entry point.

### 2.6 修复 §9 row 3 (append_dedup_digit caveat)

Update Stage 2 caveat to reference `rkmeans_inference_flat` config 自动 append.

---

## 3. 决策触发

| 条件 | 决策 |
|------|------|
| §2.2 download 引用修复 | ✅ 闭环 |
| §3 Stage 1 verifier 引用修复 | ✅ 闭环 |
| §4.2 append_dedup_digit 引用修复 | ✅ 闭环 |
| §5.3 Stage 3 verifier 引用修复 | ✅ 闭环 |
| §8 表格 7-script → 8-script 真实 | ✅ 闭环 |
| §9 Stage 2 caveat 修复 | ✅ 闭环 |
| dispatcher 5/5 PASS 不破坏 | ✅ 闭环 |

---

## 4. R11.3 自主决策

| 决策 | 选择 | 拒绝 |
|------|------|------|
| 是否创建 7 placeholder scripts | ❌ 不创建 (R2 no fallback, 不能写空 placeholder) | ✅ 创建 (但 placeholder scripts 不会真正 verify 任何东西) |
| 是否修改 REPRODUCE.md | ✅ 修改 (反映 reality, 让 reviewer 不踩坑) | ❌ 不改 (但 reviewer 困惑 "scripts 在哪") |
| §8 表格结构 | ✅ 8-script 表格 + dispatcher entry point callout | ❌ 仅指向 all_audits.py (但分散 scripts 信息丢失) |
| download script 处理 | ✅ 删除 reference, 注明 "manual download, 文档 inline" | ❌ 创建空 script 占位 |
| append_dedup_digit 处理 | ✅ 解释 inline config 自动 append (无需单独 script) | ❌ 创建占位 script |

---

## 5. 预算

| 阶段 | 估算时间 |
|------|---------|
| grep + Edit REPRODUCE.md 6 处 | ~3 min |
| dispatcher 验证 | ~3 sec |
| git commit + §16 | ~2 min |
| **总计** | **~5 min, 0 GPU** |

---

## 6. 风险与缓解

**风险 1**: 修改 §8 表格可能破坏 Task #101 创建的 REPRODUCE.md 格式
  → **缓解**: Edit 局部内容, 不动 §0 (Task #118 加的) 和 §1-§7 numbering

**风险 2**: 解释 inline append step 可能让 reviewer 困惑 "RKMeans config 怎么 append"
  → **缓解**: 简化解释 "4th digit 自动从 (N, 3) codebook index 计算, 见 rkmeans_inference_flat Hydra config"

---

## 7. 完成度跟踪

- [x] 写 descriptions/task123_reproduce_md_script_alignment.md (本文件)
- [x] 修复 §2.2 line 89 download 引用
- [x] 修复 §3 line 126 Stage 1 verifier 引用
- [x] 修复 §4.2 line 171 append_dedup_digit 引用
- [x] 修复 §5.3 line 210 Stage 3 verifier 引用
- [x] 修复 §8 表格
- [x] 修复 §9 row 3 Stage 2 caveat
- [x] dispatcher 5/5 PASS 验证
- [ ] git commit
- [ ] loop.md §16 更新

---

## 8. 关联

- 前置: Task #101 (REPRODUCE.md 初始创建) + #118 (REPRODUCE.md §0 upstream clone)
- 后置: 无 (documentation drift 闭环)

---

**核心交付**: REPRODUCE.md 7 处 script references 对齐到实际 repo state. reviewer 看 §8 表格看到 8 个真实 scripts + dispatcher entry point, 不再困惑 "scripts 在哪". dispatcher 5/5 PASS 不破坏.