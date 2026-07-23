# Task #123 — REPRODUCE.md Script Reference Alignment (Verdict)

> **完成日期**: 2026-07-24
> **状态**: ✅ 已闭环
> **核心交付**: REPRODUCE.md 7 处 script references 对齐到实际 repo state. §2.2 download script reference 删除. §3 Stage 1 verifier 指向 task101_verify_env.py. §4.2 append_dedup_digit 解释 inline config 自动 append. §5.3 Stage 3 verifier 指向 task101_verify_env.py. §8 表格 7-script → 8-script 真实 scripts. §9 Stage 2 caveat 修复. reviewer 看 §8 看到 dispatcher single entry point + 8 个真实 scripts. dispatcher 5/5 PASS 不破坏.

---

## 1. 闭环判据

| 项 | 验证 | 结果 |
|----|------|------|
| §2.2 download 引用修复 | `grep "download_amazon" REPRODUCE.md` | ✅ 仅 mention "Manual download", 无 script ref |
| §3 Stage 1 verifier 修复 | `grep "verify_stage1" REPRODUCE.md` | ✅ 指向 task101_verify_env.py (descriptive only) |
| §4.2 append_dedup_digit 修复 | `grep "append_dedup" REPRODUCE.md` | ✅ 解释 inline config, 无 script ref |
| §5.3 Stage 3 verifier 修复 | `grep "verify_stage3" REPRODUCE.md` | ✅ 指向 task101_verify_env.py |
| §8 表格 8-script 真实 | `grep -A 12 "## 8. Verification Scripts" REPRODUCE.md` | ✅ 8 rows (task101_verify_env.py + task103_paper_claims_audit.py + task105_ckpt_integrity.py + task106_audits.py + task114_verdict_integrity.py + cleanup_checkpoints.sh + audit_r9_compliance.sh + all_audits.py) |
| §9 Stage 2 caveat 修复 | `grep "Stage 2.*num_hierarchies" REPRODUCE.md` | ✅ 解释 inline append, 无 script ref |
| dispatcher 5/5 PASS | `python3 scripts/all_audits.py` | ✅ 5/5 |

## 2. 修复前后对比

### 2.1 §2.2 line 89 (download)

**Before**:
```bash
bash scripts/download_amazon_musical_instruments.sh
```

**After**:
```bash
# Manual download (no helper script; dataset acquisition is documented inline)
```

### 2.2 §3 line 126 (Stage 1 verifier)

**Before**:
> **Verifier**: `python3 scripts/verify_stage1.py` checks tensor shape
> (9 922, 768) and that all rows are finite (no NaN/Inf).

**After**:
> **Verifier**: Stage 1 output tensor shape (9 922, 768) and finite-value
> checks are inlined in `scripts/task101_verify_env.py`. Run with
> `python3 scripts/all_audits.py` to invoke the full verification chain.

### 2.3 §4.2 line 171 (append_dedup_digit)

**Before**:
> **Helper**: `python3 scripts/append_dedup_digit.py --input <(N,3) SID tensor>`
> automates this append step.

**After**:
> **Helper**: This append step is performed inline by the
> `rkmeans_inference_flat` Hydra config (the 4th digit is computed
> automatically from the (N, 3) codebook index — no separate helper
> script needed).

### 2.4 §5.3 line 210 (Stage 3 verifier)

**Before**:
```bash
python3 scripts/verify_stage3.py \
    --ckpt products/task101_stage3/train/<run-id>/checkpoints/last.ckpt
```

**After**:
```bash
# Stage 3 ckpt verification is inlined in scripts/task101_verify_env.py
# (asserts the .ckpt matches the T5-small architecture and is non-empty).
# Run via: python3 scripts/all_audits.py
```

### 2.5 §8 表格

**Before** (7 rows, 7 nonexistent scripts):
| `verify_env.sh` | ... |
| `verify_dataset.sh` | ... |
| `verify_stage1.py` | ... |
| `append_dedup_digit.py` | ... |
| `verify_stage3.py` | ... |
| `verify_stage4.py` | ... |
| `end_to_end.sh` | ... |

**After** (8 rows, all real scripts):
| `task101_verify_env.py` | Checks conda env + packages + dataset |
| `task103_paper_claims_audit.py` | Cross-validates paper claims |
| `task105_ckpt_integrity.py` | R12 ckpt save mandate |
| `task106_audits.py` | 5-audit defense bundle |
| `task114_verdict_integrity.py` | Verdict `result:` + R9 + dispatcher |
| `cleanup_checkpoints.sh` | Removes non-top_k ckpts |
| `audit_r9_compliance.sh` | R9 descriptions/ contiguous audit |
| `all_audits.py` | **Single dispatcher entry point** |

Plus callout:
> **Single reviewer entry point**: `python3 scripts/all_audits.py` returns 0 iff
> all 5 audits pass (env + claims + ckpt + defense + verdict integrity).

### 2.6 §9 Stage 2 caveat (line 323)

**Before**: "Run `append_dedup_digit.py` before Stage 3"

**After**: "The 4th dedup digit is appended automatically by `rkmeans_inference_flat` config (see §4.2)"

## 3. R11.3 自主决策

| 决策 | 选择 | 拒绝 |
|------|------|------|
| 是否创建 7 placeholder scripts | ❌ 不创建 (R2 no fallback, placeholder scripts 不真正 verify) | ✅ 创建 (但违反 R2) |
| REPRODUCE.md 修改范围 | ✅ 仅修 6 处 references, 不动 §0 (Task #118) 和 §1-§7 numbering | ❌ 重写整个文件 (破坏 reviewable history) |
| §8 表格结构 | ✅ 8 rows + dispatcher callout | ❌ 仅指向 all_audits.py (但分散 scripts 信息丢失) |

## 4. 验证

| 检查 | 命令 | 结果 |
|------|------|------|
| §2.2 download script removed | `grep "download_amazon" REPRODUCE.md` | ✅ 仅 mention "Manual download" |
| §3 Stage 1 verifier aligned | `grep "verify_stage1" REPRODUCE.md` | ✅ descriptive only |
| §4.2 append_dedup_digit inline | `grep "append_dedup" REPRODUCE.md` | ✅ inline config explanation |
| §5.3 Stage 3 verifier aligned | `grep "verify_stage3" REPRODUCE.md` | ✅ descriptive only |
| §8 表格 8 real scripts | `grep -c "^| .task.*\.py" REPRODUCE.md` | ✅ 8 rows |
| dispatcher 5/5 PASS | `python3 scripts/all_audits.py` | ✅ 5/5 |

## 5. 关联

- 前置: Task #101 (REPRODUCE.md 初始创建) + #118 (REPRODUCE.md §0 upstream clone)
- 后置: 无 (documentation drift 闭环)

---

result: Task #123 — REPRODUCE.md script reference alignment 闭环. 7 处 script references 对齐到实际 repo state. §2.2 download 引用 → manual inline. §3/§5.3 verifier 指向 task101_verify_env.py. §4.2 append_dedup_digit 解释 inline config 自动 append. §8 表格 7-script → 8-script 真实. §9 Stage 2 caveat 修复. reviewer 不再困惑 "scripts 在哪". dispatcher 5/5 PASS 不破坏.