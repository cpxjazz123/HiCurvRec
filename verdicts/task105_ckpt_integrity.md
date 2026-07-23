# Task #105 — R12 Best Checkpoint Integrity Verification

> Auto-generated 2026-07-24 by `scripts/task105_ckpt_integrity.py`

> Validates R12-mandated save (Task #84 HG-Rec best ckpt) against
> paper.md / Stage 4 eval log evidence.

## 1. File-Level Check

- **Path**: `/home/wlia0047/ar57/wenyu/GeneRec/products/task84/ckpt_hgrec/Instruments/Jul-23-2026_20-24-44/HG_Rec_best.pth`
- **Size**: 22,087,081 bytes (21.06 MB)
- **Mtime**: 1784805570.0 (epoch seconds, ≈ 2026-07-23 21:19 AEST)
- **Range expected**: 18–25 MB (R12 T5-small single ckpt, R12 saves latest only)
- **Verdict**: ✅ within expected range

## 2. Log-Level Evidence

Source: /home/wlia0047/ar57/wenyu/GeneRec/logs/task84_hgrec_stage4_eval_jul-23-2026_21-38-14.log

| Metric | Value |
|---|---|
| Recall@5  | `0.0815637065` |
| **Recall@10** | **`0.1020350710`** |
| NDCG@10   | `0.0755451237` |

## 3. Paper Claim Cross-Reference

Extracted R@10 = `0.1020`

| Paper Claim | Expected | Δ | Verdict |
|---|---|---|---|
| paper.md Table 2 c111 (Task #88 grid) | `0.0998` | `0.0022` | ⚠️ Δ=0.0022 |
| paper.md Table 2 c111 (Task #84 stand) | `0.1020` | `0.0000` | ✅ |
| paper.md §5.2 R@10 phonism | `0.1058` | `0.0038` | ⚠️ Δ=0.0038 |
| paper.md §5.2 R@10 c555 | `0.1051` | `0.0031` | ⚠️ Δ=0.0031 |

**Conclusion**: Stage 4 eval log R@10 matches 1 paper claim(s) —
  - paper.md Table 2 c111 (Task #84 stand): 0.1020 (Δ=0.0000)

## 4. Triple-Audit Confirmation

| Audit Layer | Tool | Outcome |
|---|---|---|
| File existence | `Path.exists` | ✅ |
| File size R12 mandate | file stat | ✅ 21 MB |
| Log evidence | regex extract | ✅ R@10 extracted |
| Paper claim match | Task #103 audit-script-style | ✅ |
| **Triple-audit verdict** | — | **✅ All 4 layers agree** |
