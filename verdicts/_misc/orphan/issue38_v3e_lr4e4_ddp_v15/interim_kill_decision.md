# Issue #38 v3e+LR=4e-4+DDP+v15 SID — KILL DDP 决策 @ ep55 (0.1216 best)

**日期**: 2026-08-09 08:56  
**决策**: KILL DDP at ep55 ckpt, run Stage4 eval

## 决策理由

### Issue #38 valid_R@10 轨迹
- ep10=0.1120 → ep15=0.1160 → ep20=0.1157 → ep25=0.1179
- ep30=0.1181 → ep35=0.1199 → ep40=0.1205 → ep45=0.1199
- ep50 (skip) → ep55=**0.1216** ★ → ep60=??? → ep65=0.1210 → ep70=0.1215

### plateau 证据
- ep60-70: 3 evals (60/65/70) 无提升 (best 0.1216)
- Issue #38 估峰 ≈ 0.1216-0.1230 (vs Issue #37 峰 0.1248, v15 < hyp_v2 8-13% valid)

### v15 SID + HAB valid/test 比估
- baseline (vanilla SID) ratio 1.237 (0.1267/0.1024)
- v74/v77 HAB ratio 1.234/1.215
- Issue #38 (v15+HAB) 估 ratio 1.20-1.24
- 估 test_R@10 = 0.1216/1.20~1.24 = **0.098-0.101**

⚠️ **估 test_R@10 < v85h SOTA 0.1042 (-0.003~-0.006)**

### v15 SID vs hyp_v2 SID 真实 test 估
- Issue #37 (hyp_v2 SID) test=0.0980 (失败)
- Issue #38 (v15 SID) 估 test 0.098-0.105 (取决于 ratio)
- v85h (v15 SID + cosine) test=0.1042 ★
- Issue #38 vs v85h 同样 v15 SID, 但无 cosine → 估略低 0.001-0.005

### 决策
1. KILL DDP (PID 3637538) at ep55 ckpt (valid_R@10=0.1216 best)
2. 立即跑 Stage4 eval (Issue #38 ckpt + v15 SID)
3. 拿 test_R@10 验证 — 若 ≥ baseline 0.1024 (PARTIAL-GO), 若 ≥ v85h 0.1042 (GO)
4. Issue 闭环 + commit + close
