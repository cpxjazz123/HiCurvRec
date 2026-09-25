# gate_decision_iter16

## Decision: **NO-GO** (hard target not met)

- **Mechanism**: `iter16_closed_form_layer_curvature` — `c_l = c_base * exp(α z_l)` with `c_base=0.5, α=0.2`, no cyclic, no learnable, no behavior multiplier.
- **closed_form_c_l = [0.663, 0.435, 0.433]** (constant across 100k steps).
- **Stage2**: `global_step=100000`; SID export (unique 22827/24587, hitrate@50=0.9963, H(L1|L0)=5.4549); Agent D **ALIGNED**.
- **Stage3 run**: `Sep-26-2026_00-03-22`, 150 epochs + beam test complete.

## Stage3 test (n_eval=57439)

| Metric | Value |
|--------|-------|
| test_recall@5 | 0.03794 |
| test_recall@10 | **0.05688** |
| test_ndcg@5 | 0.02551 |
| test_ndcg@10 | 0.03159 |

**Hard gate**: `test_recall@10 > 0.065` → **FAIL** (0.05688).

## A/B/C proxy (paper)

| Iter | Mechanism | test_R@10 | Δ vs iter16 |
|------|-----------|-----------|-------------|
| iter11 | learnable `u_l` + cyclic c(t) | **0.0598** | +0.0029 |
| iter8 | mgc fixed c | 0.0595 | +0.0026 |
| iter14 | + behavior `b_l` | 0.0594 | +0.0026 |
| iter15 | frozen residual `u_l` | 0.0582 | +0.0014 |
| **iter16** | **closed-form fixed `c_l`** | **0.0569** | — |

iter16 ranks **lowest** of all 16 iterations.

## Notes

- Stage-1 ceiling is still ~0.06. After 6 mechanism trials (iter11/12/13/14/15/16), **no** Stage-2 modification has lifted the HG-Rec T5 ceiling past 0.060.
- Failure attribution: `TRUE_MECHANISM_FAIL` — see `failure_attribution_iter16.md`.
- Do not PROMOTE; keep **iter11** (0.0598) / **iter8** (0.0595) as references.

- Audit commit: `988103e42fe69ba334c577d2fd0e9a0390c54e2e`
