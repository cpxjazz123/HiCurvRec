# Gate decision — iter17

## Decision: **NO-GO** (Stage3 hard target not met)

- **Mechanism:** `iter17_riemannian_adam_1_over_c` — per-layer AdamW learning-rate multipliers normalized from `1/(c_l + 1e-3)`; cyclic curvature schedule retained.
- **Stage2:** completed 100,000 steps and exported four-token SIDs (`SID_WIRING_PASS`). MVG and all three pre-registered optimizer direct effects passed. Geometry report is descriptively `ALIGNED` (`H(L1|L0)=5.5939` bits; 23,092 unique three-token SIDs / 24,587; 6.08% three-token collision rate). These Stage2 checks and metrics are descriptive only; Stage2 has no hard gate.
- **Stage3:** successful corrected 4-rank run `Sep-26-2026_02-33-02`; completed 150 epochs and final beam-20 test; `n_eval=57439`.

## Final Stage3 metrics

| Metric | Value |
|---|---:|
| `test_recall@5` | 0.038649697940423756 |
| `test_recall@10` | **0.059071362662999005** |
| `test_ndcg@5` | 0.025837394816762203 |
| `test_ndcg@10` | 0.032386492092076745 |

**Hard target:** strict `test_recall@10 > 0.065` → **FAIL**; shortfall `0.005928637337000995`.

## Comparison and attribution

- iter11: `test_recall@10=0.05976775361688052`; iter17 is lower by `0.000696390953881515`.
- iter16: `test_recall@10=0.05687773115827226`; iter17 is higher by `0.002193631504726745`.
- Agent F attribution: `TRUE_MECHANISM_FAIL`, qualified as this run's downstream target miss, not proof of a causal optimizer effect. See `failure_attribution_iter17.md`.
- No promotion. Continue with a distinct iter18 Stage2 mechanism; require a complete Stage3 result before adoption. Never use Stage2 SID geometry or hitrate@50 as a gate.

## Evidence

- Stage3: `results/stage3_T5Train/curvature_RQ-VAE_iter17/logs/Amazon_2023_Instruments/Sep-26-2026_02-33-02/test_final.json`
- Stage2 direct effects and geometry: `mvg_check_iter17.log`, `sid_geometry_iter17.md`
- Agent reports: `stage3_outcome_iter17.md`, `failure_attribution_iter17.md`

Audit commit: `4375b37da709c4e0052a9f61c81788bde8877ee5`
