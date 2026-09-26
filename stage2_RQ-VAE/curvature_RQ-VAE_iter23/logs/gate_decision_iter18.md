# Gate decision — iter18

## Decision: **NO-GO** (Stage3 hard target not met)

- **Mechanism:** `iter18_initial_curvature_beta2` — fixed per-layer AdamW `β₂` values derived from initial curvature; cyclic curvature retained; base learning rate, `β₁`, weight decay, and loss configuration unchanged.
- **Stage2:** completed 100,000 steps and exported four-token SIDs (`SID_WIRING_PASS`). The one-checkpoint/one-batch MVG and registered direct-effect checks passed. Geometry metrics are descriptive only: `full_gini=0.0642`, per-layer Gini `[0.1896, 0.2173, 0.2015]`, 22,951 unique three-token SIDs, 14,812 unique `(L0,L1)` pairs, `H(L1|L0)=5.5907` bits, and 1,636 three-token collision rows. No Stage2 metric is a gate; `hitrate@50` is not a gate.
- **Stage3:** run `Sep-26-2026_04-45-57` used the iter18 SID input (`world_size=4`, `n_digit=4`, beam 20), completed all 150 epochs, and evaluated `n_eval=57439`.

## Final Stage3 metrics

| Metric | Value |
|---|---:|
| `test_recall@5` | 0.040460314420515675 |
| `test_recall@10` | **0.05988962203380978** |
| `test_ndcg@5` | 0.02697174940916111 |
| `test_ndcg@10` | 0.03323035190128754 |

**Hard target:** strict `test_recall@10 > 0.065` → **FAIL**; shortfall `0.005110377966190224`.

## Comparison and attribution

- iter11: `test_recall@10=0.05976775361688052`; iter18 is higher by `0.00012186841692925915`.
- iter16: `test_recall@10=0.05687773115827226`; iter18 is higher by `0.0030118908755375207`.
- iter17: `test_recall@10=0.059071362662999005`; iter18 is higher by `0.0008182593708107727`.
- Attribution: qualified downstream target miss; no causal mechanism effect is established. See `failure_attribution_iter18.md`.
- Metadata caveat: `training_metrics.jsonl` says `variant="unknown_variant"`, while its `code_path` confirms the iter18 SID JSON. The launch audit records nonfatal NCCL warnings and process-group cleanup warning; the final evaluation completed.

No promotion. Continue with a distinct iter19 mechanism after evidence review and require its full Stage3 result before adoption. Never use Stage2 SID geometry or `hitrate@50` as a gate.

## Evidence

- Stage3: `results/stage3_T5Train/curvature_RQ-VAE_iter18/logs/Amazon_2023_Instruments/Sep-26-2026_04-45-57/test_final.json`
- Stage2 checks and geometry: `mvg_check_iter18.log`, `sid_geometry_iter18.md`
- Stage3 audit and attribution: `stage3_launch_iter18.log`, `stage3_outcome_iter18.md`, `failure_attribution_iter18.md`
