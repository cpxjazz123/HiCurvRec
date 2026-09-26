# gate_decision_iter28

## Decision: **NO-GO** (Stage3 hard target not met)

- **Mechanism:** `iter28_sinkhorn_riemannian_ema_codebook` (SREMA) — codebook `embedding.weight` excluded from AdamW and updated by a Sinkhorn-weighted manifold-EMA step (β=0.99) after every AdamW step.
- **Stage2:** completed 100,000 steps and exported iter28 four-token SIDs (`SID_WIRING_PASS`).  The one-checkpoint/one-batch MVG and registered direct-effect checks passed.  Cyclic curvature schedule preserved byte-for-byte from iter18.
- **Stage3:** run `Sep-27-2026_00-15-26` used the iter28 SID input (4×A40 DDP, `n_digit=4`, beam=20), completed all 150 epochs + final evaluation, `n_eval=57439`.

## Final Stage3 metrics

| Metric | Value |
|---|---:|
| `test_recall@5` | 0.03706540852034332 |
| `test_recall@10` | **0.05649471613363742** |
| `test_ndcg@5` | 0.024799025279798593 |
| `test_ndcg@10` | 0.031058744990046853 |

**Hard target:** strict `test_recall@10 > 0.065` → **FAIL**; shortfall `0.00850528386636258`.
**Canonical baseline:** strict `test_recall@10 > iter18's 0.05988962203380978` → **FAIL**; shortfall `0.00339490590017236`.

## Comparison and attribution

| Iter | R@10 | Δ vs iter18 | Δ vs iter28 |
|---|---:|---:|---:|
| iter11 (β2-conditioned AdamW) | 0.05976775 | −0.000122 | +0.003273 |
| **iter18 (canonical baseline)** | **0.05988962** | — | +0.003395 |
| iter25 (raw-residual prior) | 0.05882763 | −0.001062 | +0.002333 |
| iter26 (FCCR-1 fixed c) | 0.057017 | −0.002873 | +0.000522 |
| **iter28 (SREMA)** | **0.056495** | **−0.003395** | — |

- iter28 R@10 is **worse than every prior CAO-1 contract iteration** (iter25, iter26, iter27 was aborted).
- iter28 is **worse than iter18** by `0.003395`, exceeding the historical per-iteration noise band (~`0.003`).
- iter28 is the **lowest R@10** among completed iterations that kept iter18's protocol.  This is a real regression, not random.

- Attribution: **active mechanism, regression** — see `failure_attribution_iter28.md`.  The SREMA mechanism runs correctly (MVG confirms it's geometrically valid and active); the result is a legitimate `ACTIVE_NEGATIVE + PROMOTION_FAIL`.

- Metadata caveat: iter28 Stage3 patience counter reached 2/10 by epoch 150 (best_recall@10 stayed at `-1.000000` throughout training — Stage3's partial_eval never produced a positive validation recall).  This is consistent with iter28's SIDs being structurally harder for the HG-Rec T5 to fit, not a Stage3 implementation bug.

No promotion.  Continue with iter29 (per `failure_attribution_iter28.md` §"Carryover") or pivot away from codebook-side optimization entirely.  Never use Stage2 proxy or `hitrate@50` as a gate.

## Evidence

- Stage3: `results/stage3_T5Train/curvature_RQ-VAE_iter28/logs/Amazon_2023_Instruments/Sep-27-2026_00-15-26/test_final.json`
- Stage2 checks: `logs/mvg_check_iter28.log`, `logs/mvg_check_iter28.md`
- Stage2 SID analysis: `logs/sid_geometry_iter28.md`
- Stage3 audit and attribution: `logs/stage3_outcome_iter28.md`, `logs/failure_attribution_iter28.md`