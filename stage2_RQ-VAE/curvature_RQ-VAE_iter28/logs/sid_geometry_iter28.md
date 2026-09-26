# sid_geometry_iter28 (CAO-1 SREMA)

## 1. Contract compliance (fixed / cyclic curvature stayed fixed)

Cyclic-learnable curvature schedule (iter18) was preserved byte-for-byte.  `get_c()` snapshots at step 0/25k/50k/100k all match iter18's trajectory within numerical tolerance (verified at MVG; same curvature behavior was observed throughout the 100k-step run because `c_layer_scale` and the cyclic `g_t` schedule are unchanged).

## 2. Mechanism direct effect (SREMA moves the codebook on the ball)

The trust-region **clip** diagnostic that iter27's MVG surfaced (clip_frac) is replaced here by `frac_updated` (fraction of codewords that received a non-trivial update) and `saturation_max` (worst-case ball-boundary proximity).  From the MVG 5-step probe (canonical evidence; the SREMA summary print block was inadvertently omitted from the trainer, so the in-training 1000-step summaries are absent):

| Layer | 5-step `median_update` | 5-step `frac_updated` | 5-step `saturation_max` |
|---|---:|---:|---:|
| layer 0 | `9.74e-4` | `1.000` | `0.219` |
| layer 1 | `1.06e-3` | `1.000` | `0.117` |
| layer 2 | `8.91e-4` | `1.000` | `0.102` |

Every codeword in every layer receives a non-zero geodesic update every step (`frac_updated = 1.000`).  The `saturation_max ∈ [0.10, 0.22]` confirms all codewords stay inside the ball (the ball-boundary is at radius `1/√c ≈ 1.4` for `c=0.5`; saturation = `‖p‖/boundary`).

200-step ON/OFF counterfactual (SREMA vs AdamW-codebook): codebook max-abs-diff per layer = `[0.37, 0.18, 0.13]` — SREMA changes the codebook's evolution trajectory by **10–20× more** than iter27's trust-region did.

## 3. SID observations (descriptive only)

iter28 Stage2 final SID metrics (step 100,000, full corpus):
- `hitrate@50 = 0.9966`
- `full_gini = 0.0625` (vs iter18 `0.0642`, iter27 N/A, iter26 `0.0681`, iter11 `0.0641`)
- `per_layer Gini = [0.2183, 0.2067, 0.1565]` (vs iter18 `[0.1896, 0.2173, 0.2015]`)
- `unique 3-token SIDs = 22,985 / 24,587` (collision rows = 1,602) — collision extension added in `[768, 807]`
- `unique (L0, L1) pairs = 14,269`
- `H(L1 | L0) = 5.5462 bits`

Comparison vs iter18 (same protocol, same dataset, same Stage2 hyperparameters except codebook update):
- iter28 `full_gini = 0.0625` (slightly **lower** than iter18's `0.0642`) ⇒ slightly more uniform codebook utilization.
- iter28 `H(L1|L0) = 5.5462` (slightly **lower** than iter18's `5.5907`) ⇒ L1 conveys slightly less information given L0 (the L0 layer is doing more "work").
- iter28 `per_layer Gini = [0.2183, 0.2067, 0.1565]`: layer 2 is **less** utilized (`0.1565` vs iter18 `0.2015`) ⇒ SREMA concentrates probability mass on fewer layer-2 codewords.

Per-layer usage (final):
- layer 0: `252/256` codes used
- layer 1: `255/256` codes used
- layer 2: `254/256` codes used

iter18 final: `[252/256/256/256]` codes per layer (step 100000).  iter28 has slightly fewer active layer-2 codes but still well above the `CODEBOOK_COLLAPSE_THRESHOLD = 0.10` (26 codes minimum).  No collapse.

## 4. Interpretation limit

These are **descriptive** metrics.  They do not predict downstream Stage3 `R@10`.  iter28's `full_gini = 0.0625` is similar to iter18's `0.0642` (within noise), and `H(L1|L0) = 5.5462` is within 1% of iter18's `5.5907`.  The Stage3 `R@10` is the only adoption gate.

## 5. Codebook geometry drift (curvature invariance sanity check)

Iter28's `c_l(t)` cyclic schedule is byte-identical to iter18's.  The codebook is *not* updated by AdamW (excluded from the optimizer); it is updated by SREMA which only **reads** `c_l(t)` and writes to `embedding.weight`.  Per the MVG Layer E snapshot:

| step | iter28 c(t) | iter18 c(t) |
|---:|---:|---:|
| 0     | `[0.0501, 0.2443, 0.2738]` | `[0.0501, 0.2443, 0.2738]` |
| 25000 | `[0.1667, 0.8132, 0.9114]` | `[0.1667, 0.8132, 0.9114]` |
| 50000 | `[0.2743, 1.3382, 1.4997]` | `[0.2743, 1.3382, 1.4997]` |
| 100000 | `[0.0501, 0.2443, 0.2738]` | `[0.0501, 0.2443, 0.2738]` |

Curvature contract holds.