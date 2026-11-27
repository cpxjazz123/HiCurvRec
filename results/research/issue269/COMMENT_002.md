## Stage2 complete + Stage3 dispatched

Run `stage2_bba78cf7c160467da1c8ae7de9b6e26f`, arm `TIGER_calibrated_kappa_rq`, single-card
A100 (owner authorisation), batch 4096, seed 42, 3000 epochs, port 50202.

| item | value |
|---|---|
| wall / mean epoch | `1213.8 s` / `0.392 s` |
| final recon | `9.537e-5` (#265 parent 3000: `1.23e-4`) |
| final curvature estimate | `1.55` (target RMS `2.00`) |
| catalogue 3-code collision | **`0.1567`** (< 0.40) |
| SID | `item_sids.json` sha256 `a89110b8…`; 24,587 items |
| §63 layer record | `results/stage2_RQ-VAE/TIGER_calibrated_kappa_rq/stage2_metrics/a89110b8….json` (L3 collision `0.15630`, L2 prefixes `6,246`, L3 triples `20,744`, refinement gain **`2.849` bits**, occupancy `256/256/256`, L1 hit `0.0836`, L1 hard ER `5.72`, conditional enrichment `1.637` bits — full record in the file) |
| archive binding | `archive_binding.json` sha256 `4e51a695…` |
| SID quality gate | **allow**: collision `0.1563 < 0.40`, quant_error `0.00875` (warning: below the 0.035 diagnostic band), effective rank `4.33 < 8.0` |

The collision trajectory was noisy early (`0.82` at epoch 200) but settled: `0.25` (2550),
`0.19` (2700), `0.177` (2800), `0.151` (2950), `0.157` (3000). The full-catalogue evaluation is
the same metric the export gate reads.

**Hard-q geometry** (20,000 deterministic rows, independent reference set seed 269, calibrated
`S*`):

| space | Pearson | Spearman | sign accuracy | mean J |
|---|---|---|---|---|
| frozen `Z_ref` | `0.093` | `0.084` | `0.548` | — |
| trained hard `q` | **`0.334`** | **`0.353`** | **`0.684`** | `0.288` |

Training improves the A.1 curvature-proxy ranking about fourfold relative to the frozen
representation, but the raw graph-distance distortion is worse than `Z_ref` (~`0.17`); the two
are not the same objective. Per-role MAE and the full per-row data are in
`results/research/issue269/analysis/OWN_KAPPA.json`.

**Graphlet/AF3 fidelity** (`LOCAL_STRUCTURE.json`): triangle leaves vs open-wedge leaves at L1 are
`0.0895` vs `0.0274` (3.27x), comparable to frozen TIGER (`0.0990` vs `0.0240`); AF3 strata are
recorded per degree/triangle bucket.

**Stage3** passed the export gate (SID re-exported from the checkpoint), the collision gate and the
frozen-protocol check; single-card 150-epoch run started at 13:46, variant
`TIGER_calibrated_kappa_rq`.

Evidence: `results/research/issue269/analysis/{STAGE2_REPORT,OWN_KAPPA,A1_ZREF,LOCAL_STRUCTURE}.json`,
`results/stage2_RQ-VAE/TIGER_calibrated_kappa_rq/stage2_metrics/`.
