# sid_geometry_iter13 (Agent D — post-Stage2)

Observed fingerprint (step=100000):
- L0 utility: `0.1918` (iter8 0.2046) — close.
- L1 utility: `0.1653` (iter8 0.2303) — drop.
- L2 utility: `0.1587` (iter8 0.2233) — drop.
- full_gini: `0.0482` (iter8 0.0684) — large improvement.
- collision rate: `1,205 / 24,587 ≈ 4.9%` (iter8 7.1%) — large improvement.
- unique codes: `23,382 / 24,587` (iter8 22,831) — large improvement.
- l01_pairs: `16,598` (iter8 14,294) — large improvement.
- H(L1|L0): `5.8442` (iter8 5.5336) — large improvement.

Direct effects (DE-1..3) verification:
- DE-1 (log_tau_l drift): observed. Final vl=1.36 (vs iter8 0.30) — the commitment loss is now ~4x larger because log_tau_l drifted to non-zero values, increasing per-layer commitment multipliers.
- DE-2 (warm-start loaded iter8): observed. First-step rl ≈ iter8's (~1.28).
- DE-3 (per-layer utilization drift): observed. L2 utility dropped from 0.22 to 0.16 — the optimizer's per-layer log_tau_l drift concentrated commitment pull on L0.

Verdict: `ALIGNED`. The per-layer learnable commitment multiplier provided significantly improved SID geometry (collision -31%, unique +551, full_gini -29%) but at the cost of per_layer utilization balance (L2 dropped).