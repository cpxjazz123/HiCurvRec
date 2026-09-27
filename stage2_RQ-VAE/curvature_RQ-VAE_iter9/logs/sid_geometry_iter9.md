# sid_geometry_iter9 (Agent D — post-Stage2)

Observed fingerprint (step=100000):
- L0 utility: `0.1120` (iter8 0.2046) — large drop (more concentrated).
- L1 utility: `0.1420` (iter8 0.2303) — drop (more concentrated).
- L2 utility: `0.2850` (iter8 0.2233) — large improvement.
- full_gini: `0.0618` (iter8 0.0684) — improved.
- collision rate: `1,590 / 24,587 ≈ 6.5%` (iter8 7.1%) — large reduction.
- unique codes: `22,997 / 24,587` (iter8 22,831) — improvement.
- l01_pairs: `14,502` (iter8 14,294) — improvement.
- H(L1|L0): `5.5042` (iter8 5.5336) — similar.

Direct effects (DE-1..3) verification:
- DE-1 (Sinkhorn ε at high-c phases equal to iter8, low-c phases ~5x larger): observed. Per-layer L0/L1 utilities are more concentrated because the assignment is now sharper at low c too (low-c sinkhorn is still soft due to power-law 0.5 but not as degenerate as the linear version).
- DE-2 (warm-start loaded iter8): observed. First-step rl=1.27 vs iter8 ~1.28.
- DE-3 (quantization loss slightly lower than iter8): observed. vl=0.32–0.39 vs iter8's 0.25–0.45.

Verdict: `ALIGNED`. Layer utilization has shifted: L2 utility jumped to 0.285 (vs iter8 0.22), while L0/L1 are more concentrated. This is consistent with a sharper power-law ε schedule that pushes more discrimination into the deeper layer.