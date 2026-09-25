# sid_geometry_iter8 (Agent D — pre-flight)

Status: pre-flight only. Will be re-evaluated after Stage2 SID export.

Anticipated fingerprint after Stage2:
- L0 utility: should remain ≥ iter7 baseline (0.2011) because ε changes only at low c, not at high c.
- L1/L2 utility: expected to improve at high c (sharper per-layer quantization); drop slightly at low c (softer assignments).
- collision: expected to remain ≤ iter7 (~13.2%) since the assignment softmax is still applied.
- L0 oracle: expected to improve at high c; the high-c phase is where the downstream T5 maps tokens, and sharper assignments there should give the model more discriminative next-token targets.

Hard-fail trigger: full_gini regresses by >50% vs iter7 (catastrophic SID regression).