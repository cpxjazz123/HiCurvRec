# sid_geometry_iter9 (Agent D — pre-flight)

Status: pre-flight only.

Anticipated fingerprint after Stage2:
- L0 utility: similar to iter8 (0.2046) since c reaches the same peak.
- L1/L2 utility: similar to iter8 (0.23 each); the power-law only refines the schedule, not the maximum.
- collision: expected to remain ≤ iter8's 7.1% (the assignment softmax is still applied, just with a different ε schedule).
- full_gini: similar to iter8's 0.0684.

Hard-fail trigger: full_gini regresses by >50% vs iter8 (catastrophic).