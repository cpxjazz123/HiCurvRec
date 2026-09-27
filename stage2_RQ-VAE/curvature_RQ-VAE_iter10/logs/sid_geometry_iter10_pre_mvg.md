# sid_geometry_iter10 (Agent D — pre-flight)

Status: pre-flight only.

Anticipated fingerprint after Stage2:
- per_layer: similar to iter8 (0.20, 0.23, 0.22) since ε schedule unchanged.
- collision: similar to iter8's 7.1% (slight tightening from 5 iters).
- full_gini: similar to iter8's 0.0684.

Hard-fail trigger: full_gini regresses by >50% vs iter8 (catastrophic).