# sid_geometry_iter7 (Agent D — pre-flight)

Status: pre-flight only. Will be re-evaluated after Stage2 SID export.

Anticipated fingerprint after Stage2:
- L0 utility: should stay ≥ iter4 baseline (0.001 floor → wider curvature range lets layer-0 codes spread more).
- H(L1|L0): expected to drop slightly because wider curvature reduces effective neighborhood precision but the warm-start embedding mitigates.
- collision: expected to remain ≤ iter4 (~15.4%).
- L0 oracle: expected to improve (more discriminative c trajectory).

Hard-fail trigger: r = 0.5×iter4 (catastrophic SID regression).