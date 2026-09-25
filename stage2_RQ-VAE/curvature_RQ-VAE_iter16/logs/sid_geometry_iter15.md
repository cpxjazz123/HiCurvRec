# Agent D — SID geometry (iter15)

**Gate: ALIGNED** (mechanism direct effects observed). Stage3 gate: **NO-GO** (test_R@10=0.0582).

## Mechanism verification

- `freeze_layer_scale=True`: `u_learned` identical to calibration at steps 0, 10k, 50k, 100k (`logs/curvature_snapshots.txt`).
- Layer ordering: `c0 < c1 < c2` whenever `g(t) > 0` (e.g. step 10k, 50k).
- `curv_reg=0` throughout training (`train_migrated.log`).
- No behavior branching (`b_l`); warm-start from iter8 with frozen `c_layer_scale`.

## Final SID (step 100k export)

| Metric | Value |
|--------|-------|
| unique (3-token) | 23095 / 24587 |
| hitrate@50 | 0.9968 |
| H(L1\|L0) | 5.5962 |
| l01_pairs | 14579 |

Geometry healthy (no collapse); slightly more unique prefixes than iter14 (23003).
