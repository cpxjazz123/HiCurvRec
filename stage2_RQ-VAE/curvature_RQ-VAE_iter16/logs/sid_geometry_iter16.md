# Agent D — SID geometry (iter16)

**Gate: ALIGNED** (mechanism direct effects all observed). Stage3 gate: **NO-GO** (test_R@10=0.05688).

## Mechanism verification

- `closed_form_c_l = [0.663, 0.435, 0.433]` is **constant across all 100k training steps** (`logs/curvature_snapshots_iter16.txt`).
- c₀ > c₁ > c₂ matches the formula's `z_l` ordering.
- `curv_reg=0` throughout training (`train_migrated.log`).
- Warm-start from iter8; closed-form `_fixed_c` buffers registered, no `c_layer_scale` learnable parameter.

## Final SID (step 100k export)

| Metric | Value |
|--------|-------|
| unique (3-token) | 22827 / 24587 |
| hitrate@50 | 0.9963 |
| H(L1\|L0) | 5.4549 |
| l01_pairs | 14096 |
| full_gini | 0.0681 |
| per_layer_gini | [0.1107, 0.2243, 0.3357] |

Geometry is healthy (no collapse) but **unique 22827 < iter15 23095** (more 3-token collisions).

## Alignment verdict

- DE-1 (c_l constant): **PASS** (c_live = c_closed at steps 0/10k/50k/100k)
- DE-2 (c₀>c₁>c₂): **PASS** (0.663 > 0.435 > 0.433)
- DE-3 (no learnable curvature): **PASS** (curv_reg=0, _fixed_c buffer)

→ **ALIGNED** — geometry and direct effects confirm mechanism; **Stage3 gate is the only verdict** (NO-GO).
