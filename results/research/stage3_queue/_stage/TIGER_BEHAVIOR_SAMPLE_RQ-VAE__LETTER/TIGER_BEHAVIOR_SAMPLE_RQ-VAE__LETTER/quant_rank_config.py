"""Fixed group A/B settings for quantized behavior ranking."""

# Each entry scales that residual-quantizer level in the summed representation.
# Group A supervises all three residual-quantizer levels.
# Group B also uses all three: the experiment needs per-level gradient evidence
# at L1, L2 and L3. A level-1-only (1, 0, 0) arm could not answer that question;
# retain it as a possible future single-variable arm.
GROUP_CONFIGS = {
    "A": {"level_weights": (1.0, 1.0, 1.0)},
    "B": {"level_weights": (1.0, 1.0, 1.0)},
}
ACTIVE_GROUP = "A"
LEVEL_WEIGHTS = GROUP_CONFIGS[ACTIVE_GROUP]["level_weights"]
