"""Fixed constants and arm routing for anti-collapse mechanisms."""

ALIGN_WEIGHT = 0.1
DIVERSITY_WEIGHT = 0.1
GUMBEL_TAU = 0.5

ARM_MECHANISM = {
    "LETTER": "alignment + diversity + codebook-gradient behaviour term",
    "LETTER_nodiv": "alignment only",
    "DIGER": "gumbel soft assignment",
}

ARM_FLAGS = {
    "LETTER": {"use_align": True, "use_diversity": True, "use_gumbel": False},
    "LETTER_nodiv": {"use_align": True, "use_diversity": False, "use_gumbel": False},
    "DIGER": {"use_align": False, "use_diversity": False, "use_gumbel": False},
    # E2. The collaborative term lives in cf_alignment, not here, so none of these
    # arms ask for the encoder-self alignment; only the plan's DIV configurations
    # ask for the codebook-uniformity term, which is what use_diversity gates.
    "CF_ALIGN": {"use_align": False, "use_diversity": False, "use_gumbel": False},
    "CF_ALIGN_DIV": {"use_align": False, "use_diversity": True, "use_gumbel": False},
    "CF_ALIGN_DIV_RANK": {"use_align": False, "use_diversity": True, "use_gumbel": False},
}
