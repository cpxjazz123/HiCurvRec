"""Fixed constants and arm routing for anti-collapse mechanisms."""

ALIGN_WEIGHT = 0.1
DIVERSITY_WEIGHT = 0.1
GUMBEL_TAU = 0.5

ARM_MECHANISM = {
    "LETTER": "alignment + diversity + codebook-gradient behaviour term",
    "LETTER_nodiv": "alignment only",
    "DIGER": "gumbel soft assignment, applied in the behaviour ranking loss",
    "DIGER_FIX": "E1: behaviour ranking distances read the Gumbel-Softmax soft assignment",
}

# No arm reaches the Gumbel branch of _anti_collapse_term any more. DIGER was the
# only arm that ever set use_gumbel, and it carried no term here: the tensor that
# branch builds is consumed solely by use_align, which DIGER disables. Its soft
# assignment now lives in quant_rank_loss.soft_quantized_ranking_loss, the one
# place where it is the thing the ranking distances are computed on and so can
# both receive and pass on a gradient.
ARM_FLAGS = {
    "LETTER": {"use_align": True, "use_diversity": True, "use_gumbel": False},
    "LETTER_nodiv": {"use_align": True, "use_diversity": False, "use_gumbel": False},
    "DIGER": {"use_align": False, "use_diversity": False, "use_gumbel": False},
    "DIGER_FIX": {"use_align": False, "use_diversity": False, "use_gumbel": False},
}
