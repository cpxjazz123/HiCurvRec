"""Frozen settings for the ICML-2018 entailment-cone reproduction.

Every knob is a constant: no CLI arguments, no environment overrides. Changing
a sweep means editing this file and re-running ``python -m
bench_semantic_cones.run_paper_cones`` (project rule 1).
"""

from pathlib import Path

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/GeneRec")

# --------------------------------------------------------------------------
# The authors' released data and splits. Using them verbatim removes any
# divergence in the basic / transitive-closure split, the 0-50% supervision
# axis and the corrupted-negative sets.
# --------------------------------------------------------------------------
PAPER_DATA_DIR = Path(
    "/home/wlia0047/ar57_scratch/wenyu/refs/hyperbolic_cones"
    "/hyperbolic_cones-master/data/maxn"
)
PAPER_DATASET = "noun"
PAPER_RESULT_DIR = REPO_ROOT / "results/bench_semantic_cones/paper_wordnet_cones"
PAPER_GEOMETRIES = ("euclid", "poincare")
PAPER_DIMS = (5, 10, 16, 32)
PAPER_RATIOS = ("0percent", "10percent", "25percent", "50percent")
PAPER_SEEDS = (42, 43, 44)

# Cone geometry, exactly the reference values.
PAPER_K = 0.1
PAPER_MARGIN = 0.01
PAPER_EPSILON = 1e-5
PAPER_NEGATIVES = 10
PAPER_EPOCHS = 300

# The reference steps one 10-pair batch at a time. Gradients are summed over a
# batch, so chunking wider leaves the per-epoch update total identical while
# cutting python-side step overhead by ~3000x.
PAPER_BATCH = 32_768
# The reference steps one 10-pair batch at a time, so no node ever moves far in
# a single step and the boundary projection stays self-correcting. Summing a
# far wider chunk would push nodes onto the Poincare boundary, where the metric
# diverges. Displacements are therefore capped at this fraction of each node's
# current radius, with the same cap for both geometries.
PAPER_UPDATE_CAP = 0.05

# The reference tunes lr per method (1e-4 for HypCones/rsgd, 3e-4 for
# EuclCones at batch 10). With far fewer updates per epoch each arm is tuned at
# dim 5 / 50% and then frozen for the whole grid: euclid peaks at 0.3, poincare
# at 1.0. Both arms still optimise the same objective with the same rule.
PAPER_LR_BY_GEOMETRY = {"euclid": 0.3, "poincare": 1.0}

# Shared initialisation for both arms: random directions with radii in this
# band, which keeps both apertures non-degenerate.
PAPER_INIT_LOW = 0.12
PAPER_INIT_HIGH = 0.6

# --------------------------------------------------------------------------
# Optional Poincare NLL warm start (the reference's init stage). Off by
# default: the reimplementation drives every node onto the ball boundary,
# leaving hyperbolic apertures near one degree, which costs the hyperbolic arm
# about nine F1 points at dim 5 / 50%.
# --------------------------------------------------------------------------
PAPER_USE_WARM_START = False
PAPER_RESC_VECS = 0.7
PAPER_INIT_EPOCHS = 100
PAPER_INIT_LR = 0.03
PAPER_INIT_BURN_IN = 20
PAPER_INIT_NEG_POWER = 0.75
PAPER_INIT_MIN_RADIUS = 1e-3

# Stratification of the held-out link-prediction AUC: depth is the longest
# hypernym chain, degree the number of descendants in the closure. Cuts sit on
# the held-out apex distribution: depth terciles are 3 / 5 and degree terciles
# are 656 / 5425, so no bucket is empty and none swallows the tail.
PAPER_DEPTH_CUTS = (("shallow", 1, 3), ("mid", 4, 5), ("deep", 6, 10_000))
PAPER_DEGREE_CUTS = (("few", 0, 655), ("mid", 656, 5_424), ("many", 5_425, 10**9))

# --------------------------------------------------------------------------
# Faithful-optimisation round. The reference steps one 10-pair batch at a time,
# which is ~11.2M updates per run at 50% supervision; a step costs ~3.4ms here
# because it is kernel-launch bound, so the exact count is out of reach. This
# round instead sweeps the batch size down towards the reference on the decisive
# cell, and crosses the Poincare warm start with the cold start, to separate
# "optimisation granularity" from "geometry" as the cause of the low-dimension
# deficit.
# --------------------------------------------------------------------------
FAITHFUL_RESULT_DIR = (
    REPO_ROOT / "results/bench_semantic_cones/paper_wordnet_cones_faithful"
)
FAITHFUL_LR = {"euclid": 3e-4, "poincare": 3e-4}  # the reference's own values
FAITHFUL_UPDATE_CAP = None  # the reference has no trust region
FAITHFUL_EPOCHS = 300
# Granularity ladder at the reference learning rate, on the decisive cell.
# 256 and 64 are deliberately absent for now: a step costs ~3.4ms whatever the
# batch (the loop is kernel-launch bound), so the 256-pair cell costs 33 minutes
# and the 64-pair cell 99. The three cheap rungs answer whether the gap closes
# under a 32x refinement; a finer rung is only worth 33 minutes if it does.
FAITHFUL_BATCHES = (32_768, 4_096, 1_024)
FAITHFUL_DIMS = (5, 10)
FAITHFUL_RATIOS = ("0percent", "50percent")
FAITHFUL_INITS = ("cold", "warm")
FAITHFUL_SEED = 42
# Init axis runs in the regime where both arms actually converge (wide chunk,
# per-arm tuned lr), so it costs ~30s per cell instead of ~25 minutes and does
# not confound the initialisation effect with under-training.
FAITHFUL_BASE_BATCH = 32_768
FAITHFUL_GRID_DIM = 5
FAITHFUL_GRID_RATIO = "50percent"
# Per-arm learning rates that reach a plateau at the wide chunk; used by the
# init axis.
FAITHFUL_COARSE_LR = {"euclid": 0.3, "poincare": 1.0}
FAITHFUL_COARSE_CAP = 0.05

BOOTSTRAP_SAMPLES = 2_000
BOOTSTRAP_SEED = 2026
