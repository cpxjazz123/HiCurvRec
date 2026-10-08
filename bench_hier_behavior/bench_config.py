"""Frozen configuration for the synthetic hierarchical-behaviour benchmark.

Design
------
The question is narrow and falsifiable: on behaviour that is *generated from* a
known item tree (never from coordinates in any manifold), does a Poincare-ball
RQ-VAE beat an otherwise identical Euclidean RQ-VAE, and is that gap a
property of the negative curvature or only of the tree-shaped signal?

Four arms share one code path and differ only in the geometry plug-in and in
whether an extra (parameter-free) hierarchy loss is switched on:

  euclid_rq            Euclidean distance / residual, no hierarchy loss
  hyp_rq               Poincare distance / residual, no hierarchy loss
  euclid_rq_hier       Euclidean + Euclidean entailment-cone max-margin
  hyp_rq_cone          Poincare + Poincare entailment-cone max-margin

``euclid_rq`` vs ``hyp_rq`` isolates curvature. ``euclid_rq_hier`` vs
``hyp_rq_cone`` isolates cone geometry under identical supervision. Because the
hierarchy loss adds no parameters (parent apexes are centroids of the encoder
outputs of their own children), all four arms have identical parameter counts.

Two data arms run every configuration: ``tree`` and ``hierarchy_destroyed``.
The second relabels behaviour with a random bijection on item ids, which
preserves the popularity distribution and every sequence length exactly while
severing the link between behaviour and tree labels. An advantage that
survives the relabelling is not a hierarchy effect.

Everything is hard-coded on purpose: no CLI args, no environment overrides.
Re-run from the repository root with
``python -m bench_hier_behavior.run_bench``.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCH_DIR = REPO_ROOT / "bench_hier_behavior"
RESULT_ROOT = REPO_ROOT / "results/bench_hier_behavior"

# --------------------------------------------------------------------------
# Item trees. Every shape holds exactly 24576 leaf items, so the depth sweep
# changes tree structure at constant catalogue size. A fixed 24576-item
# 16/16/96 tree is only three levels deep, which is shallow enough that
# Euclidean space is expected to be near-lossless; deeper shapes at the same
# item count are what give negative curvature room to help.
# --------------------------------------------------------------------------
TREE_SHAPES = {
    "d3_wide": (16, 16, 96),
    "d4_balanced": (8, 8, 8, 48),
    "d5_deep": (4, 8, 8, 8, 12),
    "d6_deeper": (4, 4, 4, 4, 8, 12),
}
DEFAULT_TREE = "d3_wide"
N_ITEMS = 24576

# --------------------------------------------------------------------------
# Text-like item features. The stage-1 encoder is replaced by a bag-of-attribute
# encoder: a node-attribute half that every item under a node shares, and an
# item-attribute half that is unique per item. Siblings therefore look alike in
# Euclidean cosine exactly as T5 sentence embeddings do for real catalogue
# items, and both model arms receive byte-identical inputs.
# --------------------------------------------------------------------------
NODE_ATTR_DIM = 128
ITEM_ATTR_DIM = 128
FEATURE_DIM = NODE_ATTR_DIM + ITEM_ATTR_DIM
FEATURE_NOISE_SCALE = 0.35

# --------------------------------------------------------------------------
# Users and behaviour. The three-way mix (same interest node / same top-level
# category / different category) is swept rather than fixed, because the mix is
# the single knob that sets how much behavioural hierarchy exists at all.
# Sequences ramp linearly from EXPLORATORY_MIX at the first event to the
# preset mix at the last event, so users start broad and concentrate.
# --------------------------------------------------------------------------
N_USERS = 8000
SEQ_LEN_LOG_MEAN = 1.7
SEQ_LEN_LOG_SIGMA = 0.5
SEQ_LEN_MIN = 3
SEQ_LEN_MAX = 25

EXPLORATORY_MIX = (0.45, 0.35, 0.20)
BEHAVIOUR_MIXES = {
    "concentrated": (0.85, 0.12, 0.03),
    "balanced": (0.70, 0.20, 0.10),
    "diffuse": (0.45, 0.35, 0.20),
}
DEFAULT_MIX = "balanced"

POPULARITY_ZIPF_EXPONENT = 0.8
POPULARITY_ZIPF_OFFSET = 50.0

# Deepest internal level whose subtree still holds at most this many items is
# the "interest node" that behaviour concentrates on.
INTEREST_SUBSIZE_MAX = 128

DATA_ARMS = ("tree", "hierarchy_destroyed")

# --------------------------------------------------------------------------
# Model. Same shapes and same initial values across all four arms; only the
# geometry plug-in differs. No working-radius pinning: it has no meaning in an
# unbounded Euclidean space, and applying it to one arm only would be an
# uncontrolled advantage.
# --------------------------------------------------------------------------
CODEBOOK_NUM = 3
CODEBOOK_SIZE = (256, 256, 256)
HIDDEN_SIZES = (512, 256)
DROPOUT = 0.0
VQ_BETA = 0.25
SK_EPSILON = 0.003
SK_ITERS = 50
CURVATURE = 1.0
# Every item is encoded to the same tangent radius. Without this the encoder
# shrinks its output arbitrarily far, cone apertures become meaningless, and the
# two arms would be compared at different latent scales. GeneRec pins the same
# quantity per quantization level; a single radius here is the geometry-neutral
# version of that idea and is applied identically to both arms.
LATENT_RADIUS = 1.0

EPOCHS = 200
BATCH_SIZE = 1024
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.0
SEED = 42

# --------------------------------------------------------------------------
# Hierarchy supervision for the two *_hier / *_cone arms. Edge supervision is a
# parent/child max-margin over Ganea-style entailment cones; negatives are
# random non-descendants. The parent apex is the centroid of its supervised
# children's encoder outputs, so this adds zero parameters.
# --------------------------------------------------------------------------
EDGE_SUPERVISION_FRACTION = 0.8
CONE_TRAIN_K = 0.3
EDGE_BATCH_SIZE = 256
EDGE_APEX_SAMPLE = 16
# Chosen so the cone loss starts at roughly the same magnitude as the
# reconstruction MSE of unit-norm features, rather than swamping it.
EDGE_LOSS_WEIGHT = 0.5

# --------------------------------------------------------------------------
# Evaluation.
# --------------------------------------------------------------------------
CONE_FIT_TARGET_COVERAGE = 0.90
# A concept apex is the mean of its children's encoder outputs; below this
# tangent norm that mean is treated as collapsed and the cone undefined.
DEGENERATE_APEX_NORM = 0.1
DISTORTION_PAIR_SAMPLES = 20000
BEHAVIOUR_EVAL_USERS = 2000
BEHAVIOUR_NEGATIVES = 99
SID_PREFIX_WEIGHTS = (100.0, 10.0, 1.0)
LATENT_DIMS = (8, 16, 32)

# --------------------------------------------------------------------------
# Run grid. Stage 1 answers "can curvature preserve tree structure"; stage 2
# answers "do cones generalise to held-out edges"; stage 3 answers "does any of
# it move behaviour ranking", including the hierarchy-destroyed control.
# --------------------------------------------------------------------------
STAGE1_DIMS = (8, 16, 32)
STAGE1_TREES = ("d3_wide", "d4_balanced", "d5_deep", "d6_deeper")
STAGE1_ARMS = ("euclid_rq", "hyp_rq")

STAGE2_DIMS = (8, 16, 32)
STAGE2_TREES = ("d3_wide", "d6_deeper")
STAGE2_ARMS = ("euclid_rq_hier", "hyp_rq_cone")

STAGE3_DIMS = (8, 32)
STAGE3_TREES = ("d5_deep",)
STAGE3_MIXES = ("concentrated", "balanced", "diffuse")
STAGE3_ARMS = ("euclid_rq", "hyp_rq", "euclid_rq_hier", "hyp_rq_cone")

ARMS = ("euclid_rq", "hyp_rq", "euclid_rq_hier", "hyp_rq_cone")
ARM_GEOMETRY = {
    "euclid_rq": "euclid",
    "hyp_rq": "poincare",
    "euclid_rq_hier": "euclid",
    "hyp_rq_cone": "poincare",
}
ARM_HIERARCHY_LOSS = {
    "euclid_rq": False,
    "hyp_rq": False,
    "euclid_rq_hier": True,
    "hyp_rq_cone": True,
}
