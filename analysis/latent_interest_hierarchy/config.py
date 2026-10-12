"""Configuration for the latent-interest-hierarchy study. Hard-coded (CLAUDE.md §1).

Phase 1 only. The question is whether real purchase sequences contain a stable,
predictive latent interest hierarchy, and it is answered with matched controls
before any geometry is involved.
"""
from __future__ import annotations

from pathlib import Path

SEED_BASE = 42
SEEDS = (42, 43, 44)          # at least three random seeds

REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")
TRAIN_PARQUET = REPO / "results/stage0_build_parquet/train.parquet"
VALID_PARQUET = REPO / "results/stage0_build_parquet/valid.parquet"
TEST_PARQUET = REPO / "results/stage0_build_parquet/test.parquet"

ANALYSIS_DIR = REPO / "analysis/latent_interest_hierarchy"
OUT_DIR = REPO / "results/latent_interest_hierarchy"

# Item universe: the parquet's maximum target id plus one, so every id is
# representable and nothing is silently dropped.
N_ITEMS = 24587

# ---------------------------------------------------------------- interest model
# All three arms carry EXACTLY the same parameters: an item embedding table and 64
# interest centroids. "Hierarchical" is a factorisation of how those 64 centroids are
# combined into a user profile, never an extra parameter, so any difference in
# validation NLL cannot be attributed to capacity.
EMBED_DIM = 32
N_COARSE = 8
N_FINE = 8
N_INTEREST = N_COARSE * N_FINE          # 64 leaves, identical in every arm
MAX_HISTORY = 20                       # history items kept per instance

# ------------------------------------------------------------------- optimisation
EPOCHS = 30
BATCH_SIZE = 512
LR = 0.01
WEIGHT_DECAY = 0.0
GRAD_CLIP = 1.0
TRAIN_SUBSET_USERS = 20000             # matched across arms and seeds

# ------------------------------------------------------------------ evaluation
TOPK = 10
BOOTSTRAP_RESAMPLES = 1000
SHORT_HISTORY_QUANTILE = 0.25          # users in the shortest quartile
LOW_FREQ_QUANTILE = 0.25               # items in the lowest frequency quartile

# --------------------------------------------------------------- controls
ARMS = ("flat", "hier", "random_hier", "hier_shuffled", "flat_shuffled")

# arm -> model mode, stated once so it can never be inferred from a name prefix.
# A prefix rule once mapped random_hier onto hier and silently made the negative
# control a duplicate of the treatment.
ARM_TO_MODE = {
    "flat": "flat",
    "hier": "hier",
    "random_hier": "random_hier",
    "flat_shuffled": "flat",
    "hier_shuffled": "hier",
}
SHUFFLED_ARMS = ("flat_shuffled", "hier_shuffled")