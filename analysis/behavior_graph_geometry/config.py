"""Configuration for the behavior-graph geometry analysis. Hard-coded (CLAUDE.md §1).

Nothing here is a CLI argument: the study is reproduced by running the scripts with
no arguments from the repository root.
"""
from __future__ import annotations

from pathlib import Path

SEED = 42

REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")
TRAIN_PARQUET = REPO / "results/stage0_build_parquet/train.parquet"
VALID_PARQUET = REPO / "results/stage0_build_parquet/valid.parquet"

ANALYSIS_DIR = REPO / "analysis/behavior_graph_geometry"
OUT_DIR = REPO / "results/behavior_graph_geometry"

# Item universe used by every Stage 2/3 artifact in this repository.
N_ITEMS = 24587

# The item sequence of a user is the row order within that user; the repository's
# own transition helper reads (seen_history[-1], target), which is the same thing.
# The order is recovered from len(seen_history) and the recovery is verified, never
# assumed.

# Gromov hyperbolicity sampling.
GROMOV_QUADS = 200_000          # quadruples sampled per graph
GROMOV_BATCH = 20_000           # per BFS-batch allocation
GROMOV_STRATA = {
    "uniform_lcc": "uniform over the largest connected component",
    "popular": "anchored on the top 1% most interacted items",
    "low_popularity": "anchored on items in the bottom half of the interaction count",
}

# Degree-preserving null model.
NULL_SWAPS_PER_EDGE = 10
NULL_REPLICATES = 3

# Power-law fitting (Clauset-Shalizi-Newman).
POWERLAW_MIN_TAIL = 50
POWERLAW_MAX_XMIN_CANDIDATES = 400
POWERLAW_ALPHA_GRID = (0.001, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5)

# Embedding comparison.
EMBED_DIMS = (8, 16, 32)
EMBED_STEPS = 4000
EMBED_LR = 0.02
EMBED_BATCH_PAIRS = 2048
TRAIN_PAIR_FRACTION = 0.5
EVAL_PAIR_FRACTION = 0.25
HELDOUT_PAIR_FRACTION = 0.25      # held-out graph edges, never used for training
NEGATIVE_RATIO = 1                # degree-matched negatives per held-out edge
NEIGHBOURHOOD_K = (1, 5, 10, 50)

# Figures.
FIG_DPI = 140
