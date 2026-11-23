"""Frozen constants for the v2 build (issue #246).

Changing a value here changes the asset version; never edit a value in place while keeping the
same `VERSION` string.
"""
from __future__ import annotations

from pathlib import Path

def _repository_root() -> Path:
    """The real repository, found by the ``stage2_RQ-VAE`` marker.

    A copy of this package inside another tree (the mutation harness) must still point at the
    real data, so the marker wins over the file location.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "stage2_RQ-VAE" / "baseline").is_dir() and (parent / "stage3_T5Train").is_dir():
            return parent
    return here.parents[2]


REPO = _repository_root()

VERSION = "v2"
SPLIT_SEED = 246001
SPLIT_RATIOS = (0.8, 0.1, 0.1)          # visible / struct_val / struct_test
EGO_SEED = 246002
EGO_CAP = 32                            # neighbours, the root is extra
K1_OUT = 8
K1_IN = 8
K2_PER = 4
DISTANCE_ROOTS = 2000
CURVATURE_EDGES = 400                   # sampled visible edges for exact Ollivier-Ricci
PPR_RESTART = 0.15
PPR_TOLERANCE = 1e-10
PER_GROUP = 12000

ASSET_ROOT = REPO / "results" / "research" / "infra246" / "assets"
V1_ROOT = REPO / "results" / "research" / "infra245" / "assets"

ITEMS = REPO / "results" / "stage0_build_parquet" / "items.parquet"
TRAIN = REPO / "results" / "stage0_build_parquet" / "train.parquet"
EMBEDDINGS = REPO / "stage1_GeneEmbedding" / "output" / "sentence_t5.npy"
