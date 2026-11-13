"""Fixtures for the per-item-curvature arm.

The candidate tree is a package root (`distances`, `losses`, `model`,
`af3_per_item_curvature.py` sit beside this `tests/`), so these tests put that
directory on `sys.path` and import exactly what `train_rqvae.py` imports. The root is
found by walking up to the directory that holds `train_rqvae.py` and `distances/`: no
parent counting, no absolute path.

Sizes: `CODEBOOK_DIM` is the mechanism's `d = 32` (`D_E = D_H = 16`), so the tiny
config cannot shrink it -- a 4-dim latent is a hard error in the geometry, not a test
convenience. Only the codebook sizes and the number of levels are shrunk.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

CODEBOOK_DIM = 32
D_E = 16
D_H = 16
SCALE_HYP = 0.25


def _tree_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "train_rqvae.py").is_file() and (parent / "distances").is_dir():
            return parent
    raise RuntimeError("cannot locate the candidate tree root above " + str(here))


TREE_ROOT = _tree_root()
if str(TREE_ROOT) not in sys.path:
    sys.path.insert(0, str(TREE_ROOT))


def repository_root() -> Path:
    """The repo root, found by the marker `stage2_RQ-VAE` (for the byte-identity tests)."""
    for parent in TREE_ROOT.parents:
        if parent.name == "stage2_RQ-VAE":
            return parent.parent
    raise RuntimeError("cannot locate the repository root above " + str(TREE_ROOT))


BASELINE_TREE = TREE_ROOT.parent / "TIGER_RQ-VAE"


@pytest.fixture(scope="session")
def tree_root() -> Path:
    return TREE_ROOT


@pytest.fixture(scope="session")
def baseline_tree() -> Path:
    if not (BASELINE_TREE / "train_rqvae.py").is_file():
        raise RuntimeError("the frozen baseline tree is missing at " + str(BASELINE_TREE))
    return BASELINE_TREE


@pytest.fixture
def curvature_table() -> torch.Tensor:
    """One curvature per item row, inside the mechanism's range [0.25, 4.0]."""
    return torch.tensor([0.25, 0.35, 0.6, 1.0, 1.0, 1.7, 2.6, 4.0], dtype=torch.float32)


@pytest.fixture
def geometry(curvature_table):
    from model.per_item_curvature_geometry import PerItemCurvatureGeometry

    return PerItemCurvatureGeometry(curvature_table)


@pytest.fixture
def tiny_config() -> SimpleNamespace:
    """The tokeniser config `train_rqvae.py:_tokenizer_config()` builds, shrunk."""
    return SimpleNamespace(
        hidden_sizes=[8],
        codebook_num=2,
        codebook_size=(4, 5),
        codebook_dim=CODEBOOK_DIM,
        dropout=0.0,
        beta=0.25,
        vq_type="vq",
        ema_decay=0.99,
        fix_code_embs=False,
        sk_epsilon=0.003,
        sk_iters=3,
    )


@pytest.fixture
def tensor_pair():
    """A deterministic `(latent, codebook, rows)` triple over the 8-table rows."""
    generator = torch.Generator().manual_seed(42)
    latent = torch.randn(8, CODEBOOK_DIM, generator=generator) * 0.3
    codebook = torch.randn(6, CODEBOOK_DIM, generator=generator) * 0.3
    rows = torch.arange(8)
    return latent, codebook, rows


@pytest.fixture
def tiny_metric(curvature_table):
    from distances import build_distance

    return build_distance(name="per_item_curvature", curvature_table=curvature_table)


def reference_cost(latent, codebook, curvature):
    """`D_ik` straight from `math.json`, using the contract at the row's OWN curvature.

    Deliberately NOT the scaled rewrite the implementation uses: this calls
    `distance(., ., c_i)` with the real `c_i`, so it is a second derivation
    (`product_cost`) and the implementation is checked against
    `row_radius_equivalence` holding.
    """
    from model.poincare_contract import distance, expmap0

    c = torch.as_tensor(curvature, dtype=torch.float32).reshape(-1, 1, 1)
    r_e, r_h = latent[:, :D_E], latent[:, D_E:]
    t_e, t_h = codebook[:, :D_E], codebook[:, D_E:]
    euclid = (r_e.unsqueeze(1) - t_e.unsqueeze(0)).pow(2).sum(-1)
    x = expmap0(c * 0 + c.sqrt() * r_h.unsqueeze(1), c)     # (B, 1, H)
    y = expmap0(c.sqrt() * t_h.unsqueeze(0), c)             # (B, K, H)
    return euclid + SCALE_HYP * distance(x, y, c).pow(2)
