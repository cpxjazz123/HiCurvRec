"""Test fixtures for the AF3 curvature-drive arm.

The candidate tree is a self-contained package root (`distances`, `losses`, `model`,
`af3_potential` live beside this `tests/`), so the tests put that directory on
`sys.path` and import exactly what the trainer imports. The root is found by walking up
to the directory that holds `train_rqvae.py` and `distances/`: no parent counting, no
absolute path.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def _tree_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "train_rqvae.py").is_file() and (parent / "distances").is_dir():
            return parent
    raise RuntimeError("cannot locate the candidate tree root above " + str(here))


TREE_ROOT = _tree_root()
if str(TREE_ROOT) not in sys.path:
    sys.path.insert(0, str(TREE_ROOT))


@pytest.fixture(scope="session")
def tree_root() -> Path:
    return TREE_ROOT


@pytest.fixture
def potential_module():
    """A fresh `CodewordPotential` with a hand-written table and no training."""
    from model.codeword_potential import CodewordPotential

    return CodewordPotential((3, 3), num_items=6)


@pytest.fixture
def tiny_config() -> SimpleNamespace:
    """The tokeniser config `train_rqvae.py:_tokenizer_config()` builds, shrunk."""
    return SimpleNamespace(
        hidden_sizes=[8],
        codebook_num=2,
        codebook_size=(4, 4),
        codebook_dim=4,
        dropout=0.0,
        beta=0.25,
        vq_type="vq",
        ema_decay=0.99,
        fix_code_embs=False,
        sk_epsilon=0.003,
        sk_iters=50,
    )


@pytest.fixture
def tiny_potential():
    """`CodewordPotential` for `tiny_config` over 24 items, with a frozen table."""
    from model.codeword_potential import CodewordPotential

    generator = torch.Generator().manual_seed(7)
    module = CodewordPotential((4, 4), num_items=24)
    module.set_potentials(torch.randn(24, generator=generator))
    return module
