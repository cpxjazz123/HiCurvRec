"""The trainer-side allocation: density from TRAIN edges, table within the band.

`test_code_alpha.py` pins the cost arithmetic.  This file pins the two trainer
functions that feed it: the per-item structural density (TRAIN behaviour only,
exact triangle counts) and the refresh that turns a hard assignment plus that
density into a bounded per-codeword table.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

TREE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TREE))

from model.graphlet_allocation import ALPHA_MAX, ALPHA_MIN
from model.mixed_rq import TOTAL_DIM
from model.model import MixedRQVAE


def trainer_module():
    spec = importlib.util.spec_from_file_location("codebook_trainer", TREE / "train_rqvae.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_density_counts_triangles_on_the_train_graph_only() -> None:
    module = trainer_module()
    frame = pd.DataFrame({
        "history": [[0, 1], [1, 2], [0, 2], [3]],
        "target": [2, 0, 1, 4],
    })
    density = module._item_density(frame, n_items=5)
    # Items 0,1,2 form a triangle: every incident edge is in exactly one triangle.
    assert np.allclose(density[[0, 1, 2]], 1.0)
    # Item 3 has one edge (3-4) and no triangle; item 4 has that same edge.
    assert density[3] == 0.0 and density[4] == 0.0
    assert density.shape == (5,)


def test_refresh_writes_a_bounded_monotone_table() -> None:
    module = trainer_module()
    torch.manual_seed(0)
    config = SimpleNamespace(
        hidden_sizes=(16,), codebook_num=2, codebook_size=6, codebook_dim=TOTAL_DIM,
        dropout=0.0, beta=0.25, vq_type="vq", ema_decay=0.99, fix_code_embs=False,
        sk_epsilon=-1.0, sk_iters=-1,
    )
    model = MixedRQVAE(config, in_dim=TOTAL_DIM, curvature=1.0, code_alpha_mode=True)
    embeddings = torch.randn(24, TOTAL_DIM)
    density = np.linspace(0.0, 3.0, 24)
    summary = module._refresh_code_alpha(model, embeddings, density, torch.device("cpu"))
    table = model.rq.code_alpha.detach().numpy()
    used = table[:, :6]
    assert used.min() >= ALPHA_MIN - 1e-6
    assert used.max() <= ALPHA_MAX + 1e-6
    for level in summary["levels"]:
        assert 1 <= level["codes_used"] <= 6
        assert level["alpha_min"] is not None and level["alpha_max"] is not None
    # The refresh is deterministic for the same model and density.
    before = model.rq.code_alpha.clone()
    module._refresh_code_alpha(model, embeddings, density, torch.device("cpu"))
    assert torch.equal(before, model.rq.code_alpha)
