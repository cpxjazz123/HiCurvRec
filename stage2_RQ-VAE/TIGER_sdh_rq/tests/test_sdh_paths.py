"""Issue #194 acceptance 3: does the distillation readout select the same codes the export writes?

The export path is ``train_rqvae._encode_all``: register every row's curvature, then ``get_indices``.  The distillation step must measure itself on that
selection, not on a second Euclidean one.  ``sd_step.codes_of`` therefore registers the curvature itself, and this file pins that against an independent
implementation of the export path -- plus, as an executable record, against the shipped #181 readout that did not register anything.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

TREE = Path(__file__).resolve().parents[1]
REPO_ROOT = TREE.parents[1]
if str(TREE) not in sys.path:
    sys.path.insert(0, str(TREE))

from model import RQVAE, r48k2_assign, r48k_geometry, sd_step  # noqa: E402


def _tiny_model(seed: int = 0):
    from types import SimpleNamespace

    torch.manual_seed(seed)
    config = SimpleNamespace(
        hidden_sizes=[16, 8], dropout=0.0, codebook_num=3, codebook_size=[16, 16, 16], codebook_dim=8,
        beta=0.25, vq_type="vq", ema_decay=0.99, sk_epsilon=-1.0, sk_iters=-1, use_sk=False,
    )
    model = RQVAE(config, in_dim=8)
    model.eval()
    return model


def export_codes(model, embeddings: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """An inline copy of the trainer's export path, written here so the test does not take its oracle from the tree under test.

    The curvature is the one of the *item ids* whose rows are being selected: ``_encode_all`` registers ``arange(N)`` for the whole catalogue, which is
    the same rule -- every row carries its own item's curvature -- applied to a subset.
    """
    r48k2_assign.set_batch_kappa(r48k_geometry.kappa_of(ids))
    try:
        return model.get_indices(embeddings, infer_use_sk=False)
    finally:
        r48k2_assign.set_batch_kappa(None)


def test_the_distillation_readout_selects_exactly_what_the_export_writes():
    model = _tiny_model()
    ids = torch.tensor([3, 11, 42, 100, 250, 1000, 4999])
    embeddings = torch.randn(5000, 8)
    got = sd_step.codes_of(model, embeddings, ids)
    expected = export_codes(model, embeddings[ids], ids)
    assert torch.equal(got, expected), "the distillation readout and the export disagree"


def test_the_readout_leaves_no_curvature_registered_behind():
    model = _tiny_model()
    embeddings = torch.randn(200, 8)
    sd_step.codes_of(model, embeddings, torch.tensor([1, 2, 3]))
    assert r48k2_assign.batch_kappa() is None, "a stale curvature would change the next, unrelated forward pass"


def test_the_shipped_readout_is_the_one_that_diverged():
    """The red case: #181's readout takes no ids and therefore selects on the Euclidean matrix.

    Its body is ``module.encoder`` + ``module.rq`` with nothing registered, so it is written here rather than imported across trees.
    """
    model = _tiny_model(seed=7)
    embeddings = torch.randn(2000, 8)
    ids = torch.arange(0, 2000, 7)

    r48k2_assign.set_batch_kappa(None)
    z = model.encoder(embeddings[ids])
    _, _, _, shipped_tokens = model.rq(z)

    export_codes_plain = export_codes(model, embeddings[ids], ids)
    assert shipped_tokens.shape == export_codes_plain.shape
    assert torch.equal(shipped_tokens, model.get_indices(embeddings[ids], infer_use_sk=False)), (
        "the id-less readout is by construction the Euclidean selection"
    )
    assert not torch.equal(shipped_tokens, export_codes_plain), (
        "with this model the two paths coincide, so the case would not demonstrate anything"
    )
