"""The prior on the entry point's own paths: export reads it, the checkpoint binds it.

Two properties the framework's probes do not cover: the export must take the last
level's code from the *structured* plan (not from a plain ``argmin``), and a
restored model must refuse a structure prior other than the one its weights saw.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

TREE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TREE))

from model import struct_prior as prior  # noqa: E402
from model.mixed_rq import AlphaTable, MixedRQ, encode_tokens  # noqa: E402

CODES = 8
ITEMS = 8


def _write_prior(path: Path, seed: int = 150) -> Path:
    generator = np.random.default_rng(seed)
    layer = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    rho = np.array([0.25, 0.25, 0.25, 0.25])
    tau = generator.random(ITEMS)
    np.savez_compressed(
        path, tau=tau, layer=layer, rho=rho, item_ids=np.arange(ITEMS),
        community=np.zeros(ITEMS, dtype=np.int64),
        edges=np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]], dtype=np.int32),
        triangles=np.array([[0, 1, 2], [2, 3, 4]], dtype=np.int32),
        edge_layer=np.zeros(7, dtype=np.int64),
        metadata=np.asarray(json.dumps({"n_items": ITEMS})))
    prior.manifest_path(path).write_text(json.dumps(
        {"table": str(path), "table_sha256": prior.sha256_of(path),
         "metadata": {"n_items": ITEMS}}), encoding="utf-8")
    return path


def _quantizer(path: Path) -> MixedRQ:
    table = prior.StructPriorTable.load(path)
    rq = MixedRQ([CODES, CODES, CODES], AlphaTable(), curvature=1.0,
                 sk_epsilon=0.003, sk_iters=50)
    rq.attach_prior(table)
    rq.set_marginal(prior.equal_partition(CODES, 4), table.rho)
    return rq


def test_the_export_takes_the_last_level_from_the_structured_plan(tmp_path):
    rq = _quantizer(_write_prior(tmp_path / "p.npz"))
    rq.eval()
    generator = torch.Generator().manual_seed(3)
    latent = 0.05 * torch.randn(ITEMS, 32, generator=generator)
    with torch.no_grad():
        tokens = encode_tokens(latent, rq)
        # the same weights, a plain argmin: the plans must agree only if the
        # marginal really is what the export uses, so compare against the plan
        z_e, z_t = latent[..., :16], latent[..., 16:]
        from model.poincare_contract import expmap0
        residual_e, residual_h = z_e, expmap0(z_t, rq.curvature)
        tokens_seen: list[torch.Tensor] = []
        for level, layer in enumerate(rq.levels, start=1):
            alpha = rq.alpha_table.lookups(level, tokens_seen).to(latent.device)
            cost = layer.distances(residual_e, residual_h, alpha)
            if level == len(rq.levels):
                centred = layer.center_distance(cost).double()
                index = prior.structured_sinkhorn(centred, layer.sk_epsilon, layer.sk_iters,
                                                  rq.target_marginal.double()).argmax(-1)
            else:
                index = torch.argmin(cost, dim=-1)
            codes = layer.get_code_embs()
            codes_e, codes_h = codes[:, :16], codes[:, 16:]
            picked_e = codes_e[index]
            picked_h = expmap0(codes_h[index], rq.curvature)
            residual_e = residual_e - picked_e
            from model.poincare_contract import mobius_add, project
            residual_h = project(mobius_add(-picked_h, residual_h, rq.curvature), rq.curvature)
            tokens_seen.append(index)
    assert torch.equal(tokens, torch.stack(tokens_seen, dim=1))


def test_the_marginal_lives_on_the_codebooks_device(tmp_path):
    """DDP broadcasts every buffer, so a CPU marginal aborts the wrap."""
    rq = _quantizer(_write_prior(tmp_path / "p.npz"))
    assert rq.target_marginal.device == rq.levels[-1].embed.weight.device
    assert rq.target_marginal.dtype == torch.float32
    rq.set_marginal(prior.equal_partition(CODES, 4), [0.4, 0.3, 0.2, 0.1])
    assert rq.target_marginal.device == rq.levels[-1].embed.weight.device


def test_a_restore_refuses_another_prior(tmp_path):
    from model.model import MixedRQVAE
    from types import SimpleNamespace
    config = SimpleNamespace(hidden_sizes=(8,), codebook_dim=32, dropout=0.0,
                             codebook_size=(CODES, CODES, CODES), codebook_num=3,
                             beta=0.25, sk_epsilon=0.003, sk_iters=50)
    path = _write_prior(tmp_path / "p.npz")
    model = MixedRQVAE(config, in_dim=8, curvature=1.0)
    model.attach_prior(prior.StructPriorTable.load(path))
    state = model.state_dict()
    assert state["rq._extra_state"]["table_sha256"] == prior.sha256_of(path)
    other = _write_prior(tmp_path / "q.npz", seed=7)
    assert prior.sha256_of(other) != prior.sha256_of(path)
    with pytest.raises(ValueError, match="another structure prior"):
        MixedRQVAE.from_checkpoint(config, state, in_dim=8, structure_table=other)
    stripped = dict(state)
    stripped["rq._extra_state"] = {k: v for k, v in state["rq._extra_state"].items()
                                   if k != "table_sha256"}
    with pytest.raises(ValueError, match="does not record its structure prior"):
        MixedRQVAE.from_checkpoint(config, stripped, in_dim=8, structure_table=path)
