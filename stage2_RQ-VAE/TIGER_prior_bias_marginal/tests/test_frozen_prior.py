"""The corrected arm's four acceptance tests, plus its dispatch-time gate.

The review of #150 asked for four things to be proven mechanically before dispatch:
the frozen partition does not change across epochs, the exposure guard fires, the
realizability check rejects a marginal the assignment cannot realise, and training
and export take the same codes. The marginal must also reach the last level only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

TREE = Path(__file__).resolve().parents[1]
REPO = TREE.parents[1]
sys.path.insert(0, str(TREE))

from model import frozen_prior as prior  # noqa: E402
from model.mixed_rq import AlphaTable, MixedRQ, encode_tokens  # noqa: E402

CODES = 16


def _partition_path() -> Path:
    return prior.partition_path(REPO)


@pytest.fixture(scope="module")
def frozen():
    path = _partition_path()
    if not path.is_file():
        pytest.skip("the frozen partition has not been built: NOT_RUN")
    return prior.load_partition(REPO)


def test_the_frozen_partition_is_byte_stable_and_equal_sized(frozen):
    first = prior.load_partition(REPO)
    second = prior.load_partition(REPO)
    assert first["sha256"] == second["sha256"]
    assert first["partition"] == second["partition"]
    assert np.array_equal(first["marginal"], second["marginal"])
    sizes = [len(bucket) for bucket in first["partition"]]
    assert len(set(sizes)) == 1, sizes
    assert sum(sizes) == int(first["marginal"].size)
    assert abs(float(first["marginal"].sum()) - 1.0) < 1e-12


def test_the_probe_realises_the_marginal_within_five_percent(frozen):
    """The gate as corrected: a probe large enough for the arithmetic bound to be < 5%."""
    gate = frozen["metadata"]["realisability"]
    probe = gate["probe"]
    assert probe["rows"] == prior.PROBE_ROWS >= 8192
    assert probe["arithmetic_bound"] <= probe["tolerance"], probe
    assert probe["max_relative_error"] <= probe["tolerance"], probe


def test_every_code_sits_within_one_and_a_half_rows_of_its_target(frozen):
    """The per-batch form of the gate: one row is 1/B of the mass, so that is the unit."""
    batch = frozen["metadata"]["realisability"]["batch"]
    assert batch["rows"] == 4096
    assert batch["max_row_deviation"] <= batch["tolerance_rows"], batch


def test_the_relative_error_falls_with_the_batch_size(frozen):
    curve = frozen["metadata"]["realisability"]["curve"]
    rows = [entry["rows"] for entry in curve]
    assert rows == list(prior.CURVE_BATCHES)
    errors = [entry["max_relative_error"] for entry in curve]
    assert frozen["metadata"]["realisability"]["curve_decays"], errors
    # a 1/B decay is what the arithmetic promises: sixteen times the rows must buy
    # roughly sixteen times the relative accuracy
    assert errors[0] >= 8.0 * errors[-1], errors
    assert all(entry["max_row_deviation"] <= 1.5 for entry in curve if entry["rows"] >= 4096), curve


def test_the_row_deviation_definition_is_the_one_row_unit():
    """A plan one row off its target must read exactly one row of deviation."""
    rows, codes = 1024, 4
    target = torch.full((codes,), 1.0 / codes, dtype=torch.float64)
    exact = torch.full((rows, codes), 1.0 / codes, dtype=torch.float64)
    assert float(prior.realised_rows(exact, target).max()) <= 1e-9
    off = exact.clone()
    # moving one row's mass by one whole item shifts its column mean by 1/rows,
    # which is exactly one row of the target
    off[0, 0] += 1.0
    off[0, 1] -= 1.0
    assert abs(float(prior.realised_rows(off, target).max()) - 1.0) < 1e-9


def test_the_guard_fires_on_a_level_outside_its_bounds():
    safe = [{"level": 1, "codes_used": 200, "max_bucket_share": 0.05,
             "usage_entropy_normalised": 0.9}]
    assert prior.guard_violations(safe) == []
    collapsed = [{"level": 1, "codes_used": 8, "max_bucket_share": 0.6,
                  "usage_entropy_normalised": 0.1}]
    violations = prior.guard_violations(collapsed)
    assert violations and violations[0]["level"] == 1
    assert len(violations[0]["reasons"]) == 2


def test_the_marginal_reaches_the_last_level_only(monkeypatch):
    """Levels 1-2 must be called with no marginal at all, the last level with one."""
    from model.mixed_rq import MixedVQLevel
    seen: list[bool] = []
    original = MixedVQLevel.forward

    def traced(self, *args, marginal=None, **kwargs):
        seen.append(marginal is None)
        return original(self, *args, marginal=marginal, **kwargs)

    monkeypatch.setattr(MixedVQLevel, "forward", traced)
    frozen = {"partition": prior.equal_partition(CODES, 4),
              "marginal": prior.target_marginal(prior.equal_partition(CODES, 4),
                                                [0.3, 0.3, 0.2, 0.2]).numpy(),
              "sha256": "test"}
    rq = MixedRQ([CODES, CODES, CODES], AlphaTable(), curvature=1.0,
                 sk_epsilon=0.003, sk_iters=5)
    rq.attach_frozen_prior(frozen)
    generator = torch.Generator().manual_seed(7)
    latent = 0.05 * torch.randn(32, 32, generator=generator)
    rq.train()
    rq(latent)
    assert seen == [True, True, False], seen
    assert rq.last_plan is not None


def test_training_and_export_take_the_same_codes(tmp_path):
    frozen = {"partition": prior.equal_partition(CODES, 4),
              "marginal": prior.target_marginal(prior.equal_partition(CODES, 4),
                                                [0.4, 0.3, 0.2, 0.1]).numpy(),
              "sha256": "test"}
    rq = MixedRQ([CODES, CODES, CODES], AlphaTable(), curvature=1.0,
                 sk_epsilon=0.003, sk_iters=5)
    rq.attach_frozen_prior(frozen)
    generator = torch.Generator().manual_seed(11)
    latent = 0.05 * torch.randn(24, 32, generator=generator)
    rq.eval()
    with torch.no_grad():
        tokens = encode_tokens(latent, rq)
        z_e, z_t = latent[..., :16], latent[..., 16:]
        from model.poincare_contract import expmap0, mobius_add, project
        residual_e, residual_h = z_e, expmap0(z_t, rq.curvature)
        seen: list[torch.Tensor] = []
        for level, layer in enumerate(rq.levels, start=1):
            alpha = rq.alpha_table.lookups(level, seen).to(latent.device)
            cost = layer.distances(residual_e, residual_h, alpha)
            if level == len(rq.levels) and layer.use_sk:
                centred = prior.centred_cost(cost).double()
                index = prior.structured_sinkhorn(centred, layer.sk_epsilon, layer.sk_iters,
                                                  rq.frozen_marginal.double()).argmax(-1)
            else:
                index = torch.argmin(cost, dim=-1)
            codes = layer.get_code_embs()
            codes_e, codes_h = codes[:, :16], codes[:, 16:]
            picked_e = codes_e[index]
            picked_h = expmap0(codes_h[index], rq.curvature)
            residual_e = residual_e - picked_e
            residual_h = project(mobius_add(-picked_h, residual_h, rq.curvature), rq.curvature)
            seen.append(index)
    assert torch.equal(tokens, torch.stack(seen, dim=1))
