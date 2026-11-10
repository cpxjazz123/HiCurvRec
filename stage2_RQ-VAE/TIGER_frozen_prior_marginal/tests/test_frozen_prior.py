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


def test_the_realizability_gate_passes_for_the_frozen_marginal(frozen):
    metadata = frozen["metadata"]
    assert metadata["realised_error"] <= prior.REALISED_TOLERANCE, metadata
    assert metadata["method"] in ("affinity", "equal_fallback")


def test_the_realizability_gate_rejects_a_marginal_the_plan_cannot_realise():
    """A 20x bucket ratio at the frozen epsilon cannot be realised: the gate must say so."""
    rows, codes = 512, 32
    generator = torch.Generator().manual_seed(150)
    cost = torch.randn(rows, codes, generator=generator, dtype=torch.float64)
    centred = prior.centred_cost(cost)
    partition = prior.equal_partition(codes, 4)
    reachable = prior.target_marginal(partition, [0.25] * 4)
    extreme = prior.target_marginal(partition, [0.7, 0.1, 0.1, 0.1])
    got = prior.structured_sinkhorn(centred, 0.003, 50, extreme)
    assert float(prior.realised_error(got, extreme)) > prior.REALISED_TOLERANCE
    # the uniform target is realisable, which is the control
    flat = prior.structured_sinkhorn(centred, 0.003, 50, reachable)
    assert float(prior.realised_error(flat, reachable)) <= 1.0


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
