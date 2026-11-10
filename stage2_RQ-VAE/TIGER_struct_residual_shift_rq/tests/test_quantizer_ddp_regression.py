"""Two-rank CPU collective regressions, not four-GPU trainer admission.

The KMeans dependency is fault-injected to return different valid centers on
each rank.  This makes a missing rank-0 initialization broadcast deterministic
rather than relying on a lucky/unlucky clustering seed.  The production mixed
quantizer, its residual loop, and actual Gloo collectives run unchanged.

The active-A case runs the real DDP forward, auxiliary gradient-budget readout,
backward, and optimizer update at epoch 2250.  Its independent collective
oracle averages gradients from ordinary, non-DDP replicas of the same starting
parameters.  Rank-local batches differ, so mere rank agreement is not enough:
DDP must produce the correctly averaged gradient, not just a common one.

These tests require a runtime that permits local Gloo sockets.  In the repair
workspace initialization was denied with ``Operation not permitted``; the real
collective evidence remains NOT_RUN, not PASS.  No permission workaround is
part of this fixture.
"""
from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
import copy
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

TREE = next(parent for parent in Path(__file__).resolve().parents
            if (parent / "model" / "mixed_rq.py").is_file())
sys.path.insert(0, str(TREE))

from model.mixed_rq import AlphaTable, MixedRQ, encode_tokens  # noqa: E402


def _initialization_worker(rank: int, rendezvous: str):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=30))
    try:
        torch.manual_seed(42 + rank)
        quantizer = MixedRQ([2, 2, 2], AlphaTable(), curvature=1.0).eval()
        generator = torch.Generator().manual_seed(555)
        latent = 0.03 * torch.randn(12, 32, generator=generator)

        def different_centers(samples, count, *args, **kwargs):
            return samples[:count].clone() + float(rank) * 0.02

        with patch("model.utils.kmeans", side_effect=different_centers):
            quantizer.init_codebooks(latent, torch.device("cpu"))
        centers = torch.cat([layer.embed.weight.detach().flatten()
                             for layer in quantizer.levels])
        gathered = [torch.empty_like(centers) for _ in range(2)]
        dist.all_gather(gathered, centers)
        if not torch.equal(gathered[0], gathered[1]):
            raise AssertionError("rank-specific initial codebooks escaped the rank-0 broadcast")
        tokens = encode_tokens(latent, quantizer)
        token_tables = [torch.empty_like(tokens) for _ in range(2)]
        dist.all_gather(token_tables, tokens)
        if not torch.equal(token_tables[0], token_tables[1]):
            raise AssertionError("identical items received different SIDs on different ranks")
    finally:
        dist.destroy_process_group()


def _active_auxiliary_worker(rank: int, rendezvous: str):
    """A small one-step diagnostic, not the frozen-data trainer protocol."""
    from model.af3_auxiliary import augment_loss, prepare_state
    from model.model import MixedRQVAE

    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=30))
    try:
        torch.manual_seed(42 + rank)
        config = SimpleNamespace(hidden_sizes=(16,), codebook_num=3,
                                 codebook_size=(4, 4, 4), codebook_dim=32,
                                 dropout=0.0, beta=0.25, sk_epsilon=0.003,
                                 sk_iters=50)
        raw = MixedRQVAE(config, in_dim=8).train()
        with torch.no_grad():
            for layer in raw.rq.levels:
                layer.embed.weight.mul_(0.02)
        wrapped = torch.nn.parallel.DistributedDataParallel(raw)
        if not raw.rq.levels[-1].use_sk:
            raise AssertionError("the active fixture must exercise training's last-level Sinkhorn")

        generator = torch.Generator().manual_seed(116)
        embeddings = torch.randn(20, 8, generator=generator)
        graph = {"n_nodes": 6, "item_ids": np.array([2, 5, 9, 11, 13, 16]),
                 "rows": np.arange(5), "cols": np.arange(1, 6),
                 "weights": np.array([1.0, 5.0, 3.0, 7.0, 0.0]),
                 "af3_median": 3.0, "af3_p90": 7.0}
        state = prepare_state(graph, embeddings, torch.device("cpu"))
        # Different local examples are essential: a duplicated batch cannot
        # expose a missing or incorrect reduction of parameter gradients.
        batch = embeddings[rank * 4:(rank + 1) * 4]
        reference = copy.deepcopy(raw)
        original_table = copy.deepcopy(raw.rq.alpha_table.to_dict())
        starting = torch.cat([parameter.detach().flatten() for parameter in raw.parameters()])

        # The reference has no DDP reducer hooks.  Its only shared scalar is
        # the adapter's actual global MIN budget, followed by an explicit
        # SUM/world_size oracle for the local gradients.
        reconstruction, quantization, _, _ = reference(batch)
        reference_base, _ = reference.compute_loss(batch, reconstruction, quantization)
        reference_loss, reference_log = augment_loss(
            reference_base, reference, state, 2250, np.random.default_rng(42))
        if not reference_log["active"] or reference_log["ramp"] != 0.5:
            raise AssertionError("the fixture failed to activate the A arm halfway through its ramp")
        if not np.isfinite(reference_log["scale"]) or reference_log["scale"] <= 0:
            raise AssertionError("the fixture's auxiliary gradient contribution must be nonzero")
        if any(parameter.grad is not None for parameter in reference.parameters()):
            raise AssertionError("gradient-budget measurement mutated .grad before backward")
        reference_loss.backward()
        reference_parameters = list(reference.parameters())
        if any(parameter.grad is None for parameter in reference_parameters):
            raise AssertionError("the reference step left a trainable parameter disconnected")
        expected_gradient = torch.cat([parameter.grad.detach().flatten()
                                       for parameter in reference_parameters])
        dist.all_reduce(expected_gradient, op=dist.ReduceOp.SUM)
        expected_gradient /= 2

        # This is the real integration path.  In particular, augment_loss's
        # autograd.grad probes occur after a real DDP forward and before its
        # final backward, where reducer hook/unused-parameter bugs can surface.
        optimizer = torch.optim.SGD(wrapped.parameters(), lr=0.01)
        optimizer.zero_grad(set_to_none=True)
        reconstruction, quantization, _, _ = wrapped(batch)
        base, _ = raw.compute_loss(batch, reconstruction, quantization)
        total, record = augment_loss(base, raw, state, 2250, np.random.default_rng(42))
        if not record["active"] or record["ramp"] != 0.5:
            raise AssertionError("the DDP step did not activate A supervision")
        if not np.isclose(record["scale"], reference_log["scale"], rtol=1e-6, atol=1e-10):
            raise AssertionError("DDP hooks changed the measured auxiliary gradient budget")
        if any(parameter.grad is not None for parameter in raw.parameters()):
            raise AssertionError("A's read-only gradient probes accumulated DDP .grad")
        if not torch.isfinite(total) or float((total - base).detach()) <= 0:
            raise AssertionError("the active A loss must be finite and nontrivial")
        total.backward()
        parameters = list(raw.parameters())
        if any(parameter.grad is None for parameter in parameters):
            raise AssertionError("the active DDP step left a trainable parameter disconnected")
        actual_gradient = torch.cat([parameter.grad.detach().flatten() for parameter in parameters])
        if not torch.isfinite(actual_gradient).all():
            raise AssertionError("the active DDP step produced non-finite gradients")
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-4, atol=1e-6)
        peer_gradients = [torch.empty_like(actual_gradient) for _ in range(2)]
        dist.all_gather(peer_gradients, actual_gradient)
        torch.testing.assert_close(peer_gradients[0], peer_gradients[1], rtol=0, atol=0)

        optimizer.step()
        updated = torch.cat([parameter.detach().flatten() for parameter in parameters])
        torch.testing.assert_close(updated, starting - 0.01 * expected_gradient,
                                   rtol=1e-6, atol=1e-7)
        if torch.equal(updated, starting):
            raise AssertionError("the active diagnostic did not update any parameter")
        peer_parameters = [torch.empty_like(updated) for _ in range(2)]
        dist.all_gather(peer_parameters, updated)
        torch.testing.assert_close(peer_parameters[0], peer_parameters[1], rtol=0, atol=0)
        if raw.rq.alpha_table.to_dict() != original_table:
            raise AssertionError("A supervision must not change the fixed alpha table")
        raw.eval()
        identifiers = raw.get_indices(embeddings)
        peer_ids = [torch.empty_like(identifiers) for _ in range(2)]
        dist.all_gather(peer_ids, identifiers)
        if not torch.equal(peer_ids[0], peer_ids[1]):
            raise AssertionError("the active A step broke cross-rank global item IDs")
    finally:
        dist.destroy_process_group()


class InitialCodebookBroadcastTests(unittest.TestCase):
    def test_two_cpu_ranks_share_codebooks_before_residual_initialization(self):
        if not dist.is_available() or not dist.is_gloo_available():
            self.fail("Gloo is required for this regression; NOT_RUN is not a pass")
        with tempfile.TemporaryDirectory(prefix="mixed-rq-ddp-test-") as directory:
            rendezvous = (Path(directory) / "rendezvous").as_uri()
            mp.spawn(_initialization_worker, args=(rendezvous,), nprocs=2, join=True)


class ActiveAuxiliaryDDPTests(unittest.TestCase):
    def test_two_cpu_ranks_average_active_a_gradients_and_updates(self):
        if not dist.is_available() or not dist.is_gloo_available():
            self.fail("Gloo is required for this regression; NOT_RUN is not a pass")
        with tempfile.TemporaryDirectory(prefix="mixed-rq-active-a-ddp-test-") as directory:
            rendezvous = (Path(directory) / "rendezvous").as_uri()
            mp.spawn(_active_auxiliary_worker, args=(rendezvous,), nprocs=2, join=True)


if __name__ == "__main__":
    unittest.main()
