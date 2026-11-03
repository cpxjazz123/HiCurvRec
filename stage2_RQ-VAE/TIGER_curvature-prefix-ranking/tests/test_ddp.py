"""The distributed step: two ranks, the reducer, and one codebook.

A parameter that requires grad but never receives one kills the reducer on the
second step, and a codebook that is initialised after the DDP wrapper is built
diverges per rank. Both defects survive every single-process check, so this file
runs under ``torchrun --nproc_per_node=2`` and is skipped otherwise.
"""
import os
import sys
import unittest

import torch

if sys.version_info >= (3, 8):
    pass


def _distributed_ready() -> bool:
    return int(os.environ.get("WORLD_SIZE", "0")) > 1 and torch.cuda.device_count() >= 1


@unittest.skipUnless(_distributed_ready(), "run under torchrun --nproc_per_node=2")
class DistributedStepTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        import torch.distributed as dist
        from types import SimpleNamespace
        dist.init_process_group(backend="nccl")
        cls.rank = dist.get_rank()
        cls.local = int(os.environ.get("LOCAL_RANK", cls.rank))
        torch.cuda.set_device(cls.local)
        cls.device = torch.device("cuda", cls.local)
        from model.model import RQVAE
        cls.config = SimpleNamespace(
            hidden_sizes=(32,), codebook_num=3, codebook_size=(16, 16, 16), codebook_dim=8,
            dropout=0.0, beta=0.25, vq_type="vq", ema_decay=0.99, fix_code_embs=False,
            sk_epsilon=0.003, sk_iters=5, frechet_codebook=True, use_behaviour=True,
            use_curvature=True, frechet_eta=1.0)
        cls.RQVAE = RQVAE

    @classmethod
    def tearDownClass(cls):
        import torch.distributed as dist
        dist.destroy_process_group()

    def _model(self):
        import torch.distributed as dist
        from torch.nn.parallel import DistributedDataParallel as DDP
        torch.manual_seed(0)
        model = self.RQVAE(self.config, in_dim=16).to(self.device)
        for layer in model.rq.vq_layers:
            layer.frechet.set_weights(torch.rand(48, device=self.device) * 3 + 0.25)
        with torch.no_grad():
            model.init_codebook(torch.randn(48, 16, device=self.device))
        for layer in model.rq.vq_layers:
            dist.broadcast(layer.get_code_embs().data, 0)
        return DDP(model, device_ids=[self.local], find_unused_parameters=False)

    def test_three_steps_without_an_unused_parameter(self):
        """The reducer must be satisfied on every step, not only the first."""
        import torch.distributed as dist
        model = self._model()
        raw = model.module
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                      lr=1e-3)
        embeddings = torch.randn(48, 16, device=self.device)
        ids = torch.arange(48, device=self.device)
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            recon, quant_loss, _, _ = model(embeddings, ids)
            loss = torch.nn.functional.mse_loss(recon, embeddings) + quant_loss
            loss.backward()
            optimizer.step()
            sums = raw.frechet_step_sums()
            for _, (tangent, weight) in sums:
                dist.all_reduce(tangent)
                dist.all_reduce(weight)
            raw.apply_frechet_codebook_step(sums)
        self.assertTrue(bool(torch.isfinite(torch.tensor(float(loss.detach())))))

    def test_the_codebook_is_identical_on_every_rank(self):
        import torch.distributed as dist
        model = self._model()
        raw = model.module
        embeddings = torch.randn(48, 16, device=self.device)
        ids = torch.arange(48, device=self.device)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                      lr=1e-3)
        optimizer.zero_grad(set_to_none=True)
        recon, quant_loss, _, _ = model(embeddings, ids)
        (torch.nn.functional.mse_loss(recon, embeddings) + quant_loss).backward()
        optimizer.step()
        sums = raw.frechet_step_sums()
        for _, (tangent, weight) in sums:
            dist.all_reduce(tangent)
            dist.all_reduce(weight)
        raw.apply_frechet_codebook_step(sums)
        for layer in raw.rq.vq_layers:
            mean = layer.get_code_embs().detach().clone()
            dist.all_reduce(mean, op=dist.ReduceOp.SUM)
            mean = mean / dist.get_world_size()
            self.assertTrue(torch.allclose(mean, layer.get_code_embs().detach(), atol=1e-6))


if __name__ == "__main__":
    unittest.main()
