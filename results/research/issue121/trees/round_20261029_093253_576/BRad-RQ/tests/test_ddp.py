"""The DDP invariants of THIS candidate, with the configuration that trains.

The previous version of this file built the model with frechet_codebook=True,
which is not what the trainer runs: with radial=True and frechet_codebook=False the
layers carry no Frechet codebook at all, so the test was exercising a model that
nobody trains. The checks below are the ones that matter for this mechanism: every
parameter that requires grad is reduced on every step, the radius table is
identical on every rank, and the exported codes agree across the ranks.
"""
import os
import unittest

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

HAVE_DIST = int(os.environ.get("WORLD_SIZE", "0")) > 1


def _config():
    from types import SimpleNamespace
    return SimpleNamespace(
        hidden_sizes=(32,), codebook_num=3, codebook_size=(16, 16, 16), codebook_dim=8,
        dropout=0.0, beta=0.25, vq_type="vq", ema_decay=0.99, fix_code_embs=False,
        sk_epsilon=0.003, sk_iters=5, frechet_codebook=False, use_behaviour=True,
        use_curvature=True, frechet_eta=1.0, radial=True)


@unittest.skipUnless(HAVE_DIST, "run under torchrun --nproc_per_node=2")
class DistributedTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        dist.init_process_group(backend="nccl")
        cls.rank = dist.get_rank()
        cls.local = int(os.environ.get("LOCAL_RANK", cls.rank))
        torch.cuda.set_device(cls.local)
        cls.device = torch.device("cuda", cls.local)

    @classmethod
    def tearDownClass(cls):
        dist.destroy_process_group()

    def _model(self):
        from model.model import RQVAE
        torch.manual_seed(0)
        model = RQVAE(_config(), in_dim=16).to(self.device)
        radius = torch.linspace(0.1, 1.3, 24, device=self.device)
        model.set_radii(radius)
        with torch.no_grad():
            model.init_codebook(torch.randn(24, 16, device=self.device))
        wrapped = DDP(model, device_ids=[self.local], find_unused_parameters=False)
        return model, wrapped, radius

    def test_two_steps_finish_without_a_reducer_error(self):
        """Every parameter that requires grad must be reduced on every step."""
        model, wrapped, radius = self._model()
        optimizer = torch.optim.AdamW([p for p in wrapped.parameters() if p.requires_grad],
                                      lr=1e-3)
        embeddings = torch.randn(24, 16, device=self.device)
        ids = torch.arange(24, device=self.device)
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            recon, quant_loss, _, _ = wrapped(embeddings, ids)
            loss = torch.nn.functional.mse_loss(recon, embeddings) + quant_loss
            loss.backward()
            optimizer.step()
        self.assertTrue(bool(torch.isfinite(loss)))

    def test_the_radius_table_is_identical_on_every_rank(self):
        model, wrapped, radius = self._model()
        table = model.radius_table.detach().clone()
        dist.all_reduce(table)
        table = table / dist.get_world_size()
        self.assertTrue(torch.allclose(table, radius, atol=1e-6))

    def test_the_codes_agree_across_the_ranks(self):
        model, wrapped, radius = self._model()
        embeddings = torch.randn(24, 16, device=self.device)
        ids = torch.arange(24, device=self.device)
        with torch.no_grad():
            codes = model.get_indices(embeddings, infer_use_sk=False, item_ids=ids)
        gathered = codes.clone()
        dist.all_reduce(gathered)
        gathered = gathered / dist.get_world_size()
        self.assertTrue(torch.allclose(gathered.float(), codes.float(), atol=1e-3))


if __name__ == "__main__":
    unittest.main()
