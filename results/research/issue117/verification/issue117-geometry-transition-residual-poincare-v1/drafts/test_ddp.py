import os
import sys
import tempfile
import unittest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from pathlib import Path

TREE = Path(os.environ.get('CANDIDATE_TREE', Path(__file__).resolve().parents[1])).resolve()
sys.path.insert(0, str(TREE))
from model import RQVAE
from train_rqvae import _tokenizer_config

def _rank_worker(rank, rendezvous, output):
    dist.init_process_group('gloo', init_method='file://' + rendezvous, rank=rank, world_size=2)
    try:
        torch.manual_seed(1234)
        cfg = _tokenizer_config()
        cfg.hidden_sizes = (8,)
        cfg.codebook_num = 1
        cfg.codebook_size = (4,)
        cfg.codebook_dim = 3
        model = RQVAE(cfg, in_dim=5)
        # Actual multi-process DDP wrapping and gradient synchronization.
        model = torch.nn.parallel.DistributedDataParallel(model)
        x = torch.tensor([[rank + 1., 2., 3., 4., 5.], [2., 1., 0., 1., 2.]])
        all_x = torch.cat((x, torch.ones(2, 5)), dim=0)
        anchor = torch.tensor([0, 1])
        target = torch.tensor([1, 2])
        loss, _, _, _ = model(x, all_embeddings=all_x, edge_anchor=anchor, edge_target=target)
        loss.backward()
        grads = [p.grad.detach().clone() for p in model.module.parameters() if p.grad is not None]
        params = [p.detach().clone() for p in model.module.parameters()]
        local = torch.cat([v.reshape(-1) for v in grads + params])
        gathered = [torch.empty_like(local) for _ in range(2)]
        # Gather each rank's independently computed post-backward values.
        dist.all_gather(gathered, local)
        if rank == 0:
            torch.save([v.cpu() for v in gathered], output)
        # Make the optimizer update on both ranks, from each rank's local batch.
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        opt.step()
        updated = torch.cat([p.detach().reshape(-1) for p in model.module.parameters()])
        gathered_updates = [torch.empty_like(updated) for _ in range(2)]
        dist.all_gather(gathered_updates, updated)
        if rank == 0:
            torch.save([v.cpu() for v in gathered_updates], output + '.updated')
    finally:
        dist.destroy_process_group()

class CandidateDDPTest(unittest.TestCase):
    def test_two_real_ranks_gather_equal_gradients_parameters_and_updates(self):
        self.assertTrue(torch.distributed.is_available())
        with tempfile.TemporaryDirectory() as d:
            rendezvous = os.path.join(d, 'rendezvous')
            output = os.path.join(d, 'gather.pt')
            mp.spawn(_rank_worker, args=(rendezvous, output), nprocs=2, join=True)
            gathered = torch.load(output, map_location='cpu', weights_only=True)
            updated = torch.load(output + '.updated', map_location='cpu', weights_only=True)
            self.assertEqual(len(gathered), 2)
            self.assertFalse(torch.equal(gathered[0], torch.zeros_like(gathered[0])))
            torch.testing.assert_close(gathered[0], gathered[1], rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(updated[0], updated[1], rtol=1e-5, atol=1e-6)

if __name__ == '__main__':
    unittest.main()
