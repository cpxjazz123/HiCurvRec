import os, sys, tempfile, unittest
from pathlib import Path
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

TREE = Path("/fs04/ar57/wenyu/GeneRec/stage2_RQ-VAE/TIGER_c117_hyperbolic_item_semantic_id")
sys.path.insert(0, str(TREE))


def _rank_worker(rank, world_size, init_file, result_file):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29673"
    dist.init_process_group("gloo", init_method="file://" + init_file, rank=rank, world_size=world_size)
    try:
        torch.manual_seed(19)
        model = DistributedDataParallel(torch.nn.Linear(3, 2, bias=True))
        # Rank-distinct local data means each rank really computes its own loss/gradient.
        x = torch.tensor([[rank + 1.0, 2.0, -rank - 0.5], [0.25, rank + 0.75, 1.0]])
        y = torch.tensor([[rank * 0.2, 1.0], [0.5, -rank * 0.1]])
        opt = torch.optim.SGD(model.parameters(), lr=0.05)
        opt.zero_grad()
        loss = ((model(x) - y) ** 2).mean()
        loss.backward()
        local_grad = torch.cat([p.grad.detach().reshape(-1) for p in model.parameters()]).clone()
        gathered_grads = [torch.empty_like(local_grad) for _ in range(world_size)]
        dist.all_gather(gathered_grads, local_grad)
        # DDP gradients must be synchronized; this explicitly compares the independently gathered values.
        if not all(torch.allclose(gathered_grads[0], g, atol=1e-7, rtol=1e-6) for g in gathered_grads[1:]):
            raise AssertionError("rank-local gradients differ: " + repr([g.tolist() for g in gathered_grads]))
        opt.step()
        local_params = torch.cat([p.detach().reshape(-1) for p in model.module.parameters()]).clone()
        gathered_params = [torch.empty_like(local_params) for _ in range(world_size)]
        dist.all_gather(gathered_params, local_params)
        if not all(torch.allclose(gathered_params[0], p, atol=1e-7, rtol=1e-6) for p in gathered_params[1:]):
            raise AssertionError("rank-local updated parameters differ")
        # Save each rank's own observed values for the parent test; no broadcast is used.
        torch.save({"rank": rank, "grad": local_grad, "params": local_params}, result_file + "." + str(rank))
    finally:
        dist.destroy_process_group()


class RealDDPConsistencyTest(unittest.TestCase):
    def test_two_real_ranks_gather_independent_gradients_and_updates(self):
        self.assertTrue(dist.is_available())
        self.assertGreaterEqual(torch.get_num_threads(), 1)
        with tempfile.TemporaryDirectory() as d:
            init_file = str(Path(d) / "rendezvous")
            mp.spawn(_rank_worker, args=(2, init_file, str(Path(d) / "rank")), nprocs=2, join=True)
            rows = [torch.load(str(Path(d) / ("rank." + str(i))), weights_only=True) for i in range(2)]
            self.assertEqual([r["rank"] for r in rows], [0, 1])
            self.assertFalse(torch.equal(rows[0]["grad"], torch.zeros_like(rows[0]["grad"])))
            self.assertTrue(torch.allclose(rows[0]["grad"], rows[1]["grad"], atol=1e-7, rtol=1e-6))
            self.assertTrue(torch.allclose(rows[0]["params"], rows[1]["params"], atol=1e-7, rtol=1e-6))
