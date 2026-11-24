"""The floor-constrained assignment must keep its exact greedy result.

The repair rounds now use fixed-size scatters instead of data-dependent index
lists. The assignment they produce has to be identical to the reference
implementation below, which is the original round-for-round rule.
"""
import importlib.util
import unittest
import warnings
from pathlib import Path

import torch

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / 'tools/research/stage2/contract.py').is_file())
TREE = ROOT / 'stage2_RQ-VAE/TIGER_gnn_attention_sectional_rq'


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


floor = _load('floor252_framework', ROOT / 'tools/research/stage2/anticollapse/floor.py')
installed = _load('floor252_installed', TREE / 'assignment_floor.py')


def reference_repair(costs, assignment, floor_value, codewords):
    """The original greedy rounds, kept as an independent reference."""
    items = len(assignment)
    moves = 0
    first_wanted = torch.empty(items, dtype=torch.long, device=costs.device)
    for _ in range(codewords):
        counts_t = torch.bincount(assignment, minlength=codewords)
        deficit = (counts_t < floor_value).nonzero().flatten()
        if deficit.numel() == 0:
            break
        donatable = counts_t[assignment] > floor_value
        sub = costs[:, deficit].masked_fill(~donatable.unsqueeze(1), float('inf'))
        best_cost, best_item = sub.min(dim=0)
        order = torch.argsort(best_cost)
        wanted = best_item[order]
        positions = torch.arange(order.numel(), device=costs.device)
        first_wanted.fill_(codewords)
        first_wanted.scatter_reduce_(0, wanted, positions, reduce='amin', include_self=True)
        keep = (first_wanted[wanted] == positions) & torch.isfinite(best_cost[order])
        selected = keep.nonzero().flatten()
        if selected.numel() == 0:
            raise floor.InfeasibleFloor('no droppable item for the empty codewords')
        chosen_items = wanted[selected]
        assignment[chosen_items] = deficit[order[selected]]
        moves += chosen_items.numel()
    else:
        remaining = int((torch.bincount(assignment, minlength=codewords) < floor_value).sum())
        if remaining:
            raise floor.InfeasibleFloor('repair did not converge')
    return assignment, moves


class AssignmentFloorTests(unittest.TestCase):
    def test_installed_and_framework_copies_are_identical(self):
        first = (ROOT / 'tools/research/stage2/anticollapse/floor.py').read_bytes()
        second = (TREE / 'assignment_floor.py').read_bytes()
        self.assertEqual(first, second, 'the installed floor copy must stay identical')

    def _matrices(self, device):
        generator = torch.Generator(device='cpu').manual_seed(259)
        matrices = []
        for items, codewords, scale in ((64, 32, 1.0), (128, 64, 0.05), (256, 256, 0.2)):
            for draw in range(3):
                matrix = torch.randn((items, codewords), generator=generator) * scale
                matrices.append(matrix.to(device))
        # Concentrated costs leave most codewords empty, as the real batches do.
        concentrated = torch.randn((256, 256), generator=generator) * 0.01
        concentrated[:, :8] += 4.0
        matrices.append(concentrated.to(device))
        return matrices

    def test_repair_matches_the_reference_assignment_on_every_matrix(self):
        if not torch.cuda.is_available():
            self.skipTest('the repair rounds run on the device')
        device = torch.device('cuda:0')
        for matrix in self._matrices(device):
            with self.subTest(shape=tuple(matrix.shape)):
                costs = matrix.detach().float()
                codewords = costs.shape[1]
                start = torch.argmin(costs, dim=-1)
                expected, expected_moves = reference_repair(
                    costs.clone(), start.clone(), 1, codewords)
                actual, report = floor.constrained_assignment(
                    matrix, 1, counts_key=None)
                torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=0, atol=0)
                self.assertEqual(report['moves'], expected_moves)
                counts = torch.bincount(actual.cpu(), minlength=codewords)
                self.assertGreaterEqual(int(counts.min()), 1)

    def test_repair_reads_the_device_at_most_once_per_round(self):
        if not torch.cuda.is_available():
            self.skipTest('the repair rounds run on the device')
        device = torch.device('cuda:0')
        generator = torch.Generator(device='cpu').manual_seed(260)
        matrix = (torch.randn((256, 256), generator=generator) * 0.01).to(device)
        matrix[:, :8] += 4.0
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            torch.cuda.set_sync_debug_mode('warn')
            try:
                floor.constrained_assignment(matrix, 1, counts_key=None)
            finally:
                torch.cuda.set_sync_debug_mode('default')
        syncs = sum(1 for item in caught if 'synchronizing CUDA operation' in str(item.message))
        self.assertLessEqual(syncs, 40,
                             'the repair rounds must not read the device for every list')


if __name__ == '__main__':
    unittest.main()
