"""Floor repair must keep hard IDs without per-round tensor-to-list transfers."""
import unittest
from unittest.mock import patch
import torch
from tools.research.stage2.anticollapse import floor


class FloorVectorizationTests(unittest.TestCase):
    def test_contested_repair_without_host_item_lists(self):
        cost = torch.arange(7, dtype=torch.float32)[:, None] * 10 + torch.arange(4)[None, :]
        with patch.object(torch.Tensor, 'tolist', side_effect=AssertionError('per-round item host transfer')):
            actual, report = floor.constrained_assignment(cost, 1)
        torch.testing.assert_close(actual, torch.tensor([1, 2, 3, 0, 0, 0, 0]), rtol=0, atol=0)
        self.assertEqual(report['moves'], 3)
        self.assertEqual(report['min_count'], 1)

    def test_infeasible_batch_refuses_without_rotation(self):
        with self.assertRaises(floor.InfeasibleFloor):
            floor.constrained_assignment(torch.zeros(2, 3), 1)

    def test_uncontested_and_no_floor_hard_ids(self):
        cost = torch.tensor([[0., 5., 8.], [9., 0., 7.], [6., 2., 0.], [0., 3., 4.]])
        for required in (0, 1):
            actual, report = floor.constrained_assignment(cost, required)
            torch.testing.assert_close(actual, torch.tensor([0, 1, 2, 0]), rtol=0, atol=0)
            self.assertEqual(report['moves'], 0)


if __name__ == '__main__':
    unittest.main()
