"""CPU repair is legal only when every GPU priority has a unique finite value."""
import unittest
import numpy as np
from tools.research.stage2.anticollapse import floor


class StrictFloorTests(unittest.TestCase):
    def test_contested_unique_priorities_keep_hand_assigned_ids(self):
        costs = np.arange(7, dtype=np.float32)[:, None] * 10 + np.arange(4, dtype=np.float32)[None, :]
        assignment = np.zeros(7, dtype=np.int64)
        result = floor._strict_cpu_repair(costs, assignment, 1)
        np.testing.assert_array_equal(result[0], [1, 2, 3, 0, 0, 0, 0])
        np.testing.assert_array_equal(assignment, [0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(result[1], {'floor': 1, 'moves': 3, 'min_count': 1, 'satisfied': True})

    def test_tied_priorities_refuse_cpu_route_without_mutating_input(self):
        costs = np.zeros((7, 4), dtype=np.float32)
        assignment = np.zeros(7, dtype=np.int64)
        self.assertIsNone(floor._strict_cpu_repair(costs, assignment, 1))
        np.testing.assert_array_equal(assignment, [0, 0, 0, 0, 0, 0, 0])

    def test_nonfinite_priorities_fall_back_instead_of_selecting(self):
        costs = np.full((7, 4), np.inf, dtype=np.float32)
        self.assertIsNone(floor._strict_cpu_repair(costs, np.zeros(7, dtype=np.int64), 1))


if __name__ == '__main__':
    unittest.main()
