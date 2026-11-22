import unittest
from tools.research.science.behavior import recover_train_sequences, analyze_sequences


class BehaviorTests(unittest.TestCase):
    def test_longest_prefix_prevents_double_counting(self):
        rows = [
            {"user": 0, "history": [], "target": 1},
            {"user": 0, "history": [1], "target": 2},
            {"user": 0, "history": [1, 2], "target": 3},
            {"user": 1, "history": [4], "target": 5},
        ]
        sequences = recover_train_sequences(rows)
        self.assertEqual(sequences[0], (1, 2, 3))
        out = analyze_sequences(sequences)
        self.assertEqual(out["events"], 5)
        self.assertEqual(out["transitions"], 3)
        self.assertEqual(out["no_reverse_fraction"], 1.0)

    def test_conflicting_max_prefix_is_rejected(self):
        with self.assertRaises(ValueError):
            recover_train_sequences([
                {"user": 1, "history": [1], "target": 2},
                {"user": 1, "history": [1], "target": 3},
            ])


if __name__ == "__main__":
    unittest.main()
