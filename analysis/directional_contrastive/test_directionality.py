from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
import torch

import config
import directionality as d


class DirectionalityTests(unittest.TestCase):
    def frame(self, edges):
        return pd.DataFrame({"seen_history": [[a] for a, b in edges], "target": [b for a, b in edges]})

    def test_t1_graph_is_directed(self):
        graph = d.build_transition_graph(self.frame([(1, 2)]))
        self.assertEqual(graph["edge_counts"][(1, 2)], 1)
        self.assertNotIn((2, 1), graph["edge_counts"])

    def test_t2_reverse_classification(self):
        both = d.classify_pairs(self.frame([(1, 2), (2, 1)]), [(1, 2)])
        one = d.classify_pairs(self.frame([(1, 2)]), [(1, 2)])
        self.assertEqual(both[0], "REVERSE-OBSERVED")
        self.assertEqual(one[0], "REVERSE-UNOBSERVED")

    def test_t3_unobserved_reverse_is_not_negative(self):
        graph = d.build_transition_graph(self.frame([(1, 2)]))
        self.assertNotIn((2, 1), graph["edge_counts"])
        self.assertNotIn((2, 1), graph["negative_edges"])

    def test_t4_score_symmetry_and_directionality_mode(self):
        torch.manual_seed(4)
        a = torch.randn(4, 2, 32)
        c = d.DirectionalHeads()
        checkpoint = config.STAGE2 / config.ARMS["C"]["name"] / "rqvae_best.pth"
        mode = "randomly initialized directional heads"
        if checkpoint.is_file():
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            state = state.get("state_dict", state)
            head_state = {k.split("directional_heads.", 1)[1]: v for k, v in state.items()
                          if k.startswith("directional_heads.")}
            if head_state:
                c.load_state_dict(head_state)
                mode = "arm C checkpoint directional heads"
        self.assertTrue(torch.allclose(d.score_pairs("resample", a), d.score_pairs("resample", a.flip(1))))
        ab = d.score_pairs("resample_proj_dir", a, c)
        ba = d.score_pairs("resample_proj_dir", a.flip(1), c)
        self.assertFalse(torch.allclose(ab, ba))
        print(f"T4 mode: {mode}")

    def test_t5_per_level_metrics_finite_and_shaped(self):
        scores = np.asarray([[0.8, 0.2], [0.7, 0.1], [0.6, 0.4]], dtype=np.float64)
        m = d.summarize_level_scores(scores, np.asarray([0, 0, 0]), batch_size=2)
        self.assertEqual(len(m), 3)
        self.assertTrue(all(np.isfinite(v) for level in m for v in level.values() if isinstance(v, (float, int))))

    def test_t6_enrichment_matches_repository_definition(self):
        levels = np.asarray([[0, 1], [0, 2], [1, 3], [1, 4]])
        positives = [(0, 1), (2, 3)]
        negatives = [(0, 2), (1, 0)]
        metric = d.prefix_sharing_metrics(levels, positives, negatives)["L1"]
        self.assertEqual(metric["positive_share_rate"], 1.0)
        self.assertEqual(metric["negative_share_rate"], 0.5)
        self.assertEqual(metric["er"], 2.0)

    def test_t7_strata_partition(self):
        graph = d.build_transition_graph(self.frame([(1, 2), (2, 1), (1, 3), (3, 4)]))
        pairs = [(1, 2), (2, 3), (3, 4), (4, 1)]
        strata = d.stratify_pairs(graph, pairs)
        flattened = [pair for group in strata.values() for pair in group]
        self.assertEqual(sorted(flattened), sorted(pairs))
        self.assertEqual(len(flattened), len(set(flattened)))

    def test_t8_missing_arm(self):
        arms = d.present_arms({"A": True, "B": False, "C": False})
        self.assertEqual(arms["present"], ["A"])
        self.assertEqual(arms["missing"], ["B", "C"])


if __name__ == "__main__":
    result = unittest.main(verbosity=2, exit=False)
    successful = result.result.wasSuccessful()
    print("PASS" if successful else "FAIL")
    raise SystemExit(0 if successful else 1)
