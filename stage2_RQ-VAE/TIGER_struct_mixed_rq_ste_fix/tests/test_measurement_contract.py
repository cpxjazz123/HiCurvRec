"""Contract tests for the SID measurement used by question2.

The measurement must report occupancy、effective denominators and branch modes
explicitly.  These cases fail before `probe_measurement` exists.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import probe_measurement as M


def test_occupancy_separates_duplicate_share_from_largest_bucket():
    # buckets [56,2,2,2,1,1]: the doc's synthetic counterexample
    tokens = torch.zeros(64, 3, dtype=torch.long)
    cursor = 0
    for code, size in enumerate([56, 2, 2, 2, 1, 1]):
        tokens[cursor:cursor + size, 0] = code
        cursor += size
    stats = M.occupancy(tokens)
    l1 = stats["L1"]
    assert l1["distinct_prefixes"] == 6
    assert abs(l1["duplicate_item_share"] - 0.96875) < 1e-9
    assert abs(l1["max_bucket_share"] - 0.875) < 1e-9
    assert abs(l1["one_minus_distinct"] - (1 - 6 / 64)) < 1e-9
    assert l1["buckets"][:2] == [56, 2]


def test_edge_stats_report_undefined_without_edges():
    tokens = torch.zeros(4, 3, dtype=torch.long)
    stats = M.edge_stats(tokens, torch.empty(2, 0, dtype=torch.long))
    assert stats["kept_edges"] == 0
    assert stats["L1"]["same_prefix"] == "UNDEFINED"


def test_edge_stats_count_effective_denominator_and_random_expectation():
    tokens = torch.zeros(4, 3, dtype=torch.long)
    tokens[0, 0], tokens[1, 0], tokens[2, 0], tokens[3, 0] = 0, 0, 1, 1
    edges = torch.tensor([[0, 2], [0, 1], [2, 3]])
    stats = M.edge_stats(tokens, edges.t())
    assert stats["kept_edges"] == 3
    assert stats["L1"]["same_prefix_edges"] == 2
    assert abs(stats["L1"]["same_prefix"] - 2 / 3) < 1e-9
    # expected same-bucket share for random pairs: (2*1 + 2*1) / (4*3)
    assert abs(stats["L1"]["random_expectation"] - 4 / 12) < 1e-9


def test_branch_tokens_measure_train_mode_and_export_mode_separately():
    calls = {"sinkhorn": 0}

    class FakeRq:
        def __init__(self):
            self.vq_layers = [SimpleNamespace(use_sk=True)]

        def __call__(self, x, infer_use_sk=False):
            if (owner.train() or infer_use_sk) and self.vq_layers[0].use_sk:
                calls["sinkhorn"] += 1
            return (torch.zeros(x.shape[0], 3, dtype=torch.long),)

    owner = SimpleNamespace(rq=FakeRq(), train=lambda: owner.training)

    class FakeModel:
        training = True

        def get_indices(self, x, infer_use_sk=False):
            return self.rq(x, infer_use_sk=infer_use_sk)[0]

        def train(self, flag=True):
            self.training = flag
            owner.training = flag
            return self

        def __init__(self):
            self.rq = owner.rq

    model = FakeModel()
    owner.training = True
    M.branch_tokens(model, torch.zeros(5, 32), mode="train")
    M.branch_tokens(model, torch.zeros(5, 32), mode="export")
    assert calls["sinkhorn"] == 1, "only the train-mode call may take the SK branch"


if __name__ == "__main__":
    test_occupancy_separates_duplicate_share_from_largest_bucket()
    test_edge_stats_report_undefined_without_edges()
    test_edge_stats_count_effective_denominator_and_random_expectation()
    test_branch_tokens_measure_train_mode_and_export_mode_separately()
    print("ok")
