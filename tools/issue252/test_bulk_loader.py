"""Bulk fetch preserves the frozen distributed sampler and every emitted sample."""
from pathlib import Path
import sys
import unittest
import torch
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / 'tools/research/stage2/contract.py').is_file())
sys.path.insert(0, str(ROOT / 'stage2_RQ-VAE/TIGER_gnn_attention_sectional_rq'))


class BulkLoaderTests(unittest.TestCase):
    def test_actual_bulk_fetch_matches_all_rank_epochs_and_padding(self):
        from train_rqvae import _BatchedTensorDataset, _batch_passthrough
        features = torch.arange(77, dtype=torch.float32).reshape(11, 7)
        ids = torch.tensor([3, 7, 8, 12, 19, 21, 40, 41, 58, 63, 101])
        original = TensorDataset(features, ids)
        class Bulk(_BatchedTensorDataset):
            def __getitem__(self, index):
                raise AssertionError('avoidable per-item fetch')
        bulk = Bulk(features, ids)
        for epoch in (0, 1):
            for rank in range(4):
                left = DistributedSampler(original, num_replicas=4, rank=rank, seed=42, shuffle=True, drop_last=False)
                right = DistributedSampler(bulk, num_replicas=4, rank=rank, seed=42, shuffle=True, drop_last=False)
                left.set_epoch(epoch); right.set_epoch(epoch)
                expected = list(DataLoader(original, sampler=left, batch_size=2, num_workers=0))
                actual = list(DataLoader(bulk, sampler=right, batch_size=2, num_workers=0,
                                         collate_fn=_batch_passthrough))
                self.assertEqual(len(expected), len(actual))
                for got, want in zip(actual, expected):
                    torch.testing.assert_close(got[0], want[0], rtol=0, atol=0)
                    torch.testing.assert_close(got[1], want[1], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
