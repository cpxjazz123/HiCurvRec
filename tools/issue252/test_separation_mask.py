"""Exact separation minima without one tiny device write per codeword."""
import importlib.util
import math
from pathlib import Path
import unittest

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from tools.research.stage2.anticollapse import floor

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / 'tools/research/stage2/contract.py').is_file())


class MaskWrites(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.writes = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func in (torch.ops.aten.copy_.default, torch.ops.aten.fill_.Scalar):
            self.writes += 1
        return func(*args, **(kwargs or {}))


def implementations():
    spec = importlib.util.spec_from_file_location('issue252_floor_mask_candidate',
        ROOT / 'stage2_RQ-VAE/_archive/TIGER_gnn_attention_sectional_rq/assignment_floor.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return (floor._min_pairwise_distance, module._min_pairwise_distance)


class SeparationMaskTests(unittest.TestCase):
    def test_partial_blocks_exclude_only_self_and_preserve_cross_block_minima(self):
        points = torch.tensor([[0., 0.], [3., 0.], [3., 4.], [0., 4.], [0., 4.25]],
                              dtype=torch.float64)
        for implementation in implementations():
            for block in (1, 2, 4, 512):
                with self.subTest(implementation=implementation.__module__, block=block):
                    self.assertEqual(implementation(points, block), .25)
                    duplicated = torch.cat((points, points[:1]))
                    self.assertEqual(implementation(duplicated, block), 0.)
            self.assertTrue(math.isinf(implementation(points[:0], 2)))
            self.assertTrue(math.isinf(implementation(points[:1], 2)))

    def test_real_256_word_guard_masks_with_one_device_write_per_block(self):
        points = torch.arange(256, dtype=torch.float64)[:, None]
        for implementation in implementations():
            for block in (127, 512):
                with self.subTest(implementation=implementation.__module__, block=block):
                    with MaskWrites() as measured:
                        value = implementation(points, block)
                    self.assertEqual(value, 1.)
                    self.assertLessEqual(measured.writes, math.ceil(256 / block),
                        'Self masking must issue at most one device write per distance block')


if __name__ == '__main__':
    unittest.main()
