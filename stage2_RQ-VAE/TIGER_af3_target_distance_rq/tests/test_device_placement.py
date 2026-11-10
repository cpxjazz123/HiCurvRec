"""The regression the first dispatch died on: frozen CPU tensors meeting GPU prefixes."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

TREE = Path(__file__).resolve().parents[1]
if str(TREE) not in sys.path:
    sys.path.insert(0, str(TREE))

from model import target_distance as td  # noqa: E402


def test_cpu_targets_with_a_cuda_style_prefix_do_not_mix_devices():
    prefix_e = torch.randn(4, 4) * 0.1
    prefix_h = torch.randn(4, 4) * 0.1
    edges = torch.tensor([[0, 1], [2, 3]])
    targets = torch.tensor([0.2, 0.1], dtype=torch.float64)   # frozen table on the CPU
    reference = torch.tensor(0.3, dtype=torch.float64)
    loss = td.structure_losses([(prefix_e, prefix_h)], edges, targets,
                               torch.zeros(0, 3, dtype=torch.long), reference, 1.0)
    assert loss["struct"].device == prefix_e.device
    assert loss["struct"].dtype == prefix_e.dtype
    loss["struct"].backward  # no exception, a scalar tensor
