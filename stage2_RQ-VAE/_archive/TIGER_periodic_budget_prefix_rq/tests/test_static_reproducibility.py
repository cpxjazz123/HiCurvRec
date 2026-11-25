"""Fresh frozen consumers must agree; CUDA known answers cover empty directions."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tools.issue252.ego_operators import DeviceLayout, load_layouts
from tools.issue253.static_features import StaticFeatureMeans

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'assets/manifest.json').is_file())


def test_fresh_static_means_are_identical_for_frozen_complete_egos():
    assert torch.cuda.is_available(), 'this regression requires a launcher-owned CUDA worker'
    # An unchanged prefix of complete frozen egos, not an edge/neighbour sample.
    full = load_layouts(ROOT/'assets')[1]
    stop = int(full['offsets'][256])
    arrays = {'offsets': full['offsets'][:257], 'nodes': full['nodes'][:stop],
              'node_counts': full['node_counts'][:256]}
    for direction in ('in', 'out'):
        ptr = full[direction+'_ptr'][:stop+1]
        arrays[direction+'_ptr'] = ptr
        arrays[direction+'_col'] = full[direction+'_col'][:int(ptr[-1])]
        arrays[direction+'_edge_counts'] = full[direction+'_edge_counts'][:256]
    layout = DeviceLayout(arrays, torch.device('cuda', torch.cuda.current_device()))
    feature_root = ROOT/'assets/v5_gnn_geometry'
    with np.load(feature_root/'node_features.npz', allow_pickle=False) as asset:
        features = asset['features'].copy()
    stats = json.loads((feature_root/'node_features_report.json').read_text())['standardization']
    normalized = (features-np.asarray(stats['mean']))/np.asarray(stats['std'])
    features = torch.tensor(normalized, dtype=torch.float32, device=layout.nodes.device)
    first = StaticFeatureMeans(layout, features)
    second = StaticFeatureMeans(layout, features)
    for direction in ('incoming', 'outgoing'):
        assert torch.equal(getattr(first, direction), getattr(second, direction)), (
            'fresh frozen feature means must be bitwise identical: '+direction)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_cuda_static_means_have_hand_computed_directed_and_empty_answers(dtype):
    assert torch.cuda.is_available(), 'this regression requires a launcher-owned CUDA worker'
    arrays = {'offsets': np.array([0, 3, 4]), 'nodes': np.array([0, 1, 2, 3]),
              'node_counts': np.array([3, 1]),
              'in_ptr': np.array([0, 0, 2, 3, 3]), 'in_col': np.array([0, 2, 0]),
              'out_ptr': np.array([0, 2, 2, 3, 3]), 'out_col': np.array([1, 2, 1]),
              'in_edge_counts': np.array([3, 0]), 'out_edge_counts': np.array([3, 0])}
    layout = DeviceLayout(arrays, torch.device('cuda', torch.cuda.current_device()))
    features = torch.zeros((4, 6), dtype=dtype, device=layout.nodes.device)
    features[:, 0] = torch.tensor([1., 2., 4., 8.], dtype=dtype, device=features.device)
    means = StaticFeatureMeans(layout, features, chunk=1)
    torch.testing.assert_close(means.incoming[:, 0],
        torch.tensor([0., 2.5, 1., 0.], dtype=dtype, device=features.device), rtol=0, atol=0)
    torch.testing.assert_close(means.outgoing[:, 0],
        torch.tensor([3., 0., 2., 0.], dtype=dtype, device=features.device), rtol=0, atol=0)
    assert means.in_nonempty.tolist() == [False, True, True, False]
    assert means.out_nonempty.tolist() == [True, False, True, False]
    assert not means.incoming.requires_grad and not means.outgoing.requires_grad
