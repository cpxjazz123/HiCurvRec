"""The fused per-ego kernel must reproduce the sparse CSR operator exactly.

The kernel reads the frozen ego layout directly instead of materialising the
batch's block-diagonal CSR. Its values and gradients have to match the CSR
operator the model used before, within the contract's tolerance.
"""
import importlib.util
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / 'tools/research/stage2/contract.py').is_file())


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ego_operators = _load('ego_operators252', ROOT / 'tools/issue252/ego_operators.py')
ego_kernel = _load('ego_kernel252', ROOT / 'tools/issue252/ego_kernel.py')


class TinyStore:
    n = 5

    def node_features(self):
        return np.array([[1., 3., 2., 7., 0., 2.], [4., 1., 5., 2., 1., 4.],
                         [2., 8., 1., 4., 3., 1.], [9., 2., 3., 1., 2., 8.],
                         [5., 5., 5., 5., 5., 5.]], dtype=np.float64)

    def collate(self, items, hops=1, standardized=False):
        nodes, groups, edges = [], [], []
        for group, item in enumerate(items):
            block = [item, (item + 1) % 3] if item != 4 else [4, 0]
            start = len(nodes)
            local = {node: start + index for index, node in enumerate(block)}
            nodes.extend(block)
            groups.extend([group] * len(block))
            edges.extend((local[block[a]], local[block[b]]) for a, b in ((0, 1), (1, 0)))
        return dict(items=list(items), nodes=np.asarray(nodes, dtype=np.int64),
                    features=self.node_features()[nodes],
                    batch_index=np.asarray(groups, dtype=np.int64),
                    node_mask=np.ones(len(nodes), dtype=bool),
                    edges=np.asarray(edges, dtype=np.int64).reshape(-1, 2),
                    edge_mask=np.ones(len(edges), dtype=bool))


class FusedEgoKernelTests(unittest.TestCase):
    def _layout(self, device):
        store = TinyStore()
        arrays = ego_operators.pack_synthetic(store, 1)
        return ego_operators.DeviceLayout(arrays, device), store

    def test_fused_means_match_the_sparse_csr_operator(self):
        if not torch.cuda.is_available():
            self.fail('CUDA is required: the kernel is the device implementation')
        device = torch.device('cuda:0')
        layout, _ = self._layout(device)
        items = np.array([0, 2, 4], dtype=np.int64)
        plan = layout.plan(items)
        nodes, groups, sizes, incoming, outgoing = ego_operators.batch_operators(
            layout, torch.as_tensor(items, device=device), torch.float32, plan)
        geometry = ego_kernel.batch_geometry(layout, items, device)
        torch.manual_seed(7)
        x = torch.randn((plan.nodes, 32), device=device)
        for direction, operator in (('in', incoming), ('out', outgoing)):
            expected = torch.sparse.mm(operator, x)
            actual = ego_kernel.row_means(
                getattr(layout, direction + '_ptr'), getattr(layout, direction + '_col'),
                x, geometry)
            torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

    def test_fused_adjoint_matches_the_sparse_csr_adjoint(self):
        if not torch.cuda.is_available():
            self.fail('CUDA is required: the kernel is the device implementation')
        device = torch.device('cuda:0')
        layout, _ = self._layout(device)
        items = np.array([0, 2, 4], dtype=np.int64)
        plan = layout.plan(items)
        nodes, groups, sizes, incoming, outgoing = ego_operators.batch_operators(
            layout, torch.as_tensor(items, device=device), torch.float32, plan)
        geometry = ego_kernel.batch_geometry(layout, items, device)
        torch.manual_seed(8)
        x = torch.randn((plan.nodes, 32), device=device, requires_grad=True)
        grad = torch.randn((plan.nodes, 32), device=device)
        # Forward with the sparse operator, then the transpose adjoint.
        y = torch.sparse.mm(outgoing, x)
        expected = torch.autograd.grad(y, x, grad, retain_graph=True)[0]
        # The same adjoint with the fused kernel: the in-support rows carry the
        # out-direction degree of their column.
        scale = ego_kernel.reciprocal_degrees(layout, geometry, 'out', device)
        actual = ego_kernel.row_means(layout.in_ptr, layout.in_col, grad, geometry,
                                      scale=scale)
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

    def test_fused_geometry_is_host_sized(self):
        if not torch.cuda.is_available():
            self.fail('CUDA is required: the kernel is the device implementation')
        device = torch.device('cuda:0')
        layout, _ = self._layout(device)
        items = np.array([0, 2, 4], dtype=np.int64)
        plan = layout.plan(items)
        geometry = ego_kernel.batch_geometry(layout, items, device)
        self.assertEqual(geometry['nodes'], plan.nodes)
        self.assertEqual(int(geometry['row_ego'].numel()), plan.nodes)
        self.assertEqual(int(geometry['ego_base'][0]), 0)
        self.assertEqual(int(geometry['ego_base'][-1]), plan.nodes)
        self.assertEqual(int(geometry['layout_rows'].numel()), plan.nodes)


if __name__ == '__main__':
    unittest.main()
