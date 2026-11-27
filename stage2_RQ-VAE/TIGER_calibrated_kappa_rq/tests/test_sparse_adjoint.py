"""Exact paired CSR adjoints must avoid native transpose conversion in backward."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from test_static_means import TOKEN_POPULATIONS, TYPE_ENCODING, four_token_fixture

SPEC = importlib.util.spec_from_file_location(
    'structure252_adjoint', Path(__file__).resolve().parents[1] / 'model/structure.py')
structure = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(structure)


def directional_means(*args):
    from tools.issue253.sparse_adjoint import normalized_directional_means
    return normalized_directional_means(*args)


def dense_operators(n, edges, dtype=torch.double, device='cpu'):
    adjacency = torch.zeros(n, n, dtype=dtype, device=device)
    for source, target in edges:
        adjacency[source, target] = 1
    incoming = adjacency.T / adjacency.T.sum(1, keepdim=True).clamp_min(1)
    outgoing = adjacency / adjacency.sum(1, keepdim=True).clamp_min(1)
    return incoming, outgoing


class NoCSRConversion(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func == torch.ops.aten._to_sparse_csr.default:
            raise AssertionError('backward rebuilt sorted CSR support with aten._to_sparse_csr')
        return func(*args, **(kwargs or {}))


class TinyStore:
    n = 4

    def node_features(self):
        return np.array([[1., 3., 2., 7., 0., 2.], [4., 1., 5., 2., 1., 4.],
                         [2., 8., 1., 4., 3., 1.], [9., 2., 3., 1., 2., 8.]])

    def collate(self, items, hops=1, standardized=False):
        nodes, groups, edges = [], [], []
        for group, item in enumerate(items):
            graph = ((0, 1), (0, 2), (2, 1))
            if hops == 'in_only':
                block = sorted({item} | {a for a, b in graph if b == item})
            elif hops == 'out_only':
                block = sorted({item} | {b for a, b in graph if a == item})
            else:
                block = ([] if hops == 1 and item == 2 else [3] if item == 3 else
                         [item, (item + 1) % 3] if hops == 1 else [0, 1, 2])
            start = len(nodes)
            local = {node: start + index for index, node in enumerate(block)}
            nodes.extend(block)
            groups.extend([group] * len(block))
            edges.extend((local[a], local[b]) for a, b in ((0, 1), (0, 2), (2, 1))
                         if a in local and b in local)
        return dict(items=list(items), nodes=np.asarray(nodes, dtype=np.int64),
                    features=self.node_features()[nodes], batch_index=np.asarray(groups, dtype=np.int64),
                    node_mask=np.ones(len(nodes), dtype=bool),
                    edges=np.asarray(edges, dtype=np.int64).reshape(-1, 2),
                    edge_mask=np.ones(len(edges), dtype=bool))


def dense_tokens(model, store, ids, features):
    scales, validity = [], []
    for token_index, hops in enumerate(TOKEN_POPULATIONS):
        rows, valid = [], []
        for item in ids:
            batch = store.collate([item], hops=hops)
            if not len(batch['nodes']):
                rows.append(TYPE_ENCODING[token_index].to(features))
                valid.append(False)
                continue
            incoming, outgoing = dense_operators(
                len(batch['nodes']), batch['edges'], features.dtype, features.device)
            x = model.input(features[batch['nodes']])
            for layer in model.layers:
                x = torch.tanh(layer.self_linear(x) + layer.incoming(incoming @ x)
                               + layer.outgoing(outgoing @ x))
            rows.append(x.mean(0) + TYPE_ENCODING[token_index].to(features))
            valid.append(True)
        scales.append(torch.stack(rows))
        validity.append(torch.tensor(valid, device=features.device))
    return torch.stack(scales, 1), torch.stack(validity, 1)


class SparseAdjointTests(unittest.TestCase):
    # The four-rank diagnostic entry sets this after validating launcher context.
    production_device = 'cpu'

    def test_hand_directed_means_and_both_exact_adjoints(self):
        incoming, outgoing = dense_operators(4, ((0, 1), (0, 2), (2, 1)))
        x = torch.tensor([[1.], [2.], [4.], [8.]], dtype=torch.double, requires_grad=True)
        mean_in, mean_out = directional_means(x, incoming.to_sparse_csr(), outgoing.to_sparse_csr())
        torch.testing.assert_close(mean_in, torch.tensor([[0.], [2.5], [1.], [0.]], dtype=x.dtype))
        torch.testing.assert_close(mean_out, torch.tensor([[3.], [0.], [2.], [0.]], dtype=x.dtype))
        grad_in = torch.tensor([[2.], [3.], [5.], [7.]], dtype=x.dtype)
        grad_out = torch.tensor([[11.], [13.], [17.], [19.]], dtype=x.dtype)
        with NoCSRConversion():
            gradient, = torch.autograd.grad((mean_in, mean_out), x, (grad_in, grad_out))
        torch.testing.assert_close(gradient, torch.tensor([[6.5], [22.5], [7.], [0.]], dtype=x.dtype))

    def test_dense_values_first_gradients_and_gradgrad_irregular_empty_isolated(self):
        devices = ['cpu']
        for device in devices:
            for dtype in (torch.float, torch.double):
                for n, edges in ((6, ((0, 1), (0, 2), (0, 3), (2, 1), (3, 2), (3, 3))),
                                 (4, ()), (0, ())):
                    with self.subTest(device=device, dtype=dtype, n=n, edges=edges):
                        torch.manual_seed(252)
                        incoming, outgoing = dense_operators(n, edges, dtype, device)
                        x = torch.randn(n, 3, dtype=dtype, device=device, requires_grad=True)
                        operators = incoming.to_sparse_csr(), outgoing.to_sparse_csr()
                        with NoCSRConversion():
                            actual = directional_means(x, *operators)
                        expected = incoming @ x, outgoing @ x
                        atol, rtol = (1e-5, 1e-4) if dtype == torch.float else (1e-11, 1e-10)
                        for got, want in zip(actual, expected):
                            self.assertEqual(got.dtype, x.dtype)
                            self.assertEqual(got.device, x.device)
                            torch.testing.assert_close(got, want, atol=atol, rtol=rtol)
                        weights = tuple(torch.randn_like(value) for value in actual)
                        actual_loss = sum((value.tanh() * weight).sum() for value, weight in zip(actual, weights))
                        expected_loss = sum((value.tanh() * weight).sum() for value, weight in zip(expected, weights))
                        with NoCSRConversion():
                            actual_grad, = torch.autograd.grad(actual_loss, x, create_graph=True)
                            actual_second, = torch.autograd.grad(actual_grad.sum(), x)
                        expected_grad, = torch.autograd.grad(expected_loss, x, create_graph=True)
                        expected_second, = torch.autograd.grad(expected_grad.sum(), x)
                        torch.testing.assert_close(actual_grad, expected_grad, atol=atol, rtol=rtol)
                        torch.testing.assert_close(actual_second, expected_second, atol=atol, rtol=rtol)

    def test_either_unused_output_and_noncontiguous_inputs(self):
        incoming, outgoing = dense_operators(5, ((0, 1), (0, 2), (2, 1), (3, 0)))
        operators = incoming.to_sparse_csr(), outgoing.to_sparse_csr()
        for direction in (0, 1):
            with self.subTest(direction=direction):
                x = torch.arange(15., dtype=torch.double).reshape(3, 5).T.requires_grad_()
                self.assertFalse(x.is_contiguous())
                mean = directional_means(x, *operators)[direction]
                weight = torch.arange(15., dtype=torch.double).reshape(5, 3)
                with NoCSRConversion():
                    gradient, = torch.autograd.grad((mean * weight).sum(), x)
                torch.testing.assert_close(gradient, (incoming, outgoing)[direction].T @ weight)
                with torch.no_grad():
                    detached = directional_means(x, *operators)
                self.assertTrue(all(not value.requires_grad for value in detached))

    def test_production_backward_avoids_native_csr_transpose_conversion(self):
        torch.manual_seed(253)
        device = self.production_device
        model, store = structure.SubgraphTokens().to(device=device, dtype=torch.float), TinyStore()
        ids = [3, 0, 3, 2, 1]
        with torch.no_grad():
            four_token_fixture(structure, model, store)
            model.tokens_for_items(store, ids, device, chunk=2)
        features = store.node_features()
        normalized = torch.tensor(
            (features - features.mean(0)) / features.std(0), dtype=torch.float, device=device)
        expected, expected_valid = dense_tokens(model, store, ids, normalized)
        actual, valid = model.tokens_for_items(store, ids, device, chunk=2)
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
        self.assertTrue(torch.equal(valid, expected_valid))
        weights = torch.randn_like(actual)
        # Native C++ sparse autograd can run below Python dispatch modes.
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
            with NoCSRConversion():
                actual_gradients = torch.autograd.grad((actual * weights).sum(), tuple(model.parameters()))
        conversions = sum(event.count for event in profile.key_averages()
                          if event.key == 'aten::_to_sparse_csr')
        self.assertEqual(conversions, 0,
                         'production backward rebuilt sorted CSR support %d times' % conversions)
        expected_gradients = torch.autograd.grad((expected * weights).sum(), tuple(model.parameters()))
        for got, want in zip(actual_gradients, expected_gradients):
            torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-4)

    def test_production_live_input_and_all_parameter_gradients(self):
        from tools.issue252.ego_operators import DeviceLayout, batch_operators, pack_synthetic
        torch.manual_seed(254)
        model, store = structure.SubgraphTokens().double(), TinyStore()
        layout = DeviceLayout(pack_synthetic(store, 2), 'cpu')
        nodes, groups, sizes, incoming, outgoing = batch_operators(layout, torch.tensor([0, 3]), torch.double)
        features = torch.randn(store.n, 6, dtype=torch.double, requires_grad=True)
        local = features[nodes]
        # Independent dense static means keep the live feature-gradient reference.
        static = (incoming.to_dense() @ local, outgoing.to_dense() @ local,
                  incoming.crow_indices().diff() > 0, outgoing.crow_indices().diff() > 0)
        from tools.issue253.sparse_adjoint import normalized_directional_means
        actual = model._read_sparse(local, groups, sizes, static, model._first_affine(),
                                    lambda values: normalized_directional_means(values, incoming, outgoing))
        expected, _ = dense_tokens(model, store, [0, 3], features)
        expected_raw = expected[:, 1] - TYPE_ENCODING[1].to(expected)
        torch.testing.assert_close(actual, expected_raw, atol=1e-10, rtol=1e-9)
        weights = torch.randn_like(actual)
        variables = (features, *model.parameters())
        with NoCSRConversion():
            actual_gradients = torch.autograd.grad((actual * weights).sum(), variables)
        expected_gradients = torch.autograd.grad((expected_raw * weights).sum(), variables)
        for got, want in zip(actual_gradients, expected_gradients):
            torch.testing.assert_close(got, want, atol=1e-10, rtol=1e-9)


if __name__ == '__main__':
    unittest.main()
