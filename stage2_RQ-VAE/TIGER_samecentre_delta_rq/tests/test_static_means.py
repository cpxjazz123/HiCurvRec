"""Independent value/gradient and resource contracts for exact static GNN means."""
import copy
from contextlib import ExitStack
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

SPEC = importlib.util.spec_from_file_location(
    'structure252_static', Path(__file__).resolve().parents[1] / 'model/structure.py')
structure = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(structure)


TOKEN_POPULATIONS = (1, 2, 'in_only', 'out_only')
TYPE_ENCODING = torch.zeros(4, 32)
TYPE_ENCODING[torch.arange(4), torch.arange(4)] = 1.


def four_token_fixture(module, model, store):
    """Mock registered asset loaders with four exact synthetic ego layouts.

    Feature files use the real loader; topology and integrity are mocked only
    for this tiny fixture. No dynamic GNN output is supplied or cached.
    """
    from tools.issue252.ego_operators import pack_synthetic
    layouts = [pack_synthetic(store, population) for population in TOKEN_POPULATIONS]
    with ExitStack() as stack:
        if hasattr(store, 'root'):
            root = Path(store.root)
        else:
            directory = stack.enter_context(tempfile.TemporaryDirectory(
                dir=Path(__file__).resolve().parents[1]))
            root = Path(directory)
            features = store.node_features()
            std = features.std(axis=0)
            np.savez(root / 'node_features.npz', features=features)
            (root / 'node_features_report.json').write_text(json.dumps(
                {'standardization': {'mean': features.mean(axis=0).tolist(),
                                     'std': np.where(std < 1e-12, 1., std).tolist()}}))
        stack.enter_context(patch.object(store, 'repo', root.parent, create=True))
        stack.enter_context(patch.object(store, 'root', root, create=True))
        stack.enter_context(patch.object(module, '_verify_store_assets'))
        stack.enter_context(patch.object(module, 'load_layouts', return_value=layouts[:2]))
        stack.enter_context(patch.object(module, 'load_direction_layouts', return_value=layouts[2:]))
        model._cached_store(store)
    # Keep the first measured forward cold with respect to device/static caches.
    model._feature_cache[store]['devices'].clear()


class TinyStore:
    def __init__(self, n=4, edges=((0, 1), (0, 2), (2, 1)), features=None):
        self.n = n
        self.edges = tuple(sorted(set(edges)))
        self.calls = 0
        self.features = (np.array([[1., 3., 2., 7., 0., 2.], [4., 1., 5., 2., 1., 4.],
                                   [2., 8., 1., 4., 3., 1.], [9., 2., 3., 1., 2., 8.]])
                         if features is None and n == 4 else
                         np.arange(n * 6, dtype=float).reshape(n, 6) if features is None else
                         np.asarray(features))

    def node_features(self):
        return self.features

    def collate(self, items, hops=1, standardized=False):
        self.calls += 1
        nodes, groups, edges = [], [], []
        for group, item in enumerate(items):
            incoming = {a for a, b in self.edges if b == item}
            outgoing = {b for a, b in self.edges if a == item}
            if hops == 'in_only':
                population = {item} | incoming
            elif hops == 'out_only':
                population = {item} | outgoing
            else:
                population, frontier = {item}, {item}
                for _ in range(hops):
                    neighbours = {b for a, b in self.edges if a in frontier}
                    neighbours |= {a for a, b in self.edges if b in frontier}
                    frontier = neighbours - population
                    population |= neighbours
            block = sorted(population)
            start = len(nodes)
            local = {node: start + index for index, node in enumerate(block)}
            nodes.extend(block)
            groups.extend([group] * len(block))
            edges.extend((local[a], local[b]) for a, b in self.edges
                         if a in local and b in local)
        return dict(items=list(items), nodes=np.asarray(nodes, dtype=np.int64),
                    features=self.features[nodes], batch_index=np.asarray(groups, dtype=np.int64),
                    node_mask=np.ones(len(nodes), dtype=bool),
                    edges=np.asarray(edges, dtype=np.int64).reshape(-1, 2),
                    edge_mask=np.ones(len(edges), dtype=bool))


def dense_reference(model, store, ids, features):
    """Original two nonlinear layers, with independently formed dense operators."""
    scales, valid = [], []
    for token_index, hops in enumerate(TOKEN_POPULATIONS):
        rows, masks = [], []
        for item in ids:
            batch = store.collate([item], hops=hops)
            nodes = batch['nodes']
            if not len(nodes):
                rows.append(features.new_zeros(32))
                masks.append(False)
                continue
            a = features.new_zeros((len(nodes), len(nodes)))
            for source, target in batch['edges']:
                a[source, target] += 1
            incoming = a.T / a.T.sum(1, keepdim=True).clamp_min(1)
            outgoing = a / a.sum(1, keepdim=True).clamp_min(1)
            x = model.input(features[nodes])
            for layer in model.layers:
                x = torch.tanh(layer.self_linear(x) + layer.incoming(incoming @ x)
                               + layer.outgoing(outgoing @ x))
            rows.append(x.mean(0) + TYPE_ENCODING[token_index].to(features))
            masks.append(True)
        scales.append(torch.stack(rows))
        valid.append(torch.tensor(masks))
    return torch.stack(scales, 1), torch.stack(valid, 1)


class StaticMeansTests(unittest.TestCase):
    def test_production_retains_activations_and_only_second_layer_uses_wide_sparse_mm(self):
        torch.manual_seed(252)
        model = structure.SubgraphTokens().double()
        store = TinyStore()
        # Static construction is allowed six-wide sparse means, once only.
        with torch.no_grad():
            four_token_fixture(structure, model, store)
            model.tokens_for_items(store, [0, 1, 2, 3], 'cpu', chunk=2)
        widths, activations = [], []
        sparse_mm, tanh = torch.sparse.mm, torch.tanh

        def observe_mm(operator, dense):
            widths.append(dense.shape[1])
            return sparse_mm(operator, dense)

        def observe_tanh(value):
            activations.append(value.shape)
            return tanh(value)

        with patch.object(torch.sparse, 'mm', side_effect=observe_mm), \
                patch.object(torch, 'tanh', side_effect=observe_tanh):
            value, _ = model.tokens_for_items(store, [3, 0, 3, 1, 2], 'cpu', chunk=2)
            forward_activations = len(activations)
            forward_widths = list(widths)
            value.square().sum().backward()
        self.assertEqual(forward_widths, [32] * 16,
                         'Only the second layer may perform two 32-wide sparse means per chunk/token')
        self.assertEqual(forward_activations, 16)
        self.assertEqual(len(activations), forward_activations,
                         'Backward must use retained activations, not checkpoint replay')

    def test_tiny_directed_affine_hand_answer_and_empty_directions(self):
        from tools.issue253.static_features import StaticFeatureMeans
        from tools.issue252.ego_operators import DeviceLayout, pack_synthetic
        model = structure.SubgraphTokens().double()
        store = TinyStore()
        four_token_fixture(structure, model, store)
        layout = DeviceLayout(pack_synthetic(store, 2), 'cpu')
        features = torch.zeros(4, 6, dtype=torch.double)
        features[:, 0] = torch.tensor([1., 2., 4., 8.])
        means = StaticFeatureMeans(layout, features, chunk=1)
        # The first three complete egos have local nodes [0,1,2].
        expected_in = torch.tensor([0., 2.5, 1.] * 3 + [0.], dtype=torch.double)
        expected_out = torch.tensor([3., 0., 2.] * 3 + [0.], dtype=torch.double)
        torch.testing.assert_close(means.incoming[:, 0], expected_in)
        torch.testing.assert_close(means.outgoing[:, 0], expected_out)
        self.assertEqual(means.in_nonempty.tolist(), [False, True, True] * 3 + [False])
        self.assertEqual(means.out_nonempty.tolist(), [True, False, True] * 3 + [False])
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.input.weight[0, 0] = 2.
            model.input.bias[0] = 3.
            model.layers[0].self_linear.weight[0, 0] = .5
            model.layers[0].self_linear.bias[0] = -1.
            model.layers[0].incoming.weight[0, 0] = 1.
            model.layers[0].outgoing.weight[0, 0] = 2.
            model.layers[1].self_linear.weight[0, 0] = 1.
        actual, valid = model.tokens_for_items(store, [0, 3], 'cpu', chunk=1)
        # Use the hand features rather than TinyStore's normalization.
        layouts, _ = model._cached_store(store)
        static = [StaticFeatureMeans(part, features, chunk=1) for part in layouts]
        from tools.issue252.ego_operators import batch_operators
        nodes, groups, sizes, incoming, outgoing = batch_operators(layouts[1], torch.tensor([0, 3]), torch.double)
        from tools.issue253.sparse_adjoint import normalized_directional_means
        got = model._read_sparse(features[nodes], groups, sizes,
                                 static[1].gather(layouts[1], torch.tensor([0, 3])),
                                 model._first_affine(),
                                 lambda values: normalized_directional_means(values, incoming, outgoing))
        preactivation = torch.tensor([1.5 + 18., 2.5 + 8., 4.5 + 5. + 14.], dtype=torch.double)
        expected = torch.tanh(torch.tanh(preactivation)).mean()
        self.assertAlmostEqual(float(got[0, 0]), float(expected), places=12)
        self.assertAlmostEqual(float(got[1, 0]), float(torch.tanh(torch.tanh(torch.tensor(8.5, dtype=torch.double)))), places=12)
        self.assertTrue(valid.all())
        self.assertEqual(tuple(actual.shape), (2, 4, 32))

    def test_dense_and_original_ragged_values_and_every_parameter_gradient(self):
        for dtype, atol, rtol in ((torch.double, 1e-9, 1e-8), (torch.float, 1e-5, 1e-4)):
            with self.subTest(dtype=dtype):
                torch.manual_seed(253)
                model = structure.SubgraphTokens().to(dtype=dtype)
                store = TinyStore()
                ids = [3, 0, 3, 2, 1]
                four_token_fixture(structure, model, store)
                features = store.node_features()
                normalized = torch.tensor((features - features.mean(0)) / features.std(0), dtype=dtype)
                actual, valid = model.tokens_for_items(store, ids, 'cpu', chunk=1)
                expected, expected_valid = dense_reference(model, store, ids, normalized)
                torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
                self.assertTrue(torch.equal(valid, expected_valid))
                weights = torch.randn_like(actual)
                left = torch.autograd.grad((actual * weights).sum(), tuple(model.parameters()))
                right = torch.autograd.grad((expected * weights).sum(), tuple(model.parameters()))
                for got, want in zip(left, right):
                    torch.testing.assert_close(got, want, atol=atol, rtol=rtol)
                for position, item in enumerate(ids):
                    for scale, hops in enumerate(TOKEN_POPULATIONS):
                        batch = store.collate([item], hops=hops)
                        batch['features'] = normalized[batch['nodes']]
                        expected_token = model(batch)[0] + TYPE_ENCODING[scale].to(actual)
                        torch.testing.assert_close(actual[position, scale], expected_token, atol=atol, rtol=rtol)
                other, _ = model.tokens_for_items(store, ids, 'cpu', chunk=4)
                torch.testing.assert_close(actual, other, atol=atol, rtol=rtol)
                empty, mask = model.tokens_for_items(store, [], 'cpu')
                self.assertEqual(tuple(empty.shape), (0, 4, 32))
                self.assertEqual(tuple(mask.shape), (0, 4))

    def test_live_affine_input_and_every_parameter_gradient(self):
        from tools.issue252.ego_operators import DeviceLayout, batch_operators, pack_synthetic
        torch.manual_seed(254)
        model = structure.SubgraphTokens().double()
        store = TinyStore()
        layout = DeviceLayout(pack_synthetic(store, 2), 'cpu')
        ids = torch.tensor([0, 3])
        nodes, groups, sizes, incoming, outgoing = batch_operators(layout, ids, torch.double)
        features = torch.randn(store.n, 6, dtype=torch.double, requires_grad=True)
        local = features[nodes]
        static = (torch.sparse.mm(incoming, local), torch.sparse.mm(outgoing, local),
                  incoming.crow_indices().diff() > 0, outgoing.crow_indices().diff() > 0)
        from tools.issue253.sparse_adjoint import normalized_directional_means
        actual = model._read_sparse(local, groups, sizes, static, model._first_affine(),
                                    lambda values: normalized_directional_means(values, incoming, outgoing))
        expected, _ = dense_reference(model, store, [0, 3], features)
        expected = expected[:, 1] - TYPE_ENCODING[1].to(expected)
        torch.testing.assert_close(actual, expected, atol=1e-9, rtol=1e-8)
        weights = torch.randn_like(actual)
        variables = (features, *model.parameters())
        left = torch.autograd.grad((actual * weights).sum(), variables, retain_graph=True)
        right = torch.autograd.grad((expected * weights).sum(), variables)
        for got, want in zip(left, right):
            torch.testing.assert_close(got, want, atol=1e-9, rtol=1e-8)

    def test_cache_is_detached_static_only_and_all_weights_are_live(self):
        torch.manual_seed(255)
        model = structure.SubgraphTokens().double()
        store = TinyStore()
        original = copy.deepcopy(model.state_dict())
        four_token_fixture(structure, model, store)
        self.assertEqual(len(model._feature_cache[store]['layouts']), 4)
        model.tokens_for_items(store, [0], 'cpu', chunk=1)
        resident = model._feature_cache[store]['devices']['cpu']
        static = resident['means'][torch.double]
        self.assertEqual(len(static), 4)
        for means in static:
            self.assertEqual(set(vars(means)), {'incoming', 'outgoing', 'in_nonempty', 'out_nonempty'})
            for value in vars(means).values():
                self.assertFalse(value.requires_grad)
                self.assertIsNone(value.grad_fn)
            self.assertEqual(means.incoming.shape[1], 6)
            self.assertEqual(means.outgoing.shape[1], 6)
        baseline, _ = model.tokens_for_items(store, [0, 1, 3], 'cpu')
        for name, parameter in model.named_parameters():
            with self.subTest(parameter=name):
                with torch.no_grad():
                    parameter.add_(.05)
                changed, _ = model.tokens_for_items(store, [0, 1, 3], 'cpu')
                self.assertFalse(torch.allclose(baseline, changed, atol=1e-12, rtol=1e-12))
                with torch.no_grad():
                    parameter.copy_(original[name])
        self.assertEqual(set(model.state_dict()), set(original))
        self.assertEqual(len(tuple(model.parameters())), 10)
        self.assertEqual(set(dict(model.named_buffers())), {'type_encoding'})
        torch.testing.assert_close(model.type_encoding, TYPE_ENCODING.to(model.type_encoding),
                                   atol=0, rtol=0)
        self.assertEqual(len(original), 11)
        self.assertIs(model._feature_cache[store]['devices']['cpu']['means'][torch.double], static)

    def test_frozen_feature_root_and_bounded_full_layout_cache(self):
        from tools.issue253.static_features import StaticFeatureMeans
        from tools.issue252.ego_operators import DeviceLayout, pack_synthetic
        store = TinyStore()
        layout = DeviceLayout(pack_synthetic(store, 2), 'cpu')
        features = torch.tensor(store.node_features(), dtype=torch.double, requires_grad=True)
        import tools.issue253.static_features as module
        sizes = []
        original = module.batch_operators

        def observe(part, ids, dtype, plan=None):
            sizes.append(len(ids))
            return original(part, ids, dtype, plan)

        with patch.object(module, 'batch_operators', side_effect=observe):
            bounded = StaticFeatureMeans(layout, features, chunk=1)
        self.assertEqual(sizes, [1] * store.n)
        full = StaticFeatureMeans(layout, features, chunk=store.n)
        for name in vars(bounded):
            torch.testing.assert_close(getattr(bounded, name), getattr(full, name))
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            root = Path(directory)
            np.savez(root / 'node_features.npz', features=store.node_features())
            (root / 'node_features_report.json').write_text(json.dumps(
                {'standardization': {'mean': [1.] * 6, 'std': [2.] * 6}}))
            store.root = root
            model = structure.SubgraphTokens().double()
            four_token_fixture(structure, model, store)
            actual, valid = model.tokens_for_items(store, [3, 0, 2], 'cpu', chunk=1)
            frozen = (features.detach() - 1.) / 2.
            expected, expected_valid = dense_reference(model, store, [3, 0, 2], frozen)
            torch.testing.assert_close(actual, expected, atol=1e-9, rtol=1e-8)
            self.assertTrue(torch.equal(valid, expected_valid))
            (root / 'node_features.npz').unlink()
            again, _ = model.tokens_for_items(store, [3, 0, 2], 'cpu', chunk=3)
            torch.testing.assert_close(again, expected, atol=1e-9, rtol=1e-8)


if __name__ == '__main__':
    unittest.main()
