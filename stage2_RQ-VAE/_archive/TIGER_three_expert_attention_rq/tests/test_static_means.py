"""Independent value/gradient and resource contracts for exact static GNN means."""
import copy
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


class TinyStore:
    n = 4

    def node_features(self):
        return np.array([[1., 3., 2., 7., 0., 2.], [4., 1., 5., 2., 1., 4.],
                         [2., 8., 1., 4., 3., 1.], [9., 2., 3., 1., 2., 8.]])

    def collate(self, items, hops=1, standardized=False):
        nodes, groups, edges = [], [], []
        for group, item in enumerate(items):
            # Root 3 is an isolate; root 2 has an empty first-scale ego.
            block = ([] if hops == 1 and item == 2 else
                     [3] if item == 3 else [item, (item + 1) % 3] if hops == 1 else [0, 1, 2])
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


def dense_reference(model, store, ids, features):
    """Original two nonlinear layers, with independently formed dense operators."""
    scales, valid = [], []
    for hops in (1, 2):
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
            rows.append(x.mean(0))
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
        self.assertEqual(forward_widths, [32] * 8,
                         'Only the second layer may perform two 32-wide sparse means per ego chunk/scale')
        self.assertEqual(forward_activations, 8)
        self.assertEqual(len(activations), forward_activations,
                         'Backward must use retained activations, not checkpoint replay')

    def test_tiny_directed_affine_hand_answer_and_empty_directions(self):
        from tools.issue253.static_features import StaticFeatureMeans
        from tools.issue252.ego_operators import DeviceLayout, pack_synthetic
        model = structure.SubgraphTokens().double()
        store = TinyStore()
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
        self.assertEqual(tuple(actual.shape), (2, 2, 32))

    def test_dense_and_original_ragged_values_and_every_parameter_gradient(self):
        for dtype, atol, rtol in ((torch.double, 1e-9, 1e-8), (torch.float, 1e-5, 1e-4)):
            with self.subTest(dtype=dtype):
                torch.manual_seed(253)
                model = structure.SubgraphTokens().to(dtype=dtype)
                store = TinyStore()
                ids = [3, 0, 3, 2, 1]
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
                    for scale, hops in enumerate((1, 2)):
                        batch = store.collate([item], hops=hops)
                        batch['features'] = normalized[batch['nodes']]
                        torch.testing.assert_close(actual[position, scale], model(batch)[0], atol=atol, rtol=rtol)
                other, _ = model.tokens_for_items(store, ids, 'cpu', chunk=4)
                torch.testing.assert_close(actual, other, atol=atol, rtol=rtol)
                empty, mask = model.tokens_for_items(store, [], 'cpu')
                self.assertEqual(tuple(empty.shape), (0, 2, 32))
                self.assertEqual(tuple(mask.shape), (0, 2))

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
        expected = expected[:, 1]
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
        model.tokens_for_items(store, [0], 'cpu', chunk=1)
        resident = model._feature_cache[store]['devices']['cpu']
        static = resident['means'][torch.double]
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
        self.assertEqual(len(original), 10)
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
