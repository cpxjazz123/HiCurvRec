"""Directed, radius-preserving structure branch checks for issue #250."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch


MODEL = Path(__file__).resolve().parent.parent / 'model' / 'structure.py'
spec = importlib.util.spec_from_file_location('issue250_structure', MODEL)
structure = importlib.util.module_from_spec(spec)
spec.loader.exec_module(structure)
DirectedGate = structure.DirectedGate


import numpy as np
from scipy.sparse import csr_matrix
import torch

from tools.graph_infra.ego_gnn import EgoStore

def hand_store():
    # Directed chain with a fork, a genuine distance-three node, and an isolate.
    store = EgoStore.__new__(EgoStore)
    store.n = 6
    store.ids = np.arange(6)
    store.forward = csr_matrix((np.ones(4), ([0, 1, 1, 3], [1, 2, 3, 4])), shape=(6, 6))
    store.reverse = store.forward.T.tocsr()
    store.undirected = store.forward.maximum(store.reverse).tocsr()
    store._in_degree = np.diff(store.reverse.indptr)
    store._out_degree = np.diff(store.forward.indptr)
    store._reciprocal = np.zeros(6)
    return store


class StructureTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(250)
        self.gate = DirectedGate().double()
        self.store = hand_store()

    def test_hand_directed_means_and_true_pooling_radii(self):
        batch = self.store.collate([0], hops=2, standardized=False)
        np.testing.assert_array_equal(batch['nodes'], [0, 1, 2, 3])
        batch['features'] = np.zeros((4, 6))
        batch['features'][:, 0] = [0, 1, 2, 3]
        with torch.no_grad():
            for parameter in self.gate.parameters():
                parameter.zero_()
            self.gate.input.weight[0, 0] = 1
            self.gate.layers[0].incoming.weight[0, 0] = 1
            self.gate.layers[0].outgoing.weight[0, 0] = 2
            self.gate.layers[1].self_linear.weight[0, 0] = 1
        descriptor, weights = self.gate(batch)
        a, b, c = np.tanh([1., 2., 3.])
        # Node 0 sees outgoing node 1; node 1 sees incoming 0 and outgoing 2,3.
        first = np.tanh([2 * a, b + c, a, a])
        expected = np.zeros((1, 96))
        expected[0, 32] = first[:2].mean()  # closed radius one, not the whole ego
        expected[0, 64] = np.tanh(first).mean()  # closed radius two
        np.testing.assert_allclose(descriptor.detach(), expected, atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(weights.detach(), np.full((1, 5), .2), atol=1e-12)
        reversed_batch = dict(batch, edges=batch['edges'][:, ::-1].copy())
        reversed_descriptor, _ = self.gate(reversed_batch)
        self.assertGreater(float((descriptor - reversed_descriptor).abs().max()), .01)

    def test_duplicate_batch_chunk_and_tensor_inputs_are_invariant(self):
        items = [3, 0, 5, 0, 2]
        descriptor, weights = self.gate.gates_for_items(self.store, items, 'cpu', chunk=2)
        other_descriptor, other_weights = self.gate.gates_for_items(self.store, items, 'cpu', chunk=1)
        torch.testing.assert_close(descriptor, other_descriptor, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(weights, other_weights, atol=1e-12, rtol=1e-12)
        self.assertTrue(torch.equal(descriptor[1], descriptor[3]))
        self.assertTrue(torch.equal(weights[1], weights[3]))
        plain = self.store.collate(items, hops=2)
        tensor_batch = {key: torch.as_tensor(value) if isinstance(value, np.ndarray) else value
                        for key, value in plain.items()}
        direct_descriptor, direct_weights = self.gate(tensor_batch)
        torch.testing.assert_close(descriptor, direct_descriptor, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(weights, direct_weights, atol=1e-12, rtol=1e-12)
        alone = self.gate(self.store.collate([0], hops=2))
        torch.testing.assert_close(descriptor[1], alone[0][0], atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(weights[1], alone[1][0], atol=1e-12, rtol=1e-12)

    def test_fixed_features_are_not_rebuilt_for_each_ego_or_chunk(self):
        with patch.object(self.store, 'node_features', wraps=self.store.node_features) as features:
            self.gate.gates_for_items(self.store, [0, 1, 2, 3], 'cpu', chunk=1)
            self.gate.gates_for_items(self.store, [4, 5, 0], 'cpu', chunk=2)
        self.assertEqual(features.call_count, 1)
        self.assertNotIn('node_features', self.store.__dict__)

    def test_isolate_empty_and_behavior_off(self):
        descriptor, weights = self.gate.gates_for_items(self.store, [5], 'cpu')
        self.assertEqual(tuple(descriptor.shape), (1, 96))
        self.assertTrue(torch.isfinite(descriptor).all())
        self.assertTrue(torch.isfinite(weights).all())
        torch.testing.assert_close(weights.sum(-1), torch.ones(1, dtype=torch.float64))
        _, uniform = self.gate(self.store.collate([0, 5], hops=2), behavior_on=False)
        torch.testing.assert_close(uniform, torch.full((2, 5), .2, dtype=torch.float64))
        empty_descriptor, empty_weights = self.gate.gates_for_items(self.store, [], 'cpu')
        self.assertEqual(tuple(empty_descriptor.shape), (0, 96))
        self.assertEqual(tuple(empty_weights.shape), (0, 5))

    def test_weight_dependent_loss_reaches_both_message_layers_and_gate(self):
        _, weights = self.gate.gates_for_items(self.store, [0, 1, 3, 5], 'cpu', chunk=2)
        loss = (weights * torch.tensor([1., -2., 3., -.5, 4.], dtype=weights.dtype)).sum()
        loss.backward()
        for name, module in [('input', self.gate.input), ('layer0', self.gate.layers[0]),
                             ('layer1', self.gate.layers[1]), ('gate', self.gate.gate)]:
            norm = sum(float(parameter.grad.abs().sum()) for parameter in module.parameters()
                       if parameter.grad is not None)
            self.assertTrue(np.isfinite(norm), name)
            self.assertGreater(norm, 0., name)
        self.assertFalse(any(isinstance(module, torch.nn.Embedding) for module in self.gate.modules()))

    def test_one_hop_batch_is_rejected_instead_of_silently_truncated(self):
        batch = self.store.collate([0], hops=2)
        batch['hops'] = 1
        with self.assertRaisesRegex(ValueError, 'two-hop'):
            self.gate(batch)


if __name__ == '__main__':
    unittest.main()
