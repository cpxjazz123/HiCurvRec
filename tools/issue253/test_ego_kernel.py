"""C10 numerical oracles for the candidate's selected full-ego implementation.

CPU execution checks the sparse path. The sanctioned four-rank diagnostic sets
KernelEquivalenceTests.production_device to its local CUDA device before loading
this suite, so GPU acceptance exercises the ACTUAL kernel, never a CPU fallback.
No timing threshold is inferred from these unit tests.
"""
import importlib.util
from pathlib import Path
import unittest

import numpy as np
import torch

from tools.issue253.ego_reference import (
    FullDirectedStore, batch_reference, dense_tokens, occurrence_operators, packed_layout,
    second_layer_readout,
)

ROOT = next(path for path in Path(__file__).resolve().parents
            if (path / 'stage2_RQ-VAE').is_dir())
CANDIDATE = ROOT / 'stage2_RQ-VAE/TIGER_three_expert_attention_rq'


def load_structure():
    spec = importlib.util.spec_from_file_location(
        'issue253_structure_kernel_oracle', CANDIDATE / 'model/structure.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tolerances(dtype):
    return dict(atol=1e-9, rtol=1e-8) if dtype == torch.float64 else dict(atol=1e-5, rtol=1e-4)


def irregular_blocks():
    # Both directions include a 1049-edge row (the observed v9 maximum), and
    # every proposed vector/chunk boundary has neighbours on both sides.
    degrees = (1, 3, 7, 8, 9, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 257, 513, 1049)
    edges = set()
    for row, degree in enumerate(degrees):
        edges.update((row, target) for target in range(20, 20 + degree))
    edges.update((source, 1070) for source in range(1049))
    edges.add((1069, 1069))
    return [([], []), ([9000], []),
            ([100, 101, 102, 103], [(0, 1), (0, 2), (2, 1)]),
            (list(range(1072)), sorted(edges)),
            ([200, 201, 202], [(0, 1), (0, 1), (2, 1), (1, 2)])]


class KernelEquivalenceTests(unittest.TestCase):
    production_device = 'cpu'

    @classmethod
    def setUpClass(cls):
        cls.structure = load_structure()
        cls.device = torch.device(cls.production_device)
        if cls.device.type == 'cuda':
            # The diagnostic launcher owns GPU allocation; a bare GPU unit run
            # must not silently launch work outside the sanctioned entry.
            from tools.research.stage2.launcher import is_child_rank
            if not is_child_rank():
                raise RuntimeError('CUDA C10 checks require the sanctioned launcher')
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False

    def means(self, x, layout, ids, reference):
        geometry = self.structure.ego_kernel.batch_geometry(layout, np.asarray(ids), self.device)
        self.assertEqual(geometry['nodes'], len(x))
        self.assertTrue(torch.equal(geometry['layout_rows'], reference.layout_rows))
        self.assertTrue(torch.equal(geometry['row_ego'], reference.groups))
        operators = occurrence_operators(layout, ids, x.dtype, self.device)
        expected_occurrences = sum(len(layout.reference_blocks[item][1]) for item in ids)
        for got, want in zip(operators, (reference.incoming, reference.outgoing)):
            self.assertEqual(got._nnz(), expected_occurrences,
                             'CSR adapter must retain every directed occurrence, including duplicates')
            torch.testing.assert_close(got.to_dense(), want, **tolerances(x.dtype))
        actual = self.structure.directional_means(x, layout, geometry, operators)
        return actual, geometry

    def test_float64_adjoint_retains_degree3_and7_reciprocals(self):
        edges = [(0, target) for target in (1, 2, 3)]
        edges += [(1, target) for target in (0, 2, 3, 4, 5, 6, 7)]
        for reverse in (False, True):
            with self.subTest(reverse=reverse):
                directed = [(target, source) for source, target in edges] if reverse else edges
                layout = packed_layout([(list(range(8)), directed)], self.device)
                ref = batch_reference(layout, [0], torch.float64, self.device)
                x = torch.ones(8, 32, dtype=torch.float64, device=self.device, requires_grad=True)
                means, _ = self.means(x, layout, [0], ref)
                weights = torch.zeros_like(x)
                weights[0, 0] = weights[1, 0] = 1
                selected = means[0 if reverse else 1]
                gradient, = torch.autograd.grad((selected * weights).sum(), x)
                # Hard-coded independent rational answer; float32 reciprocal
                # promotion to double is not accurate to the float64 contract.
                torch.testing.assert_close(gradient[2, 0], x.new_tensor(10 / 21), **tolerances(x.dtype))
                operator = ref.incoming if reverse else ref.outgoing
                torch.testing.assert_close(gradient, operator.T @ weights, **tolerances(x.dtype))

    def test_known_directed_values_and_reciprocal_degree_transpose(self):
        layout = packed_layout([([0, 1, 2, 3], [(0, 1), (0, 2), (2, 1)])], self.device)
        ref = batch_reference(layout, [0], torch.float64, self.device)
        x = torch.zeros(4, 32, dtype=torch.float64, device=self.device)
        x[:, 0] = torch.tensor([1., 2., 4., 8.], device=self.device)
        x.requires_grad_()
        (incoming, outgoing), _ = self.means(x, layout, [0], ref)
        torch.testing.assert_close(incoming[:, 0], x.new_tensor([0., 2.5, 1., 0.]), **tolerances(x.dtype))
        torch.testing.assert_close(outgoing[:, 0], x.new_tensor([3., 0., 2., 0.]), **tolerances(x.dtype))
        weights_in, weights_out = torch.zeros_like(x), torch.zeros_like(x)
        weights_in[:, 0] = x.new_tensor([2., 3., 5., 7.])
        weights_out[:, 0] = x.new_tensor([11., 13., 17., 19.])
        gradient, = torch.autograd.grad((incoming, outgoing), x, (weights_in, weights_out))
        torch.testing.assert_close(gradient[:, 0], x.new_tensor([6.5, 22.5, 7., 0.]), **tolerances(x.dtype))
        self.assertEqual(torch.count_nonzero(gradient[:, 1:]).item(), 0)

    def test_empty_short_long_repeated_rows_values_and_exact_input_adjoint(self):
        ids = [3, 0, 2, 1, 4, 2]  # Restore order and repeated egos, not sorted IDs.
        layout = packed_layout(irregular_blocks(), self.device)
        for dtype in (torch.float64, torch.float32):
            with self.subTest(dtype=dtype):
                torch.manual_seed(25310)
                ref = batch_reference(layout, ids, dtype, self.device)
                x = torch.randn(len(ref.nodes), 32, dtype=dtype, device=self.device, requires_grad=True)
                actual, geometry = self.means(x, layout, ids, ref)
                expected = (ref.incoming @ x, ref.outgoing @ x)
                weights = tuple(torch.randn_like(value) for value in actual)
                for got, want in zip(actual, expected):
                    self.assertEqual(got.dtype, dtype)
                    torch.testing.assert_close(got, want, **tolerances(dtype))
                gradient, = torch.autograd.grad(actual, x, weights)
                exact_transpose = ref.incoming.T @ weights[0] + ref.outgoing.T @ weights[1]
                torch.testing.assert_close(gradient, exact_transpose, **tolerances(dtype))
                # The inner-product identity detects reversing support without
                # carrying the ORIGINAL row's reciprocal degree into backward.
                left = sum((value.detach() * weight).sum() for value, weight in zip(actual, weights))
                right = (x.detach() * exact_transpose).sum()
                torch.testing.assert_close(left, right, **tolerances(dtype))
                if self.device.type == 'cuda':
                    for direction, want in zip(('in', 'out'), expected):
                        buffer = torch.empty_like(x)
                        result = self.structure.ego_kernel.row_means(
                            getattr(layout, direction + '_ptr'), getattr(layout, direction + '_col'),
                            x.detach(), geometry, out=buffer)
                        self.assertIs(result, buffer)
                        torch.testing.assert_close(result, want.detach(), **tolerances(dtype))

    def test_zero_nodes_and_each_unused_direction(self):
        layout = packed_layout([([], []), ([0], []),
                                ([0, 1, 2, 3], [(0, 1), (0, 2), (2, 1)])], self.device)
        for ids in ([], [0], [1], [2, 0, 1]):
            for dtype in (torch.float64, torch.float32):
                for direction in (0, 1):
                    with self.subTest(ids=ids, dtype=dtype, direction=direction):
                        ref = batch_reference(layout, ids, dtype, self.device)
                        x = torch.randn(len(ref.nodes), 32, dtype=dtype, device=self.device, requires_grad=True)
                        means, _ = self.means(x, layout, ids, ref)
                        weight = torch.randn_like(x)
                        gradient, = torch.autograd.grad((means[direction] * weight).sum(), x)
                        operator = (ref.incoming, ref.outgoing)[direction]
                        torch.testing.assert_close(gradient, operator.T @ weight, **tolerances(dtype))

    def test_noncontiguous_inputs_and_finite_difference_input_gradient(self):
        layout = packed_layout([([0, 1, 2, 3], [(0, 1), (0, 2), (2, 1)])], self.device)
        for dtype in (torch.float64, torch.float32):
            with self.subTest(dtype=dtype):
                torch.manual_seed(25314)
                ref = batch_reference(layout, [0], dtype, self.device)
                x = torch.randn(32, 4, dtype=dtype, device=self.device).T.requires_grad_()
                self.assertFalse(x.is_contiguous())
                means, _ = self.means(x, layout, [0], ref)
                for got, want in zip(means, (ref.incoming @ x, ref.outgoing @ x)):
                    torch.testing.assert_close(got, want, **tolerances(dtype))
                weights = torch.randn_like(x)
                def objective():
                    incoming, outgoing = self.means(x, layout, [0], ref)[0]
                    return (torch.tanh(x + .7 * incoming - .3 * outgoing) * weights).sum()
                gradient, = torch.autograd.grad(objective(), x)
                expected = (torch.tanh(x + .7 * (ref.incoming @ x)
                                       - .3 * (ref.outgoing @ x)) * weights).sum()
                exact, = torch.autograd.grad(expected, x)
                torch.testing.assert_close(gradient, exact, **tolerances(dtype))
                if dtype == torch.float64:
                    epsilon = 1e-5
                    for index in ((0, 0), (1, 7), (2, 31), (3, 5)):
                        with torch.no_grad():
                            original = x[index].item()
                            x[index] = original + epsilon
                            positive = objective().item()
                            x[index] = original - epsilon
                            negative = objective().item()
                            x[index] = original
                        numeric = x.new_tensor((positive - negative) / (2 * epsilon))
                        torch.testing.assert_close(gradient[index], numeric, atol=1e-6, rtol=1e-5)

    def test_second_affine_readout_input_and_all_parameter_gradients(self):
        layout = packed_layout(irregular_blocks(), self.device)
        ids = [2, 0, 3, 1, 2, 4]
        for dtype in (torch.float64, torch.float32):
            with self.subTest(dtype=dtype):
                torch.manual_seed(25311)
                model = self.structure.SubgraphTokens().to(device=self.device, dtype=dtype)
                ref = batch_reference(layout, ids, dtype, self.device)
                features = torch.randn(len(ref.nodes), 6, dtype=dtype, device=self.device, requires_grad=True) * .2
                static = (ref.incoming @ features, ref.outgoing @ features,
                          ref.incoming.sum(1) > 0, ref.outgoing.sum(1) > 0)
                def operator(value):
                    return self.means(value, layout, ids, ref)[0]
                actual = model._read_sparse(features, ref.groups, ref.sizes,
                                            static, model._first_affine(), operator)
                projected = features @ model.input.weight.T + model.input.bias
                first = model.layers[0]
                hidden = torch.tanh(projected @ first.self_linear.weight.T + first.self_linear.bias
                                    + (ref.incoming @ projected) @ first.incoming.weight.T
                                    + (ref.outgoing @ projected) @ first.outgoing.weight.T)
                expected = second_layer_readout(hidden, ref.incoming, ref.outgoing,
                                                ref.groups, [len(layout.reference_blocks[item][0]) for item in ids],
                                                model.layers[1])
                torch.testing.assert_close(actual, expected, **tolerances(dtype))
                weights = torch.randn_like(actual)
                variables = (features, *model.parameters())
                actual_gradients = torch.autograd.grad((actual * weights).sum(), variables, retain_graph=True)
                expected_gradients = torch.autograd.grad((expected * weights).sum(), variables)
                for index, (got, want) in enumerate(zip(actual_gradients, expected_gradients)):
                    with self.subTest(variable=index):
                        self.assertTrue(torch.isfinite(got).all().item())
                        self.assertGreater(got.abs().sum().item(), 0)
                        torch.testing.assert_close(got, want, **tolerances(dtype))

    def test_full_induced_one_two_hop_nodes_edges_masks_and_live_parameters(self):
        store = FullDirectedStore()
        ids = [10, 0, 8, 0, 4, 3]
        for dtype in (torch.float64, torch.float32):
            with self.subTest(dtype=dtype):
                torch.manual_seed(25312)
                model = self.structure.SubgraphTokens().to(device=self.device, dtype=dtype)
                raw = store.node_features()
                features = torch.tensor((raw - raw.mean(0)) / raw.std(0), dtype=dtype, device=self.device)
                actual, mask = model.tokens_for_items(store, ids, self.device, chunk=2)
                expected, expected_mask = dense_tokens(model, store, ids, features)
                self.assertTrue(torch.equal(mask, expected_mask))
                self.assertTrue(mask.all().item())  # Isolate is root-only, not invalid.
                torch.testing.assert_close(actual, expected, **tolerances(dtype))
                torch.testing.assert_close(actual[1], actual[3], atol=0, rtol=0)
                for hops, packed in enumerate(model._feature_cache[store]['layouts'], 1):
                    for item in range(store.n):
                        nodes, edges = store.block(item, hops)
                        start, stop = packed['offsets'][item:item + 2]
                        self.assertEqual(packed['nodes'][start:stop].tolist(), nodes)
                        for direction in ('in', 'out'):
                            ptr, col = packed[direction + '_ptr'], packed[direction + '_col']
                            observed = []
                            for local_row in range(len(nodes)):
                                for local_col in col[ptr[start + local_row]:ptr[start + local_row + 1]]:
                                    observed.append((int(local_col), local_row) if direction == 'in'
                                                    else (local_row, int(local_col)))
                            self.assertEqual(sorted(observed), sorted(edges))
                weights = torch.randn_like(actual)
                left = torch.autograd.grad((actual * weights).sum(), tuple(model.parameters()))
                right = torch.autograd.grad((expected * weights).sum(), tuple(model.parameters()))
                for got, want in zip(left, right):
                    torch.testing.assert_close(got, want, **tolerances(dtype))
                original = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
                for name, parameter in model.named_parameters():
                    with self.subTest(parameter=name), torch.no_grad():
                        parameter.add_(.03125)
                        changed, _ = model.tokens_for_items(store, ids, self.device, chunk=3)
                        independent, _ = dense_tokens(model, store, ids, features)
                        torch.testing.assert_close(changed, independent, **tolerances(dtype))
                        self.assertGreater((changed - actual).abs().max().item(), 1e-6)
                        parameter.copy_(original[name])
                empty, empty_mask = model.tokens_for_items(store, [], self.device)
                self.assertEqual(tuple(empty.shape), (0, 2, 32))
                self.assertEqual(tuple(empty_mask.shape), (0, 2))

    def test_finite_difference_live_second_affine_and_first_projection(self):
        torch.manual_seed(25313)
        model = self.structure.SubgraphTokens().to(device=self.device, dtype=torch.float64)
        store, ids = FullDirectedStore(), [0, 3, 10]
        weights = torch.randn(3, 2, 32, dtype=torch.float64, device=self.device)
        def objective():
            value, _ = model.tokens_for_items(store, ids, self.device, chunk=2)
            return (value * weights).sum()
        parameters = tuple(model.parameters())
        gradients = torch.autograd.grad(objective(), parameters)
        epsilon = 1e-5
        for parameter, gradient in zip(parameters, gradients):
            index = (0,) * parameter.ndim
            with torch.no_grad():
                original = parameter[index].item()
                parameter[index] = original + epsilon
                positive = objective().item()
                parameter[index] = original - epsilon
                negative = objective().item()
                parameter[index] = original
            numeric = gradient.new_tensor((positive - negative) / (2 * epsilon))
            torch.testing.assert_close(gradient[index], numeric, atol=1e-6, rtol=1e-5)


if __name__ == '__main__':
    unittest.main()
