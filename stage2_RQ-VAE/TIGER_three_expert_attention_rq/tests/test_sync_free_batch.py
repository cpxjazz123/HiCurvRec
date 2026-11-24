"""The ego batch assembly must not synchronize the device with the host.

Every gather must be sized by host-known counts: a data-dependent shape forces a
device-to-host copy, and one copy per batch is enough to serialize the step.
"""
import importlib.util
import unittest
import warnings
from pathlib import Path

import torch

TREE = Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


structure = _load('structure252_sync', TREE / 'model/structure.py')
fixtures = _load('fixtures252_sync', Path(__file__).resolve().parent / 'test_static_means.py')


class SyncFreeAssemblyTests(unittest.TestCase):
    def _syncs(self, call):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            torch.cuda.set_sync_debug_mode('warn')
            try:
                call()
            finally:
                torch.cuda.set_sync_debug_mode('default')
        return sum(1 for item in caught if 'synchronizing CUDA operation' in str(item.message))

    def test_production_assembly_does_not_read_device_values_for_shapes(self):
        if not torch.cuda.is_available():
            self.fail('CUDA is required: this contract is about device synchronizations')
        device = str(torch.device('cuda', torch.cuda.current_device()))
        model = structure.SubgraphTokens().double().to(device)
        store = fixtures.TinyStore()
        # One cold call builds the static caches; every later call must only
        # copy the host-known ego ids, once per scale, and never ask the device
        # for a size. The observed count before this contract was 91 syncs per
        # warm call (results/research/issue252/profiling/ASSEMBLY_SYNCS.log).
        self.assertLessEqual(self._syncs(lambda: model.tokens_for_items(store, [0, 1, 2, 3], device, chunk=2)), 64)
        self.assertLessEqual(self._syncs(lambda: model.tokens_for_items(store, [0, 1, 2, 3], device, chunk=2)), 2)
        self.assertLessEqual(self._syncs(lambda: model.tokens_for_items(store, [0, 0, 1], device, chunk=2)), 2)

    def test_assembly_matches_the_unsynchronised_reference_values(self):
        if not torch.cuda.is_available():
            self.fail('CUDA is required: this contract is about device synchronizations')
        torch.manual_seed(256)
        model = structure.SubgraphTokens().double()
        store = fixtures.TinyStore()
        ids = [3, 0, 3, 1, 2]
        expected, expected_valid = model.tokens_for_items(store, ids, 'cpu', chunk=1)
        device = str(torch.device('cuda', torch.cuda.current_device()))
        moved = structure.SubgraphTokens().double().to(device)
        moved.load_state_dict(model.state_dict())
        actual, valid = moved.tokens_for_items(store, ids, device, chunk=1)
        torch.testing.assert_close(actual.cpu(), expected, atol=1e-9, rtol=1e-8)
        self.assertTrue(torch.equal(valid.cpu(), expected_valid))


if __name__ == '__main__':
    unittest.main()
