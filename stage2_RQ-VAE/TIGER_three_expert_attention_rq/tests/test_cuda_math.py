"""Compiler setup must never write the historical issue252 cache tree."""
from pathlib import Path
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'stage2_RQ-VAE').is_dir())
sys.path.insert(0, str(ROOT))
from tools.issue253 import cuda_math


class CompilerIsolationTests(unittest.TestCase):
    def test_default_cache_identity_is_issue253(self):
        self.assertEqual(cuda_math.CACHE, ROOT / 'results/research/issue253/compiler_cache')

    def test_configuration_keeps_every_cache_and_temporary_root_in_owned_tree(self):
        import torch._inductor.config as inductor
        import torch._dynamo.config as dynamo
        import torch._functorch.config as autograd
        with tempfile.TemporaryDirectory(dir=ROOT / 'results/research/issue253') as directory:
            cache = Path(directory) / 'compiler_cache'
            with patch.object(cuda_math, 'CACHE', cache), patch.dict(os.environ), \
                 patch.object(tempfile, 'tempdir', tempfile.tempdir), \
                 patch.object(inductor, 'compile_threads', inductor.compile_threads), \
                 patch.object(dynamo, 'recompile_limit', dynamo.recompile_limit), \
                 patch.object(autograd, 'donated_buffer', autograd.donated_buffer):
                cuda_math._configure()
                for key, child in (('TORCHINDUCTOR_CACHE_DIR', 'inductor'),
                                   ('TRITON_CACHE_DIR', 'triton'),
                                   ('TORCH_EXTENSIONS_DIR', 'extensions'),
                                   ('TMPDIR', 'temporary'), ('TMP', 'temporary'), ('TEMP', 'temporary')):
                    self.assertEqual(Path(os.environ[key]), cache / child)
                    self.assertTrue((cache / child).is_dir())
                self.assertEqual(Path(tempfile.tempdir), cache / 'temporary')
                self.assertEqual(inductor.compile_threads, 1)
                self.assertEqual(dynamo.recompile_limit, 64)
                self.assertFalse(autograd.donated_buffer)

    def test_compiled_api_preserves_functions_and_frozen_options(self):
        functions = tuple(lambda x, offset=i: x + offset for i in range(3))
        with patch.object(cuda_math, '_configure') as configure, \
             patch.object(cuda_math.torch, 'compile', side_effect=lambda function, **kwargs: function) as compile:
            self.assertEqual(cuda_math.compile_geometry(*functions), functions)
            self.assertIs(cuda_math.compile_probabilities(functions[0]), functions[0])
            self.assertIs(cuda_math.compile_dense(functions[1]), functions[1])
        self.assertEqual(configure.call_count, 3)
        self.assertEqual(compile.call_count, 5)
        for call in compile.call_args_list:
            self.assertEqual(call.kwargs, {'fullgraph': True, 'dynamic': True, 'mode': 'default'})


if __name__ == '__main__':
    unittest.main()
