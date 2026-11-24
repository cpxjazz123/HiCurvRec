"""The real candidate must import from the launcher's result cwd without PYTHONPATH."""
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'stage2_RQ-VAE').is_dir())
TREE = ROOT/'stage2_RQ-VAE/TIGER_three_expert_attention_rq'

class DirectImportTests(unittest.TestCase):
    def test_actual_model_imports_from_result_directory_in_isolated_python(self):
        program = ('import sys; sys.path.insert(0, '+repr(str(TREE))+'); '
                   'import train_rqvae as t; '
                   'assert t.ARM_NAME == "TIGER_three_expert_attention_rq"; '
                   'assert t.RQVAE.__module__ == "model.mechanism"')
        result = subprocess.run([sys.executable, '-I', '-c', program],
            cwd=ROOT/'results/research/issue253', capture_output=True, text=True, timeout=90)
        self.assertEqual(result.returncode, 0,
            'actual candidate imports must not depend on caller cwd/PYTHONPATH:\n'+result.stderr)

if __name__ == '__main__': unittest.main()
