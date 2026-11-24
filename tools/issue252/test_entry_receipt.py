"""Issue252 entry admission cannot spawn the legacy unlaunched single-card smoke."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from tools.research.stage2 import checks

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / 'tools/research/stage2/contract.py').is_file())


class EntryReceiptTests(unittest.TestCase):
    def test_missing_four_rank_receipt_refuses_without_single_card_process(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'results/research/issue252') as temporary:
            repo = Path(temporary)
            tree = repo / 'stage2_RQ-VAE/TIGER_gnn_attention_sectional_rq'
            tree.mkdir(parents=True)
            (tree / 'train_rqvae.py').write_text("raise AssertionError('unadmitted GPU entry')\n")
            with patch.object(checks.contract, 'resolve_root', return_value=repo), patch.object(
                    checks.subprocess, 'run', side_effect=AssertionError('single-card GPU bypass')):
                result = checks.check_entry(tree)
            self.assertEqual(result['status'], 'FAIL')
            self.assertIn('four-rank', result['detail'])


if __name__ == '__main__':
    unittest.main()
