"""The layered SID record must belong to the arm that owns the SID."""
import unittest
from pathlib import Path

from tools.issue252.layer_metrics import ARM, resolve_arm


ROOT = Path(__file__).resolve().parents[2]


class LayerRecordArmTests(unittest.TestCase):
    def test_explicit_arm_is_used_verbatim(self):
        self.assertEqual(resolve_arm(ROOT, ROOT/'x.json', 'TIGER_other_rq'), 'TIGER_other_rq')
        for value in ('', '   ', 7, ['a']):
            with self.subTest(value=value), self.assertRaises(ValueError):
                resolve_arm(ROOT, ROOT/'x.json', value)

    def test_pre_workspace_result_path_falls_back_to_the_historical_arm(self):
        legacy = ROOT/'results/stage2_RQ-VAE'/ARM/'item_sids.json'
        self.assertEqual(resolve_arm(ROOT, legacy), ARM)

    def test_malformed_immutable_result_path_fails_closed(self):
        broken = ROOT/'results/stage2_RQ-VAE/_runs'/'run'/'result/item_sids.json'
        with self.assertRaises(ValueError):
            resolve_arm(ROOT, broken)


if __name__ == '__main__':
    unittest.main()
