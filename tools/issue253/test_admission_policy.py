"""A new full-ego arm must not silently take the legacy one-GPU admission route."""
from pathlib import Path
import unittest
from tools.issue252 import training_admission as admission

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'stage2_RQ-VAE').is_dir())
ARM = 'TIGER_three_expert_attention_rq'

class AdmissionPolicyTests(unittest.TestCase):
    def test_new_three_expert_arm_requires_source_bound_four_rank_evidence(self):
        self.assertTrue(admission.required(ROOT/'stage2_RQ-VAE'/ARM),
                        'new full-ego arm must require four-rank source-bound entry evidence')
    def test_frozen_baseline_does_not_inherit_experiment_gate(self):
        self.assertFalse(admission.required(ROOT/'stage2_RQ-VAE/baseline/TIGER_RQ-VAE'))

    def test_three_queries_are_required_and_removed_queries_are_rejected(self):
        names = ['encoder','decoder','GNN.input','GNN.layer','sharedKey','sharedValue','sharedScore']
        names += ['Query%d' % i for i in range(3)] + ['codebook%d' % i for i in range(3)]
        rows = {name:{'norm':1.,'finite':True,'absent':False,'tensors':1} for name in names}
        admission._stats(rows, positive=True, experts=3)
        missing = dict(rows); del missing['Query2']
        extra = dict(rows); extra['Query3'] = rows['Query0']
        for invalid in (missing, extra):
            with self.subTest(queries=[k for k in invalid if k.startswith('Query')]):
                with self.assertRaisesRegex(ValueError, 'query evidence'):
                    admission._stats(invalid, positive=True, experts=3)

if __name__ == '__main__': unittest.main()
