"""Checkpoint launch binding and rank-local assignment history regressions."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch

TREE=Path(__file__).resolve().parents[1]
ROOT=next(p for p in TREE.parents if (p/'assets').is_dir())
sys.path.insert(0,str(TREE))

#: The candidate tree under test; the temporary repository copy below must
#: carry this identity, not the retired single-level arm.
ARM='TIGER_prefix_curvature_gate_rq'


class ExecutionTests(unittest.TestCase):
    def test_actual_launcher_changes_invalidate_source_evidence(self):
        import benchmark_entry
        with tempfile.TemporaryDirectory(dir=ROOT/'results/research/issue255') as directory:
            repo=Path(directory);tree=repo/('stage2_RQ-VAE/'+ARM)
            (tree/'config').mkdir(parents=True)
            (tree/'benchmark_entry.py').write_text('VALUE=1\n')
            (tree/'train_rqvae.py').write_text('import tools.graph_infra.ego_gnn\n')
            for name in ('assets/manifest.json','tools/issue250/launch_validation.py',
                         'tools/issue255/launch_validation.py','tools/issue253/performance.py',
                         'tools/research/stage2/launcher.py','tools/graph_infra/ego_gnn.py',
                         'tools/graph_infra/assets.py','results/research/issue255/CONTRACT.md'):
                path=repo/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('VALUE=1\n')
            with patch.object(benchmark_entry,'ROOT',repo),patch.object(
                    benchmark_entry,'__file__',str(tree/'benchmark_entry.py')):
                initial=benchmark_entry.source_binding('validation')
                (repo/'tools/issue255/launch_validation.py').write_text('VALUE=2\n')
                self.assertNotEqual(initial,benchmark_entry.source_binding('validation'),
                    'changing the executed launcher must invalidate checkpoint evidence')
                self.assertIn('tools/research/stage2/launcher.py',initial['files'])
                self.assertNotIn('tools/issue250/launch_validation.py',initial['files'])
                after_launcher=benchmark_entry.source_binding('validation')
                (repo/'tools/graph_infra/ego_gnn.py').write_text('VALUE=3\n')
                self.assertNotEqual(after_launcher,benchmark_entry.source_binding('validation'),
                    'changing an executed external source must invalidate checkpoint evidence')


    def test_rank0_calibration_does_not_advance_rotating_floor_history(self):
        import train_rqvae
        import assignment_floor
        ids=train_rqvae._calibration_ids(np.arange(512,dtype=np.int64),'cpu')
        width=max(train_rqvae.CODEBOOK_SIZE)
        history={key:value.clone() for key,value in assignment_floor._COUNTS.items()}
        assignment_floor.constrained_assignment(torch.zeros(len(ids),width),1,counts_key=984251)
        self.assertEqual(set(assignment_floor._COUNTS),set(history),
                         'rank0-only calibration cannot start a rank-local rotating history')
        for key,value in history.items():
            self.assertTrue(torch.equal(value,assignment_floor._COUNTS[key]))
        with self.assertRaisesRegex(ValueError,'codebook width'):
            train_rqvae._calibration_ids(np.arange(8,dtype=np.int64),'cpu')


if __name__=='__main__':unittest.main()
