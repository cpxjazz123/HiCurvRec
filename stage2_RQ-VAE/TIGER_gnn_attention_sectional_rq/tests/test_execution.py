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


class ExecutionTests(unittest.TestCase):
    def test_actual_launcher_changes_invalidate_source_evidence(self):
        import validation_entry
        with tempfile.TemporaryDirectory(dir=ROOT/'results/research/issue251') as directory:
            repo=Path(directory);tree=repo/'candidate';(tree/'config').mkdir(parents=True)
            (tree/'validation_entry.py').write_text('controlled entry')
            for name in ('assets/manifest.json','tools/issue250/launch_validation.py',
                         'tools/issue251/launch_validation.py','tools/research/stage2/launcher.py'):
                path=repo/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('initial')
            with patch.object(validation_entry,'ROOT',repo),patch.object(
                    validation_entry,'__file__',str(tree/'validation_entry.py')):
                initial=validation_entry.source_binding()
                (repo/'tools/issue251/launch_validation.py').write_text('changed actual launch')
                self.assertNotEqual(initial,validation_entry.source_binding(),
                    'changing the executed launcher must invalidate checkpoint evidence')
                self.assertIn('tools/research/stage2/launcher.py',initial)
                self.assertNotIn('tools/issue250/launch_validation.py',initial)

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
