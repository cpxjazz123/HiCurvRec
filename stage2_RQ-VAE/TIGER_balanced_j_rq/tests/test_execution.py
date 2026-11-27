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
ARM='TIGER_module_budget_prefix_rq'


class ExecutionTests(unittest.TestCase):
    def test_actual_launcher_changes_invalidate_source_evidence(self):
        import benchmark_entry
        with tempfile.TemporaryDirectory(dir=ROOT/'results/research/issue256/rounds/r01') as directory:
            repo=Path(directory);tree=repo/('stage2_RQ-VAE/'+ARM)
            (tree/'config').mkdir(parents=True)
            (tree/'benchmark_entry.py').write_text('VALUE=1\n')
            (tree/'train_rqvae.py').write_text('import tools.graph_infra.ego_gnn\n')
            for name in ('assets/manifest.json','tools/issue250/launch_validation.py',
                         'tools/issue256/launch_diagnostics.py','tools/issue253/performance.py',
                         'tools/research/stage2/launcher.py','tools/graph_infra/ego_gnn.py',
                         'tools/graph_infra/assets.py','results/research/issue256/rounds/r01/CONTRACT.md'):
                path=repo/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('VALUE=1\n')
            with patch.object(benchmark_entry,'ROOT',repo),patch.object(
                    benchmark_entry,'__file__',str(tree/'benchmark_entry.py')):
                initial=benchmark_entry.source_binding('validation')
                (repo/'tools/issue256/launch_diagnostics.py').write_text('VALUE=2\n')
                self.assertNotEqual(initial,benchmark_entry.source_binding('validation'),
                    'changing the executed launcher must invalidate checkpoint evidence')
                self.assertIn('tools/research/stage2/launcher.py',initial['files'])
                self.assertNotIn('tools/issue250/launch_validation.py',initial['files'])
                after_launcher=benchmark_entry.source_binding('validation')
                (repo/'tools/graph_infra/ego_gnn.py').write_text('VALUE=3\n')
                self.assertNotEqual(after_launcher,benchmark_entry.source_binding('validation'),
                    'changing an executed external source must invalidate checkpoint evidence')


    def test_calibration_uses_actual_sampler_without_advancing_rng_or_floor_history(self):
        from itertools import islice
        from torch.utils.data import DataLoader, DistributedSampler
        import train_rqvae
        import assignment_floor
        population=torch.arange(8192)
        dataset=train_rqvae._BatchedTensorDataset(population.float()[:,None],population)
        sampler=DistributedSampler(dataset,num_replicas=4,rank=0,shuffle=True,seed=42)
        loader=DataLoader(dataset,batch_size=1024,sampler=sampler,
                         collate_fn=train_rqvae._batch_passthrough)
        state=torch.random.get_rng_state()
        _,ids=train_rqvae._calibration_batch(loader,'cpu')
        expected=population.numpy()[np.fromiter(islice(iter(sampler),1024),dtype=np.int64)]
        np.testing.assert_array_equal(ids.numpy(),expected)
        self.assertTrue(torch.equal(state,torch.random.get_rng_state()),
                        'calibration identity lookup must not consume global training RNG')
        history={key:value.clone() for key,value in assignment_floor._COUNTS.items()}
        assignment_floor.constrained_assignment(torch.zeros(len(ids),256),1,counts_key=984251)
        self.assertEqual(set(assignment_floor._COUNTS),set(history))
        for key,value in history.items():
            self.assertTrue(torch.equal(value,assignment_floor._COUNTS[key]))
        with self.assertRaisesRegex(ValueError,'full per-rank batch'):
            small=train_rqvae._BatchedTensorDataset(torch.zeros(8,1),torch.arange(8))
            train_rqvae._calibration_batch(DataLoader(small,batch_size=1024,
                collate_fn=train_rqvae._batch_passthrough),'cpu')


if __name__=='__main__':unittest.main()
