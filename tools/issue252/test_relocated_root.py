"""An immutable run source lives below results, which also contains stage2_RQ-VAE."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'stage2_RQ-VAE').is_dir())
TREE=ROOT/'stage2_RQ-VAE/TIGER_gnn_attention_sectional_rq'
sys.path.insert(0,str(TREE))


class RelocatedRootTests(unittest.TestCase):
    def test_real_train_and_model_root_ignore_the_results_marker(self):
        import train_rqvae
        from model import mechanism
        relocated=ROOT/'results/stage2_RQ-VAE/_runs/example/Arm/src'
        for module,symbol,filename in ((train_rqvae,'_repository_root','train_rqvae.py'),
                                       (mechanism,'repository_root','model/mechanism.py')):
            with self.subTest(symbol=symbol),patch.object(module,'__file__',str(relocated/filename)):
                self.assertEqual(getattr(module,symbol)(),ROOT,
                                 'run-source root discovery must not select results as the repository')

    def test_direct_entry_refusal_cannot_fall_back_to_single_card_training(self):
        import os
        import torch
        import train_rqvae
        from types import SimpleNamespace
        backend=SimpleNamespace(tool_stage2_start=lambda _: {
            'started':False,'refused':'four cards unavailable; admission refused'})
        with patch.dict(os.environ,{'RANK':'-1','GENREC_SINGLE_GPU':'0'}),patch(
                'tools.stage2.server.load_shared',return_value=backend),patch.object(
                torch.cuda,'device_count',return_value=0):
            with self.assertRaisesRegex(RuntimeError,'four cards'):
                train_rqvae._launch_via_torchrun()
            self.assertEqual(os.environ['GENREC_SINGLE_GPU'],'0',
                             'a refused formal entry must not switch to an unadmitted single card')


if __name__=='__main__':unittest.main()
