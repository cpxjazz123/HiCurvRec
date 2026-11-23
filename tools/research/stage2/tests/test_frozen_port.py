"""The effective PyTorch rendezvous must honor the frozen Stage2 port."""
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import torch.distributed.run as torchrun
from tools.research.stage2 import launcher

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'assets').is_dir())
PYTHON='/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9'


class LauncherPortTests(unittest.TestCase):
    def test_effective_rendezvous_keeps_frozen_port(self):
        captured={}
        def execute(command,**kwargs):
            with patch.object(torchrun,'elastic_launch') as elastic:
                elastic.return_value=lambda *args:None
                torchrun.run(torchrun.get_args_parser().parse_args(command[1:]))
                config=elastic.call_args.kwargs['config']
                captured['endpoint']=config.rdzv_endpoint
            return subprocess.CompletedProcess(command,0)
        with tempfile.TemporaryDirectory(dir=ROOT/'results/research/issue251') as directory:
            root=Path(directory);entry=root/'entry.py';entry.write_text('')
            with patch.object(launcher,'SERIAL_LOCK',root/'serial.lock'),patch.object(
                    launcher,'_visible_cards',return_value=4),patch.object(
                    launcher.patch_tree,'apply_floor',return_value={}),patch.object(
                    launcher.graph_batches,'install',return_value=None),patch.object(
                    launcher,'_universe',return_value=256),patch.object(
                    launcher.subprocess,'run',side_effect=execute):
                self.assertEqual(launcher.launch(entry,root/'result',python=PYTHON),0)
        self.assertEqual(captured['endpoint'].rsplit(':',1)[-1],'50202',
            'PyTorch standalone mode must not silently replace the frozen master port with a random port')


if __name__=='__main__':unittest.main()
