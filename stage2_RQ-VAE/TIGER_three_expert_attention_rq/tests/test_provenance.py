"""Stale assets/configuration cannot be presented as matching checkpoint evidence."""
from pathlib import Path
import sys
import json
import hashlib
import tempfile
import unittest
import torch

TREE=Path(__file__).resolve().parents[1]
ROOT=next(p for p in TREE.parents if (p/'assets').is_dir())
sys.path.insert(0,str(TREE))
from model.provenance import binding

class ProvenanceTests(unittest.TestCase):
    def test_binding_detects_asset_drift_and_source_change(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'results/research/issue253') as directory:
            repo=Path(directory);assets=repo/'assets';assets.mkdir();tree=repo/'candidate';tree.mkdir()
            item=assets/'features.json';item.write_text('{"value":1}\n')
            train=repo/'results/stage0_build_parquet/train.parquet'
            train.parent.mkdir(parents=True);train.write_bytes(b'frozen training rows')
            (assets/'manifest.json').write_text(json.dumps({'assets':[
                {'path':'assets/features.json','sha256':hashlib.sha256(item.read_bytes()).hexdigest()}],
                'inputs':{'train':hashlib.sha256(train.read_bytes()).hexdigest()}}))
            config=tree/'config.json';config.write_text('{"experts":3}\n')
            first=binding(repo,tree)
            config.write_text('{"experts":2}\n')
            self.assertNotEqual(first,binding(repo,tree))
            item.write_text('{"value":2}\n')
            with self.assertRaisesRegex(ValueError,'asset hash mismatch'):
                binding(repo,tree)

    def test_training_split_drift_invalidates_binding(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'results/research/issue253') as directory:
            repo=Path(directory);(repo/'assets').mkdir();tree=repo/'candidate';tree.mkdir()
            train=repo/'results/stage0_build_parquet/train.parquet'
            train.parent.mkdir(parents=True);train.write_bytes(b'frozen training rows')
            manifest={'assets':[],'inputs':{'train':hashlib.sha256(train.read_bytes()).hexdigest()}}
            (repo/'assets/manifest.json').write_text(json.dumps(manifest))
            binding(repo,tree)
            train.write_bytes(b'different training split')
            with self.assertRaisesRegex(ValueError,'training input hash mismatch'):
                binding(repo,tree)

    def test_checkpoint_binding_rejects_source_drift_since_capture(self):
        from unittest.mock import patch
        from model import mechanism
        with tempfile.TemporaryDirectory(dir=ROOT/'results/research/issue253') as directory:
            repo=Path(directory);(repo/'assets').mkdir();tree=repo/'candidate'
            (tree/'model').mkdir(parents=True)
            train=repo/'results/stage0_build_parquet/train.parquet'
            train.parent.mkdir(parents=True);train.write_bytes(b'frozen training rows')
            (repo/'assets/manifest.json').write_text(json.dumps(
                {'assets':[],'inputs':{'train':hashlib.sha256(train.read_bytes()).hexdigest()}}))
            for name in ('results/research/issue253/CONTRACT.md',
                         'tools/graph_infra/ego_gnn.py','tools/graph_infra/assets.py'):
                path=repo/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('frozen')
            source=tree/'model/mechanism.py';source.write_text('executed source')
            (tree/'train_rqvae.py').write_text('import tools.graph_infra.ego_gnn\n')
            with patch.object(mechanism,'ROOT',repo),patch.object(mechanism,'__file__',str(source)):
                initial=mechanism.RQVAE.provenance(object())
                source.write_text('new disk source, not the executed source')
                with self.assertRaisesRegex(ValueError,'changed during execution'):
                    mechanism.RQVAE.provenance(object(),expected=initial)
    def test_missing_candidate_provenance_rejects_checkpoint(self):
        import train_rqvae
        from model.mechanism import RQVAE
        model=RQVAE(train_rqvae._tokenizer_config(),in_dim=768)
        with tempfile.TemporaryDirectory(dir=ROOT/'results/research/issue253') as directory:
            file=Path(directory)/'unbound.pth'
            torch.save({'state_dict':model.state_dict()},file)
            with self.assertRaisesRegex(ValueError,'provenance mismatch'):
                model.load_checkpoint(file)

if __name__=='__main__':unittest.main()
