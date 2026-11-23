"""Verification must not downgrade a newer immutable asset manifest."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from tools.graph_infra import build_assets_v5

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'assets').is_dir())

class VerifyReadonlyTests(unittest.TestCase):
    def test_verify_preserves_newer_manifest_bytes(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'results/research/issue251') as directory:
            repo=Path(directory);assets=repo/'assets';assets.mkdir()
            payload=assets/'token_schema.json';payload.write_text('{"tokens":2}\n')
            manifest=assets/'manifest.json'
            manifest.write_text(json.dumps({'version':'v6-cross-attention-251','assets':[
                {'name':'tokens','path':'assets/token_schema.json',
                 'sha256':hashlib.sha256(payload.read_bytes()).hexdigest()}],
                 'history':[{'version':'original'}]},indent=4)+'\n')
            before=manifest.read_bytes()
            with patch.object(build_assets_v5,'REPO',repo),patch.object(build_assets_v5,'ASSETS',assets),\
                 patch('sys.argv',['build_assets_v5','--verify']):
                self.assertEqual(build_assets_v5.main(),0)
            self.assertEqual(manifest.read_bytes(),before)

if __name__=='__main__':unittest.main()
