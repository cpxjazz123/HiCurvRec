import tempfile
from pathlib import Path
import json
import unittest
import numpy as np
from tools.graph_infra.interfaces_v3 import validate_ids, verify, TrainingAssets, HeldOutEvaluation
from tools.graph_infra.assets import sha256


class InterfaceTests(unittest.TestCase):
    def test_id_mismatch(self):
        for ids in [np.array([1,0]), np.array([0,0]), np.array([0.,1.])]:
            with self.assertRaises(ValueError): validate_ids(ids,np.arange(2))
        validate_ids(np.array([0,1]),np.arange(2))

    def test_tamper_and_route(self):
        root = Path('results/research/issue247')
        with tempfile.TemporaryDirectory(dir=root) as temp:
            p = Path(temp); target=p/'data.npy'; np.save(target,np.arange(2))
            manifest = p/'manifest.json'
            manifest.write_text(json.dumps({'assets':[{'name':'x','path':str(target),'sha256':sha256(target)}]}))
            verify(manifest,Path.cwd())
            target.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'manifest hash'): verify(manifest,Path.cwd())
        train = TrainingAssets.__new__(TrainingAssets)
        train.entries={'strict_test':{'use':'evaluation'}}
        with self.assertRaisesRegex(ValueError,'training cannot'): train.load('strict_test')
        evaluator = HeldOutEvaluation.__new__(HeldOutEvaluation)
        evaluator.entries={'train':{'use':'training'}}
        with self.assertRaisesRegex(ValueError,'strict evaluation'): evaluator.load('train')


if __name__ == '__main__': unittest.main()
