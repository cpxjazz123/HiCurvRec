import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch

TREE = Path(os.environ.get('CANDIDATE_TREE', Path(__file__).resolve().parents[1])).resolve()
sys.path.insert(0, str(TREE))
import train_rqvae as trainer

class CandidateEntrypointTest(unittest.TestCase):
    def test_real_trainer_main_writes_checkpoint_sid_and_uses_transition_loss(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            embeddings = np.asarray([[.1, .2, .3, .4], [.2, .1, .4, .3], [.4, .3, .2, .1], [.3, .4, .1, .2]], dtype=np.float32)
            emb_path = root / 'emb.npy'
            np.save(emb_path, embeddings)
            frame = pd.DataFrame({'user': [1, 1, 2, 2], 'history': [[0], [0, 1], [2], [2, 3]], 'target': [1, 2, 3, 0]})
            train_path = root / 'train.parquet'
            frame.to_parquet(train_path)
            ckpt, sid_path, json_path = root / 'model.pth', root / 'sid.npy', root / 'sid.json'
            config = dict(EMBEDDING_FILE=emb_path, TRAIN_FILE=train_path, CHECKPOINT=ckpt, OUTPUT_SID=sid_path, OUTPUT_JSON=json_path, METRICS_PATH=root/'metrics.jsonl', LOG_DIR=root/'logs', EPOCHS=1, BATCH_SIZE_PER_RANK=4, HIDDEN_SIZES=(8,), CODEBOOK_NUM=3, CODEBOOK_SIZE=(4, 4, 4), CODEBOOK_DIM=3, PCA_DIM=0, XAVIER_INIT=True, WARMUP_EPOCHS=0, EVAL_INTERVAL=1, NUM_WORKERS=0, GRADIENT_CLIP_NORM=1.0)
            for name, value in config.items():
                setattr(trainer, name, value)
            calls = []
            original_forward = trainer.RQVAE.forward
            def observed_forward(model, *args, **kwargs):
                result = original_forward(model, *args, **kwargs)
                calls.append((kwargs.get('all_embeddings') is not None, kwargs.get('edge_anchor') is not None, float(result[2].detach())))
                return result
            with patch.object(trainer.RQVAE, 'forward', observed_forward):
                trainer.main()
            self.assertTrue(calls, 'trainer.main did not execute model forward')
            self.assertTrue(all(a and b for a, b, _ in calls), 'real loss path omitted full item set or transition edges')
            self.assertTrue(any(np.isfinite(t) and t > 0 for _, _, t in calls), 'transition loss was not active')
            self.assertTrue(ckpt.is_file())
            saved = torch.load(ckpt, map_location='cpu', weights_only=False)
            self.assertEqual(saved['epoch'], 1)
            self.assertEqual(saved['selection'], 'final_epoch')
            self.assertTrue(saved['state_dict'])
            self.assertTrue(sid_path.is_file())
            sid = np.load(sid_path)
            self.assertEqual(sid.shape, (4, 4))
            self.assertEqual(len(np.unique(sid, axis=0)), 4)
            exported = json.loads(json_path.read_text())
            self.assertEqual(len(exported), 4)
            self.assertEqual([exported[str(i)] for i in range(4)], sid.tolist())

if __name__ == '__main__':
    unittest.main()
