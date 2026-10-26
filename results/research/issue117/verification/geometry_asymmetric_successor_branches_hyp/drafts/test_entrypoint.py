import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd
import torch

TREE = Path('/fs04/ar57/wenyu/GeneRec/stage2_RQ-VAE/TIGER_c117_asymmetric_successor_branches')
if str(TREE) not in sys.path:
    sys.path.insert(0, str(TREE))

import train_rqvae as trainer


class ActualTrainerEntrypointTest(unittest.TestCase):
    def test_main_trains_writes_checkpoint_and_exports_sid_using_geometry_loss(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            embeddings_path = root / 'embeddings.npy'
            rng = np.random.default_rng(114)
            embeddings = rng.normal(size=(8, 4)).astype('float32')
            np.save(embeddings_path, embeddings)
            # Synthetic train-only split: every row is a legal training target; no eval/test data is supplied.
            train_frame = pd.DataFrame({'target': np.arange(len(embeddings), dtype=np.int64)})
            trainer.EMBEDDING_FILE = embeddings_path
            trainer.TRAIN_FILE = root / 'train_only.parquet'
            trainer.OUTPUT_SID = root / 'sids_for_hgrec.npy'
            trainer.OUTPUT_JSON = root / 'item_sids.json'
            trainer.CHECKPOINT = root / 'rqvae_final.pth'
            trainer.METRICS_PATH = root / 'metrics.jsonl'
            trainer.LOG_DIR = root / 'logs'
            trainer.EPOCHS = 1
            trainer.BATCH_SIZE_PER_RANK = 8
            trainer.NUM_WORKERS = 0
            trainer.EVAL_INTERVAL = 1
            trainer.WARMUP_EPOCHS = 0
            trainer.HIDDEN_SIZES = (8,)
            trainer.CODEBOOK_NUM = 3
            trainer.CODEBOOK_SIZE = (2, 2, 2)
            trainer.CODEBOOK_DIM = 32
            trainer.PCA_DIM = 0
            trainer.XAVIER_INIT = True
            trainer.GRADIENT_CLIP_NORM = 1.0
            trainer._metrics_truncated = False
            initial = {}
            observed = {}
            real_factory = trainer.RQVAE

            def recording_factory(config, *, in_dim):
                model = real_factory(config, in_dim=in_dim)
                initial.update({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
                real_forward = model.forward
                def recording_forward(batch):
                    total, rec, commit, code, tokens = real_forward(batch)
                    observed['values'] = tuple(float(v.detach()) for v in (total, rec, commit, code))
                    observed['curvature'] = model.rq.curvature
                    observed['radius'] = model.rq.radius
                    observed['mechanism_class'] = type(model.rq).__name__
                    return total, rec, commit, code, tokens
                model.forward = recording_forward
                return model

            def read_train_only(path):
                if Path(path) != trainer.TRAIN_FILE:
                    raise AssertionError('trainer attempted to read an unregistered data split')
                return train_frame.copy()

            with mock.patch.object(trainer, 'RQVAE', side_effect=recording_factory), \
                 mock.patch.object(trainer.pd, 'read_parquet', side_effect=read_train_only), \
                 mock.patch.object(torch.cuda, 'is_available', return_value=False):
                trainer.main()

            self.assertTrue(trainer.CHECKPOINT.is_file(), 'real trainer did not write its checkpoint')
            checkpoint = torch.load(trainer.CHECKPOINT, map_location='cpu', weights_only=False)
            self.assertEqual(checkpoint['epoch'], 1)
            self.assertEqual(checkpoint['selection'], 'final_epoch')
            self.assertEqual(set(checkpoint['state_dict']), set(initial))
            changed = [name for name, before in initial.items() if not torch.equal(before, checkpoint['state_dict'][name])]
            self.assertTrue(changed, 'checkpoint parameters show that optimizer.step did not update the model')
            self.assertTrue(trainer.OUTPUT_SID.is_file(), 'real trainer did not export SIDs')
            sid = np.load(trainer.OUTPUT_SID)
            self.assertEqual(sid.shape, (len(embeddings), 4))
            self.assertEqual(len(np.unique(sid, axis=0)), len(embeddings))
            payload = json.loads(trainer.OUTPUT_JSON.read_text(encoding='utf-8'))
            self.assertEqual(len(payload), len(embeddings))
            self.assertEqual(observed['mechanism_class'], 'SuccessorBranchQuantiser')
            self.assertEqual(observed['curvature'], trainer.CURVATURE_C)
            self.assertEqual(observed['radius'], trainer.GEOMETRY_RADIUS)
            total, rec, commit, code = observed['values']
            self.assertTrue(all(np.isfinite(v) for v in observed['values']))
            self.assertAlmostEqual(total, rec + trainer.BETA_COMMIT * commit + trainer.BETA_CODE * code, places=5)
            self.assertGreater(rec, 0.0)

