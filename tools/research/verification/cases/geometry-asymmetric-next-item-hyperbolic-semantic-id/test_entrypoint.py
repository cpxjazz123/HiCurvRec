import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import numpy as np
import pandas as pd
import torch

SOURCE = Path('/fs04/ar57/wenyu/GeneRec/stage2_RQ-VAE/TIGER_c117_hyperbolic_item_semantic_id')
sys.path.insert(0, str(SOURCE))
import train_rqvae as trainer
from model import RQVAE

class TrainerEntrypointTest(unittest.TestCase):
    def test_main_calls_candidate_loss_and_performs_optimizer_update(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            trainer.EPOCHS = 1
            trainer.BATCH_SIZE_PER_RANK = 4
            trainer.WARMUP_EPOCHS = 0
            trainer.EVAL_INTERVAL = 1
            trainer.NUM_WORKERS = 0
            trainer.XAVIER_INIT = True
            trainer.GRADIENT_CLIP_NORM = 0.0
            trainer.OPTIMIZER = 'AdamW'
            trainer.LR = 0.02
            trainer.WEIGHT_DECAY = 0.0
            trainer.SEED = 13
            trainer.PCA_DIM = 0
            trainer.EMBEDDING_FILE = root / 'unused.npy'
            trainer.TRAIN_FILE = root / 'unused.parquet'
            trainer.CHECKPOINT = root / 'checkpoint.pth'
            trainer.OUTPUT_SID = root / 'sid.npy'
            trainer.OUTPUT_JSON = root / 'sid.json'
            trainer.METRICS_PATH = root / 'metrics.jsonl'
            trainer.LOG_DIR = root
            # Use the actual candidate model, forward, compute_loss, optimizer and main loop.
            values = np.asarray([[0.1, 0.2, -0.2], [0.7, -0.3, 0.4], [-0.4, 0.6, 0.2], [0.3, 0.5, -0.6]], dtype=np.float32)
            frame = pd.DataFrame({'target': np.arange(len(values), dtype=np.int64)})
            cfg = SimpleNamespace(hidden_sizes=(4,), codebook_num=1, codebook_size=(2,), codebook_dim=2, dropout=0.0, beta=0.25, vq_type='vq', ema_decay=0.99, fix_code_embs=False, sk_epsilon=-1.0, sk_iters=-1, geometry='hyperbolic')
            loss_calls = []
            original_loss = RQVAE.compute_loss
            def observed_loss(model, x, reconstruction, qloss):
                result = original_loss(model, x, reconstruction, qloss)
                loss_calls.append(float(result[0].detach()))
                return result
            with mock.patch.object(trainer, 'load_embeddings', return_value=values), \
                 mock.patch.object(trainer.pd, 'read_parquet', return_value=frame), \
                 mock.patch.object(trainer, '_tokenizer_config', return_value=cfg), \
                 mock.patch.object(RQVAE, 'init_codebook', autospec=True, return_value=None), \
                 mock.patch.object(RQVAE, 'compute_loss', new=observed_loss):
                trainer.main()
            self.assertEqual(len(loss_calls), 1)
            self.assertTrue(np.isfinite(loss_calls).all())
            saved = torch.load(trainer.CHECKPOINT, map_location='cpu', weights_only=False)
            self.assertEqual(saved['epoch'], 1)
            self.assertEqual(saved['selection'], 'final_epoch')
            self.assertTrue(any(torch.isfinite(t).all() for t in saved['state_dict'].values() if torch.is_floating_point(t)))
            self.assertEqual(np.load(trainer.OUTPUT_SID).shape[0], len(values))

if __name__ == '__main__':
    unittest.main()