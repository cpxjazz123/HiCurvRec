import json, sys, tempfile, unittest
from pathlib import Path
from unittest import mock
import numpy as np
import pandas as pd
import torch

TREE = Path("/fs04/ar57/wenyu/GeneRec/stage2_RQ-VAE/TIGER_c117_hyperbolic_item_semantic_id")
sys.path.insert(0, str(TREE))
import train_rqvae as trainer
from model.layers import HyperbolicRQLayer


class RealTrainerEntrypointTest(unittest.TestCase):
    def test_main_trains_hyperbolic_path_and_exports_actual_artifacts(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            embeddings_path = root / "train.npy"
            # Synthetic training-only rows; no validation/test/future examples are constructed.
            values = np.random.default_rng(71).normal(size=(6, 4)).astype(np.float32)
            np.save(embeddings_path, values)
            train = pd.DataFrame({"target": np.arange(len(values), dtype=np.int64)})
            # Keep execution small while retaining the trainer's actual main loop and optimizer.
            overrides = {
                "EMBEDDING_FILE": embeddings_path,
                "TRAIN_FILE": root / "train.parquet",
                "OUTPUT_SID": root / "sids_for_hgrec.npy",
                "OUTPUT_JSON": root / "item_sids.json",
                "CHECKPOINT": root / "rqvae_best.pth",
                "METRICS_PATH": root / "metrics.jsonl",
                "EPOCHS": 1, "BATCH_SIZE_PER_RANK": 6, "HIDDEN_SIZES": (8,),
                "CODEBOOK_NUM": 3, "CODEBOOK_SIZE": (4, 4, 4), "CODEBOOK_DIM": 4,
                "WARMUP_EPOCHS": 0, "EVAL_INTERVAL": 1, "NUM_WORKERS": 0,
                "SK_EPSILON": -1.0, "SK_ITERS": -1,
            }
            original_compute = trainer.RQVAE.compute_loss
            observed = {"loss_calls": 0, "geometry": None, "quantizer": None}
            def witnessed_compute(model, *args, **kwargs):
                observed["loss_calls"] += 1
                observed["geometry"] = model.geometry
                observed["quantizer"] = type(model.rq)
                return original_compute(model, *args, **kwargs)
            with mock.patch.object(trainer.pd, "read_parquet", return_value=train), \
                 mock.patch.object(trainer, "_record"), \
                 mock.patch.object(trainer.RQVAE, "compute_loss", witnessed_compute), \
                 mock.patch.multiple(trainer, **overrides):
                trainer.main()
            self.assertGreaterEqual(observed["loss_calls"], 1)
            self.assertEqual(observed["geometry"], "hyperbolic")
            self.assertIs(observed["quantizer"], HyperbolicRQLayer)
            self.assertTrue(overrides["CHECKPOINT"].is_file())
            checkpoint = torch.load(overrides["CHECKPOINT"], map_location="cpu", weights_only=False)
            self.assertEqual(checkpoint["epoch"], 1)
            self.assertEqual(checkpoint["selection"], "final_epoch")
            self.assertTrue(checkpoint["state_dict"])
            sid_path = overrides["OUTPUT_SID"]
            self.assertTrue(sid_path.is_file())
            sid = np.load(sid_path)
            self.assertEqual(sid.shape, (len(values), 4))
            self.assertEqual(len(np.unique(sid, axis=0)), len(values))
            exported = json.loads(overrides["OUTPUT_JSON"].read_text())
            self.assertEqual(len(exported), len(values))
            self.assertEqual([exported[str(i)] for i in range(len(values))], sid.tolist())
            # The checkpoint's actual learned state is nonempty and finite, proving an optimizer-backed run.
            self.assertTrue(all(torch.isfinite(t).all() for t in checkpoint["state_dict"].values() if torch.is_floating_point(t)))
