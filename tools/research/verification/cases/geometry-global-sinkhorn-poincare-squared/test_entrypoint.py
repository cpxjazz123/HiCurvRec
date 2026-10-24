"""The real trainer must run and produce real artifacts."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def _repo_root(start):
    for parent in [start, *start.parents]:
        if (parent / "stage2_RQ-VAE").is_dir():
            return parent
    raise RuntimeError("cannot locate the repository root")


TREE = _repo_root(Path(__file__).resolve()) / "stage2_RQ-VAE/TIGER_c117_global_sinkhorn_poincare"
sys.path.insert(0, str(TREE))
import train_rqvae as trainer


class RealTrainerEntrypointTests(unittest.TestCase):
    def test_main_trains_the_frozen_transport_and_writes_real_artifacts(self):
        with tempfile.TemporaryDirectory() as work:
            root = Path(work)
            embedding_path = root / "emb.npy"
            train_path = root / "train.parquet"
            checkpoint_path = root / "rqvae.pth"
            sid_path = root / "sid.npy"
            json_path = root / "sid.json"
            metrics_path = root / "metrics.jsonl"

            rng = np.random.default_rng(4)
            np.save(embedding_path, rng.normal(size=(12, 8)).astype("float32"))
            frame = pd.DataFrame({"user": [0, 0, 1, 1, 2, 2],
                                  "history": [[0], [1], [2], [3], [4], [5]],
                                  "target": [1, 2, 3, 4, 5, 6]})
            frame.to_parquet(train_path, index=False)

            assignments = {
                "EMBEDDING_FILE": embedding_path, "TRAIN_FILE": train_path,
                "CHECKPOINT": checkpoint_path, "OUTPUT_SID": sid_path,
                "OUTPUT_JSON": json_path, "METRICS_PATH": metrics_path,
                "EPOCHS": 1, "BATCH_SIZE_PER_RANK": 12, "HIDDEN_SIZES": (8, 4),
                "CODEBOOK_NUM": 3, "CODEBOOK_SIZE": (4, 4, 4), "CODEBOOK_DIM": 32,
                "WARMUP_EPOCHS": 0, "EVAL_INTERVAL": 1, "NUM_WORKERS": 0,
                "PCA_DIM": 0, "XAVIER_INIT": True, "GRADIENT_CLIP_NORM": 1.0,
                "LOG_DIR": root / "logs", "GEOMETRY_RADIUS": 0.9,
            }
            previous = trainer._metrics_truncated
            try:
                for name, value in assignments.items():
                    setattr(trainer, name, value)
                trainer._metrics_truncated = False
                trainer.main()
            finally:
                trainer._metrics_truncated = previous

            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            self.assertEqual(checkpoint["epoch"], 1)
            self.assertEqual(checkpoint["selection"], "final_epoch")
            self.assertTrue(checkpoint["state_dict"])

            sid = np.load(sid_path)
            self.assertEqual(sid.shape, (12, trainer.CODEBOOK_NUM + 1))
            self.assertEqual(len(np.unique(sid, axis=0)), 12)
            exported = json.loads(json_path.read_text())
            self.assertEqual([int(x) for x in exported["0"]], sid[0].tolist())

            records = [json.loads(line) for line in metrics_path.read_text().splitlines()]
            train_rows = [row for row in records if row.get("event") == "train"]
            self.assertTrue(train_rows)
            self.assertTrue(np.isfinite(train_rows[-1]["loss"]))
            self.assertGreater(train_rows[-1]["recon"], 0.0, "reconstruction absent")
            self.assertTrue(np.isfinite(train_rows[-1]["transition_loss"]))
            self.assertTrue(any(row.get("event") == "ckpt_saved" for row in records))
            self.assertTrue(any(row.get("event") == "train_end" for row in records))


if __name__ == "__main__":
    unittest.main()
