"""Issue #145 entrypoint: one real epoch of the tree's own main(), on CPU."""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd
import torch

TREE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TREE))


def _load_trainer():
    spec = importlib.util.spec_from_file_location("issue145_cpu_trainer", TREE / "train_rqvae.py")
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    return trainer


class EntryPointTest(unittest.TestCase):
    def test_one_epoch_exports_three_tokens_per_item(self):
        trainer = _load_trainer()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            arm = root / "arm"
            constants = {
                "ARM_NAME": "TIGER_af3sign_capacity", "EPOCHS": 1,
                "BATCH_SIZE_PER_RANK": 4, "HIDDEN_SIZES": (8,), "CODEBOOK_SIZE": (4, 4, 4),
                "NUM_WORKERS": 0, "EVAL_INTERVAL": 1, "WARMUP_EPOCHS": 0,
                "RESULT_DIR": arm, "CHECKPOINT": arm / "snapshot.pth",
                "OUTPUT_SID": arm / "sids.npy", "OUTPUT_JSON": arm / "sids.json",
                "LOG_DIR": arm / "logs", "METRICS_PATH": arm / "logs" / "metrics.jsonl",
            }
            embeddings = np.random.default_rng(31).normal(size=(8, 8)).astype(np.float32) * 0.1
            frame = pd.DataFrame({"target": np.arange(8),
                                  "history": [[(i + 1) % 8] for i in range(8)]})
            patcher = mock.patch.multiple(trainer, **constants)
            patcher.start()
            self.addCleanup(patcher.stop)
            patcher = mock.patch.object(trainer, "load_embeddings",
                                         lambda path: embeddings.copy())
            patcher.start()
            self.addCleanup(patcher.stop)
            patcher = mock.patch.object(trainer.pd, "read_parquet",
                                         lambda *a, **k: frame.copy())
            patcher.start()
            self.addCleanup(patcher.stop)
            patcher = mock.patch.object(torch.cuda, "is_available", lambda: False)
            patcher.start()
            self.addCleanup(patcher.stop)
            for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "GENREC_SINGLE_GPU"):
                saved = os.environ.pop(key, None)
                if saved is not None:
                    self.addCleanup(os.environ.__setitem__, key, saved)

            trainer.main()

            sid = np.load(constants["OUTPUT_SID"])
            self.assertEqual(sid.shape, (8, 4))
            self.assertTrue(int(sid[:, :3].max()) < 4)
            payload = json.loads(constants["OUTPUT_JSON"].read_text())
            self.assertEqual(len(payload), 8)
            self.assertEqual(len({tuple(value) for value in payload.values()}), 8,
                             "the disambiguation token must make every SID unique")
            events = [json.loads(line) for line in
                      constants["METRICS_PATH"].read_text().splitlines() if line.strip()]
            self.assertEqual(events[-1]["event"], "train_end")

            from model.model import RQVAE
            checkpoint = torch.load(constants["CHECKPOINT"], map_location="cpu",
                                    weights_only=True)
            restored = RQVAE(trainer._tokenizer_config(), in_dim=8).eval()
            restored.load_state_dict(checkpoint["state_dict"], strict=True)
            with torch.no_grad():
                expected = restored.get_indices(torch.from_numpy(embeddings)).cpu().numpy()
            np.testing.assert_array_equal(sid[:, :3], expected)


if __name__ == "__main__":
    unittest.main()
