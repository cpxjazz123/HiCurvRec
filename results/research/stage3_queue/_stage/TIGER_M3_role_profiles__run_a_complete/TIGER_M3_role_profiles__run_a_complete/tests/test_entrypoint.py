"""Issue #142 entrypoint: one real epoch of the tree's own main(), on CPU.

The fixture is synthetic: eight embedding rows, a toy TRAIN frame, a toy AF3
support and temporary output paths. The tree's constants and its ``main()`` are
the real ones, so what is proved here is that the mechanism the trainer runs is
the mechanism that exports the identifier -- including the post-SID objective
being active and recorded.
"""
from __future__ import annotations

import importlib.util
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

TREE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TREE))


def _load_trainer():
    spec = importlib.util.spec_from_file_location("issue142_cpu_trainer", TREE / "train_rqvae.py")
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    return trainer


def _toy_structure(n_items: int = 8):
    # Both adjacency classes must be present: the calibration matches a density,
    # and a support with no non-edge has no density to match.
    pairs = np.asarray([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 0],
                        [0, 2], [1, 3], [2, 4], [3, 5]], dtype=np.int64)
    targets = np.asarray([0.4, 0.45, 0.5, 0.55, 0.6, 0.5, 0.45, 0.4,
                          0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    graph = {"item_ids": np.arange(n_items, dtype=np.int64), "n_nodes": n_items,
             "rows": pairs[:, 0], "cols": pairs[:, 1],
             "weights": targets.astype(np.float64)}
    support = {"pairs": pairs, "is_edge": targets > 0, "edges": int((targets > 0).sum()),
               "distances": np.linspace(0.5, 1.5, len(pairs))}
    return graph, support, targets


class EntryPointTest(unittest.TestCase):
    def test_one_epoch_exports_and_records_the_mechanism(self):
        trainer = _load_trainer()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            arm = root / "arm"
            constants = {
                "ARM_NAME": "TIGER_M4_post_sid_af3", "EPOCHS": 1,
                "BATCH_SIZE_PER_RANK": 4, "HIDDEN_SIZES": (8,), "CODEBOOK_SIZE": (4, 4, 4),
                "NUM_WORKERS": 0, "EVAL_INTERVAL": 1, "WARMUP_EPOCHS": 0,
                "RESULT_DIR": arm, "CHECKPOINT": arm / "snapshot.pth",
                "OUTPUT_SID": arm / "sids.npy", "OUTPUT_JSON": arm / "sids.json",
                "LOG_DIR": arm / "logs", "METRICS_PATH": arm / "logs" / "metrics.jsonl",
                "ROLE_FIT": root / "relational_fit.json",
                "BEHAVIOUR_ON": True, "GEOMETRY_ON": True, "CURVATURE": 1.0,
            }
            embeddings = np.random.default_rng(31).normal(size=(8, 8)).astype(np.float32) * 0.1
            frame = pd.DataFrame({"target": np.arange(8),
                                  "history": [[(i + 1) % 8] for i in range(8)]})
            patcher = mock.patch.multiple(trainer, **constants)
            patcher.start()
            self.addCleanup(patcher.stop)
            for name, value in (("load_embeddings", lambda path: embeddings.copy()),
                                ("_frozen_structure", lambda: _toy_structure())):
                patcher = mock.patch.object(trainer, name, value)
                patcher.start()
                self.addCleanup(patcher.stop)
            patcher = mock.patch.object(trainer.pd, "read_parquet", lambda *a, **k: frame.copy())
            patcher.start()
            self.addCleanup(patcher.stop)
            patcher = mock.patch.object(torch.cuda, "is_available", lambda: False)
            patcher.start()
            self.addCleanup(patcher.stop)
            for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE"):
                os_environ = __import__("os").environ
                saved = os_environ.pop(key, None)
                if saved is not None:
                    self.addCleanup(os_environ.__setitem__, key, saved)

            trainer.main()

            sid = np.load(constants["OUTPUT_SID"])
            self.assertEqual(sid.shape, (8, 4))
            payload = json.loads(constants["OUTPUT_JSON"].read_text())
            self.assertEqual(len(payload), 8)
            self.assertTrue(constants["ROLE_FIT"].is_file(),
                            "the frozen relational fit must be written on the first run")
            fit = json.loads(constants["ROLE_FIT"].read_text())
            self.assertIn("a", fit)
            self.assertIn("b", fit)
            self.assertIn("sign_agrees", fit)
            events = [json.loads(line) for line in
                      constants["METRICS_PATH"].read_text().splitlines() if line.strip()]
            self.assertTrue(any(event["event"] == "role" for event in events),
                            "an active role term must be recorded every epoch")
            self.assertEqual(events[-1]["event"], "train_end")
            # The exported identifier must be reproducible from the saved weights.
            from model.model import MixedRQVAE
            checkpoint = torch.load(constants["CHECKPOINT"], map_location="cpu",
                                    weights_only=True)
            restored = MixedRQVAE(trainer._tokenizer_config(), in_dim=8, curvature=1.0).eval()
            restored.load_state_dict(checkpoint["state_dict"], strict=True)
            with torch.no_grad():
                expected = restored.get_indices(torch.from_numpy(embeddings)).cpu().numpy()
            np.testing.assert_array_equal(sid[:, :3], expected)


if __name__ == "__main__":
    unittest.main()
