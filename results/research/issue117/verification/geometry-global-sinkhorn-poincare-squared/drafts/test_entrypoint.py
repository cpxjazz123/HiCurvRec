import json, sys, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch

TREE = Path("/fs04/ar57/wenyu/GeneRec/stage2_RQ-VAE/TIGER_c117_global_sinkhorn_poincare")
sys.path.insert(0, str(TREE))
import train_rqvae as trainer

class ActualTrainerEntrypoint(unittest.TestCase):
    def test_main_writes_checkpoint_sid_and_uses_transition_loss(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            emb_path = d / "emb.npy"
            embeddings = np.random.default_rng(7).normal(size=(8, 6)).astype("float32")
            np.save(emb_path, embeddings)
            # Longest train-only user sequences produce actual directed training edges.
            frame = pd.DataFrame({"user": [0, 1, 2, 3],
                                  "history": [[0,1,2], [1,2,3], [2,3,4], [3,4,5]],
                                  "target": [3,4,5,6]})
            ckpt, sid, jsid, metrics = (d/"model.pth", d/"sid.npy", d/"sid.json", d/"metrics.jsonl")
            calls = []
            initial = {}
            original_xavier = trainer.initialize_tiger_weights
            def capture_initial(model):
                original_xavier(model)
                initial.update({k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                                if k.startswith("encoder.")})
            original_forward = trainer.RQVAE.forward
            def observed_forward(model, *args, **kwargs):
                result = original_forward(model, *args, **kwargs)
                calls.append((kwargs.get("edges", args[2] if len(args) > 2 else None),
                              [float(v.detach()) for v in result]))
                return result
            patches = [
                patch.object(trainer, "EMBEDDING_FILE", emb_path),
                patch.object(trainer, "TRAIN_FILE", d/"train.parquet"),
                patch.object(trainer, "CHECKPOINT", ckpt),
                patch.object(trainer, "OUTPUT_SID", sid),
                patch.object(trainer, "OUTPUT_JSON", jsid),
                patch.object(trainer, "METRICS_PATH", metrics),
                patch.object(trainer, "EPOCHS", 1),
                patch.object(trainer, "BATCH_SIZE_PER_RANK", 8),
                patch.object(trainer, "WARMUP_EPOCHS", 0),
                patch.object(trainer, "EVAL_INTERVAL", 1),
                patch.object(trainer, "HIDDEN_SIZES", (16, 8)),
                patch.object(trainer, "CODEBOOK_SIZE", (4, 4, 4)),
                patch.object(trainer, "CODEBOOK_NUM", 3),
                patch.object(trainer, "CODEBOOK_DIM", 4),
                patch.object(trainer, "NUM_WORKERS", 0),
                patch.object(trainer, "_record", lambda *a, **k: None),
                patch.object(trainer, "initialize_tiger_weights", capture_initial),
                patch.object(trainer.pd, "read_parquet", return_value=frame),
                patch.object(trainer.RQVAE, "forward", observed_forward),
            ]
            try:
                for p in patches: p.start()
                trainer.main()
            finally:
                for p in reversed(patches): p.stop()
            self.assertTrue(ckpt.is_file(), "real trainer did not write checkpoint")
            self.assertTrue(sid.is_file(), "real trainer did not export SID")
            state = torch.load(ckpt, map_location="cpu", weights_only=False)
            self.assertEqual(state["epoch"], 1)
            self.assertEqual(state["selection"], "final_epoch")
            sid_array = np.load(sid)
            self.assertEqual(sid_array.shape, (8, 4))
            self.assertEqual(len(np.unique(sid_array, axis=0)), 8)
            self.assertEqual(len(json.loads(jsid.read_text())), 8)
            self.assertTrue(calls, "trainer never called model forward")
            self.assertTrue(all(c[0] is not None for c in calls), "trainer loss path omitted train transition edges")
            self.assertTrue(any(c[1][4] > 0 for c in calls), "actual transition loss was not active")
            self.assertTrue(initial, "trainer did not initialize the real encoder")
            self.assertTrue(any(not torch.equal(v, state["state_dict"][k]) for k, v in initial.items()),
                            "checkpoint encoder parameters show no optimizer update")
