import json, sys, tempfile, unittest
from pathlib import Path
from unittest import mock
import numpy as np
import pandas as pd
import torch

# Resolve the candidate tree from the repository root so the suite runs both
# from the cases directory and from a materialised copy.
def _repo_root(start):
    for parent in [start, *start.parents]:
        if (parent / "stage2_RQ-VAE").is_dir():
            return parent
    raise RuntimeError("cannot locate the repository root from " + str(start))


ROOT = _repo_root(Path(__file__).resolve())
SOURCE = ROOT / "stage2_RQ-VAE/TIGER_c117_transition_residual_poincare"
sys.path.insert(0, str(SOURCE))
import train_rqvae as trainer

class TrainerEntrypointTests(unittest.TestCase):
    def test_main_trains_fixture_and_writes_real_checkpoint_and_sids(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            emb = root / "emb.npy"
            frame_path = root / "train.parquet"
            out = root / "artifacts"
            np.save(emb, np.asarray([[.1,.2,.3], [.4,.2,.1], [.2,.5,.3], [.7,.1,.2]], dtype=np.float32))
            pd.DataFrame({"user":[1,1,2], "history":[[0],[1],[2]], "target":[1,2,3]}).to_parquet(frame_path)
            small = lambda: type("C", (), dict(hidden_sizes=(8,), codebook_num=3,
                codebook_size=(2,2,2), codebook_dim=2, dropout=0., beta=.25,
                vq_type="vq", ema_decay=.99, fix_code_embs=False, sk_epsilon=.003,
                sk_iters=2, geometry="transition_residual_poincare", curvature=1.,
                tau=.1, radius=.99999, delta=1e-7))()
            with mock.patch.multiple(trainer, EMBEDDING_FILE=emb, TRAIN_FILE=frame_path,
                    EPOCHS=1, BATCH_SIZE_PER_RANK=4, HIDDEN_SIZES=(8,), CODEBOOK_NUM=3,
                    CODEBOOK_SIZE=(2,2,2), CODEBOOK_DIM=2, WARMUP_EPOCHS=0,
                    EVAL_INTERVAL=1, NUM_WORKERS=0, PCA_DIM=0, XAVIER_INIT=True,
                    LOG_DIR=root/"logs", METRICS_PATH=root/"logs"/"metrics.jsonl",
                    CHECKPOINT=out/"rqvae_best.pth", OUTPUT_SID=out/"sids.npy",
                    OUTPUT_JSON=out/"item_sids.json", _RESULT_DIR=out), \
                 mock.patch.object(trainer, "_tokenizer_config", side_effect=small), \
                 mock.patch.object(trainer, "initialize_tiger_weights", wraps=trainer.initialize_tiger_weights):
                trainer.main()
            ckpt = torch.load(out/"rqvae_best.pth", map_location="cpu", weights_only=False)
            sid = np.load(out/"sids.npy")
            exported = json.loads((out/"item_sids.json").read_text())
            self.assertEqual(ckpt["epoch"], 1)
            self.assertEqual(ckpt["selection"], "final_epoch")
            self.assertTrue(ckpt["state_dict"])
            self.assertEqual(sid.shape, (4, 4))
            self.assertEqual(len(np.unique(sid, axis=0)), 4)
            self.assertEqual(np.asarray([exported[str(i)] for i in range(4)]).tolist(), sid.tolist())
            # The actual main loop calls RQVAE with all-item and sampled-edge
            # arguments; resulting finite transition loss is recorded in its run log.
            records = [json.loads(x) for x in (root/"logs"/"metrics.jsonl").read_text().splitlines()]
            train = [r for r in records if r["event"] == "train"]
            self.assertTrue(train)
            self.assertTrue(np.isfinite(train[-1]["transition_loss"]))
            self.assertGreater(train[-1]["transition_loss"], 0.0)
