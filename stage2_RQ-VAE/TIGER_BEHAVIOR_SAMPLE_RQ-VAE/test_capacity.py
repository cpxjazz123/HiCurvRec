"""Closure and invariant checks for the frozen Stage-2 representation capacity."""

from __future__ import annotations

import importlib.util
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
STAGE2 = Path(__file__).resolve().parent
STAGE3_DATASET = ROOT / "stage3_T5Train" / "data" / "dataset.py"
if str(STAGE2) not in sys.path:
    sys.path.insert(0, str(STAGE2))

import train_rqvae as trainer
from model import RQVAE


FROZEN_DIM = 32
FROZEN_SIZES = (256, 256, 256)
FROZEN_HIDDEN = (512, 256, 128)
CLOSED_ARMS = (
    "cap_d8", "cap_d16", "cap_d64", "cap_d128", "cap_l1c128",
    "cap_l1c512", "cap_d16_l1c512", "cap_d64_l1c128", "cap_d32_wide",
)


def load_stage3_dataset_module():
    spec = importlib.util.spec_from_file_location("capacity_stage3_dataset", STAGE3_DATASET)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise AssertionError("Stage 3 dataset module has no loader")
    spec.loader.exec_module(module)
    return module


class CapacityFreezeTests(unittest.TestCase):
    def test_each_capacity_arm_import_is_rejected_with_freeze_message(self):
        source_path = Path(trainer.__file__)
        source = source_path.read_text(encoding="utf-8")
        for arm in CLOSED_ARMS:
            with self.subTest(arm=arm):
                selected_source = re.sub(r'(?m)^EXPERIMENT_ARM\s*=\s*"[^"]*"', 'EXPERIMENT_ARM = "{}"'.format(arm), source, count=1)
                self.assertNotEqual(selected_source, source, "hard-coded arm switch not found")
                namespace = {
                    "__file__": str(source_path),
                    "__name__": "capacity_closed_arm_probe",
                }
                with self.assertRaisesRegex(RuntimeError, "Capacity freeze is active"):
                    exec(compile(selected_source, str(source_path), "exec"), namespace)

    def test_frozen_invariants_hold_for_representative_open_arms(self):
        self.assertEqual(trainer.CODEBOOK_DIM, FROZEN_DIM)
        self.assertEqual(trainer.CODEBOOK_SIZE, FROZEN_SIZES)
        self.assertEqual(trainer.HIDDEN_SIZES, FROZEN_HIDDEN)
        self.assertEqual(trainer.CODEBOOK_NUM, 3)
        self.assertEqual(trainer.EPOCHS, 3000)
        self.assertEqual(trainer.SEED, 42)
        for arm in ("resample", "E0", "H1", "E1"):
            with self.subTest(arm=arm):
                self.assertNotIn(arm, trainer._CAPACITY_ARMS)
                if arm in {"E0", "H1", "E1"}:
                    shared = trainer._resolve_geometry_arm(arm)["shared"]
                    self.assertEqual(shared[0:3], (FROZEN_DIM, FROZEN_SIZES, FROZEN_HIDDEN))
                    self.assertEqual(shared[3], 3000)

        model = RQVAE(trainer._tokenizer_config(), in_dim=6)
        encoded = model.encoder(torch.randn(4, 6))
        self.assertEqual(tuple(encoded.shape), (4, FROZEN_DIM))
        self.assertEqual(tuple(layer.n_embed for layer in model.rq.vq_layers), FROZEN_SIZES)

    def test_collision_ids_and_stage3_vocab_use_frozen_codebook_base(self):
        raw = np.asarray([[0, 1, 2], [0, 1, 2], [4, 5, 6]], dtype=np.int64)
        extended = trainer._extend_collisions(raw, list(FROZEN_SIZES))
        collision_ids = extended[:, -1]
        semantic_base = sum(FROZEN_SIZES)
        self.assertTrue(np.all(collision_ids >= semantic_base))
        self.assertNotEqual(int(collision_ids[0]), int(collision_ids[1]))

        with tempfile.TemporaryDirectory() as temp_root:
            sid_json = Path(temp_root) / "item_sids.json"
            sid_json.write_text(
                json.dumps({str(i): row.tolist() for i, row in enumerate(extended)}),
                encoding="utf-8",
            )
            dataset_module = load_stage3_dataset_module()
            dataset_module.process_data = lambda *args, **kwargs: []
            dataset = dataset_module.GenRecDataset(
                dataset_path=Path(temp_root) / "synthetic.parquet",
                code_path=sid_json,
                mode="train",
                codebook_size=FROZEN_SIZES,
                max_len=4,
            )
        self.assertEqual(dataset.base_user_token, int((extended + 1).max()) + 1)
        self.assertGreater(dataset.vocab_size, dataset.max_sid_token)

    def test_frozen_encoder_and_behavior_loss_have_nonzero_gradients(self):
        torch.manual_seed(42)
        model = RQVAE(trainer._tokenizer_config(), in_dim=6)
        embeddings = torch.randn(8, 6)
        residuals = trainer._residuals_with_trace(model, embeddings)
        source_ids = torch.arange(4)
        target_ids = torch.arange(4, 8)
        behavior_loss = trainer.behavior_contrastive_loss(residuals, source_ids, target_ids)
        gradients = torch.autograd.grad(
            behavior_loss, tuple(model.encoder.parameters()), allow_unused=True,
        )
        self.assertTrue(any(
            gradient is not None and bool(torch.any(gradient != 0))
            for gradient in gradients
        ))
        self.assertTrue(bool(torch.isfinite(behavior_loss)))


if __name__ == "__main__":
    unittest.main()
