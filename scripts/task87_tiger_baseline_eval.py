#!/usr/bin/env python
"""Task #87: TIGER-aligned baseline Stage 4 evaluation.

Mirrors scripts/task14_s4_item_eval_l5.py but uses the new _tiger yamls.
Computes item-level Recall@5, Recall@10, NDCG@5, NDCG@10 on Toys.

Usage:
    python scripts/task87_tiger_baseline_eval.py \\
        --constrained_pt logs/task87_s4_tiger_inference/runs/task87_s4/pickle/merged_predictions_tensor.pt \\
        --sid logs/task87_s2_rqvae_inference/runs/task87_s2_infer/pickle/cluster_ids.pt \\
        --ckpt logs/task87_s3_tiger_train/runs/task87_s3_train/checkpoints/<best>.ckpt \\
        --out_json verdicts/task87_tiger_baseline_eval.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/GeneRec")
sys.path.insert(0, str(REPO_ROOT))
os.chdir(str(REPO_ROOT))

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import OmegaConf

import torch

from src.components.eval_metrics import NDCG, Recall, SIDRetrievalEvaluator
from src.data.loading.datamodules.sequence_datamodule import SequenceDataModule

TOYS_DATA_DIR = str(REPO_ROOT / "data" / "amazon_data" / "toys")
TOP_K_LIST = [5, 10]
SEQUENCE_LENGTH = 120
NUM_HIERARCHIES = 4  # 3 RQ-VAE levels + 1 dedup digit


def _build_patched_dl(sid_path: str, ckpt_path: str) -> "OmegaConf":
    """Compose a predict-only Hydra config using the new tiger_inference_tiger yaml."""
    GlobalHydra.instance().clear()
    initialize_config_dir(
        config_dir=str(REPO_ROOT / "configs"),
        job_name="task87_eval",
        version_base=None,
    )
    overrides = [
        "experiment=tiger_inference_tiger",
        f"data_dir={TOYS_DATA_DIR}",
        f"semantic_id_path={sid_path}",
        f"ckpt_path={ckpt_path}",
        f"num_hierarchies={NUM_HIERARCHIES}",
        f"sequence_length={SEQUENCE_LENGTH}",
        "trainer.accelerator=cpu",
        "trainer.devices=1",
        f"paths.root_dir={REPO_ROOT}",
    ]
    cfg = compose(config_name="train", overrides=overrides)

    OmegaConf.set_struct(cfg.data_loading, False)
    OmegaConf.set_struct(cfg.data_loading.predict_dataloader_config.dataloader.dataset_config, False)
    pdc = cfg.data_loading.predict_dataloader_config.dataloader

    # Inject labels + train-style collate for evaluation (so we get the next-k target).
    pdc.labels = OmegaConf.create({
        "sequence_data": {
            "transform": {
                "_target_": "src.data.loading.components.label_function.NextKTokenMasking",
                "next_k": "${model.num_hierarchies}",
            }
        }
    })
    pdc.collate_fn = OmegaConf.create({
        "_target_": "src.data.loading.components.collate_functions.collate_fn_train",
        "_partial_": True,
        "labels": {
            "sequence_data": {
                "transform": {
                    "_target_": "src.data.loading.components.label_function.NextKTokenMasking",
                    "next_k": "${model.num_hierarchies}",
                }
            }
        },
        "sequence_length": "${data_loading.predict_dataloader_config.dataloader.sequence_length}",
        "masking_token": "${data_loading.predict_dataloader_config.dataloader.masking_token}",
        "padding_token": "${data_loading.predict_dataloader_config.dataloader.padding_token}",
        "oov_token": "${data_loading.predict_dataloader_config.dataloader.oov_token}",
    })
    pdc.dataset_config.file_format = "tfrecord.gz"
    pdc.batch_size_per_device = 8
    pdc.num_workers = 4
    return cfg


@torch.no_grad()
def evaluate(
    constrained_pt: str,
    sid_path: str,
    ckpt_path: str,
) -> dict:
    # Predictions tensor: shape (n_users, top_k=10, num_hier=4)
    pred = torch.load(constrained_pt, map_location="cpu", weights_only=False).long()
    print(f"[info] constrained tensor shape: {tuple(pred.shape)} dtype={pred.dtype}")
    if pred.dim() != 3 or pred.shape[2] != NUM_HIERARCHIES:
        raise ValueError(
            f"bad shape {tuple(pred.shape)}, expected (n_users, top_k, {NUM_HIERARCHIES})"
        )
    n_total_users, top_k, num_hier = pred.shape
    print(f"[info] {n_total_users} users x {top_k} candidates x {num_hier} digits")

    # Lazy full-catalog SID lookup: (4, N_catalog)
    sid_catalog = torch.load(sid_path, map_location="cpu", weights_only=False).long()
    print(f"[info] catalog SID tensor shape: {tuple(sid_catalog.shape)}")
    if sid_catalog.shape[0] != NUM_HIERARCHIES:
        raise ValueError(
            f"SID first-dim {sid_catalog.shape[0]} != num_hierarchies {NUM_HIERARCHIES}"
        )

    cfg = _build_patched_dl(sid_path=sid_path, ckpt_path=ckpt_path)
    pdc = instantiate(cfg.data_loading.predict_dataloader_config.dataloader)
    dm = SequenceDataModule(
        train_dataloader_config=None,
        val_dataloader_config=None,
        test_dataloader_config=None,
        predict_dataloader_config=pdc,
    )

    class _StubTrainer:
        world_size = 1
        is_global_zero = True
        global_rank = 0
        local_rank = 0

    dm.trainer = _StubTrainer()  # type: ignore
    dm.prepare_data()
    dm.setup(stage="predict")

    evaluator = SIDRetrievalEvaluator(
        metrics={"Recall": Recall, "NDCG": NDCG},
        top_k_list=TOP_K_LIST,
    )
    evaluator.reset()

    predict_loader = dm.predict_dataloader()
    if isinstance(predict_loader, tuple):
        predict_loader = predict_loader[0]

    score_template = torch.arange(top_k, 0, -1, dtype=torch.float32) / top_k

    n_users = 0
    n_missing = 0
    for batch_idx, batch in enumerate(predict_loader):
        model_input, label_data = batch
        # Treat each user sequentially, regardless of internal batch size.
        # ... (See task14_s4_item_eval_l5.py for full pattern.)
        # Simplified: this script borrows the same metric-compute pattern via
        # SIDRetrievalEvaluator.update() with ground-truth + predicted SID tuples.
        # Actual implementation is filled in once Stage 3 finishes.
        break  # placeholder; remove once Stage 3 ckpt available

    return {
        "constraint_path": constrained_pt,
        "sid_path": sid_path,
        "ckpt_path": ckpt_path,
        "n_users_evaluated": n_users,
        "n_missing_gt": n_missing,
        "status": "placeholder — full implementation pending Stage 3 ckpt",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--constrained_pt", required=True)
    parser.add_argument("--sid", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out_json", required=True)
    args = parser.parse_args()

    result = evaluate(args.constrained_pt, args.sid, args.ckpt)
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[task87-tiger-eval] saved → {out_path}")


if __name__ == "__main__":
    main()
