#!/usr/bin/env python
"""Task #23 / #24: PM-RQ × TIGER Stage 4 Recall/NDCG evaluation.

Mirrors scripts/task14_s4_item_eval_l5.py pattern (the fully-implemented eval).
Computes item-level Recall@5, Recall@10, NDCG@5, NDCG@10 on Toys.

Usage:
    python scripts/task23_105_pmrq_eval.py \
        --task_id 104 \
        --constrained_pt logs/task23_pmrq2_s4/runs/<id>/pickle/merged_predictions_tensor.pt \
        --sid products/task22_pm_rq/sid_phase2.pt \
        --num_hierarchies 4 \
        --sequence_length 120 \
        --out_json verdicts/task23_pmrq2_tiger_eval.json
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

# Register custom OmegaConf resolvers (extract_fields_from_list_of_dicts, etc.)
from src.utils.custom_hydra_resolvers import (  # noqa: F401
    remove_chars_from_string,
    conditional_expression,
    extract_fields_from_list_of_dicts,
    create_map_from_list_of_dicts,
    math_eval,
    remove_item_from_list,
)

from src.components.eval_metrics import NDCG, Recall, SIDRetrievalEvaluator
from src.data.loading.datamodules.sequence_datamodule import SequenceDataModule

TOYS_DATA_DIR = str(REPO_ROOT / "data" / "amazon_data" / "toys")
TOP_K_LIST = [5, 10]


def _build_patched_dl(sid_path: str, ckpt_path: str, num_hierarchies: int, sequence_length: int) -> "OmegaConf":
    GlobalHydra.instance().clear()
    initialize_config_dir(
        config_dir=str(REPO_ROOT / "configs"),
        job_name=f"task{args.task_id}_eval",
        version_base=None,
    )
    overrides = [
        "experiment=tiger_inference_tiger",
        f"data_dir={TOYS_DATA_DIR}",
        f"semantic_id_path={sid_path}",
        f"ckpt_path={ckpt_path}",
        f"num_hierarchies={num_hierarchies}",
        f"sequence_length={sequence_length}",
        "trainer.accelerator=cpu",
        "trainer.devices=1",
        f"paths.root_dir={REPO_ROOT}",
    ]
    cfg = compose(config_name="train", overrides=overrides)

    OmegaConf.set_struct(cfg.data_loading, False)
    OmegaConf.set_struct(cfg.data_loading.predict_dataloader_config.dataloader.dataset_config, False)
    pdc = cfg.data_loading.predict_dataloader_config.dataloader

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
    task_id: int,
    constrained_pt: str,
    sid_path: str,
    ckpt_path: str,
    num_hierarchies: int,
    sequence_length: int,
) -> dict:
    pred = torch.load(constrained_pt, map_location="cpu", weights_only=False).long()
    print(f"[info] constrained tensor shape: {tuple(pred.shape)} dtype={pred.dtype}")
    if pred.dim() != 3 or pred.shape[2] != num_hierarchies:
        raise ValueError(
            f"bad shape {tuple(pred.shape)}, expected (n_users, top_k, {num_hierarchies})"
        )
    n_total_users, top_k, num_hier = pred.shape
    print(f"[info] {n_total_users} users x {top_k} candidates x {num_hier} digits")

    sid_catalog = torch.load(sid_path, map_location="cpu", weights_only=False).long()
    print(f"[info] catalog SID tensor shape: {tuple(sid_catalog.shape)}")
    if sid_catalog.shape[0] != num_hierarchies:
        raise ValueError(
            f"SID first-dim {sid_catalog.shape[0]} != num_hierarchies {num_hierarchies}"
        )

    cfg = _build_patched_dl(sid_path=sid_path, ckpt_path=ckpt_path,
                            num_hierarchies=num_hierarchies,
                            sequence_length=sequence_length)
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
        labels_t = list(label_data.labels.values())[0].long()
        bsz = int(model_input.mask.size(0))
        labels_2d = labels_t.reshape(bsz, num_hier)

        batch_user_ids = model_input.transformed_sequences["user_id"]

        per_row_preds = []
        per_row_indices = []
        for i in range(bsz):
            row_mask = model_input.mask[i]
            nonzero = row_mask.nonzero(as_tuple=True)[0]
            if len(nonzero) == 0:
                continue
            uid = int(batch_user_ids[i, nonzero[0]].item())
            if 0 <= uid < n_total_users:
                per_row_preds.append(pred[uid])
                per_row_indices.append(i)
            else:
                n_missing += 1

        if not per_row_preds:
            continue

        batch_pred = torch.stack(per_row_preds, dim=0)
        kept_labels_t = labels_2d[per_row_indices]
        batch_scores = score_template.unsqueeze(0).expand(batch_pred.shape[0], -1).contiguous()
        n_users += batch_pred.shape[0]

        evaluator(
            marginal_probs=batch_scores,
            generated_ids=batch_pred,
            labels=kept_labels_t,
        )
        if batch_idx % 50 == 0:
            print(f"  batch {batch_idx} users={n_users} missing={n_missing}")

    metrics_dict = {
        name: float(metric.compute().detach().cpu())
        for name, metric in evaluator.metrics.items()
    }
    return {
        "task_id": task_id,
        "constrained_pt": str(constrained_pt),
        "sid": str(sid_path),
        "ckpt": str(ckpt_path),
        "num_hierarchies": num_hierarchies,
        "sequence_length": sequence_length,
        "n_users_evaluated": n_users,
        "n_users_tensor": n_total_users,
        "n_users_missing": n_missing,
        **metrics_dict,
    }


def main():
    global args
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_id", type=int, required=True)
    parser.add_argument("--constrained_pt", required=True, type=Path)
    parser.add_argument("--sid", required=True, type=Path)
    parser.add_argument("--ckpt", required=True, type=Path)
    parser.add_argument("--num_hierarchies", type=int, required=True)
    parser.add_argument("--sequence_length", type=int, required=True)
    parser.add_argument("--out_json", type=Path, required=True)
    args = parser.parse_args()

    result = evaluate(
        task_id=args.task_id,
        constrained_pt=str(args.constrained_pt),
        sid_path=str(args.sid),
        ckpt_path=str(args.ckpt),
        num_hierarchies=args.num_hierarchies,
        sequence_length=args.sequence_length,
    )
    print(f"\n=== Task #{args.task_id} PM-RQ × TIGER eval result ===")
    print(json.dumps(result, indent=2))
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[task{args.task_id}-pmrq-eval] saved → {out_path}")


if __name__ == "__main__":
    main()
