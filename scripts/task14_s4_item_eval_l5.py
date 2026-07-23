"""Task 388 v4: item-level R@5/R@10/NDCG@5/NDCG@10 for Stage 4 outputs.

Usage:
    python task388v4_s4_item_eval.py \\
        --variant H_E_E_E \\
        --constrained_pt /fs04/ar57/wenyu/GeneRec/GRID/logs/inference/runs/task388v4_h_e_e_e_s4/pickle/merged_predictions_tensor.pt \\
        --sid /fs04/ar57/wenyu/GeneRec/GRID/logs/inference/runs/task388v4_hee_s22/pickle/merged_predictions_tensor.pt \\
        --ckpt /home/wlia0047/ar57/wenyu/GeneRec/GRID/task_artifacts/results/exp388v4/task388v4_h_e_e_e_tiger.ckpt \\
        --out_json /home/wlia0047/ar57/wenyu/GeneRec/GRID/task_artifacts/results/exp388v4/task388v4_h_e_e_e_eval.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

GRID_ROOT = Path("/fs04/ar57/wenyu/GeneRec/GRID")
sys.path.insert(0, str(GRID_ROOT))
os.chdir(str(GRID_ROOT))

from src.utils.custom_hydra_resolvers import (  # noqa: F401
    remove_chars_from_string,
    conditional_expression,
    extract_fields_from_list_of_dicts,
    create_map_from_list_of_dicts,
    math_eval,
    remove_item_from_list,
)

import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.components.eval_metrics import NDCG, Recall, SIDRetrievalEvaluator
from src.data.loading.datamodules.sequence_datamodule import SequenceDataModule

TOYS_DATA_DIR = "/fs04/ar57/wenyu/GeneRec/GRID/data/amazon_data/toys"
TOP_K_LIST = [5, 10]
SEQUENCE_LENGTH = 120


def _build_patched_dl(sid_path: str, ckpt_path: str):
    GlobalHydra.instance().clear()
    initialize_config_dir(
        config_dir=str(GRID_ROOT / "configs"),
        job_name="v4_eval",
        version_base=None,
    )
    overrides = [
        "experiment=tiger_inference_flat",
        f"data_dir={TOYS_DATA_DIR}",
        f"semantic_id_path={sid_path}",
        f"ckpt_path={ckpt_path}",
        "num_hierarchies=5",
        f"sequence_length={SEQUENCE_LENGTH}",
        "trainer.accelerator=cpu",
        "trainer.devices=1",
        "paths.root_dir=/fs04/ar57/wenyu/GeneRec/GRID",
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
def evaluate(variant: str, constrained_pt: str, sid_path: str, ckpt_path: str) -> dict:
    pred = torch.load(constrained_pt, map_location="cpu", weights_only=False).long()
    print(f"[info] constrained tensor shape: {tuple(pred.shape)} dtype={pred.dtype}")
    assert pred.dim() == 3 and pred.shape[2] == 5, f"bad shape {pred.shape}"  # Task #68: num_hierarchies=5
    n_total_users, top_k, num_hier = pred.shape
    print(f"[info] {n_total_users} users x {top_k} candidates x {num_hier} digits")

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
        "variant": variant,
        "constrained_pt": str(constrained_pt),
        "sid": str(sid_path),
        "ckpt": str(ckpt_path),
        "n_users_evaluated": n_users,
        "n_users_tensor": n_total_users,
        "n_users_missing": n_missing,
        **metrics_dict,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True)
    parser.add_argument("--constrained_pt", required=True, type=Path)
    parser.add_argument("--sid", required=True, type=Path)
    parser.add_argument("--ckpt", required=True, type=Path)
    parser.add_argument("--out_json", type=Path, default=None)
    args = parser.parse_args()

    result = evaluate(
        variant=args.variant,
        constrained_pt=str(args.constrained_pt),
        sid_path=str(args.sid),
        ckpt_path=str(args.ckpt),
    )
    print("\n=== task388v4 item-level eval result ===")
    print(json.dumps(result, indent=2))

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()