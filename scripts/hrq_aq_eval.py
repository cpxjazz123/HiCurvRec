#!/usr/bin/env python3
# hrq_aq_eval.py — Standalone SIDRetrievalEvaluator for HRQ/AQ
# Usage: python3 hrq_aq_eval.py --algo HRQ --sid <sid_path> --stage4 <s4_path> --ckpt <ckpt> --out <json>

import argparse, json, os, sys
from pathlib import Path

import torch

GRID_ROOT = Path("/fs04/ar57/wenyu/GeneRec/GRID")
TOYS_DATA_DIR = str(GRID_ROOT / "data/amazon_data/toys")
TOP_K_LIST = [5, 10]
SEQUENCE_LENGTH = 120


def _bootstrap():
    os.chdir(str(GRID_ROOT))
    sys.path.insert(0, str(GRID_ROOT))
    from src.utils.custom_hydra_resolvers import (  # noqa: F401
        remove_chars_from_string,
        conditional_expression,
        extract_fields_from_list_of_dicts,
        create_map_from_list_of_dicts,
        math_eval,
        remove_item_from_list,
    )


def compute_collision_rate(sid_path: str) -> float:
    """Compute SID collision rate from raw (4, N) SID tensor."""
    sid = torch.load(sid_path, weights_only=False, map_location="cpu")
    if sid.dim() == 2 and sid.shape[0] < sid.shape[1]:
        sid_t = sid.t().long()
    else:
        sid_t = sid.long()
    n_total = sid_t.shape[0]
    n_unique = torch.unique(sid_t, dim=0).shape[0]
    return 1.0 - n_total / n_total if n_total == 0 else (1.0 - n_unique / n_total)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", required=True)
    parser.add_argument("--sid", required=True, type=Path)
    parser.add_argument("--stage4", required=True, type=Path)
    parser.add_argument("--ckpt", required=True, type=Path)
    parser.add_argument("--num_hier", type=int, default=4)
    parser.add_argument("--out_json", required=True, type=Path)
    args = parser.parse_args()

    print(f"=== {args.algo} eval ===")
    print(f"  sid:    {args.sid}")
    print(f"  stage4: {args.stage4}")
    print(f"  ckpt:   {args.ckpt}")

    # Collision rate
    coll = compute_collision_rate(str(args.sid))
    print(f"  collision rate: {coll:.4f}")

    # Bootstrap (Hydra + resolvers)
    _bootstrap()

    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf

    GlobalHydra.instance().clear()
    initialize_config_dir(
        config_dir=str(GRID_ROOT / "configs"),
        job_name=f"{args.algo}_eval",
        version_base=None,
    )
    overrides = [
        "experiment=tiger_inference_flat",
        f"data_dir={TOYS_DATA_DIR}",
        f"semantic_id_path={args.sid}",
        f"ckpt_path={args.ckpt}",
        f"num_hierarchies={args.num_hier}",
        f"sequence_length={SEQUENCE_LENGTH}",
        "+model.should_check_prefix=false",
        f"+model.evaluator.top_k_list=[{TOP_K_LIST[0]},{TOP_K_LIST[1]}]",
        "+model.evaluator._target_=src.components.eval_metrics.SIDRetrievalEvaluator",
        "+model.evaluator.metrics.Recall._target_=src.components.eval_metrics.Recall",
        "+model.evaluator.metrics.NDCG._target_=src.components.eval_metrics.NDCG",
        "+model.evaluator.metrics." + "_partial_" + "=true",
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
    pdc.should_shuffle_rows = False
    pdc.drop_last = False

    from hydra.utils import instantiate
    from src.data.loading.datamodules.sequence_datamodule import SequenceDataModule
    pdc = instantiate(cfg.data_loading.predict_dataloader_config.dataloader)
    datamodule = SequenceDataModule(
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

    datamodule.trainer = _StubTrainer()  # type: ignore[attr-defined]
    datamodule.prepare_data()
    datamodule.setup(stage="predict")
    from src.components.eval_metrics import NDCG, Recall, SIDRetrievalEvaluator
    eval_metrics = SIDRetrievalEvaluator(
        metrics={"Recall": Recall, "NDCG": NDCG},
        top_k_list=TOP_K_LIST,
    )

    stage4 = torch.load(str(args.stage4), weights_only=False, map_location="cpu")
    print(f"  stage4 tensor shape: {stage4.shape}")
    sid = torch.load(str(args.sid), weights_only=False, map_location="cpu")

    pred = stage4.long()
    n_total_users, top_k, num_hier = pred.shape

    eval_metrics.reset()
    score_template = torch.arange(top_k, 0, -1, dtype=torch.float32) / top_k

    predict_loader = datamodule.predict_dataloader()
    if isinstance(predict_loader, tuple):
        predict_loader = predict_loader[0]

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

        eval_metrics(
            marginal_probs=batch_scores,
            generated_ids=batch_pred,
            labels=kept_labels_t,
        )
        if batch_idx % 50 == 0:
            print(f"  batch {batch_idx} users={n_users} missing={n_missing}", flush=True)

    results = {name: float(metric.compute().detach().cpu()) for name, metric in eval_metrics.metrics.items()}
    print(f"  eval results: {results}")
    print(f"  n_users_evaluated={n_users} n_users_missing={n_missing}")

    # Save
    out_info = {
        "algo": args.algo,
        "sid_path": str(args.sid),
        "stage4_path": str(args.stage4),
        "ckpt_path": str(args.ckpt),
        "num_hierarchies": args.num_hier,
        "collision_rate": coll,
        "metrics": {k: float(v) for k, v in results.items()},
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out_info, f, indent=2)
    print(f"  saved → {args.out_json}")


if __name__ == "__main__":
    main()