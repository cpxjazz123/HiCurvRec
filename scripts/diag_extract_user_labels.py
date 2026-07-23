#!/usr/bin/env python3
"""Extract per-user ground-truth labels (true next-K SID tuples) from the test dataloader.

This is a one-time, model-free extraction. Saves a (n_users, num_hier) int64 tensor
that can be reused for any subsequent collision-aware R@10 computation.

Output: result/collision_aware/user_labels.pt — torch.Tensor (19412, 4)
        result/collision_aware/user_labels_meta.json — num_users, num_hier, etc.
"""
import os
import sys
import json
import torch
from pathlib import Path

GRID_ROOT = Path("/home/wlia0047/ar57/wenyu/GeneRec/GRID")
TOYS_DATA_DIR = str(GRID_ROOT / "data/amazon_data/toys")
NUM_HIER = 4
SEQUENCE_LENGTH = 120
OUT_DIR = GRID_ROOT / "result" / "collision_aware"


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


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sid", required=True, help="Algorithm-specific SID tensor path (used as semantic_id_path to map item_ids → SIDs)")
    parser.add_argument("--out_name", default=None, help="Output suffix (e.g. 'A_baseline'). Default: derived from sid path.")
    args_parser = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    _bootstrap()

    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    GlobalHydra.instance().clear()
    initialize_config_dir(
        config_dir=str(GRID_ROOT / "configs"),
        job_name="extract_user_labels",
        version_base=None,
    )
    sid_path = args_parser.sid
    out_name = args_parser.out_name or Path(sid_path).stem.replace("merged_predictions_tensor", "")
    print(f"  sid_path: {sid_path}")
    print(f"  out_name: {out_name}")
    overrides = [
        "experiment=tiger_inference_flat",
        f"data_dir={TOYS_DATA_DIR}",
        f"semantic_id_path={sid_path}",
        "ckpt_path=/tmp/dummy_for_label_extraction.ckpt",
        f"num_hierarchies={NUM_HIER}",
        f"sequence_length={SEQUENCE_LENGTH}",
        "+model.should_check_prefix=false",
        "+model.evaluator.top_k_list=[5,10]",
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

    print(f"=== Extracting user labels (model-free) ===")
    print(f"  data_dir: {TOYS_DATA_DIR}")
    print(f"  num_hier: {NUM_HIER}")

    pdc_inst = instantiate(cfg.data_loading.predict_dataloader_config.dataloader)
    from src.data.loading.datamodules.sequence_datamodule import SequenceDataModule
    datamodule = SequenceDataModule(
        train_dataloader_config=None,
        val_dataloader_config=None,
        test_dataloader_config=None,
        predict_dataloader_config=pdc_inst,
    )

    class _StubTrainer:
        world_size = 1
        is_global_zero = True
        global_rank = 0
        local_rank = 0
    datamodule.trainer = _StubTrainer()  # type: ignore[attr-defined]
    datamodule.prepare_data()
    datamodule.setup("predict")

    predict_loader_tuple = datamodule.predict_dataloader()
    # Returns a tuple of (DataLoader,)
    if isinstance(predict_loader_tuple, tuple):
        predict_loader = predict_loader_tuple[0]
    else:
        predict_loader = predict_loader_tuple
    print(f"  predict_loader type={type(predict_loader).__name__}, batch_size={predict_loader.batch_size}")

    # Pass 1: find max user_id
    n_users_max = 0
    for batch in predict_loader:
        model_input, _ = batch
        uids = model_input.transformed_sequences["user_id"]
        if uids.numel() > 0:
            n_users_max = max(n_users_max, int(uids.max().item()) + 1)
    print(f"  max user_id: {n_users_max}")

    # Pass 2: extract labels
    predict_loader_tuple = datamodule.predict_dataloader()
    if isinstance(predict_loader_tuple, tuple):
        predict_loader = predict_loader_tuple[0]
    else:
        predict_loader = predict_loader_tuple
    user_labels = torch.full((n_users_max, NUM_HIER), -1, dtype=torch.long)
    n_labeled = 0
    n_skipped = 0

    for batch_idx, batch in enumerate(predict_loader):
        model_input, label_data = batch
        labels_t = list(label_data.labels.values())[0].long()
        bsz = int(model_input.mask.size(0))
        labels_2d = labels_t.reshape(bsz, NUM_HIER)
        batch_user_ids = model_input.transformed_sequences["user_id"]

        for i in range(bsz):
            row_mask = model_input.mask[i]
            nonzero = row_mask.nonzero(as_tuple=True)[0]
            if len(nonzero) == 0:
                n_skipped += 1
                continue
            uid = int(batch_user_ids[i, nonzero[0]].item())
            if 0 <= uid < n_users_max:
                user_labels[uid] = labels_2d[i]
                n_labeled += 1
            else:
                n_skipped += 1

        if batch_idx % 50 == 0:
            print(f"  batch {batch_idx}: labeled={n_labeled} skipped={n_skipped}", flush=True)

    print(f"\n  Total: labeled={n_labeled}, skipped={n_skipped}")
    print(f"  user_labels shape: {tuple(user_labels.shape)}, dtype={user_labels.dtype}")

    out_pt = OUT_DIR / f"user_labels_{out_name}.pt"
    torch.save(user_labels, out_pt)
    print(f"  Saved → {out_pt}")

    meta = {
        "n_users_max": n_users_max,
        "n_labeled": int(n_labeled),
        "n_skipped": int(n_skipped),
        "num_hier": NUM_HIER,
        "data_dir": TOYS_DATA_DIR,
        "sequence_length": SEQUENCE_LENGTH,
        "sid_path": sid_path,
        "out_name": out_name,
    }
    out_json = OUT_DIR / f"user_labels_{out_name}_meta.json"
    with open(out_json, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Saved → {out_json}")


if __name__ == "__main__":
    main()