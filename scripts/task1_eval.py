"""task3_eval.py — 用 GRID 框架 + SIDRetrievalEvaluator 评估 task15 三组的 Stage 4 tensor。

输入:
    python3 task3_eval.py <group> <sid_path> <stage4_tensor_path> <output_json>

设计:
    不再读 interactions.csv（Toys 数据是 tfrecord 分片）。改用 SequenceDataModule
    （predict mode + labels） + SIDRetrievalEvaluator（Recall/NDCG），与 task8 一致。

    直接 user_id 索引（pred[uid] = 该用户的 top-10 预测）；如果 uid 不在 [0, N) 跳过。
    SID 碰撞率单独计算（独立于评估器）。

示例:
    python3 task3_eval.py A \
        /home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/inference/runs/task13_group_a_s22/pickle/merged_predictions_tensor.pt \
        /home/wlia0047/ar57/wenyu/GeneRec/GRID/logs/inference/runs/task13_group_a_s4/pickle/merged_predictions_tensor.pt \
        /home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task5/task3_eval_A.json
"""
import argparse
import json
import os
import sys
from pathlib import Path

import torch  # noqa: E402  — 顶部 import 必须早于 @torch.no_grad()

GRID_ROOT = Path("/fs04/ar57/wenyu/GeneRec/GRID")
TOYS_DATA_DIR = str(GRID_ROOT / "data/amazon_data/toys")
TOP_K_LIST = [5, 10]
SEQUENCE_LENGTH = 120


def _bootstrap():
    """注入 conda 路径、sys.path、setcwd、Omegaconf resolvers。"""
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


def _build_patched_dl(sid_path: str, ckpt_path: str, num_hier: int):
    """复用 task8 的 dataloader 配置：predict mode + labels + collate_fn_train。"""
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf

    GlobalHydra.instance().clear()
    initialize_config_dir(
        config_dir=str(GRID_ROOT / "configs"),
        job_name="task13_eval",
        version_base=None,
    )
    overrides = [
        "experiment=tiger_inference_flat",
        f"data_dir={TOYS_DATA_DIR}",
        f"semantic_id_path={sid_path}",
        f"ckpt_path={ckpt_path}",
        f"num_hierarchies={num_hier}",
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

    # Patch the predict_dataloader to inject labels & collate_fn_train
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
    pdc.num_workers = 4  # 与 task8 一致；单进程版 _timeout 必须 0，切换多进程
    return cfg


def compute_collision_rate(sid_tensor_path: str) -> dict:
    """测碰撞率：3-layer SID unique 占比 + 完整（4-layer，含 dedup digit）占比。"""
    sid = torch.load(sid_tensor_path, map_location="cpu", weights_only=False)
    if sid.shape[0] < sid.shape[1]:
        sid = sid.transpose(0, 1)
    N = sid.shape[0]
    # 3-layer effective SID (不含 dedup digit)
    sid_eff = sid[:, :3] if sid.shape[1] >= 3 else sid
    n_unique_eff = torch.unique(sid_eff, dim=0).shape[0]
    n_unique_full = torch.unique(sid, dim=0).shape[0]
    return {
        "N": N,
        "n_unique_3layer": n_unique_eff,
        "collision_rate_3layer": float(1 - n_unique_eff / N),
        "n_unique_full": n_unique_full,
        "collision_rate_full": float(1 - n_unique_full / N),
    }


@torch.no_grad()
def evaluate(group: str, sid_path: str, stage4_path: str, ckpt_path: str) -> dict:
    """核心：加载 Stage 4 tensor，GRID dataloader 对齐 ground-truth，evaluator 出指标。"""
    from hydra.utils import instantiate
    from src.components.eval_metrics import NDCG, Recall, SIDRetrievalEvaluator
    from src.data.loading.datamodules.sequence_datamodule import SequenceDataModule

    # 1. 加载 free-form Stage 4 tensor (shape (N_users, 10, H))
    pred = torch.load(stage4_path, map_location="cpu", weights_only=False).long()
    print(f"[info] stage4 tensor shape: {tuple(pred.shape)} dtype={pred.dtype}")
    assert pred.dim() == 3, f"bad shape {pred.shape}"
    n_total_users, top_k, num_hier = pred.shape
    print(f"[info] {n_total_users} users x {top_k} candidates x {num_hier} digits")

    # 2. Build dataloader to get ground-truth labels
    cfg = _build_patched_dl(sid_path=sid_path, ckpt_path=ckpt_path, num_hier=num_hier)
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

    dm.trainer = _StubTrainer()  # type: ignore[attr-defined]
    dm.prepare_data()
    dm.setup(stage="predict")

    # 3. Build evaluator
    evaluator = SIDRetrievalEvaluator(
        metrics={"Recall": Recall, "NDCG": NDCG},
        top_k_list=TOP_K_LIST,
    )
    evaluator.reset()

    # 4. Iterate dataloader, align pred with labels via direct user_id indexing
    predict_loader = dm.predict_dataloader()
    if isinstance(predict_loader, tuple):
        predict_loader = predict_loader[0]

    # Synthetic rank-based score: top-1 highest, top-10 lowest
    score_template = torch.arange(top_k, 0, -1, dtype=torch.float32) / top_k

    n_users = 0
    n_missing = 0
    for batch_idx, batch in enumerate(predict_loader):
        model_input, label_data = batch
        labels_t = list(label_data.labels.values())[0].long()  # flat (B*next_k,)
        bsz = int(model_input.mask.size(0))
        labels_2d = labels_t.reshape(bsz, num_hier)  # (B, 4)

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
        if batch_idx % 20 == 0:
            print(f"  batch {batch_idx} users={n_users} missing={n_missing}")

    metrics_dict = {
        name: float(metric.compute().detach().cpu())
        for name, metric in evaluator.metrics.items()
    }
    return {
        "group": group,
        "sid": str(sid_path),
        "stage4_tensor": str(stage4_path),
        "ckpt": str(ckpt_path),
        "n_users_evaluated": n_users,
        "n_users_tensor": n_total_users,
        "n_users_missing": n_missing,
        "n_hier": num_hier,
        **metrics_dict,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", required=True, choices=["A", "B", "C"])
    parser.add_argument("--sid", required=True, type=Path)
    parser.add_argument("--stage4", required=True, type=Path)
    parser.add_argument("--ckpt", required=True, type=Path, help="Stage 3 best_*.ckpt 路径")
    parser.add_argument("--out_json", required=True, type=Path)
    args = parser.parse_args()

    _bootstrap()

    print(f"=== Group {args.group} 评估 ===")
    coll = compute_collision_rate(str(args.sid))
    print(f"[collision] {coll}")
    metrics = evaluate(
        group=args.group,
        sid_path=str(args.sid),
        stage4_path=str(args.stage4),
        ckpt_path=str(args.ckpt),
    )
    print(f"[metrics] {metrics}")

    result = {
        "task": "task13_gain_shape_fusion_vs_mmq",
        **coll,
        **metrics,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[saved] {args.out_json}")


if __name__ == "__main__":
    main()
