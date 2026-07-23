"""
task4_free_form_eval.py — Reuse GRID's predict_dataloader + add labels for SIDRetrievalEvaluator.

Loads a trained TIGER checkpoint, runs model.generate() on the testing dataset
with `should_check_prefix=False` (free-form beam search), and computes R@5,
R@10, NDCG@5, NDCG@10 via SIDRetrievalEvaluator — identical to what the
training loop uses.

Implementation notes:
* GRID's `predict_dataloader_config` is configured by `tiger_inference_flat.yaml`
  to point at `${paths.data_dir}/testing/` and uses `collate_fn_inference_for_sequence`
  (no labels). We patch the data_loading config in-memory *before* Hydra instantiates
  it: switch `collate_fn` to `collate_fn_train` and supply `labels` so the dataloader
  yields the model_step inputs + the ground-truth labels tensor.
* Lightning `DataModule.setup()` requires `self.trainer.world_size`. We don't want
  a real `Trainer` (would re-run training loop), so we attach a stub before
  calling `dm.setup(stage="predict")`.
* GRID's Hydra resolvers MUST be imported with `from src.utils.custom_hydra_resolvers import *`
  so `compose()` can resolve everything.

Usage (run THREE times, one per algorithm):
    cd /fs04/ar57/wenyu/GeneRec/GRID
    python /home/wlia0047/ar57/wenyu/GeneRec/scripts/task4_free_form_eval.py \\
        --algo rqvae \\
        --sid  logs/inference/runs/2026-07-07/20-17-35/pickle/merged_predictions_tensor.pt \\
        --ckpt logs/train/runs/2026-07-07/20-20-20/checkpoints/best_tiger_step1800.ckpt \\
        --out_json /home/wlia0047/ar57/wenyu/GeneRec/task4_eval_rqvae.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

GRID_ROOT = Path("/fs04/ar57/wenyu/GeneRec/GRID")
sys.path.insert(0, str(GRID_ROOT))
os.chdir(str(GRID_ROOT))

# IMPORTANT: GRID's custom OmegaConf resolvers
from src.utils.custom_hydra_resolvers import (  # noqa: F401
    remove_chars_from_string,
    conditional_expression,
    extract_fields_from_list_of_dicts,
    create_map_from_list_of_dicts,
    math_eval,
    remove_item_from_list,
)

import numpy as np
import torch

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.components.eval_metrics import NDCG, Recall, SIDRetrievalEvaluator
from src.data.loading.datamodules.sequence_datamodule import SequenceDataModule
from src.models.modules.semantic_id.tiger_generation_model import (
    SemanticIDEncoderDecoder,
)

TOYS_DATA_DIR = "/fs04/ar57/wenyu/GeneRec/GRID/data/amazon_data/toys"
TOP_K_LIST = [5, 10]
SEQUENCE_LENGTH = 120


def _build_patched_dl(sid_path: str, ckpt_path: str, num_hierarchies: int = 4):
    """Compose a config with eval-mode dataloader (labels + collate_fn_train)."""
    GlobalHydra.instance().clear()
    initialize_config_dir(
        config_dir=str(GRID_ROOT / "configs"),
        job_name="task4_eval",
        version_base=None,
    )
    overrides = [
        "experiment=tiger_inference_flat",
        f"data_dir={TOYS_DATA_DIR}",
        f"semantic_id_path={sid_path}",
        f"ckpt_path={ckpt_path}",
        f"num_hierarchies={num_hierarchies}",
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

    # Patch the predict_dataloader_config to inject labels & collate_fn_train.
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
    # Hydra's strict dataclass validation rejects file_format=None even though the
    # field is unused. Inject the suffix from TFRecordIterator.get_file_suffix().
    pdc.dataset_config.file_format = "tfrecord.gz"
    pdc.batch_size_per_device = 8
    pdc.num_workers = 4
    return cfg


def _build_model(cfg, codebooks: torch.Tensor) -> SemanticIDEncoderDecoder:
    # HydRa-resolve cfg.model down to a dict; instantiate the submodules
    # (huggingface_model and decoder) that don't depend on data_loading, then
    # wire them into SemanticIDEncoderDecoder manually so we can swap in the
    # already-loaded codebook tensor.
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    model_cfg.pop("_target_", None)
    model_cfg.pop("huggingface_model", None)
    model_cfg.pop("decoder", None)
    model_cfg.pop("codebooks", None)
    model_cfg.pop("evaluator", None)  # rebuilt by caller

    # Instantiate sub-modules via Hydra so that nested _target_ chains resolve.
    OmegaConf.set_struct(cfg.model.huggingface_model, False)
    hf_model = instantiate(cfg.model.huggingface_model)
    decoder = instantiate(cfg.model.decoder)
    # SIDRetrievalEvaluator takes metric CLASSES (not instances) and instantiates
    # them internally with top_k; build it directly in Python.
    evaluator = SIDRetrievalEvaluator(
        metrics={"Recall": Recall, "NDCG": NDCG},
        top_k_list=TOP_K_LIST,
    )

    model_cfg["evaluator"] = evaluator
    model = SemanticIDEncoderDecoder(
        huggingface_model=hf_model,
        decoder=decoder,
        codebooks=codebooks,
        **model_cfg,
    )

    raw = torch.load(cfg.ckpt_path, map_location="cpu", weights_only=False)
    state = raw.get("state_dict", raw)
    cleaned = {}
    for k, v in state.items():
        if k.startswith("model."):
            cleaned[k[len("model."):]] = v
        else:
            cleaned[k] = v
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"[warn] missing keys: {len(missing)} (first 3): {missing[:3]}")
    if unexpected:
        print(f"[warn] unexpected keys: {len(unexpected)} (first 3): {unexpected[:3]}")
    return model


@torch.no_grad()
def evaluate_algo(algo: str, sid_path: str, ckpt_path: str) -> dict:
    # If num_hierarchies was passed via CLI, override the hardcoded value
    import builtins
    num_hier = getattr(builtins, "_TASK6_NUM_HIERARCHIES", 4)
    cfg = _build_patched_dl(sid_path=sid_path, ckpt_path=ckpt_path, num_hierarchies=num_hier)

    # Instantiate ONLY the inner dataloader config; the rest of data_loading
    # has unresolvable `???` placeholders that aren't relevant to evaluation.
    # NOTE: yaml nests `predict_dataloader_config.dataloader` — the
    # SequenceDataloaderConfig dataclass — under `predict_dataloader_config`.
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

    # Pull the actual codebook tensor out of the instantiated dataset_config
    # (SemanticIDDatasetConfig.semantic_id_map["sequence_data"] is a torch.Tensor
    # after Hydra's torch.load resolver runs).
    codebooks = pdc.dataset_config.semantic_id_map["sequence_data"]
    print(f"[info] codebooks shape: {tuple(codebooks.shape)} dtype: {codebooks.dtype}")

    model = _build_model(cfg, codebooks=codebooks)
    model.eval()

    eval_metrics = model.evaluator
    eval_metrics.reset()

    predict_loader = dm.predict_dataloader()
    # get_dataloader() returns a 1-tuple of DataLoader; unpack it.
    if isinstance(predict_loader, tuple):
        predict_loader = predict_loader[0]
    n_users = 0
    for batch_idx, batch in enumerate(predict_loader):
        # collate_fn_train returns (model_input_data, model_label_data)
        model_input, label_data = batch
        try:
            generated_ids, marginal_probs = model.generate(
                attention_mask=model_input.mask,
                **{
                    model.feature_to_model_input_map.get(k, k): v
                    for k, v in model_input.transformed_sequences.items()
                },
            )
        except Exception as e:
            print(f"  batch {batch_idx} generate failed: {e.__class__.__name__}: {e}")
            continue

        labels_t = list(label_data.labels.values())[0].long()
        eval_metrics(
            marginal_probs=marginal_probs.detach(),
            generated_ids=generated_ids.detach(),
            labels=labels_t,
        )
        n_users += int(model_input.mask.size(0))
        if batch_idx % 10 == 0:
            print(f"  batch {batch_idx} users={n_users}")

    metrics_dict = {
        name: float(metric.compute().detach().cpu())
        for name, metric in eval_metrics.metrics.items()
    }
    return {
        "algo": algo,
        "ckpt": str(ckpt_path),
        "sid": str(sid_path),
        "n_users": n_users,
        **metrics_dict,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", required=True, choices=["rqvae", "rkmeans", "rvq"])
    parser.add_argument("--sid", required=True, type=Path)
    parser.add_argument("--ckpt", required=True, type=Path)
    parser.add_argument("--num_hierarchies", type=int, default=4,
                        help="Number of SID hierarchies. Default 4 (with-dedup); "
                             "use 3 for no-dedup RQ-VAE (task6).")
    parser.add_argument("--out_json", type=Path, default=None)
    args = parser.parse_args()

    # Override the hardcoded num_hierarchies in the overrides list
    import builtins
    builtins._TASK6_NUM_HIERARCHIES = args.num_hierarchies
    result = evaluate_algo(args.algo, str(args.sid), str(args.ckpt))
    print("\n=== task4 free-form eval result ===")
    print(json.dumps(result, indent=2))

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
