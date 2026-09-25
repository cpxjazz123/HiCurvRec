"""Export the *prepared* RecBole3.0 TIGER records for HG-Rec.

Use this when strict cross-run parity is required.  It asks RecBole to parse
and remap the raw dataset, perform its configured leave-one-out split, build
the sequential histories, and then serializes those exact records to the HG
parquet schema.  This avoids reimplementing the raw Amazon parser in HG-Rec.

Example (run from this directory):

    PYTHONPATH=../RecBole3.0/src python export_recbole_tiger_data.py
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent


def _frame_with_histories(frame, histories, seen_histories=None):
    result = pd.DataFrame(
        {
            "user": [int(value) for value in frame["user_id"].tolist()],
            "history": [list(history) for history in histories],
            "target": [int(value) for value in frame["item_id"].tolist()],
        }
    )
    if seen_histories is not None:
        # Model input is capped at history_max_length, whereas RecBole full
        # evaluation excludes the complete seen_item_ids sequence.
        result["seen_history"] = [list(history) for history in seen_histories]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--recbole_root",
        type=Path,
        default=ROOT.parent / "RecBole3.0",
        help="RecBole3.0 checkout containing src/ and its data cache.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=ROOT / "dataset/Amazon_2023_Instruments",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Extra Hydra override, e.g. dataset.category=Musical_Instruments.",
    )
    args = parser.parse_args()

    recbole_root = args.recbole_root.resolve()
    sys.path.insert(0, str(recbole_root / "src"))
    os.chdir(recbole_root)

    from recbole3.config import instantiate_dataclass
    from recbole3.dataset import get_dataset_spec
    from recbole3.dataset.config import SplitConfig
    from recbole3.evaluation import EvalConfig
    from recbole3.model.sequential import build_history_item_ids
    from recbole3.run import compose_config

    overrides = ["dataset=amazon2023_retrieval", "model=tiger", *args.override]
    config = compose_config(overrides=overrides)
    dataset_spec = get_dataset_spec(config.dataset.name)
    dataset_config = instantiate_dataclass(dataset_spec.config_cls, config.dataset)
    eval_config = instantiate_dataclass(EvalConfig, config.trainer.eval)
    if not isinstance(dataset_config.split, SplitConfig):
        raise TypeError("RecBole dataset split config was not instantiated")

    prepared = dataset_spec.dataset_cls(dataset_config).prepare(eval_config=eval_config)
    history_max_length = int(config.model.history_max_length)

    train_frame = prepared.get_train_dataset().frame
    valid_frame = prepared.get_eval_dataset("valid").frame
    test_frame = prepared.get_eval_dataset("test").frame
    train_histories, history_state = build_history_item_ids(
        train_frame, history_max_length=history_max_length
    )
    valid_histories, history_state = build_history_item_ids(
        valid_frame,
        initial_histories=history_state,
        history_max_length=history_max_length,
    )
    test_histories, _ = build_history_item_ids(
        test_frame,
        initial_histories=history_state,
        history_max_length=history_max_length,
    )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "train_recbole.parquet": _frame_with_histories(train_frame, train_histories),
        "valid_recbole.parquet": _frame_with_histories(
            valid_frame, valid_histories, valid_frame["seen_item_ids"].tolist()
        ),
        "test_recbole.parquet": _frame_with_histories(
            test_frame, test_histories, test_frame["seen_item_ids"].tolist()
        ),
    }
    for filename, frame in outputs.items():
        frame.to_parquet(output_dir / filename, index=False)

    print(
        f"Exported RecBole prepared data: users={prepared.get_num_users()} "
        f"items={prepared.get_num_items()} "
        + ", ".join(f"{name}={len(frame)}" for name, frame in outputs.items())
    )
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
