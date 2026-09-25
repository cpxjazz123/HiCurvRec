"""Export Amazon 2023 Instruments splits in RecBole's TIGER format.

The source interaction JSON is already ordered by user.  RecBole's retrieval
split uses the last two interactions as validation and test targets.  The
training frame contains every prefix/target pair in the remaining sequence,
including the first pair with an empty history.  Validation uses the complete
training sequence; test uses training plus the validation target.

The generated files intentionally have a ``*_recbole.parquet`` suffix so the
existing HG-Rec parquet files remain untouched.
"""

import json
from pathlib import Path

import pandas as pd


DATASET_DIR = Path(__file__).resolve().parents[1] / "dataset" / "Amazon_2023_Instruments"
INTER_JSON = DATASET_DIR / "Instruments.inter.json"


def _rows_for_prefixes(interactions):
    rows = []
    for user_id, sequence in interactions.items():
        sequence = [int(item_id) for item_id in sequence]
        for position, target in enumerate(sequence):
            rows.append(
                {
                    "user": int(user_id),
                    "history": sequence[:position],
                    "seen_history": sequence[:position],
                    "target": int(target),
                }
            )
    return rows


def build_splits(payload):
    user_ids = sorted(int(user_id) for user_id in payload)
    if user_ids != list(range(len(user_ids))):
        raise ValueError(
            "The interaction JSON must already use RecBole's zero-based user ids; "
            f"got first/last={user_ids[:2]}...{user_ids[-2:]}"
        )
    train_rows = []
    valid_rows = []
    test_rows = []
    for user_id, raw_sequence in payload.items():
        sequence = [int(item_id) for item_id in raw_sequence]
        if len(sequence) < 3:
            continue
        train_sequence = sequence[:-2]
        valid_target = sequence[-2]
        test_target = sequence[-1]
        train_rows.extend(_rows_for_prefixes({int(user_id): train_sequence}))
        valid_rows.append(
            {
                "user": int(user_id),
                "history": train_sequence,
                "seen_history": train_sequence,
                "target": int(valid_target),
            }
        )
        test_rows.append(
            {
                "user": int(user_id),
                "history": train_sequence + [valid_target],
                "seen_history": train_sequence + [valid_target],
                "target": int(test_target),
            }
        )
    return (
        pd.DataFrame(train_rows, columns=["user", "history", "seen_history", "target"]),
        pd.DataFrame(valid_rows, columns=["user", "history", "seen_history", "target"]),
        pd.DataFrame(test_rows, columns=["user", "history", "seen_history", "target"]),
    )


def main():
    with INTER_JSON.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    train_df, valid_df, test_df = build_splits(payload)
    train_path = DATASET_DIR / "train_recbole.parquet"
    valid_path = DATASET_DIR / "valid_recbole.parquet"
    test_path = DATASET_DIR / "test_recbole.parquet"
    train_df.to_parquet(train_path, index=False)
    valid_df.to_parquet(valid_path, index=False)
    test_df.to_parquet(test_path, index=False)
    print(
        f"[process_Amazon2023] wrote train={train_df.shape}, "
        f"valid={valid_df.shape}, test={test_df.shape}"
    )
    print(f"  {train_path}")
    print(f"  {valid_path}")
    print(f"  {test_path}")


if __name__ == "__main__":
    main()
