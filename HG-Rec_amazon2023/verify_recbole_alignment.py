"""Check the HG-Rec artifacts against the vanilla RecBole TIGER contract.

This checker intentionally uses only numpy/pandas so it can be run before the
training environment (PyTorch/Transformers) is activated.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _list_value(value):
    return [] if value is None else [int(item) for item in list(value)]


def _load_sid(path):
    path = Path(path)
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("RecBole SID file must be a JSON object")
        keys = sorted(int(key) for key in payload)
        if keys != list(range(len(keys))):
            raise ValueError("SID keys are not zero-based and contiguous")
        return np.asarray([payload[str(item_id)] for item_id in keys], dtype=np.int64)
    return np.load(path, allow_pickle=False).astype(np.int64)


def _expected_splits(source_path):
    source = json.loads(Path(source_path).read_text(encoding="utf-8"))
    expected = {"train_recbole.parquet": [], "valid_recbole.parquet": [], "test_recbole.parquet": []}
    for raw_user_id, raw_sequence in source.items():
        user_id = int(raw_user_id)
        sequence = [int(item_id) for item_id in raw_sequence]
        train_sequence = sequence[:-2]
        for position, target in enumerate(train_sequence):
            expected["train_recbole.parquet"].append(
                {"user": user_id, "history": train_sequence[:position], "target": target}
            )
        expected["valid_recbole.parquet"].append(
            {"user": user_id, "history": train_sequence, "target": sequence[-2]}
        )
        expected["test_recbole.parquet"].append(
            {
                "user": user_id,
                "history": train_sequence + [sequence[-2]],
                "target": sequence[-1],
            }
        )
    return expected


def _check_split(path, expected):
    frame = pd.read_parquet(path)
    if len(frame) != len(expected):
        raise AssertionError(f"{path}: rows={len(frame)} expected={len(expected)}")
    for row_id, (actual, wanted) in enumerate(zip(frame.itertuples(index=False), expected)):
        actual_history = _list_value(actual.history)
        if int(actual.user) != wanted["user"] or actual_history != wanted["history"] or int(actual.target) != wanted["target"]:
            raise AssertionError(f"{path}: mismatch at row {row_id}: {actual} != {wanted}")
    return len(frame)


def _check_split_structure(dataset_dir, *, max_len=20):
    """Validate both local exports and RecBole-produced exports without a source JSON."""
    train = pd.read_parquet(dataset_dir / "train_recbole.parquet")
    valid = pd.read_parquet(dataset_dir / "valid_recbole.parquet")
    test = pd.read_parquet(dataset_dir / "test_recbole.parquet")
    valid_users = set(int(value) for value in valid.user.tolist())
    test_users = set(int(value) for value in test.user.tolist())
    if valid_users != test_users:
        raise AssertionError("valid/test user sets differ")
    all_users = sorted(valid_users | set(int(value) for value in train.user.tolist()))
    if all_users != list(range(len(all_users))):
        raise AssertionError("user ids are not zero-based and contiguous")
    valid_targets = {int(row.user): int(row.target) for row in valid.itertuples(index=False)}

    def ordered_unique(values):
        result = []
        seen = set()
        for value in values:
            value = int(value)
            if value not in seen:
                result.append(value)
                seen.add(value)
        return result

    train_targets = {}
    for row in train.itertuples(index=False):
        user_id = int(row.user)
        history = _list_value(row.history)
        previous_targets = train_targets.setdefault(user_id, [])
        expected_history = previous_targets[-max_len:]
        if history not in (previous_targets, expected_history):
            raise AssertionError(
                f"train history is not the sequential prefix for user {user_id}: "
                f"got {history[-5:]}, expected suffix {expected_history[-5:]}"
            )
        previous_targets.append(int(row.target))

    for row in valid.itertuples(index=False):
        user_id = int(row.user)
        history = _list_value(row.history)
        expected = train_targets.get(user_id, [])
        if history not in (expected, expected[-max_len:]):
            raise AssertionError(f"valid history is not train history for user {user_id}")
        if hasattr(row, "seen_history"):
            if _list_value(row.seen_history) != ordered_unique(expected):
                raise AssertionError(f"valid seen_history is not the full train history for user {user_id}")
    for row in test.itertuples(index=False):
        user_id = int(row.user)
        history = _list_value(row.history)
        expected = train_targets.get(user_id, []) + [valid_targets[user_id]]
        if history not in (expected, expected[-max_len:]):
            raise AssertionError(f"test history is not train+valid history for user {user_id}")
        if hasattr(row, "seen_history"):
            if _list_value(row.seen_history) != ordered_unique(expected):
                raise AssertionError(f"test seen_history is not the full train+valid history for user {user_id}")
    return {"train_recbole.parquet": len(train), "valid_recbole.parquet": len(valid), "test_recbole.parquet": len(test)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", default="./dataset/Amazon_2023_Instruments")
    parser.add_argument("--sid_file", default="./dataset/Amazon_2023_Instruments/sids_for_hgrec_unique.npy")
    parser.add_argument(
        "--source",
        default=None,
        help="Optional exact source JSON. Omit when files were exported directly by RecBole.",
    )
    args = parser.parse_args()

    sid = _load_sid(args.sid_file)
    if sid.ndim != 2 or sid.shape[1] == 0:
        raise AssertionError(f"SID must be [num_items, n_digit], got {sid.shape}")
    if np.any(sid < 0):
        raise AssertionError("SID values must be non-negative")
    shifted = sid + 1
    tuples = [tuple(row) for row in shifted.tolist()]
    if len(set(tuples)) != len(tuples):
        raise AssertionError("SID rows are not one-to-one after the RecBole +1 shift")

    dataset_dir = Path(args.dataset_dir)
    if args.source:
        expected = _expected_splits(args.source)
        counts = {}
        for filename, rows in expected.items():
            counts[filename] = _check_split(dataset_dir / filename, rows)
            for row in rows:
                values = row["history"] + [row["target"]]
                if any(item_id < 0 or item_id >= len(sid) for item_id in values):
                    raise AssertionError(f"{filename}: item id outside SID rows: {values}")
        source_note = "exact source sequence: yes"
    else:
        counts = _check_split_structure(dataset_dir)
        for filename in counts:
            frame = pd.read_parquet(dataset_dir / filename)
            values = [int(item_id) for row in frame.itertuples(index=False) for item_id in (_list_value(row.history) + [int(row.target)])]
            if any(item_id < 0 or item_id >= len(sid) for item_id in values):
                raise AssertionError(f"{filename}: item id outside SID rows")
        source_note = "exact source sequence: structural check only (source not re-read)"

    max_raw_sid = int(sid.max())
    semantic_vocab_size = max_raw_sid + 1
    base_user_token = semantic_vocab_size + 1
    eos_token = base_user_token + 1  # n_user_tokens=1
    vocab_size = eos_token + 1
    print("RecBole TIGER alignment: PASS")
    print(f"  item ids: 0..{len(sid) - 1}; SID rows one-to-one: yes")
    print(f"  train/valid/test rows: {counts}")
    print(f"  {source_note}")
    print(f"  SID namespace: raw+1; n_digit={sid.shape[1]}; semantic_vocab_size={semantic_vocab_size}")
    print(f"  n_user_tokens=1; base_user_token={base_user_token}; eos={eos_token}; vocab={vocab_size}")
    print("  beam=50; top-k=[5, 10]; exclude_history=true; monitor=ndcg@10")


if __name__ == "__main__":
    main()
