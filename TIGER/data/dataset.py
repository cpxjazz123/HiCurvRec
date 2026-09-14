"""Dataset and SID utilities shared by the HG-Rec TIGER runner.

The implementation intentionally follows RecBole3.0's vanilla TIGER data
contract:

* item ids are zero-based;
* a SID row at position ``i`` belongs to item ``i``;
* SID tokens are ``raw_code + 1`` (one common token namespace, no per-level
  offsets);
* the first training prefix may have an empty history; padding is performed
  by the token collator, not here.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import Dataset


def _as_int_list(value):
    """Convert parquet/JSON/numpy list-like values to Python ints."""
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        value = value.tolist()
    return [int(x) for x in list(value)]


def _truncate_history(history, max_len):
    """Keep the same recent-history truncation as RecBole TIGER."""
    history = _as_int_list(history)
    return history[-max_len:] if len(history) > max_len else history


def process_data(file_path, mode, max_len, PAD_TOKEN=0):
    """Read an already-expanded split and build TIGER records.

    ``train_recbole.parquet`` contains one row per training prefix.  Thus a
    training row is kept as one sample; this function must not create another
    sliding window.  Evaluation rows are also one row per user, with the
    history supplied by the split generator.
    """
    del PAD_TOKEN  # Padding belongs to the collator; retained for API compat.
    data = pd.read_parquet(file_path)
    if mode not in {"train", "evaluation"}:
        raise ValueError("Mode must be 'train' or 'evaluation'.")

    records = []
    for row in data.itertuples(index=False):
        user_id = int(row.user)
        history = _as_int_list(row.history)
        seen_history = (
            _as_int_list(row.seen_history)
            if hasattr(row, "seen_history")
            else history
        )
        target = int(row.target)
        records.append(
            {
                "user": user_id,
                "history": _truncate_history(history, max_len),
                "seen_history": seen_history,
                "target_item": target,
                # Keep this field as the item id until _prepare_data replaces
                # it with the target SID tuple.
                "target": target,
            }
        )
    return records


def pad_or_truncate(sequence, max_len, PAD_TOKEN=0):
    """Compatibility helper: RecBole truncates but does not left-pad here."""
    del PAD_TOKEN
    return _truncate_history(sequence, max_len)


def _load_sid_rows(code_path):
    """Load raw SID rows from RecBole JSON or the existing HG-Rec NPY file."""
    path = Path(code_path)
    suffix = path.suffix.lower()

    if suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        if not isinstance(payload, dict):
            raise ValueError("RecBole TIGER SID JSON must be an object")

        # RecBole's item_sids.json is an object keyed by zero-based item ids.
        # Numeric string keys are accepted because JSON has no int key type.
        try:
            keys = sorted(int(key) for key in payload.keys())
        except (TypeError, ValueError) as exc:
            raise ValueError("SID JSON keys must be remapped integer item ids") from exc
        expected = list(range(len(keys)))
        if keys != expected:
            raise ValueError(
                f"SID JSON keys must be exactly 0..{len(keys) - 1}; "
                f"got first/last={keys[:2]}...{keys[-2:]}"
            )
        rows = []
        for item_id in keys:
            raw_sid = payload[str(item_id)]
            if not isinstance(raw_sid, list) or not raw_sid:
                raise ValueError(f"SID for item {item_id} must be a non-empty list")
            if any(not isinstance(value, int) or isinstance(value, bool) for value in raw_sid):
                raise ValueError(f"SID for item {item_id} must contain integers")
            rows.append(raw_sid)
        return rows

    data = np.load(path, allow_pickle=False)
    if data.ndim != 2:
        raise ValueError(f"SID array must be 2-D, got shape={data.shape} from {path}")
    return data.tolist()


def item2code(code_path, codebook_size=None):
    """Build zero-based item/SID maps using RecBole's token namespace."""
    del codebook_size  # Kept for compatibility with older HG-Rec callers.
    rows = _load_sid_rows(code_path)
    if not rows:
        raise ValueError(f"SID file is empty: {code_path}")

    width = len(rows[0])
    if width == 0:
        raise ValueError(f"SID rows must not be empty: {code_path}")

    item_to_code = {}
    code_to_item = {}
    max_token = 0
    for item_id, raw_code in enumerate(rows):
        raw_code = _as_int_list(raw_code)
        if len(raw_code) != width:
            raise ValueError(
                f"SID width is not fixed at item {item_id}: "
                f"expected {width}, got {len(raw_code)}"
            )
        if any(code < 0 for code in raw_code):
            raise ValueError(f"SID values must be non-negative at item {item_id}")

        # RecBole's TIGERSIDCodec: every raw SID component is shifted by one;
        # all codebooks share this namespace.
        tokens = tuple(code + 1 for code in raw_code)
        if tokens in code_to_item:
            other = code_to_item[tokens]
            raise ValueError(
                f"Duplicate SID after +1 shift for items {other} and {item_id}: {tokens}"
            )
        item_to_code[item_id] = tokens
        code_to_item[tokens] = item_id
        max_token = max(max_token, max(tokens))

    return item_to_code, code_to_item, max_token, width


class GenRecDataset(Dataset):
    """HG-Rec dataset with the vanilla RecBole TIGER item contract."""

    def __init__(
        self,
        dataset_path,
        code_path,
        mode,
        codebook_size,
        max_len,
        PAD_TOKEN=0,
        n_user_tokens=1,
    ):
        self.dataset_path = str(dataset_path)
        self.code_path = str(code_path)
        self.mode = mode
        self.max_len = int(max_len)
        self.PAD_TOKEN = int(PAD_TOKEN)
        self.codebook_size = codebook_size
        self.n_user_tokens = int(n_user_tokens)
        if self.n_user_tokens <= 0:
            raise ValueError("Vanilla RecBole TIGER requires n_user_tokens > 0")

        self.pad_token_id = 0
        self.eos_token = 1
        self.item_to_code, self.code_to_item, self.max_sid_token, self.n_digit = item2code(
            self.code_path, self.codebook_size
        )
        self.num_items = len(self.item_to_code)
        # RecBole's TIGER codec exposes a deterministic item-id fallback
        # order for short/invalid beam outputs.  The evaluator uses the same
        # order after filtering duplicates and seen-history items.
        self.fallback_item_ids = tuple(sorted(self.item_to_code))
        # ``max_sid_token`` is max(raw_sid + 1), which equals RecBole's
        # ``max_sid + 1``.  Do not add another one here: the highest semantic
        # token itself is already included in this vocabulary size.
        self.semantic_vocab_size = self.max_sid_token

        # Exactly RecBole TIGER's layout:
        #   PAD/decoder-start=0, EOS=1, user buckets=2..1+n,
        #   semantic SID tokens start at semantic_vocab_size+1.
        self.base_user_token = self.semantic_vocab_size + 1
        self.user_token_max = self.base_user_token + self.n_user_tokens - 1
        self.eos_token = self.base_user_token + self.n_user_tokens
        self.vocab_size = self.eos_token + 1
        self.max_token_seq_len = self.max_len * self.n_digit + 2

        self.data = self._prepare_data()

    def user_token_for(self, user_id):
        return self.base_user_token + (int(user_id) % self.n_user_tokens)

    def _prepare_data(self):
        processed_data = process_data(
            self.dataset_path, self.mode, self.max_len, self.PAD_TOKEN
        )

        for item in processed_data:
            history_items = list(item["history"])
            target_item = int(item["target_item"])
            missing_history = [x for x in history_items if x not in self.item_to_code]
            if target_item not in self.item_to_code:
                missing_history.append(target_item)
            if missing_history:
                raise KeyError(
                    f"Items absent from SID file {self.code_path}: "
                    f"{sorted(set(missing_history))[:10]}"
                )

            item["history_items"] = history_items
            # RecBole full evaluation excludes every item observed before the
            # held-out target. Keep the complete (untruncated) seen list in
            # item-id space; the input history may be truncated to max_len.
            seen_items = [int(x) for x in item.get("seen_history", [])]
            missing_seen = [x for x in seen_items if x not in self.item_to_code]
            if missing_seen:
                raise KeyError(
                    f"Items absent from SID file {self.code_path} (seen history): "
                    f"{sorted(set(missing_seen))[:10]}"
                )
            item["seen_item_ids"] = seen_items
            item["history"] = [self.item_to_code[x] for x in history_items]
            item["target_item"] = target_item
            item["target"] = self.item_to_code[target_item]
        return processed_data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
