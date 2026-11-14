"""What each training record carries besides the embedding.

The host iterates this dataset and publishes the returned mapping into
``runtime_context`` before every forward. The baseline publishes ``item_id`` only.

A mechanism that needs a different per-row field edits *this* file (it is a purpose
file like ``distances/distance.py``) and adds the column here; the trainer stays
untouched because it only forwards whatever dict comes back.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import torch
from torch.utils.data import Dataset


class RecordDataset(Dataset):
    """``(embedding, {field: value})`` per row."""

    def __init__(self, embeddings: torch.Tensor, fields: Mapping[str, torch.Tensor]):
        if embeddings.shape[0] == 0:
            raise RuntimeError("empty training table")
        for name, column in fields.items():
            if column.shape[0] != embeddings.shape[0]:
                raise RuntimeError(
                    "field %r has %d rows; embeddings have %d"
                    % (name, column.shape[0], embeddings.shape[0])
                )
        self._embeddings = embeddings
        self._fields = {name: column for name, column in fields.items()}

    def __len__(self) -> int:
        return int(self._embeddings.shape[0])

    def __getitem__(self, index: int):
        return self._embeddings[index], {name: column[index] for name, column in self._fields.items()}


def build(embeddings: torch.Tensor, train_ids: np.ndarray) -> RecordDataset:
    """The default view: embedding + item id. Add more columns here if you need them."""
    return RecordDataset(
        embeddings,
        {"item_id": torch.as_tensor(np.asarray(train_ids, dtype=np.int64))},
    )


def collate(batch):
    """Stack embeddings and each named field."""
    embeddings = torch.stack([row[0] for row in batch])
    names = batch[0][1].keys()
    values = {name: torch.stack([row[1][name] for row in batch]) for name in names}
    return embeddings, values
