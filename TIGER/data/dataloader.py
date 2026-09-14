"""DataLoader helpers implementing RecBole TIGER's padding contract."""

import torch
from torch.utils.data import DataLoader


def _pad_token_rows(rows, width, pad_value=0):
    """Right-pad variable-length token rows to a fixed encoder width."""
    result = torch.full((len(rows), width), pad_value, dtype=torch.long)
    attention = torch.zeros((len(rows), width), dtype=torch.long)
    for row_id, row in enumerate(rows):
        values = torch.as_tensor(row, dtype=torch.long)
        if values.numel() > width:
            raise ValueError(
                f"Token sequence length {values.numel()} exceeds configured width {width}"
            )
        length = values.numel()
        result[row_id, :length] = values
        attention[row_id, :length] = 1
    return result, attention


class GenRecDataLoader(DataLoader):
    """DataLoader with fixed-width right padding like vanilla TIGER.

    ``sample_collator`` must return dictionaries containing ``input_ids`` and
    ``labels``.  It may also return ``target_item`` and ``seen_item_ids`` for
    full evaluation.  The latter are padded per batch and accompanied by a
    boolean ``seen_mask``.
    """

    def __init__(
        self,
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        sampler=None,
        collate_fn=None,
        sample_collator=None,
        max_token_seq_len=None,
        include_seen=False,
        pad_token_id=0,
        **kwargs,
    ):
        self.sample_collator = sample_collator
        self.max_token_seq_len = max_token_seq_len
        self.include_seen = include_seen
        self.pad_token_id = pad_token_id

        if sample_collator is None:
            super().__init__(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=pin_memory,
                sampler=sampler,
                collate_fn=collate_fn,
                **kwargs,
            )
        else:
            super().__init__(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=pin_memory,
                sampler=sampler,
                collate_fn=self._collate_samples,
                **kwargs,
            )

    def _collate_samples(self, batch):
        processed = [self.sample_collator(item) for item in batch]
        input_rows = [item["input_ids"] for item in processed]
        label_rows = [item["labels"] for item in processed]

        width = self.max_token_seq_len or max(len(row) for row in input_rows)
        input_ids, attention_mask = _pad_token_rows(
            input_rows, width, pad_value=self.pad_token_id
        )

        # TIGER targets are fixed-width SID + EOS sequences, so stack them
        # directly.  Keeping this check catches an accidental namespace or SID
        # width mismatch before training starts.
        label_lengths = {len(row) for row in label_rows}
        if len(label_lengths) != 1:
            raise ValueError(f"Labels must have one fixed width, got {label_lengths}")
        labels = torch.as_tensor(label_rows, dtype=torch.long)

        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        if all("target_item" in item for item in processed):
            result["target_item"] = torch.as_tensor(
                [item["target_item"] for item in processed], dtype=torch.long
            )

        if self.include_seen and all("seen_item_ids" in item for item in processed):
            seen_rows = [list(item["seen_item_ids"]) for item in processed]
            seen_width = max((len(row) for row in seen_rows), default=0)
            seen_items = torch.zeros((len(seen_rows), seen_width), dtype=torch.long)
            seen_mask = torch.zeros((len(seen_rows), seen_width), dtype=torch.bool)
            for row_id, row in enumerate(seen_rows):
                if row:
                    seen_items[row_id, : len(row)] = torch.as_tensor(
                        row, dtype=torch.long
                    )
                    seen_mask[row_id, : len(row)] = True
            result["seen_item_ids"] = seen_items
            result["seen_mask"] = seen_mask
        return result
