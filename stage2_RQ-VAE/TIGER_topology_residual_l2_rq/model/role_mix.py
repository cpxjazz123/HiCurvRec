"""Role-conditioned level-2 prefix mixture.

The second semantic level of the R05 candidate takes its curvature mixture from
the item's *graphlet orbit role* instead of the item's own free gate.  The role
is a training-only behaviour label (one label per item, read from the frozen
training graphlet instances); the mixture is a learned table with one row per
role, so the level-2 geometry is a function of the role and of the checkpoint
alone.  It therefore never depends on which other items share the batch, and
the exported SID is the same global identifier in every batch and at inference.

Rows are initialised uniformly, i.e. the level-2 cost starts as the mean of the
three curvature experts; training separates the roles.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

TABLE = Path(__file__).resolve().parent.parent / 'role_table.json'
NO_ROLE = -1


class RoleTable:
    """The frozen per-item role label, with its own provenance."""

    def __init__(self, path=TABLE):
        payload = json.loads(Path(path).read_text(encoding='utf-8'))
        self.n_roles = int(payload['role_count'])
        self.n_items = int(payload['item_count'])
        self.source_sha256 = str(payload['source']['sha256'])
        self.split_rule = str(payload['split_rule'])
        self.table = np.asarray(payload['table'], dtype=np.int64)
        if len(self.table) != self.n_items:
            raise ValueError('role table length differs from the item count')

    def roles_for(self, item_ids) -> np.ndarray:
        ids = np.asarray(item_ids, dtype=np.int64).reshape(-1)
        if ids.size and (ids.min() < 0 or ids.max() >= self.n_items):
            raise ValueError('item identity outside the role table')
        return self.table[ids]

    def check_training_only(self) -> None:
        """The label may only come from the training instances of the asset."""
        if 'training' not in self.split_rule or 'heldout' in self.split_rule:
            raise ValueError('role label must come from the training instances only')


class RoleMixture(nn.Module):
    """One learnable curvature mixture per role, shared by the level-2 cost."""

    def __init__(self, table: RoleTable | None = None):
        super().__init__()
        self.table = table if table is not None else RoleTable()
        self.table.check_training_only()
        # One row carries the items that appear in no training instance; one
        # more is the single global mixture the behaviour-off switch uses.
        self.logits = nn.Parameter(torch.zeros(self.table.n_roles + 2, 3))

    @property
    def n_rows(self) -> int:
        return int(self.logits.shape[0])

    @property
    def global_row(self) -> int:
        """The row of the single global mixture."""
        return self.table.n_roles + 1

    def weights(self) -> torch.Tensor:
        """The per-role mixtures, normalised over the three curvature experts."""
        return torch.softmax(self.logits, dim=-1)

    def rows_for(self, item_ids, device) -> torch.Tensor:
        """The mixture row of every item of the batch."""
        roles = self.table.roles_for(item_ids)
        rows = np.where(roles < 0, self.table.n_roles, roles)
        return torch.as_tensor(rows, dtype=torch.long, device=device)

    def global_weights(self) -> torch.Tensor:
        """The single mixture every item shares when the behaviour side is off."""
        return self.weights()[self.global_row]

    def forward(self, item_ids, device) -> torch.Tensor:
        """The level-2 curvature mixture of every item of the batch."""
        return self.weights()[self.rows_for(item_ids, device)]
