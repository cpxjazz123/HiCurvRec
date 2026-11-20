"""The frozen baseline SID as a vector source: z, hard q and tokens per item.

The Student in issue #222 B must see exactly what the Stage 2 trainer sees: the
encoder's output `z` and the *sum of the picked code words* the real quantiser
produces.  Reading the checkpoint with the bias and the right table layout is the
whole job (CLAUDE.md 89): a forward pass without `encoder.mlp.*.bias` is off by
~5.7x, and SimVQ assigns from `embed_proj(embed.weight)`, not from the raw table.

Nothing here is fitted or tuned; it is a read-only view of
`results/stage2_RQ-VAE/<ARM>/rqvae_best.pth` and its `item_sids.json`.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DEFAULT_ARM = REPO / "results" / "stage2_RQ-VAE" / "TIGER_RQ-VAE"
DEFAULT_EMBEDDINGS = REPO / "stage1_GeneEmbedding" / "output" / "sentence_t5.npy"
CHUNK = 4096


def _level_of(name: str) -> int:
    for part in reversed(name.split(".")):
        if part.isdigit():
            return int(part)
    return 0


def encoder_layers(state: dict) -> list:
    """(weight, bias) pairs in `MLP.forward` order; every linear layer has a bias."""
    weights, biases = {}, {}
    for key, value in state.items():
        if not key.startswith("encoder.mlp"):
            continue
        parts = key.split(".")
        if len(parts) != 4 or not parts[2].isdigit():
            continue
        index = int(parts[2])
        if parts[3] == "weight" and value.ndim == 2:
            weights[index] = value
        elif parts[3] == "bias":
            biases[index] = value
    if not weights:
        raise ValueError("no encoder.mlp linear layers in the checkpoint")
    return [(weights[index], biases.get(index)) for index in sorted(weights)]


def codebook_tables(state: dict) -> list:
    """Each level's assignment table, in level order, SimVQ projected."""
    levels, projections = {}, {}
    for key, value in state.items():
        if key.endswith("embed_proj.weight"):
            levels.setdefault(key.rsplit(".", 1)[0].replace("embed_proj", "embed"), value)
            projections[key.rsplit(".", 1)[0]] = value
        elif key.endswith(".embed.weight") or (key.endswith(".embed") and value.ndim == 2):
            levels.setdefault(key, value)
    if not levels:
        return []
    ordered = sorted(levels, key=_level_of)
    tables = []
    for name in ordered:
        table = levels[name]
        projection = projections.get(name.replace(".embed", ".embed_proj"))
        tables.append(table @ projection.t() if projection is not None else table)
    return tables


class FrozenSID:
    """Read-only view of one Stage 2 arm: encoder output, hard q and tokens."""

    def __init__(self, arm_dir=DEFAULT_ARM, embeddings_path=DEFAULT_EMBEDDINGS):
        arm_dir = Path(arm_dir)
        payload = torch.load(arm_dir / "rqvae_best.pth", map_location="cpu",
                             weights_only=False)
        state = payload.get("state_dict", payload)
        self.layers = [(weight.float(), None if bias is None else bias.float())
                       for weight, bias in encoder_layers(state)]
        self.tables = [table.float() for table in codebook_tables(state)]
        if not self.tables:
            raise ValueError("no codebook tables in %s" % (arm_dir / "rqvae_best.pth"))
        if self.tables[0].shape[1] != self.layers[-1][0].shape[0]:
            raise ValueError("codebook width %d disagrees with encoder width %d"
                             % (self.tables[0].shape[1], self.layers[-1][0].shape[0]))

        raw = json.loads((arm_dir / "item_sids.json").read_text())
        width = len(next(iter(raw.values())))
        # The fourth entry is the export's collision counter, not a code.
        self.depth = min(width, len(self.tables))
        self.tokens = np.empty((len(raw), self.depth), dtype=np.int64)
        for index in range(len(raw)):
            self.tokens[index] = raw[str(index)][:self.depth]
        self.arm_dir = arm_dir

        embeddings = np.load(embeddings_path, mmap_mode="r")
        if embeddings.shape[1] != self.layers[0][0].shape[1]:
            raise ValueError("embedding width %d disagrees with encoder input %d"
                             % (embeddings.shape[1], self.layers[0][0].shape[1]))
        self.embeddings = embeddings
        self.n_items = embeddings.shape[0]
        if len(self.tokens) > self.n_items:
            raise ValueError("%d SIDs for %d embeddings" % (len(self.tokens), self.n_items))

    def _encode(self, ids: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(np.asarray(self.embeddings[ids], dtype=np.float32))
        with torch.no_grad():
            for index, (weight, bias) in enumerate(self.layers):
                x = x @ weight.t()
                if bias is not None:
                    x = x + bias
                if index < len(self.layers) - 1:
                    x = torch.relu(x)
        return x

    def z(self, ids: np.ndarray) -> torch.Tensor:
        out = []
        ids = np.asarray(ids)
        for start in range(0, len(ids), CHUNK):
            out.append(self._encode(ids[start:start + CHUNK]))
        return torch.cat(out) if out else torch.zeros(0, self.tables[0].shape[1])

    def q(self, ids: np.ndarray) -> torch.Tensor:
        ids = np.asarray(ids)
        tokens = torch.from_numpy(self.tokens[ids].T.copy())          # (depth, N)
        quantised = sum(table[tokens[level]] for level, table in enumerate(self.tables))
        return quantised

    def sid(self, ids: np.ndarray) -> np.ndarray:
        return self.tokens[np.asarray(ids)]

    def describe(self) -> dict:
        import hashlib

        checkpoint = self.arm_dir / "rqvae_best.pth"
        return {
            "arm": str(self.arm_dir),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "depth": self.depth,
            "codebook_size": [int(table.shape[0]) for table in self.tables],
            "codebook_dim": int(self.tables[0].shape[1]),
            "encoder_layers": [int(weight.shape[0]) for weight, _ in self.layers],
            "items": int(len(self.tokens)),
        }
