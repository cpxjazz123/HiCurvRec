"""Frozen token readout for the Issue #263 candidate.

The pretrained directed GNN and its prediction heads are absent from this
module. Stage2 gathers the persisted token/mask cache by global item ID and
passes it to the trainable layer-wise expert attention. No graph object, no GNN
forward and no GNN optimizer parameter exists on this path.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from tools.issue252.host_transfer import host_int64
from .layer_gate import LayerExpertAttention

ARM_NAME = "TIGER_frozen_topology_gnn_rq"
N_ITEMS = 24587
TOKEN_TYPES = ("one_hop", "two_hop", "in_only", "out_only")
TOKEN_DIM = 32


def _repository_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if ((parent / "stage2_RQ-VAE").is_dir()
                and (parent / "tools/research/stage2/contract.py").is_file()):
            return parent
    raise RuntimeError("cannot locate repository marker")


ROOT = _repository_root()
TOKEN_CACHE = ROOT / "results" / "stage2_RQ-VAE" / ARM_NAME / "token_cache.npz"
TOKEN_MANIFEST = ROOT / "results" / "stage2_RQ-VAE" / ARM_NAME / "token_manifest.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class FrozenTokenStore:
    """Minimal catalogue identity; never an EgoStore or a graph encoder."""
    def __init__(self, n: int = N_ITEMS):
        self.n = int(n)


class FrozenTokenAttention(nn.Module):
    """Load persisted tokens/mask and expose the parent gate interface."""

    def __init__(self, cache_path: Path | None = None,
                 manifest_path: Path | None = None):
        super().__init__()
        self.cache_path = Path(cache_path) if cache_path is not None else TOKEN_CACHE
        self.manifest_path = (Path(manifest_path) if manifest_path is not None
                              else TOKEN_MANIFEST)
        if not self.cache_path.is_file():
            raise FileNotFoundError("frozen token cache missing: " + str(self.cache_path))
        with np.load(self.cache_path, allow_pickle=False) as blob:
            missing = {"tokens", "mask", "item_ids"} - set(blob.files)
            if missing:
                raise ValueError("token cache missing arrays: " + str(sorted(missing)))
            tokens = np.asarray(blob["tokens"]).astype(np.float32)
            mask = np.asarray(blob["mask"]).astype(bool)
            item_ids = np.asarray(blob["item_ids"]).astype(np.int64)
        if tokens.shape != (N_ITEMS, len(TOKEN_TYPES), TOKEN_DIM):
            raise ValueError("token cache shape mismatch: " + str(tokens.shape))
        if mask.shape != tokens.shape[:2] or item_ids.shape != (N_ITEMS,):
            raise ValueError("token mask/item identity shape mismatch")
        if not np.array_equal(item_ids, np.arange(N_ITEMS, dtype=np.int64)):
            raise ValueError("token item order is not 0..24586")
        if not np.isfinite(tokens).all() or not mask.any():
            raise ValueError("invalid token values or empty mask")
        self.register_buffer("frozen_tokens", torch.from_numpy(tokens), persistent=False)
        self.register_buffer("frozen_mask", torch.from_numpy(mask), persistent=False)
        self.register_buffer("frozen_item_ids", torch.from_numpy(item_ids), persistent=False)
        self.cache_sha256 = sha256(self.cache_path)
        self.attention = LayerExpertAttention()
        self.last_tokens = None
        self.last_attention = None
        self.last_token_mask = None
        self.gnn_forward_calls = 0
        self.gnn_backward_calls = 0

    def gates_for_items(self, store, item_ids, device, chunk=1024, behavior_on=True):
        del store, chunk  # the frozen path has no graph/ego batch to chunk
        raw = item_ids.detach().cpu().numpy() if torch.is_tensor(item_ids) else np.asarray(item_ids)
        ids = np.asarray(raw, dtype=np.int64).reshape(-1)
        if ((ids < 0) | (ids >= N_ITEMS)).any():
            raise IndexError("item id outside frozen token catalogue")
        index = torch.as_tensor(ids, dtype=torch.long, device=device)
        tokens = self.frozen_tokens.to(device)[index]
        mask = self.frozen_mask.to(device)[index]
        weights, attention, effective_mask = self.attention(tokens, mask)
        self.last_tokens = tokens
        self.last_attention = attention
        self.last_token_mask = effective_mask
        if not behavior_on:
            weights = torch.full_like(weights, 1.0 / 3.0)
        return self.last_tokens, weights
