"""Checkpoint-bound SID export and verification.

The export must come from the bytes that were just saved, not from the live training
module.  This module loads the checkpoint into a fresh model, computes the argmin tokens
with the export assignment, then loads a second fresh model and repeats the computation.
A mismatch is an export failure, not a warning.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch

from model import RQVAE


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype=np.int64))
    return hashlib.sha256(array.tobytes()).hexdigest()


def load_checkpoint_model(checkpoint: Path, *, config, in_dim: int, distance, device):
    model = RQVAE(config, in_dim=in_dim, distance=distance).to(device)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing={missing[:3]} unexpected={unexpected[:3]}")
    for layer in model.rq.vq_layers:
        layer.use_sk = False
    model.eval()
    return model


def checkpoint_tokens(model, embeddings: torch.Tensor, device) -> np.ndarray:
    for layer in model.rq.vq_layers:
        layer._skip_ddp_reduce = True
    try:
        with torch.no_grad():
            tokens = model.get_indices(embeddings.to(device), infer_use_sk=False)
        return tokens.cpu().numpy().astype(np.int64)
    finally:
        for layer in model.rq.vq_layers:
            layer._skip_ddp_reduce = False


def export_tokens_from_checkpoint(
    checkpoint: Path,
    *,
    config,
    embeddings: torch.Tensor,
    in_dim: int,
    distance,
    device,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return (export tokens, re-encoded tokens, identity metadata)."""
    first = load_checkpoint_model(checkpoint, config=config, in_dim=in_dim,
                                  distance=distance, device=device)
    exported = checkpoint_tokens(first, embeddings, device)
    second = load_checkpoint_model(checkpoint, config=config, in_dim=in_dim,
                                   distance=distance, device=device)
    reencoded = checkpoint_tokens(second, embeddings, device)
    metadata = {
        "checkpoint_sha256": sha256_file(checkpoint),
        "export_tokens_sha256": sha256_array(exported),
        "reencoded_tokens_sha256": sha256_array(reencoded),
        "tokens_match": bool(np.array_equal(exported, reencoded)),
        "items": int(len(exported)),
        "assignment": "argmin",
        "model_mode": "eval",
        "export_device": str(device),
    }
    return exported, reencoded, metadata
