"""Stage 3 entry gate: three-level collision below 5% and quantisation error in band.

Both numbers are measured on the arm's own SID and checkpoint; nothing is passed
in by the caller, so the gate cannot be talked past.

* **collision** = ``1 - distinct(three-code SID) / items``.  Every arm that
  reached ``test_R@10 > 0.06`` among the 74 measured arms stayed at or below
  4.5%; arms at 20-99% never did.
* **quantisation error** = ``mean ||z - sum(codewords)||`` over the items, with
  ``z`` the Stage 2 encoder output and the codewords those the three-level SID
  selects.  Below 0.20 the codebook memorises each item (``dul_rq``: 0.160 with
  1.6% collision and recall 0.026); above 0.60 the vectors are pressed together
  (``cogeo_region_argmax``: 21.4 with 6% distinct codes).

Missing evidence is a refusal: no checkpoint, no embeddings or a codebook whose
width disagrees with the encoder output means the arm cannot be judged, and an
unjudged arm does not start Stage 3.

The import surface is deliberately stdlib-only; numpy and torch load inside the
functions so that the stdlib-only MCP server can import this module directly.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

COLLISION_LIMIT = 0.05
QUANT_ERROR_LOW = 0.20
QUANT_ERROR_HIGH = 0.60
CHUNK = 8192


def repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "stage1_GeneEmbedding").is_dir() and (parent / "stage2_RQ-VAE").is_dir():
            return parent
    raise RuntimeError("repository root not found above " + __file__)


DEFAULT_EMBEDDINGS = repo_root() / "stage1_GeneEmbedding" / "output" / "sentence_t5.npy"


def three_level_collision(payload: dict) -> float:
    triples = {tuple(value[:3]) for value in payload.values() if hasattr(value, "__len__")}
    return 1.0 - len(triples) / max(len(payload), 1)


def _encoder_weights(state: dict) -> list:
    weights = sorted(
        (int(key.split(".")[2]), value) for key, value in state.items()
        if key.startswith("encoder.mlp") and key.endswith("weight") and value.ndim == 2)
    return [value for _, value in weights]


def _codebook_tables(state: dict) -> list:
    euclid = [value for key, value in sorted(state.items())
              if "vq_layers" in key and key.endswith("embed.weight")]
    mixed = [value for key, value in sorted(state.items())
             if ".levels." in key and key.endswith("embed.weight")]
    return euclid or mixed


def quantization_error(embeddings, weights, tables, tokens) -> float:
    import torch

    depth = int(tokens.shape[1])
    if not weights or not tables:
        raise ValueError("no encoder weights or no codebook tables in the checkpoint")
    if weights[0].shape[1] != embeddings.shape[1]:
        raise ValueError("encoder input width %d disagrees with embeddings width %d"
                         % (weights[0].shape[1], embeddings.shape[1]))
    if len(tables) < depth:
        raise ValueError("%d codebook levels cannot encode a %d-code SID" % (len(tables), depth))
    if tables[0].shape[1] != weights[-1].shape[0]:
        raise ValueError("codebook width %d disagrees with encoder output width %d"
                         % (tables[0].shape[1], weights[-1].shape[0]))
    total = 0.0
    count = 0
    for start in range(0, embeddings.shape[0], CHUNK):
        block = embeddings[start:start + CHUNK]
        with torch.no_grad():
            x = torch.as_tensor(block)
            for index, weight in enumerate(weights):
                x = x @ weight.t()
                if index < len(weights) - 1:
                    x = torch.relu(x)
            quantised = None
            for level in range(depth):
                picked = tables[level][torch.as_tensor(tokens[start:start + CHUNK, level])]
                quantised = picked if quantised is None else quantised + picked
            total += float(torch.linalg.vector_norm(x - quantised, dim=1).sum())
            count += x.shape[0]
    return total / max(count, 1)


def evaluate(sid_path, embeddings_path=None, checkpoint_path=None) -> dict:
    sid_path = Path(sid_path)
    if sid_path.is_dir():
        sid_path = sid_path / "item_sids.json"
    verdict = {"sid": str(sid_path), "allow": False, "reason": "", "collision": None,
               "quant_error": None, "limits": {"collision": COLLISION_LIMIT,
                                               "quant_error": [QUANT_ERROR_LOW, QUANT_ERROR_HIGH]}}
    try:
        payload = json.loads(sid_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        verdict["reason"] = "unreadable SID json: " + repr(exc)
        return verdict
    if not isinstance(payload, dict) or not payload:
        verdict["reason"] = "SID json is not a non-empty object"
        return verdict
    collision = three_level_collision(payload)
    verdict["collision"] = collision
    if collision >= COLLISION_LIMIT:
        verdict["reason"] = ("three-level collision %.6f is not below %.2f: %.1f%% of the items "
                             "share an SID with another item"
                             % (collision, COLLISION_LIMIT, 100 * collision))
        return verdict

    checkpoint = Path(checkpoint_path) if checkpoint_path else sid_path.parent / "rqvae_best.pth"
    if not checkpoint.is_file():
        verdict["reason"] = "checkpoint missing at %s: the quantisation error cannot be measured" % checkpoint
        return verdict
    embeddings_file = Path(embeddings_path) if embeddings_path else DEFAULT_EMBEDDINGS
    if not embeddings_file.is_file():
        verdict["reason"] = "stage1 embeddings missing at %s" % embeddings_file
        return verdict

    import numpy as np
    import torch

    try:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = state.get("state_dict", state)
        weights = _encoder_weights(state)
        tables = _codebook_tables(state)
    except Exception as exc:  # noqa: BLE001 - the reason is reported, never hidden
        verdict["reason"] = "checkpoint unreadable: " + repr(exc)
        return verdict
    keys = list(payload)
    depth = min(3, min(len(value) for value in payload.values()))
    tokens = np.array([payload[key][:depth] for key in keys], dtype=np.int64)
    try:
        embeddings = np.load(embeddings_file).astype("float32")
    except Exception as exc:  # noqa: BLE001
        verdict["reason"] = "stage1 embeddings unreadable: " + repr(exc)
        return verdict
    if embeddings.shape[0] != len(tokens):
        verdict["reason"] = ("%d embeddings cannot describe %d SID rows"
                             % (embeddings.shape[0], len(tokens)))
        return verdict
    try:
        error = quantization_error(embeddings, weights, tables, tokens)
    except Exception as exc:  # noqa: BLE001
        verdict["reason"] = "quantisation error unavailable: " + repr(exc)
        return verdict
    verdict["quant_error"] = error
    if error < QUANT_ERROR_LOW:
        verdict["reason"] = ("quantisation error %.3f is below %.2f: the codebook reproduces the "
                             "item vectors instead of learning a structure that generalises"
                             % (error, QUANT_ERROR_LOW))
        return verdict
    if error > QUANT_ERROR_HIGH:
        verdict["reason"] = ("quantisation error %.3f is above %.2f: the vectors are pressed "
                             "together" % (error, QUANT_ERROR_HIGH))
        return verdict
    verdict["allow"] = True
    verdict["reason"] = "collision %.4f < %.2f and quantisation error %.3f inside [%.2f, %.2f]" % (
        collision, COLLISION_LIMIT, error, QUANT_ERROR_LOW, QUANT_ERROR_HIGH)
    return verdict


def main(argv: list) -> int:
    if len(argv) != 2:
        print(json.dumps({"allow": False, "reason": "usage: sid_quality <item_sids.json>"}))
        return 2
    print(json.dumps(evaluate(argv[1])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
