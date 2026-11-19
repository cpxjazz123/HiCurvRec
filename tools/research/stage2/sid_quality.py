"""Stage 3 entry gate: the SID must be a taxonomy, not an identity.

Three numbers are measured on the arm itself -- its SID table, its checkpoint and
the stage 1 embeddings -- and nothing is passed in by the caller, so the gate
cannot be talked past.

* **collision** = ``1 - distinct(three-code SID) / items`` below 0.10.  Among the
  73 arms with a checkpoint on disk every arm above ``test_R@10 0.06`` stayed at
  0.030-0.087; arms at 0.10-0.99 never did.
* **quantisation error** = ``mean ||z - sum(codewords)||`` inside [0.13, 0.65].
  The band is where the three codes keep a structure instead of reproducing the
  item vector (below it: ``dul_rq`` 0.160 at recall 0.026) or crushing it (above
  it: ``cogeo_region_argmax`` 21.4 at 6% distinct codes).
* **effective rank** ``PR = (sum lambda)^2 / sum lambda^2`` of the encoder output
  at or below 12.  The 12 arms above 0.06 compressed 32 dimensions into 3.6-11.5
  effective ones; the two arms the other two conditions let through kept 14.1
  and 19.1, i.e. they carried the 768-dim input through unorganised, and the
  three codes cannot cover that.

Missing evidence is a refusal: no checkpoint, no embeddings or a codebook whose
width disagrees with the encoder output means the arm cannot be judged, and an
unjudged arm does not start Stage 3.

The import surface is deliberately stdlib-only; numpy and torch load inside the
functions so that the stdlib-only MCP server can run this as a subprocess.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

COLLISION_LIMIT = 0.10
QUANT_ERROR_LOW = 0.13
QUANT_ERROR_HIGH = 0.65
EFFECTIVE_RANK_LIMIT = 12.0
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


def measure_vectors(embeddings, weights, tables, tokens) -> tuple:
    """Return (quantisation error, effective rank) of the encoder output."""
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
    error = 0.0
    count = 0
    first = None
    second = None
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
            error += float(torch.linalg.vector_norm(x - quantised, dim=1).sum())
            count += x.shape[0]
            wide = x.double()
            first = wide.sum(0) if first is None else first + wide.sum(0)
            second = wide.t() @ wide if second is None else second + wide.t() @ wide
    mean = first / max(count, 1)
    covariance = second / max(count, 1) - torch.outer(mean, mean)
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp(min=1e-12)
    effective_rank = float(eigenvalues.sum() ** 2 / (eigenvalues ** 2).sum())
    return error / max(count, 1), effective_rank


def evaluate(sid_path, embeddings_path=None, checkpoint_path=None) -> dict:
    sid_path = Path(sid_path)
    if sid_path.is_dir():
        sid_path = sid_path / "item_sids.json"
    verdict = {"sid": str(sid_path), "allow": False, "reason": "", "collision": None,
               "quant_error": None, "effective_rank": None,
               "limits": {"collision": COLLISION_LIMIT,
                          "quant_error": [QUANT_ERROR_LOW, QUANT_ERROR_HIGH],
                          "effective_rank": EFFECTIVE_RANK_LIMIT}}
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
        error, effective_rank = measure_vectors(embeddings, weights, tables, tokens)
    except Exception as exc:  # noqa: BLE001
        verdict["reason"] = "vector measurement unavailable: " + repr(exc)
        return verdict
    verdict["quant_error"] = error
    verdict["effective_rank"] = effective_rank
    if error < QUANT_ERROR_LOW:
        verdict["reason"] = ("quantisation error %.3f is below %.2f: the codebook reproduces the "
                             "item vectors instead of learning a structure that generalises"
                             % (error, QUANT_ERROR_LOW))
        return verdict
    if error > QUANT_ERROR_HIGH:
        verdict["reason"] = ("quantisation error %.3f is above %.2f: the vectors are pressed "
                             "together" % (error, QUANT_ERROR_HIGH))
        return verdict
    if effective_rank > EFFECTIVE_RANK_LIMIT:
        verdict["reason"] = ("effective rank %.2f is above %.1f: the encoder carries the input "
                             "through unorganised and three codes cannot cover it"
                             % (effective_rank, EFFECTIVE_RANK_LIMIT))
        return verdict
    verdict["allow"] = True
    verdict["reason"] = ("collision %.4f < %.2f, quantisation error %.3f inside [%.2f, %.2f] and "
                         "effective rank %.2f <= %.1f"
                         % (collision, COLLISION_LIMIT, error, QUANT_ERROR_LOW, QUANT_ERROR_HIGH,
                            effective_rank, EFFECTIVE_RANK_LIMIT))
    return verdict


def main(argv: list) -> int:
    if len(argv) != 2:
        print(json.dumps({"allow": False, "reason": "usage: sid_quality <item_sids.json>"}))
        return 2
    print(json.dumps(evaluate(argv[1])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
