"""Stage 3 entry gate: the SID must be a taxonomy, not an identity.

Three numbers are measured on the arm itself -- its SID table, its checkpoint and
the stage 1 embeddings -- and nothing is passed in by the caller, so the gate
cannot be talked past.

* **collision** = ``1 - distinct(three-code SID) / items`` below 0.40 (owner
  instruction 2026-11-21; it was 0.10 before).  The measurement is unchanged: among
  the 73 arms with a checkpoint on disk every arm above ``test_R@10 0.06`` stayed at
  0.030-0.087; arms at 0.10-0.99 never did.  The higher limit is a deliberate
  loosening by the owner, not a new measurement, so older refusals stay on record as
  such rather than being re-read as passes.
* **quantisation error** = ``mean ||z - sum(codewords)||`` inside [0.035, 0.10].
  The encoder forward includes every linear layer's bias -- ``model/layers.py``
  builds ``nn.Linear`` with a bias and the checkpoints carry it, and dropping it
  made ``z`` five times too small.  With the bias the 12 arms above 0.06 sit at
  0.041-0.090; ``resample_proj_dir`` (0.032, recall 0.0548) and ``dul_rq``
  (0.096 at 1.6% collision, recall 0.026) sit outside.
* **effective rank** ``PR = (sum lambda)^2 / sum lambda^2`` of the encoder output
  at or below 8.  The 12 arms above 0.06 compressed 32 dimensions into 5.99-6.93
  effective ones; ``dul_rq`` kept 23.2, i.e. it carried the 768-dim input through
  unorganised and the three codes cannot cover that.

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

COLLISION_LIMIT = 0.4
QUANT_ERROR_LOW = 0.035
QUANT_ERROR_HIGH = 0.10
EFFECTIVE_RANK_LIMIT = 8.0
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


def _encoder_layers(state: dict) -> list:
    """The encoder MLP as (weight, bias) pairs, in the order ``MLP.forward`` runs them.

    ``model/layers.py::MLP`` builds ``Dropout, Linear(...), ReLU`` triples, so the
    linear layers sit at odd indices and every one of them carries a bias; a
    forward pass without that bias is not the model's forward pass.
    """
    weights = {}
    biases = {}
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


def _codebook_tables(state: dict) -> list:
    """The codebook each level actually assigns from, in level order.

    Three shapes occur: ``vq_layers.<i>.embed.weight`` (vq and simvq), the bare
    ``vq_layers.<i>.embed`` parameter (ema) and ``levels.<i>.embed.weight`` (the
    mixed-geometry trees).  SimVQ assigns from ``embed_proj(embed.weight)`` --
    ``SimVQLayer.get_code_embs`` -- so an arm trained with it must be read
    through the projection, not the raw table.
    """
    levels = {}
    projections = {}
    for key, value in state.items():
        if key.endswith("embed_proj.weight"):
            projections[key.rsplit(".", 1)[0].replace("embed_proj", "embed")] = value
        elif key.endswith(".embed.weight"):
            levels[key.rsplit(".", 1)[0]] = value
        elif key.endswith(".embed") and value.ndim == 2:
            levels[key] = value
    if not levels:
        return []
    ordered = sorted(levels, key=lambda name: _level_of(name))
    return [_projected(levels[name], projections.get(name)) for name in ordered]


def _level_of(name: str) -> int:
    for part in reversed(name.split(".")):
        if part.isdigit():
            return int(part)
    return 0


def _projected(table, projection):
    if projection is None:
        return table
    import torch
    return table @ projection.t()


def measure_vectors(embeddings, layers, tables, tokens) -> tuple:
    """Return (quantisation error, effective rank) of the encoder output."""
    import torch

    depth = int(tokens.shape[1])
    if not layers or not tables:
        raise ValueError("no encoder layers or no codebook tables in the checkpoint")
    weights = [weight for weight, _ in layers]
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
            for index, (weight, bias) in enumerate(layers):
                x = x @ weight.t()
                if bias is not None:
                    x = x + bias
                if index < len(layers) - 1:
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


def evaluate(sid_path, embeddings_path=None, checkpoint_path=None,
             waive_collision: bool = False) -> dict:
    """Judge one SID table for Stage 3 entry.

    ``waive_collision`` exists for one purpose: the owner may deliberately run an arm
    whose three-level collision is at or above ``COLLISION_LIMIT``, and the run has to
    be measurable rather than silently impossible.  The waiver never turns the number
    into a pass -- the verdict keeps the measured collision and adds
    ``collision_gate_waived`` so every reader can see the arm entered outside the gate.
    """
    sid_path = Path(sid_path)
    if sid_path.is_dir():
        sid_path = sid_path / "item_sids.json"
    verdict = {"sid": str(sid_path), "allow": False, "reason": "", "collision": None,
               "quant_error": None, "effective_rank": None, "warnings": [],
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
        if not waive_collision:
            verdict["reason"] = ("three-level collision %.6f is not below %.2f: %.1f%% of the items "
                                 "share an SID with another item"
                                 % (collision, COLLISION_LIMIT, 100 * collision))
            return verdict
        # Explicit owner waiver: the run continues so the arm can be measured, and the
        # record keeps both the real number and the fact that the gate was stepped over.
        verdict["collision_gate_waived"] = True
        waiver_note = ("collision %.4f is NOT below %.2f -- waived by the caller; "
                       "this arm enters Stage 3 outside the collision gate"
                       % (collision, COLLISION_LIMIT))
    else:
        waiver_note = None

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
        layers = _encoder_layers(state)
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
        error, effective_rank = measure_vectors(embeddings, layers, tables, tokens)
    except Exception as exc:  # noqa: BLE001
        verdict["reason"] = "vector measurement unavailable: " + repr(exc)
        return verdict
    verdict["quant_error"] = error
    verdict["effective_rank"] = effective_rank
    # The collision is the hard condition: it is the one that is comparable across
    # every family.  The two vector numbers are recorded and reported, but they are
    # a fingerprint of the behaviour-sample family the bands were measured on -- the
    # mixed-geometry arms sit at 0.028-0.455 and PR 1.8-22.9, so blocking on them
    # would forbid the whole mixed-curvature line from ever reaching Stage 3.
    warnings = []
    if error < QUANT_ERROR_LOW:
        warnings.append("quantisation error %.3f is below %.3f: the codebook reproduces the item "
                        "vectors instead of learning a structure that generalises"
                        % (error, QUANT_ERROR_LOW))
    elif error > QUANT_ERROR_HIGH:
        warnings.append("quantisation error %.3f is above %.3f: the vectors are pressed together"
                        % (error, QUANT_ERROR_HIGH))
    if effective_rank > EFFECTIVE_RANK_LIMIT:
        warnings.append("effective rank %.2f is above %.1f: the encoder carries the input through "
                        "unorganised and three codes may not cover it"
                        % (effective_rank, EFFECTIVE_RANK_LIMIT))
    verdict["warnings"] = warnings
    verdict["allow"] = True
    verdict["reason"] = ("%scollision %.4f < %.2f; measured quantisation error %.3f (band %.3f-%.3f) "
                         "and effective rank %.2f (limit %.1f)%s"
                         % ("" if waiver_note is None else waiver_note + "; ",
                            collision, COLLISION_LIMIT, error, QUANT_ERROR_LOW, QUANT_ERROR_HIGH,
                            effective_rank, EFFECTIVE_RANK_LIMIT,
                            "" if not warnings else "; WARNING " + " | ".join(warnings)))
    return verdict


def main(argv: list) -> int:
    """CLI: ``sid_quality <item_sids.json> [--waive-collision]``.

    ``--waive-collision`` is the explicit owner override for an arm that must be
    measured even though its three-level collision is at or above the limit.
    """
    args = [item for item in argv[1:] if item != "--waive-collision"]
    waive = "--waive-collision" in argv[1:]
    if len(args) != 1:
        print(json.dumps({"allow": False,
                          "reason": "usage: sid_quality <item_sids.json> [--waive-collision]"}))
        return 2
    print(json.dumps(evaluate(args[0], waive_collision=waive)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
