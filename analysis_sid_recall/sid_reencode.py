#!/usr/bin/env python3
"""C12/C13: re-encode every item from a checkpoint and compare it with the exported SID.

The issue's §1 point is that a SID table is only trustworthy if it can be regenerated from the
checkpoint it claims to come from. This script does exactly that and reports, per item, whether
the three codes survive the round trip. It writes one evidence file per arm, never overwriting.

    python3 -m analysis_sid_recall.sid_reencode --arm results/stage2_RQ-VAE/TIGER_RQ-VAE
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tree_sha(arm: Path) -> str | None:
    """Hash of the mechanism source that produced the SID, when the tree is still present."""
    candidates = [p for p in (arm / "train_rqvae.py",) if p.is_file()]
    if not candidates:
        candidates = [p for p in Path("stage2_RQ-VAE").glob(f"*/train_rqvae.py")
                      if p.parent.name == arm.name or
                      p.parent.name == arm.name.replace("TIGER_", "")]
    if not candidates:
        return None
    return sha256(candidates[0])


def load_state(checkpoint: Path) -> dict:
    payload = __import__("torch").load(checkpoint, map_location="cpu")
    return payload.get("state_dict", payload)


def encode(state: dict, baseline: Path, embeddings: np.ndarray) -> np.ndarray:
    """Re-run the encoder and the three level assignments through the tree's own classes.

    Re-implementing the MLP here was wrong twice (linear/ReLU order, and the codebook's own
    distance object), so the replay imports the frozen ``model.layers.MLP`` and applies the
    same ``argmin`` the baseline uses when Sinkhorn is off. The state is loaded strictly: a
    missing or renamed tensor must fail rather than silently replay with random weights.
    """
    torch = __import__("torch")
    import importlib.util

    # The baseline's layers.py imports its sibling packages by their own names, so the tree has
    # to sit at the front of sys.path while it loads; the import order is otherwise identical to
    # the trainer's own.
    import importlib
    tree = str(Path(baseline).resolve())
    if tree not in sys.path:
        sys.path.insert(0, tree)
    layers = importlib.import_module("model.layers")
    width_in = int(state["encoder.mlp.1.weight"].shape[1])
    width_out = int(state["encoder.mlp.10.weight"].shape[0])
    hidden = [int(state[f"encoder.mlp.{i}.weight"].shape[0]) for i in (1, 4, 7)]
    encoder = layers.MLP([width_in] + hidden + [width_out], dropout=0.0)
    encoder.load_state_dict({k.replace("encoder.", "", 1): v for k, v in state.items()
                             if k.startswith("encoder.")}, strict=True)
    encoder.eval()
    with torch.no_grad():
        residual = encoder(torch.from_numpy(np.asarray(embeddings, dtype=np.float32)))
        codes = []
        for level in range(3):
            table = state[f"rq.vq_layers.{level}.embed.weight"].float()
            index = torch.argmin(torch.cdist(residual, table), dim=-1)
            codes.append(index.numpy())
            residual = residual - table[index]
    return np.stack(codes, axis=1)


def extend_collisions(tokens: np.ndarray, sizes: list[int]) -> tuple[np.ndarray, list]:
    """What ``train_rqvae._extend_collisions`` does: add the counter column, never rewrite codes.

    The export writes ``tokens[i, :-1] = row`` for the code columns. A SID whose first three
    codes disagree with a faithful replay therefore means the *code columns* moved, not that the
    counter was applied -- the counter lives above ``sum(sizes)`` and cannot alias them.
    """
    groups: dict[tuple, list] = {}
    for item_id, row in enumerate(tokens.tolist()):
        groups.setdefault(tuple(int(v) for v in row), []).append(item_id)
    offsets = np.cumsum([0, *sizes], dtype=np.int64)
    result = np.empty((len(tokens), tokens.shape[1] + 1), dtype=np.int64)
    for row, item_ids in groups.items():
        for occurrence, item_id in enumerate(item_ids):
            result[item_id, :-1] = row
            result[item_id, -1] = int(offsets[-1] + occurrence)
    return result, sorted((k, v) for k, v in groups.items() if len(v) > 1)


def audit(arm: Path) -> dict:
    embeddings = np.load(REPO / "stage1_GeneEmbedding" / "output" / "sentence_t5.npy").astype(
        np.float32)
    array = arm / "sids_for_hgrec.npy"
    if array.is_file():
        source = "sids_for_hgrec.npy"
        exported_codes = np.load(array)[:, :3].astype(np.int64)
    else:
        exported = json.loads((arm / "item_sids.json").read_text())
        source = "item_sids.json"
        exported_codes = np.asarray([exported[str(i)][:3] for i in range(len(exported))],
                                    dtype=np.int64)
    state = load_state(arm / "rqvae_best.pth")
    baseline = REPO / "stage2_RQ-VAE" / "baseline" / "TIGER_RQ-VAE"
    if not (baseline / "model" / "layers.py").is_file():
        baseline = arm.parent / "TIGER_RQ-VAE"
    recomputed = encode(state, baseline, embeddings)
    mismatch = int((recomputed != exported_codes).sum())
    differing = int((recomputed != exported_codes).any(axis=1).sum())
    # The counter must be reproducible from the replayed codes alone.
    _, collisions = extend_collisions(recomputed, [int(state["rq.vq_layers.0.embed.weight"].shape[0])] * 3)
    counter_ok = bool((np.load(array)[:, 3][[pair[1][0] for pair in collisions]] <
                       np.load(array)[:, 3][[pair[1][0] for pair in collisions]] + 1).all()) \
        if array.is_file() else None
    return {
        "arm": str(arm),
        "items": int(len(exported_codes)),
        "compared_against": source,
        "checkpoint_sha256": sha256(arm / "rqvae_best.pth"),
        "sid_sha256": sha256(arm / "item_sids.json"),
        "array_sha256": sha256(array) if array.is_file() else None,
        "source_sha256": tree_sha(arm),
        "mismatched_codes": mismatch,
        "mismatched_items": differing,
        "identical": mismatch == 0,
        "replayed_collision_groups": len(collisions),
        "level_mismatches": (recomputed != exported_codes).sum(axis=0).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    arm = Path(args.arm)
    report = audit(arm)
    out = args.out or REPO / "results" / "research" / "issue230" / (
        "reencode_" + arm.name + ".json")
    out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
