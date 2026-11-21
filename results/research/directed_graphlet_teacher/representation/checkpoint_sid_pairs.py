#!/usr/bin/env python3
"""Does a checkpoint reproduce the SID table stored next to it?

A SID table is only evidence of a training run if the checkpoint in the same directory,
fed the frozen input, produces those codes.  Anything else means the pair was assembled
by copying files, and any SID-to-recall conclusion for that arm rests on an unverified
pairing.

Reads only frozen artifacts.  Writes ``representation/checkpoint_sid_pairs.json``.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
TEACHER = HERE.parent
REPO = next(parent for parent in HERE.parents
            if (parent / "stage2_RQ-VAE").is_dir() and (parent / "results").is_dir())
sys.path.insert(0, str(HERE))

from export_latent_q import load_model, SEMANTICS  # noqa: E402

RESULTS = REPO / "results" / "stage2_RQ-VAE"
OUT = HERE / "checkpoint_sid_pairs.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stored_codes(path: Path, n_items: int) -> np.ndarray | None:
    try:
        raw = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    codes = np.full((n_items, 3), -1, dtype=np.int64)
    seen = 0
    for key, value in raw.items():
        item = int(key)
        if 0 <= item < n_items and isinstance(value, list) and len(value) >= 3:
            codes[item] = [int(v) for v in value[:3]]
            seen += 1
    if seen != n_items:
        return None
    return codes


def codes_of(model, features: np.ndarray, device) -> np.ndarray:
    with torch.no_grad():
        x = torch.from_numpy(features).to(device)
        encoded = model.encoder(x)
        _, _, _, tokens = model.rq(encoded, infer_use_sk=False)
    return tokens.cpu().numpy().astype(np.int64)[:, :3]


def strict_match(recomputed: np.ndarray, stored: np.ndarray) -> dict:
    """Token-agreement and full-SID agreement reported separately.

    Token agreement counts all three levels at once; a full SID only matches when every
    item's three codes match exactly.  A 99.9% token rate can still leave many items with
    a different SID, so both numbers are given and MATCH requires 100%.
    """
    per_item = (recomputed == stored).all(axis=1)
    return {"token_agree_fraction": float((recomputed == stored).mean()),
            "full_sid_agree_fraction": float(per_item.mean()),
            "items_exactly_matching": int(per_item.sum()),
            "items": int(len(recomputed))}


def arm_space(arm_dir: Path) -> dict:
    """How far the stored SID's first-level code space can be verified at all.

    The baseline model is rebuilt with the frozen config; an arm whose own run used a
    different metric, width or codebook cannot be reproduced that way, so the check says
    which part of the comparison is not established rather than assuming it.
    """
    notes = {"metric": "euclidean (baseline default; the arm's real metric is not "
                       "recovered by this script)"}
    config = arm_dir / "export_manifest.json"
    if config.is_file():
        try:
            notes["export_manifest"] = json.loads(config.read_text())
        except json.JSONDecodeError:
            notes["export_manifest"] = "UNREADABLE"
    else:
        notes["export_manifest"] = "ABSENT"
    ckpt = arm_dir / "rqvae_best.pth"
    if ckpt.is_file():
        try:
            payload = torch.load(ckpt, map_location="cpu")
            held = payload.get("selection") or {}
            notes["checkpoint_selection"] = held
            notes["checkpoint_keys"] = sorted(payload)[:8]
        except Exception as exc:  # noqa: BLE001
            notes["checkpoint_read"] = f"{type(exc).__name__}"
    return notes


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features = np.load(SEMANTICS).astype(np.float32)
    n_items = len(features)
    report = {"semantics_sha256": sha256(SEMANTICS), "arms": {}}
    for arm_dir in sorted(RESULTS.iterdir()):
        if not arm_dir.is_dir():
            continue
        checkpoint = arm_dir / "rqvae_best.pth"
        table = arm_dir / "item_sids.json"
        if not checkpoint.is_file() or not table.is_file():
            continue
        stored = stored_codes(table, n_items)
        if stored is None:
            report["arms"][arm_dir.name] = {"status": "UNREADABLE_SID_TABLE",
                                            "sid_sha256": sha256(table)}
            continue
        try:
            model, _ = load_model(arm_dir, features.shape[1], device)
            recomputed = codes_of(model, features, device)
        except Exception as exc:  # noqa: BLE001 - a shape mismatch is a finding, not a crash
            report["arms"][arm_dir.name] = {"status": "CHECKPOINT_UNUSABLE",
                                            "error": f"{type(exc).__name__}: {exc}"[:300],
                                            "sid_sha256": sha256(table),
                                            "checkpoint_sha256": sha256(checkpoint)}
            continue
        agreement = strict_match(recomputed, stored)
        exact = agreement["full_sid_agree_fraction"] == 1.0
        report["arms"][arm_dir.name] = {
            "status": "MATCH" if exact else "MISMATCH",
            **agreement,
            "checkpoint_sha256": sha256(checkpoint),
            "sid_sha256": sha256(table),
            "arm_space": arm_space(arm_dir)}
        print(f"{arm_dir.name}: tokens {agreement['token_agree_fraction']:.4f} "
              f"full SID {agreement['full_sid_agree_fraction']:.4f}", flush=True)
        OUT.write_text(json.dumps(report, indent=1) + "\n")
    OUT.write_text(json.dumps(report, indent=1) + "\n")
    matched = sum(1 for v in report["arms"].values() if v.get("status") == "MATCH")
    print(f"{matched} MATCH / {len(report['arms'])} arms checked; written {OUT}")


if __name__ == "__main__":
    main()
