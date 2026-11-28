#!/usr/bin/env python3
"""Scan the completed Stage2 SIDs and classify their Stage3 eligibility.

Read-only: for every results/stage2_RQ-VAE/**/item_sids.json it records the arm, the SID's
collision (from the codes), the checkpoint's run identity, the Stage3 directory state, and the
priority order (the SID's modification time).  No GPU, no training.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "results/research/stage3_queue"


def main() -> None:
    rows = []
    for sid_path in sorted(REPO.glob("results/stage2_RQ-VAE/**/item_sids.json")):
        parts = sid_path.parts
        arm = parts[parts.index("_runs") + 2] if "_runs" in parts else parts[-2]
        payload = json.loads(sid_path.read_text())
        codes = np.asarray([payload[str(i)] for i in range(len(payload))], dtype=np.int64)[:, :3]
        collision = float(1.0 - len(np.unique(codes, axis=0)) / len(codes))
        checkpoint = sid_path.parent / "rqvae_best.pth"
        identity = None
        if checkpoint.is_file():
            try:
                blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
                identity = blob.get("run_id") if isinstance(blob, dict) else None
            except Exception as exc:                     # a corrupt/unreadable checkpoint
                identity = f"UNREADABLE:{type(exc).__name__}"
        stage3 = REPO / "results/stage3_T5Train" / arm
        if any(stage3.glob("logs/*/*/test_final.json")):
            state = "DONE"
        elif any(stage3.glob("ckpt/*")) or (stage3 / "logs").is_dir():
            state = "PARTIAL"
        elif stage3.is_dir():
            state = "EMPTY"
        else:
            state = "MISSING"
        rows.append({
            "arm": arm,
            "sid": str(sid_path.relative_to(REPO)),
            "sid_mtime": sid_path.stat().st_mtime,
            "collision": collision,
            "checkpoint": str(checkpoint.relative_to(REPO)),
            "identity": bool(identity) and not str(identity).startswith("UNREADABLE"),
            "identity_value": identity,
            "stage3_state": state,
        })
    rows.sort(key=lambda r: r["sid_mtime"], reverse=True)
    eligible = [r for r in rows if r["stage3_state"] != "DONE" and r["identity"] and r["collision"] < 0.4]
    payload = {"schema": "genrec.stage3_queue/1", "items": rows,
               "summary": {
                   "total": len(rows),
                   "done": sum(1 for r in rows if r["stage3_state"] == "DONE"),
                   "missing": sum(1 for r in rows if r["stage3_state"] == "MISSING"),
                   "partial": sum(1 for r in rows if r["stage3_state"] == "PARTIAL"),
                   "eligible_now": len(eligible),
               },
               "eligible_order": [r["sid"] for r in eligible]}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "queue.json").write_text(json.dumps(payload, indent=1) + "\n")
    print(json.dumps(payload["summary"], indent=1))
    for r in eligible[:12]:
        print("NEXT", r["arm"], "coll", round(r["collision"], 4), r["sid"])


if __name__ == "__main__":
    main()
