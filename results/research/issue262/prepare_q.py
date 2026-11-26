"""Issue #262: materialise actual hard-q coordinates from immutable checkpoints.

Read-only. Does not import a trainer, run a forward pass, update a parameter or
export a SID. The SID is the authoritative code identity for q.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/research/issue262"
RAW = OUT / "_raw" / "q"
ARMS = {
    100: "TIGER_gnn_attention_sectional_rq",
    101: "TIGER_three_expert_attention_rq",
    102: "TIGER_module_budget_prefix_rq",
    103: "TIGER_periodic_budget_prefix_rq",
    104: "TIGER_role_conditioned_l1_rq",
    105: "TIGER_role_conditioned_l3_rq",
    106: "TIGER_role_conditioned_l2_rq",
    107: "TIGER_topology_residual_l2_rq",
}
EXPECTED = 24587


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def find_final(arm: str) -> Path:
    paths = sorted((ROOT / "results/stage3_T5Train" / arm).glob("**/test_final.json"))
    if len(paths) != 1:
        raise RuntimeError(f"{arm}: expected one test_final")
    return paths[0]


def main(argv):
    import torch
    index = int(argv[1])
    arm = ARMS[index]
    final = find_final(arm)
    fd = json.loads(final.read_text())
    checkpoint = Path(fd["stage2_checkpoint"])
    sid_path = Path(fd["sid_path"])
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("epoch") != 3000:
        raise ValueError("not a final 3000-epoch checkpoint")
    selections = [payload[k] for k in ("selection", "checkpoint_selection") if k in payload]
    if not selections or any(v != "final_epoch" for v in selections):
        raise ValueError("not final_epoch")
    state = payload["state_dict"]
    sid = json.loads(sid_path.read_text())
    if set(sid) != {str(i) for i in range(EXPECTED)}:
        raise ValueError("SID catalogue identity mismatch")
    rows = np.asarray([sid[str(i)] for i in range(EXPECTED)], dtype=np.int64)
    if rows.shape != (EXPECTED, 4):
        raise ValueError("SID shape mismatch")
    books = []
    for level in range(3):
        key = f"rq.vq_layers.{level}.embed.weight"
        book = state.get(key)
        if book is None or tuple(book.shape) != (256, 32):
            raise ValueError("missing codebook " + key)
        book = book.detach().cpu().numpy().astype(np.float64)
        if not np.isfinite(book).all():
            raise ValueError("nonfinite codebook")
        books.append(book)
    q = np.zeros((EXPECTED, 32), dtype=np.float64)
    for level, book in enumerate(books):
        q += book[rows[:, level]]
    scale = float(state["tangent_scale"])
    level_scales = np.asarray(state.get("level_scales", np.ones(3)), dtype=np.float64)
    curvature_weight = float(state.get("curvature_weight", np.nan))
    if not np.isfinite(scale) or scale <= 0 or not np.isfinite(q).all():
        raise ValueError("invalid q or scale")
    RAW.mkdir(parents=True, exist_ok=True)
    path = RAW / f"{index}.npz"
    np.savez_compressed(path, q=q.astype(np.float32), sid3=rows[:, :3].astype(np.int32),
                        scale=np.array(scale), level_scales=level_scales,
                        curvature_weight=np.array(curvature_weight),
                        epoch=np.array(3000), sid_sha256=np.array(sha(sid_path)),
                        checkpoint_sha256=np.array(sha(checkpoint)),
                        arm=np.array(arm))
    print(json.dumps({"index": index, "arm": arm, "q": str(path),
                      "scale": scale, "sid_sha256": sha(sid_path),
                      "checkpoint_sha256": sha(checkpoint)}, indent=2))


if __name__ == "__main__":
    main(sys.argv)
