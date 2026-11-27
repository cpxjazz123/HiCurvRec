"""Materialise the hard q, encoder z and codes of the calibrated run (read-only)."""
from __future__ import annotations

import hashlib
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
ARM = "TIGER_calibrated_kappa_rq"
OUT = ROOT / "results/research/issue269/raw"
EXPECTED = 24587


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    run = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else None
    if run is None or not (run / "item_sids.json").is_file():
        raise SystemExit("usage: prepare_qz.py <run result dir containing item_sids.json>")
    sid_path = run / "item_sids.json"
    checkpoint = run / "rqvae_best.pth"
    sid = json.loads(sid_path.read_text())
    rows = np.asarray([sid[str(i)] for i in range(EXPECTED)], dtype=np.int64)
    if rows.shape != (EXPECTED, 4):
        raise ValueError("SID shape mismatch")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if int(payload.get("epoch", -1)) != 3000:
        raise ValueError("not the final epoch checkpoint")
    books = []
    for level in range(3):
        book = payload["state_dict"][f"rq.vq_layers.{level}.embed.weight"]
        books.append(book.detach().cpu().numpy().astype(np.float64))
    q = np.zeros((EXPECTED, 32), dtype=np.float64)
    for level, book in enumerate(books):
        q += book[rows[:, level]]
    scale = float(payload["state_dict"]["tangent_scale"])
    if not np.isfinite(q).all() or not np.isfinite(scale) or scale <= 0:
        raise ValueError("invalid q/scale")
    for name in list(sys.modules):
        if name.split(".")[0] in ("model", "losses", "distances", "train_config", "train_rqvae"):
            del sys.modules[name]
    source = run.parent / "src"
    sys.path.insert(0, str(source))
    training = importlib.import_module("train_rqvae")
    if Path(training.__file__).resolve() != source / "train_rqvae.py":
        raise ValueError("candidate import did not use the run source snapshot")
    model = training.RQVAE(training._tokenizer_config(), in_dim=768,
                           distance=training.DISTANCE).cpu()
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    embeddings = np.load(ROOT / "stage1_GeneEmbedding/output/sentence_t5.npy", allow_pickle=False)
    if embeddings.shape != (EXPECTED, 768):
        raise ValueError("embedding shape mismatch")
    z = np.empty((EXPECTED, 32), dtype=np.float64)
    with torch.inference_mode():
        for start in range(0, EXPECTED, 1024):
            block = torch.from_numpy(embeddings[start:start + 1024].astype(np.float32))
            z[start:start + 1024] = model.encoder(block).cpu().numpy().astype(np.float64)
            print("[issue269] encoded %d" % min(start + 1024, EXPECTED), flush=True)
    if not np.isfinite(z).all():
        raise ValueError("nonfinite z")
    OUT.mkdir(parents=True, exist_ok=True)
    for space, values in (("q", q), ("z", z)):
        path = OUT / f"{space}_TIGER_calibrated_kappa_rq.npz"
        np.savez_compressed(path, **{space: values.astype(np.float64)},
                            scale=np.array(scale), sid3=rows[:, :3].astype(np.int64))
        print("[issue269] wrote %s sha256=%s" % (path, sha(path)), flush=True)
    identity = {
        "schema": "genrec.issue269.prepare_qz/1",
        "run": str(run),
        "item_sids": str(sid_path), "item_sids_sha256": sha(sid_path),
        "checkpoint": str(checkpoint), "checkpoint_sha256": sha(checkpoint),
        "source_tree": str(source),
        "tangent_scale": scale,
        "sid_triple_collision": 1.0 - len({tuple(r[:3]) for r in rows.tolist()}) / EXPECTED,
    }
    OUT.joinpath("PREPARE.json").write_text(json.dumps(identity, indent=2, sort_keys=True) + "\n")
    print(json.dumps(identity, indent=1))


if __name__ == "__main__":
    main()
