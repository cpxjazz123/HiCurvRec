"""Materialise q and z for a completed Stage2 run of the candidate."""
from __future__ import annotations

import hashlib
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
RAW = ROOT / "results/research/issue267/raw"
EXPECTED = 24587


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: prepare_run.py ARM")
    arm = sys.argv[1]
    candidates = sorted((ROOT / "results/stage2_RQ-VAE/_runs").glob(f"*/{arm}/result/rqvae_best.pth"),
                        key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise RuntimeError("no final checkpoint for " + arm)
    checkpoint = candidates[-1]
    result = checkpoint.parent
    sid_path = result / "item_sids.json"
    binding_path = result / "archive_binding.json"
    if not sid_path.is_file() or not binding_path.is_file():
        raise FileNotFoundError("missing SID/binding")
    binding = json.loads(binding_path.read_text())
    source = Path(binding["source_tree"])
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("epoch") != 3000:
        raise ValueError("not final epoch")
    if sha(checkpoint) != binding["checkpoint"]["sha256"]:
        raise ValueError("checkpoint hash mismatch")
    sid = json.loads(sid_path.read_text())
    rows = np.asarray([sid[str(i)] for i in range(EXPECTED)], dtype=np.int64)
    if rows.shape != (EXPECTED, 4):
        raise ValueError("SID shape mismatch")
    books = []
    for level in range(3):
        book = payload["state_dict"][f"rq.vq_layers.{level}.embed.weight"]
        if tuple(book.shape) != (256, 32):
            raise ValueError("codebook shape mismatch")
        books.append(book.detach().cpu().numpy().astype(np.float64))
    q = np.zeros((EXPECTED, 32), dtype=np.float64)
    for level, book in enumerate(books):
        q += book[rows[:, level]]
    scale = float(payload["state_dict"]["tangent_scale"])
    for name in list(sys.modules):
        if name.split(".")[0] in ("model", "losses", "distances", "train_config", "train_rqvae"):
            del sys.modules[name]
    sys.path.insert(0, str(source))
    training = importlib.import_module("train_rqvae")
    model = training.RQVAE(training._tokenizer_config(), in_dim=768,
                           distance=training.DISTANCE).cpu()
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    embeddings = np.load(ROOT / "stage1_GeneEmbedding/output/sentence_t5.npy", allow_pickle=False)
    z = np.empty((EXPECTED, 32), dtype=np.float64)
    with torch.inference_mode():
        for start in range(0, EXPECTED, 1024):
            end = min(start + 1024, EXPECTED)
            z[start:end] = model.encoder(torch.from_numpy(np.array(embeddings[start:end], copy=True))).cpu().numpy()
    if not all(np.isfinite(x).all() for x in (q, z)) or not np.isfinite(scale) or scale <= 0:
        raise ValueError("invalid representation")
    RAW.mkdir(parents=True, exist_ok=True)
    extra = {"sid3": rows[:, :3].astype(np.int32), "scale": np.array(scale),
             "checkpoint_sha256": np.array(sha(checkpoint)),
             "sid_sha256": np.array(sha(sid_path)), "arm": np.array(arm)}
    for name, value in (("q", q), ("z", z)):
        path = RAW / f"{name}_{arm}.npz"
        np.savez_compressed(path, **{name: value.astype(np.float32)}, **extra)
        print(name, path, sha(path))
    print(json.dumps({"arm": arm, "checkpoint": str(checkpoint),
                      "checkpoint_sha256": sha(checkpoint), "sid_sha256": sha(sid_path),
                      "scale": scale}, indent=2))


if __name__ == "__main__":
    main()
