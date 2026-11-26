"""Issue #267: materialise hard q and encoder z for the frozen #265 checkpoint.

Read-only: no optimizer, no parameter update, no SID rewrite. The hard q is the
sum of the three selected codebooks at the exported SID; z is the encoder output.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "results/research/issue267/raw"
ARM = "TIGER_euclid_local_curve_supervision_rq"
EXPECTED = 24587


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    finals = sorted((ROOT / "results/stage3_T5Train" / ARM).glob("**/test_final.json"))
    if len(finals) != 1:
        raise RuntimeError("expected one #265 test_final")
    final = finals[0]
    fd = json.loads(final.read_text())
    checkpoint = Path(fd["stage2_checkpoint"])
    sid_path = Path(fd["sid_path"])
    binding = json.loads(Path(fd["archive_binding"]).read_text())
    source = Path(binding["source_tree"])
    if sha(checkpoint) != binding["checkpoint"]["sha256"]:
        raise ValueError("checkpoint hash mismatch")
    if sha(sid_path) != binding["sid"]["sha256"]:
        raise ValueError("SID hash mismatch")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("epoch") != 3000:
        raise ValueError("not final epoch")
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
    if not np.isfinite(q).all() or not np.isfinite(scale) or scale <= 0:
        raise ValueError("invalid q/scale")
    for name in list(sys.modules):
        if name.split(".")[0] in ("model", "losses", "distances", "train_config", "train_rqvae"):
            del sys.modules[name]
    sys.path.insert(0, str(source))
    training = importlib.import_module("train_rqvae")
    if Path(training.__file__).resolve() != source / "train_rqvae.py":
        raise ValueError("candidate import did not use #265 source")
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
            end = min(start + 1024, EXPECTED)
            z[start:end] = model.encoder(torch.from_numpy(np.array(embeddings[start:end], copy=True))).cpu().numpy()
    if not np.isfinite(z).all():
        raise ValueError("nonfinite z")
    OUT.mkdir(parents=True, exist_ok=True)
    q_file = OUT / "q_109.npz"
    z_file = OUT / "z_109.npz"
    np.savez_compressed(q_file, q=q.astype(np.float32), sid3=rows[:, :3].astype(np.int32),
                        scale=np.array(scale), sid_sha256=np.array(sha(sid_path)),
                        checkpoint_sha256=np.array(sha(checkpoint)), arm=np.array(ARM))
    np.savez_compressed(z_file, z=z.astype(np.float32), sid3=rows[:, :3].astype(np.int32),
                        scale=np.array(scale), sid_sha256=np.array(sha(sid_path)),
                        checkpoint_sha256=np.array(sha(checkpoint)), arm=np.array(ARM))
    print(json.dumps({"arm": ARM, "checkpoint_sha256": sha(checkpoint),
                      "sid_sha256": sha(sid_path), "scale": scale,
                      "q_file": str(q_file), "q_sha256": sha(q_file),
                      "z_file": str(z_file), "z_sha256": sha(z_file)}, indent=2))


if __name__ == "__main__":
    main()
