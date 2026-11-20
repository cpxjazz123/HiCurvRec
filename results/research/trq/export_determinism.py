"""CGV item 5: the same checkpoint exported twice, in two processes, must agree.

Two independent interpreter runs load rqvae_best.pth, quantise the frozen
embeddings and extend collisions; their SID tables and the archived item_sids.json
must be identical, otherwise the checkpoint is not sufficient provenance.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def worker(arm: str, out: Path) -> None:
    tree = REPO / "stage2_RQ-VAE" / f"TIGER_trq_{arm}"
    sys.path.insert(0, str(tree))
    import torch

    import train_rqvae as T
    from model import RQVAE

    T.set_seed(T.SEED)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    embeddings = T.load_embeddings(T.EMBEDDING_FILE)
    model = RQVAE(T._tokenizer_config(), in_dim=embeddings.shape[1],
                  distance=T.build_distance(T.CONFIG)).to(device)
    state = torch.load(T.CHECKPOINT, map_location="cpu")["state_dict"]
    model.load_state_dict(state)
    model.eval()
    for layer in model.rq.vq_layers:
        layer._skip_ddp_reduce = True
    with torch.no_grad():
        tokens = model.get_indices(torch.from_numpy(embeddings).to(device))
    sid = T._extend_collisions(tokens.cpu().numpy().astype(np.int64),
                               list(model.config.codebook_size))
    np.save(out, sid)


def main(arm: str) -> None:
    result_dir = REPO / "results" / "stage2_RQ-VAE" / f"TIGER_trq_{arm}"
    if not (result_dir / "rqvae_best.pth").is_file():
        print(f"arm {arm}: no checkpoint yet")
        return
    outputs = []
    for run in (1, 2):
        out = HERE / f"export_{arm}_run{run}.npy"
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", arm, str(out)],
                       check=True, cwd=str(REPO))
        outputs.append(out)
    first, second = (np.load(path) for path in outputs)
    archived = np.array([json.loads((result_dir / "item_sids.json").read_text())[str(i)]
                         for i in range(len(first))], dtype=np.int64)
    verdict = {
        "arm": arm,
        "run1_vs_run2_mismatch": int((first != second).sum()),
        "run1_vs_archive_mismatch": int((first != archived).sum()),
        "sid_sha256": hashlib.sha256(archived.tobytes()).hexdigest(),
        "identical": bool((first == second).all() and (first == archived).all()),
    }
    (HERE / f"determinism_{arm}.json").write_text(json.dumps(verdict, indent=1))
    for path in outputs:
        path.unlink()
    print(json.dumps(verdict))


if __name__ == "__main__":
    if sys.argv[1] == "--worker":
        worker(sys.argv[2], Path(sys.argv[3]))
    else:
        main(sys.argv[1])
