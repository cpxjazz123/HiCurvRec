"""Does the evaluation path leak? Repeated get_indices calls with RSS printed after each (issue #167)."""
from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch

TREE = Path(__file__).resolve().parent
if str(TREE) not in sys.path:
    sys.path.insert(0, str(TREE))


def rss_mb() -> float:
    with open("/proc/self/status") as handle:
        for line in handle:
            if line.startswith("VmRSS:"):
                return float(line.split()[1]) / 1024.0
    return 0.0


def main() -> int:
    root = TREE.parents[1]
    import train_rqvae as trainer
    from model import item_index as ii
    from model.model import MixedRQVAE
    embeddings = np.load(root / "stage1_GeneEmbedding" / "output" / "sentence_t5.npy")
    rows = torch.from_numpy(np.ascontiguousarray(embeddings.astype(np.float32)))
    kappa = json.loads((root / "results" / "research" / "issue167" / "kappa.json").read_text())["kappa"]
    torch.manual_seed(trainer.SEED)
    model = MixedRQVAE(trainer._tokenizer_config(), in_dim=embeddings.shape[1], curvature=1.0)
    trainer.initialize_tiger_weights(model)
    model.init_codebook(rows[:4096])
    model.attach_kappa(kappa, ii.build(embeddings))
    model.eval()
    batch = rows[:8192]
    readings = []
    for repeat in range(4):
        with torch.no_grad():
            model.get_indices(batch)
        gc.collect()
        readings.append({"repeat": repeat, "rss_mb": round(rss_mb(), 1),
                         "prefixes_held": len(getattr(model.rq, "last_prefixes", []))})
        print(readings[-1], flush=True)
    payload = {"rows": int(batch.shape[0]), "readings": readings,
               "growth_mb": round(readings[-1]["rss_mb"] - readings[0]["rss_mb"], 1)}
    out = root / "results" / "research" / "issue167" / "rss_probe.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
