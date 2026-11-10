"""sigma_min of issue #157: half the initial spread of the normalised edge distances."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

TREE = Path(__file__).resolve().parent
if str(TREE) not in sys.path:
    sys.path.insert(0, str(TREE))

from model import item_index as ii  # noqa: E402
from model import rank_objective as ro  # noqa: E402
from model.model import MixedRQVAE  # noqa: E402

SIZE = 4096


def main() -> int:
    root = TREE.parents[1]
    import train_rqvae as trainer
    table = json.loads((root / "results" / "research" / "issue155" / "af3_edges.json").read_text())
    raw = np.load(root / "results" / "research" / "issue150" / "structure_prior.npz",
                  allow_pickle=True)
    item_ids = sorted(int(x) for x in raw["item_ids"])
    embeddings = np.load(root / "stage1_GeneEmbedding" / "output" / "sentence_t5.npy")
    rows = torch.from_numpy(np.ascontiguousarray(embeddings.astype(np.float32)))
    torch.manual_seed(trainer.SEED)
    model = MixedRQVAE(trainer._tokenizer_config(), in_dim=embeddings.shape[1], curvature=1.0)
    trainer.initialize_tiger_weights(model)
    model.init_codebook(rows[torch.from_numpy(np.asarray(item_ids[:4096]))])
    model.train()
    generator = torch.Generator().manual_seed(1570)
    pick = torch.randperm(len(item_ids), generator=generator)[:SIZE].numpy()
    batch_ids = np.asarray(item_ids)[pick]
    position = {int(item): index for index, item in enumerate(batch_ids)}
    edges = []
    for (tail, head) in table["edges"]:
        left, right = position.get(int(tail)), position.get(int(head))
        if left is not None and right is not None:
            edges.append((left, right))
    pairs = torch.tensor(edges, dtype=torch.long)
    with torch.no_grad():
        model(rows[torch.from_numpy(batch_ids)])
    per_level = []
    for prefix_e, prefix_h in model.rq.last_prefixes:
        rho = ro.normalise(ro.edge_distance(prefix_e, prefix_h, pairs, 1.0))
        per_level.append(float(rho.std(unbiased=False)))
    sigma_min = 0.5 * float(np.mean(per_level))
    payload = {"issue": 157, "batch": SIZE, "edges": len(edges),
               "rho_std_per_level": per_level, "mean_std": float(np.mean(per_level)),
               "sigma_min": sigma_min,
               "source_table_sha256": hashlib.sha256(
                   (root / "results" / "research" / "issue155" / "af3_edges.json").read_bytes()
               ).hexdigest()}
    out = root / "results" / "research" / "issue157" / "sigma_min.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    out.write_text(text, encoding="utf-8")
    payload["file_sha256"] = hashlib.sha256(text.encode()).hexdigest()
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
