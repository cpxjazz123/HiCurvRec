"""Gates 5-7 of issue #159: curvature coupling, STRUCTURE_OFF, and train/export agreement."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

TREE = Path(__file__).resolve().parent
if str(TREE) not in sys.path:
    sys.path.insert(0, str(TREE))

from model import rank_objective as ro  # noqa: E402
from model.model import MixedRQVAE  # noqa: E402
from model.structure_runtime import StructureSupervision  # noqa: E402

SIZE = 4096


def main() -> int:
    root = TREE.parents[1]
    import train_rqvae as trainer
    raw = np.load(root / "results" / "research" / "issue150" / "structure_prior.npz",
                  allow_pickle=True)
    item_ids = sorted(int(x) for x in raw["item_ids"])
    embeddings = np.load(root / "stage1_GeneEmbedding" / "output" / "sentence_t5.npy")
    rows = torch.from_numpy(np.ascontiguousarray(embeddings.astype(np.float32)))
    supervision = StructureSupervision.from_repo(root, 0.48247611144034613)
    torch.manual_seed(trainer.SEED)
    model = MixedRQVAE(trainer._tokenizer_config(), in_dim=embeddings.shape[1], curvature=1.0)
    trainer.initialize_tiger_weights(model)
    model.init_codebook(rows[torch.from_numpy(np.asarray(item_ids[:4096]))])
    generator = torch.Generator().manual_seed(15990)
    pick = torch.randperm(len(item_ids), generator=generator)[:SIZE].numpy()
    batch_ids = np.asarray(item_ids)[pick]
    batch = rows[torch.from_numpy(batch_ids)]
    model.train()
    model(batch)
    curved = supervision.terms(model, batch, 1.0)
    flat = supervision.terms(model, batch, 0.0)
    ro.STRUCTURE_OFF = True
    model(batch)
    off_codes = torch.stack(model.rq.last_codes, dim=1).clone()
    ro.STRUCTURE_OFF = False
    model(batch)
    on_codes = torch.stack(model.rq.last_codes, dim=1).clone()
    export = model.get_indices(batch)
    report = {
        "rows": SIZE,
        "gate6_curved": float(curved) if curved is not None else None,
        "gate6_flat": float(flat) if flat is not None else None,
        "gate6_difference": (float(abs(float(curved) - float(flat))) if curved is not None
                             and flat is not None else None),
        "gate7_structure_off_bitwise": bool(torch.equal(off_codes, on_codes)),
        "gate7_train_export_agreement_per_level": [
            float((on_codes[:, level] == export[:, level]).float().mean()) for level in range(3)],
    }
    report["gate6_pass"] = bool(report["gate6_difference"] is not None
                                and report["gate6_difference"] > 1e-6)
    target = root / "results" / "research" / "issue159" / "gate_curve_export.json"
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
