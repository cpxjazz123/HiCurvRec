"""Pre-dispatch gates 1, 2 and 4 of issue #153 on the frozen 8192-row TRAIN probe."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

TREE = Path(__file__).resolve().parent
if str(TREE) not in sys.path:
    sys.path.insert(0, str(TREE))

from model import mixed_rq as mq  # noqa: E402
from model.mixed_rq import encode_tokens, split  # noqa: E402
from model.model import MixedRQVAE  # noqa: E402
from model.poincare_contract import expmap0  # noqa: E402

ROWS = 8192


def _root() -> Path:
    for parent in TREE.parents:
        if (parent / "stage2_RQ-VAE").is_dir():
            return parent
    raise RuntimeError("cannot locate the repository root")


def _hist(codes: torch.Tensor, codes_dim: int) -> dict:
    counts = torch.bincount(codes.long(), minlength=int(codes_dim)).double()
    return {"codes_used": int((counts > 0).sum()),
            "max_share": float(counts.max() / counts.sum().clamp_min(1.0))}


def main() -> int:
    import pandas as pd
    import train_rqvae as trainer

    root = _root()
    raw = np.load(root / "results" / "research" / "issue150" / "structure_prior.npz",
                  allow_pickle=True)
    item_layers = {int(i): int(l) for i, l in zip(raw["item_ids"], raw["layer"])}
    payload = json.loads((root / "results" / "research" / "issue153"
                          / "centroids.json").read_text())
    embeddings = np.load(root / "stage1_GeneEmbedding" / "output" / "sentence_t5.npy")
    frame = pd.read_parquet(root / "results" / "stage0_build_parquet" / "train.parquet",
                            columns=["target"])
    train_ids = np.unique(frame["target"].to_numpy(dtype=np.int64))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(trainer.SEED)
    model = MixedRQVAE(trainer._tokenizer_config(), in_dim=embeddings.shape[1],
                       curvature=1.0).to(device)
    trainer.initialize_tiger_weights(model)
    rows = torch.from_numpy(np.ascontiguousarray(embeddings.astype(np.float32)))
    model.init_codebook(rows[torch.from_numpy(train_ids)].to(device))
    model.attach_item_layers(item_layers)
    generator = torch.Generator().manual_seed(153)
    pick = torch.randperm(len(train_ids), generator=generator)[:ROWS].numpy()
    picked = train_ids[pick]
    groups = torch.tensor([item_layers.get(int(item), 0) for item in picked], dtype=torch.long,
                          device=device)
    batch = rows[torch.from_numpy(picked)].to(device)
    model.train()

    def run_codes(payload_in: dict | None, zero_alpha: bool = False) -> list:
        mq.SHIFT_OFF = payload_in is None
        model.rq.centroid_payload = None
        if payload_in is not None:
            table = dict(payload_in)
            if zero_alpha:
                table["alpha"] = {key: 0.0 for key in table["alpha"]}
            model.rq.attach_centroids(table)
        with torch.no_grad():
            model.rq(model.encoder(batch), groups=groups)
        mq.SHIFT_OFF = False
        return [codes.clone().cpu() for codes in model.rq.last_codes]

    baseline = run_codes(None)
    shifted = run_codes(payload)
    zeroed = run_codes(payload, zero_alpha=True)

    changed = float((baseline[-1] != shifted[-1]).float().mean())
    per_layer = {str(layer): float((baseline[-1][groups.cpu() == layer]
                                    != shifted[-1][groups.cpu() == layer]).float().mean())
                 for layer in range(4)}
    occupancy = {}
    for level in range(len(baseline)):
        base, moved = _hist(baseline[level], 256), _hist(shifted[level], 256)
        occupancy[f"level_{level + 1}"] = {
            "baseline": base, "shifted": moved,
            "codes_ok": moved["codes_used"] >= 0.8 * base["codes_used"],
            "share_ok": moved["max_share"] <= 2.0 * base["max_share"],
        }
    with torch.no_grad():
        exported = encode_tokens(model.encoder(batch), model.rq, groups=groups).cpu()
    checks = {
        "rows": ROWS,
        "gate1_changed_last_level": changed,
        "gate1_per_layer": per_layer,
        "gate1_pass": bool(changed >= 0.10),
        "gate2": occupancy,
        "gate2_pass": bool(all(row["codes_ok"] and row["share_ok"]
                               for row in occupancy.values())),
        "gate4_alpha_zero_bitwise": bool(all(torch.equal(a, b) for a, b in
                                            zip(baseline, zeroed))),
        "gate4_train_export_identical": bool(torch.equal(exported[:, -1], shifted[-1])),
        "alpha": payload["alpha"],
    }
    checks["passed"] = bool(checks["gate1_pass"] and checks["gate2_pass"]
                            and checks["gate4_alpha_zero_bitwise"]
                            and checks["gate4_train_export_identical"])
    out = root / "results" / "research" / "issue153" / "gate.json"
    out.write_text(json.dumps(checks, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: checks[key] for key in
                      ("gate1_changed_last_level", "gate1_pass", "gate2_pass",
                       "gate4_alpha_zero_bitwise", "gate4_train_export_identical",
                       "passed")}, indent=2))
    print(json.dumps(occupancy, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
