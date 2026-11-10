"""Where does the #154 prefix graph break? One batch, four checkpoints of requires_grad."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

TREE = Path(__file__).resolve().parent
if str(TREE) not in sys.path:
    sys.path.insert(0, str(TREE))

from model import prefix_geometry as pg  # noqa: E402
from model.mixed_rq import EUCLIDEAN_DIM, split  # noqa: E402
from model.model import MixedRQVAE  # noqa: E402
from model.poincare_contract import expmap0  # noqa: E402


def main() -> int:
    root = TREE.parents[1]
    import train_rqvae as trainer
    embeddings = np.load(root / "stage1_GeneEmbedding" / "output" / "sentence_t5.npy")
    torch.manual_seed(trainer.SEED)
    model = MixedRQVAE(trainer._tokenizer_config(), in_dim=embeddings.shape[1], curvature=1.0)
    trainer.initialize_tiger_weights(model)
    rows = torch.from_numpy(np.ascontiguousarray(embeddings[:512].astype(np.float32)))
    model.init_codebook(rows)
    batch = rows[:128]
    report = {
        "encoder_params_require_grad": [bool(p.requires_grad) for p in
                                        model.encoder.parameters()][:4],
        "encoder_params_all_require_grad": all(p.requires_grad for p in
                                               model.encoder.parameters()),
        "encoder_output_requires_grad": None, "after_expmap0": None,
        "pre_cost": None, "soft_prefix": None, "prefix_h_requires_grad": None,
        "grad_mode": torch.is_grad_enabled(),
    }
    model.train()
    encoded = model.encoder(batch)
    report["encoder_output_requires_grad"] = bool(encoded.requires_grad)
    z_e, z_t = split(encoded)
    ball = expmap0(z_t, model.rq.curvature)
    report["after_expmap0"] = bool(ball.requires_grad)
    layer = model.rq.levels[0]
    codes_e = layer.embed.weight[:, :EUCLIDEAN_DIM]
    codes_h = layer.embed.weight[:, EUCLIDEAN_DIM:]
    size = codes_e.shape[0]
    pre_cost = pg.mixed_distance(z_e[:, None, :].expand(z_e.shape[0], size, -1),
                                 ball[:, None, :].expand(ball.shape[0], size, -1),
                                 codes_e[None, :, :].expand(z_e.shape[0], -1, -1),
                                 codes_h[None, :, :].expand(ball.shape[0], -1, -1), 1.0)
    report["pre_cost"] = bool(pre_cost.requires_grad)
    prefix_e, prefix_h = pg.soft_prefix(pre_cost, codes_e, codes_h)
    report["soft_prefix"] = bool(prefix_e.requires_grad)
    report["prefix_h_requires_grad"] = bool(prefix_h.requires_grad)
    out = root / "results" / "research" / "issue154" / "grad_stages.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
