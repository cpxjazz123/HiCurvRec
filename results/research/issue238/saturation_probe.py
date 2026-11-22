"""Did the curved cost saturate, and is that why the codebook collapsed?

The mechanism maps both the latent and the codeword into the ball as
`0.9 * tanh(r / tau) * x / r` with `tau = mean codeword norm`.  When the encoder's radius
grows past a few tau the map is flat: every item lands on the same shell, the geodesic
distance between them goes to zero, and the assignment has nothing to rank.  This measures
that directly on the trained checkpoint instead of inferring it from the collision.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ARM = Path("stage2_RQ-VAE/TIGER_kappa_cost_rq")
RESULTS = Path("results/stage2_RQ-VAE/TIGER_kappa_cost_rq")


def main() -> int:
    sys.path.insert(0, str(ARM.resolve()))
    import train_rqvae as entry
    from model.model import RQVAE
    from distances import curved

    payload = torch.load(RESULTS / "rqvae_best.pth", map_location="cpu", weights_only=False)
    state = payload.get("state_dict", payload)
    model = RQVAE(entry._tokenizer_config(), in_dim=int(state["encoder.mlp.1.weight"].shape[1]))
    model.load_state_dict(state, strict=False)
    model.eval()

    vectors = torch.from_numpy(np.load(entry.EMBEDDING_FILE).astype(np.float32))
    vectors = torch.from_numpy(entry.maybe_apply_pca(vectors.numpy(), entry.PCA_DIM, entry.SEED))
    with torch.no_grad():
        latent = model.encoder(vectors)
        codes = model.rq.vq_layers[0].get_code_embs()
        tau = codes.norm(dim=1).mean().clamp_min(1e-6)
        latent_radius = latent.norm(dim=1)
        ratio = latent_radius / tau
        mapped = curved._to_ball(latent, tau)
        # Saturation: the mapped radius is flat when tanh(r/tau) is at its ceiling.
        squeeze = curved.RADIUS_CAP * torch.tanh(ratio)
    print(
        "tau=%.4f  latent_radius mean=%.3f p05=%.3f p95=%.3f  r/tau mean=%.2f p95=%.2f  "
        "tanh(r/tau) mean=%.4f p95=%.4f  saturation(p95-p05)/cap=%.4f"
        % (
            float(tau), float(latent_radius.mean()),
            float(latent_radius.quantile(0.05)), float(latent_radius.quantile(0.95)),
            float(ratio.mean()), float(ratio.quantile(0.95)),
            float(squeeze.mean()), float(squeeze.quantile(0.95)),
            float((squeeze.quantile(0.95) - squeeze.quantile(0.05)) / curved.RADIUS_CAP),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
