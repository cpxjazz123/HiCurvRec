"""How close did the 300-epoch guarded run get to the margin threshold?

The guard refuses a step when more than half of the items pick their codeword by a tie.
The run completing means the threshold was never crossed; this reports the distance to it.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TREE = HERE / "trees" / "MARGIN_300"
sys.path.insert(0, str(TREE))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import train_rqvae as trainer  # noqa: E402


def main() -> None:
    embeddings = np.load(trainer.EMBEDDING_FILE).astype("float32")
    tensor = torch.tensor(embeddings, device="cuda" if torch.cuda.is_available() else "cpu")
    model = trainer.RQVAE(
        trainer._tokenizer_config(), in_dim=int(embeddings.shape[1]),
        distance=trainer.DISTANCE,
    ).to(tensor.device)
    state = torch.load(HERE / "run" / "rqvae_best.pth", map_location=tensor.device)
    model.load_state_dict(state.get("state_dict", state))
    model.eval()
    with torch.no_grad():
        encoded = model.encoder(tensor)
        for level, layer in enumerate(model.rq.vq_layers, start=1):
            cost = layer._distances(encoded)
            two = torch.topk(-cost.float(), 2, dim=-1).values
            margin = (two[:, 0] - two[:, 1]).clamp_min(0)
            scale = two[:, 0].abs().clamp_min(1.0)
            ties = float((margin <= 1e-6 * scale).float().mean())
            print(f"level {level}: tie fraction {ties:.6f}  "
                  f"median margin {float(margin.median()):.6f}  "
                  f"min margin {float(margin.min()):.6f}")


if __name__ == "__main__":
    main()
