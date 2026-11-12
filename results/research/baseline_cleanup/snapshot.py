"""Pin the baseline's numbers, so removing dead code can be shown to change nothing.

The baseline cannot be trained here (3000 epochs, four cards), so the check is a
seeded, CPU-only forward/backward/optimiser run on the real model and the real
loss. Every tensor that matters is hashed: the loss, the reconstruction loss, the
exported tokens, every parameter after three steps, and every gradient.

KMeans initialises the codebooks with no fixed seed, so `init_codebook` is skipped
and the codebooks are filled from a seeded generator instead: the comparison is
between two states of the same source tree, not between two random initialisations.

    /home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 snapshot.py [output.json]
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

TREE = Path(__file__).resolve().parents[3] / "stage2_RQ-VAE" / "TIGER_RQ-VAE"
REPO = TREE.parents[1]
sys.path.insert(0, str(TREE))
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from model import RQVAE  # noqa: E402
import train_rqvae as trainer  # noqa: E402


def digest(tensor) -> str:
    array = tensor.detach().to(torch.float64).cpu().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def build():
    torch.manual_seed(7)
    model = RQVAE(trainer._tokenizer_config(), in_dim=32)
    generator = torch.Generator().manual_seed(11)
    with torch.no_grad():
        for layer in model.rq.vq_layers:
            weight = layer.get_code_embs()
            weight.copy_(torch.randn(weight.shape, generator=generator))
    return model


def main() -> int:
    model = build()
    generator = torch.Generator().manual_seed(13)
    batch = torch.randn(48, 32, generator=generator)
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3)
    record = {"imports": {}, "steps": []}
    record["imports"]["model.utils"] = "model.utils" in sys.modules
    record["imports"]["model.poincare_contract"] = "model.poincare_contract" in sys.modules
    for step in range(3):
        model.train()
        optimiser.zero_grad(set_to_none=True)
        reconstructed, quant_loss, unused_codes, tokens = model(batch)
        loss, recon_loss = model.compute_loss(batch, reconstructed, quant_loss)
        loss.backward()
        grads = {name: digest(p.grad) for name, p in model.named_parameters()
                 if p.grad is not None}
        optimiser.step()
        params = {name: digest(p) for name, p in model.named_parameters()}
        record["steps"].append({
            "step": step,
            "loss": float(loss),
            "recon_loss": float(recon_loss),
            "quant_loss": float(quant_loss),
            "unused_codes": int(unused_codes),
            "tokens": digest(tokens.float()),
            "grads": hashlib.sha256(json.dumps(grads, sort_keys=True).encode()).hexdigest(),
            "params": hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest(),
        })
    model.eval()
    exported = model.get_indices(batch)
    record["export_tokens"] = digest(exported.float())
    record["export_shape"] = list(exported.shape)
    record["param_count"] = sum(p.numel() for p in model.parameters())
    record["module_files"] = sorted(str(p.relative_to(TREE)) for p in (TREE / "model").glob("*.py"))

    text = json.dumps(record, indent=2, sort_keys=True)
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if target is not None:
        target.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
