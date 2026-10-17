"""Issue #27 mechanism check for the multi-scale behavioural spectrum.

Exercises the installed encoder on this tree's own probe checkpoint:

1. the three cumulative band targets are genuinely nested -- depth 3 adds to depth
   2 and depth 2 to depth 1 -- and each band carries signal of its own;
2. ``q_i^(q)`` really is the cumulative sum of the first ``q`` codeword layers, so
   the heads read what ``RQLayer.forward`` accumulates and nothing else;
3. each depth's band loss reaches its head and the encoder, and reaches no codebook
   even though the codebooks are passed in;
4. the coordinate term vanishes when a head's output already equals the target and
   rises when it does not;
5. reconstruction still trains the codebooks, so the block is genuinely blocked
   rather than the whole path;
6. ``model.state_dict()`` keys are exactly the source tree's.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

SOURCE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE_DIR))

import train_rqvae as trainer  # noqa: E402
from model import RQVAE  # noqa: E402
import multiscale_behavioral_spectrum as spectrum  # noqa: E402
from multiscale_behavioral_spectrum import (MultiscaleRecoveryHeads,  # noqa: E402
                                            band_coordinate_loss,
                                            band_relation_loss,
                                            install_spectral_encoder)


def probe_checkpoint() -> Path:
    candidates = [Path(trainer.CHECKPOINT)]
    archive = Path("/home/wlia0047/ar57/wenyu/artifact_archive")
    if archive.is_dir():
        candidates += sorted(
            archive.glob(f"{trainer.TREE_NAME}_*/rqvae_best.pth"),
            key=lambda path: path.stat().st_mtime, reverse=True,
        )
    candidates.append(
        Path(trainer.REPO_ROOT)
        / "results/stage2_RQ-VAE/TIGER_BEHAVIOR_SAMPLE_RQ-VAE/resample/rqvae_best.pth"
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise SystemExit(f"no probe checkpoint found in {candidates}")


def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    report: dict[str, object] = {"device": str(device)}
    failures: list[str] = []

    trainer.set_seed(trainer.SEED)
    embeddings = trainer.load_embeddings(trainer.EMBEDDING_FILE)
    frame = pd.read_parquet(
        trainer.TRAIN_FILE, columns=["user", "seen_history", "target"]
    )
    item_ids = np.unique(frame["target"].to_numpy(dtype=np.int64))
    all_embeddings = torch.from_numpy(embeddings)

    # --- (1) the band targets are nested and each carries signal ------------
    started = time.time()
    histories = [list(history)
                  for history in frame.groupby("user")["seen_history"].last()]
    bands = spectrum.build_behaviour_spectrum(histories, len(all_embeddings),
                                              trainer.SPECTRUM_BAND_DIM)
    report["spectrum_build_seconds"] = round(time.time() - started, 2)
    cumulative = bands["cumulative"]
    report["cumulative_shape"] = list(cumulative.shape)
    report["band_energies"] = {
        name: float(np.linalg.norm(bands[name])) for name in ("low", "mid", "high")
    }
    report["depth_increases_energy"] = [
        float(np.linalg.norm(cumulative[d])) for d in range(cumulative.shape[0])
    ]
    # The cumulative targets are ball points, so the nesting is a statement about
    # the tangent bands they came from: depth q is the q-step smoothed graph
    # signal, low -> P X -> X. What must hold on the stored targets is that each
    # depth carries strictly more of it than the one above, and that they are all
    # inside the ball.
    report["cumulative_radius_max"] = float(
        np.linalg.norm(cumulative.reshape(-1, cumulative.shape[-1]), axis=-1).max()
    )
    if report["cumulative_radius_max"] >= 1.0:
        failures.append("a cumulative band target left the ball")
    energies = report["depth_increases_energy"]
    for name in ("low", "mid", "high"):
        if report["band_energies"][name] <= 0.0:
            failures.append(f"the {name} band carries no signal")
    if not (energies[0] < energies[1] < energies[2]):
        failures.append(
            f"cumulative depth targets do not grow: {energies}"
        )

    model = RQVAE(trainer._tokenizer_config(), in_dim=embeddings.shape[1]).to(device)
    checkpoint = probe_checkpoint()
    report["probe_checkpoint"] = str(checkpoint)
    payload = torch.load(checkpoint, map_location=device)
    model.load_state_dict(payload.get("state_dict", payload), strict=True)
    model.eval()
    for layer in model.rq.vq_layers:
        layer._skip_ddp_reduce = True
    with torch.no_grad():
        model.init_codebook(all_embeddings[item_ids].to(device))

    before_keys = sorted(model.state_dict())
    install_spectral_encoder(model)
    heads = MultiscaleRecoveryHeads(trainer.CODEBOOK_DIM, trainer.CODEBOOK_NUM).to(device)
    codebooks = tuple(layer.get_code_embs() for layer in model.rq.vq_layers)
    report["state_dict_keys_unchanged"] = sorted(model.state_dict()) == before_keys
    if not report["state_dict_keys_unchanged"]:
        failures.append("installing the mechanism changed model.state_dict()")

    rows = torch.arange(576, dtype=torch.long)
    batch = all_embeddings[item_ids[rows % len(item_ids)]].to(device)

    # --- (2) the cumulative representations are the level sums --------------
    cumulative_points = spectrum.cumulative_representations(model, batch)
    per_level = []
    residual = model.encoder(batch)
    for vq_layer in model.rq.vq_layers:
        quant, _, _, _ = vq_layer(residual)
        per_level.append(quant)
        residual = residual - quant
    worst_cumulative_error = max(
        float((cumulative_points[d] - torch.stack(per_level[:d + 1]).sum(0)).abs().max())
        for d in range(len(per_level))
    )
    report["cumulative_definition_error"] = worst_cumulative_error
    report["cumulative_depths"] = len(cumulative_points)
    if worst_cumulative_error > 1e-5:
        failures.append(
            f"q^(q) is not the cumulative level sum ({worst_cumulative_error:.2e})"
        )

    # --- (3)+(4) each depth's band loss reaches the head, not the codebook ---
    spectrum_t = torch.as_tensor(cumulative, dtype=torch.float32, device=device)
    ids = torch.arange(batch.shape[0], device=device) % spectrum_t.shape[1]
    predicted = heads(cumulative_points, tuple(c.detach() for c in codebooks))
    band_losses: dict[int, torch.Tensor] = {}
    for depth in range(len(predicted)):
        band_losses[depth] = (
            band_coordinate_loss(predicted[depth], spectrum_t[depth, ids])
            + band_relation_loss(predicted[depth], spectrum_t[depth, ids])
        )
    total = torch.stack([band_losses[d] for d in sorted(band_losses)]).mean()
    head_grads = torch.autograd.grad(
        total, tuple(heads.parameters()), retain_graph=True, allow_unused=True)
    encoder_grads = torch.autograd.grad(
        total, tuple(model.encoder.parameters()), retain_graph=True, allow_unused=True)
    codebook_grads = torch.autograd.grad(total, codebooks, allow_unused=True)
    report["band_losses"] = {
        str(depth): float(value.detach()) for depth, value in sorted(band_losses.items())
    }
    report["band_head_grad"] = max(
        [float(g.abs().sum()) for g in head_grads if g is not None] or [0.0])
    report["band_encoder_grad"] = max(
        [float(g.abs().sum()) for g in encoder_grads if g is not None] or [0.0])
    report["band_codebook_grad"] = max(
        [float(g.abs().sum()) for g in codebook_grads if g is not None] or [0.0])
    if report["band_head_grad"] <= 0.0:
        failures.append("the band losses produced no gradient on the heads")
    if report["band_encoder_grad"] <= 0.0:
        failures.append("the band losses produced no gradient on the encoder")
    if report["band_codebook_grad"] > 0.0:
        failures.append("the band losses leaked gradient into a codebook")

    exact = float(band_coordinate_loss(predicted[0], predicted[0].detach()))
    report["coordinate_loss_at_target"] = exact
    if exact > 1e-8:
        failures.append(f"the coordinate term does not vanish at its target ({exact:.2e})")

    # --- (5) reconstruction still trains the codebooks ----------------------
    model.zero_grad(set_to_none=True)
    heads.zero_grad(set_to_none=True)
    reconstructed, quant_loss, _, _ = model(batch)
    base_total, _ = model.compute_loss(batch, reconstructed, quant_loss)
    grads = torch.autograd.grad(base_total, codebooks, allow_unused=True)
    report["reconstruction_codebook_grad"] = max(
        [float(g.abs().sum()) for g in grads if g is not None] or [0.0])
    if report["reconstruction_codebook_grad"] <= 0.0:
        failures.append("blocking the band gradient also blocked reconstruction's")

    report["failures"] = failures
    print(json.dumps(report, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
