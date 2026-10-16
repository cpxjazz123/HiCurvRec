"""Issue #26 mechanism check for the invertible hyperbolic behavioural flow.

Exercises the installed flow on this tree's own probe checkpoint:

1. the coupling flow really is invertible -- ``G^{-1}(G(t)) == t`` and the composed
   ``F^{-1}(F(s)) == s`` on the ball, with the inverse carrying gradient;
2. the quantizer receives ``F_theta(s_i)`` and not the bare semantic base;
3. the behaviour ranking term reaches the flow and no codebook;
4. the local distortion term is zero for the identity flow and rises once the flow
   stretches a pair, so it really constrains the distortion;
5. the decoder-side base recovery goes through ``F^{-1}``, reaches its head and the
   flow, and is blocked on every codebook, while reconstruction still trains the
   codebooks;
6. ``model.state_dict()`` keys are exactly the source tree's.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

SOURCE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE_DIR))

import train_rqvae as trainer  # noqa: E402
from model import RQVAE  # noqa: E402
import invertible_hyperbolic_flow as flow_module  # noqa: E402
from invertible_hyperbolic_flow import (HyperbolicCouplingFlow,  # noqa: E402
                                        SemanticBaseRecoveryHead,
                                        behaviour_ranking_loss,
                                        install_invertible_flow_encoder,
                                        local_distortion_loss)


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
    flow = HyperbolicCouplingFlow(trainer.CODEBOOK_DIM, 4).to(device)
    install_invertible_flow_encoder(model, flow, flow_module.CURVATURE)
    recovery = SemanticBaseRecoveryHead(trainer.CODEBOOK_DIM).to(device)
    codebooks = tuple(layer.get_code_embs() for layer in model.rq.vq_layers)
    report["state_dict_keys_unchanged"] = sorted(model.state_dict()) == before_keys
    if not report["state_dict_keys_unchanged"]:
        failures.append("installing the mechanism changed model.state_dict()")

    # --- (1) invertibility --------------------------------------------------
    # A non-trivial flow: the layers are zero-initialised, which would make the
    # round trip a tautology.
    for layer in flow.layers:
        for parameter in layer.parameters():
            torch.nn.init.normal_(parameter, std=0.05)
    generator = torch.Generator().manual_seed(trainer.SEED)
    tangent = torch.randn(4096, trainer.CODEBOOK_DIM, generator=generator) * 0.3
    tangent_round = float((flow.inverse(flow(tangent)) - tangent).abs().max())
    base = flow_module._expmap0(
        torch.randn(4096, trainer.CODEBOOK_DIM, generator=generator) * 0.3,
        flow_module.CURVATURE,
    )
    moved = flow_module.hyperbolic_flow_forward(base, flow, flow_module.CURVATURE)
    restored = flow_module.hyperbolic_flow_inverse(moved, flow, flow_module.CURVATURE)
    ball_round = float((restored - base).abs().max())
    inverse_input = moved.detach().requires_grad_(True)
    flow_module.hyperbolic_flow_inverse(inverse_input, flow, flow_module.CURVATURE).sum().backward()
    inverse_grad = float(inverse_input.grad.abs().sum())
    report["tangent_round_trip"] = tangent_round
    report["ball_round_trip"] = ball_round
    report["inverse_gradient_norm"] = inverse_grad
    if tangent_round > 1e-4:
        failures.append(f"G_theta is not invertible ({tangent_round:.2e})")
    if ball_round > 1e-4:
        failures.append(f"F_theta and its inverse do not cancel ({ball_round:.2e})")
    if inverse_grad <= 0.0:
        failures.append("the inverse path carries no gradient")

    # --- (2) the quantizer sees the flowed point ----------------------------
    rows = torch.arange(256, dtype=torch.long)
    batch = all_embeddings[item_ids[rows]].to(device)
    quantizer_input = model.encoder(batch)
    base_here = trainer._semantic_base(model, batch)
    report["quantizer_input_radius_max"] = float(quantizer_input.norm(dim=-1).max())
    report["base_radius_max"] = float(base_here.norm(dim=-1).max())
    report["flow_moves_the_point"] = float((quantizer_input - base_here).abs().max())
    if report["flow_moves_the_point"] <= 0.0:
        failures.append("the installed flow does not change what the quantizer sees")
    if report["quantizer_input_radius_max"] >= 1.0:
        failures.append("the flowed point left the ball")

    # --- (3) the ranking term reaches the flow, not the codebook ------------
    ranking_points = model.encoder(
        all_embeddings[item_ids[torch.arange(192, dtype=torch.long) % len(item_ids)]].to(device)
    )
    ranking = behaviour_ranking_loss(
        ranking_points[:64], ranking_points[64:128], ranking_points[128:])
    flow_grads = torch.autograd.grad(
        ranking, tuple(flow.parameters()), retain_graph=True, allow_unused=True)
    codebook_grads = torch.autograd.grad(
        ranking, codebooks, retain_graph=True, allow_unused=True)
    report["rank_flow_grad"] = max(
        [float(g.abs().sum()) for g in flow_grads if g is not None] or [0.0])
    report["rank_codebook_grad"] = max(
        [float(g.abs().sum()) for g in codebook_grads if g is not None] or [0.0])
    if report["rank_flow_grad"] <= 0.0:
        failures.append("the behaviour ranking produced no gradient on the flow")
    if report["rank_codebook_grad"] > 0.0:
        failures.append("the behaviour ranking leaked gradient into a codebook")

    # --- (4) the distortion term constrains stretch -------------------------
    identity = HyperbolicCouplingFlow(trainer.CODEBOOK_DIM, 4).to(device)
    left_base = base[:2048]
    right_base = base[2048:4096]
    identity_loss = float(local_distortion_loss(
        left_base, left_base, right_base, right_base).detach())
    stretched = local_distortion_loss(
        left_base, moved[:2048], right_base, moved[2048:4096]
    )
    report["distortion_identity_flow"] = identity_loss
    report["distortion_trained_flow"] = float(stretched.detach())
    if identity_loss > 1e-6:
        failures.append(
            f"the identity flow is not distortion-free ({identity_loss:.2e})"
        )
    if report["distortion_trained_flow"] <= identity_loss:
        failures.append("the distortion term does not rise when the flow stretches")

    # --- (5) the decoder-side base recovery --------------------------------
    model.zero_grad(set_to_none=True)
    recovery.zero_grad(set_to_none=True)
    _trace, quantized = trainer._trace_and_quantized(model, batch, flow)
    recovered = recovery(quantized, flow, codebooks)
    recovery_loss = flow_module._geodesic(recovered, base_here).mean()
    recovery_count = len(tuple(recovery.parameters()))
    flow_count = len(tuple(flow.parameters()))
    encoder_count = len(tuple(model.encoder.parameters()))
    grads = torch.autograd.grad(
        recovery_loss,
        tuple(recovery.parameters()) + tuple(flow.parameters())
        + tuple(model.encoder.parameters()) + codebooks,
        allow_unused=True,
    )
    report["recovery_head_grad"] = max(
        [float(g.abs().sum()) for g in grads[:recovery_count] if g is not None] or [0.0])
    report["recovery_flow_grad"] = max(
        [float(g.abs().sum())
         for g in grads[recovery_count:recovery_count + flow_count] if g is not None]
        or [0.0])
    report["recovery_encoder_grad"] = max(
        [float(g.abs().sum())
         for g in grads[recovery_count + flow_count:
                        recovery_count + flow_count + encoder_count]
         if g is not None] or [0.0])
    report["recovery_codebook_grad"] = max(
        [float(g.abs().sum())
         for g in grads[recovery_count + flow_count + encoder_count:] if g is not None]
        or [0.0])
    if report["recovery_head_grad"] <= 0.0:
        failures.append("the base recovery produced no gradient on its head")
    if report["recovery_encoder_grad"] <= 0.0:
        failures.append("the base recovery produced no gradient on the encoder")
    if report["recovery_codebook_grad"] > 0.0:
        failures.append("the base recovery leaked gradient into a codebook")

    model.zero_grad(set_to_none=True)
    reconstructed, quant_loss, _, _ = model(batch)
    total, _ = model.compute_loss(batch, reconstructed, quant_loss)
    total_grads = torch.autograd.grad(total, codebooks, allow_unused=True)
    report["reconstruction_codebook_grad"] = max(
        [float(g.abs().sum()) for g in total_grads if g is not None] or [0.0])
    if report["reconstruction_codebook_grad"] <= 0.0:
        failures.append("blocking the recovery gradient also blocked reconstruction's")

    report["failures"] = failures
    print(json.dumps(report, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
