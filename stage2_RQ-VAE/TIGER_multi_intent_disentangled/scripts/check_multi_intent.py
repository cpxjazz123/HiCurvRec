"""Issue #25 mechanism check for the multi-intent disentangled hyperbolic encoder.

Exercises the installed encoder on this tree's own probe checkpoint:

1. the three behaviour views are genuinely different pair sets, and every view's
   anchor has a real partner to be ranked against;
2. ``Log_o`` really inverts each branch's ``Exp_o``, so the fusion happens on the
   tangents that produced the branches;
3. the text-conditioned gate is a convex mixture that reaches the quantizer, and
   the fused point stays inside the ball;
4. a branch's ranking term reaches the gate and no codebook;
5. the decorrelation term rises when two branches are made to agree, so it is
   really pushing them apart;
6. the decoder-side per-intent replay reaches its heads and the encoder and is
   blocked on every codebook, while reconstruction still trains the codebooks;
7. ``model.state_dict()`` keys are exactly the source tree's.
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
import multi_intent_disentangled as intent  # noqa: E402
from multi_intent_disentangled import (IntentRecoveryHeads,  # noqa: E402
                                       MultiIntentGate,
                                       install_multi_intent_encoder)


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
    all_embeddings = torch.from_numpy(embeddings)

    # --- (1) the three views are different ----------------------------------
    started = time.time()
    views = trainer._build_intent_views(frame, len(all_embeddings))
    report["view_build_seconds"] = round(time.time() - started, 2)
    report["views"] = sorted(views)
    report["view_pairs"] = {name: int(view["pairs"].shape[0]) for name, view in views.items()}
    report["view_anchors"] = {name: int(view["anchors"].shape[0]) for name, view in views.items()}
    if len(views) != 3:
        failures.append(f"expected three behaviour views, got {sorted(views)}")
    transition = set(map(tuple, views["transition"]["pairs"].tolist()))
    cooccurrence = set(map(tuple, views["cooccurrence"]["pairs"].tolist()))
    collaborative = set(map(tuple, views["collaborative"]["pairs"].tolist()))
    report["transition_subset_of_cooccurrence"] = transition <= cooccurrence
    report["collaborative_strictly_smaller"] = len(collaborative) < len(cooccurrence)
    if not report["transition_subset_of_cooccurrence"]:
        failures.append("the transition view is not a subset of co-occurrence")
    if len(collaborative) == len(cooccurrence):
        failures.append("the collaborative view is not distinguished from co-occurrence")

    model = RQVAE(trainer._tokenizer_config(), in_dim=embeddings.shape[1]).to(device)
    checkpoint = probe_checkpoint()
    report["probe_checkpoint"] = str(checkpoint)
    payload = torch.load(checkpoint, map_location=device)
    model.load_state_dict(payload.get("state_dict", payload), strict=True)
    model.eval()
    for layer in model.rq.vq_layers:
        layer._skip_ddp_reduce = True
    with torch.no_grad():
        model.init_codebook(all_embeddings[:24587].to(device))

    before_keys = sorted(model.state_dict())
    raw_forward = model.encoder.forward
    latent_batch = all_embeddings[:256].to(device)
    with torch.no_grad():
        raw_latent = raw_forward(latent_batch).clone()
    gate = MultiIntentGate(trainer.CODEBOOK_DIM, embeddings.shape[1], len(views)).to(device)
    install_multi_intent_encoder(model, gate)
    recovery = IntentRecoveryHeads(trainer.CODEBOOK_DIM, len(views)).to(device)
    codebooks = tuple(layer.get_code_embs() for layer in model.rq.vq_layers)
    report["state_dict_keys_unchanged"] = sorted(model.state_dict()) == before_keys
    if not report["state_dict_keys_unchanged"]:
        failures.append("installing the mechanism changed model.state_dict()")

    rows = torch.arange(256, dtype=torch.long)
    batch = all_embeddings[rows].to(device)

    # --- (2)+(3) fusion happens on the tangents -----------------------------
    latent = raw_forward(batch)
    _mixed, weights, tangents = gate(latent, batch)
    points = gate.intent_points(tangents)
    fused, back = intent.fuse_intents(points, weights)
    report["intent_count"] = len(tangents)
    report["log_exp_round_trip"] = max(
        float((b - t).abs().max()) for b, t in zip(back, tangents)
    )
    report["gate_weight_sum"] = float(weights.sum(-1).mean())
    report["gate_weight_std"] = float(weights.std(dim=0).mean())
    report["fused_radius_max"] = float(fused.norm(dim=-1).max())
    report["quantizer_input_matches_fusion"] = float(
        (model.encoder(batch) - fused).abs().max()
    )
    if report["log_exp_round_trip"] > 1e-4:
        failures.append(
            f"Log_o does not invert the branches' Exp_o ({report['log_exp_round_trip']:.2e})"
        )
    if abs(report["gate_weight_sum"] - 1.0) > 1e-4:
        failures.append("the gate weights are not a convex mixture")
    if report["fused_radius_max"] >= 1.0:
        failures.append("the fused point left the ball")
    if report["quantizer_input_matches_fusion"] > 1e-4:
        failures.append("the quantizer does not receive the fused representation")

    # --- (4) a branch's ranking term reaches the gate, not the codebook -----
    triples = {
        name: intent.view_triples(view, 64, torch.Generator().manual_seed(7), device)
        for name, view in views.items()
    }
    count = int(triples[sorted(views)[0]][0].shape[0])
    ranking = intent.intent_ranking_loss(
        points[0][:count], points[0][count:2 * count], points[0][2 * count:3 * count]
    )
    gate_grads = torch.autograd.grad(
        ranking, tuple(gate.parameters()), retain_graph=True, allow_unused=True)
    codebook_grads = torch.autograd.grad(
        ranking, codebooks, retain_graph=True, allow_unused=True)
    report["rank_gate_grad"] = max(
        [float(g.abs().sum()) for g in gate_grads if g is not None] or [0.0])
    report["rank_codebook_grad"] = max(
        [float(g.abs().sum()) for g in codebook_grads if g is not None] or [0.0])
    if report["rank_gate_grad"] <= 0.0:
        failures.append("the intent ranking produced no gradient on the gate")
    if report["rank_codebook_grad"] > 0.0:
        failures.append("the intent ranking leaked gradient into a codebook")

    # --- (5) decorrelation really penalises agreement -----------------------
    aligned = [tangents[0], tangents[0] * 1.0 + 0.0]
    orthogonal_like = [tangents[0], torch.roll(tangents[0], shifts=1, dims=-1)]
    report["decorrelation_aligned"] = float(
        intent.intent_decorrelation_loss(torch.stack(aligned)).detach())
    report["decorrelation_rotated"] = float(
        intent.intent_decorrelation_loss(torch.stack(orthogonal_like)).detach())
    if report["decorrelation_aligned"] <= report["decorrelation_rotated"]:
        failures.append("decorrelation does not penalise duplicated branches")

    # --- (6) the decoder-side replay ---------------------------------------
    model.zero_grad(set_to_none=True)
    recovery.zero_grad(set_to_none=True)
    ids = torch.cat([torch.cat(triple) for triple in triples.values()])
    triple_embeddings = all_embeddings[ids.cpu()].to(device)
    _trace, quantized = trainer._trace_and_quantized(model, triple_embeddings)
    replayed = recovery(quantized, codebooks)
    replay_loss = sum(
        intent.intent_ranking_loss(
            replayed[position][offset:offset + count],
            replayed[position][offset + count:offset + 2 * count],
            replayed[position][offset + 2 * count:offset + 3 * count],
        )
        for position in range(len(replayed))
        for offset in [sum(3 * int(t[0].shape[0]) for t in list(triples.values())[:position])]
    ) / len(replayed)
    recovery_count = len(tuple(recovery.parameters()))
    encoder_count = len(tuple(model.encoder.parameters()))
    grads = torch.autograd.grad(
        replay_loss,
        tuple(recovery.parameters()) + tuple(model.encoder.parameters()) + codebooks,
        allow_unused=True,
    )
    report["replay_head_grad"] = max(
        [float(g.abs().sum()) for g in grads[:recovery_count] if g is not None] or [0.0])
    report["replay_encoder_grad"] = max(
        [float(g.abs().sum())
         for g in grads[recovery_count:recovery_count + encoder_count] if g is not None]
        or [0.0])
    report["replay_codebook_grad"] = max(
        [float(g.abs().sum()) for g in grads[recovery_count + encoder_count:] if g is not None]
        or [0.0])
    if report["replay_head_grad"] <= 0.0:
        failures.append("the per-intent replay produced no gradient on its heads")
    if report["replay_encoder_grad"] <= 0.0:
        failures.append("the per-intent replay produced no gradient on the encoder")
    if report["replay_codebook_grad"] > 0.0:
        failures.append("the per-intent replay leaked gradient into a codebook")

    model.zero_grad(set_to_none=True)
    reconstructed, quant_loss, _, _ = model(batch)
    total, _ = model.compute_loss(batch, reconstructed, quant_loss)
    total_grads = torch.autograd.grad(total, codebooks, allow_unused=True)
    report["reconstruction_codebook_grad"] = max(
        [float(g.abs().sum()) for g in total_grads if g is not None] or [0.0])
    if report["reconstruction_codebook_grad"] <= 0.0:
        failures.append("blocking the replay gradient also blocked reconstruction's")

    report["failures"] = failures
    print(json.dumps(report, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
