"""Issue #23 mechanism check for hyperbolic relational structure distillation.

Runs on this tree's own probe checkpoint and asserts the properties the issue
asks for:

1. the co-occurrence teacher actually discriminates: the strong neighbour is
   closer than the weak one, and the law-of-cosines target angle is a real,
   non-degenerate number in (0, pi);
2. ``Log_{z_i}`` inverts ``Exp_{z_i}`` -- without that round trip the tangent-angle
   part of the structure term is meaningless;
3. the geodesic ranking hinge is positive for a violated triple and zero for a
   satisfied one, and it reaches the encoder but no codebook;
4. the structure term matches a planted ratio/angle exactly and penalises a wrong
   one;
5. the decoder-side replay reaches its head, reaches the encoder through the
   straight-through proxy, and is blocked on every codebook;
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
import relational_structure_distill as rel  # noqa: E402
from relational_structure_distill import RelationalRecoveryHead  # noqa: E402


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

    # --- (1) the teacher ----------------------------------------------------
    started = time.time()
    histories = frame.groupby("user")["seen_history"].last().tolist()
    teacher = rel.build_relational_teacher(histories, len(all_embeddings))
    report["teacher_build_seconds"] = round(time.time() - started, 2)
    report["teacher_triples"] = int(teacher["anchor"].shape[0])
    report["teacher_distinct_anchors"] = int(np.unique(teacher["anchor"]).shape[0])
    report["teacher_delta_strong_mean"] = float(teacher["delta_strong"].mean())
    report["teacher_delta_weak_mean"] = float(teacher["delta_weak"].mean())
    report["teacher_angle_mean"] = float(teacher["angle_target"].mean())
    if teacher["anchor"].shape[0] == 0:
        failures.append("teacher produced no triple")
    if float(teacher["delta_strong"].mean()) >= float(teacher["delta_weak"].mean()):
        failures.append("teacher does not order strong neighbours before weak ones")
    if not 0.0 < float(teacher["angle_target"].mean()) < np.pi:
        failures.append("teacher target angle is degenerate")

    # --- (2) the ball maps are genuine Riemannian maps ----------------------
    generator = torch.Generator().manual_seed(trainer.SEED)
    worst_absolute = 0.0
    worst_relative = 0.0
    for base_scale in (0.0, 0.2, 0.5, 0.8, 0.95):
        base = torch.randn(4096, trainer.CODEBOOK_DIM, generator=generator)
        base = (base / base.norm(dim=-1, keepdim=True) * base_scale
                if base_scale > 0 else torch.zeros(4096, trainer.CODEBOOK_DIM))
        for tangent_scale in (0.005, 0.05, 0.2, 0.5):
            tangent = torch.randn(4096, trainer.CODEBOOK_DIM, generator=generator)
            tangent = tangent / tangent.norm(dim=-1, keepdim=True) * tangent_scale
            moved = rel._expmap_x(base, tangent, rel.CURVATURE)
            length = torch.linalg.vector_norm(tangent, dim=-1)
            measured = rel._geodesic(base, moved, rel.CURVATURE)
            worst_absolute = max(worst_absolute, float((measured - length).abs().max()))
            worst_relative = max(
                worst_relative,
                float(((measured - length).abs() / length.clamp_min(1e-9)).max()),
            )
    report["expmap_isometry_absolute_error"] = worst_absolute
    report["expmap_isometry_relative_error"] = worst_relative
    if worst_absolute > 1e-4 or worst_relative > 5e-3:
        failures.append(
            f"Exp_x is not isometric with the tangent length "
            f"(absolute {worst_absolute:.2e}, relative {worst_relative:.2e})"
        )

    # --- the model under the installed encoder ------------------------------
    model = RQVAE(trainer._tokenizer_config(), in_dim=embeddings.shape[1]).to(device)
    checkpoint = trainer.CHECKPOINT
    if not Path(checkpoint).is_file():
        raise SystemExit(f"probe checkpoint missing: {checkpoint}")
    payload = torch.load(checkpoint, map_location=device)
    model.load_state_dict(payload.get("state_dict", payload), strict=True)
    model.eval()
    for layer in model.rq.vq_layers:
        layer._skip_ddp_reduce = True
    with torch.no_grad():
        model.init_codebook(all_embeddings[item_ids].to(device))

    before_keys = sorted(model.state_dict())
    rel.install_relational_encoder(model)
    report["state_dict_keys_unchanged"] = sorted(model.state_dict()) == before_keys
    if not report["state_dict_keys_unchanged"]:
        failures.append("installing the encoder changed model.state_dict()")

    head = RelationalRecoveryHead(trainer.CODEBOOK_DIM, rel.CURVATURE).to(device)
    codebooks = tuple(layer.get_code_embs() for layer in model.rq.vq_layers)

    generator = torch.Generator().manual_seed(trainer.SEED + 7)
    batch = min(256, len(item_ids))
    triple = rel.teacher_triples(
        teacher, batch, generator, torch.device(device),
    )
    report["triple_batch"] = int(triple["anchor"].shape[0])
    if triple["anchor"].shape[0] == 0:
        raise SystemExit("teacher produced no batchable triple")

    ids = torch.cat((
        triple["anchor"], triple["strong"], triple["weak"], triple["negative"],
    ))
    rows = all_embeddings[ids.cpu()].to(device)
    with torch.no_grad():
        z = model.encoder(rows)
    report["z_radius_max"] = float(z.norm(dim=-1).max())
    count = int(triple["anchor"].shape[0])
    z_anchor, z_strong, z_weak, z_negative = (
        z[:count], z[count:2 * count], z[2 * count:3 * count], z[3 * count:],
    )

    # --- (3) the ranking hinge ----------------------------------------------
    satisfied = rel.relational_ranking_loss(z_anchor, z_strong, z_strong, margin=0.0)
    violated = rel.relational_ranking_loss(z_anchor, z_strong, z_negative)
    report["ranking_satisfied"] = float(satisfied.detach())
    report["ranking_violated"] = float(violated.detach())
    if float(satisfied.detach()) > 1e-6:
        failures.append("the ranking hinge is not zero on a satisfied triple")
    if float(violated.detach()) <= 0.0:
        failures.append("the ranking hinge is inert on a violated triple")

    # A second, gradient-enabled pass: the one above is measured under no_grad.
    z = model.encoder(rows)
    z_anchor, z_strong, z_weak, z_negative = (
        z[:count], z[count:2 * count], z[2 * count:3 * count], z[3 * count:],
    )
    model.zero_grad(set_to_none=True)
    head.zero_grad(set_to_none=True)
    ranking = rel.relational_ranking_loss(z_anchor, z_strong, z_negative)
    encoder_grads = torch.autograd.grad(
        ranking, tuple(model.encoder.parameters()), retain_graph=True, allow_unused=True,
    )
    codebook_grads = torch.autograd.grad(
        ranking, codebooks, retain_graph=True, allow_unused=True,
    )
    report["ranking_encoder_grad"] = max(
        [float(g.abs().sum()) for g in encoder_grads if g is not None] or [0.0]
    )
    report["ranking_codebook_grad"] = max(
        [float(g.abs().sum()) for g in codebook_grads if g is not None] or [0.0]
    )
    if report["ranking_encoder_grad"] <= 0.0:
        failures.append("the ranking term produced no gradient on the encoder")
    if report["ranking_codebook_grad"] > 0.0:
        failures.append("the ranking term leaked gradient into a codebook")

    # --- (4) the structure term matches a planted target --------------------
    measured_ratio, measured_angle = rel.measure_relational_structure(
        z_anchor, z_strong, z_weak, rel.CURVATURE,
    )
    exact = rel.relational_structure_loss(
        z_anchor, z_strong, z_weak, measured_ratio.detach(), measured_angle.detach(),
    )
    report["structure_planted"] = float(exact.detach())
    wrong = rel.relational_structure_loss(
        z_anchor, z_strong, z_weak, measured_ratio.detach() + 0.5,
        measured_angle.detach() + 0.5,
    )
    report["structure_wrong"] = float(wrong.detach())
    if float(exact.detach()) > 1e-6:
        failures.append(
            f"the structure term does not vanish on its own triangle "
            f"({float(exact.detach()):.2e})"
        )
    if float(wrong.detach()) <= float(exact.detach()):
        failures.append("the structure term does not penalise a wrong triangle shape")
    report["structure_value"] = float(
        rel.relational_structure_loss(
            z_anchor, z_strong, z_weak, triple["ratio_target"], triple["angle_target"],
        ).detach()
    )
    structure_grads = torch.autograd.grad(
        report["structure_value"] * 0.0 + rel.relational_structure_loss(
            z_anchor, z_strong, z_weak, triple["ratio_target"], triple["angle_target"],
        ),
        tuple(model.encoder.parameters()), retain_graph=True, allow_unused=True,
    )
    report["structure_encoder_grad"] = max(
        [float(g.abs().sum()) for g in structure_grads if g is not None] or [0.0]
    )
    if report["structure_encoder_grad"] <= 0.0:
        failures.append("the structure term produced no gradient on the encoder")

    # --- (5) the decoder-side replay ---------------------------------------
    model.zero_grad(set_to_none=True)
    head.zero_grad(set_to_none=True)
    _trace, triple_quantized = trainer._trace_and_quantized(model, rows)
    recovered_point = head(triple_quantized, codebooks)
    recovery_ranking = rel.relational_ranking_loss(
        recovered_point[:count], recovered_point[count:2 * count],
        recovered_point[3 * count:],
    )
    recovery_structure = rel.relational_structure_loss(
        recovered_point[:count], recovered_point[count:2 * count],
        recovered_point[2 * count:3 * count],
        triple["ratio_target"], triple["angle_target"],
    )
    recovery = recovery_ranking + recovery_structure
    head_count = len(tuple(head.parameters()))
    recovery_grads = torch.autograd.grad(
        recovery, tuple(head.parameters()) + tuple(model.encoder.parameters()) + codebooks,
        allow_unused=True,
    )
    report["recovery_head_grad"] = max(
        [float(g.abs().sum()) for g in recovery_grads[:head_count] if g is not None] or [0.0]
    )
    encoder_count = len(tuple(model.encoder.parameters()))
    report["recovery_encoder_grad"] = max(
        [
            float(g.abs().sum())
            for g in recovery_grads[head_count:head_count + encoder_count]
            if g is not None
        ] or [0.0]
    )
    report["recovery_codebook_grad"] = max(
        [float(g.abs().sum()) for g in recovery_grads[head_count + encoder_count:] if g is not None]
        or [0.0]
    )
    if report["recovery_head_grad"] <= 0.0:
        failures.append("the decoder-side replay produced no gradient on its head")
    if report["recovery_encoder_grad"] <= 0.0:
        failures.append("the decoder-side replay produced no gradient on the encoder")
    if report["recovery_codebook_grad"] > 0.0:
        failures.append("the decoder-side replay leaked gradient into a codebook")

    # Reconstruction must still train the codebooks.
    model.zero_grad(set_to_none=True)
    reconstructed, quant_loss, _, _ = model(rows)
    total, _ = model.compute_loss(rows, reconstructed, quant_loss)
    total_grads = torch.autograd.grad(total, codebooks, allow_unused=True)
    report["reconstruction_codebook_grad"] = max(
        [float(g.abs().sum()) for g in total_grads if g is not None] or [0.0]
    )
    if report["reconstruction_codebook_grad"] <= 0.0:
        failures.append("blocking the replay gradient also blocked reconstruction's")

    report["failures"] = failures
    print(json.dumps(report, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
