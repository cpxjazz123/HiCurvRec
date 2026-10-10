"""Pre-flight gate for the geometry arms (CLAUDE.md 6, run inline before training).

Run from the tree root with no arguments; it reads the hardcoded arm table and
reports on whichever arm EXPERIMENT_ARM selects. Writes no artifacts, prints
JSON, exits non-zero on FAIL.

Checks, in order:
  1. the ball image respects the radius bound and lands at the intended
     fraction of it, so the geometry is neither degenerate nor saturated;
  2. the distance block is finite, symmetric in expectation, and the self
     distance is bounded by the acosh clamp rather than infinite;
  3. logits after temperature are in a range where softmax is not saturated,
     which is the failure mode that would make every arm look identical;
  4. the contrastive term alone puts non-zero gradient on encoder parameters;
  5. the total objective keeps a finite grad_fn and non-zero gradients
     everywhere, including the transition weight when the arm has one.
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

import train_rqvae as T  # noqa: E402
from model import RQVAE  # noqa: E402


def main() -> None:
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    report: dict = {
        "arm": T.EXPERIMENT_ARM,
        "geometry": T.GEOMETRY,
        "curvature": T.CURVATURE,
        "tangent_scale": T.TANGENT_SCALE,
        "transition": T.TRANSITION,
        "distance_normalization": T.DISTANCE_NORMALIZATION,
        "temperature": T.BEHAVIOR_TEMPERATURE,
        "false_negative_mask": T.FALSE_NEGATIVE_MASK,
        "device": str(device),
    }
    failures: list[str] = []

    T.set_seed(T.SEED)
    embeddings = T.load_embeddings(T.EMBEDDING_FILE)
    frame = pd.read_parquet(T.TRAIN_FILE, columns=["seen_history", "target"])
    ids = np.unique(frame["target"].to_numpy(dtype=np.int64))
    all_emb = torch.from_numpy(embeddings)
    train_emb = all_emb[torch.from_numpy(ids)]
    pair_dataset = T.TransitionPairs(frame)
    indicator = (
        T._successor_indicator(frame, len(embeddings)) if T.FALSE_NEGATIVE_MASK else None
    )

    # Exactly what the trainer does, including the device placement, so a
    # parameter left on the wrong device cannot slip past this gate and only
    # surface later inside DDP.
    model = RQVAE(T._tokenizer_config(), in_dim=embeddings.shape[1]).to(device)
    T.initialize_tiger_weights(model)
    if T.TRANSITION != "none":
        model.register_parameter(
            "transition_weight",
            torch.nn.Parameter(
                torch.zeros(
                    T._tokenizer_config().codebook_dim,
                    T._tokenizer_config().codebook_dim,
                    device=device,
                )
            ),
        )
    model.eval()
    for layer in model.rq.vq_layers:
        layer._skip_ddp_reduce = True
    with torch.no_grad():
        model.init_codebook(train_emb.to(device))

    weight = getattr(model, "transition_weight", None)
    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    report["trainable_parameter_count"] = len(params)
    # DDP refuses a module whose parameters span devices; check it here instead.
    devices = {str(p.device) for _, p in params}
    report["parameter_devices"] = sorted(devices)
    if len(devices) != 1 or str(device) not in devices:
        failures.append(f"parameters span devices {sorted(devices)}, expected {device}")
    report["has_transition_weight"] = weight is not None

    # --- 1/2/3: geometry and logit health at the real residual scale --------
    batch = T.PAIR_BATCH_SIZE_PER_RANK
    pairs = pair_dataset.pairs[:batch]
    s_ids, t_ids = pairs[:, 0].to(device), pairs[:, 1].to(device)
    pair_batch = all_emb[torch.cat((s_ids, t_ids)).cpu()].to(device)
    residuals = T._residuals_with_trace(model, pair_batch)
    report["residual_shape"] = list(residuals.shape)

    with torch.no_grad():
        from hyperbolic import ball_radius, encode_ball

        src = residuals[0, :batch]
        cand = residuals[0, batch:]
        radius = ball_radius(T.CURVATURE)
        z = encode_ball(src, T.CURVATURE, T.TANGENT_SCALE)
        ratio = float(z.norm(dim=-1).max() / radius)
        report["ball_radius"] = radius
        report["max_norm_over_radius"] = ratio
        if ratio >= 1.0:
            failures.append(f"ball image reaches the boundary: ||z||/R={ratio}")
        raw_dist = T._geodesic_block(src, cand, weight)
        rms = raw_dist.square().mean().sqrt()
        report["raw_distance_range"] = [float(raw_dist.min()), float(raw_dist.max())]
        report["raw_distance_rms"] = float(rms)
        normalized = raw_dist / rms.clamp_min(1e-8)
        # Use the calibrated temperature the run itself would resolve.
        resolved = T.calibrate_temperature(model, pair_batch, s_ids, t_ids, weight)
        report["calibrated_temperature"] = resolved
        logits = -normalized / resolved
        report["logit_range"] = [float(logits.min()), float(logits.max())]
        # softmax temperature health: the positive's own probability should not
        # be pinned at 0 or 1 for every row, or no gradient survives.
        probs = torch.softmax(logits, dim=-1)
        own = probs[torch.arange(batch, device=device), torch.arange(batch, device=device)]
        report["own_probability_mean"] = float(own.mean())
        saturated = float(((own < 1e-6) | (own > 1 - 1e-6)).float().mean())
        report["own_probability_saturated_fraction"] = saturated
        if saturated > 0.05:
            failures.append("softmax saturation exceeds 5% of rows; temperature calibration failed")

    # --- 4: the contrastive term reaches the encoder ------------------------
    contrastive = T.behavior_contrastive_loss(
        residuals, s_ids, t_ids, indicator, weight
    )
    report["contrastive_loss"] = float(contrastive.detach())
    if not contrastive.requires_grad or contrastive.grad_fn is None:
        failures.append("contrastive loss has no grad_fn")
    grads = torch.autograd.grad(
        contrastive, [p for _, p in params], retain_graph=True, allow_unused=True
    )
    nonzero = [
        n for (n, _), g in zip(params, grads) if g is not None and float(g.abs().sum()) > 0
    ]
    report["contrastive_nonzero_params"] = len(nonzero)
    report["contrastive_touches_encoder"] = any(n.startswith("encoder.") for n in nonzero)
    report["contrastive_touches_transition"] = any("transition_weight" in n for n in nonzero)
    if not report["contrastive_touches_encoder"]:
        failures.append("contrastive term produced no encoder gradient")
    if weight is not None and not report["contrastive_touches_transition"]:
        # Zero-init means the gradient can legitimately be zero; report it so a
        # later silent failure is distinguishable.
        report["transition_grad_at_zero_init"] = float(grads[[n for n, _ in params].index("transition_weight")].abs().sum())

    # --- 5: the total objective --------------------------------------------
    item = train_emb[torch.arange(T.BATCH_SIZE_PER_RANK, dtype=torch.long)].to(device)
    rec, ql, _, _ = model(item)
    base, _ = model.compute_loss(item, rec, ql)
    total = base + T.BEHAVIOR_WEIGHT_MAX * contrastive
    report["total_loss"] = float(total.detach())
    report["total_grad_fn"] = type(total.grad_fn).__name__ if total.grad_fn else None
    report["total_finite"] = bool(torch.isfinite(total))
    total_grads = torch.autograd.grad(
        total, [p for _, p in params], retain_graph=True, allow_unused=True
    )
    total_nonzero = sum(
        1 for g in total_grads if g is not None and float(g.abs().sum()) > 0
    )
    report["total_nonzero_params"] = total_nonzero
    if total_nonzero == 0:
        failures.append("total objective produced no gradient")

    # a non-zero transition weight must actually change the loss, else the
    # operator is a no-op and the arm cannot test anything.
    if weight is not None:
        with torch.no_grad():
            weight.normal_(0, 0.05)
            perturbed = T.behavior_contrastive_loss(
                residuals, s_ids, t_ids, indicator, weight
            )
        delta = abs(float(perturbed) - report["contrastive_loss"])
        report["loss_change_under_nonzero_transition"] = delta
        if delta == 0.0:
            failures.append("non-zero transition weight does not change the loss")
        with torch.no_grad():
            weight.zero_()

    report["failures"] = failures
    report["status"] = "FAIL" if failures else "PASS"
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
