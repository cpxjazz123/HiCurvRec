"""Radial-force decomposition of every Stage2 loss term.

Question under test: why does the encoder latent norm grow ~9.4x under the
hyperbolic sibling metric but only ~3.6x under the Euclidean one, when the ball
map, the initialisation and every other setting are identical?

Method. A term's radial force is the directional derivative of that term along
the encoder's own radial direction,

    g_radial(term) = d(term)/dlog||latent|| = grad_term(latent) . latent/||latent||

evaluated by autograd on the very tensor the term was built from. Positive means
the term is lowered by growing the norm, i.e. it PUSHES THE NORM OUT; negative
means it pulls back. Measuring the terms separately, on identical weights, batch,
positives and negatives, is what separates "this term causes it" from "the norm
grew and this term followed".

Every term is computed with the production functions from train_rqvae, so the
loss form under test is the loss form that trains. The only code written here is
the measurement and the detach toggle for the ablation in question 4.

No training happens in this script and nothing is written outside the directory
named on the command line.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import train_rqvae as T  # noqa: E402


def radial_force(
    loss: torch.Tensor,
    base: torch.Tensor,
    level: int | None = None,
) -> float:
    """d(loss)/dlog||latent||: positive pushes the norm outward.

    ``base`` is the tensor autograd is given; ``level`` selects one layer of a
    stacked residual trace. Differentiating the stacked tensor and indexing
    afterwards is deliberate: asking autograd for the gradient of a *view* like
    ``residuals[0]`` returns None while the true gradient is non-zero, which
    would silently report every residual-level term as zero.
    """
    if not loss.requires_grad:
        return 0.0
    grad = torch.autograd.grad(
        loss, base, retain_graph=True, allow_unused=True,
    )[0]
    if grad is None:
        return 0.0
    if level is not None:
        grad = grad[level]
        base = base[level]
    direction = base / torch.linalg.vector_norm(base, dim=-1, keepdim=True).clamp_min(1e-12)
    return float((grad * direction).sum())


def sibling_nce(
    anchor_latent: torch.Tensor,
    batch_latent: torch.Tensor,
    sibling_latent: torch.Tensor,
    batch_ids: torch.Tensor,
    anchor_ids: torch.Tensor,
    sibling_table: torch.Tensor,
    metric_name: str,
    normalizer: torch.Tensor,
    detach_scale: bool,
    temperature: float,
):
    """The production sibling InfoNCE, with one switch on the scale's detach.

    Copied from fork_geometry_terms except that ``detach_scale`` is a parameter,
    because that single detach() is the object of the ablation and toggling it
    must not require touching the training code. The two encode_ball calls stay
    separate exactly as in production, so the gradient path measured here is the
    one that trains.
    """
    anchor_ball = T.encode_ball(anchor_latent, T.CURVATURE, normalizer)
    batch_ball = T.encode_ball(batch_latent, T.CURVATURE, normalizer)
    sibling_ball = T.encode_ball(sibling_latent, T.CURVATURE, normalizer)
    if metric_name == "poincare":
        d_pos = T.poincare_distance(anchor_ball, sibling_ball, T.CURVATURE)
        d_neg = T.pair_poincare_distance(anchor_ball, batch_ball, T.CURVATURE)
    else:
        d_pos = torch.linalg.vector_norm(anchor_ball - sibling_ball, dim=-1)
        d_neg = torch.cdist(anchor_ball, batch_ball, p=2)
    scale = d_neg.square().mean().sqrt().clamp_min(1e-8)
    if detach_scale:
        scale = scale.detach()
    logits = torch.cat(
        [(-d_pos / scale / temperature)[:, None], -d_neg / scale / temperature],
        dim=1,
    )
    known_positive = (sibling_table[:, :, None] == batch_ids[None, None, :]).any(dim=1)
    exclude = known_positive | (anchor_ids[:, None] == batch_ids[None, :])
    logits = logits.masked_fill(
        torch.cat([torch.zeros_like(exclude[:, :1]), exclude], dim=1),
        float("-inf"),
    )
    usable = torch.isfinite(logits[:, 1:]).any(dim=1)
    loss = torch.zeros((), device=anchor_latent.device)
    if bool(usable.any()):
        loss = F.cross_entropy(
            logits[usable],
            torch.zeros(int(usable.sum()), dtype=torch.long, device=anchor_latent.device),
        )
    return loss, d_pos, d_neg


def build(arm: str, device: torch.device):
    """Replicate main()'s setup exactly, for one arm, without DDP."""
    T.EXPERIMENT_ARM = arm
    arm_cfg = T._ARMS[arm]
    T.GROUP_DISTANCE = arm_cfg["group_distance"]
    T.RADIAL_MODE = arm_cfg["radial_mode"]
    T.FORK_RADIAL_WEIGHT = arm_cfg["radial_weight"]
    T.RADIUS_CONTROL_ENABLED = arm_cfg.get("radius_control", False)

    T.set_seed(T.SEED)
    embeddings = T.maybe_apply_pca(
        T.load_embeddings(T.EMBEDDING_FILE), T.PCA_DIM, T.SEED
    )
    train_frame = pd.read_parquet(T.TRAIN_FILE)
    train_ids = np.unique(train_frame["target"].to_numpy(dtype=np.int64))
    all_embeddings = torch.from_numpy(embeddings)
    train_embeddings = all_embeddings[torch.from_numpy(train_ids)]

    pair_dataset = T.TransitionPairs(train_frame)
    out_degree = T.behaviour_out_degree(train_frame, len(embeddings))
    log_degree = np.log1p(out_degree.astype(np.float64))
    order = np.argsort(np.argsort(log_degree))
    rank_frac = order / max(len(order) - 1, 1)
    radius_target = torch.from_numpy(
        (T.RHO_HI - (T.RHO_HI - T.RHO_LO) * rank_frac).astype(np.float32)
    )
    fork_siblings = torch.from_numpy(
        T.behaviour_siblings(
            train_frame, len(embeddings), T.FORK_SIBLINGS_PER_ITEM, T.SEED
        )
    )

    config = T._tokenizer_config()
    model = T.RQVAE(config, in_dim=embeddings.shape[1]).to(device)
    if T.XAVIER_INIT:
        T.initialize_tiger_weights(model)
    with torch.no_grad():
        model.init_codebook(train_embeddings.to(device))

    ball_normalizer = T.resolve_normalizer(model, train_embeddings.to(device))
    normalizer = torch.as_tensor(ball_normalizer, dtype=torch.float32, device=device)

    pairs = pair_dataset.pairs[: T.PAIR_BATCH_SIZE_PER_RANK].to(device)
    pair_batch = all_embeddings[
        torch.cat((pairs[:, 0], pairs[:, 1])).cpu()
    ].to(device)
    resolved_temperature = T.calibrate_temperature(
        model, pair_batch, pairs[:, 0], pairs[:, 1], normalizer
    )

    # Item batch: the first BATCH_SIZE_PER_RANK train rows in dataset order.
    # Both arms see this identical batch, which is what makes the comparison
    # one-factor.
    dataset = TensorDataset(train_embeddings, torch.from_numpy(train_ids.copy()))
    loader = DataLoader(dataset, batch_size=T.BATCH_SIZE_PER_RANK, shuffle=False)
    batch, row_index = next(iter(loader))
    batch = batch.to(device)
    row_index = row_index.to(device)

    import hashlib

    def _sha(tensors) -> str:
        return hashlib.sha256(
            b"".join(t.detach().cpu().numpy().tobytes() for t in tensors)
        ).hexdigest()

    # Encoder AND codebook are hashed separately: the codebook hash alone proves
    # the quantizer matched, not the encoder, and this round asks whether the two
    # arms started from the same weights.
    encoder_sha = _sha(model.encoder.state_dict().values())
    codebook_sha = _sha(
        layer.embed.weight for layer in model.rq.vq_layers
    )

    ctx = dict(
        arm=arm, model=model, device=device, batch=batch, row_index=row_index,
        all_embeddings=all_embeddings, train_embeddings=train_embeddings,
        radius_target=radius_target.to(device), fork_siblings=fork_siblings.to(device),
        normalizer=normalizer, ball_normalizer=ball_normalizer,
        pairs=pairs, resolved_temperature=resolved_temperature,
        group_distance=arm_cfg["group_distance"],
        encoder_init_sha256=encoder_sha, codebook_init_sha256=codebook_sha,
    )
    return ctx


def measure(ctx: dict, detach_scale: bool = True) -> dict:
    """Radial force of every term on one identical batch."""
    model = ctx["model"]
    device = ctx["device"]
    batch, row_index = ctx["batch"], ctx["row_index"]
    normalizer = ctx["normalizer"]
    temp = ctx["resolved_temperature"]

    # The force of a term must be taken on the tensor that term was actually
    # built from. Recomputing an encoder pass would put the loss and the probe
    # on different graphs and silently report zero, so the encoder output and
    # the residual trace are captured from inside the production calls instead.
    captured: list[torch.Tensor] = []
    hook = model.encoder.register_forward_hook(
        lambda _m, _i, out: captured.append(out)
    )
    residuals_captured: list = []
    original_trace = T._residuals_with_trace

    def recording_trace(*a, **kw):
        out = original_trace(*a, **kw)
        residuals_captured.append(out)
        return out

    T._residuals_with_trace = recording_trace
    try:
        reconstructed, quant_loss, _, _ = model(batch)
        total, recon = model.compute_loss(batch, reconstructed, quant_loss)
        recon_latent = captured[-1]
        f_recon = radial_force(recon, captured[-1])

        residuals_captured.clear()
        pair_batch = ctx["all_embeddings"][
            torch.cat((ctx["pairs"][:, 0], ctx["pairs"][:, 1])).cpu()
        ].to(device)
        residuals = recording_trace(model, pair_batch)
        behavior = T.behavior_contrastive_loss(
            residuals, ctx["pairs"][:, 0], ctx["pairs"][:, 1], normalizer, temp
        )
        f_behavior = radial_force(behavior, residuals, level=0)

        per_level = []
        for level in range(residuals.shape[0]):
            lvl = T.behavior_contrastive_loss(
                residuals[level: level + 1], ctx["pairs"][:, 0],
                ctx["pairs"][:, 1], normalizer, temp,
            )
            per_level.append(radial_force(lvl, residuals, level=level))

        residuals_captured.clear()
        radial_term, angular_term, _, stats = T.fork_geometry_terms(
            model, batch, row_index, ctx["radius_target"], ctx["fork_siblings"],
            normalizer, ctx["all_embeddings"],
            torch.Generator(device=device).manual_seed(T.SEED),
        )
        fork_residuals = residuals_captured[-1]
        f_radial = radial_force(radial_term, fork_residuals, level=0)
        f_angular = radial_force(angular_term, fork_residuals, level=0)
    finally:
        hook.remove()
        T._residuals_with_trace = original_trace

    latent = model.encoder(batch)

    # Sibling InfoNCE with the detach toggle, plus the separation it produces.
    valid = ctx["fork_siblings"][row_index] >= 0
    anchor_pos, sibling_pos = valid.nonzero(as_tuple=True)
    if anchor_pos.numel() > T.FORK_ANGULAR_MAX_PAIRS:
        keep = torch.randperm(anchor_pos.numel(), device=device)[: T.FORK_ANGULAR_MAX_PAIRS]
        anchor_pos, sibling_pos = anchor_pos[keep], sibling_pos[keep]
    anchor_ids = row_index[anchor_pos]
    sibling_ids = ctx["fork_siblings"][row_index][anchor_pos, sibling_pos]
    anchor_latent = model.encoder(batch[anchor_pos])
    sibling_latent = model.encoder(ctx["all_embeddings"][sibling_ids.cpu()].to(device))
    nce, d_pos, d_neg = sibling_nce(
        anchor_latent, latent, sibling_latent, row_index, anchor_ids,
        ctx["fork_siblings"][anchor_ids], ctx["group_distance"],
        normalizer, detach_scale, T.FORK_ANGULAR_TEMPERATURE,
    )
    f_nce = radial_force(nce, anchor_latent)

    # Separation is per anchor row: the positive's distance against that same
    # row's negatives. A lower ratio means the metric pulls the behaviour
    # sibling closer to the anchor than to the batch.
    ratio = d_pos.detach()[:, None] / d_neg.detach().clamp_min(1e-8)
    return {
        "arm": ctx["arm"], "group_distance": ctx["group_distance"],
        "detach_scale": detach_scale,
        "latent_norm_median": float(torch.median(
            torch.linalg.vector_norm(latent, dim=-1))),
        "force_recon": f_recon,
        "force_behavior_3layer": f_behavior,
        "force_behavior_per_level": per_level,
        "force_sibling_nce": f_nce,
        "force_radial_mse": f_radial,
        "force_fork_angular_production": f_angular,
        "nce_value": float(nce.detach()),
        "d_pos_mean": float(d_pos.detach().mean()),
        "d_neg_mean": float(d_neg.detach().mean()),
        "d_neg_min_mean": float(d_neg.detach().min(dim=1).values.mean()),
        "pos_over_min_neg_mean": float(ratio.min(dim=1).values.mean()),
        "radius_mean": stats["radius_mean"], "radius_std": stats["radius_std"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ckpt", nargs="*", default=[],
                    help="arm=path pairs; loads final weights instead of init")
    args = ap.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ckpt = dict(a.split("=", 1) for a in args.ckpt)
    report = {"init_parity": {}, "at_init": {}, "detach_ablation": {}}

    for arm in args.arms:
        ctx = build(arm, device)
        report["init_parity"][arm] = {
            "encoder_init_sha256": ctx["encoder_init_sha256"],
            "codebook_init_sha256": ctx["codebook_init_sha256"],
            "ball_normalizer": ctx["ball_normalizer"],
            "resolved_temperature": ctx["resolved_temperature"],
            "group_distance": ctx["group_distance"],
        }
        if arm in ckpt:
            blob = torch.load(ckpt[arm], map_location=device)
            ctx["model"].load_state_dict(blob["state_dict"])
        report["at_init"][arm] = measure(ctx, detach_scale=True)
        report["detach_ablation"][arm] = {
            "with_detach": measure(ctx, detach_scale=True),
            "without_detach": measure(ctx, detach_scale=False),
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()