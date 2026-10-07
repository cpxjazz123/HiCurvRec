"""Are the behaviour and reconstruction gradients fighting over the same pairs?

iter122 showed the persistent quarter of behaviour pairs has a median angular
margin of -0.000049, i.e. the true successor and the negative are not separated
on direction at all, and that the hyperbolic distance reports that absence
faithfully. Two explanations remain and they lead to different fixes. Either the
other Stage2 terms are pushing back: the behaviour ranking pulls the successor
away from the anchor while reconstruction and quantization pull it back, so the
pairs are held in place by an opposing force. Or nothing opposes them, and the
pairwise objective simply cannot separate these pairs from that starting
configuration.

The two are told apart by asking whether the gradients actually disagree. For the
same behaviour pairs the behaviour gradient is taken with respect to the anchor's
latent, and separately the gradient of reconstruction plus quantization is taken
with respect to the same anchor latents, on the same forward pass. The cosine
between them per pair says whether the two objectives want the same change.

A negative cosine on the persistent pairs would mean the behaviour loss is being
cancelled and the fix is loss balancing. A positive cosine with a small norm on
the behaviour side would mean the behaviour loss is asking for almost nothing
there, because its own gradient vanishes once a pair is satisfied, and the fix is
the objective's form rather than its weight. Both quantities are reported, since
the cosine alone cannot distinguish "opposed" from "silent".
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

SOURCE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE_DIR))

import curvature_config as experiment
from model import RQVAE
from model.layers import (
    _hyperbolic_residual,
    _pairwise_poincare_distance_tangents,
    _poincare_distance_tangent_pairs,
)


def tokenizer_config() -> SimpleNamespace:
    return SimpleNamespace(
        hidden_sizes=experiment.HIDDEN_SIZES,
        codebook_num=3,
        codebook_size=experiment.CODEBOOK_SIZE,
        codebook_dim=experiment.CODEBOOK_DIM,
        dropout=0.0,
        beta=experiment.BETA,
        vq_type=experiment.VQ_TYPE,
        ema_decay=experiment.EMA_DECAY,
        fix_code_embs=False,
        sk_epsilon=experiment.SK_EPSILON,
        sk_iters=experiment.SK_ITERS,
        layer_curvatures=experiment.LAYER_CURVATURES,
        layer_working_radii=experiment.LAYER_WORKING_RADII,
        pin_in_s_coordinates=experiment.PIN_IN_S_COORDINATES,
    )


def transition_pairs(train_frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    sources: list[int] = []
    successors: list[int] = []
    for history, target in zip(
        train_frame["seen_history"].to_numpy(),
        train_frame["target"].to_numpy(dtype=np.int64),
    ):
        if history is None or len(history) == 0:
            continue
        sources.append(int(history[-1]))
        successors.append(int(target))
    return (
        np.asarray(sources, dtype=np.int64),
        np.asarray(successors, dtype=np.int64),
    )


def cosines(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.cosine_similarity(a, b, dim=-1)


def reconstruct_and_quantize(
    model: RQVAE, latent: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decoder output and the quantization loss, given a fixed latent.

    This mirrors RQVAE.compute_loss, which is recon + quant on the encoder's
    output, so the gradient taken here is the one training actually applies to the
    encoder from everything except the behaviour term.
    """
    reconstructed = model.decoder(latent)
    layers = model.rq.vq_layers
    quantized_x = torch.zeros_like(latent)
    quant_loss = torch.zeros((), dtype=latent.dtype, device=latent.device)
    residual = latent
    previous_codes = None
    for level, layer in enumerate(layers):
        c = float(layer.get_curvature())
        target_norm = model.rq._radius_for_level(level, c, residual)
        pinned = model.rq._pin_to_radius(residual, target_norm)
        distances = _pairwise_poincare_distance_tangents(
            pinned, layer.get_code_embs(), c
        )
        chosen = layer._indices(distances, infer_use_sk=True, bucket=previous_codes)
        quant = layer.get_code_embs()[chosen]
        quant_loss = quant_loss + (layer.beta * (quant - pinned.detach()).pow(2).mean())
        quantized_x = quantized_x + model.rq._restore_norm(
            quant, residual, target_norm
        )
        residual = model.rq._restore_norm(
            _hyperbolic_residual(pinned, quant, c), residual, target_norm
        )
        previous_codes = chosen
    return reconstructed, quant_loss / len(layers)


def main() -> None:
    experiment.STAGE2_LOG_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    generator = np.random.default_rng(experiment.DECOMP_SEED)
    torch.manual_seed(experiment.DECOMP_SEED)

    embeddings = torch.from_numpy(
        np.asarray(np.load(experiment.EMBEDDING_FILE), dtype=np.float32)
    ).to(device)
    n_items = embeddings.shape[0]
    model = RQVAE(tokenizer_config(), in_dim=embeddings.shape[1]).to(device)
    model.load_state_dict(
        torch.load(
            experiment.PARENT_CKPT, map_location=device, weights_only=False
        )["state_dict"],
        strict=True,
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    train_frame = pd.read_parquet(experiment.TRAIN_FILE)
    source_ids, successor_ids = transition_pairs(train_frame)
    pick = generator.choice(
        len(source_ids), size=experiment.DECOMP_PAIRS, replace=False
    )
    source_ids = np.ascontiguousarray(source_ids[pick])
    successor_ids = np.ascontiguousarray(successor_ids[pick])
    perm = generator.permutation(len(source_ids))
    negative_ids = source_ids[perm]

    with torch.no_grad():
        anchor = model.encoder(embeddings[torch.from_numpy(source_ids).to(device)])
        successor = model.encoder(
            embeddings[torch.from_numpy(successor_ids).to(device)]
        )
    anchor = anchor.detach()
    curvature = float(experiment.LAYER_CURVATURES[0])

    # ---- populations, from the same five checkpoints iter121 and iter122 used
    trajectory_dir = (
        experiment.REPO_ROOT
        / "results/stage2_RQ-VAE/curvature_RQ-VAE_iter120/trajectory"
    )
    idx_s = torch.from_numpy(source_ids).to(device)
    idx_p = torch.from_numpy(successor_ids).to(device)
    idx_n = torch.from_numpy(negative_ids).to(device)
    columns = []
    paths = sorted(
        trajectory_dir.glob("step_*.pth"),
        key=lambda q: int(q.stem.split("_")[1]),
    ) + [experiment.PARENT_CKPT]
    for path in paths:
        other = RQVAE(tokenizer_config(), in_dim=embeddings.shape[1]).to(device)
        other.load_state_dict(
            torch.load(path, map_location=device, weights_only=False)["state_dict"],
            strict=True,
        )
        other.eval()
        with torch.no_grad():
            a = other.encoder(embeddings[idx_s])
            b = other.encoder(embeddings[idx_p])
            c = other.encoder(embeddings[idx_n])
            positive = _poincare_distance_tangent_pairs(a, b, curvature)
            negative = _poincare_distance_tangent_pairs(a, c, curvature)
        columns.append(
            ((negative - positive) < experiment.BEHAVIOUR_MARGIN).cpu().numpy()
        )
        del other
        torch.cuda.empty_cache()
    stack = np.stack(columns, axis=1)
    persistent = stack.all(axis=1)
    never = (~stack).all(axis=1)
    rotating = ~persistent & ~never

    # ---- behaviour gradient, and the everything-else gradient, on the same pairs
    behaviour_variable = anchor.detach().requires_grad_(True)
    positive = _poincare_distance_tangent_pairs(
        behaviour_variable, successor, curvature
    )
    negative = _poincare_distance_tangent_pairs(
        behaviour_variable, anchor.detach()[perm], curvature
    )
    violation = positive + experiment.BEHAVIOUR_MARGIN - negative
    per_pair_behaviour = torch.relu(violation)
    (behaviour_grad,) = torch.autograd.grad(
        per_pair_behaviour.sum(), behaviour_variable
    )

    recon_variable = anchor.detach().requires_grad_(True)
    reconstructed, quant = reconstruct_and_quantize(model, recon_variable)
    recon = torch.nn.functional.mse_loss(reconstructed, embeddings[idx_s])
    other_loss = recon + quant
    (other_grad,) = torch.autograd.grad(other_loss, recon_variable)

    # The behaviour term carries a weight in training, and so does the rest via
    # recon and beta*quant inside compute_loss. Weight both sides the way training
    # does, so the comparison is between the forces that actually meet.
    behaviour_forced = behaviour_grad * experiment.BEHAVIOUR_LOSS_WEIGHT
    other_forced = other_grad  # recon + quant already carry their own weights

    cosine = cosines(behaviour_forced, other_forced).detach()
    behaviour_norm = behaviour_forced.norm(dim=-1).detach()
    other_norm = other_forced.norm(dim=-1).detach()

    lines: list[str] = []
    result: dict = {
        "parent_checkpoint": str(experiment.PARENT_CKPT),
        "pairs": int(len(source_ids)),
        "checkpoints_used": len(paths),
        "populations": {
            "persistent": int(persistent.sum()),
            "rotating": int(rotating.sum()),
            "never": int(never.sum()),
        },
        "behaviour_weight": experiment.BEHAVIOUR_LOSS_WEIGHT,
        "beta_quant": experiment.BETA_QUANT,
    }
    lines.append(
        f"populations over {len(paths)} checkpoints: "
        f"persistent={int(persistent.sum())}  rotating={int(rotating.sum())}  "
        f"never={int(never.sum())}"
    )
    lines.append(
        f"gradient weighting: behaviour x{experiment.BEHAVIOUR_LOSS_WEIGHT}, "
        f"reconstruction as-is, quantization x{experiment.BETA_QUANT} inside the RQ loss"
    )

    lines.append("")
    lines.append("1. does the other objective oppose the behaviour term?")
    lines.append(
        f"   {'population':<12}{'n':>7}{'cos p50':>12}{'cos mean':>12}"
        f"{'frac cos<0':>14}{'beh |g| p50':>14}"
    )
    for label, mask in (
        ("persistent", persistent),
        ("rotating", rotating),
        ("never", never),
    ):
        if mask.sum() == 0:
            continue
        c = cosine[mask].cpu().numpy()
        entry = {
            "pairs": int(mask.sum()),
            "cosine_p50": float(np.median(c)),
            "cosine_mean": float(c.mean()),
            "fraction_negative": float((c < 0).mean()),
            "behaviour_norm_p50": float(
                behaviour_norm[mask].cpu().numpy().mean()
            ),
        }
        result.setdefault("conflict", {})[label] = entry
        lines.append(
            f"   {label:<12}{entry['pairs']:>7}{entry['cosine_p50']:>12.4f}"
            f"{entry['cosine_mean']:>12.4f}"
            f"{entry['fraction_negative'] * 100:>13.2f}%"
            f"{entry['behaviour_norm_p50']:>14.3e}"
        )

    lines.append("")
    lines.append("2. is the behaviour gradient silent rather than opposed?")
    lines.append(
        f"   {'population':<12}{'beh |g| mean':>15}{'other |g| mean':>16}"
        f"{'ratio':>12}"
    )
    for label, mask in (
        ("persistent", persistent),
        ("rotating", rotating),
        ("never", never),
    ):
        if mask.sum() == 0:
            continue
        b_mean = float(behaviour_norm[mask].cpu().numpy().mean())
        o_mean = float(other_norm[mask].cpu().numpy().mean())
        lines.append(
            f"   {label:<12}{b_mean:>15.3e}{o_mean:>16.3e}"
            f"{(b_mean / o_mean if o_mean > 0 else float('nan')):>12.4f}"
        )
        result.setdefault("norms", {})[label] = {
            "behaviour_mean": b_mean, "other_mean": o_mean,
        }

    lines.append("")
    lines.append("3. cosine against the other terms separately")
    # Reconstruction alone, and quantization alone, so a conflict can be
    # attributed to one of them rather than to their sum.
    recon_only_variable = anchor.detach().requires_grad_(True)
    reconstructed_only, _ = reconstruct_and_quantize(model, recon_only_variable)
    recon_only = torch.nn.functional.mse_loss(
        reconstructed_only, embeddings[idx_s]
    )
    (recon_grad,) = torch.autograd.grad(recon_only, recon_only_variable)
    recon_cosine = cosines(behaviour_forced, recon_grad).detach().cpu().numpy()
    for label, mask in (
        ("persistent", persistent),
        ("rotating", rotating),
        ("never", never),
    ):
        if mask.sum() == 0:
            continue
        lines.append(
            f"   {label:<12}cos(g_beh, g_recon) p50="
            f"{np.median(recon_cosine[mask]):>8.4f}  "
            f"negative {float((recon_cosine[mask] < 0).mean()) * 100:5.2f}%"
        )
        result.setdefault("reconstruction_only", {})[label] = {
            "cosine_p50": float(np.median(recon_cosine[mask])),
            "fraction_negative": float((recon_cosine[mask] < 0).mean()),
        }

    lines.append("")
    lines.append("verdict")
    conflict = result.get("conflict", {}).get("persistent", {})
    norms = result.get("norms", {})
    opposed = conflict.get("fraction_negative", 0.0) > 0.5 or conflict.get(
        "cosine_p50", 0.0
    ) < 0.0
    silent = (
        norms.get("persistent", {}).get("behaviour_mean", 1.0)
        < 0.5 * norms.get("rotating", {}).get("behaviour_mean", 1.0)
    )
    if opposed:
        verdict = (
            "the other Stage2 terms actively oppose the behaviour gradient on "
            "these pairs, so the conflict is a weighting problem"
        )
    elif silent:
        verdict = (
            "the behaviour gradient is nearly silent on the persistent pairs while "
            "the rest of the objective keeps moving them, so the objective's form "
            "is the limit rather than a conflict"
        )
    else:
        verdict = (
            "neither opposed nor silent: the behaviour gradient is present and "
            "aligned, so these pairs resist the pairwise objective itself"
        )
    result["verdict"] = verdict
    result["opposed"] = bool(opposed)
    result["silent"] = bool(silent)
    lines.append(f"   opposed={opposed}  silent={silent}")
    lines.append(f"   {verdict}")

    report = "\n".join(lines)
    print(report, flush=True)
    experiment.DECOMP_JSON.write_text(
        json.dumps(result, indent=2, default=float) + "\n", encoding="utf-8"
    )
    experiment.DECOMP_LOG.write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()