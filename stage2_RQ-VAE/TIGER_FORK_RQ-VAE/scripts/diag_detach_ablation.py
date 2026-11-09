"""detach() ablation on the scale of the sibling InfoNCE.

Question 4 of the diagnosis round: does ``scale = negative_distance.detach()
.square().mean().sqrt()`` create an extra tendency for the latent norm to grow?

Method. For every saved intermediate snapshot of both arms, the sibling InfoNCE
is rebuilt twice from the SAME weights, the SAME batch and the SAME
positives/negatives, differing only in whether ``scale`` is detached. The
measured quantity is the radial force

    d(loss)/dlog||latent||  = grad(loss, latent) . latent/||latent||

positive meaning the term pushes the norm out. No training is performed, so this
is a gradient comparison and not an ablation run, exactly as required.

The sign argument being tested: with ``scale`` live, the logits are
``-d/(scale*T)`` and ``scale`` grows with the distances. Growing the norm
therefore inflates ``scale``, flattens every logit toward zero and pushes the
softmax toward uniform, which RAISES the cross-entropy. The term can lower its
own loss by keeping ``scale`` small, i.e. by shrinking the distances. Detaching
``scale`` removes that channel, leaving only the intended one: separate the
positive from the negatives.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_rqvae as T  # noqa: E402
from diag_radial_forces import build, sibling_nce, radial_force  # noqa: E402

ROOT = Path("/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/TIGER_FORK_RQ-VAE")
SNAP_EPOCHS = (1000, 1500, 1750, 2000, 2500, 3000)


def measure_snapshot(ctx, device, mode="eval"):
    """Force of the sibling InfoNCE with detach on and off.

    The gradient is taken w.r.t. the SAME batch latent tensor that production
    differentiates: in fork_geometry_terms the anchor side is ``latent[anchor_pos]``,
    a view of the batch latent, and the negative side is ``latent`` itself, so a
    single encoder pass feeds both. Taking the gradient w.r.t. only a separately
    encoded anchor block would drop the negative side's contribution and can even
    flip the sign, so the single-tensor form is used here.

    ``mode`` is explicit because the encoder MLP carries dropout: a force
    measured with dropout active is not the same quantity as one measured
    without, and the training-loop probe ran in train mode.
    """
    model = ctx["model"]
    model.train() if mode == "train" else model.eval()
    batch, row_index = ctx["batch"], ctx["row_index"]
    normalizer = ctx["normalizer"]

    latent = model.encoder(batch)
    valid = ctx["fork_siblings"][row_index] >= 0
    anchor_pos, sibling_pos = valid.nonzero(as_tuple=True)
    if anchor_pos.numel() > T.FORK_ANGULAR_MAX_PAIRS:
        keep = torch.randperm(
            anchor_pos.numel(), device=device
        )[: T.FORK_ANGULAR_MAX_PAIRS]
        anchor_pos, sibling_pos = anchor_pos[keep], sibling_pos[keep]
    anchor_ids = row_index[anchor_pos]
    sibling_ids = ctx["fork_siblings"][row_index][anchor_pos, sibling_pos]
    sibling_latent = model.encoder(
        ctx["all_embeddings"][sibling_ids.cpu()].to(device)
    )

    out = {}
    for detach in (True, False):
        nce, d_pos, d_neg = sibling_nce(
            latent[anchor_pos], latent, sibling_latent, row_index, anchor_ids,
            ctx["fork_siblings"][anchor_ids], ctx["group_distance"],
            normalizer, detach, T.FORK_ANGULAR_TEMPERATURE,
        )
        scale = d_neg.detach().square().mean().sqrt().clamp_min(1e-8)
        out["detach" if detach else "live"] = {
            "force": radial_force(nce, latent),
            "loss": float(nce.detach()),
            "scale": float(scale),
            "d_pos_mean": float(d_pos.detach().mean()),
            "d_neg_mean": float(d_neg.detach().mean()),
            "pos_over_min_neg": float(
                (d_pos.detach()[:, None] / d_neg.detach().clamp_min(1e-8))
                .min(dim=1).values.mean()
            ),
        }
    with torch.no_grad():
        out["latent_norm_median"] = float(
            torch.median(torch.linalg.vector_norm(latent, dim=-1))
        )
    return out


def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    report = {}
    for arm in ("diag_euclid_norm", "diag_hyp_norm"):
        ctx = build(arm, device)
        report[arm] = {
            "init_sha256": ctx["encoder_init_sha256"],
            "group_distance": ctx["group_distance"],
            "snapshots": {},
        }
        for ep in SNAP_EPOCHS:
            path = ROOT / arm / "snapshots" / f"ep{ep:04d}.pth"
            if not path.exists():
                report[arm]["snapshots"][ep] = {"missing": True}
                continue
            blob = torch.load(path, map_location=device)
            ctx["model"].load_state_dict(blob["state_dict"])
            # Both modes are recorded: the training-loop probe ran with dropout
            # active, so a force read off in eval mode is not the same quantity
            # and the difference has to be visible rather than assumed away.
            report[arm]["snapshots"][ep] = {
                "eval": measure_snapshot(ctx, device, "eval"),
                "train": measure_snapshot(ctx, device, "train"),
            }
    out = ROOT / "detach_ablation.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()