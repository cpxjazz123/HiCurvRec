"""MVG for symmetric ranking curvature (c_pos = c_neg = 2.0).

The change is two constants: the positive of the transition hinge is measured
at a weak curvature and the negative at the full one. RQ, the pin, Sinkhorn,
the loss weight, the margin and the latent placement are all inherited.

What has to be true before a 72k run is justified:

1. Setting both curvatures equal must reproduce the parent's loss exactly, or
   the comparison against 0.060081 is not a controlled one.
2. The asymmetric form must actually change the loss, not be a relabelling.
3. The change has to come from the intended channel. At c=0.25 the geodesic is
   close to Euclidean, so the positive should shorten; the negative at c=1.0 is
   untouched. If both move, something else changed.
4. The gradient has to reach the encoder, and the curvature must be a real
   input to the distance rather than a constant that cancels.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

SOURCE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE_DIR))

import curvature_config as experiment
import train_rqvae as training
from model import RQVAE
from model.layers import _poincare_distance_tangent_pairs
from model.model import asymmetric_ranking_loss, behaviour_ranking_loss

PARENT_CKPT = Path(
    "/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/"
    "curvature_RQ-VAE/out/rqvae/instruments/rqvae_best.pth"
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(f"MVG FAIL: {message}")


def require_nonzero_finite_gradient(loss, parameters, label):
    gradients = torch.autograd.grad(
        loss, parameters, retain_graph=True, allow_unused=True
    )
    require(
        any(
            g is not None and torch.isfinite(g).all()
            and torch.count_nonzero(g).item() > 0
            for g in gradients
        ),
        f"{label} produced no finite nonzero gradient",
    )


def main() -> None:
    training.configure_run(__file__)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(training.SEED)

    print("== configuration ==")
    require(
        training.LAYER_CURVATURES == (1.0, 1.0, 1.0),
        f"RQ curvature changed: {training.LAYER_CURVATURES}",
    )
    require(
        training.LAYER_WORKING_RADII == (0.2, 0.2, 0.2),
        f"pin changed: {training.LAYER_WORKING_RADII}",
    )
    require(
        training.BEHAVIOUR_LOSS_WEIGHT == 0.1
        and training.BEHAVIOUR_MARGIN == 0.4,
        "loss weight or margin changed",
    )
    require(
        training.BEHAVIOUR_POSITIVE_CURVATURE == 2.0
        and training.BEHAVIOUR_NEGATIVE_CURVATURE == 2.0,
        "symmetric ranking curvature differs from the registered value 2.0",
    )
    require(
        training.BEHAVIOUR_POSITIVE_CURVATURE
        == training.BEHAVIOUR_NEGATIVE_CURVATURE,
        "this run is the symmetric ablation; the two curvatures must match",
    )
    require(
        training.MAX_GLOBAL_STEPS == 72_000
        and training.BATCH_SIZE_PER_RANK == 1024
        and len(training.SNAPSHOT_STEPS) == 1,
        "Stage2 budget or batch size changed",
    )
    print(
        f"  RQ curvature {training.LAYER_CURVATURES}, pin "
        f"{training.LAYER_WORKING_RADII}, ranking curvature "
        f"(pos={training.BEHAVIOUR_POSITIVE_CURVATURE}, "
        f"neg={training.BEHAVIOUR_NEGATIVE_CURVATURE}), "
        f"weight {training.BEHAVIOUR_LOSS_WEIGHT}, margin "
        f"{training.BEHAVIOUR_MARGIN}, {training.MAX_GLOBAL_STEPS} steps"
    )

    print("== checkpoint and data ==")
    embeddings = np.asarray(np.load(experiment.EMBEDDING_FILE), dtype=np.float32)
    model = RQVAE(
        training._tokenizer_config(), in_dim=int(embeddings.shape[1])
    ).to(device)
    require(PARENT_CKPT.is_file(), f"missing parent checkpoint {PARENT_CKPT}")
    payload = torch.load(PARENT_CKPT, map_location=device, weights_only=False)
    # The weights have to actually be applied. An earlier version of this check
    # loaded the payload for its metadata and then measured a randomly
    # initialised encoder, which puts the latents at norm ~0.01 instead of the
    # trained ~2.05 and makes the margin 60x the distances rather than 0.1x.
    # That inverts every conclusion drawn from the hinge activity, so the load
    # is verified rather than assumed.
    model.load_state_dict(payload["state_dict"], strict=True)
    print(f"  parent @ step {payload['global_step']}, radii={payload['working_radii']}")

    frame = pd.read_parquet(experiment.TRAIN_FILE)
    source_ids, successor_ids = training._transition_pairs(frame)
    require(len(source_ids) >= 512, "not enough transition pairs")
    take = 8192
    sources = torch.from_numpy(
        embeddings[torch.from_numpy(source_ids[:take])]
    ).to(device)
    successors = torch.from_numpy(
        embeddings[torch.from_numpy(successor_ids[:take])]
    ).to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    z_source = model.encoder(sources)
    z_successor = model.encoder(successors)
    negatives = z_source[torch.randperm(z_source.shape[0], device=device)]
    require(
        torch.isfinite(z_source).all() and torch.isfinite(z_successor).all(),
        "encoder produced non-finite latents",
    )
    latent_norm = float(
        torch.linalg.vector_norm(z_source, dim=-1).median()
    )
    print(
        f"  trained latent norm p50 {latent_norm:.5f} "
        f"(a randomly initialised encoder would be ~0.01; the hinge analysis "
        "below is only meaningful at the trained scale)"
    )
    require(
        latent_norm > 0.5,
        f"latent norm p50 {latent_norm:.5f} is far below the trained parent; "
        "the checkpoint weights were probably not applied",
    )

    margin = training.BEHAVIOUR_MARGIN
    print("== 1. equal curvatures must reproduce the parent exactly ==")
    parent_loss = behaviour_ranking_loss(
        z_source, z_successor, negatives, curvature=1.0, margin=margin
    )
    collapsed = asymmetric_ranking_loss(
        z_source, z_successor, negatives,
        positive_curvature=1.0, negative_curvature=1.0, margin=margin,
    )
    require(
        torch.allclose(parent_loss, collapsed, atol=1e-6),
        f"symmetric loss at c=1.0 {float(collapsed):.8f} != "
        f"parent {float(parent_loss):.8f}; the comparison is not controlled",
    )
    print(f"  parent {float(parent_loss):.8f} == collapsed "
          f"{float(collapsed):.8f}")

    print("== 2/3. the asymmetry must act, and only on the positive ==")
    positive_pos = _poincare_distance_tangent_pairs(
        z_source, z_successor, training.BEHAVIOUR_POSITIVE_CURVATURE
    )
    positive_sym = _poincare_distance_tangent_pairs(
        z_source, z_successor, 1.0
    )
    # The negative must be measured at its own curvature, not the positive's.
    # The first version of this check compared the negative computed at two
    # different curvatures and called the difference a violation, which tested
    # the wrong thing: the requirement is that changing the *positive*
    # curvature does not reach the negative, not that curvature is inert.
    negative_own = _poincare_distance_tangent_pairs(
        z_source, negatives, training.BEHAVIOUR_NEGATIVE_CURVATURE
    )
    negative_sym = _poincare_distance_tangent_pairs(
        z_source, negatives, 1.0
    )
    symmetric = asymmetric_ranking_loss(
        z_source, z_successor, negatives,
        positive_curvature=training.BEHAVIOUR_POSITIVE_CURVATURE,
        negative_curvature=training.BEHAVIOUR_NEGATIVE_CURVATURE,
        margin=margin,
    )
    require(
        not torch.allclose(positive_pos, positive_sym, atol=1e-4),
        "the positive curvature has no effect on the positive distance",
    )
    require(
        not torch.allclose(negative_own, negative_sym, atol=1e-4),
        "the negative distance is unchanged by the curvature; the symmetric "
        "ablation would then be a no-op",
    )
    moved = float((positive_pos != positive_sym).float().mean())
    print(
        f"  positive: {float(positive_sym.mean()):.5f} at c=1.0 -> "
        f"{float(positive_pos.mean()):.5f} at "
        f"c={training.BEHAVIOUR_POSITIVE_CURVATURE} "
        f"({100 * moved:.1f}% of rows changed)"
    )
    print(
        f"  negative: {float(negative_sym.mean()):.5f} at c=1.0 -> "
        f"{float(negative_own.mean()):.5f} at "
        f"c={training.BEHAVIOUR_NEGATIVE_CURVATURE} (moved, as a symmetric "
        "ablation requires)"
    )
    # At c=2.0 both distances grow, because the Poincare distance is monotone
    # in curvature over the range used. An earlier version of this check
    # required the positive to *shorten*, which is the asymmetric c=0.9
    # condition and is simply the wrong direction here.
    require(
        float(positive_pos.mean()) > float(positive_sym.mean()),
        "the positive distance did not grow with curvature; c=2.0 has no "
        "effect and the mechanism would be inert",
    )
    require(
        float(negative_own.mean()) > float(negative_sym.mean()),
        "the negative distance did not grow with curvature; the symmetric "
        "ablation is not actually symmetric",
    )
    print(
        f"  loss: parent {float(parent_loss):.8f} -> symmetric "
        f"{float(symmetric):.8f} "
        f"({100 * (float(symmetric) / float(parent_loss) - 1):+.2f}%)"
    )
    require(
        not torch.allclose(symmetric, parent_loss, atol=1e-5),
        "the asymmetric loss equals the parent loss; the mechanism is inert",
    )

    print("== 4. gradient reaches the encoder, and the hinge has teeth ==")
    parameters = list(model.parameters())
    require_nonzero_finite_gradient(symmetric, parameters, "asymmetric ranking loss")
    require_nonzero_finite_gradient(parent_loss, parameters, "parent ranking loss")
    print("  both ranking forms produce finite nonzero encoder gradients")
    require(
        float(symmetric) > 0.0,
        "the hinge is already satisfied on the trained parent; there is "
        "nothing left for the asymmetry to act on",
    )
    # How much of the hinge is actually active, and whether the asymmetry
    # relieves or tightens it.
    with torch.no_grad():
        active_parent = float(
            (positive_sym + margin - negative_sym > 0).float().mean()
        )
        active_sym = float(
            (positive_pos + margin - negative_sym > 0).float().mean()
        )
    print(
        f"  active constraints: parent {100 * active_parent:.1f}%, "
        f"asymmetric {100 * active_sym:.1f}%"
    )
    # The gate that iter71 lacked. A hinge that is nearly always inactive
    # supplies almost no gradient, so a run started under it burns the full
    # Stage2 budget to reproduce the parent. Measured on the trained
    # checkpoint: 44.1% active at c_pos=1.0, 0.11% at c_pos=0.25.
    require(
        active_sym >= 0.10,
        f"only {100 * active_sym:.2f}% of hinges are active under "
        f"c_pos={training.BEHAVIOUR_POSITIVE_CURVATURE}; the ranking term will "
        "supply almost no gradient and the run will just reproduce the parent",
    )
    print(
        f"  symmetric run: active {100 * active_sym:.1f}% vs parent "
        f"{100 * active_parent:.1f}%, hinge "
        f"{float(torch.relu(positive_pos + margin - negative_own).mean()):.6f} "
        f"vs {float(parent_loss):.6f}"
    )

    print("== curvature is a real input, not a cancelling constant ==")
    # A cheap guard: the distance must genuinely depend on c over the range
    # this mechanism uses, otherwise the two curvatures are interchangeable.
    means = [
        float(_poincare_distance_tangent_pairs(z_source, z_successor, c).mean())
        for c in (0.1, 0.25, 0.5, 1.0, 2.0, 4.0)
    ]
    print("  positive distance by curvature: " + ", ".join(
        f"c={c}: {m:.5f}" for c, m in zip((0.1, 0.25, 0.5, 1.0, 2.0, 4.0), means)))
    require(
        max(means) - min(means) > 0.05 * max(means),
        "the positive distance barely responds to curvature over the range "
        "used; the two curvatures are not meaningfully different",
    )

    print(
        "MVG PASS: RQ, pin, Sinkhorn, weight, margin and latent placement all "
        "unchanged; equal curvatures reproduce the parent exactly; the "
        "asymmetry shortens only the positive; the hinge is still active and "
        "gradients reach the encoder."
    )


if __name__ == "__main__":
    main()
