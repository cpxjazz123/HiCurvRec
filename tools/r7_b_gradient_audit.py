"""Track B (issue #81): what actually reaches the optimizer, measured without a counterfactual.

R4-08 is healthy with a 314.7x weighted behavior-to-base encoder gradient ratio; the three collapsed
R6 arms sit at 146.5x-167.0x. Larger is healthier, so raw magnitude cannot be the story, and the
question this lane answers is whether the DIRECTION the optimizer applies is what differs.

Nothing here is an intervention. Every number comes from the frozen step track A already captured,
rebuilt once in a private branch, with each loss term's gradient taken separately by autograd. There
is no branch/restore, so this lane does not depend on #79 or on counterfactual validity at all.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/GeneRec")
for entry in (str(REPO_ROOT / "tools"), str(REPO_ROOT / "tools/cgv")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import r7_a_frozen_replay as R  # noqa: E402
import r7_a_replay_engine as G  # noqa: E402

ARMS = ("r6_02", "r6_03", "r6_04", "r4_08")
OBSERVED_STEPS = (96, 160)
BUNDLE_ROOT = REPO_ROOT / "results/stage2_RQ-VAE/r7_a_parity"
OUTPUT_ROOT = REPO_ROOT / "results/stage2_RQ-VAE/r7_b_gradients"


def norm(vector) -> float | None:
    return None if vector is None else float(vector.norm())


def cosine(left, right) -> float | None:
    if left is None or right is None:
        return None
    denominator = float(left.norm()) * float(right.norm())
    return None if denominator == 0.0 else float((left * right).sum() / denominator)


def dot(left, right) -> float | None:
    if left is None or right is None:
        return None
    return float((left * right).sum())


def join(parts) -> torch.Tensor | None:
    """Concatenate per-parameter gradients; None if any parameter got no gradient at all."""
    if not parts or any(part is None for part in parts):
        return None
    return torch.cat([part.reshape(-1).to(torch.float64) for part in parts])


def decompose(branch: dict[str, Any], bundle: dict[str, Any]) -> dict[str, Any]:
    """Per-term, per-block gradients for one captured step, taken separately by autograd."""
    trainer, model = branch["trainer"], branch["model"]
    model.train()
    model.zero_grad(set_to_none=True)
    device = G.DEVICE
    weight = float(bundle["applied_weight"])

    named = list(model.named_parameters())
    parameters = [parameter for _, parameter in named]
    encoder_names = [name for name, _ in named if name.startswith("encoder.")]
    encoder_positions = [index for index, (name, _) in enumerate(named)
                         if name.startswith("encoder.")]

    item_batch = bundle["item_batch"].to(device)
    pair_embeddings = bundle["pair_embeddings"]
    source_ids = bundle["source_ids"].to(device)
    target_ids = bundle["target_ids"].to(device)
    successor = bundle.get("successor")
    successor = None if successor is None else successor.to(device)

    reconstructed, quant_loss, _, _ = model(item_batch)
    base_loss, _ = model.compute_loss(item_batch, reconstructed, quant_loss)

    level_losses: dict[int, torch.Tensor] = {}
    if weight > 0.0 and pair_embeddings is not None:
        residuals = trainer._residuals_with_trace(model, pair_embeddings.to(device))
        batch = int(source_ids.shape[0])
        for level in trainer.ACTIVE_LEVELS:
            source = residuals[level, :batch]
            candidates = residuals[level, batch:]
            logits = -torch.cdist(source, candidates, p=2) / trainer.BEHAVIOR_TEMPERATURE
            level_loss = trainer._contrastive_ce(logits, source_ids, target_ids, successor)
            if level_loss is not None:
                level_losses[int(level)] = level_loss

    def grads_of(scalar: torch.Tensor, *, retain: bool = True):
        return torch.autograd.grad(scalar, parameters, retain_graph=retain, allow_unused=True)

    g_base = grads_of(base_loss)
    base_encoder = join([g_base[index] for index in encoder_positions])

    level_grads: dict[int, list] = {}
    for level, level_loss in level_losses.items():
        grads = grads_of(level_loss)
        level_grads[level] = grads

    behaviour = (torch.stack(list(level_losses.values())).mean() if level_losses
                 else torch.zeros((), device=device))
    assembled = base_loss + weight * behaviour
    g_total = grads_of(assembled, retain=False)
    total_encoder = join([g_total[index] for index in encoder_positions])

    per_level = {}
    weighted_parts = []
    share = weight / max(1, len(level_losses))
    for level, grads in level_grads.items():
        vector = join([grads[index] for index in encoder_positions])
        if vector is not None:
            weighted_parts.append(vector * share)
        per_level[str(level)] = {
            "encoder_l2": norm(vector),
            "all_params_l2": norm(join(grads)),
            "cosine_vs_base": cosine(vector, base_encoder),
        }
    deep_encoder = (torch.stack(weighted_parts).sum(dim=0) if weighted_parts else None)

    # the actual clip, applied to the real gradient, so 'preclip' and 'postclip' are the real ones
    model.zero_grad(set_to_none=True)
    assembled.backward()
    preclip = join([parameter.grad for parameter in parameters])
    clip = float(getattr(trainer, "GRADIENT_CLIP_NORM", 0.0) or 0.0)
    if clip > 0 and preclip is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    postclip = join([parameter.grad for parameter in parameters])

    return {
        "applied_weight": weight,
        "levels_present": sorted(level_losses),
        "base_encoder_l2": norm(base_encoder),
        "per_level": per_level,
        "deep_weighted_encoder_l2": norm(deep_encoder),
        "assembled_total_encoder_l2": norm(total_encoder),
        "weighted_to_base_ratio": (None if deep_encoder is None or base_encoder is None
                                    or float(base_encoder.norm()) == 0.0
                                    else float(deep_encoder.norm() / base_encoder.norm())),
        "cosine_base_vs_weighted_deep": cosine(base_encoder, deep_encoder),
        "dot_base_vs_weighted_deep": dot(base_encoder, deep_encoder),
        "additivity": {
            "definition": "sum of per-level weighted encoder norms versus the assembled deep norm",
            "sum_of_level_weighted_norms": (None if deep_encoder is None else
                                            sum(share * (per_level[str(level)]["encoder_l2"] or 0.0)
                                                for level in level_losses)),
            "assembled_deep_norm": norm(deep_encoder),
            "exactly_additive": (None if deep_encoder is None or
                                 float(deep_encoder.norm()) == 0.0 else
                                 abs(sum(share * (per_level[str(level)]["encoder_l2"] or 0.0)
                                         for level in level_losses)
                                     - float(deep_encoder.norm())) < 1e-9),
        },
        "clip": {"norm": clip, "preclip_l2": norm(preclip), "postclip_l2": norm(postclip),
                 "cosine_preclip_vs_base": cosine(preclip, base_encoder)},
        "_vectors": {"assembled_encoder": total_encoder, "base_encoder": base_encoder,
                     "deep_weighted_encoder": deep_encoder},
    }


def realized_step(branch: dict[str, Any], bundle: dict[str, Any],
                  vectors: dict[str, Any]) -> dict[str, Any]:
    """Whether the AdamW step follows the gradient the arm assembled, or its preconditioned image."""
    before = {name: value.detach().clone().cpu()
              for name, value in branch["model"].state_dict().items()}
    step = G.assemble_and_step(branch, bundle, intervention="FULL")
    after = branch["model"].state_dict()
    delta = torch.cat([(after[name].detach().to(torch.float64).cpu() - before[name].to(torch.float64)
                        ).reshape(-1) for name in sorted(before) if name in after])
    return {
        "update_l2": norm(delta),
        "cosine_update_vs_assembled_gradient": cosine(delta, vectors.get("assembled_encoder")),
        "cosine_update_vs_base_gradient": cosine(delta, vectors.get("base_encoder")),
        "cosine_update_vs_weighted_deep": cosine(delta, vectors.get("deep_weighted_encoder")),
        "update_sha256": step["update_sha256"],
        "base_loss": step["base_loss"],
        "behavior_loss": step["behavior_loss"],
        "projection_applied": step["projection_applied"],
    }


def run_arm(arm: str) -> dict[str, Any]:
    started = time.time()
    tree = REPO_ROOT / f"stage2_RQ-VAE/TIGER_{arm}"
    report: dict[str, Any] = {"arm": arm, "trainer_sha256": R.file_sha256(tree / "train_rqvae.py"),
                              "kind": "observational", "steps": {}, "blockers": []}
    for step in OBSERVED_STEPS:
        path = BUNDLE_ROOT / arm / f"step_{step:04d}" / "bundle.pt"
        if not path.is_file():
            report["blockers"].append(f"no captured bundle at step {step}; run track A first")
            continue
        bundle = torch.load(path, weights_only=False)
        branch = G.build_frozen_branch(tree, bundle, module_name=f"r7b_{arm}_{step}")
        observation = decompose(branch, bundle)
        vectors = observation.pop("_vectors")
        observation["realized_step"] = realized_step(branch, bundle, vectors)
        report["steps"][str(step)] = observation
        G.release_branch(branch)
    report["status"] = "supported" if report["steps"] and not report["blockers"] else "blocked"
    report["elapsed_s"] = round(time.time() - started, 2)
    return report


def main() -> int:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    reports = []
    for arm in ARMS:
        print(f"=== {arm} ===", flush=True)
        report = run_arm(arm)
        reports.append(report)
        (OUTPUT_ROOT / arm / "report.json").write_text(json.dumps(report, indent=2, default=str))
        print(f"{arm}: {report['status']}", flush=True)
    (OUTPUT_ROOT / "report.json").write_text(json.dumps({"arms": reports}, indent=2, default=str))
    return 0 if all(report["status"] == "supported" for report in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
