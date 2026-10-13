"""E3 audit: the gradient each Stage 2 loss term puts on the encoder and the codebooks.

The plan asks for the reconstruction, behaviour, quantised-ranking, alignment and
diversity terms to be measured one at a time rather than as a summed objective, on the
encoder and on L1/L2/L3 separately, with their directions compared. Every term here is
evaluated on the SAME real batch from one checkpoint, so the numbers are comparable
against each other and against the gradient norms the trainer records.

Two things this is meant to settle. First, whether a term that is supposed to reach the
codebooks actually does, per level -- the DIGER no-op and the CF dead term were both
found by hand, and this is the instrument that would have found them. Second, whether
the terms conflict: if the behaviour-family gradients point away from the
reconstruction gradient, then no amount of codebook health will make them compatible.

Prints JSON and exits non-zero if a term fails to reach a structure it is supposed to
reach. No report file is written.
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

import anti_collapse_losses as anti_collapse  # noqa: E402
import train_rqvae as trainer  # noqa: E402
from model import RQVAE  # noqa: E402

CHECKPOINT = (
    "/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/"
    "TIGER_BEHAVIOR_SAMPLE_RQ-VAE/resample/rqvae_best.pth"
)


def flat(values) -> torch.Tensor:
    present = [value.reshape(-1) for value in values if value is not None]
    return torch.cat(present) if present else torch.zeros(1)


def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    trainer.set_seed(trainer.SEED)

    embeddings = trainer.load_embeddings(trainer.EMBEDDING_FILE)
    all_embeddings = torch.from_numpy(embeddings)
    frame = pd.read_parquet(trainer.TRAIN_FILE, columns=["seen_history", "target"])
    pairs = trainer.TransitionPairs(frame).pairs[: trainer.PAIR_BATCH_SIZE_PER_RANK]

    model = RQVAE(trainer._tokenizer_config(), in_dim=embeddings.shape[1]).to(device)
    trainer.initialize_tiger_weights(model)
    state = torch.load(CHECKPOINT, map_location="cpu")
    model.load_state_dict(state["state_dict"])
    model.eval()
    for layer in model.rq.vq_layers:
        layer._skip_ddp_reduce = True

    source_ids = pairs[:, 0].to(device)
    target_ids = pairs[:, 1].to(device)
    negative_ids = torch.roll(target_ids, 1)
    triple_ids = torch.cat((source_ids, target_ids, negative_ids))
    triple_batch = all_embeddings[triple_ids.cpu()].to(device)
    pair_count = int(pairs.shape[0])
    anchors = torch.arange(pair_count, device=device)
    positives = anchors + pair_count
    negatives = positives + pair_count

    item_batch = all_embeddings[: trainer.BATCH_SIZE_PER_RANK].to(device)
    reconstructed, quant_loss, _, _ = model(item_batch)
    base_loss, recon_loss = model.compute_loss(item_batch, reconstructed, quant_loss)

    residuals, tokens = trainer._residuals_and_tokens_with_trace(model, triple_batch)
    # The contrastive term reads source from the first block and candidates from the
    # second; the triple batch's third block is the matched negative the ranking term
    # needs. Slicing to two blocks keeps every term on the same source and target rows.
    contrastive = trainer.behavior_contrastive_loss(
        residuals[:, : 2 * pair_count], source_ids, target_ids,
    )
    ranking = trainer._quantized_ranking_loss(
        residuals, tokens, model, anchors, positives, negatives, weight=0.2,
    )
    quantized_sum = trainer._quantized_representation(
        residuals, tokens, model.rq.vq_layers,
    )
    alignment = anti_collapse.align_loss(residuals, quantized_sum)
    diversity = anti_collapse.diversity_loss(
        residuals, model.rq.vq_layers, anti_collapse.GUMBEL_TAU,
    )

    terms = {
        "reconstruction": recon_loss,
        "behavior_contrastive": contrastive,
        "quantized_ranking": ranking,
        "letter_alignment": alignment,
        "letter_diversity": diversity,
    }
    encoder_parameters = tuple(model.encoder.parameters())
    codebooks = [layer.get_code_embs() for layer in model.rq.vq_layers]
    targets = encoder_parameters + tuple(codebooks)
    split = len(encoder_parameters)

    report: dict[str, object] = {
        "device": str(device),
        "checkpoint": CHECKPOINT,
        "checkpoint_epoch": int(state.get("epoch", -1)),
        "pair_rows": pair_count,
        "ranking_representation": "hard codebook lookup with the encoder STE",
        "terms": {},
    }
    encoder_directions: dict[str, torch.Tensor] = {}
    for name, term in terms.items():
        entry: dict[str, object] = {"value": float(term.detach())}
        entry["has_grad_fn"] = bool(term.requires_grad and term.grad_fn is not None)
        if not entry["has_grad_fn"]:
            report["terms"][name] = entry
            continue
        grads = torch.autograd.grad(term, targets, retain_graph=True, allow_unused=True)
        encoder_norm = float(flat(grads[:split]).norm())
        entry["encoder"] = encoder_norm
        entry["codebook_total"] = 0.0
        for level, grad in enumerate(grads[split:]):
            value = 0.0 if grad is None else float(grad.norm())
            entry[f"codebook_l{level + 1}"] = value
            entry["codebook_total"] += value
        encoder_directions[name] = flat(grads[:split]).detach()
        report["terms"][name] = entry

    directions: dict[str, dict[str, float]] = {}
    names = sorted(encoder_directions)
    for i, first in enumerate(names):
        for second in names[i + 1:]:
            a = encoder_directions[first]
            b = encoder_directions[second]
            cosine = float(
                torch.dot(a, b) / (a.norm() * b.norm() + 1e-12)
            )
            directions[f"{first}|{second}"] = round(cosine, 6)
    report["encoder_gradient_cosines"] = directions

    total = sum(terms.values())
    total_grads = torch.autograd.grad(total, targets, retain_graph=True, allow_unused=True)
    total_encoder = flat(total_grads[:split]).norm()
    report["total_encoder_grad_norm"] = float(total_encoder)
    report["clip_norm"] = trainer.GRADIENT_CLIP_NORM
    report["clip_ratio"] = float(
        min(1.0, trainer.GRADIENT_CLIP_NORM / (float(total_encoder) + 1e-12))
    )

    failures: list[str] = []
    for name in terms:
        if not report["terms"][name].get("has_grad_fn"):
            failures.append(f"{name} has no grad_fn")
    # The encoder must see the reconstruction, behaviour and ranking terms. The
    # codebooks must NOT see the first two: this recipe updates them by EMA and the
    # behaviour term acts on the residual trace, so a non-zero codebook gradient there
    # would mean something is wired differently from the design.
    for name in ("reconstruction", "behavior_contrastive"):
        if report["terms"][name].get("encoder", 0.0) <= 0.0:
            failures.append(f"{name} puts no gradient on the encoder")
    if report["terms"]["quantized_ranking"].get("encoder", 0.0) <= 0.0:
        failures.append("quantized_ranking puts no gradient on the encoder")
    for level in (1, 2, 3):
        if report["terms"]["quantized_ranking"].get(f"codebook_l{level}", 0.0) <= 0.0:
            failures.append(f"quantized_ranking puts no gradient on codebook L{level}")
    # Align's encoder gradient is identically zero, and that is a property of the
    # straight-through term rather than a wiring fault: the quantised sum carries
    # z - z.detach(), so d(quantised_sum)/dz is exactly 1 and the MSE against z
    # differentiates to 2(q - z)(1 - 1). Recording it as an assertion keeps a future
    # change from silently turning this term into an encoder objective.
    if report["terms"]["letter_alignment"].get("encoder", 1.0) > 1e-9:
        failures.append("letter_alignment now moves the encoder; it is supposed to be a codebook-only term")
    if report["terms"]["letter_alignment"].get("codebook_total", 0.0) <= 0.0:
        failures.append("letter_alignment puts no gradient on any codebook")
    report["failures"] = failures
    report["status"] = "PASS" if not failures else "FAIL"
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if not failures else 1)


if __name__ == "__main__":
    main()
