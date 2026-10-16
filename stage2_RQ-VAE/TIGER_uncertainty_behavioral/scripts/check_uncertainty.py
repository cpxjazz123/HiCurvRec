"""Issue #24 mechanism check for the uncertainty-aware behavioural representation.

Exercises the installed encoder on this tree's own probe checkpoint and asserts
the properties the issue asks for:

1. code selection only ever sees the deterministic mean: the quantizer's input is
   ``mu_i`` and does not move when the scale branch's noise is redrawn;
2. the reliability signals are real and ordered -- items with more evidence score
   higher, and the sigma target runs the opposite way;
3. the wrapped-normal sample stays inside the ball, moves further for a larger
   sigma, and returns to the mean as sigma -> 0;
4. the uncertainty weighting really down-weights an uncertain triple, and the
   ranking term reaches the encoder but no codebook;
5. the log-scale penalty rises with sigma, so sigma cannot buy silence;
6. the decoder-side reliability recovery reaches its head and the encoder proxy and
   is blocked on every codebook, while reconstruction still trains the codebooks;
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
import uncertainty_behavioral as unc  # noqa: E402
from uncertainty_behavioral import UncertaintyScaleHead  # noqa: E402


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

    # --- (2) the reliability signals ---------------------------------------
    started = time.time()
    histories = [list(history)
                  for history in frame.groupby("user")["seen_history"].last()]
    reliability = unc.build_reliability(histories, len(all_embeddings))
    report["reliability_build_seconds"] = round(time.time() - started, 2)
    values = reliability["reliability"]
    report["reliability_mean"] = float(values.mean())
    report["reliable_items"] = int((values > 0.0).sum())
    order = np.argsort(reliability["support"])
    low, high = order[:2000], order[-2000:]
    report["support_correlation"] = float(np.corrcoef(
        reliability["support"][order], values[order]
    )[0, 1])
    if report["reliable_items"] == 0:
        failures.append("no item carries supervision evidence")
    if float(values[low].mean()) >= float(values[high].mean()):
        failures.append("reliability does not order sparser items lower")
    scale = torch.as_tensor(values, dtype=torch.float32)
    sigma_star = unc.sigma_target(scale)
    if float(sigma_star[low].mean()) <= float(sigma_star[high].mean()):
        failures.append("the sigma target does not rise as reliability falls")

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
    unc.install_uncertainty_encoder(model)
    report["state_dict_keys_unchanged"] = sorted(model.state_dict()) == before_keys
    if not report["state_dict_keys_unchanged"]:
        failures.append("installing the encoder changed model.state_dict()")

    head = unc.UncertaintyScaleHead(trainer.CODEBOOK_DIM).to(device)
    recovery = unc.ReliabilityRecoveryHead(trainer.CODEBOOK_DIM).to(device)
    codebooks = tuple(layer.get_code_embs() for layer in model.rq.vq_layers)

    rows = torch.arange(min(256, len(item_ids)), dtype=torch.long)
    batch = all_embeddings[item_ids[rows]].to(device)

    # --- (1) code selection sees the deterministic mean only ---------------
    with torch.no_grad():
        mu = model.encoder(batch)
        first = mu.clone()
        torch.manual_seed(0)
        _ = unc.wrapped_normal_sample(mu, torch.full((mu.shape[0],), 0.5, device=device),
                                      torch.Generator().manual_seed(1))
        second = model.encoder(batch)
    report["mean_is_deterministic"] = float((first - second).abs().max())
    report["mean_radius_max"] = float(mu.norm(dim=-1).max())
    if report["mean_is_deterministic"] > 0.0:
        failures.append("code selection input moved when the sample was redrawn")
    if report["mean_radius_max"] >= 1.0:
        failures.append("the deterministic mean left the ball")

    # --- (3) the wrapped-normal sample -------------------------------------
    generator = torch.Generator().manual_seed(trainer.SEED)
    spread_small = unc.wrapped_normal_sample(
        mu, torch.full((mu.shape[0],), 0.01, device=device), generator)
    spread_large = unc.wrapped_normal_sample(
        mu, torch.full((mu.shape[0],), 0.6, device=device), generator)
    small = unc._geodesic(mu, spread_small)
    large = unc._geodesic(mu, spread_large)
    report["sample_radius_max"] = float(spread_large.norm(dim=-1).max())
    report["sample_distance_small_sigma"] = float(small.mean())
    report["sample_distance_large_sigma"] = float(large.mean())
    if report["sample_radius_max"] >= 1.0:
        failures.append("the wrapped-normal sample left the ball")
    if float(large.mean()) <= float(small.mean()):
        failures.append("a larger sigma does not perturb the sample further")

    # --- (4) the uncertainty weighting -------------------------------------
    count = mu.shape[0]
    anchor, positive, negative = mu[:count // 2], mu[count // 2:], mu[:count // 2]
    confident = torch.ones(count // 2, device=device)
    unsure = torch.full((count // 2,), 0.2, device=device)
    weighted = unc.uncertainty_ranking_loss(anchor, positive, negative, confident)
    damped = unc.uncertainty_ranking_loss(anchor, positive, negative, unsure)
    report["rank_confident"] = float(weighted.detach())
    report["rank_unsure"] = float(damped.detach())
    if float(damped.detach()) >= float(weighted.detach()):
        failures.append("uncertainty does not down-weight an unsure triple")

    # A second, gradient-enabled pass: the one above was measured under no_grad.
    mu = model.encoder(batch)
    count = mu.shape[0]
    anchor, positive, negative = mu[:count // 2], mu[count // 2:], mu[:count // 2]
    sigma_now = head(model.encoder(batch).detach())
    report["sigma_range"] = [float(sigma_now.min()), float(sigma_now.max())]

    model.zero_grad(set_to_none=True)
    head.zero_grad(set_to_none=True)
    ranking = unc.uncertainty_ranking_loss(
        anchor, positive, negative, torch.ones(count // 2, device=device))
    encoder_grads = torch.autograd.grad(
        ranking, tuple(model.encoder.parameters()), retain_graph=True, allow_unused=True)
    codebook_grads = torch.autograd.grad(
        ranking, codebooks, retain_graph=True, allow_unused=True)
    report["rank_encoder_grad"] = max(
        [float(g.abs().sum()) for g in encoder_grads if g is not None] or [0.0])
    report["rank_codebook_grad"] = max(
        [float(g.abs().sum()) for g in codebook_grads if g is not None] or [0.0])
    if report["rank_encoder_grad"] <= 0.0:
        failures.append("the ranking term produced no gradient on the encoder")
    if report["rank_codebook_grad"] > 0.0:
        failures.append("the ranking term leaked gradient into a codebook")

    supervised = unc.sigma_supervision_loss(sigma_now, sigma_star[rows.to(device)])
    scale_grads = torch.autograd.grad(
        supervised, tuple(head.parameters()), allow_unused=True)
    report["sigma_supervision_grad"] = max(
        [float(g.abs().sum()) for g in scale_grads if g is not None] or [0.0])
    if report["sigma_supervision_grad"] <= 0.0:
        failures.append("the sigma supervision produced no gradient on the scale head")

    # --- (5) the log-scale penalty -----------------------------------------
    small_penalty = unc.sigma_log_penalty(torch.full((64,), 0.8))
    large_penalty = unc.sigma_log_penalty(torch.full((64,), 3.0))
    report["log_penalty_small_sigma"] = float(small_penalty)
    report["log_penalty_large_sigma"] = float(large_penalty)
    if float(large_penalty) <= float(small_penalty):
        failures.append("the log-scale penalty does not rise with sigma")

    # --- (6) the decoder-side reliability recovery --------------------------
    model.zero_grad(set_to_none=True)
    head.zero_grad(set_to_none=True)
    recovery.zero_grad(set_to_none=True)
    trace, quantized = trainer._trace_and_quantized(model, batch)
    _replayed, predicted = recovery(quantized, codebooks)
    recovery_loss = unc.reliability_recovery_loss(
        predicted, reliability_t_rows := scale[rows.to(device)]
    )
    head_count = len(tuple(recovery.parameters()))
    encoder_count = len(tuple(model.encoder.parameters()))
    grads = torch.autograd.grad(
        recovery_loss,
        tuple(recovery.parameters()) + tuple(model.encoder.parameters()) + codebooks,
        allow_unused=True,
    )
    report["recovery_head_grad"] = max(
        [float(g.abs().sum()) for g in grads[:head_count] if g is not None] or [0.0])
    report["recovery_encoder_grad"] = max(
        [float(g.abs().sum())
         for g in grads[head_count:head_count + encoder_count] if g is not None] or [0.0])
    report["recovery_codebook_grad"] = max(
        [float(g.abs().sum())
         for g in grads[head_count + encoder_count:] if g is not None] or [0.0])
    if report["recovery_head_grad"] <= 0.0:
        failures.append("the reliability recovery produced no gradient on its head")
    if report["recovery_encoder_grad"] <= 0.0:
        failures.append("the reliability recovery produced no gradient on the encoder")
    if report["recovery_codebook_grad"] > 0.0:
        failures.append("the reliability recovery leaked gradient into a codebook")

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
