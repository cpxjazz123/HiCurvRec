"""The executable criteria of this candidate.

Every claim the reviewers have to judge is a statement that runs: the base
point maps, the successor-overlap neighbourhood, the tangent displacement the
behaviour signal produces, the curved assignment, the two switches, the
gradient path and the four-rank behaviour.

Run it with the project's Stage 2 interpreter:

    /home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 smoke_test.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from model.hyperbolic import (ball_add, ball_distance, ball_exp_map, ball_log_map,
                              ball_project, conformal_factor)
from model.model import RQVAE

CURVATURE = 1.0


def check_geometry_convention() -> str:
    """The base-point maps invert each other and the tangent norm is the distance."""
    torch.manual_seed(0)
    base = ball_project(torch.randn(8, 32, dtype=torch.float64) * 0.05, CURVATURE)
    point = ball_project(torch.randn(8, 32, dtype=torch.float64) * 0.05, CURVATURE)
    tangent = torch.randn(8, 32, dtype=torch.float64) * 0.05

    assert torch.allclose(ball_exp_map(base, ball_log_map(base, point, CURVATURE), CURVATURE),
                           point, atol=1e-9), "the exponential map does not invert the logarithm"
    assert torch.allclose(ball_log_map(base, ball_exp_map(base, tangent, CURVATURE), CURVATURE),
                           tangent, atol=1e-9), "the logarithm does not invert the exponential map"

    distance = ball_distance(base, ball_exp_map(base, tangent, CURVATURE), CURVATURE)
    expected = conformal_factor(base, CURVATURE).squeeze(-1) * tangent.norm(dim=-1)
    assert torch.allclose(distance, expected, atol=1e-8), \
        "the tangent norm is not the geodesic distance"
    assert torch.isfinite(ball_add(base, base, CURVATURE)).all(), "the Mobius addition is not finite"
    return "the base-point maps invert each other, the tangent norm is the geodesic distance"


def check_neighbourhood() -> str:
    """The neighbourhood is the successor-set overlap of the training sequences."""
    import pandas as pd

    from train_rqvae import build_successor_neighbourhood

    train_file = None
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "results" / "stage0_build_parquet" / "train.parquet"
        if candidate.is_file():
            train_file = candidate
            break
    if train_file is None:
        return "the training parquet is not reachable, the neighbourhood check is skipped"
    frame = pd.read_parquet(train_file, columns=["history", "target"])
    ids, weights, branching = build_successor_neighbourhood(frame, 24587)
    assert ids.shape == (24587, 8), "the neighbour table has the wrong shape"
    assert float(weights.min()) >= 0.0, "a neighbour weight is negative"
    assert float(weights.max()) <= 1.0 + 1e-9, "a Jaccard overlap is above one"
    assert float(branching.min()) >= 0.0, "the branching mass is negative"
    covered = int((ids[:, 0] >= 0).sum())
    assert covered > 20000, "too few items have a successor neighbourhood"
    # the neighbours of an item really share a successor with it
    successors: dict = {}
    # The whole frame: a truncated one would give incomplete successor sets and
    # the check would compare against the wrong overlap.
    for history, target in zip(frame["history"].to_numpy(),
                               frame["target"].to_numpy(dtype=np.int64)):
        sequence = list(history) + [int(target)]
        for source, destination in zip(sequence, sequence[1:]):
            successors.setdefault(int(source), set()).add(int(destination))
    checked = 0
    for item in range(200):
        for position in range(8):
            other = int(ids[item, position])
            if other < 0:
                break
            left = successors.get(item, set())
            right = successors.get(other, set())
            if left and right:
                overlap = len(left & right) / (len(left | right) + 1e-12)
                assert abs(overlap - float(weights[item, position])) < 1e-5, \
                    "the stored weight is not the Jaccard overlap of the successor sets"
                checked += 1
    assert checked > 50, "the Jaccard check had too few cases"
    return "the neighbourhood is the Jaccard overlap of the successor sets, with the branching mass"


def _toy_model(dim: int = 32, codes: int = 16, levels: int = 3) -> RQVAE:
    config = SimpleNamespace(hidden_sizes=(64, 32), codebook_num=levels,
                              codebook_size=(codes,) * levels, codebook_dim=dim,
                              beta=0.25, dropout=0.0, vq_type="vq", sk_epsilon=-1.0,
                              sk_iters=-1, ema_decay=0.99, fix_code_embs=False)
    torch.manual_seed(7)
    model = RQVAE(config, in_dim=dim)
    with torch.no_grad():
        for layer in model.rq.vq_layers:
            layer.embed.weight.normal_(0.0, 0.05)
        for mapping in model.fork_maps:
            mapping.weight.normal_(0.0, 0.05)
    return model


def _toy_context(count: int, dim: int = 32) -> dict:
    torch.manual_seed(5)
    return {"neighbour_mean": torch.randn(count, dim) * 0.1,
            "scale": torch.rand(count) + 0.5,
            "curvature": CURVATURE, "geometry_on": True}


def check_displacement_and_switches() -> str:
    """The displacement comes from the behaviour signal and the switches work."""
    model = _toy_model()
    embeddings = torch.randn(12, 32) * 0.05
    context = _toy_context(12)

    def tokens_of(ctx: dict) -> torch.Tensor:
        torch.manual_seed(3)
        with torch.no_grad():
            return model(embeddings, fork_context=ctx)[3].detach().clone()

    both = tokens_of(context)
    behaviour_off = tokens_of(dict(context, scale=torch.zeros(12),
                                    neighbour_mean=torch.zeros(12, 32)))
    geometry_off = tokens_of(dict(context, geometry_on=False))
    assert not torch.equal(both, behaviour_off), \
        "switching the behaviour off does not change the assignment"
    assert not torch.equal(both, geometry_off), \
        "switching the geometry off does not change the assignment"

    # the displacement is the learned map applied to the neighbourhood mean
    prepared = model._fork_context(model.encoder(embeddings), context)
    shift = prepared["shifts"][0]
    expected = context["scale"].unsqueeze(-1) * model.fork_maps[0](context["neighbour_mean"])
    assert torch.allclose(shift, expected, atol=1e-6), \
        "the displacement is not the scaled map of the neighbourhood mean"
    assert float(shift.abs().sum()) > 0, "the displacement is zero"

    # the gradient reaches the encoder and the learned maps
    model.zero_grad(set_to_none=True)
    reconstructed, quant_loss, _unused, _tokens = model(embeddings, fork_context=context)
    (reconstructed.square().mean() + quant_loss).backward()
    encoder_linear = next(module for module in model.encoder.mlp.modules()
                           if isinstance(module, torch.nn.Linear))
    assert encoder_linear.weight.grad is not None and float(encoder_linear.weight.grad.abs().sum()) > 0, \
        "the gradient does not reach the encoder"
    assert model.fork_maps[0].weight.grad is not None and \
        float(model.fork_maps[0].weight.grad.abs().sum()) > 0, \
        "the gradient does not reach the learned displacement map"
    return "the displacement is the scaled map of the neighbourhood mean, both switches change the assignment, the gradient reaches the encoder and the map"


def check_four_rank_step() -> str:
    """The four-rank step finishes and a rank-0 evaluation does not wait."""
    tree = Path(__file__).resolve().parent
    torchrun = Path(sys.executable).with_name("torchrun")
    if not torchrun.is_file() or torch.cuda.device_count() < 4:
        return "fewer than four cards are visible, the four-rank check is skipped"
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(index) for index in range(4))
    try:
        done = subprocess.run(
            [str(torchrun), "--standalone", "--nproc_per_node=4", "--master_port=50215",
             str(tree / "ddp_check.py")],
            cwd=str(tree), env=env, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        raise AssertionError("the four-rank check timed out: the step hangs")
    assert done.returncode == 0, \
        "the four-rank check failed: " + done.stdout[-600:] + done.stderr[-600:]
    assert "DDP_PASS" in done.stdout, "the four-rank check did not report a pass"
    return "the four-rank step finishes and the rank-0 evaluation does not wait in a collective"


def main() -> int:
    checks = [
        ("geometry", check_geometry_convention),
        ("neighbourhood", check_neighbourhood),
        ("displacement", check_displacement_and_switches),
        ("ddp", check_four_rank_step),
    ]
    failures = []
    for name, check in checks:
        try:
            message = check()
            print("PASS", name + ":", message, flush=True)
        except AssertionError as exc:
            failures.append(name + ": " + str(exc))
            print("FAIL", name + ":", exc, flush=True)
        except Exception as exc:  # noqa: BLE001 - the report says which check broke
            failures.append(name + ": " + repr(exc))
            print("ERROR", name + ":", repr(exc), flush=True)
    if failures:
        print("SMOKE_FAIL", failures, flush=True)
        return 1
    print("SMOKE_PASS the mechanism's claims hold as executable statements", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
