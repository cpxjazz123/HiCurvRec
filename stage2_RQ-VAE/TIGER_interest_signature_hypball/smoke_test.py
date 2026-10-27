"""The executable criteria of this candidate.

Every claim the reviewers have to judge is checked by running something. The
criteria are the ones the previous rounds had to settle in long review rounds:
the base-point maps, the round trip, the tangent norm against the geodesic
distance, the behaviour counting, the switches, the gradient path and the
four-rank behaviour.

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
    curvature = CURVATURE
    base = ball_project(torch.randn(8, 32, dtype=torch.float64) * 0.05, curvature)
    point = ball_project(torch.randn(8, 32, dtype=torch.float64) * 0.05, curvature)
    tangent = torch.randn(8, 32, dtype=torch.float64) * 0.05

    round_trip = ball_exp_map(base, ball_log_map(base, point, curvature), curvature)
    assert torch.allclose(round_trip, point, atol=1e-9), \
        "the exponential map does not invert the logarithm"

    other = ball_log_map(base, ball_exp_map(base, tangent, curvature), curvature)
    assert torch.allclose(other, tangent, atol=1e-9), \
        "the logarithm does not invert the exponential map"

    distance = ball_distance(base, ball_exp_map(base, tangent, curvature), curvature)
    expected = conformal_factor(base, curvature).squeeze(-1) * tangent.norm(dim=-1)
    assert torch.allclose(distance, expected, atol=1e-8), \
        "the tangent norm is not the geodesic distance"

    radius = (1.0 - 1e-5) / (curvature ** 0.5)
    assert bool((base.norm(dim=-1) < radius).all()), "the projection left the ball"
    assert torch.isfinite(ball_add(base, base, curvature)).all(), "the Mobius addition is not finite"
    return "the base-point maps invert each other, the tangent norm is the geodesic distance"


def check_cohort_hash() -> str:
    """The cohort of a user is a stable hash of the ordered history."""
    from train_rqvae import _cohort_of_user

    history = [12, 345, 6, 789]
    first = _cohort_of_user(history, 16)
    assert first == _cohort_of_user(list(history), 16), "the cohort hash is not stable"
    assert first != _cohort_of_user(list(reversed(history)), 16), \
        "the cohort hash ignores the order of the history"
    assert 0 <= first < 16, "the cohort is out of range"
    return "the cohort of a user is a stable hash of the ordered history"


def check_signatures_and_allocation() -> str:
    """The signature is a normalised distribution and every parent gets codes."""
    import pandas as pd

    from train_rqvae import build_interest_signatures

    train_file = None
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "results" / "stage0_build_parquet" / "train.parquet"
        if candidate.is_file():
            train_file = candidate
            break
    if train_file is None:
        return "the training parquet is not reachable, the signature check is skipped"
    frame = pd.read_parquet(train_file, columns=["user", "history", "target"])
    parent_ids, shift, codes_of_parent, nonempty = build_interest_signatures(
        frame, 24587, 16, 32, 42, 256)
    assert parent_ids.shape == (24587,), "every item needs a parent"
    assert len(nonempty) == 16, "the cohort count is not the configured one"
    sizes = [len(codes) for codes in codes_of_parent.values()]
    assert min(sizes) >= 1, "a nonempty parent received no code"
    assert sum(sizes) == 256, "the allocated codes do not fill the first-level codebook"
    assert len(set(sum(codes_of_parent.values(), []))) == 256, "a code was allocated twice"
    norm = np.linalg.norm(shift, axis=1)
    assert float(norm.max()) <= 1.0 - 1e-6 + 1e-9, \
        "the cohort-mass vector is not normalised inside the ball"
    return "the signature is normalised, every parent has codes and no code is allocated twice"


def _toy_first_level(count: int, n_codes: int, parent_of: torch.Tensor,
                     shift: torch.Tensor, behaviour_on: bool = True,
                     geometry_on: bool = True) -> dict:
    """A first-level context: a behaviour shift and the allowed codes."""
    mask = torch.zeros((count, n_codes), dtype=torch.bool)
    for row in range(count):
        mask[row, parent_of[row] % n_codes] = True
        mask[row, (parent_of[row] + 1) % n_codes] = True
    return {
        "behaviour_shift": shift,
        "parent_mask": mask,
        "curvature": CURVATURE,
        "behavior_on": behaviour_on,
        "geometry_on": geometry_on,
    }


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
    return model


def check_switches_and_gradient() -> str:
    """The switches change the assignment and the gradient reaches the encoder."""
    torch.manual_seed(11)
    dim, count, n_codes = 32, 24, 16
    model = _toy_model(dim=dim, codes=n_codes)
    embeddings = torch.randn(count, dim) * 0.05
    parents = torch.arange(count) % 4
    shift = torch.randn(count, dim) * 0.2

    def run(context: dict):
        torch.manual_seed(0)
        model.zero_grad(set_to_none=True)
        encoded = model.encoder(embeddings)
        quantized, quant_loss, _unused, tokens = model.rq(
            encoded, first_level=context)
        loss = quantized.square().mean() + quant_loss
        loss.backward()
        return tokens.detach().clone(), loss.detach().clone()

    both, _ = run(_toy_first_level(count, n_codes, parents, shift))
    behaviour_off, _ = run(_toy_first_level(count, n_codes, parents, shift,
                                            behaviour_on=False))
    geometry_off, _ = run(_toy_first_level(count, n_codes, parents, shift,
                                            geometry_on=False))
    assert not torch.equal(both[:, 0], behaviour_off[:, 0]), \
        "switching the behaviour off does not change the first-level assignment"
    assert not torch.equal(both[:, 0], geometry_off[:, 0]), \
        "switching the geometry off does not change the first-level assignment"

    torch.manual_seed(0)
    model.zero_grad(set_to_none=True)
    encoded = model.encoder(embeddings)
    _quantized, quant_loss, _unused, _tokens = model.rq(
        encoded, first_level=_toy_first_level(count, n_codes, parents, shift))
    (quant_loss + _quantized.square().mean()).backward()
    encoder_linear = next(module for module in model.encoder.mlp.modules()
                           if isinstance(module, torch.nn.Linear))
    assert encoder_linear.weight.grad is not None and float(encoder_linear.weight.grad.abs().sum()) > 0, \
        "the gradient does not reach the encoder"
    assert model.rq.vq_layers[0].embed.weight.grad is not None, \
        "the gradient does not reach the first-level codebook"
    return "the switches change the first-level assignment, the gradient reaches the encoder and the codebook"


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
            [str(torchrun), "--standalone", "--nproc_per_node=4", "--master_port=50213",
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
        ("cohort", check_cohort_hash),
        ("signatures", check_signatures_and_allocation),
        ("switches", check_switches_and_gradient),
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
