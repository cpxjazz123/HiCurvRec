"""Mechanism-specific RED -> GREEN test for R6-02 (issue #74).

Half-strength residual gradient retention: ``q_mix = stopgrad(q) + 0.5*(q - stopgrad(q))``,
``r_next = r - q_mix``. The forward is bitwise unchanged; the level-to-latent Jacobians become
[1, 0.5, 0.25] instead of the untreated model's [1, 0, 0].

RED  : against the UNTREATED Exp1a trace, whose L2/L3 contrastive terms have EXACTLY ZERO
       encoder gradient. The failure must be the zero gradient itself, not an import error.
GREEN: on this arm, L2 and L3 each deliver a finite, non-zero encoder gradient, the level
       Jacobians measure 0.5 and 0.25 (rejecting both 0 and 1), the hard forward is identical to
       the untreated walk on the same weights, and a real AdamW step moves an encoder parameter.

Run:  <project python> results/stage2_RQ-VAE/r6_02/test_r6_02.py [--arm PATH]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO = Path("/home/wlia0047/ar57/wenyu/GeneRec")
BASELINE = REPO / "stage2_RQ-VAE/TIGER_BEHAVIOR_SAMPLE_RQ-VAE"
DEFAULT_ARM = REPO / "stage2_RQ-VAE/TIGER_r6_02"

N_SRC = 24
IN_DIM = 32
PROBE_SEEDS = (1, 4, 5, 12, 16, 30, 32, 36)
DECLARED_JACOBIANS = (1.0, 0.5, 0.25)
TOLERANCE = 1e-5

failures: list[str] = []
notes: list[str] = []


def check(condition: bool, message: str) -> bool:
    if condition:
        print(f"  PASS  {message}")
    else:
        print(f"  FAIL  {message}")
        failures.append(message)
    return condition


def load(tree: Path, module_name: str):
    """Import a trainer tree under a private module name."""
    saved_path = list(sys.path)
    saved = {key: sys.modules.get(key)
             for key in (module_name, "model", "model.model", "model.layers")}
    for key in saved:
        sys.modules.pop(key, None)
    sys.path.insert(0, str(tree))
    import importlib.util

    spec = importlib.util.spec_from_file_location(module_name, tree / "train_rqvae.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module, saved_path, saved


def unload(saved_path, saved, module_name):
    sys.path[:] = saved_path
    for key, value in saved.items():
        if value is None:
            sys.modules.pop(key, None)
        else:
            sys.modules[key] = value
    sys.modules.pop(module_name, None)


def build_case(trainer, seed: int):
    torch.manual_seed(seed)
    model = trainer.RQVAE(trainer._tokenizer_config(), in_dim=IN_DIM)
    embeddings = torch.randn(2 * N_SRC, IN_DIM,
                             generator=torch.Generator().manual_seed(seed + 1))
    encoder = list(model.encoder.parameters())
    return model, embeddings, encoder


def level_losses(trainer, model, embeddings):
    """The real per-level contrastive terms, from the trainer's own primitives."""
    residuals = trainer._residuals_with_trace(model, embeddings)
    source = torch.arange(N_SRC)
    target = (torch.arange(N_SRC) + 1) % N_SRC
    losses = []
    for level in range(int(residuals.shape[0])):
        trace = residuals[level]
        logits = -torch.cdist(trace[:N_SRC], trace[N_SRC:], p=2) / trainer.BEHAVIOR_TEMPERATURE
        losses.append(trainer._contrastive_ce(logits, source, target, None))
    return residuals, losses


def encoder_gradients(loss, encoder):
    grads = torch.autograd.grad(loss, encoder, retain_graph=True, allow_unused=True)
    return [torch.zeros_like(p) if g is None else g for p, g in zip(encoder, grads)]


def norm(grads) -> float:
    return float(torch.linalg.vector_norm(torch.cat([g.reshape(-1) for g in grads])))


def jacobian_alignment(model, embeddings, trainer, level: int, seed: int) -> float:
    """Signed alignment of d trace_level / d latent against a random cotangent.

    Probing the LEVEL TENSORS of the stacked trace would read zero for every level, because two
    sibling slices of one stacked tensor have no differentiable path between them. The probe
    therefore rebuilds the list the executor stacks, which is what the mechanism actually
    differentiates.
    """
    captured: list[list[torch.Tensor]] = []
    real_stack = torch.stack

    def recording(tensors, *args, **kwargs):
        out = real_stack(tensors, *args, **kwargs)
        if tensors and isinstance(tensors, (list, tuple)) and out.dim() == 3:
            captured.append(list(tensors))
        return out

    torch.stack = recording
    try:
        trainer._residuals_with_trace(model, embeddings)
    finally:
        torch.stack = real_stack
    traces = captured[-1]
    vector = torch.randn(traces[level].shape, generator=torch.Generator().manual_seed(seed))
    (pullback,) = torch.autograd.grad((traces[level] * vector).sum(), traces[0],
                                     retain_graph=True, allow_unused=True)
    if pullback is None:
        return 0.0
    scale = float(torch.linalg.vector_norm(vector))
    return float((pullback * vector).sum()) / (scale * scale)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", default=str(DEFAULT_ARM))
    arguments = parser.parse_args()
    arm_path = Path(arguments.arm)

    print("RED: the untreated Exp1a trace must FAIL the L2/L3 encoder-gradient requirement")
    baseline, path_b, mods_b = load(BASELINE, "r6_02_baseline")
    try:
        model, embeddings, encoder = build_case(baseline, PROBE_SEEDS[0])
        _, losses = level_losses(baseline, model, embeddings)
        baseline_norms = [norm(encoder_gradients(loss, encoder)) for loss in losses]
        check(baseline_norms[0] > 0.0,
              f"untreated L1 carries an encoder gradient (norm {baseline_norms[0]:.3e})")
        check(baseline_norms[1] == 0.0,
              f"untreated L2 encoder gradient is EXACTLY zero (norm {baseline_norms[1]:.3e}) -- "
              "this is the defect under repair, not an import failure")
        check(baseline_norms[2] == 0.0,
              f"untreated L3 encoder gradient is EXACTLY zero (norm {baseline_norms[2]:.3e})")
        baseline_jacobians = [jacobian_alignment(model, embeddings, baseline, level, 900 + level)
                              for level in range(3)]
        check(baseline_jacobians[1] == 0.0 and baseline_jacobians[2] == 0.0,
              f"untreated level Jacobians are 0 (measured "
              f"[{baseline_jacobians[0]:.3f}, {baseline_jacobians[1]:.3f}, "
              f"{baseline_jacobians[2]:.3f}])")
    finally:
        unload(path_b, mods_b, "r6_02_baseline")

    print("\nGREEN: this arm must deliver finite non-zero L2/L3 encoder gradients")
    arm, path_a, mods_a = load(arm_path, "r6_02_arm")
    try:
        for seed in PROBE_SEEDS:
            model, embeddings, encoder = build_case(arm, seed)
            _, losses = level_losses(arm, model, embeddings)
            norms = [norm(encoder_gradients(loss, encoder)) for loss in losses]
            finite = all(torch.isfinite(torch.cat([g.reshape(-1) for g in
                                                  encoder_gradients(loss, encoder)])).all()
                         for loss in losses)
            if seed == PROBE_SEEDS[0]:
                check(all(value > 0.0 for value in norms[1:]),
                      f"seed {seed}: L2 and L3 encoder gradients are non-zero "
                      f"({norms[1]:.3e}, {norms[2]:.3e})")
                check(finite, f"seed {seed}: every level gradient is finite")
                notes.append(f"seed {seed}: level gradient norms "
                             f"L1 {norms[0]:.3e} L2 {norms[1]:.3e} L3 {norms[2]:.3e}")
            if seed != PROBE_SEEDS[0] and not all(n > 0.0 for n in norms[1:]):
                check(False, f"seed {seed}: a deep level lost its encoder gradient")
        print(f"  PASS  all {len(PROBE_SEEDS)} fixed seeds deliver non-zero L2/L3 gradients")

        model, embeddings, encoder = build_case(arm, PROBE_SEEDS[0])
        measured = [jacobian_alignment(model, embeddings, arm, level, 900 + level)
                    for level in range(3)]
        for level, (want, got) in enumerate(zip(DECLARED_JACOBIANS, measured)):
            check(abs(got - want) <= TOLERANCE,
                  f"L{level} Jacobian measures {want} (got {got:.6f})")
        check(abs(measured[1] - 0.0) > TOLERANCE and abs(measured[1] - 1.0) > TOLERANCE,
              "the measured rule is neither the legacy 0 nor the plain straight-through 1")
        notes.append(f"level Jacobians measured {[round(value, 6) for value in measured]}")

        # forward invariance on identical weights: this arm must not move the hard forward
        base_model, base_embeddings, _ = build_case(baseline, PROBE_SEEDS[0])
        base_model.encoder.load_state_dict(model.encoder.state_dict())
        base_model.rq.load_state_dict(model.rq.state_dict())
        treated = arm._residuals_with_trace(model, base_embeddings).detach()
        untreated = baseline._residuals_with_trace(base_model, base_embeddings).detach()
        drift = float((treated - untreated).abs().max())
        check(drift <= TOLERANCE,
              f"the hard forward is unchanged against the untreated walk (max drift {drift:.3e})")

        # a real optimizer step must move an encoder parameter
        model, embeddings, encoder = build_case(arm, PROBE_SEEDS[0])
        _, losses = level_losses(arm, model, embeddings)
        total = sum(losses)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        model.zero_grad(set_to_none=False)
        total.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        before = [p.detach().clone() for p in encoder]
        optimizer.step()
        moved = max(float((p.detach() - old).abs().max()) for p, old in zip(encoder, before))
        check(moved > 0.0, f"a real AdamW step moved an encoder parameter by {moved:.3e}")
    finally:
        unload(path_a, mods_a, "r6_02_arm")

    print()
    for note in notes:
        print(f"  note  {note}")
    if failures:
        print(f"\nFAILED: {len(failures)} check(s)")
        for item in failures:
            print(f"  - {item}")
        return 1
    print("\nPASSED: R6-02 restores L2/L3 encoder gradients with the declared decay, and leaves "
          "the hard forward untouched")
    return 0


if __name__ == "__main__":
    sys.exit(main())
