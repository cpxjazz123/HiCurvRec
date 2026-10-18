"""CGV -- Computational Graph & Gradient Verification.

Six gates that every Stage2 mechanism must pass BEFORE training is allowed to start.

    G1 forward consistency   the mechanism changes only the computation it declares
    G2 gradient path         every loss term it depends on reaches its target parameters
    G3 gradient direction    the executed gradient matches the mechanism's own formula
    G4 parameter update      the optimizer really performs the predicted update
    G5 DDP consistency       multi-rank gradients match an independent pooled reference
    G6 real training entry   the verified path is the one training actually executes

Design notes
------------
*Output isolation is mandatory.* Some gates execute the trainer's real ``main()``. The
trainer writes its checkpoint, SID export and metrics into its production result directory, so
:func:`isolated_outputs` redirects every one of those paths into a temporary directory FIRST.
Without it, merely verifying a mechanism destroys the artifacts of the run under audit -- a
mistake this module exists partly to make impossible.

*Dependency discovery is functional, not declared.* G2 finds which loss terms an arm actually
depends on by substituting a random probe tensor into each candidate term and observing whether
the executed encoder gradient changes. An arm that reads a term is then required to receive a
non-zero gradient through it. This is what catches a term that participates in the objective but
contributes nothing to the target parameters.

*Unknown mechanisms fail closed.* G3 needs the mechanism's own formula. An arm whose mechanism is
absent from :data:`MECHANISMS` is refused with an actionable message rather than passed by
default, so a new mechanism cannot skip the direction check by being unrecognised.
"""
from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import tempfile
import time
import types
import warnings
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/GeneRec")
CGV_DIR = REPO_ROOT / "results/stage2_RQ-VAE/_cgv"
CGV_PY = "/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9"

N_SRC = 24
IN_DIM = 32
BEHAVIOR_WEIGHT = 0.5
CHILD_TIMEOUT_S = 240
G6_EPOCHS = 2

ATOL, RTOL = 1e-5, 1e-5

GATES = ("G1", "G2", "G3", "G4", "G5", "G6")

GATE_TITLES = {
    "G1": "forward consistency",
    "G2": "gradient path",
    "G3": "gradient direction",
    "G4": "parameter update",
    "G5": "DDP consistency",
    "G6": "real training entry",
}

GATE_PREVENTS = {
    "G1": "unintended changes to the loss, the quantisation result, or the sampling",
    "G2": "a loss term that contributes nothing to the targeted parameters (the R5-10 class)",
    "G3": "wrong projection direction, coefficient, or sign",
    "G4": "a mechanism that is present but has no effect on the parameters",
    "G5": "double all-reduce, or per-rank divergence under DDP",
    "G6": "tests passing against a code path training never executes",
}


# --------------------------------------------------------------------------------------
# mechanism registry: the formula G3 verifies, keyed by the arm's own mechanism flag
# --------------------------------------------------------------------------------------
def _pcgrad(g_l1: torch.Tensor, g_ref: torch.Tensor, *, eps: float = 1e-12) -> torch.Tensor:
    base_sq = float((g_ref * g_ref).sum())
    if base_sq <= 0.0:
        return g_l1
    dot = float((g_l1 * g_ref).sum())
    if dot >= 0.0:
        return g_l1
    return g_l1 - (dot / (base_sq + eps)) * g_ref


MECHANISMS: dict[str, dict[str, Any]] = {
    "r4_08": {
        "detector": ("R4_08_PCGRAD_ENC",),
        "on_flag": "R4_08_PCGRAD_ENC",
        "rule": "project_l1_against_base",
        "reference_term": "base_loss",
        "summary": "project g_L1 against g_base when dot < 0",
    },
    "r5_01": {
        "detector": ("R5_01_REVERSE_PCGRAD",),
        "on_flag": "R5_01_REVERSE_PCGRAD",
        "rule": "project_base_against_l1",
        "reference_term": "l1_loss",
        "summary": "project g_base against g_L1, assemble the unprojected g_L1",
    },
    "r5_02": {
        "detector": ("R5_02_RECON_REF",),
        "on_flag": "R5_02_RECON_REF",
        "rule": "project_l1_against_base",
        "reference_term": "recon_loss",
        "summary": "project g_L1 against the recon-only gradient, assemble full g_base",
    },
    "r5_03": {
        "detector": ("R5_03_QUANT_REF",),
        "on_flag": "R5_03_QUANT_REF",
        "rule": "project_l1_against_base",
        "reference_term": "quant_loss",
        "summary": "project g_L1 against the quant-only gradient",
    },
    "r5_04": {
        "detector": ("R5_04_BLOCKWISE",),
        "on_flag": "R5_04_BLOCKWISE",
        "rule": "blockwise_l1_against_base",
        "reference_term": "base_loss",
        "summary": "per-Linear-block projection instead of one global vector",
    },
    "r5_05": {
        "detector": ("R5_05_ALPHA",),
        "on_flag": "R5_05_ALPHA",
        "rule": "project_l1_against_base",
        "reference_term": "base_loss",
        "alpha": 0.5,
        "summary": "half-strength projection",
    },
    "r5_06": {
        "detector": ("R5_06_NORM_PRESERVE",),
        "on_flag": "R5_06_NORM_PRESERVE",
        "rule": "norm_restoring_projection",
        "reference_term": "base_loss",
        "summary": "restore the original ||g_L1|| after projecting",
    },
    "r5_07": {
        "detector": ("R5_07_COS_THRESHOLD",),
        "on_flag": "R5_07_COS_THRESHOLD",
        "rule": "cosine_gated_projection",
        "reference_term": "base_loss",
        "summary": "project only when cos(g_L1, g_base) < -0.2",
    },
    "r5_08": {
        "detector": ("R5_08_LATE_START_EPOCH",),
        "on_flag": "R5_08_LATE_START_EPOCH",
        "rule": "project_l1_against_base",
        "reference_term": "base_loss",
        "summary": "apply the projection only from epoch 2250",
    },
    "r5_09": {
        "detector": ("R5_09_EMA_DECAY",),
        "on_flag": "R5_09_EMA_DECAY",
        "rule": "project_l1_against_base",
        "reference_term": "ema_of_base",
        "warmup_steps": 2,          # the first call only seeds the buffer with g_base
        "ema_seed_offset": 1,       # ... so the priming step must use a DIFFERENT batch
        "summary": "project g_L1 against the EMA-smoothed base gradient",
    },
    "r5_10": {
        "detector": ("R5_10_INTERLEVEL",),
        "on_flag": "R5_10_INTERLEVEL",
        "rule": "project_l1_against_base",
        "reference_term": "other_loss",
        "summary": "project g_L1 against the L2+L3 behaviour gradient",
    },
}


# --------------------------------------------------------------------------------------
# isolation and loading
# --------------------------------------------------------------------------------------
# The files a trainer WRITES. Inputs (stage1 embeddings, stage0 parquet, dataset frames) share
# the repository root, so "anything under the repo" is the wrong rule: it would move the inputs
# into a temporary directory and the run would fail before it ever reached the mechanism.
TRAINER_WRITES = {
    "training_metrics.jsonl",
    "rqvae_best.pth",
    "sids_for_hgrec.npy",
    "sids_raw.npy",
    "item_sids.json",
    "launcher.log",
}
TRAINER_OUTPUT_DIRS = {"logs", "results"}


@contextlib.contextmanager
def isolated_outputs(T, tmp: Path):
    """Redirect every path the trainer WRITES into ``tmp``.

    MANDATORY before executing anything that can reach the trainer's artifact-writing code.
    ``train_rqvae.main()`` writes its checkpoint, SID export and metrics into its production
    directories, so verifying a mechanism without this redirection overwrites the very artifacts
    being audited.

    The redirection is DISCOVERED, not listed by hand. ``METRICS_PATH`` for instance is computed
    from ``LOG_DIR`` at import time, so reassigning ``LOG_DIR`` afterwards silently leaves the
    metrics stream pointing at the real run -- which is exactly how a two-epoch verification
    truncated a 3003-line training log. Seeds are the module paths named like something a trainer
    writes, and the redirection then expands to every module path underneath them. Read-only
    inputs are left exactly where they are.
    """
    saved: dict[str, Path] = {}
    roots: list[Path] = []

    def is_seed(value: Path) -> bool:
        return value.name in TRAINER_WRITES or value.name in TRAINER_OUTPUT_DIRS

    for name, value in list(vars(T).items()):
        if not isinstance(value, Path) or not is_seed(value):
            continue
        try:
            resolved = value.resolve()
        except OSError:
            continue
        if not str(resolved).startswith(str(REPO_ROOT.resolve()) + os.sep):
            continue
        saved[name] = value
        roots.append(resolved)
        try:
            relative = resolved.relative_to(REPO_ROOT.resolve())
            setattr(T, name, tmp / "redirected" / relative)
        except ValueError:
            setattr(T, name, tmp / "redirected" / value.name)

    if roots:
        for name, value in list(vars(T).items()):
            if name in saved or not isinstance(value, Path):
                continue
            try:
                resolved = value.resolve()
            except OSError:
                continue
            if any(resolved == root or root in resolved.parents for root in roots):
                saved[name] = value
                try:
                    relative = resolved.relative_to(REPO_ROOT.resolve())
                    setattr(T, name, tmp / "redirected" / relative)
                except ValueError:
                    setattr(T, name, tmp / "redirected" / value.name)

    for root in (tmp / "redirected", tmp / "out", tmp / "logs"):
        root.mkdir(parents=True, exist_ok=True)
    try:
        yield sorted(saved)
    finally:
        for name, value in saved.items():
            setattr(T, name, value)


def load_trainer(tree: Path, module_name: str = "cgv_trainer"):
    """Import ``tree/train_rqvae.py`` under a private name so arms cannot collide."""
    tree = Path(tree).resolve()
    entry = tree / "train_rqvae.py"
    if not entry.is_file():
        raise FileNotFoundError(f"no train_rqvae.py in {tree}")
    saved_path = list(sys.path)
    saved_mods = {k: sys.modules.get(k) for k in
                  (module_name, "model", "model.model", "model.layers")}
    for key in saved_mods:
        sys.modules.pop(key, None)
    sys.path.insert(0, str(tree))
    module = types.ModuleType(module_name)
    module.__file__ = str(entry)
    module.__dict__["__name__"] = module_name
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            spec = importlib.util.spec_from_file_location(module_name, entry)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)  # type: ignore[union-attr]
        except RuntimeError:
            # Single-arm guard: read EXPERIMENT_ARM from the source and neutralise only that
            # guard, which exists to stop an accidental rerun, not to change behaviour.
            import ast

            parsed = ast.parse(entry.read_text(encoding="utf-8"), filename=str(entry))
            parsed.body = [
                node for node in parsed.body
                if not (isinstance(node, ast.If)
                        and ast.unparse(node.test).startswith("EXPERIMENT_ARM !="))
            ]
            module = types.ModuleType(module_name)
            module.__file__ = str(entry)
            module.__dict__["__name__"] = module_name
            sys.modules[module_name] = module
            exec(compile(parsed, str(entry), "exec"), module.__dict__)  # noqa: S102
    return module, saved_path, saved_mods


def unload_trainer(saved_path, saved_mods, module_name="cgv_trainer"):
    sys.path[:] = saved_path
    for key, value in saved_mods.items():
        if value is None:
            sys.modules.pop(key, None)
        else:
            sys.modules[key] = value
    sys.modules.pop(module_name, None)


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------------------
# shared mini-case
# --------------------------------------------------------------------------------------
def flat(grads):
    return torch.cat([g.reshape(-1) for g in grads])


def unflatten(flat_vec, template):
    out, offset = [], 0
    for g in template:
        count = g.numel()
        out.append(flat_vec[offset:offset + count].view_as(g))
        offset += count
    return out


def build_case(T, seed: int = 1234, n_src: int = N_SRC):
    """A deterministic mini training case for one trainer module."""
    torch.manual_seed(seed)
    model = T.RQVAE(T._tokenizer_config(), in_dim=IN_DIM)
    emb = torch.randn(2 * n_src, IN_DIM, generator=torch.Generator().manual_seed(seed + 1))
    reconstructed, quant_loss, _, _ = model.forward(emb)
    base_loss, recon_loss = model.compute_loss(emb, reconstructed, quant_loss)
    residuals = T._residuals_with_trace(model, emb)
    l1_loss, other_loss, valid = T.behavior_contrastive_components(
        residuals, torch.arange(n_src), (torch.arange(n_src) + 1) % n_src, None)
    encoder = list(model.encoder.parameters())
    return {
        "T": T, "model": model, "emb": emb,
        "reconstructed": reconstructed.detach(), "quant_loss": quant_loss.detach(),
        "base_loss": base_loss, "recon_loss": recon_loss,
        "l1_loss": l1_loss, "other_loss": other_loss, "valid_levels": valid,
        "encoder": encoder,
        "non_encoder": [p for p in model.parameters() if not any(p is q for q in encoder)],
        "scale": BEHAVIOR_WEIGHT / max(valid, 1),
    }


def arm_kwargs(T, case) -> dict:
    import inspect

    accepted = inspect.signature(T._r4_08_apply_projection).parameters
    kwargs = {}
    if "recon_loss" in accepted:
        kwargs["recon_loss"] = case["recon_loss"]
    if "encoder_module" in accepted:
        kwargs["encoder_module"] = case["model"].encoder
    return kwargs


def term_grads(T, case, key: str):
    """Encoder gradient of one named loss term, before any pooling."""
    source = {
        "base_loss": case["base_loss"],
        "l1_loss": case["l1_loss"],
        "other_loss": case["other_loss"],
        "recon_loss": case["recon_loss"],
        "quant_loss": case["base_loss"] - case["recon_loss"],
    }[key]
    grads = T._r4_08_encoder_gradients(source, case["encoder"], retain=True)
    return [torch.zeros_like(p) if g is None else g
            for p, g in zip(case["encoder"], grads)]


def max_abs_diff(a, b) -> float:
    return max(float((x - y).abs().max()) for x, y in zip(a, b))


def l2(a, b) -> float:
    return float(torch.sqrt(sum(((x - y) ** 2).sum() for x, y in zip(a, b))))


def executed_encoder_grad(T, case_hook=None,
                          epoch: int | None = None,
                          seed: int = 1234) -> tuple[list[torch.Tensor], dict]:
    """Run the production gradient assembly on a FRESH case and return what it wrote.

    A fresh case per call is required, not a convenience: ``_r4_08_apply_projection`` ends with
    an ``autograd.grad(..., retain_graph=False)`` for the L2+L3 term, which frees the graph, so
    the same case cannot be executed twice.

    ``case_hook(case) -> dict`` lets a caller substitute loss terms built ON THIS CASE's graph.
    That anchoring matters: a probe anchored on a different case's parameters would carry a zero
    gradient, and substituting it would look like "this term is unused" for every arm.
    """
    case = build_case(T, seed=seed)
    if case_hook is not None:
        case.update(case_hook(case))
    model = case["model"]
    model.zero_grad(set_to_none=False)
    gated = False
    if epoch is not None and hasattr(T, "R5_08_LATE_START_EPOCH"):
        gated = epoch < T.R5_08_LATE_START_EPOCH
    if gated:
        # Production takes the `else: loss.backward()` branch when the projection is off, and
        # that `loss` is the FULLY ASSEMBLED one. Backwarding base_loss alone here would drop the
        # behaviour term and compare against the wrong expectation.
        assembled = (case["base_loss"]
                     + BEHAVIOR_WEIGHT * (case["l1_loss"] + case["other_loss"])
                     / max(case["valid_levels"], 1))
        assembled.backward()
    else:
        case["base_loss"].backward(retain_graph=True)
        T._r4_08_apply_projection(case["base_loss"], case["l1_loss"], case["other_loss"],
                                  case["encoder"], BEHAVIOR_WEIGHT,
                                  valid_levels=case["valid_levels"], **arm_kwargs(T, case))
    return [p.grad.detach().clone() for p in case["encoder"]], case


def detect_mechanism(T) -> tuple[str | None, dict | None]:
    """Identify the mechanism from the arm's own flags.

    ``R4_08_PCGRAD_ENC`` is present in EVERY arm, so the parent must be tested LAST; otherwise
    every arm would be certified against the parent rule.
    """
    ordered = sorted(MECHANISMS.items(), key=lambda kv: kv[0] == "r4_08")
    for name, spec in ordered:
        if all(hasattr(T, flag) for flag in spec["detector"]):
            return name, spec
    return None, None


def identify_arm(T) -> str | None:
    """The arm the trainer actually declares, not one guessed from the directory name."""
    arm = getattr(T, "EXPERIMENT_ARM", None)
    return str(arm) if arm else None


# --------------------------------------------------------------------------------------
# G1 -- forward consistency
# --------------------------------------------------------------------------------------
def gate_g1(T, spec) -> dict:
    """The mechanism must not change the forward: identical model weights and inputs must give
    identical reconstructed embeddings, quantisation loss, tokens and losses with the mechanism
    on and off."""
    case_on = build_case(T)
    fingerprint_on = {
        "reconstructed": case_on["reconstructed"],
        "quant_loss": case_on["quant_loss"],
        "base_loss": float(case_on["base_loss"].detach()),
        "recon_loss": float(case_on["recon_loss"].detach()),
    }
    tokens_on = T.RQVAE(T._tokenizer_config(), in_dim=IN_DIM)
    tokens_on.load_state_dict(case_on["model"].state_dict())
    tokens_on.eval()
    with torch.no_grad():
        tok_on = tokens_on.get_indices(case_on["emb"], infer_use_sk=False)

    flag = spec.get("on_flag")
    saved = getattr(T, flag) if flag else None
    if saved is not None:
        setattr(T, flag, False if isinstance(saved, bool) else 0)
    try:
        case_off = build_case(T)
        tokens_off_model = T.RQVAE(T._tokenizer_config(), in_dim=IN_DIM)
        tokens_off_model.load_state_dict(case_off["model"].state_dict())
        tokens_off_model.eval()
        with torch.no_grad():
            tok_off = tokens_off_model.get_indices(case_off["emb"], infer_use_sk=False)
    finally:
        if saved is not None:
            setattr(T, flag, saved)

    diffs = {
        "reconstructed": float((fingerprint_on["reconstructed"] - case_off["reconstructed"].detach())
                               .abs().max()),
        "quant_loss": abs(float(fingerprint_on["quant_loss"]) - float(case_off["quant_loss"])),
        "base_loss": abs(fingerprint_on["base_loss"] - float(case_off["base_loss"].detach())),
        "recon_loss": abs(fingerprint_on["recon_loss"] - float(case_off["recon_loss"].detach())),
        "tokens": int((tok_on != tok_off).sum()),
    }
    worst = max(diffs["reconstructed"], diffs["quant_loss"], diffs["base_loss"],
                diffs["recon_loss"])
    ok = worst == 0.0 and diffs["tokens"] == 0
    return {
        "gate": "G1", "title": GATE_TITLES["G1"], "passed": ok,
        "detail": ("forward is bitwise identical with the mechanism on and off"
                   if ok else "the mechanism changed the forward pass"),
        "measurements": diffs,
        "finding": None if ok else (
            "a gradient-only mechanism must leave the forward bitwise identical; it changed "
            f"{diffs}"),
    }


# --------------------------------------------------------------------------------------
# G2 -- gradient path
# --------------------------------------------------------------------------------------
def _probe_substitution(T, key: str, seed: int = 77) -> float:
    """Replace one loss term with a random probe and see whether the encoder gradient moves.

    The probe is built on top of the base loss's graph so it REQUIRES GRAD and carries a
    non-zero encoder gradient of its own. A detached scalar would make the arm's
    ``autograd.grad`` raise, and a zero-gradient probe would be indistinguishable from the real
    term -- neither would answer the question this probe exists to answer.
    """
    reference, _ = executed_encoder_grad(T)

    def hook(case):
        raw = torch.randn((), generator=torch.Generator().manual_seed(seed))
        return {key: case["base_loss"] * 3.7 + raw}

    moved, _ = executed_encoder_grad(T, case_hook=hook)
    return l2(moved, reference)


def gate_g2(T, spec) -> dict:
    """Every loss term the mechanism DEPENDS ON must deliver gradient to its target parameters.

    Dependency discovery is functional: a random probe carrying its own non-zero encoder
    gradient is substituted for each candidate term, and the term counts as "read" when the
    executed encoder gradient moves.

    Only the mechanism's DECLARED reference term is blocking. A term that is merely added by the
    shared assembly is reported as a warning instead: it is a family-wide property, and failing
    every arm on it would make the gate useless for judging a new mechanism. The declared term is
    the mechanism's own claim, so a zero gradient there means the mechanism cannot work, which is
    exactly the failure this gate exists to catch.
    """
    dependencies = {}
    # base_loss is included because a mechanism may reference it directly, through a split of it
    # (r5_03 uses base - recon), or through an EMA of it (r5_09).
    for key in ("base_loss", "l1_loss", "other_loss", "recon_loss"):
        moved = _probe_substitution(T, key)
        if moved > 1e-6:
            dependencies[key] = moved

    declared = spec.get("reference_term")
    case = build_case(T)
    norms = {}
    for key in ("base_loss", "l1_loss", "other_loss", "recon_loss", "quant_loss"):
        norms[key] = float(flat(term_grads(T, case, key)).norm())

    # A derived reference has no substitution of its own; it is assembled from terms the probe
    # can reach. `ema_of_base` seeds with g_base on the first step, so its gradient is g_base's.
    READS = {
        "quant_loss": lambda: "base_loss" in dependencies or "recon_loss" in dependencies,
        "ema_of_base": lambda: "base_loss" in dependencies,
        "base_loss": lambda: True,
    }
    effective = {"quant_loss": "quant_loss", "ema_of_base": "base_loss"}.get(declared, declared)
    declared_norm = norms.get(effective, 0.0)
    declared_read = READS.get(declared, lambda: declared in dependencies)()
    blocking = []
    if declared_norm <= 0.0:
        blocking.append(
            f"the declared reference term {declared!r} has encoder gradient norm "
            f"{declared_norm:.3e}: the mechanism projects against a vector that carries no "
            "gradient, so it can never change the update")
    if not declared_read:
        blocking.append(f"the declared reference term {declared!r} is never read by this arm")

    warnings = [f"{key} is read by the shared assembly but delivers no encoder gradient "
                f"(norm {norms[key]:.3e})" for key in sorted(dependencies)
                if norms.get(key, 0.0) <= 0.0]
    passed = not blocking
    return {
        "gate": "G2", "title": GATE_TITLES["G2"], "passed": passed,
        "detail": (f"the declared reference term {declared!r} delivers gradient to the encoder "
                   f"(norm {declared_norm:.3e}) and is read by this arm")
        if passed else "; ".join(blocking),
        "measurements": {"discovered_dependencies": dependencies,
                         "encoder_gradient_norms": norms,
                         "declared_reference_term": declared,
                         "declared_reference_effective_term": effective,
                         "declared_reference_norm": declared_norm},
        "warnings": warnings,
        "finding": None if passed else "; ".join(blocking),
    }


# --------------------------------------------------------------------------------------
# G3 -- gradient direction
# --------------------------------------------------------------------------------------
def _reference_gradient(T, spec, case, *, epoch: int | None = None,
                        ema_ref: list[torch.Tensor] | None = None) -> list[torch.Tensor]:
    """Rebuild the encoder gradient from the mechanism's declared formula.

    ``epoch`` makes the reference epoch-aware: an arm whose mechanism is gated off below a
    threshold must produce the UNPROJECTED assembly at such an epoch, not the projected one.
    ``ema_ref`` supplies the lagged EMA reference for the mechanisms that use one.
    """
    g_base = term_grads(T, case, "base_loss")
    g_l1 = term_grads(T, case, "l1_loss")
    g_other = term_grads(T, case, "other_loss")

    ref_term = spec.get("reference_term")
    if ref_term == "base_loss":
        g_ref = g_base
    elif ref_term == "l1_loss":
        g_ref = g_l1
    elif ref_term == "recon_loss":
        g_ref = term_grads(T, case, "recon_loss")
    elif ref_term == "quant_loss":
        g_ref = term_grads(T, case, "quant_loss")
    elif ref_term == "other_loss":
        g_ref = g_other
    elif ref_term == "ema_of_base":
        g_ref = ema_ref if ema_ref is not None else g_base
    else:
        raise KeyError(ref_term)

    rule = spec["rule"]
    alpha = float(spec.get("alpha", 1.0))
    eps = float(getattr(T, "R4_08_EPS", 1e-12))

    if epoch is not None and hasattr(T, "R5_08_LATE_START_EPOCH") and \
            epoch < T.R5_08_LATE_START_EPOCH:
        # the gate is closed: the arm must fall back to the plain, unprojected assembly
        return [g_base[i] + case["scale"] * (g_l1[i] + g_other[i])
                for i in range(len(g_base))]

    if rule == "project_base_against_l1":
        base_proj = unflatten(_pcgrad(flat(g_base), flat(g_l1), eps=eps), g_base)
        return [base_proj[i] + case["scale"] * (g_l1[i] + g_other[i])
                for i in range(len(g_base))]
    if rule == "blockwise_l1_against_base":
        blocks = T._r5_04_linear_blocks(case["model"].encoder, case["encoder"])
        projected = [g.clone() for g in g_l1]
        for block in blocks:
            b = torch.cat([g_base[i].reshape(-1) for i in block])
            l = torch.cat([g_l1[i].reshape(-1) for i in block])
            out = _pcgrad(l, b, eps=eps)
            offset = 0
            for i in block:
                count = g_l1[i].numel()
                projected[i] = out[offset:offset + count].view_as(g_l1[i])
                offset += count
        return [g_base[i] + case["scale"] * (projected[i] + g_other[i])
                for i in range(len(g_base))]
    if rule == "norm_restoring_projection":
        out = _pcgrad(flat(g_l1), flat(g_ref), eps=eps)
        norm = float(torch.linalg.vector_norm(flat(g_l1)))
        proj_norm = float(torch.linalg.vector_norm(out))
        if proj_norm <= 1e-8 * max(norm, 1e-12):
            out = torch.zeros_like(out)
        else:
            out = out * (norm / proj_norm)
        projected = unflatten(out, g_l1)
        return [g_base[i] + case["scale"] * (projected[i] + g_other[i])
                for i in range(len(g_base))]
    if rule == "cosine_gated_projection":
        threshold = float(getattr(T, "R5_07_COS_THRESHOLD", -0.2))
        l = flat(g_l1)
        r = flat(g_ref)
        lsq = float((l * l).sum())
        rsq = float((r * r).sum())
        if rsq <= 0.0 or lsq <= 0.0:
            out = l
        else:
            cos = float((l * r).sum()) / (math.sqrt(lsq) * math.sqrt(rsq) + eps)
            out = _pcgrad(l, r, eps=eps) if cos < threshold else l
        projected = unflatten(out, g_l1)
        return [g_base[i] + case["scale"] * (projected[i] + g_other[i])
                for i in range(len(g_base))]

    # default family rule
    if alpha != 1.0:
        base_sq = float((flat(g_ref) * flat(g_ref)).sum())
        dot = float(flat(g_l1) @ flat(g_ref))
        out = (flat(g_l1) if base_sq <= 0.0 or dot >= 0.0
               else flat(g_l1) - alpha * (dot / (base_sq + eps)) * flat(g_ref))
    else:
        out = _pcgrad(flat(g_l1), flat(g_ref), eps=eps)
    projected = unflatten(out, g_l1)
    return [g_base[i] + case["scale"] * (projected[i] + g_other[i])
            for i in range(len(g_base))]


def _reset_ema(T) -> None:
    """R5-09 keeps its EMA in module-global state; a leftover buffer would make the executed
    reference lag and stop matching the first-step formula."""
    state = getattr(T, "R5_09_EMA_STATE", None)
    if state is not None:
        state["buffers"] = None


def gate_g3(T, spec) -> dict:
    """The executed gradient must match the mechanism's declared formula on every probe seed,
    and the formula must be non-vacuous: on at least one seed the mechanism must actually fire
    and differ from the parent rule.

    Scanning seeds is not optional. On a seed with no gradient conflict the projection is a
    passthrough, so EVERY PCGrad variant collapses onto the parent rule and a single-seed check
    would either pass a broken mechanism or refuse a working one, depending on which seed it
    happened to pick.
    """
    seeds = (1, 4, 5, 12, 16, 30, 32, 36)
    distinguishes = spec is not MECHANISMS["r4_08"]
    warmup = int(spec.get("warmup_steps", 1))
    # an epoch-gated mechanism (R5-08) IS the parent rule once its gate opens; its difference is
    # visible while the gate is CLOSED, so that epoch must be probed as well
    epochs = [T.EPOCHS]
    if hasattr(T, "R5_08_LATE_START_EPOCH"):
        epochs.append(1)
    per_seed = []
    worst_formula = 0.0
    fired_seeds = []
    for index, seed in enumerate(seeds):
      for epoch in epochs:
        _reset_ema(T)
        ema_ref = None
        g_base = term_grads(T, build_case(T, seed=seed), "base_loss")
        for step_index in range(warmup):
            if step_index == warmup - 1:
                # prime on a DIFFERENT batch: with the same batch the EMA converges to g_base
                # immediately and can never lag, so the mechanism would look like the parent
                step_seed = seed
            else:
                offset = int(spec.get("ema_seed_offset", 1))
                step_seed = seeds[(index + offset * (step_index + 1)) % len(seeds)]
            executed, _ = executed_encoder_grad(T, epoch=epoch, seed=step_seed)
            if step_index == warmup - 2:
                prime_grads = term_grads(T, build_case(T, seed=step_seed), "base_loss")
                decay = float(getattr(T, "R5_09_EMA_DECAY", 0.9))
                ema_ref = [decay * a + (1.0 - decay) * b for a, b in zip(prime_grads, g_base)]
        case = build_case(T, seed=seed)
        reference = _reference_gradient(T, spec, case, epoch=epoch, ema_ref=ema_ref)
        diff = max_abs_diff(executed, reference)
        worst_formula = max(worst_formula, diff)
        entry = {"seed": seed, "epoch": epoch, "deviation_from_formula": diff,
                 "dot_l1_base":
                 float(torch.cat([g.reshape(-1) for g in term_grads(T, case, "l1_loss")])
                       @ torch.cat([g.reshape(-1) for g in term_grads(T, case, "base_loss")]))}
        if distinguishes:
            parent_case = build_case(T, seed=seed)
            # the parent arm has no epoch gate, so its reference is always its projected rule;
            # passing the child's epoch here would make parent and child agree while the gate
            # is closed and hide exactly the difference this gate is meant to measure
            parent_reference = _reference_gradient(T, MECHANISMS["r4_08"], parent_case)
            entry["distance_from_parent_rule"] = l2(executed, parent_reference)
            if entry["distance_from_parent_rule"] > 1e-6:
                fired_seeds.append(f"seed{seed}@epoch{epoch}")
        per_seed.append(entry)

    formula_ok = worst_formula < ATOL
    non_vacuous = (bool(fired_seeds) if distinguishes else True)
    passed = formula_ok and non_vacuous
    total = len(seeds) * len(epochs)
    detail_bits = [f"max deviation from {spec['rule']} over {total} (seed, epoch) probes = "
                   f"{worst_formula:.3e}"]
    if distinguishes:
        detail_bits.append(
            f"the mechanism changes the encoder gradient on {len(fired_seeds)}/{total} probes"
            + (f" (e.g. {fired_seeds})" if fired_seeds else ""))
    else:
        detail_bits.append("this arm is the parent rule itself")
    return {
        "gate": "G3", "title": GATE_TITLES["G3"], "passed": passed,
        "detail": "; ".join(detail_bits) if passed else
        ("; ".join(detail_bits) + " -- " + (
            "the executed gradient does not match the declared formula"
            if not formula_ok else
            "the projection never fires on any probe seed, so the mechanism is indistinguishable "
            "from the parent rule and cannot be shown to do anything")),
        "measurements": {"per_seed": per_seed,
                         "max_deviation_from_formula": worst_formula,
                         "seeds_where_mechanism_differs": fired_seeds,
                         "rule": spec["rule"],
                         "reference_term": spec.get("reference_term"),
                         "distinguishes_from_parent": distinguishes},
        "finding": None if passed else (
            f"the mechanism's declared formula {spec['rule']!r} with reference "
            f"{spec.get('reference_term')!r} does not reproduce the executed gradient"
            if not formula_ok else
            "the projection never changes the encoder gradient on any probe seed: direction, "
            "trigger condition or coefficient leave the mechanism inert"),
    }


# --------------------------------------------------------------------------------------
# G4 -- parameter update
# --------------------------------------------------------------------------------------
def _adamw_first_step(param: torch.Tensor, clipped_grad: torch.Tensor,
                      lr: float, weight_decay: float, eps: float = 1e-8) -> torch.Tensor:
    """Closed-form AdamW first step, derived independently of torch.optim internals.

    With bias correction, ``v_hat == g**2`` on the first step, so ``g_hat == g/(|g|+eps)``.
    """
    direction = clipped_grad / (clipped_grad.abs() + eps)
    return param * (1.0 - lr * weight_decay) - lr * direction


def gate_g4(T, spec) -> dict:
    """One real optimizer step: every parameter must move by exactly what AdamW predicts from the
    gradient that was verified, and the targeted parameters must actually move."""
    lr, weight_decay, eps = 1e-3, 0.01, 1e-8
    case = build_case(T)
    model = case["model"]
    before = [p.detach().clone() for p in model.parameters()]
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.zero_grad(set_to_none=False)
    case["base_loss"].backward(retain_graph=True)
    T._r4_08_apply_projection(case["base_loss"], case["l1_loss"], case["other_loss"],
                              case["encoder"], BEHAVIOR_WEIGHT,
                              valid_levels=case["valid_levels"], **arm_kwargs(T, case))
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    clipped = [None if p.grad is None else p.grad.detach().clone() for p in model.parameters()]
    optimizer.step()
    after = [p.detach().clone() for p in model.parameters()]

    enc_index = {id(p) for p in case["encoder"]}
    nonenc_index = {id(p) for p in case["non_encoder"]}
    enc_moved = 0.0
    nonenc_moved = 0.0
    worst_enc = 0.0
    worst_nonenc = 0.0
    for param, old, grad, new in zip(model.parameters(), before, clipped, after):
        if grad is None:
            continue
        expected = _adamw_first_step(old, grad, lr, weight_decay, eps)
        delta = float((new - expected).abs().max())
        moved = float((new - old).abs().max())
        if id(param) in enc_index:
            enc_moved = max(enc_moved, moved)
            worst_enc = max(worst_enc, delta)
        elif id(param) in nonenc_index:
            nonenc_moved = max(nonenc_moved, moved)
            worst_nonenc = max(worst_nonenc, delta)
    steps = {int(state["step"].item()) for state in optimizer.state.values() if "step" in state}

    prediction_ok = worst_enc < 1e-6 and worst_nonenc < 1e-6
    passed = prediction_ok and enc_moved > 0.0 and steps == {1}
    return {
        "gate": "G4", "title": GATE_TITLES["G4"], "passed": passed,
        "detail": (f"encoder moved {enc_moved:.3e}, non-encoder {nonenc_moved:.3e}, both within "
                   f"{max(worst_enc, worst_nonenc):.3e} of the closed-form AdamW prediction, "
                   f"optimizer recorded exactly {sorted(steps)} step(s)")
        if passed else
        (f"encoder moved {enc_moved:.3e} (prediction off by {worst_enc:.3e}), non-encoder "
         f"{nonenc_moved:.3e} (off by {worst_nonenc:.3e}), optimizer steps {sorted(steps)}"),
        "measurements": {"encoder_max_abs_delta": enc_moved,
                         "non_encoder_max_abs_delta": nonenc_moved,
                         "encoder_update_prediction_error": worst_enc,
                         "non_encoder_update_prediction_error": worst_nonenc,
                         "optimizer_steps": sorted(steps)},
        "finding": None if passed else (
            "the optimizer did not perform the predicted update, or the targeted parameters did "
            "not move at all: the mechanism is present in the graph but has no effect"),
    }


# --------------------------------------------------------------------------------------
# G5 -- DDP consistency (real two-rank gloo, run in a fresh subprocess)
# --------------------------------------------------------------------------------------
DDP_DRIVER = '''
import sys, json
sys.path.insert(0, {engine!r})
from pathlib import Path
import cgv_ddp_child as C
sys.exit(C.main({tree!r}, {tag!r}, {timeout!r}))
'''


def gate_g5(tree: Path, spec) -> dict:
    payload = CGV_DIR / f"g5_{tree.name}"
    payload.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [CGV_PY, str(Path(__file__).with_name("cgv_ddp_child.py")),
         str(tree), str(payload), str(CHILD_TIMEOUT_S)],
        capture_output=True, text=True, timeout=CHILD_TIMEOUT_S + 120)
    summary_path = payload / "summary.json"
    if not summary_path.is_file():
        return {
            "gate": "G5", "title": GATE_TITLES["G5"], "passed": False,
            "detail": "the two-rank DDP run produced no summary",
            "measurements": {"driver_rc": proc.returncode,
                             "stderr_tail": proc.stderr.strip()[-600:]},
            "finding": "G5 could not execute the two-rank path; treat as a failure, not a skip",
        }
    summary = json.loads(summary_path.read_text())
    return {
        "gate": "G5", "title": GATE_TITLES["G5"],
        "passed": bool(summary.get("passed")),
        "detail": summary.get("detail", ""),
        "measurements": summary.get("measurements", {}),
        "finding": summary.get("finding"),
    }


# --------------------------------------------------------------------------------------
# G6 -- real training entry
# --------------------------------------------------------------------------------------
def gate_g6(T, spec, tmp: Path) -> dict:
    """Execute the trainer's real ``main()`` -- with outputs isolated -- and prove the verified
    gradient surgery is actually reached by the training loop.

    A plain two-epoch run is not enough: the behaviour weight ramps from 0, so an early run never
    reaches the mechanism. The ramp is forced open and the mechanism's functions are counted.
    """
    counter = {"projection": 0, "conflicting": 0, "steps": 0}

    def instrument(module, fn_name):
        original = getattr(module, fn_name)

        def wrapper(*args, **kwargs):
            counter["projection"] += 1
            try:
                first, second = args[0], args[1]
                if float(first.reshape(-1) @ second.reshape(-1)) < 0.0:
                    counter["conflicting"] += 1
            except Exception:  # noqa: BLE001
                pass
            return original(*args, **kwargs)

        setattr(module, fn_name, wrapper)

    instrument(T, "r4_project_conflict")

    saved_epochs = T.EPOCHS
    saved_weight_max = getattr(T, "BEHAVIOR_WEIGHT_MAX", None)
    saved_ramp = getattr(T, "RAMP_START_EPOCH", None)
    saved_gate = getattr(T, "R4_08_PCGRAD_ENC", None)
    T.EPOCHS = G6_EPOCHS
    if saved_weight_max is not None:
        T.BEHAVIOR_WEIGHT_MAX = 1.0
    if saved_ramp is not None:
        T.RAMP_START_EPOCH = 0
    if hasattr(T, "R5_08_LATE_START_EPOCH"):
        T.R5_08_LATE_START_EPOCH = 0
    if hasattr(T, "R5_09_EMA_STATE"):
        T.R5_09_EMA_STATE["buffers"] = None

    error = None
    started = time.time()
    try:
        with isolated_outputs(T, tmp):
            try:
                T.main()
            except Exception as exc:  # noqa: BLE001
                error = f"{type(exc).__name__}: {exc}"
    finally:
        T.EPOCHS = saved_epochs
        if saved_weight_max is not None:
            T.BEHAVIOR_WEIGHT_MAX = saved_weight_max
        if saved_ramp is not None:
            T.RAMP_START_EPOCH = saved_ramp
        if saved_gate is not None:
            T.R4_08_PCGRAD_ENC = saved_gate

    produced = sorted(p.name for p in (tmp / "out").glob("*"))
    passed = error is None and counter["projection"] > 0
    return {
        "gate": "G6", "title": GATE_TITLES["G6"], "passed": passed,
        "detail": (f"the real training entry executed the gradient surgery "
                   f"{counter['projection']} time(s) ({counter['conflicting']} on a genuine "
                   f"conflict) in {time.time() - started:.1f}s")
        if passed else
        (f"the real training entry never reached the verified gradient surgery "
         f"({counter['projection']} calls)"
         + (f"; main() raised {error}" if error else "")),
        "measurements": {"projection_calls": counter["projection"],
                         "conflicting_calls": counter["conflicting"],
                         "epochs_run": G6_EPOCHS,
                         "isolated_outputs": produced,
                         "elapsed_s": round(time.time() - started, 1),
                         "error": error},
        "finding": None if passed else (
            "training does not execute the code path the other gates verified -- either the "
            "ramp keeps the behaviour weight at 0, or the mechanism is gated off"),
    }


# --------------------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------------------
def run_cgv(tree: Path, *, gates: tuple[str, ...] = GATES) -> dict:
    tree = Path(tree).resolve()
    trainer_path = tree / "train_rqvae.py"
    trainer_sha = sha256_file(trainer_path)
    result: dict[str, Any] = {
        "arm": None,
        "tree": str(tree),
        "trainer": str(trainer_path),
        "trainer_sha256": trainer_sha,
        "checked_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gates": {},
        "passed": False,
    }
    if trainer_sha is None:
        result["refused"] = f"no train_rqvae.py in {tree}"
        return result

    T, saved_path, saved_mods = load_trainer(tree)
    detected, spec = detect_mechanism(T)
    if spec is None:
        result["refused"] = (
            f"{tree.name}: no registered mechanism. CGV verifies the gradient change against the "
            f"mechanism's own formula, so the formula must be added to tools/cgv/engine.py "
            f"MECHANISMS before this arm can be certified.")
        result["gates"] = {"G3": {"gate": "G3", "title": GATE_TITLES["G3"], "passed": False,
                                  "detail": "unregistered mechanism",
                                  "measurements": {}, "finding": result["refused"]}}
        unload_trainer(saved_path, saved_mods)
        return result
    arm = identify_arm(T)
    result["arm"] = arm
    result["mechanism"] = {"name": detected, **spec}
    if detected != arm:
        result["mechanism_name_mismatch"] = f"tree looks like {arm} but implements {detected}"

    try:
        for gate in GATES:
            if gate not in gates:
                result["gates"][gate] = {"gate": gate, "title": GATE_TITLES[gate],
                                        "passed": None, "skipped": True,
                                        "detail": "not selected", "measurements": {},
                                        "finding": None}
                continue
            if gate == "G1":
                result["gates"][gate] = gate_g1(T, spec)
            elif gate == "G2":
                result["gates"][gate] = gate_g2(T, spec)
            elif gate == "G3":
                result["gates"][gate] = gate_g3(T, spec)
            elif gate == "G4":
                result["gates"][gate] = gate_g4(T, spec)
            elif gate == "G5":
                result["gates"][gate] = gate_g5(tree, spec)
            elif gate == "G6":
                with tempfile.TemporaryDirectory() as raw:
                    result["gates"][gate] = gate_g6(T, spec, Path(raw))
    finally:
        unload_trainer(saved_path, saved_mods)

    result["passed"] = all(result["gates"][g].get("passed") is True for g in gates)
    result["summary"] = {
        g: {"passed": result["gates"][g].get("passed"),
            "title": GATE_TITLES[g],
            "prevents": GATE_PREVENTS[g]}
        for g in gates
    }
    return result


def write_certificate(result: dict) -> Path:
    CGV_DIR.mkdir(parents=True, exist_ok=True)
    arm = result.get("arm") or (result.get("mechanism") or {}).get("name") or "unknown"
    path = CGV_DIR / f"certificate_{arm}.json"
    path.write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
    return path


def load_certificate(tree: Path) -> dict | None:
    """Find a certificate for this tree by mechanism name, then arm name."""
    for name in MECHANISMS:
        path = CGV_DIR / f"certificate_{name}.json"
        if path.is_file():
            try:
                cert = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if str(Path(cert.get("tree", "")).resolve()) == str(Path(tree).resolve()):
                return cert
    return None


def certificate_is_valid(tree: Path, gates: tuple[str, ...] = GATES) -> tuple[bool, str]:
    """A certificate is valid only if it passed every gate AND still describes this trainer."""
    tree = Path(tree)
    cert = load_certificate(tree)
    if cert is None:
        return False, ("no CGV certificate. Run cgv_check(source_dir=...) and all six gates must "
                       "pass before this tree may train")
    if not cert.get("passed"):
        failed = [g for g in gates if (cert.get("gates", {}).get(g) or {}).get("passed") is not True]
        return False, f"CGV gates not passing: {failed}. Re-run cgv_check before training"
    current = sha256_file(tree / "train_rqvae.py")
    if cert.get("trainer_sha256") != current:
        return False, ("the trainer changed after the certificate was issued; the mechanism must "
                       "be re-verified with cgv_check before it may train")
    return True, f"CGV certificate {cert.get('checked_utc')} covers trainer {current[:16]}"
