"""Track A (issue #80): non-perturbing frozen-state replay.

Issue #79's causal contrasts were inconclusive because the audit watched the training loop from
INSIDE it: it ran measurement forwards between ``backward()`` and ``optimizer.step()``, captured and
restored state around it, and advanced the loop to take counterfactual steps. Every one of those
touches a run that is supposed to be the unmodified one, and the trajectory drifted away from an
independent probe.

This module removes the cause rather than tuning it. The training loop is instrumented only to
COPY things out of it -- a capture that reads state and stores tensors advances no RNG, runs no
forward, and restores train/eval mode before returning. The loop then continues exactly as it would
have, untouched. Counterfactual steps are taken in a FRESH model, optimizer and scheduler built
from the frozen state, invoking the ARM's own trace, contrastive, projection and update code on the
captured minibatch. Nothing is ever un-done in the original run.

The rule that makes this honest: a branch is admissible only after the FULL control reproduces an
uninstrumented golden replay bit for bit. Until then the module reports the control failure and
measures nothing.
"""
from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import os
import random
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/GeneRec")
for _entry in (str(REPO_ROOT / "tools/cgv"), str(REPO_ROOT / "tools")):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

import contract as C  # noqa: E402
import engine as E  # noqa: E402

FORK_STEPS = (32, 96, 160)
BASELINE_STEPS = 32
ACTIVE_STEPS = 128
ACTIVE_WEIGHT = 0.20
SEED = 42


# --------------------------------------------------------------------------------------
# hashing: identity is content, never a count
# --------------------------------------------------------------------------------------
def tensor_digest(tensor: torch.Tensor) -> str:
    with torch.no_grad():
        value = tensor.detach().to(torch.float64).cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def state_digest(state_dict: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        value = state_dict[name]
        digest.update(name.encode("utf-8"))
        digest.update(tensor_digest(value).encode("ascii") if isinstance(value, torch.Tensor)
                      else repr(value).encode("utf-8"))
    return digest.hexdigest()


def file_sha256(path: Path) -> str | None:
    """Hash a file, or report its absence rather than assuming it is unchanged."""
    path = Path(path)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def optimizer_digest_of(optimizer_state: dict[str, Any]) -> str:
    """Hash an AdamW state, including every moment, so a missing moment cannot hide."""
    digest = hashlib.sha256()
    state = optimizer_state.get("state") or {}
    for index in sorted(state, key=lambda key: int(key)):
        entry = state[index]
        digest.update(str(index).encode("ascii"))
        for field in sorted(entry):
            value = entry[field]
            digest.update(field.encode("utf-8"))
            digest.update((tensor_digest(value) if isinstance(value, torch.Tensor)
                           else repr(value)).encode("utf-8"))
    return digest.hexdigest()


def source_hashes(tree: Path) -> dict[str, str | None]:
    """Hash the trainer and every module the mechanism can live in."""
    result: dict[str, str | None] = {}
    for relative in ("train_rqvae.py", "model/model.py", "model/layers.py"):
        path = tree / relative
        result[relative] = (hashlib.sha256(path.read_bytes()).hexdigest()
                            if path.is_file() else None)
    return result


# --------------------------------------------------------------------------------------
# the frozen bundle
# --------------------------------------------------------------------------------------
class FrozenBundle:
    """Everything one counterfactual step needs, copied OUT of a live run without disturbing it."""

    def __init__(self, step: int, tree: Path, spec: dict[str, Any]):
        self.step = step
        self.tree = Path(tree)
        self.spec = spec
        self.model_state: dict[str, Any] = {}
        self.optimizer_state: dict[str, Any] = {}
        self.scheduler_state: dict[str, Any] | None = None
        self.item_batch: torch.Tensor | None = None
        self.pair_embeddings: torch.Tensor | None = None
        self.source_ids: torch.Tensor | None = None
        self.target_ids: torch.Tensor | None = None
        self.successor: Any = None
        self.applied_weight: float = 0.0
        self.valid_levels: int = 1
        self.training_mode: bool = True
        self.rng: dict[str, Any] = {}
        self.parameter_digest: str | None = None
        self.optimizer_digest: str | None = None
        self.parameter_sha256_after: str | None = None
        self.lr_at_step: float | None = None
        self.reference_after_state: dict[str, Any] = {}
        self.problems: list[str] = []

    def summary(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "parameter_sha256": self.parameter_digest,
            "optimizer_sha256": self.optimizer_digest,
            "applied_behavior_weight": self.applied_weight,
            "lr_at_step": self.lr_at_step,
            "item_batch_sha256": tensor_digest(self.item_batch) if self.item_batch is not None
            else None,
            "pair_embeddings_sha256": (tensor_digest(self.pair_embeddings)
                                       if self.pair_embeddings is not None else None),
            "source_ids_sha256": (tensor_digest(self.source_ids)
                                  if self.source_ids is not None else None),
            "target_ids_sha256": (tensor_digest(self.target_ids)
                                  if self.target_ids is not None else None),
            "source_hashes": source_hashes(self.tree),
            "training_mode": self.training_mode,
            "parameter_sha256_after_reference": self.parameter_sha256_after,
            "has_optimizer_moments": bool(self.optimizer_state.get("state")),
        }

    def verify(self, expect_deep: bool = True) -> list[str]:
        """Structural checks on the bundle itself, before anything is replayed from it.

        ``expect_deep`` is False for steps the frozen #78 schedule leaves at zero deep weight: the
        behaviour path is genuinely not taken there, so the absence of a pair minibatch is a fact
        about the schedule, not a capture failure.
        """
        problems: list[str] = []
        if not self.optimizer_state.get("state"):
            problems.append("missing AdamW moments: the optimizer state carries no per-parameter "
                            "entry, so a replay from this bundle would not be the same step")
        if self.item_batch is None:
            problems.append("no item minibatch was captured")
        if self.training_mode is not True:
            problems.append("the model was not in training mode at the captured step")
        if expect_deep:
            if self.source_ids is None or self.target_ids is None:
                problems.append("no source/target labels were captured although the deep path ran")
            if self.pair_embeddings is None:
                problems.append("no pair minibatch was captured although the deep weight is live")
            if self.applied_weight <= 0.0:
                problems.append(f"the applied deep weight is {self.applied_weight} although the "
                                f"capture step is inside the active window")
        elif self.pair_embeddings is not None:
            problems.append("a pair minibatch was captured at a step whose deep weight is zero, so "
                            "the capture is not reading the step the loop took")
        return problems

    def save(self, path: Path) -> Path:
        path.mkdir(parents=True, exist_ok=True)
        payload = {
            "step": self.step,
            "tree": str(self.tree),
            "model_state": self.model_state,
            "optimizer_state": self.optimizer_state,
            "scheduler_state": self.scheduler_state,
            "item_batch": self.item_batch,
            "pair_embeddings": self.pair_embeddings,
            "source_ids": self.source_ids,
            "target_ids": self.target_ids,
            "successor": self.successor,
            "applied_weight": self.applied_weight,
            "valid_levels": self.valid_levels,
            "training_mode": self.training_mode,
            "parameter_sha256_after_reference": self.parameter_sha256_after,
            "rng": self.rng,
            "summary": self.summary(),
            "reference_after_state": self.reference_after_state,
            "lr_at_step": self.lr_at_step,
            "problems": self.problems,
        }
        torch.save(payload, path / "bundle.pt")
        (path / "summary.json").write_text(json.dumps(self.summary(), indent=2, default=str))
        return path


# --------------------------------------------------------------------------------------
# capture: read-only, from inside a real run
# --------------------------------------------------------------------------------------
def capture_from_run(tree: Path, spec: dict[str, Any], *,
                     fork_steps: tuple[int, ...] = FORK_STEPS,
                     save_root: Path | None = None,
                     baseline_steps: int = BASELINE_STEPS,
                     active_steps: int = ACTIVE_STEPS,
                     active_weight: float = ACTIVE_WEIGHT,
                     seed: int = SEED,
                     instrument: bool = True) -> dict[str, Any]:
    """Run the arm's real path to each fork step and copy the state out.

    The instrumented callbacks only COPY. None of them runs a forward, touches `.grad`, steps the
    optimizer, or draws from an RNG: the loop continues as if the capture were not there, which is
    the property issue #79 lacked.
    """
    tree = Path(tree).resolve()
    started = time.time()
    trainer, saved_path, saved_mods = E.load_trainer(tree, module_name="r7a_capture_trainer")
    optimizer_cls = getattr(torch.optim, trainer.OPTIMIZER)
    originals = {
        "init": trainer.RQVAE.__init__,
        "step": optimizer_cls.step,
        "zero_grad": optimizer_cls.zero_grad,
        "weight": trainer.behavior_weight,
        "hook": getattr(trainer, "_PROBE_STEP_WEIGHT_HOOK", None),
        "epochs": trainer.EPOCHS,
        "seed": trainer.SEED,
        "trace": trainer._residuals_with_trace,
        "ce": trainer._contrastive_ce,
        "components": getattr(trainer, "behavior_contrastive_components", None),
    }
    state: dict[str, Any] = {"step": 0, "slot": None, "pending": None, "bundles": {},
                             "last_weight": 0.0, "initial_saved": False}
    models: list[Any] = []
    scheduler_holder: dict[str, Any] = {}
    fork_set = set(int(value) for value in fork_steps)
    budget = baseline_steps + active_steps

    def capturing_init(inner_self, *args, **kwargs):
        originals["init"](inner_self, *args, **kwargs)
        models.append(inner_self)
        # Step 0 is the post-KMeans initialisation: there is no optimizer and no minibatch yet, so
        # what is captured is the model state alone. A geometry observer can measure it; a replay
        # cannot, and says so rather than pretending otherwise.
        if 0 in fork_set and not state["initial_saved"] and save_root is not None:
            (Path(save_root) / "step_0000").mkdir(parents=True, exist_ok=True)
            torch.save({"model_state": {name: value.detach().clone().cpu()
                                        for name, value in inner_self.state_dict().items()},
                        "training_mode": inner_self.training,
                        "applied_weight": 0.0, "item_batch": None, "pair_embeddings": None,
                        "source_ids": None, "target_ids": None, "successor": None,
                        "optimizer_state": {}, "step": 0,
                        "summary": {"note": "initialisation only; no optimizer step has run"}},
                       Path(save_root) / "step_0000" / "bundle.pt")
            state["initial_saved"] = True

    has_behavior_weight = hasattr(trainer, "behavior_weight")

    def weight_schedule(epoch, current=None):
        if current is None:
            return originals["weight"](epoch) if has_behavior_weight else 0.0
        step_about_to_run = state["step"] + 1
        if step_about_to_run <= baseline_steps or step_about_to_run > budget:
            state["last_weight"] = 0.0
            return 0.0
        state["last_weight"] = active_weight
        return active_weight

    def tracing(raw_module, embeddings):
        out = originals["trace"](raw_module, embeddings)
        slot = state["slot"]
        if slot is not None:
            # COPY ONLY: no forward, no grad, no RNG draw, no mode change
            slot["pair_embeddings"] = embeddings.detach().clone()
        return out

    def observing_ce(logits, source_ids, target_ids, successor_indicator=None):
        slot = state["slot"]
        if slot is not None:
            slot["source_ids"] = source_ids.detach().clone()
            slot["target_ids"] = target_ids.detach().clone()
            slot["successor"] = (None if successor_indicator is None else successor_indicator)
        return originals["ce"](logits, source_ids, target_ids, successor_indicator)

    def arming_zero_grad(self, *args, **kwargs):
        # ``zero_grad`` is the first call of every loop iteration, before the forward. Arming here
        # is what makes the capture see the step that is about to run rather than the previous one.
        step_about_to_run = state["step"] + 1
        if (step_about_to_run in fork_set
                and step_about_to_run not in state["bundles"]):
            state["slot"] = {"item_batch": None, "pair_embeddings": None, "source_ids": None,
                             "target_ids": None, "successor": None}
        return originals["zero_grad"](self, *args, **kwargs)

    def capturing_forward(raw_module, *args, **kwargs):
        # read-only: record the item minibatch the loop is using on this step
        slot = state["slot"]
        if slot is not None and args:
            with torch.no_grad():
                slot["item_batch"] = args[0].detach().clone()
        return originals["forward"](raw_module, *args, **kwargs)

    def measured_step(self, *args, **kwargs):
        step_about_to_run = state["step"] + 1
        slot = state["slot"]
        if slot is not None and models and step_about_to_run in fork_set:
            model = models[-1]
            bundle = FrozenBundle(step_about_to_run, tree, spec)
            bundle.model_state = {name: value.detach().clone().cpu()
                                  for name, value in model.state_dict().items()}
            bundle.optimizer_state = copy.deepcopy(self.state_dict())
            bundle.scheduler_state = (copy.deepcopy(scheduler_holder["scheduler"].state_dict())
                                      if scheduler_holder.get("scheduler") is not None else None)
            bundle.training_mode = model.training
            bundle.rng = {
                "torch_cpu": torch.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "numpy": np.random.get_state(),
                "python": random.getstate(),
            }
            bundle.applied_weight = state["last_weight"]
            bundle.lr_at_step = float(self.param_groups[0]["lr"])
            bundle.item_batch = slot["item_batch"]
            bundle.pair_embeddings = slot["pair_embeddings"]
            bundle.source_ids = slot["source_ids"]
            bundle.target_ids = slot["target_ids"]
            bundle.successor = slot["successor"]
            bundle.parameter_digest = state_digest(bundle.model_state)
            bundle.optimizer_digest = optimizer_digest_of(bundle.optimizer_state)
            expect_deep = baseline_steps < step_about_to_run <= budget
            bundle.problems = bundle.verify(expect_deep=expect_deep)
            state["pending"] = bundle
        result = originals["step"](self, *args, **kwargs)
        state["step"] = step_about_to_run
        if state["pending"] is not None:
            # The golden reference: the state the REAL loop is in after this step. A FULL branch must
            # reproduce this exactly. It is recorded here and never used to drive anything.
            bundle = state["pending"]
            reference_after = {name: value.detach().clone().cpu()
                               for name, value in models[-1].state_dict().items()}
            bundle.reference_after_state = reference_after
            bundle.parameter_sha256_after = state_digest(reference_after)
            state["bundles"][step_about_to_run] = {
                "summary": bundle.summary(),
                "problems": bundle.problems,
                "deep_active": expect_deep,
                "parameter_sha256_after": bundle.parameter_sha256_after,
            }
            if save_root is not None:
                bundle.save(Path(save_root) / f"step_{step_about_to_run:04d}")
            state["pending"] = None
            state["slot"] = None
        return result

    def digest_only_step(self, *args, **kwargs):
        """The hook-free arm of the non-perturbation control.

        It touches nothing: it counts steps so the weight ramp behaves identically, and once at the
        end it READS the parameters to hash them. No forward, no loss hook, no state copy, no RNG
        draw, no mode change. If the instrumented run disagrees with this one, the capture is
        perturbing and every number downstream is suspect.
        """
        result = originals["step"](self, *args, **kwargs)
        state["step"] += 1
        if models and state["step"] == budget:
            state["final_parameter_sha256"] = state_digest(
                {name: value.detach().clone().cpu()
                 for name, value in models[-1].state_dict().items()})
        return result

    # The weight ramp IS the frozen protocol, so it belongs to both arms of the non-perturbation
    # control. Only the read-only capture hooks are conditional.
    if has_behavior_weight:
        trainer.behavior_weight = weight_schedule
    # the loop reads the module global at run time, so installing it works whether or not the tree
    # ever defined one
    trainer._PROBE_STEP_WEIGHT_HOOK = weight_schedule
    if instrument:
        trainer.RQVAE.__init__ = capturing_init
        trainer._residuals_with_trace = tracing
        trainer._contrastive_ce = observing_ce
        originals["forward"] = trainer.RQVAE.forward
        trainer.RQVAE.forward = capturing_forward
        optimizer_cls.step = measured_step
        optimizer_cls.zero_grad = arming_zero_grad
    else:
        # recording the module reference is not instrumentation: it neither calls the model nor
        # reads a tensor, and without it there is nothing to hash at the end of the run.
        trainer.RQVAE.__init__ = capturing_init
        optimizer_cls.step = digest_only_step
    trainer.EPOCHS = budget + 2
    trainer.SEED = seed

    scratch = Path(tempfile.mkdtemp(prefix="r7a_capture_"))
    failures: list[str] = []
    try:
        with E.isolated_outputs(trainer, scratch):
            try:
                trainer.main()
            except (SystemExit, Exception) as exc:  # noqa: BLE001
                if type(exc).__name__ not in ("SystemExit", "ProbeStop"):
                    failures.append(f"the training loop raised {type(exc).__name__}: {exc}\n"
                                    + "".join(traceback.format_exc()))
    finally:
        trainer.RQVAE.__init__ = originals["init"]
        if instrument:
            trainer.RQVAE.forward = originals["forward"]
        if originals.get("weight") is not None:
            trainer.behavior_weight = originals["weight"]
        trainer._PROBE_STEP_WEIGHT_HOOK = originals["hook"]
        trainer.EPOCHS = originals["epochs"]
        trainer.SEED = originals["seed"]
        trainer._residuals_with_trace = originals["trace"]
        trainer._contrastive_ce = originals["ce"]
        if originals["components"] is not None:
            trainer.behavior_contrastive_components = originals["components"]
        optimizer_cls.step = originals["step"]
        optimizer_cls.zero_grad = originals["zero_grad"]
        E.unload_trainer(saved_path, saved_mods, "r7a_capture_trainer")
        with contextlib.suppress(OSError):
            for path in sorted(scratch.rglob("*"), reverse=True):
                path.unlink() if path.is_file() else path.rmdir()
            scratch.rmdir()

    final_digest = state.get("final_parameter_sha256")
    if final_digest is None and models:
        final_digest = state_digest({name: value.detach().clone().cpu()
                                     for name, value in models[-1].state_dict().items()})

    return {
        "capture": {
            "instrumented": instrument,
            "final_parameter_sha256": final_digest,
            "fork_steps": sorted(state["bundles"]),
            "bundles": state["bundles"],
            "executed_steps": state["step"],
            "failures": failures,
            "elapsed_s": round(time.time() - started, 2),
            "schedule": {"baseline_steps": baseline_steps, "active_steps": active_steps,
                         "active_weight": active_weight, "seed": seed, "budget": budget},
        },
    }
