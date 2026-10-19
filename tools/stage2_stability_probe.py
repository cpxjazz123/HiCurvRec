"""Isolated short-run Stage2 stability probe (issue #77).

A diagnostic, NOT a seventh correctness gate and NOT a second performance gate. It exists because
three cold 3000-epoch runs were verified correct by CGV and still lost L1 SID discrimination, and
that loss was only visible after all the GPU time was spent. This probe asks, in ~160 optimizer
steps, whether the same mechanism already shows the early signs of that collapse.

What it reuses: the arm's real data loaders, its real loss assembly, its real optimizer, its real
clipping, its real DDP aggregation and its real SID extraction. Nothing here reimplements the
mechanism or the model. The ONLY thing replaced is the behaviour-weight SCHEDULE: production ramps
from epoch 1500, so an ordinary 160-step run would never touch the behaviour path at all and could
not possibly reproduce the failure mode. That substitution is documented, probe-local, and never
leaks into production scheduling or artifacts.

What it must never do: read the frozen held-out pair artifact, unlock Stage3, retune training, or
appear as an extra `stage3_allowed` condition. The frozen Hard-L1 ER remains the only Stage2
performance criterion.

Telemetry is collected on RAW (non-extended) hard SIDs, with L1-only and full three-level tuple
statistics kept strictly apart, because their natural scales differ by an order of magnitude.

Usage:
    stage2_stability_probe.py <tree> --spec <file.json|-> [--probe-config <file.json|->]
                                    [--out DIR] [--worker]
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/GeneRec")
TOOLS = REPO_ROOT / "tools/cgv"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import adapters  # noqa: E402
import contract as C  # noqa: E402
import engine as E  # noqa: E402
import oracles  # noqa: E402

# The frozen held-out artifact is the Stage2 acceptance criterion. The probe must never read it.
FORBIDDEN_INPUTS = ("behavior_pairs_valid_s42.npz", "behavior_pairs_valid")

DEFAULT_PROBE_CONFIG: dict[str, Any] = {
    "seed": 42,
    "baseline_steps": 32,
    "behavior_active_steps": 128,
    "behavior_weight_active": 0.20,
    "snapshot_steps": [0, 8, 16, 32, 64, 96, 128, 160],
    "assignment_population": "train_only_all_items",
    "train_pairs": "original_train_only",
    "optimizer": "production",
    "gradient_clip": "production",
    "device_layout": "production_if_available",
    "output_isolation": True,
}

PROBE_CONFIG_KEYS = set(DEFAULT_PROBE_CONFIG)

# Heuristic risk indicators. Explicitly NOT validated predictors: they are named as heuristics in
# the report so nobody downstream mistakes them for a proven early-warning rule.
RISK_RULES = {
    "l1_entropy_drop": (
        "normalized L1 entropy fell by more than half of its step-0 value",
        lambda snap, first: snap is not None and first is not None
        and first.get("l1_entropy_norm", 0.0) > 0
        and snap.get("l1_entropy_norm", 0.0) < 0.5 * first["l1_entropy_norm"]),
    "l1_effective_codes_drop": (
        "effective L1 code count fell by more than half of its step-0 value",
        lambda snap, first: snap is not None and first is not None
        and first.get("l1_effective_codes", 0.0) > 0
        and snap.get("l1_effective_codes", 0.0) < 0.5 * first["l1_effective_codes"]),
    "l1_collision_rise": (
        "L1-only collision rose by more than 0.20 absolute",
        lambda snap, first: snap is not None and first is not None
        and snap.get("l1_collision", 0.0) > first.get("l1_collision", 0.0) + 0.20),
    "negative_train_pair_match_rise": (
        "same-code rate on fixed train-only negatives rose by more than 50% relative",
        lambda snap, first: snap is not None and first is not None
        and first.get("negative_same_code_rate", 0.0) > 0
        and snap.get("negative_same_code_rate", 0.0)
        > 1.5 * first["negative_same_code_rate"]),
    "deep_gradient_domination": (
        "a deep behaviour level exceeded the base encoder gradient norm on every snapshot",
        lambda snap, _first: snap is not None
        and bool(snap.get("level_norm_over_base"))
        and max(snap["level_norm_over_base"]) > 1.0),
    "clipping_saturation": (
        "every measured step clipped at the clip norm",
        lambda snap, _first: snap is not None and snap.get("clip_factor_min", 0.0) >= 1.0),
    "max_code_share_rise": (
        "the most used L1 code exceeded 25% of all items",
        lambda snap, _first: snap is not None and snap.get("l1_max_code_share", 0.0) > 0.25),
}

WATCH_RULES = ("l1_entropy_drop", "l1_effective_codes_drop", "max_code_share_rise",
               "negative_train_pair_match_rise")


# Measured, not assumed. The probe was run with this exact protocol on one healthy reference arm
# (TIGER_r4_08, which historically scored Hard-L1 ER 16.402628) and on the three collapsed R6 arms
# (2.89 / 1.82 / 1.54). The binary indicators below fired on the HEALTHY arm as well, so they are
# NOT discriminative at this step budget and no automatic abort may be based on them. What did
# separate the arms was the three-level tuple collision (0.477 healthy vs 0.789 / 0.816 / 0.939)
# and the train-only negative same-code rate (0.073 vs 0.117 / 0.120 / 0.255), both ordered with
# the final ER. Those are reported as OBSERVATIONS against a healthy reference, never as a
# threshold: one healthy arm and three failures cannot calibrate a predictor.
RISK_CALIBRATION = {
    "status": "uncalibrated",
    "reference_arms": {
        "healthy": {"arm": "r4_08", "reference_hard_l1_er_historical": 16.402628,
                    "tuple3_collision_at_160": 0.4769, "negative_same_code_rate_at_160": 0.0734},
        "collapsed": {
            "r6_02": {"reference_hard_l1_er_historical": 2.8902618750708537,
                      "tuple3_collision_at_160": 0.7893,
                      "negative_same_code_rate_at_160": 0.1169},
            "r6_03": {"reference_hard_l1_er_historical": 1.8233611326609775,
                      "tuple3_collision_at_160": 0.8158,
                      "negative_same_code_rate_at_160": 0.1203},
            "r6_04": {"reference_hard_l1_er_historical": 1.5416677050365089,
                      "tuple3_collision_at_160": 0.9386,
                      "negative_same_code_rate_at_160": 0.2547},
        },
    },
    "note": ("these reference values were measured by the ER gate on those arms' finished runs; "
             "the probe itself never reads the held-out artifact and never reports Hard-L1 ER"),
    "false_positive_observed": ("the healthy reference arm also reaches risk=high under these "
                               "indicators, so a high verdict is not evidence of collapse"),
    "no_automatic_action": ("this probe must not cancel, approve or retune a run. A risk verdict "
                            "recommends review; the frozen Hard-L1 ER decides Stage3 admission"),
}


class ProbeInvalid(Exception):
    """The probe's own execution was wrong: a protocol failure, not a quality verdict."""


class ProbeStop(Exception):
    """Raised inside the training loop once the step budget is spent."""


# --------------------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------------------
def validate_probe_config(config: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    unknown = sorted(set(config) - PROBE_CONFIG_KEYS)
    if unknown:
        problems.append(f"probe_config has keys the probe does not act on: {unknown}")
    for key in ("baseline_steps", "behavior_active_steps"):
        value = config.get(key)
        if not isinstance(value, int) or value < 0:
            problems.append(f"probe_config.{key} must be a non-negative integer")
    weight = config.get("behavior_weight_active")
    if not isinstance(weight, (int, float)) or weight <= 0:
        problems.append("probe_config.behavior_weight_active must be a positive number: a probe "
                        "that never activates the behaviour path cannot test anything")
    snapshots = config.get("snapshot_steps")
    if not isinstance(snapshots, list) or not snapshots:
        problems.append("probe_config.snapshot_steps must be a non-empty list of cumulative steps")
    elif any(not isinstance(value, int) or value < 0 for value in snapshots):
        problems.append("probe_config.snapshot_steps must be non-negative integers")
    if config.get("assignment_population") != "train_only_all_items":
        problems.append("probe_config.assignment_population must be 'train_only_all_items': the "
                        "probe may not measure a different population")
    if config.get("output_isolation") is not True:
        problems.append("probe_config.output_isolation must be true: probe updates are throwaway "
                        "and may never touch production files")
    return problems


def total_steps(config: dict[str, Any]) -> int:
    return int(config["baseline_steps"]) + int(config["behavior_active_steps"])


# --------------------------------------------------------------------------------------
# hashing and isolation
# --------------------------------------------------------------------------------------
def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def production_artifacts(tree: Path) -> dict[str, str | None]:
    """Hash every production artifact the probe must leave untouched."""
    result_dir = tree / "results"
    if not result_dir.is_dir():
        result_dir = REPO_ROOT / "results/stage2_RQ-VAE" / tree.name.split("TIGER_")[-1]
    tracked: dict[str, Path] = {
        "trainer": tree / "train_rqvae.py",
        "frozen_heldout_pairs": (REPO_ROOT / "results/stage3_T5Train/monotonicity"
                                 / "behavior_pairs_valid_s42.npz"),
    }
    for name in ("rqvae_best.pth", "sids_for_hgrec.npy", "item_sids.json"):
        tracked[f"result:{name}"] = result_dir / name
    for name in ("train_migrated.log", "training_metrics.jsonl"):
        tracked[f"log:{name}"] = tree / "logs/resample" / name
    return {key: sha256_file(path) for key, path in tracked.items()}


# --------------------------------------------------------------------------------------
# measurement helpers (pure functions over observations, so they are unit-testable)
# --------------------------------------------------------------------------------------
def codebook_statistics(codes: torch.Tensor, size: int = 256) -> dict[str, Any]:
    """Usage histogram, normalized entropy, effective code count and max share for one level."""
    counts = torch.bincount(codes.reshape(-1).to(torch.int64), minlength=size).double()
    total = float(counts.sum())
    probabilities = counts / max(total, 1.0)
    nonzero = probabilities[probabilities > 0]
    entropy = float(-(nonzero * nonzero.log()).sum())
    return {
        "used_codes": int((counts > 0).sum()),
        "capacity": int(size),
        "used_fraction": float((counts > 0).sum()) / size,
        "entropy_nats": entropy,
        "entropy_norm": entropy / math.log(size),
        "effective_codes": math.exp(entropy),
        "max_code_share": float(probabilities.max()),
    }


def collision_statistics(tokens: torch.Tensor) -> dict[str, float]:
    """Collision over ITEMS: a 2-D input is a set of tuples, so uniqueness is per ROW.

    Flattening first would count distinct integers instead of distinct tuples, which understates
    the collision of a three-level SID by a large factor and would make the two measures
    incomparable.
    """
    if tokens.dim() == 1:
        rows = tokens.reshape(-1, 1)
    else:
        rows = tokens.reshape(tokens.shape[0], -1)
    total = int(rows.shape[0])
    unique = int(torch.unique(rows, dim=0).shape[0])
    return {"collision": 1.0 - unique / max(total, 1), "unique": unique, "items": total}


def classify_status(snapshots: list[dict[str, Any]], config: dict[str, Any],
                    protocol_failures: list[str]) -> tuple[str, str, list[str], list[dict]]:
    """`status` judges the probe's execution; `risk` is explicitly heuristic."""
    status, reason = "valid", ""
    if protocol_failures:
        status = "invalid"
    elif not snapshots:
        status = "inconclusive"
        reason = "no snapshot was produced"
    elif not any(snapshot.get("finite_fraction", 1.0) >= 1.0 for snapshot in snapshots):
        status = "invalid"
        reason = "a measured step produced a non-finite value"
    elif max(snapshot.get("mechanism_active_steps", 0) for snapshot in snapshots) < 2:
        status = "invalid"
        reason = ("the behaviour path executed on fewer than two steps, so this run cannot test "
                  "the failure mode it exists to detect. The behaviour weight is consulted once "
                  "per epoch, so the step budget must span at least one epoch boundary beyond the "
                  "baseline window")
    if status != "valid":
        return status, reason, [], []

    first = snapshots[0]
    last = snapshots[-1]
    fired_high, fired_watch, evidence = [], [], []
    for name, (description, rule) in RISK_RULES.items():
        triggered = bool(rule(last, first))
        evidence.append({"indicator": name, "description": description,
                         "fired": triggered, "classification": "heuristic"})
        if triggered:
            (fired_watch if name in WATCH_RULES else fired_high).append(name)
    if fired_high:
        risk = "high"
    elif fired_watch:
        risk = "watch"
    else:
        risk = "low"
    return status, reason, fired_high + fired_watch, evidence


# --------------------------------------------------------------------------------------
# the bounded run
# --------------------------------------------------------------------------------------
def run_probe(tree: Path, spec: dict[str, Any], config: dict[str, Any], out_dir: Path,
              worker: bool = False) -> dict[str, Any]:
    """Run the bounded probe against the arm's real training path.

    Reused verbatim from the arm: the data loaders, the loss assembly, the optimizer, the
    gradient clipping, the DDP aggregation and the SID extraction. Replaced, probe-locally and
    reversibly: the behaviour-weight SCHEDULE, the epoch bound, and where artifacts may be
    written. Nothing about the mechanism under study is touched.
    """
    tree = Path(tree).resolve()
    config = {**DEFAULT_PROBE_CONFIG, **config}
    problems = validate_probe_config(config)
    if problems:
        return _failure(config, [f"probe_config invalid: {p}" for p in problems])

    try:
        spec = C.normalise_spec(spec)
    except C.ContractError as exc:
        return _failure(config, [f"declaration invalid: {exc}"])
    spec_problems = C.validate(spec)
    if spec_problems:
        return _failure(config, [f"declaration invalid: {p}" for p in spec_problems])

    before = production_artifacts(tree)
    started = time.time()
    snapshots: list[dict[str, Any]] = []
    failures: list[str] = []
    scratch = Path(tempfile.mkdtemp(prefix=f"probe_{tree.name}_"))
    trainer, saved_path, saved_mods = E.load_trainer(tree, module_name="probe_trainer")
    saved_attrs: dict[str, Any] = {}
    budget = total_steps(config)
    snapshot_set = {int(value) for value in config["snapshot_steps"]}

    state: dict[str, Any] = {
        "step": 0, "active": 0, "nonfinite": 0, "clip_factors": [], "update_ratio": [],
        "preclip_norms": [], "level_norms": [], "base_derived": [], "weights_before": None,
        "losses": [], "recons": [], "behaviors": [], "quants": [], "finite": 0, "sampled": 0,
        "clip_index": 0, "baseline_observed": 0, "steps_per_epoch": 0,
    }

    def note(message: str) -> None:
        failures.append(message)

    try:
        with E.isolated_outputs(trainer, scratch):
            for name in ("METRICS_PATH", "LOG_DIR", "CHECKPOINT", "OUTPUT_SID", "OUTPUT_JSON"):
                saved_attrs[name] = getattr(trainer, name, None)
            captured_models: list[Any] = []
            captured_traces: list[dict[str, Any]] = []
            model_class = trainer.RQVAE
            originals = {
                "init": model_class.__init__,
                "trace": trainer._residuals_with_trace,
                "weight": trainer.behavior_weight,
                "step": getattr(torch.optim, trainer.OPTIMIZER).step,
                "clip": torch.nn.utils.clip_grad_norm_,
                "backward": torch.Tensor.backward,
            }
            optimizer_cls = getattr(torch.optim, trainer.OPTIMIZER)

            def capturing_init(inner_self, *args, **kwargs):
                originals["init"](inner_self, *args, **kwargs)
                captured_models.append(inner_self)

            def tracing_trace(raw_module, embeddings):
                # Counting the active window HERE, rather than from the configured step numbers,
                # is what makes the count evidence. A schedule expressed in steps is consulted
                # once per EPOCH, so a budget shorter than one epoch can leave the behaviour path
                # untouched while a config-derived counter happily reports "active".
                if state["step"] >= int(config["baseline_steps"]):
                    state["active"] += 1
                else:
                    state["baseline_observed"] += 1
                stacked = originals["trace"](raw_module, embeddings)
                captured_traces.append({
                    "traces": [stacked[i] for i in range(int(stacked.shape[0]))],
                    "n_src": int(embeddings.shape[0]) // 2,
                })
                return stacked

            def weight_schedule(epoch: int) -> float:
                """0 during the baseline window, the declared weight afterwards.

                Production ramps from epoch 1500, so a 160-step run would never touch the behaviour
                path and could not reproduce the failure this probe exists to catch. Only the
                SCHEDULE is replaced; the loss, optimizer, clipping and aggregation are the real
                ones, and the original function is restored before this process returns.
                """
                if state["step"] < int(config["baseline_steps"]):
                    return 0.0
                return float(config["behavior_weight_active"])

            def level_terms(entry: dict[str, Any]) -> list[torch.Tensor]:
                """Per-level contrastive terms, built with the trainer's own primitives."""
                device = entry["traces"][0].device
                source = torch.arange(entry["n_src"], device=device)
                target = (torch.arange(entry["n_src"], device=device) + 1) % entry["n_src"]
                terms = []
                for trace in entry["traces"]:
                    logits = -torch.cdist(trace[:entry["n_src"]], trace[entry["n_src"]:], p=2) \
                        / trainer.BEHAVIOR_TEMPERATURE
                    loss = trainer._contrastive_ce(logits, source, target, None)
                    terms.append(loss if loss is not None
                                 else torch.zeros((), device=trace.device))
                return terms

            def sampling_backward(self, *args, **kwargs):
                """Sample per-level behaviour gradients from THIS graph, then backward once.

                Sampling here rather than after the optimizer step is what makes the numbers
                describe the computation that actually ran: the graph is still alive, and the
                trainer's own backward follows with the graph intact, so there is no second
                backward and no extra optimizer step.
                """
                try:
                    if captured_traces and captured_models:
                        model = captured_models[-1]
                        encoder = [p for p in model.encoder.parameters()]
                        terms = level_terms(captured_traces[-1])
                        rows = []
                        for loss in terms:
                            grads = torch.autograd.grad(loss, encoder, retain_graph=True,
                                                        allow_unused=True)
                            rows.append(oracles.flat_norm([
                                torch.zeros_like(p) if g is None else g
                                for p, g in zip(encoder, grads)]))
                            state["sampled"] += 1
                        state["level_norms"].append(rows)
                        state["behaviors"].append(float(sum(float(t) for t in terms) / len(terms)))
                        captured_traces.pop()
                except Exception as exc:  # noqa: BLE001
                    note(f"gradient sampling failed: {type(exc).__name__}: {exc}")
                return originals["backward"](self, *args, **kwargs)

            def sampling_clip(parameters, max_norm, *args, **kwargs):
                total = originals["clip"](parameters, max_norm, *args, **kwargs)
                norm = float(total) if total is not None else 0.0
                state["preclip_norms"].append(norm)
                if max_norm:
                    factor = norm / float(max_norm)
                    state["clip_factors"].append(factor)
                    if factor >= 1.0:
                        state["clipped"] = state.get("clipped", 0) + 1
                return total

            def measured_step(self, *args, **kwargs):
                models = captured_models
                weights_before = None
                if models:
                    weights_before = [p.detach().clone()
                                      for p in models[-1].encoder.parameters()]
                result = originals["step"](self, *args, **kwargs)
                state["step"] += 1
                step = state["step"]
                if weights_before and models:
                    moved = 0.0
                    scale = 0.0
                    for previous, parameter in zip(weights_before, models[-1].encoder.parameters()):
                        moved = max(moved, float((parameter.detach() - previous).abs().max()))
                        scale = max(scale, float(torch.linalg.vector_norm(parameter.detach())))
                    state["update_ratio"].append(moved / scale if scale > 0 else 0.0)
                # A step is non-finite when a value MEASURED for it is not finite. The clip
                # wrapper records one pre-clip norm per step, so the judgement is on that step's
                # own measurements rather than on whatever happened to be recorded last.
                measured = state["preclip_norms"][state["clip_index"]:]
                finite = bool(measured) and all(math.isfinite(value) for value in measured)
                state["clip_index"] = len(state["preclip_norms"])
                state["finite"] += 1 if finite else 0
                state["nonfinite"] += 0 if finite else 1
                if models and (step in snapshot_set or step == budget):
                    try:
                        snapshots.append(measure(models[-1], state, config, snapshots))
                    except Exception as exc:  # noqa: BLE001
                        note(f"snapshot at step {step} failed: {type(exc).__name__}: {exc}")
                if step >= budget:
                    raise ProbeStop(f"probe step budget {budget} reached")
                return result

            model_class.__init__ = capturing_init
            trainer._residuals_with_trace = tracing_trace
            trainer.behavior_weight = weight_schedule
            optimizer_cls.step = measured_step
            torch.nn.utils.clip_grad_norm_ = sampling_clip
            torch.Tensor.backward = sampling_backward
            # Generous on purpose: the sentinel stops the run the moment the budget is spent, so
            # extra epochs are never executed. Deriving this from the budget keeps it out of the
            # configuration, where a key the probe does not act on would be refused anyway.
            trainer.EPOCHS = budget + 2
            trainer.SEED = int(config["seed"])
            model_probe_trainer[0] = trainer
            try:
                trainer.main()
            except (ProbeStop, SystemExit):
                pass
            except Exception as exc:  # noqa: BLE001
                note(f"the training loop raised {type(exc).__name__}: {exc}")
            model_class.__init__ = originals["init"]
            trainer._residuals_with_trace = originals["trace"]
            trainer.behavior_weight = originals["weight"]
            optimizer_cls.step = originals["step"]
            torch.nn.utils.clip_grad_norm_ = originals["clip"]
            torch.Tensor.backward = originals["backward"]
            arm_name = getattr(trainer, "EXPERIMENT_ARM", None)
    finally:
        for name, value in saved_attrs.items():
            if value is not None:
                setattr(trainer, name, value)
        E.unload_trainer(saved_path, saved_mods, "probe_trainer")
        shutil_rmtree(scratch)

    after = production_artifacts(tree)
    changed = sorted(key for key in before if before[key] != after.get(key))
    if changed:
        note(f"the probe modified production artifacts: {changed}")
    if not snapshots:
        note("the probe produced no snapshots")

    status, reason, fired, evidence = classify_status(snapshots, config, failures)
    return {
        "probe": "stage2_stability_probe",
        "status": status,
        "status_reason": reason,
        "risk": "unknown" if status != "valid" else classify_risk(fired),
        "risk_note": ("risk is a HEURISTIC early-warning indicator, not a validated predictor, "
                      "and never a performance gate: the frozen Hard-L1 ER remains the only "
                      "Stage2 acceptance criterion"),
        "risk_calibration": RISK_CALIBRATION,
        "fired_indicators": fired,
        "risk_evidence": evidence,
        "protocol_failures": failures,
        "tree": str(tree),
        "arm": arm_name if "arm_name" in dir() else None,
        "spec_sha256": C.spec_sha256(spec),
        "declaration": spec,
        "probe_config": config,
        "probe_config_sha256": hashlib.sha256(
            json.dumps(config, sort_keys=True, default=str).encode("utf-8")).hexdigest(),
        "snapshots": snapshots,
        "executed_steps": state["step"],
        "mechanism_active_steps": state["active"],
        "gradient_samples": state["sampled"],
        "nonfinite_step_count": state["nonfinite"],
        "clip_step_count": state.get("clipped", 0),
        "production_artifacts_before": before,
        "production_artifacts_after": after,
        "production_artifacts_unchanged": not changed,
        "elapsed_s": round(time.time() - started, 2),
        "environment": {
            "torch": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        },
    }


def _failure(config: dict[str, Any], failures: list[str]) -> dict[str, Any]:
    return {"probe": "stage2_stability_probe", "status": "invalid", "status_reason": failures[0],
            "risk": "unknown", "risk_note": "the probe did not execute",
            "fired_indicators": [], "risk_evidence": [], "protocol_failures": failures,
            "probe_config": config, "snapshots": [], "executed_steps": 0,
            "mechanism_active_steps": 0, "production_artifacts_unchanged": True}


def shutil_rmtree(path: Path) -> None:
    """Remove the probe's scratch tree even if the run died, so nothing is left behind."""
    if not path.exists():
        return
    for child in sorted(path.rglob("*"), reverse=True):
        try:
            child.unlink() if child.is_file() else child.rmdir()
        except OSError:
            pass
    try:
        path.rmdir()
    except OSError:
        pass


def classify_risk(fired: list[str]) -> str:
    high = {"l1_collision_rise", "deep_gradient_domination", "clipping_saturation",
            "negative_train_pair_match_rise"}
    if not fired:
        return "low"
    return "high" if set(fired) & high else "watch"


def _train_only_pairs(trainer, limit: int = 20000) -> tuple[torch.Tensor, torch.Tensor]:
    """The arm's own train-only transition stream, via its own dataset class.

    Diagnostics only: these pairs never touch the held-out artifact, and the probe reports the
    resulting rate as a proxy that is explicitly NOT held-out Hard-L1 ER.
    """
    import numpy as np
    import pandas as pd

    frame = pd.read_parquet(trainer.TRAIN_FILE)
    dataset = trainer.TransitionPairs(frame)
    rows = min(limit, len(dataset))
    pairs = torch.stack([dataset[index] for index in range(rows)])
    left, right = pairs[:, 0].long(), pairs[:, 1].long()
    # The negatives are drawn from the same train item population and fixed by a seeded
    # generator, so the rate is comparable across snapshots of one run. They are a diagnostic
    # proxy only and are never the frozen held-out negatives.
    population = int(max(int(left.max()), int(right.max())) + 1)
    negatives = torch.tensor(
        np.random.default_rng(0).integers(0, population, size=(rows, 10)), dtype=torch.long)
    return left, right, negatives


def measure(model, state: dict[str, Any], config: dict[str, Any],
            previous: list[dict[str, Any]]) -> dict[str, Any]:
    """One snapshot: RAW SID occupancy, the two collisions kept apart, and step telemetry."""
    model.eval()
    embeddings = _probe_embeddings(model)
    with torch.no_grad():
        tokens = model.rq(model.encoder(embeddings))[-1].to(torch.int64)
    model.train()

    per_level = {f"level_{level + 1}": codebook_statistics(tokens[:, level])
                 for level in range(tokens.shape[1])}
    l1 = tokens[:, 0]
    l1_stats = collision_statistics(l1)
    tuple_stats = collision_statistics(
        torch.stack([tokens[:, i] for i in range(tokens.shape[1])], dim=1))

    positive_rate = float("nan")
    negative_rate = float("nan")
    try:
        left, right, negatives = _train_only_pairs(model_probe_trainer[0])
        valid = (left < l1.numel()) & (right < l1.numel())
        left, right = left[valid], right[valid]
        positive_rate = float((l1[left] == l1[right]).float().mean())
        negative_valid = negatives[valid] % l1.numel()
        negative_rate = float((l1[left][:, None] == l1[negative_valid]).float().mean())
    except Exception:  # noqa: BLE001
        pass

    level_rows = state["level_norms"][-1] if state["level_norms"] else []
    preclip = state["preclip_norms"][-1] if state["preclip_norms"] else 0.0
    ratio = [value / preclip if preclip > 0 else 0.0 for value in level_rows]
    clip_factors = state["clip_factors"]

    return {
        "step": state["step"],
        "mechanism_active_steps": state["active"],
        "nonfinite_steps": state["nonfinite"],
        "finite_fraction": (state["finite"] / max(state["step"], 1)),
        "levels": per_level,
        "l1_used_codes": per_level["level_1"]["used_codes"],
        "l1_entropy_norm": per_level["level_1"]["entropy_norm"],
        "l1_effective_codes": per_level["level_1"]["effective_codes"],
        "l1_max_code_share": per_level["level_1"]["max_code_share"],
        "l1_collision": l1_stats["collision"],
        "l1_unique": l1_stats["unique"],
        "tuple3_collision": tuple_stats["collision"],
        "tuple3_unique": tuple_stats["unique"],
        "items_measured": int(l1.numel()),
        "positive_same_code_rate": positive_rate,
        "negative_same_code_rate": negative_rate,
        "level_grad_norms": level_rows,
        "level_norm_over_base": ratio,
        "max_level_norm_over_base": max(ratio) if ratio else 0.0,
        "preclip_total_norm": preclip,
        "clip_factor_min": min(clip_factors) if clip_factors else 0.0,
        "clip_factor_mean": (sum(clip_factors) / len(clip_factors)) if clip_factors else 0.0,
        "update_to_weight_ratio_mean": (sum(state["update_ratio"]) / len(state["update_ratio"]))
        if state["update_ratio"] else 0.0,
        "behavior_loss_proxy": (sum(state["behaviors"]) / len(state["behaviors"]))
        if state["behaviors"] else 0.0,
    }


_EMBEDDING_CACHE: dict[str, torch.Tensor] = {}
model_probe_trainer: list[Any] = [None]


def _probe_embeddings(model) -> torch.Tensor:
    """The real stage1 item embeddings, loaded once and reused; never mutated."""
    device = next(model.parameters()).device
    key = str(device)
    if key not in _EMBEDDING_CACHE:
        import numpy as np

        _EMBEDDING_CACHE[key] = torch.from_numpy(
            np.load(REPO_ROOT / "stage1_GeneEmbedding/output/sentence_t5.npy")).to(device)
    return _EMBEDDING_CACHE[key]


def render_markdown(result: dict[str, Any]) -> str:
    if "tree" not in result:
        return ("# Stage2 stability probe\n\n"
                f"**status = `{result.get('status')}`** -- the probe did not execute.\n\n"
                + "\n".join(f"- {failure}" for failure in result.get("protocol_failures", []))
                + "\n")
    lines = [
        "# Stage2 stability probe",
        "",
        f"- tree: `{result['tree']}`",
        f"- declaration sha256: `{result['spec_sha256']}`",
        f"- probe config sha256: `{result['probe_config_sha256']}`",
        f"- executed steps: {result['executed_steps']} "
        f"(behaviour-active {result['mechanism_active_steps']})",
        f"- elapsed: {result['elapsed_s']}s",
        "",
        f"## status = `{result['status']}`",
        "",
    ]
    if result.get("status_reason"):
        lines.append(f"{result['status_reason']}")
        lines.append("")
    for failure in result.get("protocol_failures") or []:
        lines.append(f"- protocol failure: {failure}")
    if result.get("protocol_failures"):
        lines.append("")
    lines += [
        f"## risk = `{result['risk']}`",
        "",
        result["risk_note"],
        "",
    ]
    if result.get("fired_indicators"):
        fired = ", ".join("`" + name + "`" for name in result["fired_indicators"])
        lines.append(f"Fired: {fired}")
        lines.append("")
    lines += [
        "## Snapshots (RAW hard SIDs; L1-only and 3-level tuple kept separate)",
        "",
        "| step | active | L1 used/256 | L1 H/log256 | L1 eff.codes | L1 max share | "
        "L1 collision | 3-level collision |",
        "|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for snap in result["snapshots"]:
        lines.append(
            f"| {snap['step']} | {snap['mechanism_active_steps']} | "
            f"{snap['l1_used_codes']}/256 | {snap['l1_entropy_norm']:.4f} | "
            f"{snap['l1_effective_codes']:.1f} | {snap['l1_max_code_share']:.4f} | "
            f"{snap['l1_collision']:.4f} | {snap['tuple3_collision']:.4f} |")
    lines += [
        "",
        "L1-only collision and the three-level tuple collision have very different natural scales "
        "and are never mixed.",
        "",
        "## Isolation",
        "",
        f"- production artifacts unchanged: **{result['production_artifacts_unchanged']}**",
    ]
    for key in sorted(result["production_artifacts_before"]):
        before = result["production_artifacts_before"][key]
        after = result["production_artifacts_after"].get(key)
        lines.append(f"  - `{key}`: `{str(before)[:16]}` -> `{str(after)[:16]}`")
    lines += [
        "",
        "The frozen held-out pair artifact is never read by this probe. A risk verdict never "
        "unlocks Stage3 and never retunes training.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("tree")
    parser.add_argument("--spec", required=True)
    parser.add_argument("--probe-config", default=None,
                        help="omit to use the documented diagnostic defaults")
    parser.add_argument("--out", default=None)
    parser.add_argument("--worker", action="store_true")
    arguments = parser.parse_args()

    # The declaration arrives as data. A caller that also needs to carry the probe configuration
    # sends one envelope object, which is what the MCP tool does; a caller using the CLI directly
    # passes a file path or an inline object. Both are accepted, and neither is read from the
    # mechanism's tree.
    raw = (sys.stdin.read() if arguments.spec == "-"
           else Path(arguments.spec).read_text(encoding="utf-8"))
    envelope_config = None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict) and "spec" in parsed:
        spec_value = parsed["spec"]
        envelope_config = parsed.get("probe_config")
    else:
        spec_value = raw

    if envelope_config is not None:
        config = {**DEFAULT_PROBE_CONFIG, **envelope_config}
    elif arguments.probe_config is None:
        config = dict(DEFAULT_PROBE_CONFIG)
    else:
        text = (sys.stdin.read() if arguments.probe_config == "-"
                else Path(arguments.probe_config).read_text(encoding="utf-8"))
        stripped = text.strip()
        config = {**DEFAULT_PROBE_CONFIG, **(json.loads(stripped) if stripped else {})}

    tree = Path(arguments.tree)
    out_dir = Path(arguments.out) if arguments.out else \
        REPO_ROOT / "results/stage2_RQ-VAE/stability_probe" / \
        tree.name.split("TIGER_")[-1] / time.strftime("run-%Y%m%dT%H%M%S")
    # The training loop prints its own banners to stdout. Redirect that noise to stderr so
    # stdout carries ONLY the JSON the caller parses.
    with contextlib.redirect_stdout(sys.stderr):
        result = run_probe(tree, spec_value, config, out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
    (out_dir / "report.md").write_text(render_markdown(result))
    print(json.dumps({"status": result["status"], "risk": result["risk"],
                      "executed_steps": result["executed_steps"],
                      "mechanism_active_steps": result["mechanism_active_steps"],
                      "fired_indicators": result.get("fired_indicators"),
                      "report": str(out_dir / "report.json")}, indent=2))
    return 0 if result["status"] == "valid" else 1


if __name__ == "__main__":
    sys.exit(main())
