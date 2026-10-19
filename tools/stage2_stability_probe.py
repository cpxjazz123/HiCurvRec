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
        "a behaviour level's WEIGHTED encoder gradient exceeded ||grad(base_loss, encoder)||, "
        "both measured on the same step",
        lambda snap, _first: snap is not None
        and (snap.get("attribution") or {}).get("max_norm_over_base") is not None
        and snap["attribution"]["max_norm_over_base"] > 1.0),
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
    "measurement_version": 2,
    "supersedes": ("version 1 divided the per-level gradient norm by the total pre-clip norm of "
                   "every parameter and every loss, which is not the base-loss encoder gradient, "
                   "and regenerated contrastive labels with a cyclic shift instead of observing "
                   "the trainer's own terms. Those values are invalid and are not comparable with "
                   "version 2; the version-1 reports are left in place and are not claimed to be "
                   "correct"),
    "risk_labels_reevaluated": ("after the correction the labels were recomputed on the same four "
                                "arms; the outcome is recorded in RETROSPECTIVE_v2.md rather than "
                                "asserted here"),
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
    elif any(snapshot.get("nonfinite_steps", 0) > 0 for snapshot in snapshots):
        status = "invalid"
        worst = max(snapshot.get("nonfinite_steps", 0) for snapshot in snapshots)
        reason = (f"{worst} optimizer step(s) produced a non-finite measurement. A finite early "
                  "snapshot does not excuse a non-finite later one, so this is a failure rather "
                  "than an advisory field")
    elif max(snapshot.get("active_optimizer_steps", 0) for snapshot in snapshots) < 2:
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
        "observed_ce": [], "observed_base": None, "active_optimizer_steps": 0,
        "mechanism_call_count": 0, "applied_weights": [], "zero_measured": False,
        "nonfinite_detail": [], "step_gradients": [], "schedule": [],
        "current_weight": 0.0,
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
                "ce": trainer._contrastive_ce,
                "compute_loss": model_class.compute_loss,
                "weight": trainer.behavior_weight,
                "step": getattr(torch.optim, trainer.OPTIMIZER).step,
                "clip": torch.nn.utils.clip_grad_norm_,
                "backward": torch.Tensor.backward,
                "hook": getattr(trainer, "_PROBE_STEP_WEIGHT_HOOK", None),
            }
            optimizer_cls = getattr(torch.optim, trainer.OPTIMIZER)

            def capturing_init(inner_self, *args, **kwargs):
                originals["init"](inner_self, *args, **kwargs)
                captured_models.append(inner_self)

            def observing_ce(logits, source_ids, target_ids, successor_indicator=None):
                """Record the loss the TRAINER built, on the batch the TRAINER chose.

                The previous version regenerated a contrastive target with a cyclic shift and
                dropped `successor_indicator`, so the gradients it reported were not the training
                terms at all. Observing the call keeps the real labels, the real mask and the real
                temperature, and the value is returned unchanged.
                """
                loss = originals["ce"](logits, source_ids, target_ids, successor_indicator)
                if state["active"] and loss is not None:
                    state["observed_ce"].append(loss)
                    state["mechanism_call_count"] += 1
                return loss

            def observing_compute_loss(inner_self, embeddings, reconstructed, quant_loss):
                base_loss, recon_loss = originals["compute_loss"](
                    inner_self, embeddings, reconstructed, quant_loss)
                if state["active"]:
                    state["observed_base"] = base_loss
                    state["applied_weights"].append(
                        float(state.get("current_weight", 0.0)))
                return base_loss, recon_loss

            def tracing_trace(raw_module, embeddings):
                # The active window is counted where the mechanism actually runs, and the hook that
                # sets the per-step weight reads this same counter, so the schedule and the count
                # cannot disagree.
                stacked = originals["trace"](raw_module, embeddings)
                return stacked

            def weight_schedule(epoch: int, current: float | None = None) -> float:
                """The probe's schedule, in OPTIMIZER STEPS.

                Production's weight is an epoch constant evaluated once per epoch, so a schedule
                expressed in steps cannot be expressed through it. The trainer therefore exposes a
                probe-only hook evaluated per mini-batch; this function is what that hook calls.
                The step about to run is ``state["step"] + 1``, so steps 1..baseline use exactly 0
                and steps baseline+1..budget use exactly the declared weight.
                """
                if current is None:
                    # The per-EPOCH read. Production evaluates the weight once per epoch and the
                    # crowding ramp, the hyperbolic alpha and the gating that follows all read
                    # that value, so this must return exactly what production returns. Only the
                    # per-step hook below replaces the behaviour weight, and only for the
                    # behaviour term.
                    return originals["weight"](epoch)
                step_about_to_run = state["step"] + 1
                if step_about_to_run <= int(config["baseline_steps"]):
                    applied = 0.0
                elif step_about_to_run > budget:
                    applied = 0.0
                else:
                    applied = float(config["behavior_weight_active"])
                state["current_weight"] = applied
                # The SAME value decides whether the observers record, so the schedule and what
                # was observed cannot disagree: there is one source of truth, not a flag beside it.
                state["active"] = applied > 0.0
                state["schedule"].append({"step": step_about_to_run, "weight": applied})
                return applied

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

            def _encoder_grad(loss, encoder):
                """The real VJP of a loss the trainer built, on this step's graph."""
                if loss is None or not loss.requires_grad:
                    return None
                grads = torch.autograd.grad(loss, encoder, retain_graph=True, allow_unused=True)
                if all(g is None for g in grads):
                    return None
                return [torch.zeros_like(p) if g is None else g for p, g in zip(encoder, grads)]

            def sampling_backward(self, *args, **kwargs):
                """Sample the TRUE per-term gradients from THIS graph, then backward exactly once.

                The terms come from the trainer's own loss construction, observed as it ran: the
                real pair minibatch, the real source/target ids, the real duplicate and
                false-negative mask, the real temperature. Sampling before the trainer's own
                backward keeps the graph alive, so there is no second backward, no extra optimizer
                step and no mutation of `.grad`.
                """
                try:
                    if captured_models and state["active"]:
                        model = captured_models[-1]
                        encoder = [p for p in model.encoder.parameters()]
                        g_base = _encoder_grad(state["observed_base"], encoder)
                        g_levels = [_encoder_grad(loss, encoder) for loss in state["observed_ce"]]
                        state["step_gradients"].append({
                            "g_base": g_base,
                            "g_levels_raw": g_levels,
                            "level_count": len([g for g in g_levels if g is not None]),
                        })
                        state["sampled"] += len(g_levels)
                except Exception as exc:  # noqa: BLE001
                    note(f"gradient sampling failed: {type(exc).__name__}: {exc}")
                finally:
                    state["observed_ce"] = []
                    state["observed_base"] = None
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
                # Step 0: measured at the top of the first optimizer step, where the parameters
                # are still the cold initial ones -- the gradients exist but have not been applied.
                if models and 0 in snapshot_set and not state["zero_measured"]:
                    state["zero_measured"] = True
                    try:
                        snapshots.append(measure(models[-1], state, config, snapshots))
                    except Exception as exc:  # noqa: BLE001
                        note(f"step-0 snapshot failed: {type(exc).__name__}: {exc}")
                weights_before = None
                if models:
                    weights_before = [p.detach().clone()
                                      for p in models[-1].encoder.parameters()]
                result = originals["step"](self, *args, **kwargs)
                state["step"] += 1
                step = state["step"]
                if state["step"] > int(config["baseline_steps"]):
                    state["active_optimizer_steps"] += 1
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
                if models and step in snapshot_set and step not in {s["step"] for s in snapshots}:
                    try:
                        snapshots.append(measure(models[-1], state, config, snapshots))
                    except Exception as exc:  # noqa: BLE001
                        note(f"snapshot at step {step} failed: {type(exc).__name__}: {exc}")
                if step >= budget:
                    raise ProbeStop(f"probe step budget {budget} reached")
                return result

            model_class.__init__ = capturing_init
            trainer._residuals_with_trace = tracing_trace
            trainer._contrastive_ce = observing_ce
            model_class.compute_loss = observing_compute_loss
            trainer.behavior_weight = weight_schedule
            trainer._PROBE_STEP_WEIGHT_HOOK = weight_schedule
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
            trainer._contrastive_ce = originals["ce"]
            model_class.compute_loss = originals["compute_loss"]
            trainer.behavior_weight = originals["weight"]
            trainer._PROBE_STEP_WEIGHT_HOOK = originals["hook"]
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

    # The configured schedule must be what actually ran, not merely what was asked for: the
    # issue requires asserting the two agree, so the observed weights are compared step by step.
    expected_active = int(config["behavior_active_steps"])
    expected_baseline = int(config["baseline_steps"])
    schedule_rows = state["schedule"]
    baseline_steps_seen = [row for row in schedule_rows if row["weight"] == 0.0]
    active_steps_seen = [row for row in schedule_rows
                         if row["weight"] == float(config["behavior_weight_active"])]
    schedule_agrees = (len(schedule_rows) == budget
                       and len(baseline_steps_seen) == expected_baseline
                       and len(active_steps_seen) == expected_active
                       and all(row["step"] <= expected_baseline
                               for row in baseline_steps_seen)
                       and all(row["step"] > expected_baseline
                               for row in active_steps_seen))
    if not schedule_agrees:
        failures.append(
            f"schedule_mismatch: configured {expected_baseline} baseline + {expected_active} "
            f"active, observed {len(baseline_steps_seen)} baseline + {len(active_steps_seen)} "
            f"active over {len(schedule_rows)} step(s)")
    if state["active_optimizer_steps"] != expected_active:
        failures.append(
            f"active_step_mismatch: expected_active_steps={expected_active} "
            f"observed_active_steps={state['active_optimizer_steps']}")

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
        "schedule": {
            "configured_baseline_steps": expected_baseline,
            "configured_active_steps": expected_active,
            "observed_baseline_steps": len(baseline_steps_seen),
            "observed_active_steps": len(active_steps_seen),
            "agrees": schedule_agrees,
            "applied_weights_seen": sorted({row["weight"] for row in schedule_rows}),
        },
        "executed_steps": state["step"],
        "active_optimizer_steps": state["active_optimizer_steps"],
        "mechanism_call_count": state["mechanism_call_count"],
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
            "active_optimizer_steps": 0, "mechanism_call_count": 0,
            "production_artifacts_unchanged": True}


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


MEASUREMENT_VERSION = 2


def _norm(grads) -> float | None:
    if grads is None:
        return None
    return float(torch.linalg.vector_norm(torch.cat([g.reshape(-1) for g in grads])))


def attribute_gradients(step_record: dict[str, Any], weight: float,
                        valid_levels: int) -> dict[str, Any]:
    """Per-term encoder gradient attribution, with the CORRECT denominator.

    ``level_norm_over_base`` previously divided by ``preclip_total_norm`` -- the norm of EVERY
    parameter's gradient from EVERY loss, pre-clipping. That is not the base-loss encoder gradient
    and the two are not interchangeable, so the ratio and the `deep_gradient_domination` indicator
    built on it were wrong. The denominator here is ``||grad(base_loss, encoder_parameters)||``
    from the SAME step.

    A ratio is UNDEFINED when either norm is exactly zero; it is reported as ``None`` with a
    reason rather than as 0 or as a fabricated large number.
    """
    g_base = step_record.get("g_base")
    base_norm = _norm(g_base)
    rows = []
    for index, grads in enumerate(step_record.get("g_levels_raw") or []):
        raw = _norm(grads)
        if raw is None:
            rows.append({"level": index, "raw_norm": None, "weighted_norm": None,
                         "norm_over_base": None, "cosine_with_base": None,
                         "undefined_reason": "the level's gradient is disconnected from the "
                                             "encoder parameters on this step"})
            continue
        weighted = raw * (weight / max(valid_levels, 1))
        if base_norm is None or base_norm == 0.0 or weighted == 0.0:
            rows.append({"level": index, "raw_norm": raw, "weighted_norm": weighted,
                         "norm_over_base": None, "cosine_with_base": None,
                         "undefined_reason": "zero denominator: "
                                             + ("||g_base_enc|| == 0" if not base_norm
                                                else "||g_Lk_enc_weighted|| == 0")})
            continue
        flat_base = torch.cat([g.reshape(-1) for g in g_base])
        flat_level = torch.cat([g.reshape(-1) for g in grads])
        cosine = float((flat_level * flat_base).sum()) / (
            float(torch.linalg.vector_norm(flat_level)) * float(torch.linalg.vector_norm(flat_base)))
        rows.append({"level": index, "raw_norm": raw, "weighted_norm": weighted,
                     "norm_over_base": weighted / base_norm, "cosine_with_base": cosine,
                     "undefined_reason": None})
    ratios = [row["norm_over_base"] for row in rows if row["norm_over_base"] is not None]
    return {
        "g_base_enc_norm": base_norm,
        "levels": rows,
        "max_norm_over_base": max(ratios) if ratios else None,
        "ratios_defined": len(ratios),
    }


def measure(model, state: dict[str, Any], config: dict[str, Any],
            previous: list[dict[str, Any]]) -> dict[str, Any]:
    """One snapshot: RAW SID occupancy, the two collisions kept apart, and TRUE attribution."""
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

    record = state["step_gradients"][-1] if state["step_gradients"] else {}
    weight = state["applied_weights"][-1] if state["applied_weights"] else 0.0
    attribution = attribute_gradients(record, weight, int(config.get("valid_levels", 3)))
    preclip = state["preclip_norms"][-1] if state["preclip_norms"] else 0.0

    return {
        "measurement_version": MEASUREMENT_VERSION,
        "step": state["step"],
        "active_optimizer_steps": state["active_optimizer_steps"],
        "mechanism_call_count": state["mechanism_call_count"],
        "applied_behavior_weight": weight,
        "nonfinite_steps": state["nonfinite"],
        "finite_fraction": state["finite"] / max(state["step"], 1),
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
        "attribution": attribution,
        "preclip_total_norm": preclip,
        "clip_factor_min": min(state["clip_factors"]) if state["clip_factors"] else 0.0,
        "clip_factor_mean": (sum(state["clip_factors"]) / len(state["clip_factors"]))
        if state["clip_factors"] else 0.0,
        "update_to_weight_ratio_mean": (sum(state["update_ratio"]) / len(state["update_ratio"]))
        if state["update_ratio"] else 0.0,
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
        f"(behaviour-active optimizer steps {result['active_optimizer_steps']}, "
        f"mechanism calls {result['mechanism_call_count']})",
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
        "| step | active steps | mechanism calls | L1 used/256 | L1 H/log256 | L1 eff.codes | "
        "L1 max share | L1 collision | 3-level collision |",
        "|--:|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for snap in result["snapshots"]:
        lines.append(
            f"| {snap['step']} | {snap.get('active_optimizer_steps', 0)} | "
            f"{snap.get('mechanism_call_count', 0)} | "
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
    parser.add_argument("--out-root", default="results/stage2_RQ-VAE/stability_probe_v2",
                        help="never overwrite a previous measurement version's reports")
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
        REPO_ROOT / arguments.out_root / \
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
                      "active_optimizer_steps": result["active_optimizer_steps"],
                      "mechanism_call_count": result["mechanism_call_count"],
                      "fired_indicators": result.get("fired_indicators"),
                      "report": str(out_dir / "report.json")}, indent=2))
    return 0 if result["status"] == "valid" else 1


if __name__ == "__main__":
    sys.exit(main())
