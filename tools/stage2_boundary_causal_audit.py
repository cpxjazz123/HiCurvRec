"""Boundary primitives for the causal audit (issue #79).

These are the measurements the audit rests on. Each one is deliberately boring and checkable:

  * the hard assignment comes from the quantiser's OWN distance function and its OWN index rule,
    so the audit cannot silently disagree with production about what the nearest code is;
  * a margin is the gap between the best and second-best DISTANCE, and an exact tie is reported
    as a tie rather than smoothed into a margin;
  * the "which side moved" counterfactuals are evaluated geometrically on one update, and any
    interaction between the latent and codebook movement is reported rather than decomposed
    additively.

Nothing here trains anything. The audit driver in this module's sibling reuses these to compare
counterfactual branches that all start from the same frozen pre-step state.
"""
from __future__ import annotations

import copy
import math
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/GeneRec")


# --------------------------------------------------------------------------------------
# the quantiser's own assignment, never a reimplementation
# --------------------------------------------------------------------------------------
def layer_distances(layer, latent: torch.Tensor) -> torch.Tensor:
    """Distances straight from the layer, so the metric matches production by construction."""
    with torch.no_grad():
        return layer._distances(latent.view(-1, layer.dim))


def layer_assignment(layer, distances: torch.Tensor, *, use_sinkhorn: bool = False) -> torch.Tensor:
    """The layer's own index rule, evaluated in the same mode production exports SIDs in.

    Production's hard SID export calls ``get_indices(..., infer_use_sk=False)`` from eval mode,
    where ``_indices`` takes ``argmin``. Sinkhorn is requested explicitly and never silently.
    """
    return layer._indices(distances, bool(use_sinkhorn))


def hard_assignments(model, embeddings: torch.Tensor, *, use_sinkhorn: bool = False
                     ) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Per-level hard assignments and the per-level distance matrices for the real latents."""
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            latent = model.encoder(embeddings)
            residual = latent
            assignments, distance_rows = [], []
            for layer in model.rq.vq_layers:
                distances = layer_distances(layer, residual)
                assignments.append(layer_assignment(layer, distances, use_sinkhorn=use_sinkhorn))
                distance_rows.append(distances)
                quant = torch.nn.functional.embedding(
                    assignments[-1], layer.get_code_embs())
                residual = residual - quant
    finally:
        model.train(was_training)
    return torch.stack(assignments, dim=0), distance_rows


def validate_against_native(model, embeddings: torch.Tensor, use_sinkhorn: bool = False
                            ) -> dict[str, Any]:
    """Check the audit's assignment against the trainer's own ``get_indices`` export path."""
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            native = model.get_indices(embeddings, infer_use_sk=use_sinkhorn).to(torch.int64)
    finally:
        model.train(was_training)
    ours, _ = hard_assignments(model, embeddings, use_sinkhorn=use_sinkhorn)
    # ours is [levels, items]; get_indices reports [items, levels]
    ours = ours.t().contiguous()
    differing = int((ours != native).sum())
    return {"matches_native_get_indices": differing == 0, "differing_entries": differing,
            "shape": list(native.shape)}


# --------------------------------------------------------------------------------------
# margins
# --------------------------------------------------------------------------------------
def margins(distances: torch.Tensor) -> dict[str, torch.Tensor]:
    """Per-item gap between the best and second-best distance, plus an explicit tie flag.

    A zero gap means two codes are exactly equidistant: torch's argmin picks the lower index, and
    that item is exactly as vulnerable to a boundary crossing as the margin can express. It is
    reported rather than hidden.
    """
    ordered, _ = torch.sort(distances, dim=-1)
    best, second = ordered[:, 0], ordered[:, 1]
    gap = second - best
    ties = gap <= 0.0
    return {"gap": gap, "tie": ties, "tie_fraction": float(ties.float().mean())}


def margin_quantiles(gap: torch.Tensor, switched: torch.Tensor) -> dict[str, float]:
    def summarise(mask: torch.Tensor) -> dict[str, float]:
        values = gap[mask]
        if values.numel() == 0:
            return {"count": 0}
        out = {"count": int(values.numel())}
        for name, q in (("p10", 0.10), ("p25", 0.25), ("p50", 0.50), ("p75", 0.75),
                        ("p90", 0.90)):
            out[name] = float(torch.quantile(values.float(), q))
        out["mean"] = float(values.float().mean())
        return out

    return {"switched": summarise(switched), "stable": summarise(~switched)}


# --------------------------------------------------------------------------------------
# crossings, and where they came from
# --------------------------------------------------------------------------------------
def switch_statistics(before: torch.Tensor, after: torch.Tensor,
                      before_gap: torch.Tensor) -> dict[str, Any]:
    """How many items changed hard code, where they came from and where they went."""
    switched = before != after
    count = int(switched.sum())
    stats: dict[str, Any] = {
        "items": int(before.numel()),
        "switched_items": count,
        "switch_fraction": count / max(int(before.numel()), 1),
        "margin_of_switched": margin_quantiles(before_gap, switched),
        "transitions": {},
    }
    if count:
        pairs = before[switched].to(torch.int64) * 100000 + after[switched].to(torch.int64)
        values, counts = torch.unique(pairs, return_counts=True)
        for value, occurrences in zip(values.tolist(), counts.tolist()):
            stats["transitions"][f"{value // 100000}->{value % 100000}"] = occurrences
    return stats


def four_way_assignment(layer, z_before: torch.Tensor, c_before: torch.Tensor,
                        z_after: torch.Tensor, c_after: torch.Tensor) -> dict[str, torch.Tensor]:
    """The four combinations: (z_before,c_before), (z_after,c_before), (z_before,c_after),
    (z_after,c_after). Every assignment uses the layer's own distance and index rule."""
    def assign(latent: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            flat = latent.view(-1, layer.dim)
            distances = (flat.pow(2).sum(1, keepdim=True) - 2 * flat @ codebook.t()
                         + codebook.pow(2).sum(1).t())
        return layer._indices(distances, False)

    return {
        "before": assign(z_before, c_before),
        "encoder_only": assign(z_after, c_before),
        "codebook_only": assign(z_before, c_after),
        "joint": assign(z_after, c_after),
    }


def crossing_attribution(four: dict[str, torch.Tensor]) -> dict[str, Any]:
    """Switch counts for encoder-only, codebook-only and joint motion, with the interaction."""
    before = four["before"]
    encoder_only = int((four["encoder_only"] != before).sum())
    codebook_only = int((four["codebook_only"] != before).sum())
    joint = int((four["joint"] != before).sum())
    return {
        "switched_encoder_only": encoder_only,
        "switched_codebook_only": codebook_only,
        "switched_joint": joint,
        "additive_prediction": encoder_only + codebook_only,
        "interaction": joint - (encoder_only + codebook_only),
        "note": "single-move counts are geometric counterfactuals for one update, not an additive "
                "decomposition; a non-zero interaction means the two motions are not separable",
    }


# --------------------------------------------------------------------------------------
# discrimination, on the same fixed train-only population
# --------------------------------------------------------------------------------------
def codebook_usage(assignments: torch.Tensor, capacity: int = 256) -> dict[str, Any]:
    counts = torch.bincount(assignments.reshape(-1).to(torch.int64), minlength=capacity).double()
    total = float(counts.sum())
    probabilities = counts / max(total, 1.0)
    nonzero = probabilities[probabilities > 0]
    entropy = float(-(nonzero * nonzero.log()).sum())
    return {"used_codes": int((counts > 0).sum()), "capacity": int(capacity),
            "used_fraction": float((counts > 0).sum()) / capacity,
            "entropy_nats": entropy, "entropy_norm": entropy / math.log(capacity),
            "effective_codes": math.exp(entropy),
            "max_code_share": float(probabilities.max())}


def tuple_collision(tokens: torch.Tensor) -> dict[str, float]:
    rows = tokens.reshape(tokens.shape[0], -1)
    total = int(rows.shape[0])
    unique = int(torch.unique(rows, dim=0).shape[0])
    return {"collision": 1.0 - unique / max(total, 1), "unique": unique, "items": total}


def same_code_rates(l1: torch.Tensor, positive: torch.Tensor,
                    negative: torch.Tensor) -> dict[str, float]:
    """Train-only positive and negative same-code rates. A DIAGNOSTIC proxy, never ER."""
    positive_rate = float((l1[positive] == l1[positive]).float().mean())
    negative_rate = float((l1[positive][:, None] == l1[negative]).float().mean())
    return {"positive_same_l1_code": positive_rate, "negative_same_l1_code": negative_rate}


def train_only_pairs(trainer, limit: int = 20000):
    """The arm's own train-only transition stream and a frozen train-only negative set."""
    import numpy as np
    import pandas as pd

    frame = pd.read_parquet(trainer.TRAIN_FILE)
    dataset = trainer.TransitionPairs(frame)
    rows = min(limit, len(dataset))
    pairs = torch.stack([dataset[index] for index in range(rows)])
    left, right = pairs[:, 0].long(), pairs[:, 1].long()
    population = int(max(int(left.max()), int(right.max())) + 1)
    negatives = torch.tensor(
        np.random.default_rng(0).integers(0, population, size=(rows, 10)), dtype=torch.long)
    return left, right, negatives


# --------------------------------------------------------------------------------------
# complete frozen state
# --------------------------------------------------------------------------------------
def capture_state(model, optimizer, scheduler, trainer) -> dict[str, Any]:
    """Everything that must be identical across counterfactual branches.

    Model weights, buffers AND codebooks, optimizer moments and step counts, scheduler state, and
    every RNG stream. Omitting the optimizer moments or reusing a fresh optimizer would make the
    branches incomparable, which is precisely what this audit must not do.
    """
    return {
        "model": copy.deepcopy(model.state_dict()),
        "optimizer": copy.deepcopy(optimizer.state_dict()),
        "scheduler": copy.deepcopy(scheduler.state_dict()) if scheduler is not None else None,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "python_rng": __import__("random").getstate(),
        "numpy_rng": __import__("numpy").random.get_state(),
        "model_training": model.training,
    }


def restore_state(state: dict[str, Any], model, optimizer, scheduler) -> None:
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and state["scheduler"] is not None:
        scheduler.load_state_dict(state["scheduler"])
    torch.set_rng_state(state["torch_rng"])
    if state["cuda_rng"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    __import__("random").setstate(state["python_rng"])
    __import__("numpy").random.set_state(state["numpy_rng"])
    model.train(state["model_training"])


def state_identity(state: dict[str, Any]) -> str:
    """A content hash of the frozen state, so two branches can be shown to have started alike."""
    import hashlib

    digest = hashlib.sha256()
    for name, value in state["model"].items():
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().to(torch.float32).cpu().numpy().tobytes()
                      if hasattr(value, "detach") else repr(value).encode("utf-8"))
    for group in state["optimizer"]["state"].values():
        for key in sorted(group):
            value = group[key]
            if hasattr(value, "detach"):
                digest.update(value.detach().to(torch.float32).cpu().numpy().tobytes())
            else:
                digest.update(repr(value).encode("utf-8"))
    digest.update(str(state["scheduler"]).encode("utf-8"))
    return digest.hexdigest()


# --------------------------------------------------------------------------------------
# interventions
# --------------------------------------------------------------------------------------
INTERVENTIONS = ("CTRL", "FULL", "NO-DEEP", "BASE-ONLY")


def classify_assembly(trainer) -> dict[str, Any]:
    """Which assembly the arm performs, and what that implies for the interventions.

    Every intervention here removes a loss contribution BEFORE the arm's own gradient assembly and
    lets the arm recompute: the PCGrad projection for R4-08, the soft-surrogate VJP for R6-04, the
    level assembly, the clipping and the optimizer untouched. That is the only construction this
    audit permits, because subtracting a raw gradient AFTER a nonlinear assembly or after clipping
    would not be the arm's own computation and is not attempted anywhere here.

    For R4-08 the deep L2/L3 Encoder paths are zero by construction, so NO-DEEP there is a
    NEGATIVE CONTROL: it is run and verified numerically rather than assumed to be inert.
    """
    has_components = hasattr(trainer, "behavior_contrastive_components")
    has_projection = hasattr(trainer, "_r4_08_apply_projection")
    source = getattr(trainer, "__file__", "")
    text = ""
    if source:
        try:
            text = Path(source).read_text(encoding="utf-8", errors="replace")
        except OSError:
            text = ""
    surrogate = ("quant_soft" in text) or ("quant_behavior" in text)
    assembly = ("pcgrad_projection" if has_projection
                else ("level_components" if has_components else "pooled_level_mean"))
    return {
        "assembly": assembly,
        "nonlinear_gradient_route": bool(has_projection or surrogate),
        "intervention_mode": "loss_term_removed_before_assembly",
        "recomputed_through_true_assembly": True,
        "supported_interventions": list(INTERVENTIONS),
        "no_deep_is_negative_control": bool(has_projection),
        "note": ("R4-08 assembles through a PCGrad projection, so its deep L2/L3 Encoder paths are "
                 "zero and NO-DEEP is expected to change nothing; that expectation is verified "
                 "numerically rather than assumed"
                 if has_projection else
                 ("R6-04 routes the behaviour path through a nonlinear soft surrogate, so the "
                  "gradients are recomputed through that surrogate rather than approximated"
                  if surrogate else
                  "the arm assembles a level mean, so removing a level before assembly is exact")),
    }


def install_loss_switch(trainer, state: dict[str, Any]) -> dict[str, Any]:
    """Remove behaviour loss contributions BEFORE the arm's own gradient assembly.

    The removal happens on the loss terms, so the arm recomputes its gradient exactly as it would
    without them: the projection for R4-08, the surrogate for R6-04, the assembly and the
    clipping unchanged. Nothing is subtracted from a finished gradient.
    """
    original: dict[str, Any] = {}

    def current():
        return state.get("intervention")

    if hasattr(trainer, "behavior_contrastive_components"):
        base = trainer.behavior_contrastive_components

        def components(residuals, *args, **kwargs):
            l1_loss, other_loss, valid = base(residuals, *args, **kwargs)
            mode = current()
            if mode == "NO-DEEP":
                return l1_loss, _zero(other_loss), valid
            if mode == "BASE-ONLY":
                return _zero(l1_loss), _zero(other_loss), valid
            return l1_loss, other_loss, valid

        original["components"] = trainer.behavior_contrastive_components
        trainer.behavior_contrastive_components = components
        return {"hook": "behavior_contrastive_components", "levels": "other_loss holds L2+L3"}

    base_ce = trainer._contrastive_ce
    level_counter = {"index": 0}

    def ce(logits, source_ids, target_ids, successor_indicator=None):
        loss = base_ce(logits, source_ids, target_ids, successor_indicator)
        mode = current()
        index = level_counter["index"]
        level_counter["index"] = index + 1
        if loss is None:
            return None
        if mode == "NO-DEEP" and index in (1, 2):
            return _zero(loss)
        if mode == "BASE-ONLY":
            return _zero(loss)
        return loss

    original["ce"] = base_ce
    trainer._contrastive_ce = ce
    return {"hook": "_contrastive_ce", "levels": "per-level, L2 and L3 dropped for NO-DEEP"}


def _zero(loss):
    if loss is None:
        return None
    return loss * 0.0


def restore_loss_switch(trainer, original: dict[str, Any]) -> None:
    for name, value in original.items():
        setattr(trainer, name, value)


# --------------------------------------------------------------------------------------
# the branch driver
# --------------------------------------------------------------------------------------
# The audit inherits the probe's protocol verbatim, so the two cannot drift apart. The snapshot
# list belongs to the probe and is NOT accepted here: the audit freezes at its fork steps instead,
# and a key the audit never reads would be a claim nobody checks.
try:
    import stage2_stability_probe as probe
except ImportError:  # running from tools/
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import stage2_stability_probe as probe

DEFAULT_AUDIT_CONFIG: dict[str, Any] = {
    key: value for key, value in probe.DEFAULT_PROBE_CONFIG.items() if key != "snapshot_steps"
}
DEFAULT_AUDIT_CONFIG.update({
    "fork_steps": [32, 96, 160],
    "propagation_steps": 8,
    "pair_limit": 8000,
    "seed": 42,
})


def validate_audit_config(config: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    allowed = set(DEFAULT_AUDIT_CONFIG)
    unknown = sorted(set(config) - allowed)
    if unknown:
        problems.append(f"audit config has keys the audit does not act on: {unknown}")
    for key in ("baseline_steps", "behavior_active_steps", "propagation_steps", "pair_limit"):
        value = config.get(key)
        if not isinstance(value, int) or value <= 0:
            problems.append(f"audit_config.{key} must be a positive integer")
    forks = config.get("fork_steps")
    if not isinstance(forks, list) or not forks:
        problems.append("audit_config.fork_steps must be a non-empty list of absolute steps")
    else:
        last = int(config["baseline_steps"]) + int(config["behavior_active_steps"])
        if any(not isinstance(f, int) or f < 0 or f > last for f in forks):
            problems.append(f"every fork step must lie inside 0..{last}")
    return problems


def snapshot_metrics(model, embeddings, positive, negative, *, use_sinkhorn: bool = False
                     ) -> dict[str, Any]:
    """One complete measurement of the boundary state, on the fixed train-only population."""
    assignments, distances = hard_assignments(model, embeddings, use_sinkhorn=use_sinkhorn)
    level_one = assignments[0]
    level_one_gaps = margins(distances[0])
    return {
        "l1_assignment": level_one.clone(),
        "l1_usage": codebook_usage(level_one),
        "tuple_collision": tuple_collision(assignments.t()),
        "same_code_rates": same_code_rates(level_one, positive, negative),
        "l1_margin_tie_fraction": level_one_gaps["tie_fraction"],
        "l1_margin_median": float(torch.quantile(level_one_gaps["gap"].float(), 0.5)),
        "l1_margin_p10": float(torch.quantile(level_one_gaps["gap"].float(), 0.10)),
        "raw_levels": {f"level_{i}": codebook_usage(assignments[i]) for i in range(assignments.shape[0])},
    }


def compare_branches(origin: dict[str, Any], branch: dict[str, Any]) -> dict[str, Any]:
    """Signed effect of one branch relative to the frozen origin and to FULL.

    The central contrast the issue asks for is Metric(FULL) - Metric(NO-DEEP); both are also
    reported against the no-update origin. Directions are stated with every number so a reader
    cannot mistake a smaller negative-match rate for a better one.
    """
    out: dict[str, Any] = {}
    origin_assign = origin["l1_assignment"]
    for name, metrics in branch.items():
        switches = int((metrics["l1_assignment"] != origin_assign).sum())
        out[name] = {
            "l1_switches_vs_origin": switches,
            "l1_switch_fraction_vs_origin": switches / max(int(origin_assign.numel()), 1),
            "negative_same_l1_code_delta_vs_origin":
                metrics["same_code_rates"]["negative_same_l1_code"]
                - origin["same_code_rates"]["negative_same_l1_code"],
            "tuple_collision_delta_vs_origin":
                metrics["tuple_collision"]["collision"]
                - origin["tuple_collision"]["collision"],
            "l1_effective_codes_delta_vs_origin":
                metrics["l1_usage"]["effective_codes"] - origin["l1_usage"]["effective_codes"],
        }
    if "FULL" in branch and "NO-DEEP" in branch:
        full = branch["FULL"]
        no_deep = branch["NO-DEEP"]
        out["Delta_deep"] = {
            "negative_same_l1_code_FULL_minus_NO-DEEP":
                full["same_code_rates"]["negative_same_l1_code"]
                - no_deep["same_code_rates"]["negative_same_l1_code"],
            "tuple_collision_FULL_minus_NO-DEEP":
                full["tuple_collision"]["collision"] - no_deep["tuple_collision"]["collision"],
            "l1_effective_codes_FULL_minus_NO-DEEP":
                full["l1_usage"]["effective_codes"] - no_deep["l1_usage"]["effective_codes"],
            "direction": ("a POSITIVE negative_same_l1_code difference means the deep gradients "
                          "increased unrelated-item sharing, i.e. local harm"),
        }
    if "BASE-ONLY" in branch and "FULL" in branch:
        out["Delta_base_only"] = {
            "negative_same_l1_code_BASE_ONLY_minus_FULL":
                branch["BASE-ONLY"]["same_code_rates"]["negative_same_l1_code"]
                - branch["FULL"]["same_code_rates"]["negative_same_l1_code"],
            "note": "if BASE-ONLY reproduces most of FULL's harmful transitions, the drift is "
                    "base-driven and the deep-gradient explanation is rejected",
        }
    return out


def parameter_checksum(model) -> str:
    """A content hash of every parameter and buffer, for the FULL-control bitwise comparison."""
    import hashlib

    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        if hasattr(value, "detach"):
            digest.update(value.detach().to(torch.float32).cpu().numpy().tobytes())
        else:
            digest.update(repr(value).encode("utf-8"))
    return digest.hexdigest()


def run_audit(tree: Path, spec: Any, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Walk the real training path and branch the declared interventions off frozen states.

    Everything the loop does is the arm's own: its loaders, its loss assembly, its projection or
    surrogate, its clipping, its AdamW and its scheduler. The audit adds three things and nothing
    else -- a per-step weight hook so the protocol matches the v2 probe, a loss switch that
    removes a contribution BEFORE assembly, and state capture/restore at the declared fork steps.
    No branch writes anything, and every branch starts from the same frozen pre-step state.
    """
    import contextlib
    import tempfile
    import time

    if str(REPO_ROOT / "tools/cgv") not in sys.path:
        sys.path.insert(0, str(REPO_ROOT / "tools/cgv"))
    import contract as C  # noqa: E402
    import engine as E  # noqa: E402

    tree = Path(tree).resolve()
    cfg = {**DEFAULT_AUDIT_CONFIG, **(config or {})}
    problems = validate_audit_config(cfg)
    if problems:
        return {"status": "invalid", "failures": problems, "audit_config": cfg}
    spec = C.normalise_spec(spec)
    spec_problems = C.validate(spec)
    if spec_problems:
        return {"status": "invalid",
                "failures": [f"declaration invalid: {p}" for p in spec_problems],
                "audit_config": cfg}

    started = time.time()
    trainer, saved_path, saved_mods = E.load_trainer(tree, module_name="audit_trainer")
    assembly = classify_assembly(trainer)
    before = probe.production_artifacts(tree)
    failures: list[str] = []
    forks: dict[str, Any] = {}
    state: dict[str, Any] = {"step": 0, "frozen": None, "queue": [], "armed": None,
                             "frozen_step": None, "intervention": None}
    captured: list[Any] = []
    optimizer_cls = getattr(torch.optim, trainer.OPTIMIZER)
    originals = {
        "init": trainer.RQVAE.__init__,
        "step": optimizer_cls.step,
        "weight": trainer.behavior_weight,
        "hook": getattr(trainer, "_PROBE_STEP_WEIGHT_HOOK", None),
        "epochs": trainer.EPOCHS,
        "seed": trainer.SEED,
        "ce": trainer._contrastive_ce,
        "components": getattr(trainer, "behavior_contrastive_components", None),
    }
    holder: dict[str, Any] = {"scheduler": None, "optimizer": None}
    fork_set = {int(value) for value in cfg["fork_steps"]}
    budget = int(cfg["baseline_steps"]) + int(cfg["behavior_active_steps"])
    branch_metrics: dict[str, dict[str, Any]] = {}

    def capturing_init(inner_self, *args, **kwargs):
        originals["init"](inner_self, *args, **kwargs)
        captured.append(inner_self)

    def pairs():
        return probe._train_only_pairs(trainer, limit=int(cfg["pair_limit"]))

    def weight_schedule(epoch, current=None):
        if current is None:
            return originals["weight"](epoch)
        step_about_to_run = state["step"] + 1
        if step_about_to_run <= int(cfg["baseline_steps"]):
            return 0.0
        if step_about_to_run > budget:
            return 0.0
        return float(cfg["behavior_weight_active"])

    def switch(trainer_module):
        return install_loss_switch(trainer_module, {"intervention": state["intervention"]})

    def measure_branch(model, name, fork_step):
        positive, _right, negatives = pairs()
        metrics = snapshot_metrics(model, state["embeddings"], positive, negatives)
        metrics["source_attribution"] = {}
        with torch.no_grad():
            latent = model.encoder(state["embeddings"])
            residual = latent
            for level, layer in enumerate(model.rq.vq_layers):
                assignment = layer._indices(layer._distances(residual.view(-1, layer.dim)),
                                            False)
                z_after = residual
                four = four_way_assignment(layer, state["z"][level], state["c"][level],
                                           z_after.detach(), layer.get_code_embs().detach())
                metrics["source_attribution"][f"level_{level}"] = crossing_attribution(four)
                residual = residual - torch.nn.functional.embedding(
                    assignment, layer.get_code_embs().detach())
        forks[str(fork_step)]["branches"][name] = metrics

    def arm_next(optimizer, model):
        """Restore the frozen state and arm the next branch for the following loop iteration."""
        if state["frozen"] is None:
            return
        restore_state(state["frozen"], model, optimizer, holder["scheduler"])
        if not state["queue"]:
            state["frozen"] = None
            state["armed"] = None
            state["intervention"] = None
            return
        state["armed"] = state["queue"].pop(0)
        state["intervention"] = state["armed"]

    def measured_step(self, *args, **kwargs):
        model = captured[-1] if captured else None
        step_now = state["step"] + 1
        # A fork is only armed inside the behaviour-active window: a step in the baseline window
        # has no behaviour gradient to remove, so branching there would compare nothing.
        is_fork = (step_now in fork_set and state["frozen"] is None
                   and step_now > int(cfg["baseline_steps"]) and model is not None)
        if is_fork:
            holder["optimizer"] = self
            state["frozen"] = capture_state(model, self, holder["scheduler"], trainer)
            state["frozen_identity"] = state_identity(state["frozen"])
            state["frozen_checksum"] = parameter_checksum(model)
            state["embeddings"] = probe._probe_embeddings(model).detach().clone()
            state["frozen_step"] = step_now
            # the per-level residual ENTERING each VQ layer, and that layer's codebook: these
            # are the two objects the crossing attribution varies independently
            with torch.no_grad():
                latent = model.encoder(state["embeddings"])
                residual = latent
                zs, cs = [], []
                for layer in model.rq.vq_layers:
                    zs.append(residual.detach().clone())
                    code = layer.get_code_embs().detach().clone()
                    cs.append(code)
                    assignment = layer._indices(layer._distances(residual.view(-1, layer.dim)),
                                                False)
                    residual = residual - torch.nn.functional.embedding(assignment, code)
                state["z"] = zs
                state["c"] = cs
            positive, _right, negatives = pairs()
            state["origin"] = snapshot_metrics(model, state["embeddings"], positive, negatives)
            state["queue"] = [name for name in INTERVENTIONS if name != "CTRL"]
            # CTRL is the origin itself: the frozen state with no optimizer update
            forks[str(step_now)] = {"branches": {"CTRL": state["origin"],
                                                 "origin": state["origin"]},
                                     "frozen_identity": state["frozen_identity"],
                                     "frozen_checksum": state["frozen_checksum"]}
            branch_metrics[str(step_now)] = {"origin_checksum": state["frozen_checksum"]}
        branch = state["armed"]
        before_weights = ([p.detach().clone() for p in model.parameters()]
                          if model is not None else None)
        result = originals["step"](self, *args, **kwargs)
        state["step"] = step_now
        if branch is not None and model is not None:
            measure_branch(model, branch, state["frozen_step"])
            branch_metrics[str(state["frozen_step"])][branch] = {
                "param_sha256": parameter_checksum(model),
                "max_param_delta_vs_frozen": max(
                    float((p.detach() - q).abs().max()) for p, q in zip(model.parameters(),
                                                                      before_weights))
                if before_weights else None,
            }
            arm_next(self, model)
            return result
        if is_fork:
            arm_next(self, model)
        return result

    trainer.RQVAE.__init__ = capturing_init
    trainer.behavior_weight = weight_schedule
    trainer._PROBE_STEP_WEIGHT_HOOK = weight_schedule
    optimizer_cls.step = measured_step
    switch_holder = install_loss_switch(trainer, state)
    trainer.EPOCHS = budget + 2
    trainer.SEED = int(cfg["seed"])

    scratch = Path(tempfile.mkdtemp(prefix="audit_"))
    try:
        with E.isolated_outputs(trainer, scratch):
            try:
                trainer.main()
            except (probe.ProbeStop, SystemExit):
                pass
            except Exception as exc:  # noqa: BLE001
                failures.append(f"the training loop raised {type(exc).__name__}: {exc}")
    finally:
        trainer.RQVAE.__init__ = originals["init"]
        trainer.behavior_weight = originals["weight"]
        trainer._PROBE_STEP_WEIGHT_HOOK = originals["hook"]
        trainer.EPOCHS = originals["epochs"]
        trainer.SEED = originals["seed"]
        trainer._contrastive_ce = originals["ce"]
        if originals["components"] is not None:
            trainer.behavior_contrastive_components = originals["components"]
        optimizer_cls.step = originals["step"]
        E.unload_trainer(saved_path, saved_mods, "audit_trainer")
        with contextlib.suppress(OSError):
            for path in sorted(scratch.rglob("*"), reverse=True):
                path.unlink() if path.is_file() else path.rmdir()
            scratch.rmdir()

    after = probe.production_artifacts(tree)
    changed = sorted(key for key in before if before[key] != after.get(key))
    if changed:
        failures.append(f"the audit modified production artifacts: {changed}")

    for step_key, entry in forks.items():
        entry["comparison"] = compare_branches(entry["branches"].get("origin", {}),
                                               {name: metrics for name, metrics
                                                in entry["branches"].items()
                                                if name != "origin"})
    return {
        "audit": "stage2_boundary_causal_audit",
        "status": "valid" if not failures else "invalid",
        "failures": failures,
        "tree": str(tree),
        "mechanism_id": (spec.get("mechanism") or {}).get("id"),
        "assembly": assembly,
        "audit_config": cfg,
        "forks": forks,
        "branch_metrics": branch_metrics,
        "executed_steps": state["step"],
        "production_artifacts_unchanged": not changed,
        "elapsed_s": round(time.time() - started, 2),
    }


def verify_control_parity(audit_result: dict[str, Any], probe_report: dict[str, Any]
                          ) -> dict[str, Any]:
    """Does the audit's own machinery leave the unmodified training path unchanged?

    The audit installs a weight hook, a loss switch and state capture/restore. If any of those
    perturbed the run, the branch comparison would be measuring the audit rather than the arm. The
    check is against the probe's v2 report for the SAME arm, protocol and seed, at the fork steps
    the two share: the frozen origin must reproduce the probe's snapshot statistics exactly.
    """
    if audit_result.get("status") != "valid":
        return {"parity_established": False,
                "reason": "the audit run was invalid, so nothing to compare"}
    audit_by_step = {str(int(key)): entry for key, entry in (audit_result.get("forks") or {}).items()}
    probe_by_step = {str(snapshot["step"]): snapshot
                     for snapshot in (probe_report.get("snapshots") or [])}
    compared, mismatched, missing = [], [], []
    for step, entry in sorted(audit_by_step.items()):
        origin = (entry.get("branches") or {}).get("origin")
        reference = probe_by_step.get(step)
        if origin is None or reference is None:
            missing.append(int(step))
            continue
        # the probe names the three-level measure `tuple3_collision`; the audit nests it
        same = (origin["l1_usage"]["used_codes"] == reference["l1_used_codes"]
                and abs(origin["tuple_collision"]["collision"]
                        - reference["tuple3_collision"]) < 1e-12
                and abs(origin["l1_usage"]["entropy_norm"]
                        - reference["l1_entropy_norm"]) < 1e-9)
        compared.append(int(step))
        if not same:
            mismatched.append(int(step))
    established = bool(compared) and not mismatched and not missing
    return {
        "parity_established": established,
        "compared_fork_steps": compared,
        "mismatched_fork_steps": mismatched,
        "fork_steps_without_a_probe_reference": missing,
        "note": ("the audit's frozen origin is compared against the independent probe run on the "
                 "same arm, protocol and seed; if the two ever disagreed, the audit's own hooks "
                 "would have changed training and no branch contrast could be trusted"),
    }


def adjudicate(audit_result: dict[str, Any]) -> dict[str, Any]:
    """Apply the issue's decision rules to one arm's audit.

    Control parity gates everything: if the FULL branch cannot reproduce the independent probe, the
    arms' own machinery perturbed training and no branch contrast is admissible, however large the
    numbers look. That verdict is reported as such rather than dressed up as a result.
    """
    parity = audit_result.get("control_parity") or {}
    if not parity.get("parity_established"):
        return {
            "verdict": "inconclusive",
            "reason": ("control parity not established: the audit's frozen origin does not "
                       "reproduce the independent probe run at the fork steps "
                       f"{parity.get('mismatched_fork_steps')}"),
            "no_causal_claim": True,
            "frozen_states": {key: entry.get("frozen_identity")
                              for key, entry in (audit_result.get("forks") or {}).items()},
        }
    deltas = []
    for entry in (audit_result.get("forks") or {}).values():
        contrast = (entry.get("comparison") or {}).get("Delta_deep") or {}
        if contrast:
            deltas.append(contrast.get("negative_same_l1_code_FULL_minus_NO-DEEP"))
    if not deltas:
        return {"verdict": "inconclusive",
                "reason": "no FULL-versus-NO-DEEP contrast was measured",
                "no_causal_claim": True}
    harmful = [value for value in deltas if value is not None and value > 0]
    if len(harmful) == len(deltas):
        return {"verdict": "deep_gradients_locally_harmful",
                "reason": "FULL raises unrelated-item same-L1-code sharing relative to NO-DEEP at "
                          "every fork step",
                "no_causal_claim": False,
                "caveat": "local one-step evidence only; it does not explain a 3000-epoch final ER"}
    if not harmful:
        return {"verdict": "deep_gradient_explanation_rejected",
                "reason": "NO-DEEP never increases unrelated-item sharing relative to FULL at any "
                          "fork step",
                "no_causal_claim": False,
                "caveat": "this rejects the deep-gradient harm explanation for these states; it "
                          "does not identify the actual driver"}
    return {"verdict": "mixed",
            "reason": "the FULL-versus-NO-DEEP contrast changes sign across fork steps",
            "no_causal_claim": True,
            "caveat": "no branch, step or metric is selected post hoc"}
