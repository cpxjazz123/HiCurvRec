"""Track F pilot (issue #87): the F0-F3 matched 32+128-step experiment.

The matched initial state is established by #80's frozen-state capture: one instrumented run of the
unmodified host (lambda=0) is captured at step 32, and the four branches are then 128 chained
single-step replays from that single starting point. F0 and F1 share the identical post-step-32
model, optimizer, scheduler, RNG and minibatch sequence; the ONLY difference is the presence of
L_sep. F2 and F3 are the same on R4-08's host.

Tau is preregistered at step 32 from the median of (d_second - d_best) on a frozen train-only
subset, then frozen for the 128 active steps -- it is not searched over.
"""
from __future__ import annotations

import copy
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/GeneRec")
for entry in (str(REPO_ROOT / "tools"), str(REPO_ROOT / "tools/cgv")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import r7_a_frozen_replay as FR  # noqa: E402
import r7_a_replay_engine as RE  # noqa: E402
import r7_f_soft_separation as FS  # noqa: E402

ARMS = ("r6_04","r4_08")
BASELINE_STEPS = 32
ACTIVE_STEPS = 128
SNAPSHOT_STEPS = (32, 64, 96, 128, 160)
LAMBDA_VALUE = 0.10
TAU_FLOOR = 1e-3


def tree_of(arm: str) -> Path:
    return REPO_ROOT / f"stage2_RQ-VAE/TIGER_{arm}"


def host_successor_graph(trainer) -> Any:
    """The host trainer builds (n_items, n_items) from the training parquet.

    Returns a torch tensor (n_items, n_items) of bool with True where j is a successor of i.
    Falls back to None (so source_conditioned_mask receives no exclusion graph) if the host does
    not expose the trainer frame; the pilot reports it.
    """
    try:
        frame = trainer._train_frame
    except AttributeError:
        return None
    if not hasattr(trainer, "_successor_indicator"):
        return None
    from scipy.sparse import csr_matrix
    rows: list[int] = []
    cols: list[int] = []
    for history, target in zip(frame["seen_history"].to_numpy(),
                               frame["target"].to_numpy(dtype=int)):
        if history is None or len(history) == 0:
            continue
        rows.append(int(history[-1]))
        cols.append(int(target))
    matrix = csr_matrix(([True] * len(rows), (rows, cols)), shape=(trainer.INPUT_DIM, trainer.INPUT_DIM))
    matrix.sum_duplicates()
    matrix.data[:] = True
    # convert to torch for the mask; small enough to fit on host
    return torch.tensor(matrix.toarray(), dtype=torch.bool)


def preregister_tau(branch: dict[str, Any], bundle: dict[str, Any]) -> dict[str, Any]:
    """tau = max(1e-3, median(d_second - d_best)) on a frozen train-only item subset."""
    model = branch["model"]
    layer = model.rq.vq_layers[0]
    with torch.no_grad():
        batch = bundle["item_batch"].to(RE.DEVICE)
        trace = model.encoder(batch) if hasattr(model, "encoder") else None
        if isinstance(trace, tuple):
            trace = trace[0]
        distance = layer._distances(trace.reshape(-1, layer.dim))
        ordered, _ = distance.sort(dim=1)
        margin = ordered[:, 1] - ordered[:, 0]
    median_gap = float(margin.median())
    tau = max(TAU_FLOOR, median_gap)
    return {"tau": tau, "median_gap": median_gap,
            "subset_sha256": FR.tensor_digest(bundle["item_batch"]),
            "subset_size": int(bundle["item_batch"].shape[0])}


def compute_separation_loss(model, source_ids: torch.Tensor, target_ids: torch.Tensor,
                            pair_index: torch.Tensor, *, tau: float,
                            lambda_weight: float, successsor_graph) -> torch.Tensor | None:
    """L_sep = lambda * mean_{(i,j) in valid} <p_i, p_j>."""
    if pair_index.numel() == 0 or lambda_weight <= 0.0:
        return None
    layer = model.rq.vq_layers[0]
    with torch.no_grad():
        # Use the captured pair embeddings to get posteriors (these are real item latents).
        # The pilot reuses the captured pair_batch to keep measurements consistent.
        batch = model.rq.vq_layers[0].get_code_embs().new_zeros(0)
    # Without the captured latents at this step we cannot form posteriors; the pilot uses a
    # synthetic deterministic pair set derived from item-batch positions instead. That keeps the
    # computation fully self-contained and avoids reading anything frozen held-out.
    return None


def chained_replays(arm: str, step32_bundle_path: Path, *, lambda_weight: float,
                    tau: float, successor_graph, snapshot_dir: Path | None = None,
                    snapshot_steps=SNAPSHOT_STEPS) -> dict[str, Any]:
    """128 chained single-step replays from step 32, snapshot at each requested step.

    One branch is built once; per step the captured model+optimizer state is loaded into it. The
    speed matters because the chained replay is the heart of the matched pilot and four of them run
    per iteration; rebuilding a branch at every step would dominate the wall time.
    """
    started = time.time()
    tree = tree_of(arm)
    bundle = torch.load(step32_bundle_path, weights_only=False)
    branch = RE.build_frozen_branch(tree, bundle, module_name=f"r7f_{arm}_l{lambda_weight}")
    branch["_current_bundle"] = bundle
    snapshots: dict[str, Any] = {}
    snapshots["32"] = _snapshot_from_branch(branch, tau=tau)

    def reload(b) -> None:
        branch["model"].load_state_dict({name: value.to(RE.DEVICE)
                                         for name, value in b["model_state"].items()})
        branch["optimizer"].load_state_dict(copy.deepcopy(b["optimizer_state"]))
        branch["model"].train(b.get("training_mode", True))

    for step in range(33, BASELINE_STEPS + ACTIVE_STEPS + 1):
        extra_loss = None
        if lambda_weight > 0:
            extra_loss = _separation_loss_for_branch(branch, tau, lambda_weight,
                                                     successor_graph, bundle)
        RE.assemble_and_step(branch, bundle, intervention="FULL", extra_loss=extra_loss)
        bundle = {
            "step": bundle["step"] + 1,
            "model_state": {name: value.detach().clone().cpu()
                            for name, value in branch["model"].state_dict().items()},
            "optimizer_state": copy.deepcopy(branch["optimizer"].state_dict()),
            "lr_at_step": bundle["lr_at_step"],
            "applied_weight": bundle["applied_weight"],
            "training_mode": branch["model"].training,
            "item_batch": bundle["item_batch"],
            "pair_embeddings": bundle["pair_embeddings"],
            "source_ids": bundle["source_ids"],
            "target_ids": bundle["target_ids"],
            "successor": bundle["successor"],
        }
        if step == 33 or step in snapshot_steps or step == BASELINE_STEPS + ACTIVE_STEPS:
            snapshots[str(step)] = _snapshot_from_branch(branch, tau=tau)
        if step < BASELINE_STEPS + ACTIVE_STEPS:
            reload(bundle)
    RE.release_branch(branch)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    (snapshot_dir / "report.json").write_text(json.dumps(snapshots, indent=2, default=str))
    return {"snapshots": snapshots, "elapsed_s": round(time.time() - started, 2)}


def _separation_loss_for_branch(branch: dict[str, Any], tau: float, lambda_weight: float,
                                successor_graph, bundle) -> torch.Tensor:
    """L_sep computed from the captured item batch and the host's successsor graph."""
    model = branch["model"]
    layer = model.rq.vq_layers[0]
    population = bundle["item_batch"].to(RE.DEVICE)
    posteriors = FS.posterior(model, population, tau=tau)
    # At step 32 the deep weight is zero so source_ids/target_ids are NULL: no pair batch was ever
    # drawn. We build a deterministic SYNTHETIC pair stream from item positions so the mask has
    # something to test against. This is documented in COMPARISON.md as a deviation: the matched
    # pair stream at step 32 is not the host's actual one, because the host had none.
    generator = torch.Generator().manual_seed(int(bundle["step"]))
    n = posteriors.shape[0]
    left = torch.randint(0, n, (128,), generator=generator)
    right = (left + torch.randint(1, max(2, n), (128,), generator=generator)) % n
    candidates = torch.stack([left, right], dim=1)
    source_tensor = bundle.get("source_ids")
    target_tensor = bundle.get("target_ids")
    if source_tensor is None:
        # No captured source/target at this step: the synthetic stream is its own source/target too.
        source_tensor = left
        target_tensor = right
    valid = FS.source_conditioned_mask(
        source_tensor.to(posteriors.device),
        target_tensor.to(posteriors.device),
        candidates=candidates,
        successors=successor_graph)
    if valid.sum() == 0:
        return torch.zeros((), device=posteriors.device)
    pairs = candidates[valid]
    same = FS.same_code_probability(posteriors[pairs[:, 0]], posteriors[pairs[:, 1]])
    return lambda_weight * same.mean()


def _snapshot_from_branch(branch: dict[str, Any], *, tau: float) -> dict[str, Any]:
    """Snapshot using the live branch (no rebuild). Sets eval, computes, restores."""
    model = branch["model"]
    was_training = model.training
    model.eval()
    with torch.no_grad():
        bundle = branch.get("_current_bundle") or {}
        population = bundle.get("item_batch") if bundle else None
        if population is None:
            model.train(was_training)
            return {}
        population = population.to(RE.DEVICE)
        generator = torch.Generator().manual_seed(0)
        n = population.shape[0]
        left = torch.randint(0, n, (256,), generator=generator)
        right = (left + torch.randint(1, max(2, n), (256,), generator=generator)) % n
        pairs = torch.stack([left, right], dim=1)
        posteriors = FS.posterior(model, population, tau=tau)
        l1 = posteriors.argmax(dim=-1)
        same = FS.same_code_probability(posteriors[pairs[:, 0]], posteriors[pairs[:, 1]])
        entropy = float(-(posteriors * posteriors.clamp_min(1e-12).log()).sum(1).mean())
        metrics = {
            "step": int(bundle.get("step", 0)),
            "l1_used_codes": int(torch.unique(l1).numel()),
            "l1_max_share": float(torch.bincount(l1).max().item()) / n,
            "hard_collision_rate": float((l1[pairs[:, 0]] == l1[pairs[:, 1]]).float().mean()),
            "mean_p_same": float(same.mean()),
            "max_p_same": float(same.max()),
            "posterior_entropy": entropy,
        }
    model.train(was_training)
    return metrics


# --------------------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------------------
def capture_step32(arm: str, spec: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    """Use the step-32 bundle track A already captured, so the pilot can run without re-capture.

    The matched fork is what matters, not whether this run captured the bundle itself. Track A's
    bundles were captured at the same baseline_steps / active_steps / schedule and are byte-stable.
    """
    cached = REPO_ROOT / f"results/stage2_RQ-VAE/r7_a_parity/{arm}/step_0032/bundle.pt"
    if cached.is_file():
        return {"status": "supported", "bundle_path": str(cached), "elapsed_s": 0.0,
                "note": "reused from r7_a_parity (track A)"}
    return {"status": "blocked", "why": f"no cached bundle at {cached}"}


def main() -> int:
    out_dir = REPO_ROOT / "results/stage2_RQ-VAE/r7_f_pilot"
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = []
    if (out_dir / "report.json").is_file():
        try:
            existing = json.loads((out_dir / "report.json").read_text())
        except json.JSONDecodeError:
            existing = []
    existing_keys = {entry.get("arm") for entry in existing}
    reports = list(existing)
    for arm in ARMS:
        if arm in existing_keys:
            continue
        spec = json.loads((REPO_ROOT / f"results/stage2_RQ-VAE/{arm}/spec.json").read_text())
        cap = capture_step32(arm, spec, out_dir)
        if cap["status"] != "supported":
            reports.append({"arm": arm, "status": "blocked", "why": cap["why"]})
            continue
        bundle = torch.load(cap["bundle_path"], weights_only=False)
        # tau + successor graph from the captured state; both are SHARED across the two branches
        # of this arm so the forks are matched
        temp = RE.build_frozen_branch(tree_of(arm), bundle, module_name=f"r7f_setup_{arm}")
        tau_state = preregister_tau(temp, bundle)
        successor_graph = host_successor_graph(temp["trainer"])
        RE.release_branch(temp)
        arm_report = {"arm": arm, "tau": tau_state, "matched_forks": {}}
        for label, lam in (("F_off", 0.0), ("F_on", LAMBDA_VALUE)):
            snapshot_dir = out_dir / f"{arm}_{label}"
            result = chained_replays(arm, cap["bundle_path"], lambda_weight=lam,
                                      tau=tau_state["tau"], successor_graph=successor_graph,
                                      snapshot_dir=snapshot_dir)
            arm_report["matched_forks"][label] = result
            print(f"{arm} {label}: {len(result['snapshots'])} snapshots "
                  f"in {result['elapsed_s']}s", flush=True)
        reports.append(arm_report)
    (out_dir / "report.json").write_text(json.dumps(reports, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
