"""Track C (issue #82): what does L1 geometry actually do, measured without touching the run.

Every number here comes from a snapshot copied out of a live run and then measured on its own, in
its own process, against the fixed train-item population. Nothing is measured inside the training
loop, because measuring inside it is exactly what made issue #79's contrasts unusable.

The read-only claim is not assumed. ``non_perturbation_control`` runs the identical schedule twice --
once instrumented, once not -- and compares the final parameters. If the instrumented run lands
somewhere else, every geometry number below is measuring the instrumentation instead of the arm.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/GeneRec")
for entry in (str(REPO_ROOT / "tools"), str(REPO_ROOT / "tools/cgv")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import r7_a_frozen_replay as R  # noqa: E402
import r7_a_replay_engine as G  # noqa: E402

SNAPSHOT_STEPS = (0, 8, 16, 32, 64, 96, 128, 160)
BASELINE_STEPS = 32
ACTIVE_STEPS = 128
ACTIVE_WEIGHT = 0.20
SEED = 42


# --------------------------------------------------------------------------------------
# the read-only guarantee
# --------------------------------------------------------------------------------------
def non_perturbation_control(tree: Path, spec: dict[str, Any], *, module_suffix: str) -> dict[str, Any]:
    """Same schedule twice, one instrumented and one not; the final parameters must agree."""
    instrumented = R.capture_from_run(tree, spec, fork_steps=(BASELINE_STEPS + ACTIVE_STEPS,),
                                      baseline_steps=BASELINE_STEPS,
                                      active_steps=ACTIVE_STEPS)
    plain = R.capture_from_run(tree, spec, fork_steps=(), baseline_steps=BASELINE_STEPS,
                               active_steps=ACTIVE_STEPS)
    left = instrumented["capture"]["bundles"].get(str(BASELINE_STEPS + ACTIVE_STEPS),
                                                 {}).get("parameter_sha256_after")
    right = plain["capture"].get("uninstrumented_parameter_sha256")
    return {
        "definition": "an instrumented run and an uninstrumented run of the same schedule must "
                      "reach the same final parameters",
        "instrumented_sha256": left,
        "uninstrumented_sha256": right,
        "identical": bool(left and right and left == right),
        "note": "step 0 is outside the instrumented capture set, so the uninstrumented digest is "
                "taken by an identical run with no hooks installed",
        "module_suffix": module_suffix,
    }


# --------------------------------------------------------------------------------------
# geometry, measured on a frozen snapshot
# --------------------------------------------------------------------------------------
def l1_geometry(model, embeddings: torch.Tensor, *, chunk: int = 8192) -> dict[str, Any]:
    """Per-item L1 distances to the winning and runner-up code, in the quantizer's own metric.

    Ties are counted, not smoothed: a Voronoi boundary with exactly equal distances is a different
    situation from one with a 1e-9 gap, and averaging the two hides exactly the fragility this
    lane is looking for.
    """
    was_training = model.training
    model.eval()
    layer = model.rq.vq_layers[0]
    device = embeddings.device
    all_margin = []
    all_winner = []
    all_second = []
    all_tied = []
    with torch.no_grad():
        for start in range(0, embeddings.shape[0], chunk):
            block = embeddings[start:start + chunk].to(device)
            trace = model.encoder(block)
            if isinstance(trace, tuple):
                trace = trace[0]
            distance = layer._distances(trace.reshape(-1, layer.dim))
            ordered, index = distance.sort(dim=1)
            winner = ordered[:, 0]
            runner_up = ordered[:, 1]
            margin = runner_up - winner
            tied = margin <= 0.0
            all_margin.append(margin.double().cpu())
            all_winner.append(winner.cpu())
            all_second.append(runner_up.cpu())
            all_tied.append(tied.cpu())
    model.train(was_training)
    margin = torch.cat(all_margin)
    winner = torch.cat(all_winner)
    runner_up = torch.cat(all_second)
    tied = torch.cat(all_tied)
    return {
        "items": int(margin.numel()),
        "margin_mean": float(margin.mean()),
        "margin_p05": float(margin.quantile(0.05)),
        "margin_p50": float(margin.quantile(0.50)),
        "margin_median": float(margin.median()),
        "exact_ties": int(tied.sum()),
        "exact_tie_rate": float(tied.float().mean()),
        "margin_below_1e_3": int((margin < 1e-3).sum()),
        "indices": winner.to(torch.long),
        "winner_distance_mean": float(torch.cat(all_winner).mean()),
        "runner_up_distance_mean": float(torch.cat(all_second).mean()),
    }


def occupancy(codes: torch.Tensor, size: int) -> dict[str, Any]:
    counts = torch.bincount(codes, minlength=size).to(torch.float64)
    total = counts.sum()
    if total <= 0:
        return {"used_codes": 0, "gini": 0.0, "entropy": 0.0, "effective_codes": 0.0}
    share = counts / total
    nonzero = share[share > 0]
    entropy = float(-(nonzero * nonzero.log()).sum())
    ordered = torch.sort(counts, descending=True).values
    index = torch.arange(1, ordered.numel() + 1, dtype=torch.float64)
    gini = float((2.0 * (index * ordered).sum()) / (ordered.sum() * ordered.numel())
                 - (ordered.numel() + 1.0) / ordered.numel())
    return {
        "used_codes": int((counts > 0).sum()),
        "codebook_size": int(size),
        "gini": gini,
        "entropy": entropy,
        "effective_codes": float(math.exp(entropy)),
        "max_share": float(share.max()),
    }


def switches(previous: torch.Tensor, current: torch.Tensor) -> dict[str, Any]:
    changed = previous != current
    total = int(changed.numel())
    return {
        "items": total,
        "switched": int(changed.sum()),
        "switch_rate": float(changed.float().mean()) if total else 0.0,
    }


def switch_back_rate(first: torch.Tensor, middle: torch.Tensor, last: torch.Tensor) -> dict[str, Any]:
    """How many items that moved away from their original code came straight back."""
    left = first != middle
    returned = left & (first == last)
    switched = max(1, int(left.sum()))
    return {
        "items_that_left": int(left.sum()),
        "returned_to_original": int(returned.sum()),
        "switch_back_rate": float(returned.sum()) / switched,
    }


# --------------------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------------------
ARMS = ("r6_02", "r6_03", "r6_04", "r4_08")
OUTPUT_ROOT = REPO_ROOT / "results/stage2_RQ-VAE/r7_c_geometry"


def run_arm(arm: str) -> dict:
    """Snapshot the fixed train-item population at each declared step and measure it offline."""
    import json
    import time

    import r7_a_frozen_replay as RC

    started = time.time()
    tree = REPO_ROOT / f"stage2_RQ-VAE/TIGER_{arm}"
    spec_path = REPO_ROOT / f"results/stage2_RQ-VAE/{arm}/spec.json"
    spec = json.loads(spec_path.read_text()) if spec_path.is_file() else {}
    out = OUTPUT_ROOT / arm
    capture = RC.capture_from_run(tree, spec, fork_steps=SNAPSHOT_STEPS, save_root=out)["capture"]
    report: dict = {"arm": arm, "kind": "observational_read_only",
                    "snapshot_steps": list(SNAPSHOT_STEPS), "snapshots": {}, "blockers": []}
    report["capture_failures"] = capture.get("failures", [])

    embeddings = None
    for step in SNAPSHOT_STEPS:
        path = out / f"step_{step:04d}" / "bundle.pt"
        if not path.is_file():
            report["blockers"].append(f"no snapshot at step {step}")
            continue
        bundle = torch.load(path, weights_only=False)
        branch = G.build_frozen_branch(tree, bundle, module_name=f"r7c_{arm}_{step}")
        population = bundle["item_batch"]
        geometry = l1_geometry(branch["model"], population.to(G.DEVICE))
        indices = geometry.pop("indices")
        codebook_size = branch["model"].rq.vq_layers[0].n_embed
        report["snapshots"][str(step)] = {
            "margin": geometry,
            "occupancy": occupancy(indices, codebook_size),
            "deep_active": float(bundle["applied_weight"]) > 0.0,
        }
        embeddings = population
        G.release_branch(branch)

    ordered = sorted(report["snapshots"], key=int)
    for previous, current in zip(ordered, ordered[1:]):
        left = torch.load(out / f"step_{previous:04d}" / "bundle.pt", weights_only=False)
        right = torch.load(out / f"step_{current:04d}" / "bundle.pt", weights_only=False)
        before = G.hard_assignments(G.build_frozen_branch(
            tree, left, module_name=f"r7c_{arm}_{previous}_cmp")["model"],
            left["item_batch"].to(G.DEVICE))
        after_branch = G.build_frozen_branch(tree, right, module_name=f"r7c_{arm}_{current}_cmp")
        after = G.hard_assignments(after_branch["model"], right["item_batch"].to(G.DEVICE))
        report["snapshots"][current]["switches_from_previous"] = switches(before["indices"],
                                                                        after["indices"])
        G.release_branch(after_branch)
    report["status"] = "supported" if not report["blockers"] else "blocked"
    report["elapsed_s"] = round(time.time() - started, 2)
    return report


def main() -> int:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    reports = []
    for arm in ARMS:
        print(f"=== {arm} ===", flush=True)
        report = run_arm(arm)
        reports.append(report)
        (OUTPUT_ROOT / arm / "report.json").write_text(json.dumps(report, indent=2, default=str))
        print(f"{arm}: {report['status']}", flush=True)
    (OUTPUT_ROOT / "report.json").write_text(json.dumps({"arms": reports}, indent=2, default=str))
    return 0 if all(report["status"] == "supported" for report in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
