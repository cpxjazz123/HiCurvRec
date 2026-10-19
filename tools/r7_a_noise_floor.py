"""How large is run-to-run noise in the metrics #78 and #79 actually reported?

Track A's FULL replay is bit-exact against the live loop, which is a WITHIN-run statement: same
process, same tensors, same cuBLAS algorithm. It says nothing about two separate runs of the same
code, which is the comparison #79 made when its audited origin failed to match an independent
probe.

Two runs of the unmodified trainer with no hooks reach different final parameters even with
CUBLAS_WORKSPACE_CONFIG pinned, so the environment is not bit-reproducible. This script measures
what that costs in the units the earlier issues reported -- L1 used codes, three-level tuple
collision, L1 entropy -- so a reader can tell a real mechanism effect from run-to-run spread.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/GeneRec")
for entry in (str(REPO_ROOT / "tools"), str(REPO_ROOT / "tools/cgv")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import r7_a_frozen_replay as R  # noqa: E402
import r7_a_replay_engine as G  # noqa: E402

FORK_STEPS = (96, 160)


def summarise(model, embeddings: torch.Tensor) -> dict[str, float]:
    """The #78-style metrics, computed on one frozen snapshot."""
    was_training = model.training
    model.eval()
    layer = model.rq.vq_layers[0]
    with torch.no_grad():
        trace = model.encoder(embeddings)
        if isinstance(trace, tuple):
            trace = trace[0]
        rows = []
        for level, vq in enumerate(model.rq.vq_layers):
            distance = vq._distances(trace.reshape(-1, vq.dim))
            rows.append(distance.argmin(dim=-1))
            trace = trace.reshape(-1, vq.dim) - vq._quantize_from_indices(
                distance.argmin(dim=-1)) if hasattr(vq, "_quantize_from_indices") else trace
        codes = torch.stack(rows, dim=1)
    model.train(was_training)
    used = [int(torch.unique(codes[:, level]).numel()) for level in range(codes.shape[1])]
    tuple_collision = 1.0 - float(torch.unique(codes, dim=0).shape[0]) / codes.shape[0]
    counts = torch.bincount(codes[:, 0], minlength=layer.n_embed).to(torch.float64)
    share = counts[counts > 0] / counts.sum()
    entropy = float(-(share * share.log()).sum())
    return {
        "l1_used_codes": used[0],
        "l2_used_codes": used[1] if len(used) > 1 else None,
        "l3_used_codes": used[2] if len(used) > 2 else None,
        "raw_tuple3_collision": tuple_collision,
        "l1_entropy_norm": entropy / float(torch.log(torch.tensor(float(layer.n_embed)))),
    }


def run(tree: Path, spec: dict, tag: str, out: Path) -> dict:
    capture = R.capture_from_run(tree, spec, fork_steps=FORK_STEPS, save_root=out / tag)["capture"]
    branch_metrics = {}
    for step in FORK_STEPS:
        path = out / tag / f"step_{step:04d}" / "bundle.pt"
        if not path.is_file():
            continue
        bundle = torch.load(path, weights_only=False)
        branch = G.build_frozen_branch(tree, bundle, module_name=f"r7n_{tag}_{step}")
        branch_metrics[str(step)] = summarise(branch["model"],
                                              bundle["item_batch"].to(G.DEVICE))
        G.release_branch(branch)
    return {"tag": tag, "steps": branch_metrics,
            "final_parameter_sha256": capture.get("final_parameter_sha256")}


def main() -> int:
    tree = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO_ROOT / "stage2_RQ-VAE/TIGER_r6_02"
    spec_name = tree.name.replace("TIGER_", "")
    spec_path = REPO_ROOT / f"results/stage2_RQ-VAE/{spec_name}/spec.json"
    spec = json.loads(spec_path.read_text()) if spec_path.is_file() else {}
    out = REPO_ROOT / "results/stage2_RQ-VAE/r7_a_parity/noise_floor"
    runs = [run(tree, spec, tag, out) for tag in ("run_1", "run_2")]
    rows = []
    for step in FORK_STEPS:
        key = str(step)
        left, right = runs[0]["steps"].get(key, {}), runs[1]["steps"].get(key, {})
        if not left or not right:
            continue
        rows.append({
            "step": step,
            "run_1": left, "run_2": right,
            "l1_used_codes_delta": left["l1_used_codes"] - right["l1_used_codes"],
            "tuple3_collision_delta": (left["raw_tuple3_collision"]
                                       - right["raw_tuple3_collision"]),
            "l1_entropy_delta": left["l1_entropy_norm"] - right["l1_entropy_norm"],
        })
    summary = {
        "definition": "two runs of the UNMODIFIED trainer, no hooks, identical seed and protocol; "
                      "the spread below is what a cross-run comparison can attribute to the code",
        "arm": spec_name,
        "cublas_workspace_config": "pinned by the caller when set",
        "runs_identical": (runs[0]["final_parameter_sha256"]
                           == runs[1]["final_parameter_sha256"]),
        "run_1_sha256": runs[0]["final_parameter_sha256"],
        "run_2_sha256": runs[1]["final_parameter_sha256"],
        "per_step": rows,
    }
    (out / "report.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary["per_step"], indent=2, default=str))
    print("runs_identical:", summary["runs_identical"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
