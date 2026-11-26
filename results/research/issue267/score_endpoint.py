"""Issue #267 pre-Stage3 score prediction on the new candidate.

Read-only: final Stage2 checkpoint forward/autograd, no optimizer, no update, no
SID rewrite. Must run after Stage2 and before Stage3/reading new Recall.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/research/issue267"
ARM = "TIGER_relative_curvature_ranking_rq"
RUNS = sorted((ROOT / "results/stage2_RQ-VAE/_runs").glob(f"*/{ARM}/result/rqvae_best.pth"),
              key=lambda path: path.stat().st_mtime)
if not RUNS:
    raise SystemExit("no completed Stage2 checkpoint for " + ARM)
CHECKPOINT = RUNS[-1]
RESULT = CHECKPOINT.parent
SID = RESULT / "item_sids.json"
TREE = RESULT.parent / "src"
EMBEDDING = ROOT / "stage1_GeneEmbedding/output/sentence_t5.npy"
TRAIN = ROOT / "results/stage0_build_parquet/train.parquet"
EXPECTED = 24587
BATCH_SHA = "251dced08f5c42703e2ce66b3c56924e91d14200a15a3771f45c8d8c66f1c457"
R_RANGE = (0.00008205277845263481, 0.00010617653606459498)
G_RANGE = (0.0003398736207435491, 0.05941492674715333)
L_RANGE = (0.00004965698388494281, 0.00015923990773577049)
WEIGHTS = (0.52137, 0.19389, 0.28474)
HISTORICAL = [
    (101, "TIGER_three_expert_attention_rq", 0.046431866850049616, 0.2281179533),
    (104, "TIGER_role_conditioned_l1_rq", 0.0490085133794112, 0.4475563464),
    (100, "TIGER_gnn_attention_sectional_rq", 0.0492870697609638, 0.5689364228),
    (107, "TIGER_topology_residual_l2_rq", 0.049600445690210486, 0.5782102943),
    (106, "TIGER_role_conditioned_l2_rq", 0.05034906596563311, 0.5789591370),
    (105, "TIGER_role_conditioned_l3_rq", 0.05151552081338463, 0.5797449191),
    (108, "TIGER_frozen_topology_gnn_rq", 0.05189853583801946, 0.6454995374),
    (109, "TIGER_euclid_local_curve_supervision_rq", 0.052594926791900976, 0.6535990918),
    (103, "TIGER_periodic_budget_prefix_rq", 0.05266456588728913, 0.6698993639),
    (102, "TIGER_module_budget_prefix_rq", 0.05273420498267727, 0.7411182932),
]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def norm(value: float, lo: float, hi: float) -> float:
    if not (math.isfinite(value) and math.isfinite(lo) and math.isfinite(hi) and hi > lo):
        raise ValueError("invalid normalization")
    return (value - lo) / (hi - lo)


def norm_grad(grads) -> float:
    return float(math.sqrt(sum(float((g.double() ** 2).sum()) for g in grads)))


def main() -> None:
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(CHECKPOINT)
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    if payload.get("epoch") != 3000:
        raise ValueError("not a final 3000-epoch checkpoint")
    for name in list(sys.modules):
        if name.split(".")[0] in ("model", "losses", "distances", "train_config", "train_rqvae"):
            del sys.modules[name]
    sys.path.insert(0, str(TREE))
    training = importlib.import_module("train_rqvae")
    if Path(training.__file__).resolve() != TREE / "train_rqvae.py":
        raise ValueError("candidate import did not use the candidate tree")
    model = training.RQVAE(training._tokenizer_config(), in_dim=768,
                           distance=training.DISTANCE).cpu()
    model.load_state_dict(payload["state_dict"], strict=True)
    # `_relative_alpha` is a pre-training scalar, not a state_dict buffer; restore
    # it from the immutable Stage2 calibration record so the native L_struct is
    # measured with the actual trained objective.
    relative_alpha = None
    metrics_path = RESULT / "logs/training_metrics.jsonl"
    if metrics_path.is_file():
        for line in metrics_path.read_text().splitlines():
            event = json.loads(line)
            if event.get("event") == "relative_calibration":
                relative_alpha = float(event["alpha"])
    if relative_alpha is None or not math.isfinite(relative_alpha) or relative_alpha <= 0:
        raise RuntimeError("relative alpha not found in Stage2 calibration")
    model._relative_alpha = relative_alpha
    model.eval()
    embeddings = np.load(EMBEDDING, allow_pickle=False)
    frame = pd.read_parquet(TRAIN, columns=["target"])
    train_ids = np.unique(frame["target"].to_numpy(dtype=np.int64))
    batch_ids = np.sort(np.random.default_rng(42).choice(train_ids, size=4096, replace=False))
    batch_sha = hashlib.sha256(batch_ids.astype(np.int64).tobytes()).hexdigest()
    if batch_sha != BATCH_SHA:
        raise ValueError("fixed batch identity mismatch")
    batch = torch.from_numpy(np.array(embeddings[batch_ids], copy=True))
    catalogue = torch.from_numpy(np.array(embeddings, copy=True))
    sample = model.structural_sampler().draw(0, 0, 1, track=False)
    before = hashlib.sha256()
    for _, value in sorted(model.state_dict().items()):
        before.update(value.detach().cpu().numpy().tobytes())
    model.train()
    for layer in model.rq.vq_layers:
        layer._skip_ddp_reduce = True
    try:
        model.zero_grad(set_to_none=True)
        reconstructed, parts, _, _, curvature = model(
            batch, torch.from_numpy(batch_ids), catalogue_embeddings=catalogue,
            structural_sample=sample)
        original, components = training.CRITERION(reconstructed, batch, parts)
        parameters = tuple(p for p in model.parameters() if p.requires_grad)
        g_original = torch.autograd.grad(original, parameters, retain_graph=True,
                                         allow_unused=True)
        g_structural = torch.autograd.grad(curvature, parameters, retain_graph=False,
                                           allow_unused=True)
    finally:
        for layer in model.rq.vq_layers:
            layer._skip_ddp_reduce = False
        model.zero_grad(set_to_none=True)
    g_original = tuple(torch.zeros_like(p) if g is None else g for p, g in zip(parameters, g_original))
    g_structural = tuple(torch.zeros_like(p) if g is None else g for p, g in zip(parameters, g_structural))
    index = {id(p): i for i, p in enumerate(parameters)}
    gate = tuple(p for name, p in model.named_parameters()
                 if name.startswith("structure.") and ("attention" in name or "gate" in name))
    l3 = tuple(model.rq.vq_layers[2].parameters())
    gate_idx = [index[id(p)] for p in gate if id(p) in index]
    l3_idx = [index[id(p)] for p in l3 if id(p) in index]
    if not gate_idx or not l3_idx:
        raise RuntimeError("missing gate or L3 parameter group")
    R = float(components["reconstruction"].detach())
    G_gate = norm_grad([g_structural[i] for i in gate_idx])
    G_L3 = norm_grad([g_original[i] for i in l3_idx])
    if not all(math.isfinite(x) and x > 0 for x in (R, G_gate, G_L3)):
        raise ValueError("zero/nonfinite score input; formula not applicable")
    score = (WEIGHTS[0] * (1.0 - norm(R, *R_RANGE))
             + WEIGHTS[1] * norm(math.log(G_gate), math.log(G_RANGE[0]), math.log(G_RANGE[1]))
             + WEIGHTS[2] * norm(math.log(G_L3), math.log(L_RANGE[0]), math.log(L_RANGE[1])))
    order = sorted(HISTORICAL, key=lambda row: row[2])
    lower = [row for row in order if row[3] <= score]
    upper = [row for row in order if row[3] > score]
    result = {
        "schema": "genrec.issue267.score_prediction/1",
        "issue": 267, "arm": ARM,
        "frozen": {"R_range": list(R_RANGE), "G_gate_range": list(G_RANGE),
                   "G_L3_range": list(L_RANGE), "weights": list(WEIGHTS),
                   "formula": "0.52137*(1-N_R(R))+0.19389*N_gate(log(G_gate))+0.28474*N_L3(log(G_L3))"},
        "metrics": {"R": R, "G_gate": G_gate, "G_L3": G_L3,
                    "relative_alpha": float(relative_alpha),
                    "N_R": norm(R, *R_RANGE),
                    "N_gate": norm(math.log(G_gate), math.log(G_RANGE[0]), math.log(G_RANGE[1])),
                    "N_L3": norm(math.log(G_L3), math.log(L_RANGE[0]), math.log(L_RANGE[1])),
                    "S": score},
        "identity": {"checkpoint": str(CHECKPOINT), "checkpoint_sha256": sha(CHECKPOINT),
                     "sid": str(SID) if SID.is_file() else None,
                     "sid_sha256": sha(SID) if SID.is_file() else None,
                     "source_tree": str(TREE), "tree_sha256": sha(TREE / "model/mechanism.py"),
                     "batch_sha256": batch_sha,
                     "sample_ids_sha256": hashlib.sha256(
                         np.asarray(sample["ids"], dtype=np.int64).tobytes()).hexdigest(),
                     "sample_centre": int(sample["centre"]),
                     "sample_rows": int(len(sample["target"])),
                     "contract_sha256": sha(OUT / "SCORE_CONTRACT.md"),
                     "git_head": __import__("subprocess").check_output(
                         ["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True).strip(),
                     "parameter_unchanged": before.digest() == hashlib.sha256(
                         b"".join(v.detach().cpu().numpy().tobytes()
                                  for _, v in sorted(model.state_dict().items()))).digest(),
                     "optimizer_steps": 0},
        "historical": [{"index": i, "arm": a, "recall@10": r, "S": s} for i, a, r, s in order],
        "prediction": {"S": score, "lower_historical": lower[-1] if lower else None,
                       "upper_historical": upper[0] if upper else None,
                       "outside_below": not lower, "outside_above": not upper},
        "scope": "before Stage3; no new Recall read; diagnostic only",
    }
    (OUT / "SCORE_PREDICTION.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({**result["metrics"], "S": score,
                      "prediction": result["prediction"]}, indent=2))


if __name__ == "__main__":
    main()
