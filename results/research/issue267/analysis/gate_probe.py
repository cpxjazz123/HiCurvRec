"""Issue #267: low-cost gate-vs-equal-weight curvature probe on 109.

Read-only. Uses the frozen structural sampler draws and the actual #265 model.
No optimizer, no parameter update, no SID write.
"""
from __future__ import annotations

import importlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "results/research/issue267/analysis"
ARM = "TIGER_euclid_local_curve_supervision_rq"
DRAWS = 32


def main() -> None:
    final = sorted((ROOT / "results/stage3_T5Train" / ARM).glob("**/test_final.json"))[0]
    fd = json.loads(final.read_text())
    checkpoint = Path(fd["stage2_checkpoint"])
    source = Path(json.loads(Path(fd["archive_binding"]).read_text())["source_tree"])
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    for name in list(sys.modules):
        if name.split(".")[0] in ("model", "losses", "distances", "train_config", "train_rqvae"):
            del sys.modules[name]
    sys.path.insert(0, str(source))
    training = importlib.import_module("train_rqvae")
    model = training.RQVAE(training._tokenizer_config(), in_dim=768, distance=training.DISTANCE).cpu()
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    embeddings = np.load(ROOT / "stage1_GeneEmbedding/output/sentence_t5.npy", allow_pickle=False)
    catalogue = torch.from_numpy(np.array(embeddings, copy=True))
    sampler = model.structural_sampler()
    records = []
    with torch.no_grad():
        for step in range(DRAWS):
            sample = sampler.draw(step, 0, 1, track=False)
            host = np.asarray(sample["ids"], dtype=np.int64)
            ids = torch.from_numpy(host)
            _, weights = model.structure.gates_for_items(model.store, host, torch.device("cpu"),
                                                         behavior_on=True)
            q, _, _, _ = model.quantize(model.encoder(catalogue[ids]), weights, balanced=False)
            centre = int(sample["centre"])
            local = int(np.flatnonzero(host == centre)[0])
            centre_weights = weights[local].mean(0)
            equal_weights = torch.full_like(centre_weights, 1.0 / 3.0)
            scale = model._tangent_scale_value
            target = torch.as_tensor(sample["target"], dtype=q.dtype)
            def predict(w):
                local_squared = lambda x, y: model.assignment_cost(x, y, w.unsqueeze(0))
                # assignment_cost is Euclidean in #265; use mixed geometry for the
                # structural probe, matching sampled_curvature.
                from model.geometry import mixed_cost
                def sq(x, y):
                    return mixed_cost(x, y, w, scale=scale, curvature_on=True)
                def paired(x, y):
                    return mixed_cost(x[:, None, :], y[:, None, :], w, scale=scale,
                                      curvature_on=True).reshape(-1)
                from model.estimator import sampled_values
                return sampled_values(q, sample["triples"], sample["references"], sq,
                                      paired_squared=paired)
            gate_pred = predict(centre_weights)
            equal_pred = predict(equal_weights)
            records.append({
                "step": step, "centre": centre, "rows": int(len(target)),
                "target": target.detach().numpy().astype(float).tolist(),
                "gate": gate_pred.detach().numpy().astype(float).tolist(),
                "equal": equal_pred.detach().numpy().astype(float).tolist(),
                "gate_weights": centre_weights.detach().numpy().astype(float).tolist(),
            })
    target = np.concatenate([np.asarray(r["target"]) for r in records])
    gate = np.concatenate([np.asarray(r["gate"]) for r in records])
    equal = np.concatenate([np.asarray(r["equal"]) for r in records])
    def metrics(pred):
        return {
            "mae": float(np.mean(np.abs(pred - target))),
            "rmse": float(np.sqrt(np.mean((pred - target) ** 2))),
            "pearson": float(pearsonr(target, pred).statistic),
            "spearman": float(spearmanr(target, pred).statistic),
            "pred_mean": float(pred.mean()), "pred_var": float(pred.var()),
            "target_mean": float(target.mean()), "target_var": float(target.var()),
        }
    pair = gate - equal
    result = {
        "schema": "genrec.issue267.gate_probe/1",
        "arm": ARM, "draws": DRAWS, "rows": int(len(target)),
        "identity": {"checkpoint": str(checkpoint), "final": str(final),
                     "source_tree": str(source)},
        "gate": metrics(gate), "equal": metrics(equal),
        "delta_gate_minus_equal": {"mae": float(np.mean(np.abs(gate - target)) -
                                                np.mean(np.abs(equal - target))),
                                   "prediction_mean": float(pair.mean()),
                                   "prediction_var": float(pair.var())},
        "scope": "training-side sampler draws; low-cost diagnostic, not full 28M evaluation",
        "records": records,
    }
    (OUT / "GATE_PROBE.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("draws", "rows", "gate", "equal",
                                             "delta_gate_minus_equal")}, indent=2))


if __name__ == "__main__":
    main()
