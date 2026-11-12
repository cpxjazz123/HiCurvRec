"""Issue #198 part C: does the two-regime curvature actually change the codes, and does that explain the level-1 collapse?

Three measurements on the shipped checkpoint and the shipped SID tables, all inference only:
  * the whole catalogue re-exported with the gate on and with curvature 1 everywhere, giving the per-level disagreement rate;
  * the level-1 usage histograms of SIGN and of the frozen baseline, side by side;
  * one batch with only the twenty-eight items' curvature switched on, to see how far a single code assignment can spread.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
TREE = REPO_ROOT / "stage2_RQ-VAE" / "TIGER_sign_rq"
SIGN_RESULT = REPO_ROOT / "results" / "stage2_RQ-VAE" / "TIGER_sign_rq"
BASE_RESULT = REPO_ROOT / "results" / "stage2_RQ-VAE" / "TIGER_RQ-VAE"
CHUNK = 4096
BATCH = 1024
SEED = 42


def load():
    sys.path.insert(0, str(TREE))
    spec = importlib.util.spec_from_file_location("arm_sign", TREE / "train_rqvae.py")
    arm = importlib.util.module_from_spec(spec)
    sys.modules["arm_sign"] = arm
    spec.loader.exec_module(arm)
    from model import RQVAE, sign_metric
    return arm, RQVAE, sign_metric


def codes(model, embeddings: torch.Tensor, ids: torch.Tensor, device) -> torch.Tensor:
    with torch.no_grad():
        rows = []
        for start in range(0, embeddings.shape[0], CHUNK):
            part = embeddings[start:start + CHUNK].to(device)
            chunk_ids = ids[start:start + CHUNK]
            rows.append(model.get_indices(part, infer_use_sk=False, item_ids=chunk_ids.to(device)).cpu())
        return torch.cat(rows, dim=0)


def histogram(path: Path) -> dict:
    payload = json.loads(path.read_text())
    level1 = Counter(value[0] for value in payload.values())
    total = sum(level1.values())
    counts = np.array(sorted(level1.values(), reverse=True), dtype=np.float64)
    p = counts / counts.sum()
    return {
        "items": total,
        "codes_used": len(level1),
        "top1_share": float(p[0]),
        "top5_share": float(p[:5].sum()),
        "entropy_share_of_max": float(-(p * np.log(p)).sum() / np.log(min(len(level1), total))),
        "counts_top10": [int(x) for x in counts[:10]],
        "sid": str(path.relative_to(REPO_ROOT)),
    }


def main() -> int:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    arm, RQVAE, sign = load()
    embeddings = arm.load_embeddings(arm.EMBEDDING_FILE)
    frame = arm.pd.read_parquet(arm.TRAIN_FILE, columns=["seen_history", "target"])
    train_ids = np.unique(frame["target"].to_numpy(dtype=np.int64))
    edges = sorted({(int(h[-1]), int(t)) for h, t in
                    zip(frame["seen_history"].to_numpy(), frame["target"].to_numpy(np.int64))
                    if h is not None and len(h) > 0})
    groups = sign.components(sign.closed_subgraph(edges))
    delta = sign.delta_of(edges, sign.af3_weights(edges), groups)
    enabled = sorted(i for i, v in delta.items() if v is not None and v > 0.0)

    model = RQVAE(arm._tokenizer_config(), in_dim=embeddings.shape[1]).to(device)
    model.load_state_dict(torch.load(SIGN_RESULT / "rqvae_best.pth", map_location=device, weights_only=False)["state_dict"])
    model.eval()
    model.delta = delta

    all_embeddings = torch.from_numpy(np.ascontiguousarray(embeddings)).float()
    all_ids = torch.arange(all_embeddings.shape[0])

    gate_on = codes(model, all_embeddings, all_ids, device)
    saved = model.delta
    model.delta = None
    gate_off = codes(model, all_embeddings, all_ids, device)
    model.delta = saved

    per_level = [int((gate_on[:, level] != gate_off[:, level]).sum()) for level in range(gate_on.shape[1])]
    unchanged = int((gate_on == gate_off).all(dim=1).sum())

    rng = np.random.default_rng(SEED)
    batch_ids = torch.from_numpy(np.sort(rng.choice(train_ids, size=BATCH, replace=False)))
    batch = all_embeddings[batch_ids]
    with_gate = codes(model, batch, batch_ids, device)
    model.delta = None
    without = codes(model, batch, batch_ids, device)
    model.delta = saved
    batch_level1_changed = int((with_gate[:, 0] != without[:, 0]).sum())
    switched = [int(x) for x in batch_ids.tolist() if int(x) in set(enabled)]

    report = {
        "device": str(device),
        "items": int(all_embeddings.shape[0]),
        "gate_on_vs_off_per_level_changed": per_level,
        "items_with_all_codes_identical": unchanged,
        "per_level_rates": [per_level[i] / int(all_embeddings.shape[0]) for i in range(len(per_level))],
        "batch": {"size": BATCH, "enabled_items_in_batch": switched, "level1_codes_changed": batch_level1_changed},
        "level1_histogram_sign": histogram(SIGN_RESULT / "item_sids.json"),
        "level1_histogram_baseline": histogram(BASE_RESULT / "item_sids.json"),
    }
    (HERE / "audit_c.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
