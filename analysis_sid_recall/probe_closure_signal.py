"""Issue #216 diagnostic: where does the closure signal live -- encoder or SID?

For every arm, encode all 24587 items with the trained encoder (the pre-quantization
representation z) and score each heldout pair by cos(z_b, z_c).  The AUC of
closure-vs-branch separates two stories:

* high AUC with a collapsed SID prefix lift  -> the encoder learned the distinction
  and the quantizer is what loses it (readout bypass);
* low AUC as well                            -> the supervision never entered the
  representation at all.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
TREE = REPO / "stage2_RQ-VAE" / "TIGER_g4_closure_rq"
POOL = REPO / "results" / "research" / "g4_closure" / "pairs_heldout.npz"
TRAIN_POOL = REPO / "results" / "research" / "g4_closure" / "pairs_train.npz"
OUT = REPO / "results" / "research" / "g4_closure" / "probe.json"
ARMS = {
    "A": REPO / "results/stage2_RQ-VAE/TIGER_g4_closure_A/rqvae_best.pth",
    "B": REPO / "results/stage2_RQ-VAE/TIGER_g4_closure_B/rqvae_best.pth",
    "C": REPO / "results/stage2_RQ-VAE/TIGER_g4_closure_C/rqvae_best.pth",
    "D": REPO / "results/stage2_RQ-VAE/TIGER_g4_closure_D/rqvae_best.pth",
    "baseline": REPO / "results/stage2_RQ-VAE/TIGER_RQ-VAE/rqvae_best.pth",
}


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    positives = labels == 1
    n_pos, n_neg = int(positives.sum()), int((~positives).sum())
    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def main() -> None:
    sys.path.insert(0, str(TREE))
    from model.layers import MLP  # noqa: E402 - the fitted encoder lives in the tree

    config = json.loads((TREE / "config" / "train_config.json").read_text())
    embeddings = np.load(REPO / "stage1_GeneEmbedding/output/sentence_t5.npy")
    pool = np.load(POOL)
    triples, labels = pool["triples"], pool["labels"]
    train_pool = np.load(TRAIN_POOL)
    train_triples, train_labels = train_pool["triples"], train_pool["labels"]
    print(f"embeddings {embeddings.shape}, heldout pairs {len(triples)}, train pairs {len(train_triples)}")
    results = {}
    for arm, path in ARMS.items():
        if not path.is_file():
            print(f"{arm}: no checkpoint"); continue
        payload = torch.load(path, map_location="cpu", weights_only=False)
        state = payload.get("state_dict") or payload.get("model") or payload
        sizes = [embeddings.shape[1], *config["HIDDEN_SIZES"], config["CODEBOOK_DIM"]]
        encoder = MLP(sizes, dropout=float(config.get("DROPOUT") or 0.0))
        encoder.load_state_dict({k[len("encoder."):]: v for k, v in state.items()
                                 if k.startswith("encoder.")})
        encoder.eval()
        with torch.no_grad():
            z = encoder(torch.from_numpy(embeddings).float()).numpy()
        first = z[triples[:, 1].astype(np.int64)]
        second = z[triples[:, 2].astype(np.int64)]
        cosine = (first * second).sum(1) / (np.linalg.norm(first, axis=1) * np.linalg.norm(second, axis=1))
        score = auc(cosine, labels)
        # A cosine is a one-dimensional readout; a ridge probe on the pair features
        # |z_b - z_c| and z_b * z_c asks whether the information is in z at all.
        def features(pair_triples):
            left = z[pair_triples[:, 1].astype(np.int64)]
            right = z[pair_triples[:, 2].astype(np.int64)]
            return np.concatenate([np.abs(left - right), left * right], axis=1)
        train_x, heldout_x = features(train_triples), features(triples)
        design = np.concatenate([train_x, np.ones((len(train_x), 1))], axis=1)
        ridge = np.linalg.solve(design.T @ design + 1e-1 * np.eye(design.shape[1]),
                                design.T @ (train_labels == 1).astype(np.float64))
        heldout_design = np.concatenate([heldout_x, np.ones((len(heldout_x), 1))], axis=1)
        probe = auc(heldout_design @ ridge, labels)
        results[arm] = {"auc": score, "ridge_probe_auc": probe,
                        "positive_mean": float(cosine[labels == 1].mean()),
                        "negative_mean": float(cosine[labels == 0].mean()),
                        "gap": float(cosine[labels == 1].mean() - cosine[labels == 0].mean())}
        print(f"{arm:9s} cos-AUC {score:.4f}  ridge-AUC {probe:.4f}  "
              f"cos+ {results[arm]['positive_mean']:.4f} cos- {results[arm]['negative_mean']:.4f}")
    OUT.write_text(json.dumps(results, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
