"""C12/C13 of issue #239: curvature fidelity and SID structure for both arms.

Three representations are compared against the same graph target, in the same units, on the same
frozen samples:

  z      the encoder output before quantisation
  q_hard the hard codeword sum from the model's own argmin assignment
  q_sid  the codeword sum rebuilt from the exported `item_sids.json`

Reporting only `z` would repeat the mistake #238 was closed for: a continuous-space number says
nothing about the discrete SID that Stage3 reads.  `q_sid` is the one that matters.

Everything here is diagnostic except the pass/fail on "did the hard-q fidelity improve relative to
frozen TIGER", which issue #239 states as the success criterion (C14).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path("/fs04/ar57/wenyu/GeneRec")
CANDIDATE = "TIGER_curvloss_rq"
BASELINE = "TIGER_RQ-VAE"
GRAPH_DIR = REPO / "results" / "research" / "behavior_graph"
OUT = REPO / "results" / "research" / "issue239" / "evaluation.json"


def spearman(xs, ys):
    def ranks(values):
        order = sorted(range(len(values)), key=lambda i: values[i])
        rank = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            for k in range(i, j + 1):
                rank[order[k]] = (i + j) / 2.0 + 1.0
            i = j + 1
        return rank

    rx, ry = ranks(xs), ranks(ys)
    n = len(rx)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else float("nan")


def main() -> int:
    arm_dir = REPO / "stage2_RQ-VAE" / CANDIDATE
    sys.path.insert(0, str(arm_dir))
    import train_rqvae as entry
    import curv_supervision as cs
    import graph_curv
    from distances import mixed
    from model.model import RQVAE

    calibration = json.loads((REPO / "results" / "stage2_RQ-VAE" / CANDIDATE / "calibration.json")
                             .read_text(encoding="utf-8"))
    mixed.set_scale(float(calibration["distance"]["mapping_scale"]))
    scale_s = float(calibration["target"]["global_scale_s"])

    vectors = torch.from_numpy(
        entry.maybe_apply_pca(entry.load_embeddings(entry.EMBEDDING_FILE),
                              entry.PCA_DIM, entry.SEED))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    targets = cs.prepare(calibration, GRAPH_DIR, vectors.shape[0], seed=entry.SEED)
    m, b, c = targets["m"], targets["b"], targets["c"]
    # the degree stratum is the one of the triplet's middle node (m), not of an arbitrary item

    refs = targets["refs"]
    k_g = targets["k_g"].numpy()
    near_zero = 1e-3
    strata = {
        "positive": k_g > near_zero,
        "negative": k_g < -near_zero,
    }
    degree = np.zeros(vectors.shape[0])
    import pandas as pd
    edges = pd.read_parquet(GRAPH_DIR / "edges.parquet")
    np.add.at(degree, edges.iloc[:, 0].to_numpy(), 1.0)
    np.add.at(degree, edges.iloc[:, 1].to_numpy(), 1.0)
    quintile = np.digitize(degree, np.quantile(degree, [0.2, 0.4, 0.6, 0.8]))

    def curvature_of(points: torch.Tensor) -> np.ndarray:
        """K of an arbitrary point set, same references and formula as the target."""
        points = points.to(device)
        out = torch.full((m.numel(),), float("nan"))
        for start in range(0, m.numel(), 2048):
            sl = slice(start, start + 2048)
            qm, qb, qc = points[m[sl].to(device)], points[b[sl].to(device)], points[c[sl].to(device)]
            qa = points[refs.to(device)]
            d2_am = mixed.distances(qa, qm)
            d2_ab = mixed.distances(qa, qb)
            d2_ac = mixed.distances(qa, qc)
            d2_bc = mixed.pairwise_sq(qb, qc).clamp_min(0.0)
            out[sl] = (d2_am + d2_bc.unsqueeze(0) / 4.0 - (d2_ab + d2_ac) / 2.0).mean(0).cpu()
        return out.numpy()

    quintile_m = np.digitize(degree[m.numpy()], np.quantile(degree, [0.2, 0.4, 0.6, 0.8]))

    def statistics(k_q: np.ndarray) -> dict:
        ok = np.isfinite(k_q)
        resid = scale_s * k_q[ok] - k_g[ok]
        result = {
            "n": int(ok.sum()),
            "mae": float(np.abs(resid).mean()),
            "rmse": float(np.sqrt((resid ** 2).mean())),
            "spearman": round(spearman(k_q[ok], k_g[ok]), 4),
            "bias": float(resid.mean()),
        }
        for name, mask in strata.items():
            sel = ok & mask
            if sel.any():
                result[f"mae_{name}"] = float(np.abs(scale_s * k_q[sel] - k_g[sel]).mean())
                result[f"sign_agreement_{name}"] = float(
                    (np.sign(scale_s * k_q[sel]) == np.sign(k_g[sel])).mean())
        mae_q = []
        for q in range(5):
            sel = ok & (quintile_m == q)
            mae_q.append(float(np.abs(scale_s * k_q[sel] - k_g[sel]).mean()) if sel.any() else None)
        result["mae_by_degree_quintile"] = mae_q
        return {key: (round(value, 6) if isinstance(value, float) else value)
                for key, value in result.items()}

    def sid_statistics(arm: str) -> dict:
        payload = json.loads((REPO / "results" / "stage2_RQ-VAE" / arm / "item_sids.json")
                             .read_text(encoding="utf-8"))
        rows = [payload[str(i)][:3] for i in range(len(payload))]
        triples = {}
        for row in rows:
            triples[tuple(row)] = triples.get(tuple(row), 0) + 1
        counts = [len({row[level] for row in rows}) for level in range(3)]
        entropy = []
        for level in range(3):
            freq = {}
            for row in rows:
                freq[row[level]] = freq.get(row[level], 0) + 1
            total = sum(freq.values())
            entropy.append(round(-sum(v / total * math.log2(v / total) for v in freq.values()), 4))
        shared = {}
        for level in (1, 2):
            prefixes = {}
            for row in rows:
                prefixes[tuple(row[:level])] = prefixes.get(tuple(row[:level]), 0) + 1
            shared[f"level{level}_prefixes"] = len(prefixes)
            shared[f"level{level}_max_share"] = round(max(prefixes.values()) / len(rows), 4)
        return {
            "items": len(rows),
            "distinct_triples": len(triples),
            "collision": round(1 - len(triples) / len(rows), 6),
            "used_per_level": counts,
            "entropy_per_level": entropy,
            "max_bucket_share": round(max(triples.values()) / len(rows), 4),
            **shared,
        }

    report = {"scale_s": scale_s, "delta": calibration["target"]["huber_delta"],
              "triplets": int(m.numel()), "arms": {}}
    for arm in (BASELINE, CANDIDATE):
        checkpoint = REPO / "results" / "stage2_RQ-VAE" / arm / "rqvae_best.pth"
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = state.get("state_dict", state)
        model = RQVAE(entry._tokenizer_config(), in_dim=int(state["encoder.mlp.1.weight"].shape[1]))
        model.load_state_dict(state, strict=False)
        model.to(device).eval()
        with torch.no_grad():
            latent = model.encoder(vectors.to(device)).cpu()
            tokens = model.get_indices(vectors.to(device)).cpu()
            codes = [layer.get_code_embs().cpu() for layer in model.rq.vq_layers]
            q_hard = sum(codes[level][tokens[:, level]] for level in range(3))
            payload = json.loads((REPO / "results" / "stage2_RQ-VAE" / arm / "item_sids.json")
                                 .read_text(encoding="utf-8"))
            sid_rows = torch.tensor([payload[str(i)][:3] for i in range(len(payload))])
            q_sid = sum(codes[level][sid_rows[:, level]] for level in range(3))
            reconstructed = model.decoder(q_hard.to(device)).cpu()
            mse = float((reconstructed - vectors).pow(2).mean())

        entry_stats = {"reconstruction_mse": round(mse, 8), "sid": sid_statistics(arm)}
        for name, points in (("z", latent), ("q_hard", q_hard), ("q_sid", q_sid)):
            entry_stats[name] = statistics(curvature_of(points))
        # diagnostic: real edges should be closer than a random partner
        rng = np.random.default_rng(0)
        sample = edges.sample(n=4000, random_state=42)
        src = torch.as_tensor(sample.iloc[:, 0].to_numpy())
        dst = torch.as_tensor(sample.iloc[:, 1].to_numpy())
        neg = torch.as_tensor(rng.integers(0, vectors.shape[0], len(src)))
        with torch.no_grad():
            pos_d = mixed.pairwise_sq(q_sid[src].to(device), q_sid[dst].to(device))
            neg_d = mixed.pairwise_sq(q_sid[src].to(device), q_sid[neg].to(device))
            entry_stats["edge_vs_random_closer_fraction"] = round(
                float((pos_d < neg_d).float().mean()), 4)
        report["arms"][arm] = entry_stats
        del model
        torch.cuda.empty_cache()

    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
