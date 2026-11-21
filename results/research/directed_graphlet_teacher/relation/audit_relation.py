#!/usr/bin/env python3
"""Audit the existing 128-d relation target before using it as structure supervision.

Three questions, each answered by a number:

1. stability   -- does an independently retrained teacher (other seed) reproduce it?
2. beyond text -- does relation cosine discriminate held-out behaviour pairs better
                  than the 768-d Stage1 embedding, and does it keep doing so after a
                  ridge model has removed everything predictable from the 768-d vector?
3. SID ceiling -- do already-exported discrete SIDs (trained on the 768-d vector only)
                  already group the relation-nearest items, i.e. how much of the
                  relation structure is reachable without touching the encoder input.

Only reads frozen artifacts; trains nothing. Writes relation/audit_relation.json.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TEACHER = HERE.parent
sys.path.insert(0, str(TEACHER))

import graph_teacher3 as gt3  # noqa: E402

REPO = Path("/fs04/ar57/wenyu/GeneRec")
OUT = HERE
TARGETS = TEACHER / "teacher" / "structure" / "v3" / "structure_targets_v3.npz"
SID_ARMS = ("TIGER_RQ-VAE", "TIGER_RQ-VAE__canonical", "TIGER_mixed_rq_F0")
K = 10
PROBES = 2000
LAM_FRAC = 1e-3
SEED = 20261121


# ---------------------------------------------------------------- pure functions
def l2norm(a: np.ndarray, axis: int = -1) -> np.ndarray:
    n = np.linalg.norm(a, axis=axis, keepdims=True)
    return a / np.maximum(n, 1e-12)


def auc_score(positive: np.ndarray, negative: np.ndarray) -> float:
    """Mann-Whitney AUC with average ranks (ties count as half)."""
    positive = np.asarray(positive, dtype=np.float64).ravel()
    negative = np.asarray(negative, dtype=np.float64).ravel()
    if positive.size == 0 or negative.size == 0:
        raise ValueError("both classes need at least one score")
    scores = np.concatenate([positive, negative])
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(scores.size, dtype=np.float64)
    sorted_scores = scores[order]
    i = 0
    while i < scores.size:
        j = i
        while j + 1 < scores.size and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    rank_pos = ranks[:positive.size].sum()
    u = rank_pos - positive.size * (positive.size + 1) / 2.0
    return float(u / (positive.size * negative.size))


def prefix_score(codes_a: np.ndarray, codes_b: np.ndarray) -> np.ndarray:
    """Shared leading-code count (0..3) for aligned rows of two code matrices."""
    if codes_a.shape != codes_b.shape or codes_a.shape[1] < 1:
        raise ValueError("codes must be aligned (n, depth) matrices")
    score = np.zeros(len(codes_a), dtype=np.float64)
    alive = np.ones(len(codes_a), dtype=bool)
    for level in range(codes_a.shape[1]):
        alive &= codes_a[:, level] == codes_b[:, level]
        score += alive
    return score


def pair_cosine(states: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    return (states[pairs[:, 0]] * states[pairs[:, 1]]).sum(axis=1)


def ridge_predict(x: np.ndarray, y: np.ndarray, lam_frac: float = LAM_FRAC) -> tuple:
    """Closed-form ridge with a relative penalty, centred inputs, no bias term needed."""
    xc = x - x.mean(axis=0, keepdims=True)
    yc = y - y.mean(axis=0, keepdims=True)
    gram = xc.T @ xc
    lam = lam_frac * float(np.trace(gram)) / x.shape[1]
    w = np.linalg.solve(gram + lam * np.eye(gram.shape[0]), xc.T @ yc)
    return w, x.mean(axis=0, keepdims=True), y.mean(axis=0, keepdims=True)


def top_k(sim_row: np.ndarray, k: int, exclude: int) -> np.ndarray:
    row = sim_row.copy()
    row[exclude] = -np.inf
    return np.argpartition(-row, k)[:k]


def graph_only(edges: np.ndarray, n_items: int) -> list:
    """Undirected neighbour sets from edge rows, self loops removed."""
    neighbours = [set() for _ in range(n_items)]
    for left, right in edges:
        if left == right:
            continue
        neighbours[int(left)].add(int(right))
        neighbours[int(right)].add(int(left))
    return neighbours


def graph_precision(sim_block: np.ndarray, probes: np.ndarray, neighbours: list,
                    k: int = K) -> float:
    hits = 0
    for row, item in enumerate(probes):
        knn = top_k(sim_block[row], k, int(item))
        hits += sum(int(c) in neighbours[int(item)] for c in knn)
    return hits / (len(probes) * k)


def code_purity(sim_block: np.ndarray, probes: np.ndarray, codes: np.ndarray,
                k: int = K) -> tuple:
    """Fraction of each item's top-k whose first SID code equals the item's own,
    against the same first-code share over the whole catalogue (chance level)."""
    own = codes[probes, 0]
    frequency = np.bincount(codes[:, 0])
    chance = frequency[own] / len(codes)
    hit = 0
    for row, item in enumerate(probes):
        knn = top_k(sim_block[row], k, int(item))
        hit += int((codes[knn, 0] == codes[item, 0]).sum())
    return hit / (len(probes) * k), float(chance.mean())


def jaccard(a: set, b: set) -> float:
    return len(a & b) / max(len(a | b), 1)


def r2_multi(predicted: np.ndarray, target: np.ndarray) -> float:
    """Fraction of target variance explained, over all entries."""
    return float(1.0 - ((target - predicted) ** 2).sum()
                 / ((target - target.mean(axis=0, keepdims=True)) ** 2).sum())


def procrustes_align(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Rotate `source` rows onto `target` rows (orthogonal Procrustes)."""
    u, _, vt = np.linalg.svd(source.T @ target)
    return source @ (u @ vt)


def stable_sample(index: np.ndarray, count: int, seed: int = SEED) -> np.ndarray:
    """Deterministic subsample: order by sha256(item id + seed), take first `count`."""
    if len(index) <= count:
        return np.sort(index)
    keys = np.array([int.from_bytes(
        hashlib.sha256(f"{seed}:{int(i)}".encode()).digest()[:8], "big") for i in index],
        dtype=np.uint64)
    order = np.argsort(keys)
    return np.sort(index[order[:count]])


# ---------------------------------------------------------------------- encoding
def build_variant_cached(name: str, path: Path, device):
    import torch
    variant = gt3.build_variant(name)
    gt3.load_checkpoint(path, variant)
    return variant.to(device).eval()


def encode_seed(path: Path, features, adjacency, device) -> np.ndarray:
    import torch
    variant = gt3.build_variant("C_residual")
    gt3.load_checkpoint(path, variant)
    variant.to(device).eval()
    with torch.no_grad():
        state = variant.encode(features, adjacency)
    return l2norm(state.detach().cpu().numpy().astype(np.float64))


def load_sid_codes(arm: str, n_items: int) -> np.ndarray | None:
    path = REPO / "results" / "stage2_RQ-VAE" / arm / "item_sids.json"
    if not path.exists():
        return None
    raw = json.loads(path.read_text())
    codes = np.full((n_items, 3), -1, dtype=np.int64)
    for key, value in raw.items():
        item = int(key)
        if item < n_items:
            codes[item] = [int(v) for v in value[:3]]
    return codes


# -------------------------------------------------------------------------- main
def main() -> None:
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = gt3.prepare_data(device)
    features, adjacency = data["features"], data["adjacency"]
    n_items = data["n_items"]

    targets = np.load(TARGETS)
    relation = l2norm(targets["relation"].astype(np.float64))
    covered = targets["coverage_neighbour"] > 0          # where relation is a target
    own_covered = targets["coverage"] > 0                 # where own-orbit is a target
    semantics = l2norm(np.asarray(data["features"].cpu(), dtype=np.float64))

    sample = stable_sample(np.flatnonzero(covered), 400)
    edge = data["test_positive"]
    negative = data["test_negative"]
    covered_pairs = covered[edge[:, 0]] & covered[edge[:, 1]]

    report = {"inputs": {
        "targets_sha256": hashlib.sha256(TARGETS.read_bytes()).hexdigest(),
        "semantics": "stage1_GeneEmbedding/output/sentence_t5.npy",
        "covered_items": int(covered.sum()),
        "own_orbit_covered_items": int(own_covered.sum()),
        "test_pairs": int(len(edge)),
        "test_pairs_both_covered": int(covered_pairs.sum()),
    }}

    # 1. provenance and seed stability -------------------------------------------
    relation_by_seed = {}
    provenance = {}
    for seed in (226, 227):
        path = TEACHER / "teacher" / "opt" / f"C_residual_edge_main_seed{seed}.pth"
        if not path.exists():
            continue
        rel = encode_seed(path, features, adjacency, device)
        relation_by_seed[seed] = rel
        provenance[str(seed)] = {
            "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "cosine_to_stored_relation": float(
                (rel[sample] * relation[sample]).sum(axis=1).mean()),
        }
    report["provenance"] = provenance
    saved = TEACHER / "teacher" / "structure" / "v3" / "teacher_structure_v3.pth"
    if saved.exists():
        import torch as _torch
        payload = _torch.load(saved, map_location="cpu", weights_only=False)
        meta = payload.get("meta", {})
        report["provenance"]["carrier_file"] = {
            "sha256": hashlib.sha256(saved.read_bytes()).hexdigest(),
            "source_checkpoint": meta.get("source_checkpoint"),
            "source_checkpoint_sha256": meta.get("source_checkpoint_sha256"),
            "graph_sha_matches_contract": meta.get("graph_sha") == data["meta"]["graph_sha"],
            "split_sha_matches_contract": meta.get("split_sha") == data["split_sha"],
            "note": "carrier is 768-only; relation itself needs the graph",
        }

    stability = {}
    if len(relation_by_seed) == 2:
        a, b = relation_by_seed[226], relation_by_seed[227]
        cos = (a[covered] * b[covered]).sum(axis=1)
        sim_a = a[sample] @ a.T
        sim_b = b[sample] @ b.T
        overlap = [jaccard(set(top_k(sim_a[r], K, int(item)).tolist()),
                           set(top_k(sim_b[r], K, int(item)).tolist()))
                   for r, item in enumerate(sample)]
        stability = {
            "items": int(covered.sum()),
            "cosine_mean": float(cos.mean()),
            "cosine_median": float(np.median(cos)),
            "cosine_p10": float(np.quantile(cos, 0.10)),
            f"top{K}_jaccard_mean": float(np.mean(overlap)),
        }
    report["stability"] = stability

    # 2. held-out discrimination and structure beyond semantics ------------------
    w, x_mean, y_mean = ridge_predict(semantics[covered], relation[covered])
    predicted = (semantics - x_mean) @ w + y_mean
    residual = l2norm(relation - predicted)
    variance = 1.0 - ((relation - predicted) ** 2).sum() / ((relation - relation.mean(0)) ** 2).sum()

    scores = {
        "relation": pair_cosine(relation, edge),
        "semantics_768": pair_cosine(semantics, edge),
        "residual_after_ridge": pair_cosine(residual, edge),
    }
    rng = np.random.default_rng(SEED)
    aucs = {}
    for name, value in scores.items():
        auc = auc_score(value, value[negative])
        boot = []
        for _ in range(200):
            idx = rng.integers(0, len(value), len(value))
            boot.append(auc_score(value[idx], value[negative[idx]]))
        aucs[name] = {"auc": auc, "p05": float(np.quantile(boot, 0.05)),
                      "p95": float(np.quantile(boot, 0.95))}
        both = covered_pairs
        if both.sum() > 100:
            aucs[name]["auc_both_covered"] = auc_score(value[both], value[both][negative[both]])
    report["ridge"] = {"r2_variance_explained": float(variance),
                       "lam_frac": LAM_FRAC,
                       "relation_norm_kept": float(np.linalg.norm(residual, axis=1).mean())}
    report["pair_auc"] = aucs

    # 2b. seed stability that is invariant to the contrastive rotation -----------
    if len(relation_by_seed) == 2:
        a = relation_by_seed[226][covered]
        b = relation_by_seed[227][covered]
        aligned = procrustes_align(b, a)
        cos_aligned = (aligned * a).sum(axis=1)
        per_seed_auc = {}
        for seed, rel in relation_by_seed.items():
            value = pair_cosine(rel, edge)
            per_seed_auc[str(seed)] = auc_score(value, value[negative])
        report["stability"]["procrustes_cosine_mean"] = float(cos_aligned.mean())
        report["stability"]["procrustes_cosine_median"] = float(np.median(cos_aligned))
        report["stability"]["per_seed_pair_auc"] = per_seed_auc

    # 2c. held-out ridge: how much of relation a 768-only linear map really keeps --
    fit_items = np.flatnonzero(covered)[: int(0.8 * covered.sum())]
    eval_items = np.setdiff1d(np.flatnonzero(covered), fit_items)
    w_fit, xm, ym = ridge_predict(semantics[fit_items], relation[fit_items])
    heldout = (semantics[eval_items] - xm) @ w_fit + ym
    report["ridge"]["r2_heldout"] = r2_multi(heldout, relation[eval_items])
    report["ridge"]["heldout_items"] = int(len(eval_items))

    # 2d. where the edge signal actually lives: cosine geometry vs learned head ---
    neighbour_relation = l2norm(targets["neighbour_relation"].astype(np.float64))
    nb_value = pair_cosine(neighbour_relation, edge)
    report["pair_auc"]["neighbour_relation"] = {
        "auc": auc_score(nb_value, nb_value[negative]),
        "auc_both_covered": auc_score(nb_value[covered_pairs],
                                       nb_value[covered_pairs][negative[covered_pairs]]),
    }
    readout = {}
    opt = TEACHER / "teacher" / "opt"
    for name, filename in (("C_residual", "C_residual_edge_main_seed226.pth"),
                           ("A_embedding", "A_embedding_edge_main_seed226.pth")):
        path = opt / filename
        if not path.exists():
            continue
        import torch as _torch
        variant = build_variant_cached(name, path, device)
        with _torch.no_grad():
            scores = gt3.pair_scores(variant, features, adjacency, edge, negative, device)
        readout[name] = {"checkpoint": filename, "pair_head_auc": gt3.auc_of(scores)}
    report["learned_readout"] = readout

    # 3. neighbourhood agreement with the observed TRAIN graph -------------------
    neighbours = graph_only(data["message_edges"], n_items)
    degree = np.array([len(n) for n in neighbours])
    probe_pool = np.flatnonzero(covered & (degree >= 3))
    probes = stable_sample(probe_pool, PROBES)
    knn = {}
    for name, states in (("relation", relation),
                         ("neighbour_relation", neighbour_relation),
                         ("semantics_768", semantics),
                         ("residual_after_ridge", residual)):
        block = states[probes] @ states.T
        knn[name] = graph_precision(block, probes, neighbours)
    report["graph_precision"] = {"k": K, "probes": int(len(probes)),
                                 "probe_pool": int(len(probe_pool)),
                                 "points": knn}

    # 4. what existing 768-trained SIDs already retain ---------------------------
    sid = {}
    rel_block = relation[probes] @ relation.T
    sem_block = semantics[probes] @ semantics.T
    for arm in SID_ARMS:
        codes = load_sid_codes(arm, n_items)
        if codes is None:
            sid[arm] = {"status": "MISSING"}
            continue
        valid = codes[:, 0] >= 0
        pair_ok = valid[edge[:, 0]] & valid[edge[:, 1]]
        prefix = prefix_score(codes[edge[:, 0]], codes[edge[:, 1]])
        prefix_neg = prefix_score(codes[edge[:, 0]], codes[negative])
        covered_probe = probes[valid[probes]]
        rel_purity, rel_chance = code_purity(rel_block, covered_probe, codes)
        sem_purity, _ = code_purity(sem_block, covered_probe, codes)
        sid[arm] = {
            "sid_sha256": hashlib.sha256(
                (REPO / "results" / "stage2_RQ-VAE" / arm / "item_sids.json").read_bytes()
            ).hexdigest(),
            "items": int(valid.sum()),
            "prefix_auc": auc_score(prefix[pair_ok], prefix_neg[pair_ok]),
            "first_code_purity_relation_top": rel_purity,
            "first_code_purity_semantics_top": sem_purity,
            "first_code_chance": rel_chance,
            "probes": int(len(covered_probe)),
        }
    report["sid_retention"] = sid

    report["readings"] = readings(report)
    OUT.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, indent=1, ensure_ascii=False)
    (OUT / "audit_relation.json").write_text(payload + "\n")
    print(payload)


def readings(report: dict) -> dict:
    """Plain-language numbers, so the conclusion is not a matter of taste."""
    out = {}
    auc = report.get("pair_auc", {})
    if "relation" in auc and "semantics_768" in auc:
        out["relation_beats_semantics"] = (
            auc["relation"]["auc"] > auc["semantics_768"]["p95"])
    if "residual_after_ridge" in auc:
        out["residual_above_chance"] = auc["residual_after_ridge"]["p05"] > 0.5
    stab = report.get("stability", {})
    if stab:
        out["seed_stable"] = (stab.get("procrustes_cosine_median", 0.0) >= 0.9
                             and stab[f"top{K}_jaccard_mean"] >= 0.5)
    sid = report.get("sid_retention", {})
    if sid and all(v.get("first_code_purity_relation_top") is not None
                   for v in sid.values()):
        out["sid_relation_group_gap"] = {
            arm: round(v["first_code_purity_relation_top"]
                       / max(v["first_code_purity_semantics_top"], 1e-9), 3)
            for arm, v in sid.items()}
    return out


if __name__ == "__main__":
    main()
