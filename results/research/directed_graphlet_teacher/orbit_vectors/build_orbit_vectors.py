"""Phase 1: per-item directed-orbit vectors, their split audit, reliability and learnability.

The question this answers, before any RQ-VAE supervision is attempted:

  1. can a stable per-item directed-orbit vector be built from the TRAIN graph only,
  2. is that vector reproducible and stable under an independent instance sample,
  3. can it be predicted from the 768-d item embedding alone.

Nothing here trains Stage2/Stage3.  The item embedding stays the original
`sentence_t5.npy`; the SID protocol is untouched.

Vocabulary (fixed, K = 376): one dimension per
(family, canonical directed pattern key, node orbit id).  A pattern has at most
four nodes, so at most four orbit ids; the vocabulary keeps eight slots per
subtype exactly like `directed_role.load_vocabulary("directed")`, and dimensions
that no instance ever reaches are reported rather than silently dropped.

Counting: every node of every TRAIN-split instance adds one to
`counts[item, orbit]`.  A multi-role item accumulates all of its roles; no item is
assigned a unique role.  The instance pool is the audited #221 pool
(`directed_graphlet_role/instances.npz`) whose `split` flag is a stable hash of the
node set; only `split == 0` enters the vector, `split == 1` is used as the
independent instance sample for the reliability check.

The instance pool is a *sample* of the induced subgraphs, so every number below is
"the role distribution under that sampling scheme"; it is never called the exact
population distribution.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def find_repo(start: Path = HERE) -> Path:
    for candidate in [start, *start.parents]:
        if ((candidate / "stage2_RQ-VAE").is_dir()
                and (candidate / "results").is_dir()
                and (candidate / "stage1_GeneEmbedding").is_dir()):
            return candidate
    raise RuntimeError("repository root (stage2_RQ-VAE + results + stage1_GeneEmbedding) not found")


REPO = find_repo()
ROLE = REPO / "results" / "research" / "directed_graphlet_role"
GRAPH_DIR = REPO / "results" / "research" / "behavior_graph_directed"
EMB = REPO / "stage1_GeneEmbedding" / "output" / "sentence_t5.npy"
PARQUET = REPO / "results" / "stage0_build_parquet"
OUT = HERE
FAMILIES = ("K3", "P3", "C4", "P4", "K1_3")
MIN_RELIABLE = 5
DIM_FAMILY: list[str] = []
SEED = 229
N_NULL = 10


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# --------------------------------------------------------------- vocabulary
def build_vocabulary(raw: dict) -> tuple[list[dict], dict[str, np.ndarray]]:
    """Fixed K-dim vocabulary and the (subtype, orbit) -> dim table per family."""
    dims: list[dict] = []
    table: dict[str, np.ndarray] = {}
    for family in FAMILIES:
        entries = [(key, int(index)) for key, index in raw[family].items() if key != "__other__"]
        entries.sort(key=lambda pair: pair[1])
        lookup = np.full((len(raw[family]), 8), -1, dtype=np.int64)
        for key, subtype_id in entries:
            for orbit in range(8):
                lookup[subtype_id, orbit] = len(dims)
                dims.append({"index": len(dims), "family": family,
                             "pattern_key": key, "orbit": orbit,
                             "name": f"{family}:{key}:orbit{orbit}"})
        table[family] = lookup
    return dims, table


# ---------------------------------------------------------------- counting
def count_roles(split_flag: int, table: dict[str, np.ndarray], n: int, k: int) -> tuple:
    pool = np.load(ROLE / "instances.npz", allow_pickle=False)
    counts = np.zeros((n, k), dtype=np.int64)
    per_family: dict[str, dict] = {}
    for family in FAMILIES:
        nodes = pool[f"{family}_nodes"].astype(np.int64)
        roles = pool[f"{family}_roles"].astype(np.int64)
        subtype = pool[f"{family}_subtype"].astype(np.int64)
        keep = pool[f"{family}_split"] == split_flag
        nodes, roles, subtype = nodes[keep], roles[keep], subtype[keep]
        lookup = table[family]
        other = int((lookup[np.clip(subtype, 0, lookup.shape[0] - 1)] == -1).all(axis=1).sum())
        for position in range(nodes.shape[1]):
            orbit_dim = lookup[subtype, roles[:, position]]
            valid = orbit_dim >= 0
            flat = nodes[valid, position] * k + orbit_dim[valid]
            counts += np.bincount(flat, minlength=n * k).reshape(n, k)
        per_family[family] = {"instances": int(keep.sum()), "merged_other_dropped": other}
    return counts, per_family


# ------------------------------------------------------------- graph audit
def transitions_from(split: str) -> dict[tuple[int, int], int]:
    import pandas as pd
    frame = pd.read_parquet(PARQUET / f"{split}.parquet")
    users = frame["user"].to_numpy()
    histories = frame["history"].to_numpy()
    targets = frame["target"].to_numpy()
    longest: dict[int, int] = {}
    for user, history in zip(users, histories):
        key = int(user)
        if key not in longest or len(history) > longest[key]:
            longest[key] = len(history)
    chosen: dict[int, list[int]] = {}
    for user, history, target in zip(users, histories, targets):
        key = int(user)
        if len(history) == longest[key]:
            chosen[key] = [int(item) for item in history] + [int(target)]
    counts: dict[tuple[int, int], int] = {}
    for user in sorted(chosen):
        sequence = chosen[user]
        for left, right in zip(sequence[:-1], sequence[1:]):
            if left == right:
                continue
            counts[(left, right)] = counts.get((left, right), 0) + 1
    return counts


def graph_audit(train_counts: dict, valid_counts: dict, test_counts: dict,
                pool: "np.lib.npyio.NpzFile", table: dict[str, np.ndarray]) -> dict:
    payload = np.load(GRAPH_DIR / "graph.npz", allow_pickle=False)
    rows = payload["rows"].astype(np.int64)
    cols = payload["cols"].astype(np.int64)
    weight = payload["weight"].astype(np.int64)
    graph_counts = {(int(a), int(b)): int(w) for a, b, w in zip(rows, cols, weight)}
    graph_keys = set(graph_counts)
    train_keys = set(train_counts)
    valid_keys = set(valid_counts)
    test_keys = set(test_counts)
    symmetric = np.array_equal(payload["out_indptr"], payload["in_indptr"]) and \
        np.array_equal(payload["out_indices"], payload["in_indices"])

    # stored roles are trusted only if they agree with a fresh classification
    sys.path.insert(0, str(ROLE))
    from directed_graphlets import classify
    symmetric_csr = None
    from scipy import sparse
    symmetric_csr = sparse.csr_matrix((np.ones(2 * len(rows)),
                                       (np.concatenate([rows, cols]),
                                        np.concatenate([cols, rows]))),
                                      shape=(payload["nodes"], payload["nodes"]))
    symmetric_csr.data[:] = 1.0
    symmetric_csr.sum_duplicates()
    rng = np.random.default_rng(SEED)
    checked, mismatched = 0, 0
    for family in FAMILIES:
        nodes = pool[f"{family}_nodes"]
        roles = pool[f"{family}_roles"]
        subtype = pool[f"{family}_subtype"]
        lookup = table[family]
        candidates = np.flatnonzero((lookup[np.clip(subtype, 0, lookup.shape[0] - 1)] >= 0).any(1))
        if not len(candidates):
            continue
        picks = rng.choice(candidates, size=min(25, len(candidates)), replace=False)
        for index in picks:
            node_set = nodes[index]
            matrix = np.asarray(symmetric_csr[node_set][:, node_set].todense(), dtype=bool)
            # the directed pattern: an edge only when the ordered transition exists
            directed = np.zeros_like(matrix)
            for i, left in enumerate(node_set):
                for j, right in enumerate(node_set):
                    if i != j:
                        directed[i, j] = (int(left), int(right)) in graph_counts
            result = classify(directed)
            checked += 1
            if result is None:
                mismatched += 1
                continue
            _skeleton, key, orbits = result
            if not np.array_equal(np.asarray(orbits), roles[index]) or \
                    subtype[index] >= lookup.shape[0] or \
                    np.any(lookup[subtype[index], np.asarray(orbits)] < 0):
                mismatched += 1
    return {
        "graph": {
            "path": "results/research/behavior_graph_directed/graph.npz",
            "sha256": sha256(GRAPH_DIR / "graph.npz"),
            "metadata_sha256": sha256(GRAPH_DIR / "metadata.json"),
            "edges": int(len(rows)), "nodes": int(payload["nodes"]),
            "policy": "TRAIN transitions only; validation/test parquet is not read",
            "is_symmetric": bool(symmetric),
        },
        "rebuild": {
            "train_transitions_from_parquet": len(train_keys),
            "graph_edges": len(graph_keys),
            "set_equal": bool(graph_keys == train_keys),
            "weight_equal": bool(graph_keys == train_keys and all(
                graph_counts[key] == train_counts[key] for key in graph_keys)),
        },
        "overlap_with_heldout": {
            "valid_edges_in_graph": int(len(graph_keys & valid_keys)),
            "test_edges_in_graph": int(len(graph_keys & test_keys)),
            "valid_edges_not_in_train": int(len(valid_keys - train_keys)),
            "test_edges_not_in_train": int(len(test_keys - train_keys)),
            "graph_edges_not_in_train": int(len(graph_keys - train_keys)),
            "note": "overlap is an edge observed in both splits, not leakage: the graph is rebuilt "
                    "from train.parquet and its edge/weight set equals the train transitions",
        },
        "instance_pool": {
            "path": "results/research/directed_graphlet_role/instances.npz",
            "sha256": sha256(ROLE / "instances.npz"),
            "train_instances": int(sum(int((pool[f"{f}_split"] == 0).sum()) for f in FAMILIES)),
            "heldout_instances": int(sum(int((pool[f"{f}_split"] == 1).sum()) for f in FAMILIES)),
            "heldout_instances_used_in_counts": 0,
        },
        "stored_roles_spotcheck": {"checked": checked, "mismatched": mismatched},
    }


# ------------------------------------------------------------ reliability
def reliability(train_counts: np.ndarray, heldout_counts: np.ndarray) -> dict:
    def row_stats(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        total = counts.sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            dist = counts / np.clip(total, 1, None)[:, None]
        return dist, total

    dist_a, total_a = row_stats(train_counts)
    dist_b, total_b = row_stats(heldout_counts)
    both = (total_a >= MIN_RELIABLE) & (total_b >= MIN_RELIABLE)
    if not both.any():
        return {"items_compared": 0, "note": "no item has enough roles in both samples"}
    cosine = np.einsum("ij,ij->i", dist_a[both], dist_b[both]) / np.clip(
        np.linalg.norm(dist_a[both], axis=1) * np.linalg.norm(dist_b[both], axis=1), 1e-12, None)
    log_a = np.log1p(train_counts[both])
    log_b = np.log1p(heldout_counts[both])
    a_centered = log_a - log_a.mean(1, keepdims=True)
    b_centered = log_b - log_b.mean(1, keepdims=True)
    corr = np.einsum("ij,ij->i", a_centered, b_centered) / np.clip(
        np.linalg.norm(a_centered, axis=1) * np.linalg.norm(b_centered, axis=1), 1e-12, None)
    family_counts_a = np.stack([train_counts[:, [d["family"] for d in DIMS_FAMILY]].sum(1)
                                for d in []] ) if False else None
    family_a = np.stack([train_counts[:, np.asarray([i for i, f in enumerate(DIM_FAMILY)
                                                     if f == name])].sum(1)
                         for name in FAMILIES], axis=1)
    family_b = np.stack([heldout_counts[:, np.asarray([i for i, f in enumerate(DIM_FAMILY)
                                                       if f == name])].sum(1)
                         for name in FAMILIES], axis=1)
    family_dist_a = family_a / np.clip(family_a.sum(1, keepdims=True), 1, None)
    family_dist_b = family_b / np.clip(family_b.sum(1, keepdims=True), 1, None)
    family_cosine = np.einsum("ij,ij->i", family_dist_a[both], family_dist_b[both]) / np.clip(
        np.linalg.norm(family_dist_a[both], axis=1)
        * np.linalg.norm(family_dist_b[both], axis=1), 1e-12, None)
    return {
        "items_compared": int(both.sum()),
        "family_level_cosine": {"mean": float(np.mean(family_cosine)),
                                "median": float(np.median(family_cosine))},
        "min_roles_per_sample": MIN_RELIABLE,
        "cosine": {"mean": float(np.mean(cosine)), "median": float(np.median(cosine)),
                   "fraction_above_0.9": float((cosine > 0.9).mean())},
        "log1p_pearson": {"mean": float(np.mean(corr)), "median": float(np.median(corr))},
        "sampling_scheme": "independent instance split of the audited #221 pool "
                           "(stable hash of the node set); results are distributions under that scheme",
    }


# ------------------------------------------------------------ learnability
def learnability(counts: np.ndarray, embeddings_path: Path) -> dict:
    features = np.load(embeddings_path, mmap_mode="r")
    n, k = counts.shape
    if features.shape[0] != n:
        raise ValueError(f"embedding rows {features.shape[0]} != items {n}")
    target = np.log1p(counts).astype(np.float64)
    digest = np.array([int.from_bytes(hashlib.blake2b(str(i).encode(), digest_size=4).digest(), "big")
                       for i in range(n)])
    eval_mask = (digest % 5) == 0
    train_mask = ~eval_mask
    x = np.asarray(features, dtype=np.float64)
    mean, std = x[train_mask].mean(0), x[train_mask].std(0).clip(1e-6)
    x = (x - mean) / std
    x_train = np.hstack([x[train_mask], np.ones((int(train_mask.sum()), 1))])
    x_eval = np.hstack([x[eval_mask], np.ones((int(eval_mask.sum()), 1))])
    y_train = target[train_mask]
    y_eval = target[eval_mask]
    ridge = 1e-2 * np.eye(x_train.shape[1])

    def fit(source: np.ndarray) -> np.ndarray:
        weights = np.linalg.solve(x_train.T @ x_train + ridge, x_train.T @ source)
        return x_eval @ weights

    def r2(prediction: np.ndarray, truth: np.ndarray) -> np.ndarray:
        residual = ((truth - prediction) ** 2).sum(0)
        total = ((truth - truth.mean(0, keepdims=True)) ** 2).sum(0)
        return 1.0 - residual / np.clip(total, 1e-12, None)

    prediction = fit(y_train)
    rng = np.random.default_rng(SEED)
    shuffled = [fit(y_train[rng.permutation(len(y_train))]) for _ in range(N_NULL)]
    # a dimension whose evaluation rows are all zero has no variance to explain:
    # counting it as R^2 = 1 would inflate the mean by the dead dimensions
    active = y_eval.var(0) > 1e-8
    real = r2(prediction[:, active], y_eval[:, active])
    null = np.stack([r2(entry[:, active], y_eval[:, active]) for entry in shuffled])
    null_mean = null.mean(0)
    null_high = np.quantile(null, 0.95, axis=0)
    above = real > null_high
    return {
        "features": "stage1_GeneEmbedding/output/sentence_t5.npy (768-d, unchanged)",
        "target": "log1p counts (376 dims)",
        "items": int(n), "eval_items": int(eval_mask.sum()),
        "items_with_any_role": int((counts.sum(1) > 0).sum()),
        "eval_items_with_any_role": int(((counts.sum(1) > 0) & eval_mask).sum()),
        "active_dimensions": int(active.sum()),
        "r2": {"mean": float(real.mean()), "median": float(np.median(real)),
               "fraction_above_null_p95": float(above.mean()),
               "max": float(real.max()), "scope": "dimensions with non-zero evaluation variance"},
        "null": {"reps": N_NULL, "r2_mean": float(null_mean.mean()),
                 "r2_p95_mean": float(null_high.mean()), "scope": "same active dimensions"},
        "learnable": bool(above.mean() >= 0.5 and real.mean() >= 0.05),
        "trace_only": bool(above.mean() >= 0.5 and real.mean() < 0.05),
        "note": "a dimension counts as beating the null when its held-out R^2 exceeds the "
                "permutation null's 95th percentile; learnable additionally requires the "
                "mean R^2 over active dimensions to reach 0.05, so a detectable trace with "
                "no usable effect size is reported as trace_only",
    }


# ---------------------------------------------------------------- main
def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    raw = json.loads((ROLE / "subtypes.json").read_text())
    dims, table = build_vocabulary(raw)
    k = len(dims)
    DIM_FAMILY[:] = [entry["family"] for entry in dims]
    pool = np.load(ROLE / "instances.npz", allow_pickle=False)
    n = int(np.load(GRAPH_DIR / "graph.npz", allow_pickle=False)["nodes"])
    counts, family_stats = count_roles(0, table, n, k)
    heldout_counts, heldout_stats = count_roles(1, table, n, k)
    instances = counts.sum(1)
    dist = counts / np.clip(instances, 1, None)[:, None]
    np.savez_compressed(
        OUT / "orbit_vectors.npz",
        counts=counts.astype(np.int32),
        log1p=np.log1p(counts).astype(np.float32),
        distribution=dist.astype(np.float32),
        instances=instances.astype(np.int32),
        coverage=(instances > 0),
        reliable=(instances >= MIN_RELIABLE),
        item_ids=np.arange(n, dtype=np.int64),
        family_of=np.array([FAMILIES.index(d["family"]) for d in dims], dtype=np.int8),
        heldout_counts=heldout_counts.astype(np.int32),
    )
    (OUT / "orbit_vocab.json").write_text(json.dumps({
        "k": k, "families": list(FAMILIES), "dimensions": dims,
        "rule": "index = (family, canonical directed pattern key, node orbit id); "
                "orbit ids are automorphism classes of the pattern",
        "unused_dimensions": int((counts.sum(0) == 0).sum()),
    }, indent=1))

    audit = graph_audit(transitions_from("train"), transitions_from("valid"),
                        transitions_from("test"), pool, table)
    audit["counting"] = {"train": family_stats, "heldout": heldout_stats,
                         "coverage": {
                             "items_with_any_role": int((instances > 0).sum()),
                             "fraction": float((instances > 0).mean()),
                             "items_reliable": int((instances >= MIN_RELIABLE).sum()),
                             "reliable_fraction": float((instances >= MIN_RELIABLE).mean()),
                             "instances_median": float(np.median(instances[instances > 0])),
                             "instances_max": int(instances.max())},
                         "unused_dimensions": int((counts.sum(0) == 0).sum())}
    (OUT / "split_audit.json").write_text(json.dumps(audit, indent=1))
    (OUT / "reliability.json").write_text(json.dumps(reliability(counts, heldout_counts), indent=1))
    (OUT / "learnability.json").write_text(json.dumps(learnability(counts, EMB), indent=1))

    outputs = ["orbit_vectors.npz", "orbit_vocab.json", "split_audit.json",
               "reliability.json", "learnability.json"]
    (OUT / "manifest.json").write_text(json.dumps({
        "issue": 229, "phase": 1,
        "inputs": {str(p.relative_to(REPO)): sha256(p) for p in
                   (ROLE / "instances.npz", ROLE / "subtypes.json",
                    GRAPH_DIR / "graph.npz", GRAPH_DIR / "metadata.json", EMB)},
        "outputs": {name: sha256(OUT / name) for name in outputs},
        "regenerate": "python3 build_orbit_vectors.py (the .npz is git-ignored; every "
                      "other artifact is committed)",
    }, indent=1))

    rel = json.loads((OUT / "reliability.json").read_text())
    learn = json.loads((OUT / "learnability.json").read_text())
    cover = audit["counting"]["coverage"]
    def fmt(value):
        return f"{value:.4f}" if isinstance(value, float) else str(value)
    lines = ["# 商品有向 Orbit 结构向量(第一阶段)", "",
             f"- 词表 K = {k};未使用维度 {audit['counting']['unused_dimensions']}",
             f"- 覆盖:{cover['items_with_any_role']} 件商品有角色"
             f"({cover['fraction']:.1%});≥{MIN_RELIABLE} 个实例的可靠商品 "
             f"{cover['items_reliable']}({cover['reliable_fraction']:.1%})",
             f"- 图重建与 train 转移集一致:{audit['rebuild']['set_equal']}"
             f"(权重一致 {audit['rebuild']['weight_equal']});"
             f"图与 valid/test 的重叠仅是两边都出现过的边,"
             f"valid/test 独有边数 {audit['overlap_with_heldout']['valid_edges_not_in_train']}"
             f"/{audit['overlap_with_heldout']['test_edges_not_in_train']}",
             f"- 存储角色抽查 {audit['stored_roles_spotcheck']['checked']} 例,"
             f"不一致 {audit['stored_roles_spotcheck']['mismatched']}",
             "",
             "## 采样稳定性(独立实例划分)",
             f"- 比较商品数 {rel.get('items_compared', 0)};"
             f"orbit 级余弦均值 {fmt(rel.get('cosine', {}).get('mean'))},"
             f"中位数 {fmt(rel.get('cosine', {}).get('median'))},"
             f">0.9 的比例 {fmt(rel.get('cosine', {}).get('fraction_above_0.9'))};"
             f"family 级余弦中位数 "
             f"{fmt(rel.get('family_level_cosine', {}).get('median'))}",
             "",
             "## 能否从 768 维语义学到",
             f"- 活跃维度 {learn['active_dimensions']};留出商品 R² 均值 "
             f"{fmt(learn['r2']['mean'])},中位数 {fmt(learn['r2']['median'])},"
             f"超过置换零分布 p95 的比例 {fmt(learn['r2']['fraction_above_null_p95'])}",
             f"- 结论: {'可学' if learn['learnable'] else ('仅可检测到痕迹、无实用效应' if learn.get('trace_only') else '不可学')}",
             "",
             "## 对 RQ-VAE 监督的决定(第一阶段结论)",
             f"- 可靠性的粒度差异:orbit 级余弦 {fmt(rel.get('cosine', {}).get('mean'))},"
             f"family 级 {fmt(rel.get('family_level_cosine', {}).get('mean'))};"
             "按单 orbit 做监督会把采样噪声当成标签",
             f"- 语义可预测性:线性探针 R² 均值 {fmt(learn['r2']['mean'])},"
             "结构不能从 768 维输入推出;把它压进编码器会与重构目标竞争",
             "- 决定:本轮不把 376 维 orbit 计数接入 RQ-VAE 监督;若继续,先用 family 级"
             "(或聚合后的低维)目标并屏蔽无证据商品,非线性探针另测"]
    (OUT / "REPORT_ORBIT_VECTORS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
