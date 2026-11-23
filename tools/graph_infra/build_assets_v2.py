"""Build the v2 frozen assets (issue #246).

Stages: ``graph``, ``ego``, ``motif``, ``pairs``, ``distance``, ``curvature``.
Run without arguments to build everything; the manifest is written last and verified.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from tools.graph_infra import assets, config, graph_build, graphlet, isolation, metrics  # noqa: E402
from tools.graph_infra import pairs, splits, topology  # noqa: E402

ROOT = config.ASSET_ROOT
COMMAND = "python3 tools/graph_infra/build_assets_v2.py"


def _sha(path) -> str:
    return assets.sha256(Path(path))


def code_sha256() -> str:
    files = sorted((REPO / "tools" / "graph_infra").glob("*.py"))
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode())
        digest.update(assets.sha256(path).encode())
    return digest.hexdigest()


def source_hashes() -> dict:
    return {"items": _sha(config.ITEMS), "train": _sha(config.TRAIN),
            "embeddings": _sha(config.EMBEDDINGS)}


def graph_version(split: str) -> str:
    return f"{config.VERSION}/{split}/{source_hashes()['train'][:12]}"


def _entry(name, path, *, schema, shape, dtype, split, legal_use, purpose, dependencies,
           extra=None):
    return assets.entry_v2(
        name, path, repo=REPO, schema=schema, shape=shape, dtype=dtype,
        graph_version=graph_version(split), legal_use=legal_use, version=config.VERSION,
        purpose=purpose, command=COMMAND, source_sha256=source_hashes()["train"],
        code_sha256=code_sha256(), dependencies=dependencies, extra=extra)


# ------------------------------------------------------------------ graph stage
def build_graph() -> dict:
    info = graph_build.item_index()
    sequences = graph_build.sequences_from_parquet(config.TRAIN)
    users = np.array(sorted(sequences), dtype=np.int64)
    split = splits.three_way(users, config.SPLIT_SEED, config.SPLIT_RATIOS)
    audit = splits.audit(split, users)
    if not (audit["disjoint"] and audit["union_complete"]):
        raise SystemExit(f"split audit failed: {audit}")

    def counts_for(name):
        chosen = split[name]
        return graph_build.count_edges([sequences[int(u)] for u in chosen.tolist()])

    full = graph_build.count_edges([sequences[int(u)] for u in users.tolist()])
    visible = counts_for("visible")
    struct_val = counts_for("struct_val")
    struct_test = counts_for("struct_test")
    total = visible.total() + struct_val.total() + struct_test.total()
    if total != full.total():
        raise SystemExit(f"split edge totals do not add up: {total} vs {full.total()}")

    n = int(info["items"])
    graphs = {"full_description": full, "visible": visible,
              "struct_val": struct_val, "struct_test": struct_test}
    payload = {}
    for name, counts in graphs.items():
        csr = counts.to_csr(n)
        payload[name] = {"forward": csr, "reverse": csr.T.tocsr(),
                         "undirected": graph_distance_undirected(csr)}
    evaluation = (graphs["struct_val"].to_csr(n) + graphs["struct_test"].to_csr(n)).tocsr()
    held = isolation.held_out_support(payload["visible"]["forward"], evaluation)

    graph_dir = ROOT / "graph"
    graph_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for name, block in payload.items():
        for kind, matrix in block.items():
            path = graph_dir / f"{name}_{kind}.npz"
            assets.save_sparse(path, matrix)
            entries.append(_entry(
                f"graph_{name}_{kind}", path, schema="csr.int64", shape=list(matrix.shape),
                dtype="int64", split=name,
                legal_use="description" if name == "full_description" else "generalisation",
                purpose=f"{kind} graph of the {name} split",
                dependencies=["train.parquet"]))
    held_path = graph_dir / "held_out_support.npz"
    assets.save_sparse(held_path, held)
    entries.append(_entry("held_out_support", held_path, schema="csr.bool", shape=list(held.shape),
                          dtype="int8", split="visible", legal_use="generalisation",
                          purpose="undirected support of edges held out from the visible graph "
                                  "(struct_val ∪ struct_test minus visible), both directions",
                          dependencies=["train.parquet"]))
    users_path = graph_dir / "split_users.npz"
    np.savez_compressed(users_path, **{name: split[name] for name in splits.NAMES})
    entries.append(_entry("split_users", users_path, schema="npz.split.v1",
                          shape=[len(split[name]) for name in splits.NAMES], dtype="int64",
                          split="visible", legal_use="generalisation",
                          purpose="real user ids of the three train splits",
                          dependencies=["train.parquet"],
                          extra={"seed": config.SPLIT_SEED, "ratios": list(config.SPLIT_RATIOS)}))
    stats = {
        "items": n, "users": int(len(users)), "split": audit,
        "edges": {name: {"distinct": len(counts), "weight": counts.total()}
                  for name, counts in graphs.items()},
        "self_loops_dropped": int(graph_build.self_loops(
            [sequences[int(u)] for u in users.tolist()])),
        "held_out_support_entries": int(held.nnz),
        "graph_stats": {name: assets.graph_stats(block["forward"])
                        for name, block in payload.items()},
        "independent_crosscheck": independent_edge_crosscheck(sequences, payload["full_description"]["forward"]),
    }
    (graph_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    entries.append(_entry("graph_stats", graph_dir / "stats.json", schema="json",
                          shape=[len(stats)], dtype="json", split="visible",
                          legal_use="generalisation",
                          purpose="counts, split audit, self loops and the independent cross-check",
                          dependencies=["train.parquet"]))
    return {"entries": entries, "split": split, "graphs": payload, "held": held,
            "sequences": sequences, "items": n, "stats": stats}


def graph_distance_undirected(csr: sp.spmatrix) -> sp.csr_matrix:
    from tools.graph_infra import graph_distance
    return graph_distance.undirected_projection(csr)


def independent_edge_crosscheck(sequences, forward: sp.csr_matrix) -> dict:
    """Count edges a second time, by a different route, and compare edge sets exactly."""
    counted: dict = {}
    for user in sequences:
        sequence = sequences[user]
        for left, right in zip(sequence[:-1], sequence[1:]):
            left, right = int(left), int(right)
            if left == right:
                continue
            counted[(left, right)] = counted.get((left, right), 0) + 1
    coo = forward.tocoo()
    csr_edges = {(int(a), int(b)): int(w) for a, b, w in zip(coo.row, coo.col, coo.data)}
    only_reference = set(counted) - set(csr_edges)
    only_csr = set(csr_edges) - set(counted)
    weight_mismatch = [(edge, counted[edge], csr_edges[edge]) for edge in set(counted) & set(csr_edges)
                       if counted[edge] != csr_edges[edge]]
    return {"reference_edges": len(counted), "csr_edges": len(csr_edges),
            "only_reference": len(only_reference), "only_csr": len(only_csr),
            "weight_mismatch": len(weight_mismatch),
            "ok": not only_reference and not only_csr and not weight_mismatch}


# ------------------------------------------------------------------ ego stage
def _collect_ego(roots, out_csr, in_csr, mode: str, cap: int) -> dict:
    node_parts, edge_parts, hop_parts, depth_parts = [], [], [], []
    role_out_parts, role_in_parts = [], []
    root_local, empty = [], []
    node_offsets = [0]
    edge_offsets = [0]
    for root in roots:
        if mode == "full1":
            block = topology.ego_full(int(root), out_csr, in_csr, radius=1)
        else:
            rng = np.random.default_rng([config.EGO_SEED, int(root)])
            block = topology.ego_induced(int(root), out_csr, in_csr, rng,
                                         radius=2, cap=cap)
        node_parts.append(block["nodes"])
        edge_parts.append(block["edges"])
        hop_parts.append(block["hop"].astype(np.int8))
        depth_parts.append(block["traversal_depth"].astype(np.int8))
        role_out_parts.append(block["role_out"])
        role_in_parts.append(block["role_in"])
        root_local.append(block["root_local"])
        empty.append(block["empty_neighbourhood"])
        node_offsets.append(node_offsets[-1] + len(block["nodes"]))
        edge_offsets.append(edge_offsets[-1] + len(block["edges"]))
    return {
        "nodes": np.concatenate(node_parts) if node_parts else np.zeros(0, dtype=np.int64),
        "edges": np.concatenate(edge_parts).reshape(-1, 2) if edge_parts else np.zeros((0, 2), np.int64),
        "hop": np.concatenate(hop_parts) if hop_parts else np.zeros(0, dtype=np.int8),
        "traversal_depth": np.concatenate(depth_parts) if depth_parts else np.zeros(0, np.int8),
        "role_out": np.concatenate(role_out_parts) if role_out_parts else np.zeros(0, bool),
        "role_in": np.concatenate(role_in_parts) if role_in_parts else np.zeros(0, bool),
        "root_local": np.array(root_local, dtype=np.int64),
        "empty": np.array(empty, dtype=bool),
        "roots": np.array(roots, dtype=np.int64),
        "node_offsets": np.array(node_offsets, dtype=np.int64),
        "edge_offsets": np.array(edge_offsets, dtype=np.int64),
        "cap": cap, "cap_includes_root": False, "mode": mode,
    }


def build_ego(graph: dict) -> dict:
    n = graph["items"]
    visible = graph["graphs"]["visible"]["forward"]
    reverse = graph["graphs"]["visible"]["reverse"]
    full = graph["graphs"]["full_description"]["forward"]
    full_reverse = graph["graphs"]["full_description"]["reverse"]
    ego_dir = ROOT / "ego"
    ego_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    roots = np.arange(n)
    variants = (
        ("ego_full_1hop_visible", visible, reverse, "full1", None, "generalisation"),
        ("ego_sampled_visible", visible, reverse, "sampled", config.EGO_CAP, "generalisation"),
        ("ego_sampled_full_description", full, full_reverse, "sampled", config.EGO_CAP, "description"),
    )
    report = {}
    for name, out_csr, in_csr, mode, cap, use in variants:
        started = time.time()
        block = _collect_ego(roots, out_csr, in_csr, mode, cap)
        path = ego_dir / f"{name}.npz"
        np.savez_compressed(path, **block)
        sizes = np.diff(block["node_offsets"])
        report[name] = {
            "blocks": int(len(roots)), "mean_nodes": float(sizes.mean()),
            "max_nodes": int(sizes.max()), "empty_fraction": float(block["empty"].mean()),
            "edges": int(len(block["edges"])), "seconds": round(time.time() - started, 1),
        }
        entries.append(_entry(
            name, path, schema="ego.offset.v2", shape=[len(block["nodes"])], dtype="mixed",
            split="visible" if "visible" in name else "full_description", legal_use=use,
            purpose=f"{mode} ego blocks: nodes, induced local edges, true hop, traversal depth, "
                    f"real role masks, explicit root_local (cap={cap} neighbours, root extra)",
            dependencies=[f"graph_{'visible' if 'visible' in name else 'full_description'}_forward"],
            extra={"cap": cap, "mean_nodes": report[name]["mean_nodes"],
                   "max_nodes": report[name]["max_nodes"]}))
    (ego_dir / "ego_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    entries.append(_entry("ego_report", ego_dir / "ego_report.json", schema="json",
                          shape=[len(report)], dtype="json", split="visible",
                          legal_use="generalisation",
                          purpose="ego block sizes, coverage and empty fractions",
                          dependencies=["ego_sampled_visible"]))
    return {"entries": entries, "report": report}


# ------------------------------------------------------------------ motif stage
def build_motif(graph: dict) -> dict:
    visible = graph["graphs"]["visible"]["forward"]
    reverse = graph["graphs"]["visible"]["reverse"]
    motif_dir = ROOT / "motifs"
    motif_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    counted = graphlet.family_counts(visible, reverse)
    names = counted["names"]
    counts = counted["counts"]
    covered = int((counts.sum(axis=1) > 0).sum())
    centres = stratified_centres(visible, reverse, 2000)
    sampled = graphlet.sampled_instances(visible, reverse, centres, 40, config.EGO_SEED + 11,
                                         graph_version=graph_version("visible"))
    four = []
    for centre in stratified_centres(visible, reverse, 500, seed=config.EGO_SEED + 12):
        four.extend(graphlet.four_node_instances(centre, visible, reverse, 10,
                                                 seed=config.EGO_SEED + 13 + int(centre)))
    stats_path = motif_dir / "motif_counts.npz"
    np.savez_compressed(stats_path, counts=counts, names=np.array(names),
                        instances=counted["instances"])
    path = motif_dir / "motif_instances_visible.npz"
    _save_instances(path, sampled)
    four_path = motif_dir / "motif_instances_4node_visible.npz"
    _save_instances(four_path, four)
    total_pairs = int(counted["instances"].sum())
    report = {
        "families": names,
        "totals": {name: int(counts[:, index].sum()) for index, name in enumerate(names)},
        "coverage": covered / graph["items"],
        "instances_total": total_pairs,
        "sampling": "3-node counts are exact over every neighbour pair; instances are sampled",
        "instance_sample": {"centres": len(centres), "max_per_centre": 40,
                            "instances": len(sampled),
                            "coverage": len(centres) / graph["items"]},
        "four_node_sample": {"centres": 500, "max_instances_per_centre": 10,
                             "instances": len(four), "sampling": "sampled, not full"},
        "seconds": round(time.time() - started, 1),
    }
    (motif_dir / "motif_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    entries = [
        _entry("motif_counts", stats_path, schema="npz.motif_counts.v2", shape=list(counts.shape),
               dtype="int64", split="visible", legal_use="generalisation",
               purpose="exact 3-node family counts per centre over every neighbour pair",
               dependencies=["graph_visible_forward"]),
        _entry("motif_instances_visible", path, schema="npz.instances.v2", shape=[len(sampled)],
               dtype="mixed", split="visible", legal_use="generalisation",
               purpose="sampled 3-node instances with node ids, induced edges, graphlet id, orbit",
               dependencies=["graph_visible_forward"],
               extra={"coverage": len(centres) / graph["items"]}),
        _entry("motif_instances_4node_visible", four_path, schema="npz.instances.v2",
               shape=[len(four)], dtype="mixed", split="visible", legal_use="generalisation",
               purpose="sampled 4-node instances (declared sampling, not full)",
               dependencies=["graph_visible_forward"],
               extra={"coverage": 500 / graph["items"]}),
        _entry("motif_report", motif_dir / "motif_report.json", schema="json", shape=[len(names)],
               dtype="json", split="visible", legal_use="generalisation",
               purpose="family totals, coverage and the sampling statement",
               dependencies=["motif_counts"]),
    ]
    return {"entries": entries, "report": report}


def _save_instances(path: Path, records: list) -> None:
    if not records:
        np.savez_compressed(path, centers=np.zeros(0, np.int64), nodes=np.zeros((0, 3), np.int64))
        return
    vertices = len(records[0]["node_ids"])
    centers = np.array([record["center"] for record in records], dtype=np.int64)
    nodes = np.array([record["node_ids"] for record in records], dtype=np.int64)
    codes = np.array([record["code"] for record in records], dtype=np.int64)
    graphlets = np.array([record["graphlet_id"] for record in records], dtype=np.int64)
    family_names = graphlet.family_names() if vertices == 3 else []
    if vertices == 3:
        slots = np.array([family_names.index(record["family"]) for record in records], dtype=np.int64)
    else:
        slots = np.full(len(records), -1, dtype=np.int64)
    orbits = np.array([record.get("orbit", -1) for record in records], dtype=np.int64)
    roles = np.array([record["center_role"] for record in records], dtype=np.int64)
    edge_parts, edge_offsets = [], [0]
    for record in records:
        edge_parts.append(np.array(record["edges"], dtype=np.int64).reshape(-1, 2))
        edge_offsets.append(edge_offsets[-1] + len(record["edges"]))
    np.savez_compressed(path, centers=centers, nodes=nodes, codes=codes, graphlets=graphlets,
                        families=slots, family_names=np.array(family_names), orbits=orbits,
                        center_roles=roles,
                        edges=np.concatenate(edge_parts) if edge_parts else np.zeros((0, 2), np.int64),
                        edge_offsets=np.array(edge_offsets, dtype=np.int64),
                        vertices=np.array(vertices))


def stratified_centres(csr: sp.spmatrix, reverse: sp.spmatrix, count: int, seed: int = 246050) -> np.ndarray:
    degree = np.asarray(csr.getnnz(axis=1) + reverse.getnnz(axis=1)).ravel()
    order = np.argsort(degree)
    chunks = np.array_split(order, max(1, count // 10))
    rng = np.random.default_rng(seed)
    picked = []
    per_chunk = max(1, count // max(1, len(chunks)))
    for chunk in chunks:
        if len(chunk) == 0:
            continue
        take = min(per_chunk, len(chunk))
        picked.extend(rng.choice(chunk, size=take, replace=False).tolist())
    return np.array(sorted(picked), dtype=np.int64)


STAGES = ("graph", "ego", "motif", "pairs", "curvature")

def load_graph_state() -> dict:
    """Rebuild the graph state from the frozen files, so later stages can run alone."""
    graph_dir = ROOT / "graph"
    graphs = {}
    for name in ("full_description", "visible", "struct_val", "struct_test"):
        block = {}
        for kind in ("forward", "reverse", "undirected"):
            block[kind] = assets.load_sparse(graph_dir / f"{name}_{kind}.npz")
        graphs[name] = block
    held = assets.load_sparse(graph_dir / "held_out_support.npz")
    blob = np.load(graph_dir / "split_users.npz")
    split = {name: blob[name] for name in splits.NAMES}
    stats = json.loads((graph_dir / "stats.json").read_text(encoding="utf-8"))
    return {"graphs": graphs, "held": held, "split": split, "items": int(stats["items"]),
            "sequences": graph_build.sequences_from_parquet(config.TRAIN), "stats": stats}


# ------------------------------------------------------------------ pair banks + features
def _bank_groups(forward, reverse, n, rng, per_group):
    from tools.graph_infra import pairs as pair_ops
    return {
        "direct": pair_ops.direct_pairs(forward, rng, per_group),
        "shared_out": pair_ops.shared_successor_pairs(forward, n, rng, per_group),
        "shared_in": pair_ops.shared_predecessor_pairs(reverse, n, rng, per_group),
        "two_hop": pair_ops.two_hop_pairs(forward, n, rng, per_group),
        "degree_matched": pair_ops.degree_matched_pairs(forward, reverse, n, rng, per_group),
    }


def _visible_support(forward):
    return ((forward + forward.T) > 0).astype(np.int8).tocsr()


def build_pairs(graph: dict) -> dict:
    from tools.graph_infra import bank_features, pairs as pair_ops
    n = graph["items"]
    visible = graph["graphs"]["visible"]["forward"]
    reverse = graph["graphs"]["visible"]["reverse"]
    undirected = graph["graphs"]["visible"]["undirected"]
    held = graph["held"]
    support = _visible_support(visible)
    pair_dir = ROOT / "pairs"
    pair_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    rng = np.random.default_rng(config.EGO_SEED + 21)
    per_group = config.PER_GROUP

    def unseen(pairs):
        keep = [(int(a), int(b)) for a, b in np.asarray(pairs).tolist()
                if support[a, b] == 0 and support[b, a] == 0]
        return np.array(keep, dtype=np.int64).reshape(-1, 2)

    def seen(pairs):
        keep = [(int(a), int(b)) for a, b in np.asarray(pairs).tolist()
                if support[a, b] > 0 or support[b, a] > 0]
        return np.array(keep, dtype=np.int64).reshape(-1, 2)

    strict = {}
    relaxed = {}
    for split_name, limit in (("struct_val", 3000), ("struct_test", 3000)):
        forward = graph["graphs"][split_name]["forward"]
        back = graph["graphs"][split_name]["reverse"]
        groups = _bank_groups(forward, back, n, np.random.default_rng(config.EGO_SEED + 31), limit)
        strict[split_name] = {name: unseen(pairs) for name, pairs in groups.items()}
        relaxed[split_name] = {name: seen(pairs) for name, pairs in groups.items()}
    strict_keys = set()
    for group in strict.values():
        for produced in group.values():
            strict_keys |= isolation.pair_keys(produced)
    blocked_matrix = held.copy().tolil()
    for left, right in strict_keys:
        blocked_matrix[left, right] = 1
        blocked_matrix[right, left] = 1
    blocked_matrix = blocked_matrix.tocsr().astype(np.int8)

    train = {}
    over = per_group * 2
    groups = _bank_groups(visible, reverse, n, rng, over)
    for name, produced in groups.items():
        kept = pair_ops.exclude_undirected(produced, blocked_matrix)[:per_group]
        train[name] = kept
        pair_ops.verify_group(visible, reverse, kept, name)
    probe = pair_ops.probe_bank(visible, points=64)["pairs"]
    probe = pair_ops.exclude_undirected(probe, blocked_matrix)
    # the probe must not repeat a training pair either
    taken = np.zeros((n, n), dtype=np.int8)
    for produced in train.values():
        for left, right in produced.tolist():
            taken[left, right] = 1
            taken[right, left] = 1
    probe = pair_ops.exclude_undirected(probe, sp.csr_matrix(taken))

    # fail fast: the per-bank call path is exercised on a one-pair probe before the real work
    bank_features.compute(np.array([[0, 1]], dtype=np.int64), visible, reverse, undirected,
                          with_ppr=False)
    report = {"train": {}, "strict": {}, "relaxed": {}, "probe": int(len(probe)),
              "blocked_undirected": int(blocked_matrix.nnz // 2)}
    for name, produced in train.items():
        fields = bank_features.compute(produced, visible, reverse, undirected, with_ppr=False)
        fields.update(bank_features.witness_fields(visible, reverse, produced, name))
        path = pair_dir / f"train_{name}.npz"
        np.savez_compressed(path, pairs=produced, group=np.array(name),
                            **{key: value for key, value in fields.items() if key != "ppr_meta"})
        report["train"][name] = {"pairs": int(len(produced)),
                                 "reachable": float(fields["reachable_undirected"].mean())}
        entries.append(_entry(f"bank_train_{name}", path, schema="bank.v2",
                              shape=[int(len(produced)), 2], dtype="mixed", split="visible",
                              legal_use="generalisation",
                              purpose=f"{name} training bank with per-pair graph features and witnesses",
                              dependencies=["graph_visible_forward", "held_out_support"],
                              extra={"pairs": int(len(produced))}))
    for kind, container in (("strict", strict), ("relaxed", relaxed)):
        for split_name, groups in container.items():
            for name, produced in groups.items():
                if len(produced) == 0:
                    report[kind].setdefault(split_name, {})[name] = {"pairs": 0}
                    continue
                fields = bank_features.compute(produced, visible, reverse, undirected, with_ppr=False)
                fields.update(bank_features.witness_fields(visible, reverse, produced, name))
                path = pair_dir / f"{kind}_{split_name}_{name}.npz"
                np.savez_compressed(path, pairs=produced, group=np.array(name),
                                    **{key: value for key, value in fields.items() if key != "ppr_meta"})
                report[kind].setdefault(split_name, {})[name] = {
                    "pairs": int(len(produced)),
                    "reachable": float(fields["reachable_undirected"].mean())}
                entries.append(_entry(
                    f"bank_{kind}_{split_name}_{name}", path, schema="bank.v2",
                    shape=[int(len(produced)), 2], dtype="mixed",
                    split="visible" if kind == "relaxed" else split_name,
                    legal_use="evaluation",
                    purpose=f"{kind} evaluation bank on the {split_name} graph "
                            f"({'held out from the visible graph' if kind == 'strict' else 'user-isolated, edge seen by the visible graph'})",
                    dependencies=["graph_visible_forward", "held_out_support"],
                    extra={"pairs": int(len(produced))}))
    probe_path = pair_dir / "probe.npz"
    np.savez_compressed(probe_path, pairs=probe, graph_hops=np.ones(len(probe), dtype=np.int64),
                        seed=np.array(config.EGO_SEED), alpha=np.array(config.PPR_RESTART))
    entries.append(_entry("gradient_probe", probe_path, schema="npz.probe.v2",
                          shape=[int(len(probe)), 2], dtype="int64", split="visible",
                          legal_use="generalisation",
                          purpose="gradient probe; every pair is outside the blocked set",
                          dependencies=["graph_visible_forward"]))
    (pair_dir / "banks_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    entries.append(_entry("banks_report", pair_dir / "banks_report.json", schema="json",
                          shape=[len(report)], dtype="json", split="visible",
                          legal_use="generalisation", purpose="bank sizes, PPR residuals and coverage",
                          dependencies=["bank_train_direct"]))
    return {"entries": entries, "report": report, "blocked": blocked_matrix}


# ------------------------------------------------------------------ curvature
def build_curvature(graph: dict) -> dict:
    from tools.graph_infra import curvature
    undirected = graph["graphs"]["visible"]["undirected"]
    directory = ROOT / "curvature"
    directory.mkdir(parents=True, exist_ok=True)
    started = time.time()
    nodes = stratified_centres(undirected, undirected, config.DISTANCE_ROOTS,
                               seed=config.EGO_SEED + 41)
    result = curvature.node_curvature(undirected, nodes, max_edges_per_node=3,
                                      seed=config.EGO_SEED + 42)
    values = [record["mean"] for record in result["nodes"].values()]
    summary = curvature.summarise(values)
    missing = sum(1 for record in result["nodes"].values() if record["mean"] is None)
    path = directory / "curvature_visible.npz"
    np.savez_compressed(path,
                        nodes=np.array(sorted(result["nodes"]), dtype=np.int64),
                        mean=np.array([result["nodes"][node]["mean"] if result["nodes"][node]["mean"] is not None
                                       else np.nan for node in sorted(result["nodes"])]),
                        edges=np.array([result["nodes"][node]["edges"] for node in sorted(result["nodes"])],
                                       dtype=np.int64))
    payload = {"summary": summary, "missing_nodes": missing,
               "reference_nodes": int(len(nodes)), "sampled_edges": result["sampled_edges"],
               "coverage": len(nodes) / graph["items"],
               "unit": "dimensionless (1 - W1/hop)",
               "sampling": "stratified reference nodes, up to 3 incident edges each; not full",
               "seconds": round(time.time() - started, 1)}
    (directory / "curvature_visible.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {"report": payload, "entries": [
        _entry("curvature_visible", path, schema="npz.curvature.v2",
               shape=[int(len(nodes))], dtype="mixed", split="visible",
               legal_use="generalisation",
               purpose="exact Ollivier-Ricci on the visible graph for frozen reference nodes",
               dependencies=["graph_visible_undirected"],
               extra={"coverage": payload["coverage"], "missing": missing}),
        _entry("curvature_report", directory / "curvature_visible.json", schema="json",
               shape=[len(payload)], dtype="json", split="visible", legal_use="generalisation",
               purpose="curvature distribution, unit, sampling coverage and missing count",
               dependencies=["curvature_visible"]),
    ]}



def main(argv) -> int:
    stages = tuple(argv[1:]) or STAGES
    started = time.time()
    ROOT.mkdir(parents=True, exist_ok=True)
    state = {}
    entries = []
    if "graph" in stages:
        state.update(build_graph())
        entries.extend(state["entries"])
    elif set(stages) & {"pairs", "curvature"}:
        state.update(load_graph_state())
    if "ego" in stages:
        state.update(build_ego(state))
        entries.extend(state["entries"])
    if "motif" in stages:
        state.update(build_motif(state))
        entries.extend(state["entries"])
    if "pairs" in stages:
        state.update(build_pairs(state))
        entries.extend(state["entries"])
    if "curvature" in stages:
        state.update(build_curvature(state))
        entries.extend(state["entries"])
    manifest = ROOT / "manifest.json"
    if manifest.exists() and set(stages) != set(STAGES):
        previous = assets.load_manifest(manifest)["assets"]
        names = {entry["name"] for entry in entries}
        entries = [entry for entry in previous if entry["name"] not in names] + entries
    if set(stages) == set(STAGES):
        assets.write_manifest_v2(manifest, entries, issue=246,
                                 notes="v2 assets built from config.VERSION; v1 stays untouched at "
                                       "results/research/infra245/assets")
        audit = assets.audit_manifest(manifest)
        if not audit["ok"]:
            raise SystemExit(f"manifest audit failed: {audit['issues'][:5]}")
        verified = assets.verify_manifest(manifest)
        print(json.dumps({"stages": stages, "assets": len(entries),
                          "verified": len(verified["verified"]),
                          "seconds": round(time.time() - started, 1)}, indent=2))
    else:
        print(json.dumps({"stages": stages, "assets": len(entries),
                          "seconds": round(time.time() - started, 1)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
