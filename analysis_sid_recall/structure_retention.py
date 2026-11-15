"""How much of a role's behaviour edges survive as a shared SID prefix, per arm.

What this measures, stated plainly: ORCA plus NMF put every item into one of six roles, and a
behaviour edge is attributed to the role of its history item; the table then asks whether the two ends
of that edge share the level-1, level-2 or all three codes.  It is *not* the per-graphlet distance test
-- that lives in ``analysis.py patterns`` -- and it does not subtract a random baseline.  The question
here is narrower: do different structural roles keep their behaviour connections differently, and does
that pattern track the arm's recall.

Does a structure's retention predict the arm's Stage 3 recall?

Every archived pair in `r2/` carries a SID table and a recall.  This script asks a
narrower question than `analyze.py` does: not "how well does the SID keep behaviour
in general", but "does the STRUCTURE of the local behaviour graph -- the four-node
graphlets an item sits in, from ORCA -- survive equally well in every arm, and does
the pattern of survival track the downstream number".

Structure, fixed once for all arms and independent of any SID:

  * ORCA's 15 node orbits of graphlets up to four nodes, computed on the undirected
    TRAIN behaviour graph (`results/research/issue213/orca/node4.out`);
  * six roles from NMF over those orbits plus four cheap invariants -- the same
    role assignment `results/research/issue213/graphlet_roles.py` reports, reused
    here through `features_and_roles` so the two analyses cannot drift apart;
  * the assignment is cached in `structure_roles5.npz` (or `structure_roles.npz` for four-node
    orbits) behind a fingerprint of the graph, the orbit table and the overlap mode, because a cache
    keyed only on the node count can silently survive a graph change;
  * items with no behaviour edge (31 of 24,587 in the current graph) have no role and never appear in
    an edge, so they cannot enter this table; the count is written into the output.

Retention of one pair, per role:

    prefix@1   both endpoints share the level-1 code
    prefix@2   they share levels 1 and 2
    prefix@3   they share all three codes (the full semantic triple)

    Only the same SID position is ever compared: a level-1 code of one item is never counted as a
    match for the level-2 code of the other.

An edge is attributed to the role of its first endpoint, the history item, exactly
as the 213 audit does, so the two tables can be read side by side.

The correlation table is Spearman rho of each retention series against
`test_recall@10` across pairs, with a percentile bootstrap interval over pairs
(2000 resamples, seed 20261109 -- the same seed and resamples as `analyze.py`).
n is printed next to every row: 56 pairs with a recall are 56 arms from 56
families, not a population, and no row here is an admission gate (Stage 3
`test_recall@10` stays the only verdict, CLAUDE.md §19/§25).

    python3 analysis_sid_recall/structure_retention.py
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import random
import sys
from pathlib import Path

import networkx as nx
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
ROLES_MODULE = REPO / "results" / "research" / "issue213" / "graphlet_roles.py"
RESULTS = HERE / "results"
REPORT = RESULTS / "report.json"
LEDGER = REPO / "results" / "research" / "issue139" / "r2_ledger.json"
MIRROR = HERE / "r2"
ROLE_CACHE = HERE / "structure_roles.npz"
ROLE_CACHE5 = HERE / "structure_roles5.npz"
ROLE_CACHE_4NODE_REFIT = HERE / "structure_roles_4node_refit.npz"
OUT_JSON = RESULTS / "structure_retention.json"
OUT_MD = RESULTS / "STRUCTURE_RETENTION.md"
OUT_JSON5 = RESULTS / "structure_retention_5node.json"
OUT_MD5 = RESULTS / "STRUCTURE_RETENTION_5NODE.md"

RECALL = "test_recall@10"
#: Above this level-1 share the level-1 code is nearly a constant, so "structure retention" at that
#: level says nothing about structure.  Fixed before any correlation was computed.
COARSE_L1 = 0.5
#: prefixes of the SAME SID positions: an edge counts when both ends agree on level 1, on 1+2, on all three
LEVELS = ("prefix@1", "prefix@2", "prefix@3")


def load_roles_module():
    spec = importlib.util.spec_from_file_location("graphlet_roles", ROLES_MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def train_graph(gr):
    edges = np.asarray(gr.edges_of(), dtype=np.int64)
    graph = nx.Graph()
    graph.add_edges_from(edges)
    adj = {node: set(graph.neighbors(node)) for node in graph.nodes()}
    degrees = dict(graph.degree())
    return edges, graph, degrees, adj


def role_assignment(gr, graph, degrees, adj, five=False, overlap_mode="mean",
                    scaling="clip", normalise=False, seeds=(42,)):
    """The six roles the arm analyses use, read from the one cache ``analysis.py cache`` writes.

    Fitting a second partition here would let this table and the pattern/independence analyses
    describe different role systems under the same labels -- the review caught exactly that, with a
    Role 1 of 3,521 items in the profile and 2,448 in the retention rates.  The 5-node path therefore
    reads the shared cache and refuses a stale one (the fingerprint covers the behaviour adjacency,
    the orbit table and the overlap mode).  Only the 4-node audit refits, into its own file, because
    nothing else writes a 4-node partition.

    Two review findings also live here: ``overlap`` is undefined for a node with fewer than two
    neighbours and is imputed at the mean of the defined values (``overlap_mode="mean"``; ``"old"``
    reproduces the historical 1.0), and the cache is keyed on that fingerprint so a changed graph
    cannot be reused silently.
    """
    import analysis as study

    n_items, behaviour_adjacency, _ = study.load_graph()
    role_of = {}
    if five:
        role_arr = study.load_roles(n_items, behaviour_adjacency, overlap_mode,
                                    scaling=scaling, normalise=normalise, seed=seeds[0])
        fingerprint = study.feature_fingerprint(behaviour_adjacency, study.ORBITS, overlap_mode)[:16]
        cached = np.load(study.ROLE_CACHE)
        variants = json.loads(str(cached["variants"])) if "variants" in cached.files else {}
        source = {"cache": study.ROLE_CACHE.name, "fingerprint": fingerprint, "refit": False,
                  "variants": variants}
        print(f"roles: read {int((role_arr >= 0).sum())} assignments from {study.ROLE_CACHE.name} "
              f"(fingerprint {fingerprint}); stability {variants}", flush=True)
    else:
        orbit_path = gr.ORCA_NODE4
        if not Path(orbit_path).exists():
            raise SystemExit(f"missing orbit table {orbit_path}")
        adjacency_list = [set(behaviour_adjacency[node]) for node in range(n_items)]
        nodes = np.asarray(sorted(graph.nodes()), dtype=np.int64)
        fingerprint = study.feature_fingerprint(adjacency_list, orbit_path, overlap_mode)
        key = f"{fingerprint[:16]}|k=6|seed={seeds[0]}|{scaling}|{int(normalise)}"
        refit_path = ROLE_CACHE_4NODE_REFIT
        role_arr = None
        if refit_path.exists():
            cached = np.load(refit_path)
            if (cached["nodes"].shape[0] == nodes.shape[0]
                    and np.array_equal(cached["nodes"], nodes)
                    and str(cached["fingerprint"]) == key):
                print(f"roles: {refit_path.name} reused (fingerprint matches)", flush=True)
                role_arr = np.full(n_items, -1, dtype=np.int64)
                role_arr[cached["nodes"]] = cached["role"]
        if role_arr is None:
            matrix, info = study.build_features(n_items, adjacency_list, orbit_path, overlap_mode)
            block = matrix[nodes]
            fitted = study.fit_roles(block, 6, seeds[0], scaling, normalise)
            print(f"roles: refit on {nodes.shape[0]} items from {matrix.shape[1]} features "
                  f"({info['orbit_width']} orbits, overlap mode {overlap_mode}); "
                  f"sizes {np.bincount(fitted, minlength=6).tolist()}", flush=True)
            variant = {}
            for seed in seeds[1:]:
                variant[f"seed={seed}"] = study.adjusted_rand(
                    fitted, study.fit_roles(block, 6, seed, scaling, normalise))
            if not normalise:
                variant["normalised_components"] = study.adjusted_rand(
                    fitted, study.fit_roles(block, 6, seeds[0], scaling, True))
            if variant:
                print(f"roles: stability {variant}", flush=True)
            np.savez(refit_path, nodes=nodes, role=fitted.astype(np.int64), fingerprint=key)
            role_arr = np.full(n_items, -1, dtype=np.int64)
            role_arr[nodes] = fitted
        source = {"cache": refit_path.name, "fingerprint": fingerprint[:16], "refit": True}

    # restricted to the items the train graph has, so an item that carries a role from the behaviour
    # graph but no train edge is not counted here (it could make items_without_a_role negative)
    train_nodes = {int(node) for node in graph.nodes()}
    role_of = {int(node): int(value) for node, value in enumerate(role_arr)
               if value >= 0 and int(node) in train_nodes}
    sizes = np.bincount(np.asarray(list(role_of.values()), dtype=np.int64), minlength=6).tolist()
    routing = {"cache_hit": True, **source,
               "role_sizes": {f"role{index + 1}": int(size) for index, size in enumerate(sizes)},
               "items_without_a_role": int(graph.number_of_nodes()) - len(role_of),
               "items_in_assignment": len(role_of)}
    print(f"roles: sizes {routing['role_sizes']} | train items without a role "
          f"{routing['items_without_a_role']}", flush=True)
    return role_of, routing


def retention(tokens: np.ndarray, u, v, role_of_u, n_roles: int) -> dict:
    """prefix@1, prefix@3 and 'any token' shares, overall and per role of the first endpoint."""
    a1, a2, a3 = tokens[0], tokens[1], tokens[2]
    # same position on both sides: a level-1 code of one item never counts as a match for the
    # level-2 code of the other, which is what the earlier "any token" column allowed.
    m1 = (a1[u] == a1[v]) & (a1[u] >= 0)
    m2 = m1 & (a2[u] == a2[v]) & (a2[u] >= 0)
    m3 = m2 & (a3[u] == a3[v]) & (a3[u] >= 0)

    def block(mask):
        out = {"edges": int(mask.sum())}
        for name, hit in (("prefix@1", m1), ("prefix@2", m2), ("prefix@3", m3)):
            out[name] = float(hit[mask].mean()) if mask.any() else None
        return out

    roles = {}
    for role in range(n_roles):
        mask = role_of_u == role
        roles[role + 1] = block(mask)          # one-based, to match the report and analysis.py
    overall = block(np.ones(u.shape[0], dtype=bool))
    return {"overall": overall, "roles": roles}


def parse_options():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--five", action="store_true", help="use the 73 five-node orbits")
    parser.add_argument("--overlap-mode", default="mean", choices=("mean", "old"))
    parser.add_argument("--nmf-scaling", default="clip", choices=("clip", "shift"))
    parser.add_argument("--normalise-components", action="store_true")
    parser.add_argument("--seeds", default="42", help="comma separated; the first one is used")
    return parser.parse_args()


def main() -> int:
    options = parse_options()
    five = options.five
    out_json = OUT_JSON5 if five else OUT_JSON
    out_md = OUT_MD5 if five else OUT_MD
    gr = load_roles_module()
    def spearman(xs, ys):
        def rank(v):
            order = sorted(range(len(v)), key=lambda i: v[i])
            out, i = [0.0] * len(v), 0
            while i < len(order):
                j = i
                while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                    j += 1
                avg = (i + j) / 2.0 + 1.0
                for k in range(i, j + 1):
                    out[order[k]] = avg
                i = j + 1
            return out

        rx, ry = rank(xs), rank(ys)
        n = len(xs)
        mx, my = sum(rx) / n, sum(ry) / n
        num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
        den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
        return num / den if den else None

    def bootstrap(xs, ys, seed=20261109, draws=2000):
        rng = random.Random(seed)
        n = len(xs)
        rhos = []
        for _ in range(draws):
            idx = [rng.randrange(n) for _ in range(n)]
            rho = spearman([xs[i] for i in idx], [ys[i] for i in idx])
            if rho is not None:
                rhos.append(rho)
        rhos.sort()
        if not rhos:
            return None, None
        return (rhos[int(0.025 * (len(rhos) - 1))], rhos[int(0.975 * (len(rhos) - 1))])

    report = json.loads(REPORT.read_text())
    ledger = json.loads(LEDGER.read_text())
    # Enumerate the MIRROR, not report.json: the report covers numbers 01-56 while the archive
    # holds 01-68, so reading it would silently drop twelve arms that do have a recall.
    pairs = []
    for number in sorted(p.name for p in MIRROR.iterdir() if p.is_dir()):
        entry = ledger.get(number) or {}
        if entry.get("retired"):
            continue
        recall_path = MIRROR / number / "recall.json"
        if not recall_path.exists():
            continue
        value = json.loads(recall_path.read_text()).get(RECALL)
        if value is None:
            continue
        arm = (entry.get("arm") or (entry.get("pair") or {}).get("variant")
               or next((p.name for p in (MIRROR / number).iterdir() if p.is_dir()), number))
        pairs.append({"number": number, "arm": arm, RECALL: value,
                      "sid_sha256": (entry.get("files") or {}).get("item_sids.json")})
    in_report = {p["number"] for p in report["pairs"] if p.get(RECALL) is not None}
    report_sid = {p["number"]: p.get("sid_sha256") for p in report["pairs"]}
    print(f"pairs from the mirror: {len(pairs)} | report.json covers {len(in_report)} of them "
          f"({len(pairs) - len(in_report)} arms are archived but absent from the report)", flush=True)

    edges, graph, degrees, adj = train_graph(gr)
    u, v = edges[:, 0], edges[:, 1]
    n_nodes = int(max(edges.max(), u.max()) + 1)
    role_of, routing = role_assignment(
        gr, graph, degrees, adj, five=five, overlap_mode=options.overlap_mode,
        scaling=options.nmf_scaling, normalise=options.normalise_components,
        seeds=tuple(int(value) for value in options.seeds.split(",")))
    role_arr = np.full(n_nodes, -1, dtype=np.int64)
    for node, role in role_of.items():
        role_arr[node] = role
    n_roles = 6
    role_of_u = role_arr[u]

    rows = []
    skipped = []
    for pair in pairs:
        sid_path = MIRROR / pair["number"] / "item_sids.json"
        if not sid_path.exists():
            skipped.append({"number": pair["number"], "reason": "item_sids.json missing from the mirror"})
            continue
        digest = hashlib.sha256(sid_path.read_bytes()).hexdigest()
        # The ledger is the provenance record; report.json is a derived table and can be stale --
        # six of these pairs (51-56) disagree, and in every one of them the mirror matches the
        # ledger, so the archive is what the SID is verified against here.
        attested = ((ledger.get(pair["number"]) or {}).get("files") or {}).get("item_sids.json")
        if attested is not None and digest != attested:
            skipped.append({"number": pair["number"],
                            "reason": f"mirror sha256 {digest[:12]} is not the ledger's {str(attested)[:12]}"})
            continue
        if attested is None and digest != pair.get("sid_sha256"):
            skipped.append({"number": pair["number"],
                            "reason": f"sha256 {digest[:12]} is not the recorded {str(pair.get('sid_sha256'))[:12]}"})
            continue
        recorded = report_sid.get(pair["number"])
        record_mismatch = recorded is not None and recorded != digest
        payload = json.loads(sid_path.read_text())
        tokens = np.full((3, n_nodes), -1, dtype=np.int64)
        for key, value in payload.items():
            index = int(key)
            if index >= n_nodes or not value:
                continue
            for pos in range(min(3, len(value))):
                tokens[pos, index] = int(value[pos])
        block = retention(tokens, u, v, role_of_u, n_roles)
        spreads = [r["prefix@1"] for r in block["roles"].values() if r["prefix@1"] is not None]
        rows.append({
            "number": pair["number"], "arm": pair["arm"], RECALL: pair[RECALL],
            "sid_sha256": digest, "items_in_sid": len(payload),
            "record_disagrees_with_archive": record_mismatch,
            "overall": block["overall"], "roles": block["roles"],
            "role_gap": (max(spreads) - min(spreads)) if len(spreads) > 1 else None,
            "role_spread": float(np.std(spreads)) if len(spreads) > 1 else None,
        })
        print(f"  {pair['number']}: prefix@1 {block['overall']['prefix@1']:.4f} "
              f"prefix@3 {block['overall']['prefix@3']:.4f} recall {pair[RECALL]:.5f}", flush=True)

    recalls = [row[RECALL] for row in rows]

    def series(getter):
        xs, ys = [], []
        for row in rows:
            value = getter(row)
            if value is None:
                continue
            xs.append(value)
            ys.append(row[RECALL])
        return xs, ys

    correlations = {}

    def record(name: str, getter) -> None:
        xs, ys = series(getter)
        if len(xs) < 8:
            correlations[name] = {"n": len(xs), "rho": None, "ci95": None}
            return
        rho = spearman(xs, ys)
        lo, hi = bootstrap(xs, ys)
        correlations[name] = {"n": len(xs), "rho": rho, "ci95": [lo, hi]}

    for level in LEVELS:
        record(f"overall/{level}", lambda row, level=level: row["overall"][level])
    for role in range(n_roles):
        for level in LEVELS:
            record(f"role{role + 1}/{level}",
                   lambda row, role=role, level=level: row["roles"][role + 1][level])
    for name, getter in (("contrast/role_gap", lambda row: row["role_gap"]),
                         ("contrast/role_spread", lambda row: row["role_spread"])):
        record(name, getter)
    # The contrast that removes the pair's own level: a role's retention minus the overall one.
    # Without this, a role series can correlate with recall simply because the arm is a coarse or
    # a fine quantiser overall, which says nothing about the structure.
    degenerate = [row["number"] for row in rows if row["overall"]["prefix@1"] > COARSE_L1]
    print(f"pairs excluded from the robustness rows (level-1 share > {COARSE_L1}): {degenerate}", flush=True)
    kept = [row for row in rows if row["number"] not in degenerate]
    for role in (2, 3, 5, 6):
        xs = [row["roles"][role]["prefix@1"] - row["overall"]["prefix@1"] for row in kept]
        ys = [row[RECALL] for row in kept]
        correlations[f"robust/centered/role{role}/prefix@1"] = {
            "n": len(xs), "rho": spearman(xs, ys), "ci95": list(bootstrap(xs, ys)),
            "excluded": degenerate}
    xs = [row["role_spread"] for row in kept]
    ys = [row[RECALL] for row in kept]
    correlations["robust/contrast/role_spread"] = {
        "n": len(xs), "rho": spearman(xs, ys), "ci95": list(bootstrap(xs, ys)), "excluded": degenerate}

    for role in range(n_roles):
        record(f"centered/role{role + 1}/prefix@1",
               lambda row, role=role: (None if row["roles"][role + 1]["prefix@1"] is None
                                       else row["roles"][role + 1]["prefix@1"]
                                       - row["overall"]["prefix@1"]))

    # The role profile comes from this run's assignment and graph, never from an older fit: the 213
    # file describes a different partition, so pairing it with these retention rates mislabels the roles.
    import networkx as nx

    clustering = nx.clustering(graph)
    core_number = nx.core_number(graph)
    role_profiles = {}
    for role in range(6):
        members = [node for node, value in role_of.items() if value == role]
        if not members:
            role_profiles[str(role + 1)] = {"members": 0}
            continue
        role_profiles[str(role + 1)] = {
            "members": len(members),
            "mean_degree": float(np.mean([degrees[node] for node in members])),
            "mean_clustering": float(np.mean([clustering[node] for node in members])),
            "mean_core": float(np.mean([core_number[node] for node in members])),
            "mean_neighbour_degree": float(np.mean([
                np.mean([degrees[other] for other in adj[node]]) if adj[node] else 0.0
                for node in members]))}
    covered = sum(entry["members"] for entry in role_profiles.values())
    assert covered == len(role_of), f"profile covers {covered} of {len(role_of)} items"
    assert [role_profiles[str(role + 1)]["members"] for role in range(6)] == \
        list(routing["role_sizes"].values()), "profile and routing disagree on role sizes"

    out = {
        "recall_metric": RECALL,
        "n_pairs": len(rows),
        "n_skipped": len(skipped),
        "skipped": skipped,
        # the profile table is the one this run fitted, asserted against the routing sizes below
        "roles": role_profiles,
        "pairs": rows,
        "routing": routing,
        "correlations": correlations,
        "report_coverage": {"pairs_in_mirror": len(rows), "pairs_in_report": len(in_report)},
        "limits": [
            ("structure is ORCA's node orbits of graphlets up to 5 nodes (88 columns) plus four invariants; "
             "the six roles are NMF with k fixed by the 213 script, not selected here" if five else
             "structure is ORCA's node orbits of graphlets up to 4 nodes plus four invariants; the six roles are NMF with k fixed by the 213 script, not selected here"),
            "an edge is attributed to the role of its first endpoint (the history item), the same convention as the 213 audit",
            "every column compares the same SID position on both sides; no cross-level matching",
            f"{len(rows)} pairs are {len(rows)} arms under distinct family names: no within-family replication exists, so the interval is over arms, not over runs of one arm",
            f"Spearman on {len(rows)} arms with no multiple-comparison correction is exploratory; a row that clears zero is a hypothesis, not a finding",
            f"the robust/ rows drop the arms whose level-1 share exceeds {COARSE_L1} ({', '.join(degenerate)}): a level-1 code that coarse cannot carry structure, "
            "and the rule was fixed before any correlation was computed",
            "report.json disagrees with the archive on " +
            ", ".join(sorted(r["number"] for r in rows if r["record_disagrees_with_archive"])) +
            "; the ledger and the mirror agree with each other, so the archive is what these numbers use",
        ],
    }
    out_json.write_text(json.dumps(out, indent=1, default=float))

    order = sorted(correlations.items(), key=lambda kv: -(kv[1]["rho"] if kv[1]["rho"] is not None else -9))
    lines = [
        ("# 结构保留度 vs 下游 recall（ORCA 4+5-节点轨道 88 维 -> 6 个角色）" if five
         else "# 结构保留度 vs 下游 recall（ORCA 4-节点轨道 -> 6 个角色）"),
        "",
        f"配对：**{len(rows)}** 个有 recall 的臂（跳过 {len(skipped)} 个），每个臂一条 PARTITION，结构固定、不随臂变化。",
        "",
        "## 相关系数（Spearman，2000 次 bootstrap，n = 有效臂数）",
        "",
        "| 序列 | n | rho | 95% bootstrap |",
        "|---|---|---|---|",
    ]
    for name, block in order:
        rho = "—" if block["rho"] is None else f"{block['rho']:+.3f}"
        ci = "—" if block["ci95"] is None else f"[{block['ci95'][0]:+.3f}, {block['ci95'][1]:+.3f}]"
        lines.append(f"| `{name}` | {block['n']} | {rho} | {ci} |")
    lines += [
        "",
        "## 各角色的平均保留度（跨臂均值，`prefix@1`）",
        "",
        "| 角色 | 成员 | 平均度 | 平均聚类 | 平均 core | 平均 prefix@1 | 平均 prefix@3 | 臂数 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for role in range(n_roles):
        vals = [row["roles"][role + 1] for row in rows]
        p1 = [x["prefix@1"] for x in vals if x["prefix@1"] is not None]
        p3 = [x["prefix@3"] for x in vals if x["prefix@3"] is not None]
        profile = out["roles"][str(role + 1)]
        lines.append(f"| {role + 1} | {profile['members']} "
                     f"| {profile.get('mean_degree', float('nan')):.1f} "
                     f"| {profile.get('mean_clustering', float('nan')):.3f} "
                     f"| {profile.get('mean_core', float('nan')):.1f} "
                     f"| {np.mean(p1):.4f} | {np.mean(p3):.4f} | {len(p1)} |")
    lines += ["", "## 边界", ""] + [f"- {item}" for item in out["limits"]] + [""]
    out_md.write_text("\n".join(lines))

    for name, block in order[:12]:
        rho = "—" if block["rho"] is None else f"{block['rho']:+.3f}"
        print(f"  {name:<28} n={block['n']:>3} rho={rho}")
    print(f"written: {out_json.name}, {out_md.name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
