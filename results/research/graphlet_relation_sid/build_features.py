"""Build the edge-level table for Graphlet Role x relation x SID-prefix analysis.

One row per undirected TRAIN behaviour edge.  Every feature is computed from TRAIN only; the TEST
split contributes only the *outcome* columns (does the edge reappear as a future transition), never
an input to the graph, the roles, or the SID.

Run with the Stage2 interpreter (torch is only needed by the optional codebook extractor, which this
script reads as a plain npz):

    /home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 \
        results/research/graphlet_relation_sid/build_features.py

Writes (inside results/research/graphlet_relation_sid/):
    edge_features.parquet    one row per undirected TRAIN edge
    item_features.parquet    item -> role / community / degree / popularity / SID
    graph_stats.json         sizes, edge counts, session stats, pattern validation
    graphlet_names.json      bit -> graphlet name, plus instance-validation evidence
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
TRAIN = REPO / "results" / "stage0_build_parquet" / "train.parquet"
TEST = REPO / "results" / "stage0_build_parquet" / "test.parquet"
ITEMS = REPO / "results" / "stage0_build_parquet" / "items.parquet"
EMB = REPO / "stage1_GeneEmbedding" / "output" / "sentence_t5.npy"
ITEM_IDS = REPO / "stage1_GeneEmbedding" / "output" / "item_ids.json"
SID_JSON = REPO / "results" / "stage2_RQ-VAE" / "TIGER_RQ-VAE" / "item_sids.json"
ROLES = REPO / "results" / "research" / "issue215" / "orca_roles_frozen.csv"
INSTANCES = REPO / "results" / "research" / "issue215" / "graphlet_instances.npz"
CODEBOOKS = HERE / "codebooks.npz"
OUT_EDGES = HERE / "edge_features.parquet"
OUT_ITEMS = HERE / "item_features.parquet"
OUT_STATS = HERE / "graph_stats.json"
OUT_GRAPHLETS = HERE / "graphlet_names.json"

#: the canonical order of issue215's sampled graphlet patterns
PATTERNS = ("K5", "double_triangle", "K4_tail", "K2_3", "P5_centre", "C5", "P5_hops", "K1_4", "T5")
STOPWORDS = set("""
a an the and or for with without of to in on at by from as is are was were be been being this that
these those it its his her their our your my you we they he she them him us i not no nor but if then
than so such into over under between within per each other more most less least very can could may
might will would should do does did done has have had having use used using new set pack piece pieces
""".split())


def tokenize(title: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", str(title).lower())
    return {word for word in words if word not in STOPWORDS and len(word) > 1}


def load_sessions() -> dict[int, list[int]]:
    """One full TRAIN session per user: the longest prefix row plus its target.

    The parquet is prefix-expanded, so counting pairs over rows would count the same user-session
    pair many times; the longest row is the user's whole TRAIN portion (the last two interactions
    are held out for valid/test and are never read here).
    """
    frame = pd.read_parquet(TRAIN, columns=["user", "seen_history", "target"])
    best: dict[int, tuple[int, list[int]]] = {}
    for user, history, target in zip(
        frame["user"].to_numpy(), frame["seen_history"].to_numpy(), frame["target"].to_numpy(np.int64)
    ):
        history = [] if history is None else [int(x) for x in history]
        user = int(user)
        if user not in best or len(history) > best[user][0]:
            best[user] = (len(history), [*history, int(target)])
    return {user: session for user, (_, session) in best.items()}


def load_items() -> pd.DataFrame:
    frame = pd.read_parquet(ITEMS)
    frame["item_id"] = frame["item_id"].astype(np.int64)
    return frame.sort_values("item_id").reset_index(drop=True)


def load_sids(n_items: int) -> np.ndarray:
    payload = json.loads(SID_JSON.read_text())
    rows = np.zeros((n_items, 3), dtype=np.int64)
    for item in range(n_items):
        raw = payload[str(item)]
        rows[item] = [int(x) for x in raw[:3]]
    return rows


def load_roles(n_items: int) -> np.ndarray:
    frame = pd.read_csv(ROLES)
    roles = np.zeros(n_items, dtype=np.int64)
    for item, role in zip(frame["item_id"].to_numpy(np.int64), frame["role"].to_numpy(np.int64)):
        if 0 <= item < n_items:
            roles[item] = role
    return roles


def title_jaccard(titles: list[str], left: np.ndarray, right: np.ndarray) -> np.ndarray:
    sets = [tokenize(title) for title in titles]
    sizes = np.array([len(value) for value in sets], dtype=np.float64)
    out = np.zeros(len(left), dtype=np.float64)
    for index, (a, b) in enumerate(zip(left.tolist(), right.tolist())):
        first, second = sets[a], sets[b]
        union = sizes[a] + sizes[b] - len(first & second)
        out[index] = len(first & second) / union if union else 0.0
    return out


def semantic_cosine(embeddings: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1)
    unit = embeddings / np.clip(norms, 1e-12, None)[:, None]
    out = np.zeros(len(left), dtype=np.float64)
    step = 50_000
    for start in range(0, len(left), step):
        stop = min(start + step, len(left))
        out[start:stop] = np.einsum("ij,ij->i", unit[left[start:stop]], unit[right[start:stop]])
    return out


def graphlet_membership(adjacency, edges_index) -> tuple[np.ndarray, np.ndarray, dict]:
    """Bit per sampled pattern, plus how many sampled instances contain the edge.

    The cache stores at most 1000 instances per pattern (119 for double_triangle), so this is a
    sample of structural membership, not an exact count; the exact triangle/common-neighbour counts
    in the edge table carry the complement.
    """
    stored = np.load(INSTANCES, allow_pickle=True)
    names = [name for name in PATTERNS if name in stored.files]
    mask = np.zeros(len(edges_index), dtype=np.uint16)
    count = np.zeros(len(edges_index), dtype=np.int16)
    validation = {}
    for bit, name in enumerate(names):
        instances = stored[name]
        induced_hist = Counter()
        degree_hist = Counter()
        for nodes in instances.tolist():
            nodes = [int(node) for node in nodes]
            induced = []
            for left, right in combinations(nodes, 2):
                if right in adjacency[left]:
                    induced.append((left, right) if left < right else (right, left))
            induced_hist[len(induced)] += 1
            degree_hist[tuple(sorted((sum(1 for u, v in induced if u == node or v == node) for node in nodes)))] += 1
            for edge in induced:
                position = edges_index.get(edge)
                if position is not None:
                    mask[position] |= np.uint16(1 << bit)
                    count[position] += 1
        validation[name] = {
            "instances": int(len(instances)),
            "induced_edge_histogram": {str(k): int(v) for k, v in sorted(induced_hist.items())},
            "degree_sequence_histogram": {
                "|".join(str(value) for value in key): int(value) for key, value in degree_hist.most_common(10)
            },
            "edges_with_this_pattern": int(((mask >> bit) & 1).sum()),
        }
        print(name, validation[name]["induced_edge_histogram"], flush=True)
    return mask, count, {"order": names, "validation": validation}


def main() -> int:
    items = load_items()
    n_items = len(items)
    titles = items["title"].tolist()
    item_ids = json.loads(ITEM_IDS.read_text())
    assert item_ids == list(range(n_items)), "embedding row order is not item_id order"
    embeddings = np.load(EMB)

    sessions = load_sessions()
    n_users = len(sessions)
    print(f"users {n_users}, sessions mean {np.mean([len(s) for s in sessions.values()]):.2f}", flush=True)

    trans: Counter[tuple[int, int]] = Counter()
    cooc: Counter[tuple[int, int]] = Counter()
    popularity = np.zeros(n_items, dtype=np.int64)
    edge_set: set[tuple[int, int]] = set()
    for session in sessions.values():
        for item in session:
            popularity[item] += 1
        for left, right in zip(session, session[1:]):
            trans[(left, right)] += 1
            edge_set.add((left, right) if left < right else (right, left))
        unique = sorted(set(session))
        for left, right in combinations(unique, 2):
            cooc[(left, right)] += 1

    edges = np.array(sorted(edge_set), dtype=np.int64)
    left, right = edges[:, 0], edges[:, 1]
    n_edges = len(edges)
    print(f"edges {n_edges}", flush=True)

    adjacency: list[set[int]] = [set() for _ in range(n_items)]
    for a, b in edges.tolist():
        adjacency[a].add(b)
        adjacency[b].add(a)
    degree = np.array([len(value) for value in adjacency], dtype=np.int64)
    edges_index = {(int(a), int(b)): index for index, (a, b) in enumerate(edges.tolist())}

    print("triangles / 4-cliques", flush=True)
    triangles = np.zeros(n_edges, dtype=np.int64)
    clique4 = np.zeros(n_edges, dtype=np.int64)
    for index, (a, b) in enumerate(edges.tolist()):
        common = adjacency[a] & adjacency[b]
        triangles[index] = len(common)
        if len(common) >= 2:
            clique4[index] = sum(1 for x, y in combinations(common, 2) if y in adjacency[x])

    print("networkx structure", flush=True)
    import networkx as nx

    graph = nx.Graph()
    graph.add_nodes_from(range(n_items))
    graph.add_edges_from(edges.tolist())
    bridges = {tuple(sorted(edge)) for edge in nx.bridges(graph)}
    is_bridge = np.array([(int(a), int(b)) in bridges for a, b in edges.tolist()], dtype=np.int8)
    core = nx.core_number(graph)
    clustering = nx.clustering(graph)
    communities = nx.community.louvain_communities(graph, seed=42)
    community = np.zeros(n_items, dtype=np.int64)
    for number, members in enumerate(communities):
        for item in members:
            community[item] = number
    print(f"communities {len(communities)}", flush=True)

    mask, gcount, graphlet_meta = graphlet_membership(adjacency, edges_index)
    graphlet_meta["bridges"] = int(is_bridge.sum())
    graphlet_meta["communities"] = int(len(communities))

    roles = load_roles(n_items)
    sids = load_sids(n_items)

    same_l1 = (sids[left, 0] == sids[right, 0])
    same_l2 = (sids[left, 1] == sids[right, 1])
    same_l3 = (sids[left, 2] == sids[right, 2])
    prefix1 = same_l1
    prefix2 = same_l1 & same_l2
    prefix3 = same_l1 & same_l2 & same_l3
    lcp = np.where(same_l1, np.where(same_l2, np.where(same_l3, 3, 2), 1), 0).astype(np.int8)

    print("semantic / title similarity", flush=True)
    sem_cos = semantic_cosine(embeddings, left, right)
    title_jac = title_jaccard(titles, left, right)

    if CODEBOOKS.exists():
        codebooks = np.load(CODEBOOKS)["codebooks"]
        quantised = np.zeros((n_items, codebooks.shape[2]), dtype=np.float32)
        for level in range(3):
            quantised += codebooks[level][sids[:, level]]
        quantised = quantised.astype(np.float64)
        q_norms = np.linalg.norm(quantised, axis=1)
        q_l2 = np.linalg.norm(quantised[left] - quantised[right], axis=1)
        q_cos = np.einsum("ij,ij->i", quantised[left], quantised[right]) / np.clip(
            q_norms[left] * q_norms[right], 1e-12, None
        )
        level_l2 = {}
        for level in range(3):
            delta = codebooks[level][sids[left, level]] - codebooks[level][sids[right, level]]
            distance = np.linalg.norm(delta.astype(np.float64), axis=1)
            level_l2[level] = np.where(sids[left, level] == sids[right, level], 0.0, distance)
    else:
        quantised = None
        q_l2 = q_cos = np.full(n_edges, np.nan)
        level_l2 = {level: np.full(n_edges, np.nan) for level in range(3)}

    print("TEST future outcomes", flush=True)
    test_frame = pd.read_parquet(TEST, columns=["seen_history", "target"])
    test_pairs: Counter[tuple[int, int]] = Counter()
    exposure = np.zeros(n_items, dtype=np.int64)
    target_count = np.zeros(n_items, dtype=np.int64)
    test_transition: set[tuple[int, int]] = set()
    n_test = len(test_frame)
    for history, target in zip(test_frame["seen_history"].to_numpy(), test_frame["target"].to_numpy(np.int64)):
        history = [] if history is None else [int(x) for x in history]
        target = int(target)
        target_count[target] += 1
        for item in set(history):
            exposure[item] += 1
        for item in history:
            test_pairs[(item, target)] += 1
        if history:
            test_transition.add((history[-1], target))
    base_rate = (target_count + 0.5) / (n_test + 1.0)
    out_count: Counter[int] = Counter()
    for (source, _), value in trans.items():
        out_count[source] += value

    def direction_features(a: int, b: int) -> tuple[float, float, int, int, float, float]:
        trans_ab = trans.get((a, b), 0)
        trans_ba = trans.get((b, a), 0)
        out_a = out_count[a]
        out_b = out_count[b]
        p_ab = (trans_ab + 0.5) / (out_a + 1.0)
        p_ba = (trans_ba + 0.5) / (out_b + 1.0)
        future_ab = test_pairs.get((a, b), 0)
        future_ba = test_pairs.get((b, a), 0)
        expected_ab = exposure[a] * base_rate[b]
        expected_ba = exposure[b] * base_rate[a]
        lift_ab = np.log((future_ab + 0.5) / (expected_ab + 0.5))
        lift_ba = np.log((future_ba + 0.5) / (expected_ba + 0.5))
        return p_ab, p_ba, future_ab, future_ba, float(lift_ab), float(lift_ba)

    rows = defaultdict(list)
    for index, (a, b) in enumerate(edges.tolist()):
        p_ab, p_ba, future_ab, future_ba, lift_ab, lift_ba = direction_features(a, b)
        rows["left"].append(a)
        rows["right"].append(b)
        rows["degree_left"].append(int(degree[a]))
        rows["degree_right"].append(int(degree[b]))
        rows["popularity_left"].append(int(popularity[a]))
        rows["popularity_right"].append(int(popularity[b]))
        rows["core_left"].append(int(core[a]))
        rows["core_right"].append(int(core[b]))
        rows["cluster_left"].append(float(clustering[a]))
        rows["cluster_right"].append(float(clustering[b]))
        rows["role_left"].append(int(roles[a]))
        rows["role_right"].append(int(roles[b]))
        rows["community_left"].append(int(community[a]))
        rows["community_right"].append(int(community[b]))
        rows["same_community"].append(int(community[a] == community[b]))
        rows["triangles"].append(int(triangles[index]))
        rows["clique4_edges"].append(int(clique4[index]))
        rows["is_bridge"].append(int(is_bridge[index]))
        rows["graphlet_mask"].append(int(mask[index]))
        rows["graphlet_count"].append(int(gcount[index]))
        rows["sem_cos"].append(float(sem_cos[index]))
        rows["title_jaccard"].append(float(title_jac[index]))
        rows["q_l2"].append(float(q_l2[index]))
        rows["q_cos"].append(float(q_cos[index]))
        rows["q_level1_l2"].append(float(level_l2[0][index]))
        rows["q_level2_l2"].append(float(level_l2[1][index]))
        rows["q_level3_l2"].append(float(level_l2[2][index]))
        rows["cooc_users"].append(int(cooc.get((a, b), 0)))
        rows["trans_ab"].append(int(trans.get((a, b), 0)))
        rows["trans_ba"].append(int(trans.get((b, a), 0)))
        rows["p_trans_ab"].append(float(p_ab))
        rows["p_trans_ba"].append(float(p_ba))
        rows["p_trans_max"].append(float(max(p_ab, p_ba)))
        rows["p_trans_asym"].append(float(abs(p_ab - p_ba) / (p_ab + p_ba + 1e-12)))
        rows["future_ab"].append(int(future_ab))
        rows["future_ba"].append(int(future_ba))
        rows["future_sum"].append(int(future_ab + future_ba))
        rows["future_hit"].append(int(future_ab + future_ba > 0))
        rows["future_lift_ab"].append(float(lift_ab))
        rows["future_lift_ba"].append(float(lift_ba))
        rows["future_lift_max"].append(float(max(lift_ab, lift_ba)))
        rows["test_edge_ab"].append(int((a, b) in test_transition))
        rows["test_edge_ba"].append(int((b, a) in test_transition))
        rows["lcp"].append(int(lcp[index]))
        rows["prefix1"].append(int(prefix1[index]))
        rows["prefix2"].append(int(prefix2[index]))
        rows["prefix3"].append(int(prefix3[index]))
        rows["sid_l1_equal"].append(int(same_l1[index]))
        rows["sid_l2_equal"].append(int(same_l2[index]))
        rows["sid_l3_equal"].append(int(same_l3[index]))

    edge_frame = pd.DataFrame(rows)
    edge_frame.to_parquet(OUT_EDGES, index=False)
    print(f"wrote {OUT_EDGES} {edge_frame.shape}", flush=True)

    item_frame = pd.DataFrame(
        {
            "item_id": np.arange(n_items, dtype=np.int64),
            "role": roles,
            "community": community,
            "degree": degree,
            "popularity": popularity,
            "test_exposure": exposure,
            "test_target_count": target_count,
            "sid1": sids[:, 0],
            "sid2": sids[:, 1],
            "sid3": sids[:, 2],
        }
    )
    item_frame.to_parquet(OUT_ITEMS, index=False)

    stats = {
        "train_rows": int(len(pd.read_parquet(TRAIN, columns=["user"]))),
        "users": int(n_users),
        "items": int(n_items),
        "nodes_with_edges": int((degree > 0).sum()),
        "edges": int(n_edges),
        "mean_degree": float(degree.mean()),
        "max_degree": int(degree.max()),
        "mean_session_length": float(np.mean([len(s) for s in sessions.values()])),
        "max_session_length": int(max(len(s) for s in sessions.values())),
        "communities": int(len(communities)),
        "bridges": int(is_bridge.sum()),
        "test_rows": int(n_test),
        "test_users_with_history": int((pd.read_parquet(TEST, columns=["seen_history"])["seen_history"].map(len) > 0).sum()),
        "train_transitions": int(sum(trans.values())),
        "cooc_pairs": int(len(cooc)),
        "test_pairs": int(len(test_pairs)),
        "prefix_rates": {
            "prefix1": float(prefix1.mean()),
            "prefix2": float(prefix2.mean()),
            "prefix3": float(prefix3.mean()),
            "mean_lcp": float(lcp.mean()),
        },
        "sem_cos": {"mean": float(sem_cos.mean()), "p10": float(np.percentile(sem_cos, 10)),
                    "p50": float(np.percentile(sem_cos, 50)), "p90": float(np.percentile(sem_cos, 90))},
        "title_jaccard": {"mean": float(title_jac.mean()), "p10": float(np.percentile(title_jac, 10)),
                          "p50": float(np.percentile(title_jac, 50)), "p90": float(np.percentile(title_jac, 90))},
        "categories": {
            "available": bool(pd.read_parquet(ITEMS, columns=["categories"])["categories"].notna().any()),
            "note": "items.parquet carries categories=None for every item; category was not recoverable from the shipped artifacts",
        },
    }
    OUT_STATS.write_text(json.dumps(stats, indent=2) + "\n")
    OUT_GRAPHLETS.write_text(json.dumps(graphlet_meta, indent=2) + "\n")
    print(json.dumps(stats, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
