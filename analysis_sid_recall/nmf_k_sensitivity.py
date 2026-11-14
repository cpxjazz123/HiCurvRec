"""Does the story survive a different NMF granularity? K = 3, 4, 5, 6, 8, 10.

The independence test found that at K = 6 only one role carries information the others do not.  That
could be a real structural fact or an artefact of the granularity, so the same pipeline is rerun for
six values of K:

    K = 3, 4, 5     does any role still add something out of sample?
    K = 6           the current answer: one role stands out
    K = 8, 10       does that role split, and do the extra groups carry new information?

Everything is held fixed while K moves: the ORCA five-node orbit matrix, the four invariants, the
NMF settings (`nndsvda`, seed 42, 800 iterations), the nine pattern instance sets, their matched null
groups, the anchor-based per-item averaging, and the out-of-sample protocol (repeated 5-fold CV, ridge
with an inner penalty, marginal contribution = CV R^2 with the role minus without it).

    python3 analysis_sid_recall/nmf_k_sensitivity.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import orca5_role_preservation as m  # noqa: E402
import role_independence as ri  # noqa: E402
import role_pattern_gain as rpg  # noqa: E402

OUT = HERE / "nmf_k_sensitivity.json"
ORBITS = m.REPO / "results" / "research" / "issue213" / "orca" / "node5.out"
KS = (3, 4, 5, 6, 8, 10)
SEED = 42
MIN_ITEMS = rpg.MIN_ITEMS
RECALL = m.RECALL
BASELINE_K = 6
BASELINE_ROLE = 2          # the NMF role the report calls Role 3


def build_features(n: int, adjacency: list[set[int]]) -> np.ndarray:
    """Exactly the feature block of `graphlet_roles.features_and_roles`, minus the NMF and the argmax."""
    import itertools

    import networkx as nx

    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    for node in range(n):
        for other in adjacency[node]:
            if node < other:
                graph.add_edge(node, other)
    degrees = {node: len(adjacency[node]) for node in range(n)}
    clustering = nx.clustering(graph)
    core = nx.core_number(graph)
    mean_neighbour_degree = {
        node: float(np.mean([degrees[other] for other in adjacency[node]])) if adjacency[node] else 0.0
        for node in range(n)}
    overlap = {}
    for node in range(n):
        near = list(adjacency[node])
        if len(near) < 2:
            overlap[node] = 1.0
            continue
        total = 0.0
        for first, second in itertools.combinations(near, 2):
            union = len(adjacency[first] | adjacency[second])
            total += len(adjacency[first] & adjacency[second]) / union if union else 0.0
        overlap[node] = total / (len(near) * (len(near) - 1) / 2)

    orbits = np.loadtxt(ORBITS, dtype=np.float64)
    rows = []
    for node in range(n):
        rows.append([np.log1p(degrees[node]), clustering.get(node, 0.0), core.get(node, 0),
                     np.log1p(mean_neighbour_degree[node]), overlap[node],
                     *[np.log1p(value) for value in orbits[node]]])
    matrix = np.asarray(rows, dtype=np.float64)
    return (matrix - matrix.mean(0)) / matrix.std(0).clip(min=1e-9)


def fit_roles(matrix: np.ndarray, k: int) -> np.ndarray:
    from sklearn.decomposition import NMF

    model = NMF(n_components=k, init="nndsvda", random_state=SEED, max_iter=800)
    weights = model.fit_transform(matrix.clip(min=0.0))
    return np.argmax(weights, axis=1)


def alignment(first: np.ndarray, second: np.ndarray) -> float:
    """Share of items whose roles agree after matching each role of `first` to its most common partner."""
    usable = second >= 0
    mapping = {}
    for role in np.unique(first[usable]):
        values = second[usable & (first == role)]
        mapping[role] = int(np.bincount(values).argmax())
    mapped = np.array([mapping.get(role, -1) for role in first[usable]])
    return float(np.mean(mapped == second[usable]))


def main() -> int:
    n, adjacency, _ = m.load_graph()
    print("building the ORCA feature block ...", flush=True)
    matrix = build_features(n, adjacency)

    roles_by_k = {}
    for k in KS:
        assignment = fit_roles(matrix, k)
        roles_by_k[k] = assignment
        sizes = np.bincount(assignment, minlength=k)
        print(f"  K={k:<3} role sizes {sizes.tolist()}", flush=True)
    cached = np.full(n, -1, dtype=np.int64)
    for node, role in zip(np.load(m.ROLE_CACHE5)["nodes"], np.load(m.ROLE_CACHE5)["role"]):
        cached[int(node)] = int(role)
    agreement = alignment(roles_by_k[BASELINE_K], cached)
    print(f"  K={BASELINE_K} agrees with the cached assignment on {agreement:.1%} of items")

    assignments = dict(roles_by_k)
    assignments["6cached"] = cached

    prepared = rpg.prepare()
    patterns = prepared["patterns"]
    subsets = prepared["subsets"]
    incidences = prepared["incidences"]

    # per pattern: item -> instance lists flattened once, so every K reuses the same arithmetic
    flat = {}
    for name, incidence in incidences.items():
        items = sorted(incidence)
        item_index = np.concatenate([np.full(len(incidence[item]), item, dtype=np.int64)
                                     for item in items]) if items else np.zeros(0, dtype=np.int64)
        inst_index = np.concatenate([np.asarray(incidence[item], dtype=np.int64)
                                     for item in items]) if items else np.zeros(0, dtype=np.int64)
        counts = np.bincount(item_index, minlength=n)
        flat[name] = (item_index, inst_index, counts)

    ledger = json.loads(rpg.LEDGER.read_text())
    arms = []
    for number in sorted(p.name for p in rpg.MIRROR.iterdir() if p.is_dir()):
        entry = ledger.get(number) or {}
        if entry.get("retired"):
            continue
        sid_path, recall_path = rpg.MIRROR / number / "item_sids.json", rpg.MIRROR / number / "recall.json"
        if not (sid_path.exists() and recall_path.exists()):
            continue
        recall = json.loads(recall_path.read_text()).get(RECALL)
        if recall is None:
            continue
        attested = (entry.get("files") or {}).get("item_sids.json")
        import hashlib

        if attested is not None and hashlib.sha256(sid_path.read_bytes()).hexdigest() != attested:
            continue
        arms.append({"number": number, "recall": float(recall), "sid_path": sid_path})
    print(f"{len(arms)} arms", flush=True)

    # per arm and pattern: the per-instance psi on the real instances and on the matched groups
    per_arm: list[dict] = []
    for count, arm in enumerate(arms, start=1):
        payload = json.loads(arm["sid_path"].read_text())
        tokens = np.full((n, 3), -1, dtype=np.int64)
        for key, row in payload.items():
            item = int(key)
            if 0 <= item < n:
                tokens[item] = [int(code) for code in row[:3]]

        def distance(left: int, right: int) -> float:
            matches = 0
            for x, y in zip(tokens[left].tolist(), tokens[right].tolist()):
                if x != y:
                    break
                matches += 1
            return 1.0 - matches / 3.0

        triples = {(int(row[0]), int(row[1]), int(row[2])) for row in tokens[tokens[:, 0] >= 0]}
        collision = 1.0 - len(triples) / max(int((tokens[:, 0] >= 0).sum()), 1)
        blocks = {}
        for name in patterns:
            chosen, nulls = subsets[name]
            real_values, null_values = [], []
            for instance, draws in zip(chosen, nulls):
                if not draws:
                    continue
                computed = rpg.per_instance(name, instance, draws, distance)
                if computed is None:
                    continue
                real_values.append(computed[0])
                null_values.append(computed[1])
            blocks[name] = (np.asarray(real_values), np.asarray(null_values))
        per_arm.append({"number": arm["number"], "recall": arm["recall"],
                        "collision": collision, "blocks": blocks})
        print(f"  [{count}/{len(arms)}] {arm['number']}", flush=True)

    def role_rates(assignment: np.ndarray, k: int, block: dict) -> np.ndarray:
        """Anchor-based satisfaction rate per role for one arm, equal weights over the role's patterns."""
        patterns_of_role = {}
        for role in range(k):
            # a pattern enters the role's set when at least MIN_ITEMS items of the role touch it
            patterns_of_role[role] = [
                name for name, (item_index, _, counts) in flat.items()
                if int(np.sum(counts[assignment == role] > 0)) >= MIN_ITEMS]
        totals = np.full(k, np.nan)
        accumulators = [[] for _ in range(k)]
        for name, (item_index, inst_index, counts) in flat.items():
            real_values, _ = block[name]
            if real_values.size == 0:
                continue
            sums = np.bincount(item_index, weights=real_values[inst_index], minlength=n)
            present = counts > 0
            item_means = np.full(n, np.nan)
            item_means[present] = sums[present] / counts[present]
            for role in range(k):
                if name not in patterns_of_role[role]:
                    continue
                mask = (assignment == role) & present
                if mask.any():
                    accumulators[role].append(float(np.nanmean(item_means[mask])))
        for role in range(k):
            if accumulators[role]:
                totals[role] = float(np.mean(accumulators[role]))
        return totals

    recalls = np.array([arm["recall"] for arm in per_arm])
    collisions = np.array([arm["collision"] for arm in per_arm])
    rng = np.random.default_rng(SEED)

    result = {"ks": list(KS), "n_arms": len(arms), "min_items": MIN_ITEMS,
              "agreement_with_cached_k6": agreement, "results": {}}
    for k in KS + ("6cached",):
        assignment = assignments[k]
        roles = int(assignment.max()) + 1
        table = np.array([role_rates(assignment, roles, arm["blocks"]) for arm in per_arm])
        sizes = np.bincount(assignment[assignment >= 0], minlength=roles)
        kept = [role for role in range(roles) if int(sizes[role]) >= MIN_ITEMS]
        entry = {"role_sizes": sizes.tolist(), "kept_roles": [role + 1 for role in kept],
                 "roles": {}}
        print(f"\n=== K={k}: role rates vs recall (n={len(arms)}) ===")
        print(f"  {'role':<8}{'rate mean':>10}{'rho':>8}{'p':>7}{'r':>8}{'p':>7}{'partS':>9}{'partP':>9}")
        for role in range(roles):
            column = table[:, role]
            usable = np.isfinite(column)
            if usable.sum() < 5:
                entry["roles"][str(role)] = {"n": int(usable.sum())}
                continue
            first, second = column[usable], recalls[usable]
            rho, r = rpg.spearman(first, second), rpg.pearson(first, second)
            entry["roles"][str(role)] = {
                "role": role + 1, "rate_mean": float(first.mean()), "n": int(usable.sum()),
                "spearman": rho, "pearson": r,
                "partial_spearman": rpg.partial(first, second, collisions[usable], True),
                "partial_pearson": rpg.partial(first, second, collisions[usable], False)}
            print(f"  {role + 1:<8}{first.mean():>10.3f}{rho:>+8.3f}{0:>7.3f}{r:>+8.3f}"
                  f"{0:>7.3f}{rpg.partial(first, second, collisions[usable], True):>+9.3f}"
                  f"{rpg.partial(first, second, collisions[usable], False):>+9.3f}")

        features = np.zeros((len(per_arm), len(kept)))
        for position, role in enumerate(kept):
            column = table[:, role]
            features[:, position] = np.where(np.isfinite(column), column,
                                             np.nanmean(column[np.isfinite(column)]))
        full = ri.cross_validate(features, recalls, rng)
        entry["cv_r2_all"] = full["r2_mean"]
        print(f"  all {k} roles: CV R^2 {full['r2_mean']:+.4f}")
        best, best_drop = None, -np.inf
        drops = {}
        for position, role in enumerate(kept):
            keep = [index for index in range(features.shape[1]) if index != position]
            without = ri.cross_validate(features[:, keep], recalls, rng)
            drop = full["r2_mean"] - without["r2_mean"]
            drops[str(role)] = {"without_r2": without["r2_mean"], "drop": drop}
            if drop > best_drop:
                best, best_drop = role, drop
        entry["marginal"] = drops
        entry["best_role"] = best + 1
        entry["best_role_drop"] = best_drop
        # permutation p for the role that matters most
        position = kept.index(best)
        keep = [index for index in range(features.shape[1]) if index != position]
        without = ri.cross_validate(features[:, keep], recalls, rng)
        shuffle_rng = np.random.default_rng(SEED + 17)
        null_drops = []
        for _ in range(100):
            shuffled = features.copy()
            shuffled[:, best] = shuffle_rng.permutation(features[:, best])
            null_drops.append(ri.cross_validate(shuffled, recalls, rng)["r2_mean"]
                              - without["r2_mean"])
        entry["best_role_p"] = float((np.sum(np.array(null_drops) >= best_drop) + 1) / 101)
        print(f"  strongest marginal: role {best + 1}, drop {best_drop:+.4f}, "
              f"p {entry['best_role_p']:.3f}")

        if k == "6cached":
            base = np.flatnonzero(roles_by_k[BASELINE_K] == BASELINE_ROLE)
            sizes = np.bincount(assignment[base][assignment[base] >= 0], minlength=roles)
            entry["baseline_role3_refined"] = {
                "sizes": sizes.tolist(),
                "shares": (sizes / max(len(base), 1)).tolist()}
            print(f"  the K={BASELINE_K} Role 3 splits into {k} roles as {sizes.tolist()}")
        result["results"][str(k)] = entry

    OUT.write_text(json.dumps(result, indent=1, default=float))
    print(f"\nwritten {OUT.relative_to(m.REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
