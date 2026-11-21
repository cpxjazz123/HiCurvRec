#!/usr/bin/env python3
"""Does the discrete SID carry the graph's structural roles?

No training, no edge predictor.  The question is read directly off the three codes:

* within an induced graphlet instance, do nodes in the **same orbit** share an L1 / L1+L2 /
  full prefix more often than nodes in **different orbits** (the ``role_gap``);
* what share of each graphlet type's nodes share a prefix at all;
* for directed merge and fork centres, do the in- (out-) neighbours share prefixes with
  each other while the centre does not share with them;
* and the plain three-code collision rate.

Instances and orbit labels come from ``analysis_sid_recall/build_graphlet_roles.py`` so the
definitions are the repository's own, not re-invented here.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
INSTANCES = REPO / "results" / "research" / "graphlet_roles" / "instances.npz"
GRAPH = REPO / "results" / "research" / "behavior_graph" / "graph.npz"
SIDS = REPO / "results" / "research" / "directed_graphlet_teacher" / "graph_loss" / "sids"
BASELINE = REPO / "results" / "stage2_RQ-VAE" / "TIGER_drq_D1" / "item_sids.json"
OUT = Path(__file__).resolve().parent / "sid_structure.json"
TYPES = ("K3", "P3", "C4", "P4", "K1_3")
CENTRE_MIN = 3
CENTRE_CAP = 400
CONTROL_ROUNDS = 200


def load_sid(path: Path) -> np.ndarray:
    payload = json.loads(Path(path).read_text())
    return np.array([payload[str(index)][:3] for index in range(len(payload))], dtype=np.int64)


def pair_stats(codes: np.ndarray, nodes: np.ndarray, roles: np.ndarray, depth: int):
    """Per instance: P(share | same orbit), P(share | different orbits), both countable."""
    sliced = codes[nodes, :depth]
    same_code = (sliced[:, :, None, :] == sliced[:, None, :, :]).all(-1)
    same_role = roles[:, :, None] == roles[:, None, :]
    off_diagonal = ~np.eye(nodes.shape[1], dtype=bool)[None, :, :]
    same_mask = same_role & off_diagonal
    diff_mask = (~same_role) & off_diagonal
    same_hits = (same_code & same_mask).sum(1)
    diff_hits = (same_code & diff_mask).sum(1)
    return same_hits, same_mask.sum(1), diff_hits, diff_mask.sum(1)


def prefix_share_all(codes: np.ndarray, nodes: np.ndarray, depth: int) -> np.ndarray:
    sliced = codes[nodes, :depth]
    same = (sliced[:, :, None, :] == sliced[:, None, :, :]).all(-1)
    pairs = same.shape[1]
    return (same.sum(axis=(1, 2)) - pairs) / (pairs * (pairs - 1))


def cross_instance_gap(codes: np.ndarray, nodes: np.ndarray, roles: np.ndarray,
                       depth: int, samples: int = 300000, seed: int = 42):
    """Same-orbit vs different-orbit prefix sharing **across** instances.

    The within-instance version cannot separate "same role" from "adjacent in the graph":
    a P3's leaves are both role 0 yet they are two different products hanging off one
    centre.  Pairing nodes of the same orbit from *different* instances removes that
    confound, which is why this is the primary number.
    """
    flat_nodes = nodes.ravel()
    flat_roles = roles.ravel()
    uniq, inverse = np.unique(flat_nodes, return_inverse=True)
    votes = np.zeros((len(uniq), int(flat_roles.max()) + 1), dtype=np.int64)
    np.add.at(votes, (inverse, flat_roles), 1)
    role_of = votes.argmax(1)
    generator = np.random.default_rng(seed)
    left = generator.integers(0, len(uniq), samples)
    right = generator.integers(0, len(uniq), samples)
    keep = left != right
    left, right = left[keep], right[keep]
    same_role = role_of[left] == role_of[right]
    agree = (codes[uniq[left], :depth] == codes[uniq[right], :depth]).all(-1)
    same_share = float(agree[same_role].mean()) if same_role.any() else None
    diff_share = float(agree[~same_role].mean()) if (~same_role).any() else None
    gap = None if (same_share is None or diff_share is None) else same_share - diff_share
    interval = None
    if gap is not None:
        draw = generator.integers(0, len(agree), (400, len(agree)))
        gaps = (agree[draw] & same_role[draw]).sum(1) / np.maximum(same_role[draw].sum(1), 1) \
            - (agree[draw] & ~same_role[draw]).sum(1) / np.maximum((~same_role[draw]).sum(1), 1)
        interval = [float(np.percentile(gaps, 2.5)), float(np.percentile(gaps, 97.5))]
    return {"same_orbit_share": same_share, "different_orbit_share": diff_share,
            "role_gap": gap, "role_gap_ci95": interval,
            "nodes": int(len(uniq)), "pairs": int(len(agree))}


def directed_roles(rows: np.ndarray, cols: np.ndarray, items: int):
    """Merge / fork centres and their neighbour sets, straight from the directed edges."""
    in_partners: dict = {}
    out_partners: dict = {}
    for a, b in zip(rows.tolist(), cols.tolist()):
        out_partners.setdefault(a, set()).add(b)
        in_partners.setdefault(b, set()).add(a)
    return in_partners, out_partners


def neighbour_sharing(codes: np.ndarray, sets: dict, depth: int, rng, rounds: int,
                      pool):
    """P(share prefix) among a centre's partners vs a size-matched degree-matched control.

    The control draws from items that are *also* someone's neighbour (the pool), never from
    the whole item space: popular targets would otherwise be compared against cold items,
    and any code-frequency difference would read as structure.
    """
    observed, control, centre_share, centre_control, deltas = [], [], [], [], []
    pool = np.asarray(sorted(pool), dtype=np.int64)
    for centre, partners in sets.items():
        partners = sorted(partners)
        if len(partners) < 2:
            continue
        chosen = partners[:12]
        stacked = codes[chosen, :depth]
        agreement = (stacked[:, None, :] == stacked[None, :, :]).all(-1)
        pairs = len(chosen) * (len(chosen) - 1)
        share = float((agreement.sum() - len(chosen)) / pairs)  # off-diagonal only
        observed.append((pairs, share))
        centre_hit = float((codes[chosen, :depth] == codes[centre, :depth]).all(-1).mean())
        centre_share.append((pairs, centre_hit))
        draws = []
        blocked = set(chosen) | {centre}
        allowed = np.array([item for item in pool.tolist() if item not in blocked],
                           dtype=np.int64)
        for _ in range(rounds):
            if len(allowed) < len(chosen):
                break
            random_set = rng.choice(allowed, size=len(chosen), replace=False)
            stacked = codes[random_set, :depth]
            agreement = (stacked[:, None, :] == stacked[None, :, :]).all(-1)
            draws.append(float(agreement.sum() - len(chosen)) /
                         (len(chosen) * (len(chosen) - 1)))
        control_mean = float(np.mean(draws))
        control.append((pairs, control_mean))
        deltas.append(share - control_mean)
        centre_control.append((pairs, float(
            np.mean([bool(np.array_equal(codes[rng.choice(allowed), :depth],
                                         codes[centre, :depth])) for _ in range(20)]))))
    def weighted(entries):
        if not entries:
            return None
        weights = np.array([entry[0] for entry in entries], dtype=np.float64)
        values = np.array([entry[1] for entry in entries], dtype=np.float64)
        return float((weights * values).sum() / weights.sum())
    if len(deltas) >= 10:
        generator = np.random.default_rng(7)
        array = np.asarray(deltas)
        means = [array[generator.integers(0, len(array), len(array))].mean()
                 for _ in range(1000)]
        interval = [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]
    else:
        interval = None
    return {"observed": weighted(observed), "control": weighted(control),
            "delta": float(np.mean(deltas)) if len(deltas) else None,
            "delta_ci95": interval,
            "centre_vs_partner": weighted(centre_share),
            "centre_vs_random": weighted(centre_control),
            "centres": len(observed)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    arms = {"baseline": BASELINE}
    arms.update({name: SIDS / f"{name}.json" for name in
                 ("A_none", "B_undirected", "C_random", "D_directed")})
    for role_arm in ("E", "T", "R", "Q", "QR"):
        candidate = (REPO / "results" / "stage2_RQ-VAE" / f"TIGER_graphlet_role_{role_arm}"
                     / "item_sids.json")
        if candidate.is_file():
            arms[f"role_{role_arm}"] = candidate
    instances = np.load(INSTANCES)
    graph = np.load(GRAPH)
    rows, cols = graph["rows"].astype(np.int64), graph["cols"].astype(np.int64)
    sys.path.insert(0, str(SIDS.parents[1] / "augment_gnn"))
    import augment_gnn as base
    (msg_rows, msg_cols), _, _, _ = base.split_edges(rows, cols)
    in_partners, out_partners = directed_roles(msg_rows, msg_cols, len(rows))
    targets = set(msg_cols.tolist())
    sources = set(msg_rows.tolist())
    merge_sets = {k: v for k, v in in_partners.items() if CENTRE_MIN <= len(v) and k in targets}
    fork_sets = {k: v for k, v in out_partners.items() if CENTRE_MIN <= len(v) and k in sources}
    merge_items = sorted(merge_sets)[:CENTRE_CAP]
    fork_items = sorted(fork_sets)[:CENTRE_CAP]
    report = {"protocol": {
        "instances": str(INSTANCES.relative_to(REPO)),
        "orbit_definitions": "analysis_sid_recall/build_graphlet_roles.py (induced "
                             "undirected graphlets; P3 and K1_3 have centre/leaf orbits, "
                             "P4 has middle/end orbits)",
        "directed_roles": f"merge centre = in-degree >= {CENTRE_MIN} (train edges), "
                          f"fork centre = out-degree >= {CENTRE_MIN}; "
                          f"{len(merge_items)} / {len(fork_items)} centres used",
        "share": "P(two nodes agree on the first k codes), k = 1, 2, 3",
        "no_training": True},
        "arms": {}}
    for name, path in arms.items():
        if not Path(path).is_file():
            report["arms"][name] = {"missing": str(path)}
            continue
        rng = np.random.default_rng(42)  # same draws for every arm
        codes = load_sid(Path(path))
        entry = {"sid": str(Path(path).relative_to(REPO)),
                 "items": int(len(codes)),
                 "collision": float(1.0 - len(np.unique(codes, axis=0)) / len(codes))}
        for kind in TYPES:
            nodes = instances[f"{kind}_nodes"]
            roles = instances[f"{kind}_roles"]
            split = instances[f"{kind}_split"]
            held = nodes[split == 1], roles[split == 1]  # 1 = heldout in instances.npz
            entry[kind] = {}
            for depth in (1, 2, 3):
                same_hits, same_total, diff_hits, diff_total = pair_stats(
                    codes, held[0], held[1], depth)
                share_same = float(same_hits.sum() / max(same_total.sum(), 1))
                share_diff = float(diff_hits.sum() / max(diff_total.sum(), 1))
                # K3 and C4 are vertex-transitive: every node has the same orbit, so the
                # same-vs-different contrast is undefined and only the absolute share is
                # meaningful there.
                gapped = bool(diff_total.sum() > 0)
                entry[kind][f"prefix@{depth}"] = {
                    "same_orbit_share": share_same,
                    "different_orbit_share": share_diff if gapped else None,
                    "role_gap": (share_same - share_diff) if gapped else None}
            entry[kind]["cross_instance"] = {
                f"prefix@{depth}": cross_instance_gap(codes, held[0], held[1], depth)
                for depth in (1, 2)}
            entry[kind]["all_nodes_share@1"] = float(
                prefix_share_all(codes, held[0], 1).mean())
        merge_pool = set().union(*(merge_sets[k] for k in merge_items))
        fork_pool = set().union(*(fork_sets[k] for k in fork_items))
        entry["merge"] = neighbour_sharing(codes, {k: merge_sets[k] for k in merge_items},
                                           1, rng, 50, merge_pool)
        entry["fork"] = neighbour_sharing(codes, {k: fork_sets[k] for k in fork_items},
                                          1, rng, 50, fork_pool)
        entry["merge"]["control_pool"] = "items that are in-neighbours of some merge centre"
        entry["fork"]["control_pool"] = "items that are out-neighbours of some fork centre"
        report["arms"][name] = entry
        cross = entry["P3"]["cross_instance"]["prefix@1"]
        print(f"{name}: collision {entry['collision']:.4f} "
              f"P3 cross-instance gap@1 {cross['role_gap']:+.5f} "
              f"ci {[round(v, 5) for v in cross['role_gap_ci95']]} "
              f"P3 within gap@1 {entry['P3']['prefix@1']['role_gap']:+.4f} "
              f"(same {entry['P3']['prefix@1']['same_orbit_share']:.4f} / "
              f"diff {entry['P3']['prefix@1']['different_orbit_share']:.4f}) "
              f"merge {entry['merge']['observed']:.4f} vs {entry['merge']['control']:.4f} "
              f"fork {entry['fork']['observed']:.4f} vs {entry['fork']['control']:.4f}",
              flush=True)
        Path(args.out).write_text(json.dumps(report, indent=1) + "\n")
    print("written", args.out)


if __name__ == "__main__":
    main()
