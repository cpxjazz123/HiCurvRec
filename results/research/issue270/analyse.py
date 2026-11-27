"""Issue #270: arm-, item- and structure-level analysis of curvature -> SID -> recall."""
from __future__ import annotations
import json, math
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "results/research/issue270"
G = OUT / "assets/geometry"
S = OUT / "assets/sid"
P = OUT / "assets/pairs"
AN = OUT / "analysis"
BASE = "TIGER_RQ-VAE"
PRIMARY = ["TIGER_RQ-VAE", "TIGER_euclid_local_curve_supervision_rq",
           "TIGER_calibrated_kappa_rq"]
RNG = np.random.default_rng(270)


def load(name: str, kind: str) -> dict:
    path = {"g": G, "s": S, "p": P}[kind] / (name + ".json")
    return json.loads(path.read_text())


def ranks(values):
    order = np.argsort(values)
    r = np.empty(len(values))
    r[order] = np.arange(1, len(values) + 1)
    return r


def spearman(x, y):
    if len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return float("nan")
    return float(stats.spearmanr(x, y).statistic)


def fam_bootstrap(rows, xkey, ykey, draws=2000):
    by = defaultdict(list)
    for r in rows:
        by[r["family"]].append(r)
    fams = list(by)
    out = []
    for _ in range(draws):
        pick = RNG.choice(len(fams), len(fams), replace=True)
        sample = [r for i in pick for r in by[fams[i]]]
        out.append(spearman([r[xkey] for r in sample], [r[ykey] for r in sample]))
    out = np.asarray([v for v in out if np.isfinite(v)])
    return {"n_draws": len(out), "mean": float(out.mean()),
            "ci95": [float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))]}


def permutation_p(x, y, draws=5000):
    obs = spearman(x, y)
    y = np.asarray(y)
    hits = 0
    for _ in range(draws):
        if abs(spearman(x, RNG.permutation(y))) >= abs(obs):
            hits += 1
    return float((hits + 1) / (draws + 1))


def rank_files(final_json: dict) -> int:
    final = Path(final_json["stage3_final"]).parent
    return len(list(final.glob("predictions/rank*.jsonl")))


def main() -> None:
    cohort = json.loads((OUT / "COHORT.json").read_text())
    rows = [r for r in cohort["rows"] if r["usable_geometry"]]
    tiger_g = load(BASE, "g")
    tiger_s = load(BASE, "s")
    tiger_p = load(BASE, "p")
    tiger_abs = tiger_g["mean_abs_error"]
    tiger_l2 = tiger_s["distinct"]["L2"]
    table = []
    for r in rows:
        try:
            g = load(r["arm"], "g"); s = load(r["arm"], "s"); p = load(r["arm"], "p")
        except FileNotFoundError:
            continue
        table.append({
            "arm": r["arm"], "family": r["family"], "binding": r["binding"],
            "record_verified": r["record_verified"], "protocol": "single" if rank_files(r) == 1 else "multi",
            "recall": r["recall@10"], "delta_recall": r["recall@10"] - tiger_p["hit@10"],
            "hit10": p["hit@10"], "ndcg10": p["ndcg@10_from_rank"],
            "mean_abs": g["mean_abs_error"], "improve_abs": tiger_abs - g["mean_abs_error"],
            "sign_acc": g["sign_accuracy"], "L2": s["distinct"]["L2"],
            "l2_ratio": s["distinct"]["L2"] / tiger_l2, "collision": s["collision"],
            "share3": s["mean_all3_L3_share"], "share1": s["mean_all3_L1_share"],
            "l2_per_l1": s["l2_per_l1"],
            "j": (json.loads((G / (r["arm"] + ".distortion.json")).read_text())["mean_j"]
                  if (G / (r["arm"] + ".distortion.json")).is_file() else None),
            "j_kappa": (json.loads((G / (r["arm"] + ".distortion_kappa.json")).read_text())["mean_j"]
                        if (G / (r["arm"] + ".distortion_kappa.json")).is_file() else None),
        })
    AN.mkdir(parents=True, exist_ok=True)
    x = [r["improve_abs"] for r in table]
    y = [r["delta_recall"] for r in table]
    arm_level = {
        "schema": "genrec.issue270.arm_level/1", "n": len(table),
        "spearman": spearman(x, y), "pearson": float(stats.pearsonr(x, y).statistic),
        "kendall": float(stats.kendalltau(x, y).statistic),
        "permutation_p": permutation_p(x, y),
        "family_bootstrap": fam_bootstrap(table, "improve_abs", "delta_recall"),
        "single_protocol": {"n": sum(r["protocol"] == "single" for r in table),
                            "spearman": spearman([r["improve_abs"] for r in table if r["protocol"] == "single"],
                                                 [r["delta_recall"] for r in table if r["protocol"] == "single"])},
        "multi_protocol": {"n": sum(r["protocol"] == "multi" for r in table),
                           "spearman": spearman([r["improve_abs"] for r in table if r["protocol"] == "multi"],
                                                [r["delta_recall"] for r in table if r["protocol"] == "multi"])},
        "conditional_sid": {},
        "leave_one_family": {},
        "table": table,
    }
    strata_samples = {}
    for keep in (True, False):
        sub = [r for r in table if (r["l2_ratio"] >= 0.9) == keep]
        name = "l2_ratio>=0.9" if keep else "l2_ratio<0.9"
        arm_level["conditional_sid"][name] = {
            "n": len(sub), "spearman": spearman([r["improve_abs"] for r in sub],
                                                [r["delta_recall"] for r in sub]),
            "family_bootstrap": fam_bootstrap(sub, "improve_abs", "delta_recall") if len(sub) >= 3 else None}
        strata_samples[keep] = sub
    # family-cluster bootstrap of the difference between the two strata
    diffs = []
    fam2 = defaultdict(list)
    for r in table:
        fam2[r["family"]].append(r)
    fams2 = list(fam2)
    for _ in range(2000):
        pick = RNG.choice(len(fams2), len(fams2), replace=True)
        sample = [r for i in pick for r in fam2[fams2[i]]]
        hi = [r for r in sample if r["l2_ratio"] >= 0.9]
        lo = [r for r in sample if r["l2_ratio"] < 0.9]
        if len(hi) >= 3 and len(lo) >= 3:
            diffs.append(spearman([r["improve_abs"] for r in hi], [r["delta_recall"] for r in hi]) -
                         spearman([r["improve_abs"] for r in lo], [r["delta_recall"] for r in lo]))
    diffs = np.asarray([d for d in diffs if np.isfinite(d)])
    arm_level["conditional_sid"]["difference_ci95"] = [
        float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))] if len(diffs) else None
    # mediation: is the SID an observable intermediate?
    d_sid = [r["l2_ratio"] - 1.0 for r in table]
    def ranks_local(v):
        return rankdata(v)
    from scipy.stats import rankdata as _rankdata
    re_ = _rankdata(x); ry_ = _rankdata(y); rs_ = _rankdata(d_sid)
    def residual(u, v):
        beta = np.polyfit(v, u, 1)
        return u - np.polyval(beta, v)
    mediation = {
        "curvature_to_recall": spearman(x, y),
        "curvature_to_sid": spearman(x, d_sid),
        "sid_to_recall": spearman(d_sid, y),
        "partial_curvature_recall_given_sid": spearman(residual(re_, rs_), residual(ry_, rs_)),
        "n": len(table),
    }
    arm_level["mediation"] = mediation
    fams = sorted({r["family"] for r in table})
    for fam in fams:
        sub = [r for r in table if r["family"] != fam]
        arm_level["leave_one_family"][fam] = {
            "n": len(sub), "spearman": spearman([r["improve_abs"] for r in sub],
                                                [r["delta_recall"] for r in sub])}
    # item-level pairing against TIGER
    tp = np.load(P / (BASE + ".npz"))
    tg = np.load(G / (BASE + ".npz"))
    tiger_item_abs = tg["centre_abs"] / np.maximum(tg["centre_count"], 1)
    tiger_item_absgt = tg["centre_absgt"] / np.maximum(tg["centre_count"], 1)
    tiger_hit = tp["per_target_cur"] / np.maximum(tp["per_target_n"], 1)
    item_level = {"schema": "genrec.issue270.item_level/1", "arms": {}}
    pooled_err, pooled_hit, pooled_idx = [], [], []
    for r in table:
        if r["arm"] == BASE:
            continue
        try:
            gg = np.load(G / (r["arm"] + ".npz")); pp = np.load(P / (r["arm"] + ".npz"))
            ss = np.load(S / (r["arm"] + ".npz"))
            sg = np.load(S / (BASE + ".npz"))
        except FileNotFoundError:
            continue
        n = pp["per_target_n"]
        cur_abs = gg["centre_abs"] / np.maximum(gg["centre_count"], 1)
        cur_hit = pp["per_target_cur"] / np.maximum(n, 1)
        ok = (n >= 5) & (gg["centre_count"] > 0)
        d_err = cur_abs[ok] - tiger_item_abs[ok]
        d_hit = cur_hit[ok] - tiger_hit[ok]
        share_cur = ss["per_item"][ok, 8]
        share_base = sg["per_item"][ok, 8]
        both = (d_err != 0) & (d_hit != 0)
        feats = {"spearman_d_err_d_hit": spearman(d_err, d_hit),
                 "sign_concordance": float(np.mean(np.sign(-d_err[both]) == np.sign(d_hit[both])))
                 if both.any() else None,
                 "sign_ties": float(1 - both.mean())}
        improve = d_err < 0
        sid_improve = share_cur > share_base
        quad = {}
        for a, name_a in ((True, "curv_improve"), (False, "curv_worsen")):
            for b, name_b in ((True, "sid_improve"), (False, "sid_worsen")):
                mask = (improve == a) & (sid_improve == b)
                quad[f"{name_a}|{name_b}"] = {"n": int(mask.sum()),
                                              "mean_d_hit": float(d_hit[mask].mean()) if mask.any() else None}
        feats["quadrants"] = quad
        item_level["arms"][r["arm"]] = feats
        if r["arm"] in PRIMARY:
            item_level.setdefault("primary", {})[r["arm"]] = feats
        pooled_err.append(d_err); pooled_hit.append(d_hit)
        pooled_idx.append(np.flatnonzero(ok))
    all_err = np.concatenate(pooled_err); all_hit = np.concatenate(pooled_hit)
    item_ids = np.concatenate(pooled_idx).astype(np.int64)
    both = (all_err != 0) & (all_hit != 0)
    item_level["pooled"] = {"n": int(len(all_err)), "spearman": spearman(all_err, all_hit),
                            "sign_concordance": float(np.mean(np.sign(-all_err[both]) == np.sign(all_hit[both]))),
                            "sign_ties": float(1 - both.mean())}
    from scipy.stats import rankdata
    re_all, rh_all = rankdata(all_err), rankdata(all_hit)
    uniq = np.unique(item_ids)
    boots = []
    for _ in range(2000):
        draw = RNG.choice(uniq, len(uniq), replace=True)
        mult = np.zeros(24587, dtype=np.float64)
        np.add.at(mult, draw, 1.0)
        w = mult[item_ids]
        keep = w > 0
        ww = w[keep]
        mx = np.average(re_all[keep], weights=ww); my = np.average(rh_all[keep], weights=ww)
        cov = np.average((re_all[keep] - mx) * (rh_all[keep] - my), weights=ww)
        vx = np.average((re_all[keep] - mx) ** 2, weights=ww)
        vy = np.average((rh_all[keep] - my) ** 2, weights=ww)
        boots.append(cov / math.sqrt(vx * vy) if vx > 0 and vy > 0 else np.nan)
    boots = np.asarray([b for b in boots if np.isfinite(b)])
    item_level["pooled"]["bootstrap_ci95"] = [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))]
    item_level["pooled"]["bootstrap_draws"] = int(len(boots))
    # #265 vs #269 diagnosis
    a265 = np.load(G / "TIGER_euclid_local_curve_supervision_rq.npz")
    a269 = np.load(G / "TIGER_calibrated_kappa_rq.npz")
    p265 = np.load(P / "TIGER_euclid_local_curve_supervision_rq.npz")
    p269 = np.load(P / "TIGER_calibrated_kappa_rq.npz")
    s265 = np.load(S / "TIGER_euclid_local_curve_supervision_rq.npz")
    s269 = np.load(S / "TIGER_calibrated_kappa_rq.npz")
    n = p269["per_target_n"]
    ok = (n >= 5) & (a265["centre_count"] > 0) & (a269["centre_count"] > 0)
    d_err = (a269["centre_abs"] / np.maximum(a269["centre_count"], 1))[ok] - \
            (a265["centre_abs"] / np.maximum(a265["centre_count"], 1))[ok]
    d_hit = (p269["per_target_cur"] / np.maximum(n, 1))[ok] - (p265["per_target_cur"] / np.maximum(n, 1))[ok]
    # L2 split/merge through shared items
    l2_265 = s265["per_item"][ok, 1].astype(np.int64)
    l2_269 = s269["per_item"][ok, 1].astype(np.int64)
    pairs = defaultdict(set)
    for x_, y_ in zip(l2_265, l2_269):
        pairs[int(x_)].add(int(y_))
    split = sum(1 for v in pairs.values() if len(v) > 1)
    rev = defaultdict(set)
    for x_, y_ in zip(l2_265, l2_269):
        rev[int(y_)].add(int(x_))
    merge = sum(1 for v in rev.values() if len(v) > 1)
    diagnosis = {
        "schema": "genrec.issue270.diag_265_269/1",
        "items": int(ok.sum()),
        "delta_abs_269_265": float(load("TIGER_calibrated_kappa_rq", "g")["mean_abs_error"]
                                    - load("TIGER_euclid_local_curve_supervision_rq", "g")["mean_abs_error"]),
        "delta_abs_269_vs_tiger": float(load("TIGER_calibrated_kappa_rq", "g")["mean_abs_error"] - tiger_abs),
        "delta_abs_265_vs_tiger": float(load("TIGER_euclid_local_curve_supervision_rq", "g")["mean_abs_error"] - tiger_abs),
        "delta_recall_269_265": float(table[[r["arm"] for r in table].index("TIGER_calibrated_kappa_rq")]["recall"]
                                      - table[[r["arm"] for r in table].index("TIGER_euclid_local_curve_supervision_rq")]["recall"]),
        "item_spearman_d_err_d_hit": spearman(d_err, d_hit),
        "item_sign_concordance": float(np.mean(np.sign(-d_err[(d_err != 0) & (d_hit != 0)])
                                                == np.sign(d_hit[(d_err != 0) & (d_hit != 0)]))),
        "item_sign_ties": float(1 - np.mean((d_err != 0) & (d_hit != 0))),
        "l2_groups_265": len(pairs), "l2_groups_split_into_multiple_269": split,
        "l2_groups_269": len(rev), "l2_groups_merged_from_multiple_265": merge,
        "hit10_gain": int((p269["delta_hit10"] == 1).sum()),
        "hit10_loss": int((p269["delta_hit10"] == -1).sum()),
    }
    # counterexamples and class strata for the primary arms
    counter = {}
    centre_cache = OUT / "assets/centre_ids.npy"
    if not centre_cache.is_file():
        ids = []
        for shard in sorted((REPO / "assets/shards").glob("shard_*.npz")):
            with np.load(shard) as blob:
                ids.append(blob["centre"][blob["valid"]].astype(np.int32))
        np.save(centre_cache, np.concatenate(ids))
    centre_ids = np.load(centre_cache)
    for arm in PRIMARY:
        if arm == BASE:
            continue
        gg = np.load(G / (arm + ".npz")); pp = np.load(P / (arm + ".npz"))
        n = pp["per_target_n"]
        cur_abs = gg["centre_abs"] / np.maximum(gg["centre_count"], 1)
        d_err = cur_abs - tiger_item_abs
        d_hit = (pp["per_target_cur"] / np.maximum(n, 1)) - tiger_hit
        ok = (n >= 5) & (gg["centre_count"] > 0)
        idx = np.flatnonzero(ok)
        order = idx[np.argsort(d_err[idx])]
        improve_lost = [int(i) for i in order[:10] if d_hit[i] < 0][:5]
        order2 = idx[np.argsort(-d_err[idx])]
        worsen_gain = [int(i) for i in order2[:10] if d_hit[i] > 0][:5]
        # class strata via the dominant class of each item
        cls = gg["cls"].astype(np.int32)
        counts = np.zeros((128, 24587), dtype=np.float64)
        np.add.at(counts, (cls, centre_ids), 1.0)
        dominant = counts.argmax(0)
        per_class = {}
        for k in range(128):
            mask = ok & (dominant == k)
            if mask.sum() < 20:
                continue
            per_class[str(k)] = {"items": int(mask.sum()),
                                 "mean_d_abs": float(d_err[mask].mean()),
                                 "mean_d_hit": float(d_hit[mask].mean()),
                                 "n_samples": int(n[mask].sum())}
        counter[arm] = {"improve_but_hit_loss": improve_lost,
                        "worsen_but_hit_gain": worsen_gain,
                        "class_strata": per_class}
    # class consistency across the primary arms
    consistency = {}
    for k in range(128):
        vals = [counter[a]["class_strata"][str(k)]["mean_d_hit"]
                for a in counter if str(k) in counter[a]["class_strata"]]
        if len(vals) >= 2:
            consistency[str(k)] = {"arms": len(vals), "vals": vals,
                                   "same_sign": bool(all(v > 0 for v in vals) or all(v < 0 for v in vals))}
    # FDR over the pre-registered headline tests
    pvals = {"arm_spearman_P1": arm_level["permutation_p"]}
    # BH over the tests with an exact/permutation p; the others report their bootstrap CI
    ordered = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(ordered)
    fdr = {}
    prev = 1.0
    for rank, (name, p) in reversed(list(enumerate(ordered, 1))):
        prev = min(prev, p * m / rank)
    for rank, (name, p) in enumerate(ordered, 1):
        fdr[name] = {"p": p, "bh_q": min(1.0, p * m / rank)}
    # variance by family: between-family spread of delta_recall vs within
    fam_means = defaultdict(list)
    for r in table:
        fam_means[r["family"]].append(r["delta_recall"])
    family_sd = float(np.std([np.mean(v) for v in fam_means.values()]))
    within_sd = float(np.mean([np.std(v) if len(v) > 1 else 0.0 for v in fam_means.values()]))
    out = {"arm_level": arm_level, "item_level": item_level, "diagnosis": diagnosis,
           "family_variance": {"between_family_sd": family_sd, "within_family_sd": within_sd,
                               "n_families": len(fam_means)},
           "p_values": pvals, "fdr": fdr}
    out["counterexamples"] = counter
    out["class_consistency"] = consistency
    (AN / "RELATION.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"arm_n": len(table), "arm_spearman": arm_level["spearman"],
                      "perm_p": arm_level["permutation_p"],
                      "boot_ci": arm_level["family_bootstrap"]["ci95"],
                      "item_pooled_spearman": item_level["pooled"]["spearman"],
                      "diag_delta_abs_269_265": diagnosis["delta_abs_269_265"]}, indent=1))


if __name__ == "__main__":
    main()
