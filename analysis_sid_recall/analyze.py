"""The one analysis script: every R2 pair, one JSON, one table.

    python3 analysis_sid_recall/analyze.py            # analyse r2/<NN>/, write report
    python3 analysis_sid_recall/analyze.py --download # first mirror r2:genrec into r2/
    python3 analysis_sid_recall/analyze.py --check     # verify the mirror against the ledger

Pairs come from `results/research/issue139/r2_ledger.json` (number -> arm/key,
per-file sha256). Each `r2/<NN>/` carries exactly what was archived: the Stage 2
source folder, `item_sids.json`, `rqvae_best.pth`, `recall.json`.

Per pair, from the identifier table alone (labels of the table are tokens, the
behaviour needed for hit rates is the TRAIN split only):

  l1_codes           distinct level-1 codes actually used (of 256)
  l1_er              exp(entropy of the level-1 distribution): effective number
  l1_max_share       largest level-1 code's share of items
  l2_prefixes        distinct (t1,t2) prefixes
  l3_distinct        distinct (t1,t2,t3) triples
  l3_collision       1 - l3_distinct / n_items: share of items sharing its triple
  l3_gain_bits       log2(n_items / l3_distinct): bits the triple does not separate

and prefix hit rates on TRAIN-only pairs (a fixed sample, seed 20261109): the
share of observed (history item -> target) transitions whose level-k prefix is
identical, next to the same statistic on random item pairs, with the gap.

Across pairs: Spearman rho of each metric against `test_recall@10`, with a
percentile bootstrap interval over pairs (2000 resamples); n = number of pairs,
printed next to every number because 28 arms are 28 arms, not a population.
"""
from __future__ import annotations

import random
import math
import json
import sys
import hashlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
ISSUE139 = REPO / "results" / "research" / "issue139"
TRAIN = REPO / "results" / "stage0_build_parquet" / "train.parquet"
LEDGER = ISSUE139 / "r2_ledger.json"
MIRROR = HERE / "r2"
REPORT_JSON = HERE / "report.json"
REPORT_MD = HERE / "report.md"

SEED = 20261109
POSITIVES = 200_000
NEGATIVES = 200_000
BOOT = 2000


def mirror(download: bool, check_only: bool) -> int:
    sys.path.insert(0, str(ISSUE139))
    import r2_layout as L

    ledger = json.loads(LEDGER.read_text())
    MIRROR.mkdir(exist_ok=True)
    bad, verified = [], 0
    for number in sorted(ledger):
        entry = ledger[number]
        if entry.get("retired"):
            continue
        dest = MIRROR / number
        if download and not check_only:
            dest.mkdir(exist_ok=True)
            done = L.rclone(["copy", "%s/%s/" % (L.BUCKET, number), str(dest),
                             "--s3-no-check-bucket", "--transfers", "16",
                             "--checkers", "32", "-q"])
            if done.returncode != 0:
                print("[%s] copy failed: %s" % (number, (done.stderr or done.stdout)[-300:]))
        files = entry.get("files", {})
        missing = [rel for rel in files if not (dest / rel).is_file()]
        wrong = [rel for rel, sha in files.items()
                 if (dest / rel).is_file() and L.sha256(dest / rel) != sha]
        verified += len(files) - len(missing) - len(wrong)
        if missing or wrong:
            bad.append((number, missing[:3], wrong[:3]))
        print("[%s] %-46s files=%d missing=%d bad=%d"
              % (number, entry.get("arm") or entry.get("key", "")[:40],
                 len(files), len(missing), len(wrong)))
    print("verified files: %d; problems: %d" % (verified, len(bad)))
    return 0 if not bad else 1


def train_pairs(seed: int):
    """Observed (history -> target) transitions and random unobserved pairs."""
    import pyarrow.parquet as pq

    table = pq.read_table(TRAIN)
    hist = table.column("history").to_pylist()
    target = table.column("target").to_pylist()
    rng = random.Random(seed)
    positives, seen = set(), set()
    order = list(range(len(target)))
    rng.shuffle(order)
    for row in order:
        h, t = hist[row], target[row]
        if t <= 0:
            continue
        for item in h or ():
            if item <= 0 or item == t:
                continue
            pair = (item, t) if item < t else (t, item)
            if pair in seen:
                continue
            seen.add(pair)
            positives.add(pair)
            if len(positives) >= POSITIVES:
                break
        if len(positives) >= POSITIVES:
            break
    items = sorted({x for pair in positives for x in pair})
    lo, hi = (min(items), max(items)) if items else (1, 1)
    negatives, guard = set(), 0
    while len(negatives) < NEGATIVES and guard < NEGATIVES * 50:
        guard += 1
        a, b = rng.randint(lo, hi), rng.randint(lo, hi)
        if a == b:
            continue
        pair = (a, b) if a < b else (b, a)
        if pair in seen:
            continue
        negatives.add(pair)
    return sorted(positives), sorted(negatives)


def sid_metrics(tokens: list[list[int]], positives, negatives) -> dict:
    n = len(tokens)
    level = [[tuple(t[:k]) for t in tokens] for k in (1, 2, 3)]
    out = {"n_items": n}
    counts = {}
    for t in tokens:
        counts[t[0]] = counts.get(t[0], 0) + 1
    entropy = -sum((c / n) * math.log2(c / n) for c in counts.values())
    out["l1_codes"] = len(counts)
    out["l1_er"] = 2.0 ** entropy
    out["l1_max_share"] = max(counts.values()) / n
    out["l2_prefixes"] = len(set(level[1]))
    l3 = set(level[2])
    out["l3_distinct"] = len(l3)
    out["l3_collision"] = 1.0 - len(l3) / n
    out["l3_gain_bits"] = math.log2(n / len(l3)) if l3 else None

    def hit(sample, k):
        index = level[k - 1]
        ok = total = 0
        for a, b in sample:
            if a >= n or b >= n:
                continue
            total += 1
            ok += index[a] == index[b]
        return ok / total if total else None

    for k in (1, 2, 3):
        pos, neg = hit(positives, k), hit(negatives, k)
        out["prefix_overlap@%d_pos" % k] = pos
        out["prefix_overlap@%d_neg" % k] = neg
        out["prefix_overlap@%d_gap" % k] = (pos - neg) if pos is not None and neg is not None else None
    return out


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


def bootstrap(xs, ys, seed=SEED):
    rng = random.Random(seed)
    n = len(xs)
    rhos = []
    for _ in range(BOOT):
        idx = [rng.randrange(n) for _ in range(n)]
        rho = spearman([xs[i] for i in idx], [ys[i] for i in idx])
        if rho is not None:
            rhos.append(rho)
    rhos.sort()
    if not rhos:
        return None, None
    lo = rhos[int(0.025 * (len(rhos) - 1))]
    hi = rhos[int(0.975 * (len(rhos) - 1))]
    return lo, hi


def main() -> int:
    if "--download" in sys.argv or "--check" in sys.argv:
        return mirror("--download" in sys.argv, "--check" in sys.argv)

    ledger = json.loads(LEDGER.read_text())
    positives, negatives = train_pairs(SEED)
    pairs, rows = [], []
    for number in sorted(ledger):
        if ledger[number].get("retired"):
            continue
        folder = MIRROR / number
        sid_path, recall_path = folder / "item_sids.json", folder / "recall.json"
        stage2_only = folder / "stage2_record.json"
        if not sid_path.is_file() or not (recall_path.is_file() or stage2_only.is_file()):
            print("[%s] incomplete mirror, skipped" % number)
            continue
        raw = json.loads(sid_path.read_text())
        tokens = [raw[k] for k in sorted(raw, key=lambda s: int(s))]
        tokens = [t for t in tokens if len(t) >= 3]
        recall = json.loads(recall_path.read_text()) if recall_path.is_file() else {}
        row = {"number": number, "arm": ledger[number].get("arm") or ledger[number].get("key"),
               "test_recall@10": recall.get("test_recall@10"),
               "test_recall@5": recall.get("test_recall@5"),
               "test_ndcg@10": recall.get("test_ndcg@10")}
        row.update(sid_metrics(tokens, positives, negatives))
        row["sid_sha256"] = hashlib.sha256(sid_path.read_bytes()).hexdigest()
        pairs.append(row)
    metrics = ["l1_codes", "l1_er", "l1_max_share", "l2_prefixes", "l3_distinct",
               "l3_collision", "l3_gain_bits", "prefix_overlap@1_gap",
               "prefix_overlap@2_gap", "prefix_overlap@3_gap"]
    scored = [p for p in pairs if p["test_recall@10"] is not None]
    seen_sha, unique = set(), []
    for p in scored:
        if p["sid_sha256"] in seen_sha:
            continue
        seen_sha.add(p["sid_sha256"])
        unique.append(p)
    recalls = [p["test_recall@10"] for p in scored]
    correlations = {}
    for m in metrics:
        xs = [p[m] for p in scored]
        if any(v is None for v in xs):
            correlations[m] = {"n": len([v for v in xs if v is not None]), "rho": None}
            continue
        rho = spearman(xs, recalls)
        lo, hi = bootstrap(xs, recalls)
        correlations[m] = {"n": len(xs), "rho": rho, "ci95": [lo, hi]}
    unique_corr = {}
    for m in metrics:
        xs = [p[m] for p in unique]
        if any(v is None for v in xs) or len(xs) < 4:
            unique_corr[m] = {"n": len(xs), "rho": None}
            continue
        unique_corr[m] = {"n": len(xs), "rho": spearman(xs, [p["test_recall@10"] for p in unique])}
    result = {"seed": SEED, "positives": len(positives), "negatives": len(negatives),
              "pairs": pairs, "correlations": correlations,
              "correlations_unique_sid": unique_corr}
    REPORT_JSON.write_text(json.dumps(result, indent=2) + "\n")

    lines = ["# R2 pairs, one pass over %d pairs" % len(pairs), "",
             "Sample: %d TRAIN-only transitions, %d random pairs (seed %d)."
             % (len(positives), len(negatives), SEED), "",
             "| NN | arm | R@10 | L1 codes | L1 ER | L1 max | L2 | L3 | L3 coll | gain bits |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for p in pairs:
        lines.append("| %s | `%s` | %s | %d | %.1f | %.3f | %d | %d | %.3f | %.2f |"
                     % (p["number"], p["arm"],
                        "%.4f" % p["test_recall@10"] if p["test_recall@10"] is not None else "— (Stage2-only)",
                        p["l1_codes"],
                        p["l1_er"], p["l1_max_share"], p["l2_prefixes"], p["l3_distinct"],
                        p["l3_collision"], p["l3_gain_bits"]))
    lines += ["", "## Spearman against test_recall@10 (n = %d pairs with Stage 3)" % len(scored), "",
              "| metric | rho | 95% bootstrap |", "|---|---|---|"]
    for m, c in correlations.items():
        ci = c.get("ci95") or [None, None]
        lines.append("| %s | %s | %s |" % (
            m,
            "%.3f" % c["rho"] if c["rho"] is not None else "n/a",
            "[%.3f, %.3f]" % (ci[0], ci[1]) if ci[0] is not None else "n/a"))
    lines += ["", "## Same, one row per distinct item_sids.json (n = %d)" % len(unique), "",
              "| metric | rho |", "|---|---|"]
    for m, c in unique_corr.items():
        lines.append("| %s | %s |" % (m, "%.3f" % c["rho"] if c["rho"] is not None else "n/a"))
    REPORT_MD.write_text("\n".join(lines) + "\n")
    print("pairs: %d; report: %s" % (len(pairs), REPORT_MD))
    for m, c in correlations.items():
        print("%-24s rho=%s n=%d" % (m, "%.3f" % c["rho"] if c["rho"] is not None else "n/a", c["n"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
