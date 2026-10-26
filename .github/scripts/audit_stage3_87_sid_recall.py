#!/usr/bin/env python3
"""Recover historical Stage3 runs and correlate SID structure with Recall@10."""
import ast
import csv
import hashlib
import json
import math
import os
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

from scipy.stats import kendalltau, pearsonr, rankdata, spearmanr

AUDIT_REF = os.environ.get("AUDIT_REF", "84b4486a8a21c0c1b64de5d9bc0ab0c14254bd74")
EXPECTED_RUNS = int(os.environ.get("EXPECTED_RUNS", "87"))
OUT = Path(os.environ.get("OUTPUT_DIR", "artifacts/stage3-87-sid-recall"))
ROOT = "results/stage3_T5Train/"
CACHE = {}


def git(*args, check=True):
    p = subprocess.run(["git", *args], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if check and p.returncode:
        raise RuntimeError("git command failed: " + p.stderr.decode("utf-8", "replace")[-1000:])
    return p


def versions(path):
    if not path:
        return []
    if path in CACHE:
        return CACHE[path]
    log = git("log", "--format=%H %ct", AUDIT_REF, "--", path, check=False)
    seen, result = set(), []
    for line in log.stdout.decode("utf-8", "replace").splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        commit, timestamp = parts[0], int(parts[1])
        revpath = f"{commit}:{path}"
        blob = git("rev-parse", "--verify", revpath, check=False)
        if blob.returncode:
            continue
        blob_id = blob.stdout.decode().strip()
        if blob_id in seen:
            continue
        seen.add(blob_id)
        content = git("show", revpath, check=False)
        if not content.returncode:
            result.append({
                "blob_id": blob_id, "commit": commit, "commit_ts": timestamp,
                "content": content.stdout,
            })
    CACHE[path] = result
    return result


def latest(path):
    found = versions(path)
    return found[0] if found else None


def normalize_path(path):
    if not path:
        return None
    value = str(path).replace("\\", "/")
    for marker in ("results/", "stage3_T5Train/"):
        index = value.rfind(marker)
        if index >= 0:
            return value[index:]
    return value.lstrip("./") if not value.startswith("/") else None


def run_paths():
    cmd = git(
        "log", "--format=", "--name-only", AUDIT_REF, "--",
        ":(glob)results/stage3_T5Train/**/test_final.json", check=False,
    )
    paths = {
        x.strip() for x in cmd.stdout.decode("utf-8", "replace").splitlines()
        if x.strip().startswith(ROOT) and x.strip().endswith("/test_final.json")
    }
    if not paths:
        cmd = git(
            "log", "--format=", "--name-only", AUDIT_REF, "--",
            "results/stage3_T5Train", check=False,
        )
        paths = {
            x.strip() for x in cmd.stdout.decode("utf-8", "replace").splitlines()
            if x.strip().startswith(ROOT) and x.strip().endswith("/test_final.json")
        }
    return sorted(paths)


def inventory_map():
    blob = latest("results/stage3_T5Train/monotonicity/model_inventory.json")
    if not blob:
        return {}
    try:
        inventory = json.loads(blob["content"])
    except Exception:
        return {}
    result = {}
    for model in inventory.get("models", []):
        for run in model.get("stage3_runs", []):
            result[ROOT + run.strip("/")] = {
                "sid_path": normalize_path(model.get("sid_path")),
                "sha_prefix": str(model.get("sid_sha256", "")),
                "sid_mtime": model.get("sid_mtime"),
            }
    return result


def run_meta(run_dir):
    metric = latest(run_dir + "/training_metrics.jsonl")
    log = latest(run_dir + "/HG_Rec.log")
    start = {}
    if metric:
        for line in metric["content"].decode("utf-8", "replace").splitlines():
            try:
                event = json.loads(line)
            except Exception:
                continue
            if event.get("event") == "train_start":
                start = event
                break
    config = {}
    if log:
        for line in log["content"].decode("utf-8", "replace").splitlines():
            if "Configuration:" not in line:
                continue
            try:
                config = ast.literal_eval(line.split("Configuration:", 1)[1].strip())
                break
            except Exception:
                pass
    code_path = start.get("code_path") or config.get("code_path")
    return {
        "code_path": str(code_path or ""),
        "sid_path": normalize_path(code_path),
        "start_ts": start.get("timestamp"),
        "config": config,
        "has_training_metrics": bool(metric),
        "has_hg_rec_log": bool(log),
    }


def sid_data(blob):
    try:
        obj = json.loads(blob["content"])
        keys = sorted(int(k) for k in obj)
        if keys != list(range(len(keys))):
            return None
        rows = [tuple(obj[str(k)]) for k in keys]
        if not rows or any(not row for row in rows):
            return None
        width = len(rows[0])
        if any(len(row) != width or any(not isinstance(v, int) or isinstance(v, bool) for v in row) for row in rows):
            return None
        return rows
    except Exception:
        return None


def features(rows, sha, n_user_tokens):
    n, width = len(rows), len(rows[0])
    result = {
        "sid_sha256": sha,
        "sid_num_items": n,
        "sid_width": width,
        "sid_max_raw_code": max(max(row) for row in rows),
        "sid_vocab_size_from_codes": max(max(row) for row in rows) + n_user_tokens + 3,
    }
    for depth in range(1, min(width, 4) + 1):
        counts = Counter(tuple(row[:depth]) for row in rows)
        probs = [v / n for v in counts.values()]
        result[f"prefix{depth}_distinct"] = len(counts)
        result[f"prefix{depth}_entropy_bits"] = -sum(p * math.log2(p) for p in probs if p)
        result[f"prefix{depth}_collision_share"] = sum(v for v in counts.values() if v > 1) / n
        result[f"prefix{depth}_max_bucket_share"] = max(counts.values()) / n
    for level in range(1, min(width, 4) + 1):
        counts = Counter(row[level - 1] for row in rows)
        probs = [v / n for v in counts.values()]
        result[f"token{level}_distinct"] = len(counts)
        result[f"token{level}_entropy_bits"] = -sum(p * math.log2(p) for p in probs if p)
    return result


def resolve_sid(meta, expected):
    path = meta.get("sid_path")
    if not path:
        return {"sid_status": "no_logged_code_path"}
    candidates = []
    for item in versions(path):
        rows = sid_data(item)
        if rows is None:
            continue
        raw = item["content"]
        item = dict(item)
        item["rows"] = rows
        item["sha256"] = hashlib.sha256(raw).hexdigest()
        item["max_raw"] = max(max(row) for row in rows)
        item["width"] = len(rows[0])
        candidates.append(item)
    if not candidates:
        return {"sid_path": path, "sid_status": "no_sid_blob_in_git_history", "sid_candidate_count": 0}

    config = meta.get("config", {})
    expected_width = config.get("sid_length")
    expected_vocab = config.get("vocab_size")
    n_user_tokens = int(config.get("n_user_tokens", 1) or 1)
    target_ts = float((expected or {}).get("sid_mtime") or meta.get("start_ts") or 0)
    prefix = (expected or {}).get("sha_prefix", "")
    selected = None
    status = ""
    if prefix:
        match = [c for c in candidates if c["sha256"].startswith(prefix)]
        if len(match) == 1:
            selected, status = match[0], "inventory_sha16_match"
    if selected is None:
        def match_config(c):
            width_ok = expected_width is None or c["width"] == int(expected_width)
            vocab = c["max_raw"] + n_user_tokens + 3
            vocab_ok = expected_vocab is None or vocab == int(expected_vocab)
            return width_ok and vocab_ok
        exact = [c for c in candidates if match_config(c)]
        pool = exact or [
            c for c in candidates
            if expected_width is None or c["width"] == int(expected_width)
        ] or candidates
        selected = min(pool, key=lambda c: abs(c["commit_ts"] - target_ts))
        if len(exact) == 1:
            status = "unique_vocab_width_match"
        elif len(exact) > 1:
            status = "ambiguous_vocab_width_nearest_time"
        elif len(candidates) == 1:
            status = "single_historical_sid_version"
        else:
            status = "no_config_match_nearest_time"

    sid_path = path
    result = {
        "sid_path": sid_path,
        "sid_blob_file": "sid_blobs/" + selected["sha256"] + ".json",
        "sid_source_commit": selected["commit"],
        "sid_source_commit_time": selected["commit_ts"],
        "sid_candidate_count": len(candidates),
        "sid_status": status,
        "sid_runtime_vocab_size": expected_vocab,
        "sid_runtime_length": expected_width,
        "sid_hash_matches_inventory": (
            selected["sha256"].startswith(prefix) if prefix else None
        ),
        **features(selected["rows"], selected["sha256"], n_user_tokens),
        "_blob": selected["content"],
    }
    return result


def recall_value(path):
    blob = latest(path)
    if not blob:
        return None, {}
    try:
        data = json.loads(blob["content"])
    except Exception:
        return None, {}
    for key in ("test_recall@10", "test_R@10", "recall@10", "R@10"):
        try:
            value = float(data[key])
            return (value if math.isfinite(value) else None), data
        except Exception:
            pass
    return None, data


def family_of(sid_path, run_dir):
    if sid_path and sid_path.startswith("results/stage2_RQ-VAE/"):
        return sid_path[len("results/stage2_RQ-VAE/"):].split("/")[0]
    return run_dir[len(ROOT):].split("/")[0]


def collect():
    paths = run_paths()
    inv = inventory_map()
    rows = []
    for test_path in paths:
        run_dir = test_path.rsplit("/", 1)[0]
        recall, final = recall_value(test_path)
        meta = run_meta(run_dir)
        sid = resolve_sid(meta, inv.get(run_dir))
        row = {
            "run_id": run_dir[len(ROOT):],
            "test_final_path": test_path,
            "recall_at_10": recall,
            "ndcg_at_10": final.get("test_ndcg@10"),
            "n_eval": final.get("n_eval"),
            "code_path_logged": meta["code_path"],
            "training_metrics_present": meta["has_training_metrics"],
            "hg_rec_log_present": meta["has_hg_rec_log"],
            "sid_family": family_of(meta.get("sid_path"), run_dir),
            **{k: v for k, v in sid.items() if k != "_blob"},
        }
        rows.append(row)
        if sid.get("_blob"):
            dest = OUT / "sid_blobs" / (sid["sid_sha256"] + ".json")
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(sid["_blob"])
    return rows, paths


def finite(x):
    try:
        return x is not None and math.isfinite(float(x))
    except Exception:
        return False


def bh(pvalues):
    q = [None] * len(pvalues)
    valid = sorted((p, i) for i, p in enumerate(pvalues) if finite(p))
    m, running = len(valid), 1.0
    for rank in range(m, 0, -1):
        p, index = valid[rank - 1]
        running = min(running, p * m / rank)
        q[index] = min(1.0, running)
    return q


def family_adjusted(x, y, families):
    groups = defaultdict(list)
    for i, family in enumerate(families):
        groups[family].append(i)
    groups = [idx for idx in groups.values() if len(idx) > 1]
    indices = [i for group in groups for i in group]
    if len(indices) < 4:
        return None, None
    rx, ry = rankdata([x[i] for i in indices]), rankdata([y[i] for i in indices])
    by_family = defaultdict(list)
    for j, i in enumerate(indices):
        by_family[families[i]].append(j)
    for idx in by_family.values():
        rx[idx] -= rx[idx].mean()
        ry[idx] -= ry[idx].mean()
    if len(set(rx)) < 2 or len(set(ry)) < 2:
        return None, None
    value = pearsonr(rx, ry)
    return float(value.statistic), float(value.pvalue)


def correlations(rows, scope):
    if not rows:
        return []
    features_list = [
        k for k in rows[0]
        if k.startswith("prefix") or k.startswith("token")
    ] + ["sid_num_items", "sid_width", "sid_max_raw_code", "sid_vocab_size_from_codes"]
    output = []
    for feature in features_list:
        subset = [r for r in rows if finite(r.get(feature)) and finite(r.get("recall_at_10"))]
        if len(subset) < 3:
            continue
        x = [float(r[feature]) for r in subset]
        y = [float(r["recall_at_10"]) for r in subset]
        try:
            sp, kd = spearmanr(x, y), kendalltau(x, y)
            rho, p = float(sp.statistic), float(sp.pvalue)
            tau, kp = float(kd.statistic), float(kd.pvalue)
        except Exception:
            rho = p = tau = kp = None
        ar, ap = family_adjusted(x, y, [str(r.get("sid_family", "unknown")) for r in subset])
        output.append({
            "scope": scope, "feature": feature, "n": len(subset),
            "spearman_rho": rho, "spearman_p": p,
            "spearman_q_bh": None, "kendall_tau": tau, "kendall_p": kp,
            "family_adjusted_rank_r": ar, "family_adjusted_p": ap,
        })
    qs = bh([r["spearman_p"] for r in output])
    for row, q in zip(output, qs):
        row["spearman_q_bh"] = q
    return output


def unique_sids(rows):
    groups = defaultdict(list)
    for row in rows:
        if row.get("sid_sha256") and finite(row.get("recall_at_10")):
            groups[row["sid_sha256"]].append(row)
    result = []
    for group in groups.values():
        row = dict(group[0])
        row["recall_at_10"] = sum(float(r["recall_at_10"]) for r in group) / len(group)
        result.append(row)
    return result


def write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    cols = []
    for row in rows:
        for key in row:
            if not key.startswith("_") and key not in cols:
                cols.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fmt(value):
    return "NA" if not finite(value) else f"{float(value):.4g}"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, paths = collect()
    numeric = [r for r in rows if finite(r.get("recall_at_10"))]
    resolved = [r for r in numeric if r.get("sid_sha256")]
    strict_status = {
        "inventory_sha16_match", "unique_vocab_width_match",
        "single_historical_sid_version",
    }
    strict = [r for r in resolved if r.get("sid_status") in strict_status]
    fingerprinted = [r for r in resolved if r.get("sid_status") == "inventory_sha16_match"]
    unique = unique_sids(resolved)
    unique_fingerprinted = unique_sids(fingerprinted)
    corr = (
        correlations(resolved, "all_run_selected_sid")
        + correlations(unique, "unique_sid_mean_recall")
        + correlations(strict, "inventory_or_unique_version")
        + correlations(unique_fingerprinted, "inventory_fingerprint_unique_sid")
    )
    write_csv(OUT / "historical_87_pairs.csv", rows)
    write_csv(OUT / "sid_recall_correlations.csv", corr)

    status_counts = dict(Counter(str(r.get("sid_status")) for r in rows))
    summary = {
        "audit_ref": AUDIT_REF,
        "expected_historical_runs": EXPECTED_RUNS,
        "historical_test_final_paths": len(paths),
        "numeric_recall_rows": len(numeric),
        "rows_with_logged_code_path": sum(bool(r.get("code_path_logged")) for r in rows),
        "rows_with_selected_sid_blob": len(resolved),
        "inventory_or_unique_version_rows": len(strict),
        "inventory_sha16_match_rows": len(fingerprinted),
        "unique_inventory_fingerprint_sid_hashes": len(unique_fingerprinted),
        "ambiguous_version_rows": sum(str(r.get("sid_status", "")).startswith("ambiguous") for r in rows),
        "unresolved_rows": sum(not r.get("sid_sha256") for r in rows),
        "unique_selected_sid_hashes": len(unique),
        "sid_status_counts": status_counts,
    }
    (OUT / "metadata.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report = [
        "# Historical Stage3 SID–Recall correlation audit", "",
        f"- Audit cutoff: {AUDIT_REF}",
        f"- Expected runs: {EXPECTED_RUNS}; distinct historical test_final paths: {len(paths)}",
        f"- Numeric Recall@10: {len(numeric)}",
        f"- SID blobs selected from Git history: {len(resolved)}",
        f"- Inventory fingerprint or unique configuration/version matches: {len(strict)}",
        f"- Stored 16-character inventory fingerprint matches: {len(fingerprinted)} rows / {len(unique_fingerprinted)} unique SIDs",
        f"- Ambiguous version choices: {summary['ambiguous_version_rows']}; unresolved rows: {summary['unresolved_rows']}",
        f"- Unique selected SID SHA-256 values: {len(unique)}", "",
        "The pair CSV contains one row per historical test_final path, with Recall@10, "
        "SID path, selected SID SHA-256, resolution status, and derived structural features. "
        "The artifact includes the selected historical SID JSON files under sid_blobs/.", "",
    ]
    for scope in ("all_run_selected_sid", "unique_sid_mean_recall", "inventory_or_unique_version", "inventory_fingerprint_unique_sid"):
        subset = [r for r in corr if r["scope"] == scope]
        report.extend([f"## {scope.replace('_', ' ')}", ""])
        if not subset:
            report.extend(["No complete observations for this scope.", ""])
            continue
        subset.sort(key=lambda r: abs(float(r["spearman_rho"])) if finite(r["spearman_rho"]) else -1, reverse=True)
        report.extend([
            "| Feature | n | Spearman rho | p | BH q | Kendall tau | Family-adjusted rank r |",
            "|:--|--:|--:|--:|--:|--:|--:|",
        ])
        for r in subset[:12]:
            report.append(
                f"| {r['feature']} | {r['n']} | {fmt(r['spearman_rho'])} | "
                f"{fmt(r['spearman_p'])} | {fmt(r['spearman_q_bh'])} | "
                f"{fmt(r['kendall_tau'])} | {fmt(r['family_adjusted_rank_r'])} |"
            )
        report.append("")
    robust = [
        r for r in corr
        if r["scope"] == "unique_sid_mean_recall"
        and finite(r.get("spearman_q_bh")) and float(r["spearman_q_bh"]) < 0.05
    ]
    report.extend(["## Conclusion", ""])
    if robust:
        robust.sort(key=lambda r: abs(float(r["spearman_rho"])), reverse=True)
        report.append(
            "After BH correction in the unique-SID analysis, q < 0.05 for: "
            + "; ".join(f"{r['feature']} (rho={fmt(r['spearman_rho'])}, q={fmt(r['spearman_q_bh'])})" for r in robust[:5])
            + ". These are associations across this experiment set, not causal effects."
        )
    else:
        report.append(
            "No feature has BH-adjusted q < 0.05 in the unique-SID analysis. "
            "Run-level correlations are descriptive; repeated SIDs, model-family "
            "differences and protocol variation limit causal interpretation."
        )
    if len(paths) != EXPECTED_RUNS:
        report.append("The discovered run count differs from 87; do not describe this as a complete 87-run analysis.")
    if len(resolved) != EXPECTED_RUNS:
        report.append("Some runs lack a recoverable SID blob; inspect sid_status before claiming full SID coverage.")
    fingerprint_sig = [
        r for r in corr
        if r["scope"] == "inventory_fingerprint_unique_sid"
        and finite(r.get("spearman_q_bh")) and float(r["spearman_q_bh"]) < 0.05
    ]
    if fingerprint_sig:
        report.append(
            "Within the stored-inventory-fingerprint-only unique-SID subset, BH-adjusted "
            "q < 0.05 for: " + "; ".join(
                f"{r['feature']} (rho={fmt(r['spearman_rho'])}, q={fmt(r['spearman_q_bh'])})"
                for r in fingerprint_sig[:5]
            ) + "."
        )
    else:
        report.append(
            f"In the {len(unique_fingerprinted)} unique SIDs matched to the stored 16-character "
            "inventory fingerprint, no tested structural feature has BH-adjusted q < 0.05."
        )
    report.extend([
        "",
        "An inventory SHA-prefix match is direct fingerprint evidence. A unique SID "
        "version matching the logged vocabulary size and SID length is a reconstruction; "
        "the runtime did not record full SID SHA-256 for every run. Ambiguous candidates "
        "are marked in the pair table. Correlation does not establish causality.", "",
    ])
    (OUT / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print((OUT / "report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
