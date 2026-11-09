"""Inventory of every locally completed Stage 2 / Stage 3 experiment pair.

A "model" here is one Stage 2 checkpoint, identified by the content hash of its
Semantic ID file. Several Stage 3 runs can consume the SAME SID file, and those
runs are NOT independent models: they differ only by Stage 3 run-to-run
nondeterminism. They are grouped into one entry that keeps every run, because
their spread is also the only available estimate of how much Recall@10 moves
when the input is held fixed. Any metric that claims to explain Recall has to
be read against that spread.

The SID-to-Stage3 link is taken from ``code_path`` in the run's own HG_Rec.log
configuration, never inferred from directory or file naming.

Protocol comparability is verified from the same log rather than assumed:
seed, beam size, epochs, no_eval, skip_test, test file and the screening
baseline. Runs whose protocol differs, whose SID file is missing or malformed, or
whose SID duplicates another, are excluded with the reason recorded.

Read-only. Nothing here trains, modifies or deletes anything.
"""
from __future__ import annotations

import ast
import csv
import glob
import hashlib
import json
import os
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path("/home/wlia0047/ar57/wenyu/GeneRec")
S2 = ROOT / "results/stage2_RQ-VAE"
S3 = ROOT / "results/stage3_T5Train"
OUT_JSON = ROOT / "results/stage3_T5Train/monotonicity/model_inventory.json"
OUT_CSV = ROOT / "results/stage3_T5Train/monotonicity/model_inventory.csv"

N_ITEMS = 24587
N_LEVELS = 3
# Protocol every comparable run must satisfy, taken from the frozen Stage 3
# configuration rather than from any particular run.
PROTOCOL = {
    "seed": 42,
    "beam_size": 20,
    "num_epochs": 150,
    "no_eval": True,
    "skip_test": False,
    "test_file": "test.parquet",
    "screen_baseline_log": "",
    # Three runs set early_stop=10 together with screen_skip_test_on_fail.
    # Early stopping is forbidden for these comparisons, so those runs are
    # not protocol-comparable and are excluded.
    "early_stop": None,
    "screen_skip_test_on_fail": None,
    # One run tokenized a 5-token SID; every comparable run uses 4 tokens
    # (three codebooks plus the collision extension token).
    "sid_length": 4,
}


def read_config(log: Path) -> dict:
    if not log.exists():
        return {}
    txt = log.read_text(errors="ignore")
    m = re.search(r"Configuration: (\{.*?\})\n", txt, re.S)
    if not m:
        return {}
    try:
        return ast.literal_eval(m.group(1))
    except Exception:
        return {}


def predicted_vocab_size(path: Path) -> int:
    """Reproduce Stage 3's own tokenizer vocabulary size from a SID file.

    data/dataset.py builds ``tokens = code + 1`` over ALL stored columns, takes
    ``max_sid_token = max(tokens)`` and then ``vocab_size = max_sid_token + 3``.
    So ``vocab_size == max(all columns) + 4``. Recomputing it here is what lets a
    run be tied to the SID CONTENT it actually consumed, which the file path
    cannot do: ``curvature_RQ-VAE/item_sids.json`` was overwritten between runs
    and served at least six different vocabularies.
    """
    raw = json.loads(path.read_text())
    return max(max(v) for v in raw.values()) + 4


def sid_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect_sid(path: Path) -> tuple[dict | None, str]:
    try:
        raw = json.loads(path.read_text())
    except Exception as exc:
        return None, f"unreadable SID json: {exc}"
    if not isinstance(raw, dict):
        return None, "SID json is not an object"
    n_items = len(raw)
    widths = {len(v) for v in raw.values() if isinstance(v, (list, tuple))}
    if n_items != N_ITEMS:
        return None, f"SID covers {n_items} items, expected {N_ITEMS}"
    if not widths:
        return None, "SID entries are not lists"
    width = widths.pop()
    if len(widths):
        return None, "SID entries have inconsistent widths"
    if width < N_LEVELS:
        return None, f"SID has {width} tokens, fewer than {N_LEVELS} levels"
    return {
        "n_items": n_items,
        "n_tokens": width,
        "levels_analysed": N_LEVELS,
        "collision_token_present": bool(width > N_LEVELS),
    }, "ok"


def protocol_deviations(cfg: dict) -> list[str]:
    dev = []
    for key, want in PROTOCOL.items():
        got = cfg.get(key)
        if got != want:
            dev.append(f"{key}={got!r} (expected {want!r})")
    return dev


# The repository is reachable under two prefixes that are the same tree (a bind
# mount). Runs recorded either prefix must normalise to one, or path arithmetic
# against the canonical root fails.
PATH_ALIASES = ("/fs04/ar57/wenyu/GeneRec", "/home/wlia0047/ar57/wenyu/GeneRec")


def normalise(path: str) -> str:
    for alias in PATH_ALIASES:
        if path.startswith(alias):
            return str(ROOT) + path[len(alias):]
    return path


def _unique_name(sid_path: str) -> str:
    """Leaf name, disambiguated by the experiment sub-family when it collides."""
    q = Path(normalise(sid_path))
    if not q.is_absolute():
        q = ROOT / q
    rel = q.relative_to(S2)
    parts = rel.parts[:-1]
    leaf = parts[-1]
    return "/".join(parts[-2:]) if len(parts) >= 2 else leaf


def _family(sid_path: str) -> str:
    """Experiment family = the directory under results/stage2_RQ-VAE."""
    q = Path(normalise(sid_path))
    if not q.is_absolute():
        q = ROOT / q
    rel = q.relative_to(S2)
    return rel.parts[0] if len(rel.parts) > 1 else "."


def main() -> None:
    runs = []
    for tf in sorted(glob.glob(str(S3 / "**/test_final.json"), recursive=True)):
        tf = Path(tf)
        d = json.loads(tf.read_text())
        log = tf.parent / "HG_Rec.log"
        cfg = read_config(log)
        # Full path INCLUDING the timestamp directory. Dropping it collapses
        # several distinct runs of the same arm onto one file, which silently
        # hides Stage 3 run-to-run spread.
        run_name = str(tf.parent.relative_to(S3))
        runs.append({
            "run": run_name,
            "test_final": str(tf.relative_to(ROOT)),
            "recall@10": d.get("test_recall@10"),
            "ndcg@10": d.get("test_ndcg@10"),
            "n_eval": d.get("n_eval"),
            "code_path": cfg.get("code_path"),
            "vocab_size": cfg.get("vocab_size"),
            "cfg": cfg,
        })

    excluded: list[dict] = []
    usable: list[dict] = []
    for r in runs:
        if r["recall@10"] is None or r["n_eval"] is None:
            excluded.append({"run": r["run"], "reason": "test_final.json incomplete"})
            continue
        if r["n_eval"] != 57439:
            excluded.append({
                "run": r["run"],
                "reason": f"n_eval={r['n_eval']}, not comparable to 57439",
            })
            continue
        dev = protocol_deviations(r["cfg"])
        if dev:
            excluded.append({"run": r["run"], "reason": "protocol deviation: " + "; ".join(dev)})
            continue
        if not r["code_path"]:
            excluded.append({"run": r["run"], "reason": "no code_path in run configuration"})
            continue
        p = Path(normalise(r["code_path"]))
        if not p.exists():
            excluded.append({"run": r["run"], "reason": f"SID file missing on disk: {p}"})
            continue
        r["sid_path"] = str(p.relative_to(ROOT))
        r["sid_mtime"] = p.stat().st_mtime
        usable.append(r)

    # Group by SID IDENTITY, not by path. The path is not an identity:
    # TIGER_RQ-VAE/item_sids.json served vocab 799 to one run and 813 to
    # another, so grouping by path merged two different SID contents and the
    # verification below then rejected the whole group, silently dropping the
    # 0.053674 baseline. The tokenizer fingerprint (vocab_size, eos_token_id,
    # sid_length) is what pins a run to the SID content it consumed.
    by_sid: dict[tuple, list[dict]] = defaultdict(list)
    for r in usable:
        key = (
            r["sid_path"],
            r["vocab_size"],
            r["cfg"].get("eos_token_id"),
            r["cfg"].get("sid_length"),
        )
        by_sid[key].append(r)

    models = []
    seen_digest: dict[str, str] = {}
    for key4, rs in sorted(by_sid.items()):
        sid_path = key4[0]
        p = ROOT / sid_path
        expected_vocab = predicted_vocab_size(p)
        if expected_vocab != rs[0]["vocab_size"]:
            # The file at this path no longer holds what the run consumed.
            for r in rs:
                excluded.append({
                    "run": r["run"],
                    "reason": (
                        f"SID at {r['sid_path']} no longer matches the run: "
                        f"vocab_size recorded {r['vocab_size']}, file now implies "
                        f"{expected_vocab}; the file was overwritten after the run"
                    ),
                })
            continue
        info, why = inspect_sid(p)
        if info is None:
            for r in rs:
                excluded.append({"run": r["run"], "reason": why})
            continue
        digest = sid_digest(p)
        if digest in seen_digest:
            for r in rs:
                excluded.append({
                    "run": r["run"],
                    "reason": f"SID content identical to {seen_digest[digest]}; same checkpoint",
                })
            continue
        seen_digest[digest] = sid_path
        recalls = sorted(x["recall@10"] for x in rs)
        models.append({
            # Not just the leaf directory: two experiments both contain a
            # D_poincare_cone_s42 (l2_cone_arms and category_cone_arms), and a
            # leaf-only key silently dropped one of them.
            "model": _unique_name(sid_path),
            "sid_path": sid_path,
            "sid_sha256": digest[:16],
            "sid_mtime": p.stat().st_mtime,
            "sid_info": info,
            "family": _family(sid_path),
            "n_stage3_runs": len(rs),
            "recall@10": recalls[len(recalls) // 2],
            "recall@10_runs": recalls,
            "recall@10_spread": (recalls[-1] - recalls[0]) if len(recalls) > 1 else 0.0,
            "ndcg@10": rs[0]["ndcg@10"],
            "n_eval": rs[0]["n_eval"],
            "vocab_size": rs[0]["vocab_size"],
            "stage3_runs": [x["run"] for x in rs],
        })

    models.sort(key=lambda m: -m["recall@10"])
    spreads = [m["recall@10_spread"] for m in models if m["n_stage3_runs"] > 1]
    report = {
        "n_stage3_results_scanned": len(runs),
        "n_models_valid": len(models),
        "n_runs_excluded": len(excluded),
        "protocol_required": PROTOCOL,
        "stage3_repeat_noise": {
            "models_with_repeat_runs": len(spreads),
            "max_recall_spread": max(spreads) if spreads else None,
            "per_model_spreads": {
                m["model"]: m["recall@10_spread"] for m in models if m["n_stage3_runs"] > 1
            },
        },
        "models": models,
        "excluded": excluded,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2))
    with OUT_CSV.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([
            "model", "family", "recall@10", "ndcg@10", "n_eval", "vocab_size",
            "n_stage3_runs", "recall@10_spread", "sid_path", "sid_sha256",
        ])
        for m in models:
            w.writerow([
                m["model"], m["family"], m["recall@10"], m["ndcg@10"], m["n_eval"],
                m["vocab_size"], m["n_stage3_runs"], m["recall@10_spread"],
                m["sid_path"], m["sid_sha256"],
            ])
    print(f"scanned {len(runs)} stage3 results -> {len(models)} valid models, "
          f"{len(excluded)} excluded runs")
    if spreads:
        print(f"stage3 repeat noise: max Recall@10 spread {max(spreads):.6f} "
              f"over {len(spreads)} models with repeat runs")
    print(f"written: {OUT_JSON}\nwritten: {OUT_CSV}")


if __name__ == "__main__":
    main()