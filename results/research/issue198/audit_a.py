"""Issue #198 part A: is SIGN's recipe the frozen one it was copied from?

Compares the two trees' constants with the values their own training logs recorded, computes steps_per_epoch from the item count and the global batch, and flags
every field that differs. Read-only; nothing is trained.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
TREES = {"baseline": "TIGER_mixed_rq", "sign": "TIGER_sign_rq"}
FIELDS = ("epochs", "batch_size_per_rank", "total_batch_size", "world_size", "optimizer", "lr",
          "weight_decay", "warmup_epochs", "seed", "codebook_num", "codebook_size", "codebook_dim",
          "beta", "vq_type", "sk_epsilon", "sk_iters", "pca_dim", "gradient_clip_norm", "num_workers", "eval_interval")


def train_start(arm: str) -> dict:
    # the trainer writes its metrics into the tree; the results directory only holds the exported artefacts
    path = REPO_ROOT / "stage2_RQ-VAE" / arm / "logs" / "training_metrics.jsonl"
    if not path.exists():
        return {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("event") == "train_start":
            return row
    return {}


def constants(arm: str) -> dict:
    text = (REPO_ROOT / "stage2_RQ-VAE" / arm / "train_rqvae.py").read_text().splitlines()
    wanted = ("EPOCHS", "BATCH_SIZE_PER_RANK", "LR", "WEIGHT_DECAY", "WARMUP_EPOCHS", "SEED", "CODEBOOK_NUM",
              "CODEBOOK_SIZE", "CODEBOOK_DIM", "BETA", "VQ_TYPE", "SK_EPSILON", "SK_ITERS", "PCA_DIM",
              "GRADIENT_CLIP_NORM", "NUM_WORKERS", "EVAL_INTERVAL", "OPTIMIZER")
    out = {}
    for line in text:
        stripped = line.split("=", 1)
        if len(stripped) != 2:
            continue
        name = stripped[0].strip()
        if name in wanted and not line.startswith(" "):
            out[name] = stripped[1].split("#", 1)[0].strip()
    return out


def main() -> int:
    report = {"trees": {}, "differences": [], "steps_per_epoch": {}}
    for label, arm in TREES.items():
        start = train_start(arm)
        report["trees"][label] = {
            "arm": arm,
            "logged": {field: start.get(field) for field in FIELDS},
            "constants": constants(arm),
            "has_metrics": bool(start),
        }
        items, global_batch = start.get("train_items"), start.get("total_batch_size")
        if items and global_batch:
            report["steps_per_epoch"][label] = {
                "train_items": int(items),
                "global_batch": int(global_batch),
                "expected": int(math.ceil(items / global_batch)),
            }

    base, sign = report["trees"]["baseline"], report["trees"]["sign"]
    for field in FIELDS:
        if base["logged"].get(field) != sign["logged"].get(field):
            report["differences"].append({"field": field, "baseline": base["logged"].get(field), "sign": sign["logged"].get(field)})
    for name in sorted(set(base["constants"]) | set(sign["constants"])):
        if base["constants"].get(name) != sign["constants"].get(name):
            report["differences"].append({"constant": name, "baseline": base["constants"].get(name), "sign": sign["constants"].get(name)})

    report["recipe_identical"] = not report["differences"]
    report["global_batch_is_frozen_4096"] = report["trees"]["sign"]["logged"].get("total_batch_size") in (4096, "4096")
    (HERE / "audit_a.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
