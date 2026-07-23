"""task58_p0_eval.py — Task 450 P0.4 eval helper.

Wrapper around task13_eval.evaluate() with arbitrary --tag instead of fixed group.
Used to evaluate Stage 4 predictions for both P0.2 (Euclidean-Concat) and
P0.3 (Mixed-Curvature).
"""
import argparse
import json
import os
import sys
from pathlib import Path

# Bootstrap sys.path and cwd like task13_eval does
GRID_ROOT = "/home/wlia0047/ar57/wenyu/GeneRec/GRID"
os.chdir(GRID_ROOT)
sys.path.insert(0, GRID_ROOT)
sys.path.insert(0, str(Path(GRID_ROOT) / "task_artifacts" / "scripts"))
from src.utils.custom_hydra_resolvers import (  # noqa: F401
    remove_chars_from_string,
    conditional_expression,
    extract_fields_from_list_of_dicts,
    create_map_from_list_of_dicts,
    math_eval,
    remove_item_from_list,
)
from task13_eval import evaluate, compute_collision_rate  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="any tag e.g. P02 or P03")
    ap.add_argument("--sid", required=True, type=Path)
    ap.add_argument("--stage4", required=True, type=Path)
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--out_json", required=True, type=Path)
    args = ap.parse_args()

    print(f"=== {args.tag} eval ===")
    coll = compute_collision_rate(str(args.sid))
    print(f"[collision] {coll}")
    metrics = evaluate(
        group=args.tag,
        sid_path=str(args.sid),
        stage4_path=str(args.stage4),
        ckpt_path=str(args.ckpt),
    )
    print(f"[metrics] {metrics}")
    result = {
        "task": "task58_mixed_curvature",
        "tag": args.tag,
        **coll,
        **metrics,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[saved] {args.out_json}")


if __name__ == "__main__":
    main()