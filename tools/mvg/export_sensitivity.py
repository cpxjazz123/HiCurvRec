"""The export-sensitivity test: did a mechanism touch the exported SIDs at all?

The #104 campaign produced two arms whose exported SIDs were **byte-identical** to the
control's, and two whose exported SIDs were wrecked. Neither fact was visible in the
training loss, and neither was visible in a "the mechanism ran" check: the mechanism did
run, the loss did move, and the exported codes did not change by one bit. A probe that
only asks "did the parameters move" cannot tell those three outcomes apart.

This test answers the question the export actually cares about, by comparing an arm's
exported SIDs with a **matched control's** exported SIDs, item by item:

* how many items changed their L1, L1+L2 or full 3-token code, and by how much;
* the occupancy at each level, and the collision rate at each level;
* the TRAIN-SIDE behaviour-pair discrimination -- how often a transition's source and
  target share a level-0 code, against how often a source shares its code with a random
  item. This is the training-side measure; the frozen held-out ER is a different thing
  and is never used here.

The verdict is one of three words: ``inert`` (the export did not change), ``collapsed``
(the arm lost codes relative to the control) or ``active`` (the export changed and the
code structure survived). The verdict is an admission signal, never a scientific success:
an active arm can still be a bad arm, and only the frozen evaluation decides that.

Run with the project's own environment (numpy and pandas):

    /home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 \
        tools/mvg/export_sensitivity.py \
          --arm-sid results/stage2_RQ-VAE/<arm>/r8_control/sids_for_hgrec.npy \
          --control-sid results/stage2_RQ-VAE/r8_control/sids_for_hgrec.npy \
          --train-parquet results/stage0_build_parquet/train.parquet \
          --out /tmp/export_sensitivity.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Below this fraction of changed items the arm is inert: the export is the same object.
INERT_CHANGED_FRACTION = 1e-6

#: Below this fraction of the control's level-0 occupancy the arm has collapsed.
COLLAPSE_OCCUPANCY_RATIO = 0.5


def load_sids(path: Path) -> np.ndarray:
    """The arm's exported SIDs as an [N, levels] integer matrix."""
    path = Path(path)
    if path.suffix == ".npy":
        return np.load(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    max_id = max(int(key) for key in raw)
    matrix = np.zeros((max_id + 1, 4), dtype=np.int64)
    for key, value in raw.items():
        matrix[int(key)] = value
    return matrix


def occupancy(sids: np.ndarray, levels: int) -> Dict[str, Any]:
    """How many distinct codes the arm uses at the first ``levels`` levels, and the
    collision rate the export therefore has."""
    codes = sids[:, :levels]
    distinct = int(len(np.unique(codes, axis=0)))
    items = int(codes.shape[0])
    return {
        "levels": levels,
        "distinct_codes": distinct,
        "items": items,
        "occupancy_fraction": (distinct / items) if items else 0.0,
        # The fraction of items that share their code with at least one other item.
        "collision_rate": float(1.0 - distinct / items) if items else 0.0,
    }


def changed_fraction(arm: np.ndarray, control: np.ndarray, levels: int) -> float:
    """The fraction of items whose code at the first ``levels`` levels differs."""
    size = min(arm.shape[0], control.shape[0])
    if size == 0:
        return 0.0
    a = arm[:size, :levels]
    c = control[:size, :levels]
    return float((a != c).any(axis=1).mean())


def train_pairs(parquet: Path, limit: int = 200000) -> Tuple[np.ndarray, np.ndarray]:
    """The (source item, target item) transitions of the training split."""
    import pandas as pd

    frame = pd.read_parquet(parquet)
    columns = {name.lower(): name for name in frame.columns}
    history = columns.get("history")
    target = columns.get("target")
    if history is None or target is None:
        raise ValueError(
            f"the training parquet has no history/target columns: {sorted(frame.columns)}"
        )
    sources: List[int] = []
    targets: List[int] = []
    for past, item in zip(frame[history][:limit], frame[target][:limit]):
        if past is None or item is None:
            continue
        sequence = list(past) if isinstance(past, (list, tuple, np.ndarray)) else [past]
        if not sequence:
            continue
        sources.append(int(sequence[-1]))
        targets.append(int(item))
    return np.asarray(sources, dtype=np.int64), np.asarray(targets, dtype=np.int64)


def train_side_discrimination(
    sids: np.ndarray, sources: np.ndarray, targets: np.ndarray, seed: int = 42
) -> Dict[str, Any]:
    """How well the exported level-0 codes separate a transition from a random pair.

    This is the TRAINING-SIDE measure: the source and target items are the ones the
    model was trained on. It is not the frozen held-out enrichment ratio.
    """
    rng = np.random.default_rng(seed)
    size = min(len(sources), len(targets))
    if size == 0:
        return {"pos_share": None, "neg_share": None, "discrimination": None, "pairs": 0}
    sources = sources[:size]
    targets = targets[:size]
    n_items = sids.shape[0]
    if int(sources.max()) >= n_items or int(targets.max()) >= n_items:
        raise ValueError("the training pairs reference items the export does not contain")
    codes = sids[:, 0]
    positives = codes[sources] == codes[targets]
    negatives = codes[sources] == codes[rng.integers(0, n_items, size=size)]
    pos_share = float(positives.mean())
    neg_share = float(negatives.mean())
    discrimination = (pos_share / neg_share) if neg_share > 0 else None
    return {
        "pos_share": pos_share,
        "neg_share": neg_share,
        "discrimination": discrimination,
        "pairs": int(size),
    }


def compare(
    arm_sid: Path,
    control_sid: Path,
    parquet: Path,
    *,
    limit: int = 200000,
    seed: int = 42,
) -> Dict[str, Any]:
    """The whole test: the arm's export against the matched control's export."""
    arm = load_sids(arm_sid)
    control = load_sids(control_sid)
    sources, targets = train_pairs(parquet, limit=limit)
    levels = min(arm.shape[1], control.shape[1], 3)
    arm_levels = [occupancy(arm, level) for level in (1, 2, 3)]
    control_levels = [occupancy(control, level) for level in (1, 2, 3)]
    changed = {
        f"levels_{level}": changed_fraction(arm, control, level) for level in (1, 2, 3)
    }
    changed["levels_3token_identical"] = bool(
        arm.shape[0] == control.shape[0]
        and bool((arm[: min(arm.shape[0], control.shape[0]), :3]
                  == control[: min(arm.shape[0], control.shape[0]), :3]).all())
    )
    ratio = None
    if control_levels[0]["occupancy_fraction"] > 0:
        ratio = arm_levels[0]["occupancy_fraction"] / control_levels[0]["occupancy_fraction"]
    verdict = "active"
    reasons: List[str] = []
    if changed["levels_3"] < INERT_CHANGED_FRACTION:
        verdict = "inert"
        reasons.append(
            f"not one item's level-0/1/2 code changed relative to the matched control "
            f"({changed['levels_3']:.1e} of items changed)"
        )
    if ratio is not None and ratio < COLLAPSE_OCCUPANCY_RATIO:
        verdict = "collapsed"
        reasons.append(
            f"the arm uses {arm_levels[0]['occupancy_fraction']:.1%} of the codes against the "
            f"control's {control_levels[0]['occupancy_fraction']:.1%} (ratio {ratio:.3f})"
        )
    return {
        "ok": True,
        "arm_sid": str(arm_sid),
        "control_sid": str(control_sid),
        "train_parquet": str(parquet),
        "verdict": verdict,
        "reasons": reasons,
        "changed_fraction": changed,
        "arm_levels": arm_levels,
        "control_levels": control_levels,
        "level0_occupancy_ratio_vs_control": ratio,
        "train_side_discrimination": {
            "arm": train_side_discrimination(arm, sources, targets, seed=seed),
            "control": train_side_discrimination(control, sources, targets, seed=seed),
        },
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="tools.mvg.export_sensitivity", description=__doc__)
    parser.add_argument("--arm-sid", required=True, help="the arm's exported SIDs")
    parser.add_argument("--control-sid", required=True, help="the matched control's SIDs")
    parser.add_argument(
        "--train-parquet",
        default=str(REPO_ROOT / "results" / "stage0_build_parquet" / "train.parquet"),
        help="the training split, for the train-side discrimination measure",
    )
    parser.add_argument("--limit", type=int, default=200000, help="how many train pairs to read")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="", help="where to write the report")
    args = parser.parse_args(argv)
    report = compare(
        Path(args.arm_sid),
        Path(args.control_sid),
        Path(args.train_parquet),
        limit=args.limit,
        seed=args.seed,
    )
    text = json.dumps(report, indent=2, default=str)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
