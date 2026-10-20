"""The discriminating experiments of the learning loop (issue #104, Workstream B).

Two explanations for what #103 observed, each with a falsifiable prediction and a minimal
experiment that uses only artifacts that already exist. Neither experiment trains anything:
they measure, so a hypothesis can be falsified before a GPU-hour is spent on it.

**H1 -- where the soft-to-hard implication breaks.**
Observation: constraining the *soft* assignment to the host's occupancy (39.5% of the
host's level-0 codes) and enforcing the host's quota exactly (24.2%) both failed to
preserve the *exported hard* codes.

* E1a "the argmin": the exported code is ``argmin_c ||z - e_c||``. A constraint on the
  assignment's marginal is not a constraint on that argmin, so the soft marginal can be
  exactly the host's while the hard counts are not.
  * prediction: a large share of the soft mass sits on codes the hard map never uses, and
    the soft marginal and the hard counts disagree in rank. Falsified if they agree.
* E1b "the codebook moves": the constrained terms still carry a codebook gradient, so the
  codebook itself moves and the occupancy follows the codebook's geometry.
  * prediction: the hard occupancy is bounded by the codebook's own geometry -- codes with
    no nearest items have no mass however the assignment is constrained. Falsified if the
    unused codes are geometrically reachable (their nearest-item distances are comparable).

**H2 -- high level-0 occupancy with degraded Hard-L1 ER.**
Observation: the arm with 86% of the host's level-0 occupancy scored 14.54 against the
control's 16.36; the R8 lanes show occupancy and ER moving apart in both directions.

* E2a "usage is not discrimination": the occupancy counts *how many* codes are used; the ER
  measures whether real observed transitions are more likely than negatives to share a
  code. A code can be used by many items and still not separate successors from negatives.
  * prediction: across the eleven recorded lanes, occupancy and ER are weakly correlated,
    while the train-side pair discrimination tracks the ER.
* E2b "the deeper levels compensate": the level-2 prefix recovers what the level-1 code
  loses, so a level-1 occupancy loss is invisible downstream.
  * prediction: among the pairs that DO share a level-1 code, the level-2 prefix separates
  them as often as it does on the control. Falsified if the conditional level-2 match rate
  falls with the ER.

Run with the project's environment (torch and numpy):

    /home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 \
        tools/mvg/experiments.py --experiment all --out <dir>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CGV_DIR = REPO_ROOT / "tools" / "cgv"
if str(CGV_DIR) not in sys.path:
    sys.path.insert(0, str(CGV_DIR))

EMBEDDINGS = REPO_ROOT / "results" / "stage2_RQ-VAE" / "_input" / "item_emb.npy"
ITEM_IDS = REPO_ROOT / "results" / "stage2_RQ-VAE" / "_input" / "item_ids.json"
TRAIN = REPO_ROOT / "results" / "stage0_build_parquet" / "train.parquet"
LEADERBOARD = REPO_ROOT / "results" / "stage2_RQ-VAE" / "r8_ten_mechanisms" / "leaderboard.json"
CONTROL_TREE = REPO_ROOT / "stage2_RQ-VAE" / "TIGER_r8_control"

LANES = (
    "r8_control",
    "r8_01_directed_l1_transition_nce",
    "r8_02_l1_next_item_prediction",
    "r8_03_successor_profile_js",
    "r8_04_behavior_conditioned_ot",
    "r8_05_three_level_distillation",
    "r8_06_train_graph_diffusion",
    "r8_07_hyperbolic_gyro",
    "r8_08_hyperbolic_frechet",
    "r8_09_hyperbolic_branching",
    "r8_10_hyperbolic_gw",
)


def _rank(values: Sequence[float]) -> np.ndarray:
    order = np.argsort(np.asarray(values, dtype=float), kind="stable")
    ranks = np.empty(len(order), dtype=float)
    ranks[order] = np.arange(len(order), dtype=float)
    return ranks


def spearman(left: Sequence[float], right: Sequence[float]) -> float:
    """Spearman's rho, computed here so nothing else has to be installed."""
    a = _rank(left)
    b = _rank(right)
    if a.std() == 0.0 or b.std() == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def load_sid_matrix(path: Path) -> np.ndarray:
    path = Path(path)
    if not path.is_file():
        return np.zeros((0, 4), dtype=np.int64)
    if path.suffix == ".npy":
        return np.load(path)
    document = json.loads(path.read_text(encoding="utf-8"))
    max_id = max(int(k) for k in document)
    matrix = np.zeros((max_id + 1, 4), dtype=np.int64)
    for key, value in document.items():
        matrix[int(key)] = value
    return matrix


# --------------------------------------------------------------------------------------
# H1 -- where the soft-to-hard implication breaks
# --------------------------------------------------------------------------------------
def experiment_soft_hard(out_dir: Path, *, temperature: float = 0.1) -> Dict[str, Any]:
    """Measure the soft assignment's marginal against the hard code's occupancy."""
    import torch

    import engine as cgv_engine  # type: ignore  # noqa: E402

    if not CONTROL_TREE.is_dir():
        return {"ok": False, "error": f"the control tree is not here: {CONTROL_TREE}"}
    if not EMBEDDINGS.is_file():
        return {"ok": False, "error": f"the embeddings are not here: {EMBEDDINGS}"}
    trainer, saved_path, saved_mods = cgv_engine.load_trainer(CONTROL_TREE, module_name="exp_h1")
    try:
        checkpoint = Path(str(trainer.CHECKPOINT))
        if not checkpoint.is_file():
            return {"ok": False, "error": f"no control checkpoint at {checkpoint}"}
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
        embeddings = np.load(EMBEDDINGS).astype("float32")
        model = trainer.RQVAE(trainer._tokenizer_config(), in_dim=int(embeddings.shape[1]))
        model.load_state_dict(state, strict=False)
        model.eval()
        with torch.no_grad():
            encoded = model.encoder(torch.from_numpy(embeddings)).detach()
            codebook = model.rq.vq_layers[0].get_code_embs().detach()
            # the level-0 distances, exactly as the quantiser computes them
            distances = torch.cdist(encoded, codebook, p=2)
            hard = distances.argmin(dim=1).numpy()
            soft = torch.softmax(-distances / float(temperature), dim=1).numpy()
        items = int(hard.shape[0])
        codes = int(codebook.shape[0])
        hard_counts = np.bincount(hard, minlength=codes).astype(float)
        soft_mass = soft.mean(axis=0)
        used = int((hard_counts > 0).sum())
        # how much of the soft mass sits on codes the hard map never uses
        unused = hard_counts == 0
        mass_on_unused = float(soft_mass[unused].sum())
        # do the two distributions agree in rank?
        rho = spearman(soft_mass, hard_counts)
        # the geometry explanation: the distance from each code to its nearest item
        nearest = distances.min(dim=0).values.numpy()
        reachable = nearest <= float(np.quantile(nearest, 0.5))
        return {
            "ok": True,
            "items": items,
            "codes": codes,
            "hard_used": used,
            "hard_fraction": round(used / codes, 6),
            "soft_mass_on_unused_codes": round(mass_on_unused, 6),
            "soft_top_decile_mass": round(float(np.sort(soft_mass)[-max(1, codes // 10):].sum()), 6),
            "spearman_soft_vs_hard": round(rho, 6),
            "unused_codes_geometrically_reachable": int((unused & reachable).sum()),
            "prediction": {
                "E1a": "a large soft mass on unused codes and a weak rank correlation "
                       "(|rho| small) => the argmin is not constrained by the marginal",
                "E1b": "unused codes are geometrically unreachable (no item is close to them) "
                       "=> the occupancy is set by the codebook's geometry",
            },
        }
    finally:
        cgv_engine.unload_trainer(saved_path, saved_mods, module_name="exp_h1")


# --------------------------------------------------------------------------------------
# H2 -- usage against discrimination
# --------------------------------------------------------------------------------------
def _train_pairs(limit: int = 20000) -> tuple[np.ndarray, np.ndarray]:
    """Observed train transitions: (the last item of the history, the target)."""
    import pandas as pd

    frame = pd.read_parquet(TRAIN, columns=["history", "target"])
    anchors: List[int] = []
    targets: List[int] = []
    for history, target in zip(frame["history"], frame["target"]):
        if len(history) == 0:
            continue
        anchors.append(int(history[-1]))
        targets.append(int(target))
        if len(anchors) >= limit:
            break
    return np.asarray(anchors, dtype=np.int64), np.asarray(targets, dtype=np.int64)


def experiment_occupancy_discrimination(out_dir: Path, *, negatives: int = 10,
                                        seed: int = 42) -> Dict[str, Any]:
    """Occupancy against train-side pair discrimination, across the recorded lanes."""
    import numpy as np

    anchors, targets = _train_pairs()
    if anchors.size == 0:
        return {"ok": False, "error": "no train transitions could be read"}
    rng = np.random.default_rng(seed)
    negative_ids = rng.integers(0, int(anchors.max()) + 1, size=(anchors.size, negatives))
    recorded: Dict[str, Any] = {}
    if LEADERBOARD.is_file():
        for lane in json.loads(LEADERBOARD.read_text(encoding="utf-8")).get("lanes", []):
            recorded[str(lane.get("lane"))] = lane.get("hard_l1_er")

    rows: List[Dict[str, Any]] = []
    for lane in LANES:
        sid = load_sid_matrix(REPO_ROOT / "results" / "stage2_RQ-VAE" / lane / "item_sids.json")
        if sid.size == 0:
            continue
        valid = (anchors < sid.shape[0])
        a = anchors[valid]
        t = targets[valid][a < sid.shape[0]]
        a = a[a < sid.shape[0]]
        if a.size == 0:
            continue
        neg = negative_ids[valid]
        neg = np.clip(neg, 0, sid.shape[0] - 1)
        l1_anchor = sid[a, 0]
        l1_pos = sid[t, 0]
        l1_neg = sid[neg, 0]
        pos_l1 = float((l1_anchor == l1_pos).mean())
        neg_l1 = float((l1_anchor[:, None] == l1_neg).mean())
        l2_anchor = sid[a, :2]
        l2_pos = sid[t, :2]
        pos_l2 = float(np.all(l2_anchor == l2_pos, axis=1).mean())
        neg_l2 = float(np.all(l2_anchor[:, None, :] == sid[neg, :2], axis=2).mean())
        shared = l1_anchor == l1_pos
        conditional_l2 = float(np.all(l2_anchor[shared] == l2_pos[shared], axis=1).mean()) if shared.any() else float("nan")
        rows.append(
            {
                "lane": lane,
                "items": int(sid.shape[0]),
                "l1_used": int(len(np.unique(sid[:, 0]))),
                "l1_fraction": round(len(np.unique(sid[:, 0])) / sid.shape[0], 6),
                "l1l2_used": int(len(np.unique(sid[:, :2], axis=0))),
                "n_pairs": int(a.size),
                "train_pos_l1": round(pos_l1, 6),
                "train_neg_l1": round(neg_l1, 6),
                "train_train_er_l1": round(pos_l1 / neg_l1, 6) if neg_l1 else None,
                "train_pos_l1l2": round(pos_l2, 6),
                "train_neg_l1l2": round(neg_l2, 6),
                "conditional_l2_among_shared_l1": None if conditional_l2 != conditional_l2 else round(conditional_l2, 6),
                "recorded_hard_l1_er": recorded.get(lane),
            }
        )

    if not rows:
        return {"ok": False, "error": "no lane's SIDs could be read"}
    with_er = [row for row in rows if isinstance(row.get("recorded_hard_l1_er"), (int, float))]
    occupancy = [row["l1_fraction"] for row in with_er]
    frozen = [float(row["recorded_hard_l1_er"]) for row in with_er]
    train_side = [row["train_train_er_l1"] or 0.0 for row in with_er]
    conditional = [
        row["conditional_l2_among_shared_l1"] or 0.0
        for row in with_er
        if row.get("conditional_l2_among_shared_l1") is not None
    ]
    return {
        "ok": True,
        "lanes": rows,
        "correlations": {
            "occupancy_vs_frozen_er": round(spearman(occupancy, frozen), 6),
            "train_side_er_vs_frozen_er": round(spearman(train_side, frozen), 6),
            "conditional_l2_vs_frozen_er": round(spearman(conditional, frozen), 6)
            if len(conditional) > 2
            else None,
        },
        "prediction": {
            "E2a": "occupancy and the frozen ER are weakly correlated, and the train-side "
                   "pair discrimination tracks the frozen ER => usage is not discrimination",
            "E2b": "the conditional level-2 match rate among shared level-1 pairs falls with "
                   "the ER => the deeper level does not compensate",
        },
    }


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tools.mvg.experiments", description=__doc__)
    parser.add_argument("--experiment", default="all",
                        choices=("all", "soft_hard", "occupancy_discrimination"))
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    report: Dict[str, Any] = {}
    if args.experiment in ("all", "soft_hard"):
        report["soft_hard"] = experiment_soft_hard(out_dir)
        (out_dir / "h1_soft_hard.json").write_text(
            json.dumps(report["soft_hard"], indent=2, default=str) + "\n", encoding="utf-8"
        )
    if args.experiment in ("all", "occupancy_discrimination"):
        report["occupancy_discrimination"] = experiment_occupancy_discrimination(out_dir)
        (out_dir / "h2_occupancy_discrimination.json").write_text(
            json.dumps(report["occupancy_discrimination"], indent=2, default=str) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, indent=2, default=str)[:4000])
    return 0 if all(section.get("ok") for section in report.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
