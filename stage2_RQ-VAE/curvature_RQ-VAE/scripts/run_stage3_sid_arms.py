"""Run the frozen Stage3 evaluator on two already-saved Stage2 SID sets.

Arm A  the accepted hyperbolic model: 256 codes per level, no category cone.
Arm B  the L2=64 no-cone arm, whose level-1 and level-2 prefixes carry
       category structure (fine AMI 0.0901 against 0.0001 for A) but whose raw
       three-token SIDs collide on 3.15% of the items.

Nothing is retrained here.  Stage2 already wrote both ``item_sids.json`` files
and Stage3 consumes them as they are, four tokens wide: the first three tokens
come from the RQ-VAE residual levels and the fourth is the collision-extension
id that Stage2 appends to keep every item's sequence unique.

Stage3 is the frozen evaluator, so both arms run the same trainer with the same
architecture, budget, seed and metric definitions, and each writes into its own
checkpoint and log subtree so no two SID configurations can share weights.  The
pre-flight block prints the offset, vocabulary and mapping facts that make the
two arms comparable, and prints the raw three-token collision rate next to the
four-token one, because the fourth token makes every item unique by
construction and would otherwise hide the Stage2 difference.

Each arm is launched with the trainer's own torchrun helper, which re-executes
this file under torchrun.  ``SIDARM`` is launcher plumbing that tells the
re-executed child which arm it is evaluating; it is not a hyperparameter.
"""

import glob
import importlib.util
import json
import os
import sys
from pathlib import Path

try:
    import torch
    _TORCHVISION_NMS_LIB = torch.library.Library("torchvision", "DEF")
    _TORCHVISION_NMS_LIB.define(
        "nms(Tensor dets, Tensor scores, float iou_threshold) -> Tensor"
    )
except (ImportError, RuntimeError):
    pass

STAGE3_DIR = "/home/wlia0047/ar57/wenyu/GeneRec/stage3_T5Train"
sys.path.insert(0, STAGE3_DIR)
spec = importlib.util.spec_from_file_location(
    "genrec_stage3_sid_arms", f"{STAGE3_DIR}/train_HG-Rec.py"
)
if spec is None or spec.loader is None:
    raise RuntimeError("Could not load Stage3 trainer")
trainer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trainer)

RESULTS = "/home/wlia0047/ar57/wenyu/GeneRec/results"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from launch_utils import launch_arm  # noqa: E402

# name -> (item_sids.json, variant directory under results/stage3_T5Train)
ARMS = {
    "A": (
        f"{RESULTS}/stage2_RQ-VAE/curvature_RQ-VAE/item_sids.json",
        "sidarm_A_l2_256_nocone",
    ),
    "B": (
        f"{RESULTS}/stage2_RQ-VAE/curvature_RQ-VAE/"
        "l2_cone_arms/B_poincare_nocone_s42/item_sids.json",
        "sidarm_B_l2_64_nocone",
    ),
    "D": (
        f"{RESULTS}/stage2_RQ-VAE/curvature_RQ-VAE/"
        "l2_cone_arms/D_poincare_cone_s42/item_sids.json",
        "sidarm_D_l2_64_cone",
    ),
    "A4": (
        f"{RESULTS}/stage2_RQ-VAE/curvature_RQ-VAE/"
        "l2_cone_arms/A_euclid_nocone_s42/item_sids.json",
        "sidarm_A4_euclid_l2_64_nocone",
    ),
    # The euclidean counterpart of the accepted model: same 72,000 Stage2
    # steps, same 256 codes per level, same seed, no cone, and zero raw
    # three-token collisions like the hyperbolic model, so this pair differs by
    # geometry alone.
    "E72": (
        f"{RESULTS}/stage2_RQ-VAE/curvature_RQ-VAE/"
        "euclid_l2_256_72k_s42/item_sids.json",
        "sidarm_E72_euclid_l2_256_72k",
    ),
    # The round-five hyperbolic cone arm, the one whose held-out containment
    # already has the shape the mechanism is supposed to produce: 0.91 positive
    # against 0.29/0.16 negative, while its euclidean twin collapsed to zero.
    # It has never been through Stage3, so it is the direct test of whether that
    # containment advantage transfers.
    "R5D": (
        f"{RESULTS}/stage2_RQ-VAE/curvature_RQ-VAE/"
        "l2_cone_arms_fixed_rejection/D_poincare_cone_s42/item_sids.json",
        "sidarm_R5D_poincare_cone_rejection_fixed",
    ),
    # Metric-margin cone arms: the same objective in both geometries, which for
    # euclid degenerates to the plain angular cone loss.
    "ME": (
        f"{RESULTS}/stage2_RQ-VAE/curvature_RQ-VAE/"
        "metric_cone_arms/A_euclid_metriccone_s42/item_sids.json",
        "sidarm_ME_euclid_metriccone",
    ),
    # Metric-smooth quantiser arms: no cone, no partition change, judged on the
    # same held-out containment readings and then on Stage3.
    "SE": (
        f"{RESULTS}/stage2_RQ-VAE/curvature_RQ-VAE/"
        "smoothness_arms/A_euclid_smooth_s42/item_sids.json",
        "sidarm_SE_euclid_smooth",
    ),
    "SH": (
        f"{RESULTS}/stage2_RQ-VAE/curvature_RQ-VAE/"
        "smoothness_arms/B_poincare_smooth_s42/item_sids.json",
        "sidarm_SH_poincare_smooth",
    ),
    # The same mechanism with both arms at equal relative pressure, which is the
    # matched pair the first smoothness round did not have.
    "SFE": (
        f"{RESULTS}/stage2_RQ-VAE/curvature_RQ-VAE/"
        "smooth_fair_arms/A_euclid_smoothfair_s42/item_sids.json",
        "sidarm_SFE_euclid_smoothfair",
    ),
    "SFP": (
        f"{RESULTS}/stage2_RQ-VAE/curvature_RQ-VAE/"
        "smooth_fair_arms/B_poincare_smoothfair_s42/item_sids.json",
        "sidarm_SFP_poincare_smoothfair",
    ),
    # Position test: the same smoothness term at the same per-level weight,
    # applied to level 3 alone and to levels 1-2 together.
    "PLE": (
        f"{RESULTS}/stage2_RQ-VAE/curvature_RQ-VAE/"
        "position_arms/A_euclid_l3only_s42/item_sids.json",
        "sidarm_PLE_euclid_l3only",
    ),
    "PLP": (
        f"{RESULTS}/stage2_RQ-VAE/curvature_RQ-VAE/"
        "position_arms/B_poincare_l3only_s42/item_sids.json",
        "sidarm_PLP_poincare_l3only",
    ),
    "PME": (
        f"{RESULTS}/stage2_RQ-VAE/curvature_RQ-VAE/"
        "position_arms/C_euclid_l12only_s42/item_sids.json",
        "sidarm_PME_euclid_l12only",
    ),
    "PMP": (
        f"{RESULTS}/stage2_RQ-VAE/curvature_RQ-VAE/"
        "position_arms/D_poincare_l12only_s42/item_sids.json",
        "sidarm_PMP_poincare_l12only",
    ),
    # The surviving mechanism at two further seeds: metric-smooth quantisation,
    # hyperbolic arm only, against the bar's own seed spread.
    "M43": (
        f"{RESULTS}/stage2_RQ-VAE/curvature_RQ-VAE/"
        "multiseed_arms/smooth_poincare_s43/item_sids.json",
        "sidarm_M43_smooth_poincare_s43",
    ),
    "M44": (
        f"{RESULTS}/stage2_RQ-VAE/curvature_RQ-VAE/"
        "multiseed_arms/smooth_poincare_s44/item_sids.json",
        "sidarm_M44_smooth_poincare_s44",
    ),
    # Cone-family separation: the last implemented mechanism, which asks
    # different categories' cones to stay apart by a margin in the arm's metric.
    "SPE": (
        f"{RESULTS}/stage2_RQ-VAE/curvature_RQ-VAE/"
        "separation_arms/A_euclid_separation_s42/item_sids.json",
        "sidarm_SPE_euclid_separation",
    ),
    "SPP": (
        f"{RESULTS}/stage2_RQ-VAE/curvature_RQ-VAE/"
        "separation_arms/B_poincare_separation_s42/item_sids.json",
        "sidarm_SPP_poincare_separation",
    ),
    # The bar's own configuration at two further training seeds: the noise the
    # configuration has against itself, which every mechanism comparison needs
    # before its few-percent differences can be read.
    "E43": (
        f"{RESULTS}/stage2_RQ-VAE/curvature_RQ-VAE/"
        "euclid_l2_256_72k_s43/item_sids.json",
        "sidarm_E43_euclid_l2_256_72k_s43",
    ),
    "E44": (
        f"{RESULTS}/stage2_RQ-VAE/curvature_RQ-VAE/"
        "euclid_l2_256_72k_s44/item_sids.json",
        "sidarm_E44_euclid_l2_256_72k_s44",
    ),
    # Coarse cone moved onto level 1, level 2 left to reconstruction, at the
    # production 256 codes per level so the numbers compare to the bar.
    "CLE": (
        f"{RESULTS}/stage2_RQ-VAE/curvature_RQ-VAE/"
        "coarse_l1_arms/A_euclid_coarse_l1_s42/item_sids.json",
        "sidarm_CLE_euclid_coarse_l1",
    ),
    "CLP": (
        f"{RESULTS}/stage2_RQ-VAE/curvature_RQ-VAE/"
        "coarse_l1_arms/B_poincare_coarse_l1_s42/item_sids.json",
        "sidarm_CLP_poincare_coarse_l1",
    ),
    "MP": (
        f"{RESULTS}/stage2_RQ-VAE/curvature_RQ-VAE/"
        "metric_cone_arms/B_poincare_metriccone_s42/item_sids.json",
        "sidarm_MP_poincare_metriccone",
    ),
}


def preflight(code_path):
    """Print the token facts the two arms have to agree on to be comparable."""
    rows = json.load(open(code_path, encoding="utf-8"))
    codes = [list(v) for v in rows.values()]
    width = len(codes[0])
    per_column_max = [max(c[i] for c in codes) for i in range(width)]
    raw = [tuple(c[: width - 1]) for c in codes]
    full = [tuple(c) for c in codes]
    raw_unique = len(set(raw))
    print(f"[sidarm] code file      : {code_path}", flush=True)
    print(f"[sidarm] items          : {len(codes)}", flush=True)
    print(f"[sidarm] n_digit        : {width}", flush=True)
    print(f"[sidarm] per-column max : {per_column_max}", flush=True)
    print(f"[sidarm] max token (+1) : {max(per_column_max) + 1}", flush=True)
    print(
        f"[sidarm] raw 3-token collisions : "
        f"{len(codes) - raw_unique} items "
        f"({100.0 * (len(codes) - raw_unique) / len(codes):.2f}%)",
        flush=True,
    )
    print(
        f"[sidarm] 4-token collisions     : "
        f"{len(codes) - len(set(full))} items",
        flush=True,
    )


def configure(arm):
    code_path, variant = ARMS[arm]
    trainer.CODE_PATH = code_path
    trainer.RQVAE_VARIANT = variant
    trainer.LOG_PATH = f"{RESULTS}/stage3_T5Train/{variant}/logs/"
    trainer.SAVE_PATH = f"{RESULTS}/stage3_T5Train/{variant}/ckpt/"
    trainer._LAUNCHER["script"] = os.path.abspath(__file__)
    trainer._LAUNCHER["log"] = os.path.join(trainer.LOG_PATH, "_stage3_launcher.log")


def finished(variant):
    """True when this arm already produced its final-test event."""
    pattern = (
        f"{RESULTS}/stage3_T5Train/{variant}/logs/"
        "Amazon_2023_Instruments/*/training_metrics.jsonl"
    )
    for path in sorted(glob.glob(pattern)):
        with open(path, encoding="utf-8") as handle:
            if any('"event": "test"' in line for line in handle):
                return True
    return False


if "RANK" in os.environ:
    # The torchrun child re-executes this file, so it has to apply the arm
    # itself; without this the trainer would evaluate its own default TIGER
    # codebook instead of the arm the launcher selected.
    arm = os.environ.get("SIDARM")
    if arm not in ARMS:
        raise RuntimeError(f"SIDARM must name an arm, got {arm!r}")
    configure(arm)
    if trainer.CODE_PATH != ARMS[arm][0]:
        raise RuntimeError(
            f"arm {arm} resolved CODE_PATH {trainer.CODE_PATH}, "
            f"expected {ARMS[arm][0]}"
        )
    preflight(trainer.CODE_PATH)
    trainer.main()
else:
    for arm in ARMS:
        if finished(ARMS[arm][1]):
            print(f"[sidarm] arm {arm} already evaluated, skipping", flush=True)
            continue
        if not Path(ARMS[arm][0]).is_file():
            # An arm whose Stage2 has not run yet. Skipping keeps a batch that
            # was queued before that round existed from aborting on it.
            print(f"[sidarm] arm {arm} has no SIDs yet, skipping", flush=True)
            continue
        print(f"[sidarm] ===== arm {arm} =====", flush=True)
        preflight(ARMS[arm][0])
        configure(arm)
        # launch_arm blocks and returns; the trainer's own launcher ends the
        # process, which would leave every later arm unrun.
        launch_arm(trainer, {"SIDARM": arm})
    print("[sidarm] both arms finished", flush=True)