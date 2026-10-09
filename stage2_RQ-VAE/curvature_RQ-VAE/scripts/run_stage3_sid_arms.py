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
        print(f"[sidarm] ===== arm {arm} =====", flush=True)
        preflight(ARMS[arm][0])
        configure(arm)
        os.environ["SIDARM"] = arm
        trainer._launch_via_torchrun()
    print("[sidarm] both arms finished", flush=True)