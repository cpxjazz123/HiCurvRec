"""A mechanism shape the guard was never tuned on: different codebook size per level.

The frozen recipe uses 256/256/256 at dimension 32. A capacity-allocation mechanism may
want 512/256/64, or a different codebook dimension. Nothing in the guard assumes a single
size -- `constrained_assignment` and the checks read K from the cost matrix and the
codebook -- but that is an argument, not a measurement. This arm measures it.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO / "tools" / "research"))

from stage2.anticollapse import apply_floor  # noqa: E402

BASELINE = REPO / "stage2_RQ-VAE" / "baseline" / "TIGER_RQ-VAE"
ARM = "DIM16"

REBASE = f"""
_R22D = Path("{HERE / ARM / 'run'}")
RESULT_DIR = _R22D
OUTPUT_SID = _R22D / "sids_for_hgrec.npy"
OUTPUT_JSON = _R22D / "item_sids.json"
CHECKPOINT = _R22D / "rqvae_best.pth"
LOG_DIR = _R22D / "logs"
METRICS_PATH = LOG_DIR / "training_metrics.jsonl"
EXPORT_MANIFEST = _R22D / "export_manifest.json"
"""


def main() -> None:
    target = HERE / "trees" / ARM
    if target.exists():
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(BASELINE, target, symlinks=True,
                    ignore=shutil.ignore_patterns("__pycache__", "logs"))
    print("floor:", apply_floor(target)["changed"], flush=True)

    config_path = target / "config" / "train_config.json"
    config = json.loads(config_path.read_text())
    config["CODEBOOK_SIZE"] = [256, 256, 256]
    config["CODEBOOK_DIM"] = 16
    config_path.write_text(json.dumps(config, indent=2) + "\n")

    trainer = target / "train_rqvae.py"
    text = trainer.read_text(encoding="utf-8")
    text = text.replace(
        '        if (parent / "stage2_RQ-VAE").is_dir():',
        '        if (parent / "stage2_RQ-VAE").is_dir() and (parent / "CLAUDE.md").is_file():', 1)
    anchor = "EPOCHS              = CONFIG.EPOCHS"
    text = text.replace(anchor, anchor + "\nEPOCHS = 60" + REBASE, 1)
    trainer.write_text(text, encoding="utf-8")
    print("built", ARM, "(codebook dim 16)", flush=True)


if __name__ == "__main__":
    main()
