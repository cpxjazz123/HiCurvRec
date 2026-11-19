"""The frozen-batch version of K > batch: 8192 codewords against 4096 items per step.

Round23 exercised the rotating regime at a diagnostic batch of 256. I said the frozen-scale
version would need an 8192-cluster KMeans and was not worth the budget; measured, that
KMeans takes 21 seconds, so the excuse was wrong. This runs it.
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
ARM = "K8192"

REBASE = f"""
_R24 = Path("{HERE / 'run'}")
RESULT_DIR = _R24
OUTPUT_SID = _R24 / "sids_for_hgrec.npy"
OUTPUT_JSON = _R24 / "item_sids.json"
CHECKPOINT = _R24 / "rqvae_best.pth"
LOG_DIR = _R24 / "logs"
METRICS_PATH = LOG_DIR / "training_metrics.jsonl"
EXPORT_MANIFEST = _R24 / "export_manifest.json"
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
    config["CODEBOOK_SIZE"] = [8192, 256, 64]      # L1: 8192 codewords, 4096 items
    config_path.write_text(json.dumps(config, indent=2) + "\n")

    trainer = target / "train_rqvae.py"
    text = trainer.read_text(encoding="utf-8")
    text = text.replace(
        '        if (parent / "stage2_RQ-VAE").is_dir():',
        '        if (parent / "stage2_RQ-VAE").is_dir() and (parent / "CLAUDE.md").is_file():', 1)
    anchor = "EPOCHS              = CONFIG.EPOCHS"
    text = text.replace(anchor, anchor + "\nEPOCHS = 30" + REBASE, 1)
    trainer.write_text(text, encoding="utf-8")
    print("built", ARM, "(8192/256/64 codes, frozen batch 4096)", flush=True)


if __name__ == "__main__":
    main()
