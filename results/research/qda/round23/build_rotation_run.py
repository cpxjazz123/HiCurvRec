"""Exercise the rotating regime in the real trainer: K > batch.

The frozen recipe has batch 4096, so a K > 4096 codebook would need an 8192-cluster KMeans
to test it. The *regime* is what matters, not the absolute size: a diagnostic batch of 256
with a 512-code first level triggers exactly the same path (items < codewords) at a cost
that fits in a minute.
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
ARM = "ROTATION"

REBASE = f"""
_R23 = Path("{HERE / 'run'}")
RESULT_DIR = _R23
OUTPUT_SID = _R23 / "sids_for_hgrec.npy"
OUTPUT_JSON = _R23 / "item_sids.json"
CHECKPOINT = _R23 / "rqvae_best.pth"
LOG_DIR = _R23 / "logs"
METRICS_PATH = LOG_DIR / "training_metrics.jsonl"
EXPORT_MANIFEST = _R23 / "export_manifest.json"
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
    config["CODEBOOK_SIZE"] = [512, 256, 64]
    config["BATCH_SIZE_PER_RANK"] = 256          # 256 items against 512 codewords
    config_path.write_text(json.dumps(config, indent=2) + "\n")

    trainer = target / "train_rqvae.py"
    text = trainer.read_text(encoding="utf-8")
    text = text.replace(
        '        if (parent / "stage2_RQ-VAE").is_dir():',
        '        if (parent / "stage2_RQ-VAE").is_dir() and (parent / "CLAUDE.md").is_file():', 1)
    anchor = "EPOCHS              = CONFIG.EPOCHS"
    text = text.replace(anchor, anchor + "\nEPOCHS = 10" + REBASE, 1)
    trainer.write_text(text, encoding="utf-8")
    print("built", ARM, "(512/256/64 codes, batch 256)", flush=True)


if __name__ == "__main__":
    main()
