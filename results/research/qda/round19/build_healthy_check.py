"""A healthy run with the separation check installed: it must not fire.

The near-collapse adversary is refused by `verify_separation`; this arm exists to show the
same check does not stop an ordinary run.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO / "tools" / "research"))

from stage2.anticollapse import apply_floor  # noqa: E402

BASELINE = REPO / "stage2_RQ-VAE" / "baseline" / "TIGER_RQ-VAE"
ARM = "HEALTHY_60"

REBASE = f"""
_R19 = Path("{HERE / 'run'}")
RESULT_DIR = _R19
OUTPUT_SID = _R19 / "sids_for_hgrec.npy"
OUTPUT_JSON = _R19 / "item_sids.json"
CHECKPOINT = _R19 / "rqvae_best.pth"
LOG_DIR = _R19 / "logs"
METRICS_PATH = LOG_DIR / "training_metrics.jsonl"
EXPORT_MANIFEST = _R19 / "export_manifest.json"
"""


def main() -> None:
    target = HERE / "trees" / ARM
    if target.exists():
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(BASELINE, target, symlinks=True,
                    ignore=shutil.ignore_patterns("__pycache__", "logs"))
    print("floor:", apply_floor(target)["changed"], flush=True)
    trainer = target / "train_rqvae.py"
    text = trainer.read_text(encoding="utf-8")
    text = text.replace(
        '        if (parent / "stage2_RQ-VAE").is_dir():',
        '        if (parent / "stage2_RQ-VAE").is_dir() and (parent / "CLAUDE.md").is_file():', 1)
    anchor = "EPOCHS              = CONFIG.EPOCHS"
    text = text.replace(anchor, anchor + "\nEPOCHS = 60" + REBASE, 1)
    trainer.write_text(text, encoding="utf-8")
    print("built", ARM, flush=True)


if __name__ == "__main__":
    main()
