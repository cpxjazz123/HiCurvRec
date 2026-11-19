"""The frozen 3000-epoch budget with the *complete* guard installed.

The earlier full-budget run predates the relative spacing check, so the strongest claim
had a hole: 3000 epochs with everything *except* that check. This run closes it, and it is
the only way to know whether a healthy codebook drifts below a tenth of its initial
spacing over the full budget.
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
ARM = "GUARDED_FULL_V2"

REBASE = f"""
_R16 = Path("{HERE / 'run'}")
RESULT_DIR = _R16
OUTPUT_SID = _R16 / "sids_for_hgrec.npy"
OUTPUT_JSON = _R16 / "item_sids.json"
CHECKPOINT = _R16 / "rqvae_best.pth"
LOG_DIR = _R16 / "logs"
METRICS_PATH = LOG_DIR / "training_metrics.jsonl"
EXPORT_MANIFEST = _R16 / "export_manifest.json"
"""


def main() -> None:
    target = HERE / "trees" / ARM
    if target.exists():
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(BASELINE, target, symlinks=True,
                    ignore=shutil.ignore_patterns("__pycache__", "logs"))
    report = apply_floor(target)
    print("floor:", report["changed"], report.get("family"), flush=True)

    trainer = target / "train_rqvae.py"
    text = trainer.read_text(encoding="utf-8")
    text = text.replace(
        '        if (parent / "stage2_RQ-VAE").is_dir():',
        '        if (parent / "stage2_RQ-VAE").is_dir() and (parent / "CLAUDE.md").is_file():',
        1)
    text = text.replace('ARM_NAME = "TIGER_RQ-VAE"', f'ARM_NAME = "{ARM}"', 1)
    anchor = "EPOCHS              = CONFIG.EPOCHS"
    text = text.replace(anchor, anchor + REBASE, 1)     # keeps the frozen 3000
    trainer.write_text(text, encoding="utf-8")
    print("built", ARM, "(3000 epochs, the frozen budget)", flush=True)


if __name__ == "__main__":
    main()
