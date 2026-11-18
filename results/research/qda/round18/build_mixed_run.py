"""300 epochs of the mixed family with the balanced assignment floored.

The euclid family has been run to the full budget; the mixed family -- the one whose last
level balances with Sinkhorn -- has not been run since the guard stopped switching that
balancing off. This arm closes that: same tree, guard installed, balancing preserved.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO / "tools" / "research"))

from stage2.anticollapse import apply_floor  # noqa: E402

SOURCE = REPO / "results" / "research" / "qda" / "trees" / "F0_seed42"
ARM = "MIXED_GUARDED_300"

REBASE = f"""
_R18 = Path("{HERE / 'run'}")
RESULT_DIR = _R18
OUTPUT_SID = _R18 / "sids_for_hgrec.npy"
OUTPUT_JSON = _R18 / "item_sids.json"
CHECKPOINT = _R18 / "rqvae_best.pth"
LOG_DIR = _R18 / "logs"
METRICS_PATH = LOG_DIR / "training_metrics.jsonl"
EXPORT_MANIFEST = _R18 / "export_manifest.json"
"""


def main() -> None:
    target = HERE / "trees" / ARM
    if target.exists():
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(SOURCE, target, symlinks=True,
                    ignore=shutil.ignore_patterns("__pycache__", "logs", "observations"))
    print("floor:", apply_floor(target)["changed"], flush=True)

    trainer = target / "train_rqvae.py"
    text = trainer.read_text(encoding="utf-8")
    text = text.replace(
        '        if (parent / "stage2_RQ-VAE").is_dir():',
        '        if (parent / "stage2_RQ-VAE").is_dir() and (parent / "CLAUDE.md").is_file():', 1)
    anchor = "EPOCHS              = 3000"
    assert anchor in text, "the F0 trainer's budget constant changed"
    text = text.replace(anchor, "EPOCHS              = 300" + REBASE, 1)
    trainer.write_text(text, encoding="utf-8")
    print("built", ARM, flush=True)


if __name__ == "__main__":
    main()
