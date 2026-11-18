"""Run the guarded frozen baseline for 300 epochs and watch the margin health.

The guard refuses a step when more than half of the items pick their codeword by a tie.
That threshold is safe at initialisation, but nobody has checked it late in training, where
codewords could legitimately drift together. This arm answers that with a real run: the
guard is active, so a false positive would kill the run at the epoch where it happens.

    python build_margin_run.py && cd trees/<ARM> && python -u train_rqvae.py
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
ARM = "MARGIN_300"


def main() -> None:
    target = HERE / "trees" / ARM
    if target.exists():
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(BASELINE, target, symlinks=True,
                    ignore=shutil.ignore_patterns("__pycache__", "logs"))
    report = apply_floor(target)
    print("floor:", report["changed"], report.get("family"))

    trainer = target / "train_rqvae.py"
    text = trainer.read_text(encoding="utf-8")
    text = text.replace(
        '        if (parent / "stage2_RQ-VAE").is_dir():',
        '        if (parent / "stage2_RQ-VAE").is_dir() and (parent / "CLAUDE.md").is_file():', 1)
    text = text.replace('ARM_NAME = "TIGER_RQ-VAE"', f'ARM_NAME = "{ARM}"', 1)
    # A diagnostic must never write into a published result directory. ARM_NAME alone is
    # not enough: every output path is rebased here, explicitly.
    rebase = (
        f'\n_R14 = Path("/fs04/ar57/wenyu/GeneRec/results/research/qda/round14/run")\n'
        f"RESULT_DIR = _R14\n"
        f"OUTPUT_SID = _R14 / 'sids_for_hgrec.npy'\n"
        f"OUTPUT_JSON = _R14 / 'item_sids.json'\n"
        f"CHECKPOINT = _R14 / 'rqvae_best.pth'\n"
        f"LOG_DIR = _R14 / 'logs'\n"
        f"METRICS_PATH = LOG_DIR / 'training_metrics.jsonl'\n"
        f"EXPORT_MANIFEST = _R14 / 'export_manifest.json'\n"
    )
    anchor = "EPOCHS              = CONFIG.EPOCHS"
    assert anchor in text
    text = text.replace(anchor, anchor + "\nEPOCHS              = 300   # diagnostic budget" + rebase, 1)
    trainer.write_text(text, encoding="utf-8")
    print("built", ARM)


if __name__ == "__main__":
    main()
