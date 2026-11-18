"""Try to defeat the guard without ever duplicating a codeword.

Every check in the guard is exact: empty codewords, identical rows, tie-decided assignment.
A near-collapse -- all codewords pulled into a tiny ball, still pairwise distinct, margins
still non-zero -- is the shape none of them is designed to catch. This arm produces it on
purpose so the boundary is measured rather than asserted: an auxiliary loss pulls every
codeword toward the codebook mean, and the run is allowed to proceed if the guard lets it.
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
ARM = "NEAR_COLLAPSE"

REBASE = f"""
_R17 = Path("{HERE / 'run'}")
RESULT_DIR = _R17
OUTPUT_SID = _R17 / "sids_for_hgrec.npy"
OUTPUT_JSON = _R17 / "item_sids.json"
CHECKPOINT = _R17 / "rqvae_best.pth"
LOG_DIR = _R17 / "logs"
METRICS_PATH = LOG_DIR / "training_metrics.jsonl"
EXPORT_MANIFEST = _R17 / "export_manifest.json"
"""

CONSTANT = "NEAR_COLLAPSE_WEIGHT = 5.0\n"

INJECT_ANCHOR = "            loss, components = CRITERION(reconstructed, batch, quant_parts)"
INJECT = """            loss, components = CRITERION(reconstructed, batch, quant_parts)
            if NEAR_COLLAPSE_WEIGHT:
                _pull = sum(((layer.get_code_embs()
                              - layer.get_code_embs().mean(dim=0)) ** 2).mean()
                            for layer in raw_module.rq.vq_layers)
                loss = loss + NEAR_COLLAPSE_WEIGHT * _pull"""


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
    text = text.replace('ARM_NAME = "TIGER_RQ-VAE"', f'ARM_NAME = "{ARM}"', 1)
    anchor = "EPOCHS              = CONFIG.EPOCHS"
    text = text.replace(anchor, anchor + "\nEPOCHS = 60" + REBASE, 1)
    text = text.replace("BETA                = CONFIG.BETA", "BETA                = CONFIG.BETA\n" + CONSTANT, 1)
    assert INJECT_ANCHOR in text
    text = text.replace(INJECT_ANCHOR, INJECT, 1)
    trainer.write_text(text, encoding="utf-8")
    print("built", ARM, "(60 epochs, near-collapse loss)", flush=True)


if __name__ == "__main__":
    main()
