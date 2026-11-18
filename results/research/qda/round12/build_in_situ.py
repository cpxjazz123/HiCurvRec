"""Build an arm whose mechanism collapses the codebook on purpose, in the real trainer.

The unit tests prove the guard refuses a duplicate table when it is handed one. This arm
proves the same check is live inside the frozen trainer: at a fixed epoch the mechanism
copies codeword 0 over every other codeword -- exactly what a badly written codebook
re-normalisation does -- and the very next forward must refuse the step.

Nothing else changes: same tree, same recipe, same 3000-epoch budget constant.
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
ARM = "COLLAPSE_AT_10"

CONSTANTS = """#: The mechanism collapses the table at this epoch, on purpose (diagnostic arm).
COLLAPSE_AT_EPOCH = 10
"""

INJECT_ANCHOR = """            optimizer.step()
            scheduler.step()"""
INJECT = """            optimizer.step()
            scheduler.step()
            if epoch == COLLAPSE_AT_EPOCH:
                # A codebook re-normalisation that copies one vector everywhere. The guard
                # is expected to refuse the next forward, inside this same epoch.
                with torch.no_grad():
                    for _layer in raw_module.rq.vq_layers:
                        _codes = _layer.get_code_embs()
                        _codes[1:] = _codes[0]
                print("[collapse] duplicated codeword 0 across every codeword", flush=True)"""


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
    # This diagnostic runs in the round workspace instead of a promoted arm directory, and
    # `results/` also holds a `stage2_RQ-VAE`, so the single-marker search stops one level
    # too early. `CLAUDE.md` exists only at the true root.
    source = trainer.read_text(encoding="utf-8")
    marker = '        if (parent / "stage2_RQ-VAE").is_dir():'
    assert marker in source
    source = source.replace(
        marker,
        '        if (parent / "stage2_RQ-VAE").is_dir() and (parent / "CLAUDE.md").is_file():',
        1)
    trainer.write_text(source, encoding="utf-8")
    text = trainer.read_text(encoding="utf-8")
    marker = 'ARM_NAME            = "TIGER_RQ-VAE"'
    if marker not in text:
        marker = [line for line in text.splitlines() if line.startswith("ARM_NAME")][0]
    text = text.replace(marker, marker + "\n" + CONSTANTS, 1)
    text = text.replace(marker, marker.replace("TIGER_RQ-VAE", ARM), 1)
    assert INJECT_ANCHOR in text
    text = text.replace(INJECT_ANCHOR, INJECT, 1)
    trainer.write_text(text, encoding="utf-8")
    print("built", ARM)


if __name__ == "__main__":
    main()
