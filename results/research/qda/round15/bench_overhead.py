"""What does the guard cost per epoch, on the frozen baseline's own recipe?

Two arms, same recipe, same seed, 30 epochs each: one exactly as published, one with the
guard installed. Every output path is rebased into this round, so neither run can touch a
published result directory.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO / "tools" / "research"))

from stage2.anticollapse import apply_floor  # noqa: E402

BASELINE = REPO / "stage2_RQ-VAE" / "baseline" / "TIGER_RQ-VAE"
EPOCHS = 60

REBASE = """
_R15 = Path("{run}")
RESULT_DIR = _R15
OUTPUT_SID = _R15 / "sids_for_hgrec.npy"
OUTPUT_JSON = _R15 / "item_sids.json"
CHECKPOINT = _R15 / "rqvae_best.pth"
LOG_DIR = _R15 / "logs"
METRICS_PATH = LOG_DIR / "training_metrics.jsonl"
EXPORT_MANIFEST = _R15 / "export_manifest.json"
"""


def build(name: str, guarded: bool) -> Path:
    tree = HERE / "trees" / name
    if tree.exists():
        shutil.rmtree(tree)
    tree.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(BASELINE, tree, symlinks=True,
                    ignore=shutil.ignore_patterns("__pycache__", "logs"))
    if guarded:
        apply_floor(tree)
    trainer = tree / "train_rqvae.py"
    text = trainer.read_text(encoding="utf-8")
    text = text.replace(
        '        if (parent / "stage2_RQ-VAE").is_dir():',
        '        if (parent / "stage2_RQ-VAE").is_dir() and (parent / "CLAUDE.md").is_file():',
        1)
    anchor = "EPOCHS              = CONFIG.EPOCHS"
    text = text.replace(
        anchor,
        anchor + f"\nEPOCHS              = {EPOCHS}" + REBASE.format(run=HERE / "run" / name),
        1)
    trainer.write_text(text, encoding="utf-8")
    return tree


def run(tree: Path) -> float:
    started = time.time()
    done = subprocess.run([sys.executable, "-u", "train_rqvae.py"], cwd=str(tree),
                          capture_output=True, text=True, timeout=1800)
    elapsed = time.time() - started
    assert done.returncode == 0, done.stderr[-800:]
    epochs = re.findall(r"time=([0-9.]+)s", done.stdout)
    per_epoch = [float(value) for value in epochs]
    print(f"{tree.name}: {len(per_epoch)} epochs, wall {elapsed:.1f}s, "
          f"steady mean {sum(per_epoch[1:]) / max(1, len(per_epoch) - 1):.4f}s")
    return sum(per_epoch[1:]) / max(1, len(per_epoch) - 1)


def build_floor_only(name: str) -> Path:
    """The floor without the per-step detectors: what does the assignment alone cost?"""
    tree = build(name, guarded=True)
    module = tree / "model" / "layers.py"
    text = module.read_text(encoding="utf-8")
    text = text.replace("        verify_codebook(self.get_code_embs(), distance=False)\n", "")
    text = text.replace("        verify_margins(cost)\n", "")
    module.write_text(text, encoding="utf-8")
    return tree


def main() -> None:
    plain = build("PLAIN_60", guarded=False)
    guarded = build("GUARDED_60", guarded=True)
    floor_only = build_floor_only("FLOOR_60")
    a = run(plain)
    c = run(floor_only)
    b = run(guarded)
    print(f"floor only: {100.0 * (c - a) / a:+.1f}%  (detectors add "
          f"{100.0 * (b - c) / c:+.1f}% on top)")
    print(f"overhead: {100.0 * (b - a) / a:+.1f}%  (plain {a:.4f}s -> guarded {b:.4f}s)")


if __name__ == "__main__":
    main()
