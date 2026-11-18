"""Measure the separation ratio over training, without gating on it.

`min codeword distance / quantiser resolution` separates a collapsed codebook from a
healthy one at convergence, but a gate at 1.0 refused a healthy run early in training.
This probe asks the next question: does the ratio stay within an order of magnitude of its
*own initial value*? If it does for healthy runs and not for the adversarial one, a
calibration-free rule exists (relative to the run's own initialization).
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
EPOCHS = 60

LOGGER = '''

# -- diagnostic-only probe (round20): record the separation ratio, never gate on it ------
import atexit as _atexit
import json as _json
import os as _os

_RATIOS: dict[str, list] = {}


def record_separation(layer, cost):
    import torch
    codes = layer.get_code_embs().detach().float().reshape(layer.n_embed, -1)
    pairwise = torch.cdist(codes, codes)
    pairwise.fill_diagonal_(float("inf"))
    min_pair = float(pairwise.min())
    resolution = float(cost.detach().float().min(dim=-1).values.mean().clamp_min(0).sqrt())
    ratio = min_pair / resolution if resolution > 0 else float("inf")
    _RATIOS.setdefault(str(id(layer)), []).append(ratio)


@_atexit.register
def _dump():
    path = _os.environ.get("SEPARATION_OUT")
    if path:
        with open(path, "w") as handle:
            _json.dump(_RATIOS, handle)
'''

CALL = "        record_separation(self, cost)\n"

REBASE = f"""
_R20 = Path("{HERE / 'run'}")
RESULT_DIR = _R20
OUTPUT_SID = _R20 / "sids_for_hgrec.npy"
OUTPUT_JSON = _R20 / "item_sids.json"
CHECKPOINT = _R20 / "rqvae_best.pth"
LOG_DIR = _R20 / "logs"
METRICS_PATH = LOG_DIR / "training_metrics.jsonl"
EXPORT_MANIFEST = _R20 / "export_manifest.json"
"""


def build(name: str, source: Path, collapse: bool) -> Path:
    tree = HERE / "trees" / name
    if tree.exists():
        shutil.rmtree(tree)
    tree.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, tree, symlinks=True,
                    ignore=shutil.ignore_patterns("__pycache__", "logs", "observations"))
    apply_floor(tree)

    module = tree / "model" / ("layers.py" if (tree / "model" / "layers.py").is_file()
                              and "mixed" not in source.name.lower() else "mixed_rq.py")
    if name.startswith("ADVERSARY"):
        module = tree / "model" / "layers.py"
    text = module.read_text(encoding="utf-8")
    text = text.replace("verify_codebook, verify_margins)",
                        "verify_codebook, verify_margins, record_separation)", 1)
    anchor = "        verify_codebook(self.get_code_embs(), distance=False)\n"
    assert anchor in text, module
    text = text.replace(anchor, CALL + anchor, 1)
    module.write_text(text, encoding="utf-8")

    floor_module = tree / "assignment_floor.py"
    floor_module.write_text(floor_module.read_text(encoding="utf-8") + LOGGER, encoding="utf-8")

    trainer = tree / "train_rqvae.py"
    body = trainer.read_text(encoding="utf-8")
    body = body.replace(
        '        if (parent / "stage2_RQ-VAE").is_dir():',
        '        if (parent / "stage2_RQ-VAE").is_dir() and (parent / "CLAUDE.md").is_file():', 1)
    for anchor_line in ("EPOCHS              = CONFIG.EPOCHS", "EPOCHS              = 3000"):
        if anchor_line in body:
            body = body.replace(anchor_line, anchor_line + f"\nEPOCHS = {EPOCHS}" + REBASE, 1)
            break
    if collapse:
        body = body.replace(
            "BETA                = CONFIG.BETA",
            "BETA                = CONFIG.BETA\nNEAR_COLLAPSE_WEIGHT = 5.0", 1)
        anchor = "            loss, components = CRITERION(reconstructed, batch, quant_parts)"
        assert anchor in body
        body = body.replace(
            anchor,
            anchor + "\n            if NEAR_COLLAPSE_WEIGHT:\n"
            "                _pull = sum(((l.get_code_embs() - l.get_code_embs().mean(dim=0)) ** 2).mean()"
            " for l in raw_module.rq.vq_layers)\n"
            "                loss = loss + NEAR_COLLAPSE_WEIGHT * _pull", 1)
    trainer.write_text(body, encoding="utf-8")
    return tree


def main() -> None:
    healthy = build("HEALTHY", BASELINE, collapse=False)
    adversary = build("ADVERSARY", BASELINE, collapse=True)
    print("healthy:", healthy)
    print("adversary:", adversary)


if __name__ == "__main__":
    main()
